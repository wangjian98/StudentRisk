#!/usr/bin/env python3
"""
CS2-D · 段内卡点诊断模型（序列内 listwise 排序）
=================================================================================
定位：**判别线不再投深度模型**（CS2-SPC / CS2-FWD 实测：deep 一律低于 flat RF），
深度模型投到它唯一有优势的位置——**段内"卡在哪一题"的诊断**。

依据：
  · 81.8% 的掉队发生在段内中途（最终段完成比例中位 0.474）→ 卡点可定位
  · Transformer > MLP（+0.02 ~ +0.05 pooled）→ 序列里确实有 flat 聚合丢掉的时序信息
  · 卡住表征随进度变化（rem_over_ins p2–3 HR 2.5–2.7；peer_z_paste p4–5 HR 2.2–5.2）
  · 周报原型现在的"疑似卡点"判断是**手写启发式**（编辑多/运行少/焦点churn）→ 可学习化

任务（listwise，避免 0.58% 基准率的类别不平衡）：
  对每个含停点的 (生,段)：给出该段内每道题一个分数，目标是**停点题排第一**。
  损失 = 段内 softmax 交叉熵（唯一正例 = 停点题）。

评估：
  · top-1 命中率（argmax == 停点题）、top-3 命中率、MRR
  · 对照启发式基线：最大事件数 / 最大编辑数 / 最大焦点 churn / 三者组合
  · peer-relative 版本：token 级同伴 z（同一 part、同一题号、跨学生的中位/MAD，训练折统计量）

跑法（246）：
  python3 analysis/hazard_model/run_cs2_diag.py --out results/cs2_diag
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    'cs2', os.path.join(_HERE, 'run_cs2_spc.py'))
cs2 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(cs2)

STALL_CNT = 7       # 事件类型数（ins, rem, paste, fg, fl, run, submit）


def build_tokens_and_labels(raw_ex, ex_sid, ex_par, ex_exr, part_key, part_tot,
                            drop_rows):
    """返回 per-row: (T[22], peer_block, stall_idx or -1)"""
    sp_rows = defaultdict(list)
    for i in range(len(ex_sid)):
        sp_rows[(int(ex_sid[i]), int(ex_par[i]))].append(i)
    out = {}
    for (s, p), idxs in sp_rows.items():
        r = part_key.get((s, p))
        if r is None:
            continue
        idxs = sorted(idxs, key=lambda i: ex_exr[i])
        last_ex = int(ex_exr[idxs[-1]])
        # 停点 = 该段最后触碰的习题（若该段是掉队事件且未做完该段）
        stall = -1
        if r in drop_rows and last_ex < part_tot.get(p, last_ex):
            stall = len(idxs) - 1
        out[r] = (s, p, idxs, last_ex, stall)
    return out


def token_matrix(raw_ex, idxs, base_ex, ex_ids, pstats, part):
    """22 维 token 特征 + 7 维 token 级同伴 z（同一 part、同一题号、跨学生对照）"""
    T, is_end, u = cs2.token_features(raw_ex, idxs, base_ex)
    if len(T) == 0:
        return T, is_end, u, np.zeros((0, STALL_CNT), np.float32)
    # 屏蔽位置类特征（idx/base、坐次内位置、坐次号、坐次末标记）：
    # 停点按定义就是该段最后一个 token，留着位置特征会让模型作弊
    T[:, 16] = 0.0
    T[:, 19] = 0.0
    T[:, 20] = 0.0
    T[:, 21] = 0.0
    g = raw_ex[idxs]
    cnt = g[:, 1:8]
    P = np.zeros((len(T), STALL_CNT), np.float32)
    for i in range(len(T)):
        st = pstats.get((int(part), int(ex_ids[i])))
        if st is None:
            continue
        med, mad = st
        P[i] = np.clip((cnt[i] - med) / np.maximum(mad, 1e-3), -5, 5)
    return T, is_end, u, P


def peer_token_stats(raw_ex, ex_par, ex_exr, rows_idx):
    """训练折内：(part, 题号) → 该题跨学生的事件计数中位/MAD（token 级同伴基准）"""
    cells = defaultdict(list)
    for i in rows_idx:
        cells[(int(ex_par[i]), int(ex_exr[i]))].append(i)
    out = {}
    for key, idxs in cells.items():
        c = raw_ex[np.array(idxs), 1:8]
        med = np.median(c, axis=0)
        mad = np.maximum(np.median(np.abs(c - med), axis=0), 1e-3)
        out[key] = (med, mad)
    return out


def make_model(torch, nn, d_in, d=64, nlayer=2, heads=4, ff=128, maxlen=64):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d))
            self.pos = nn.Parameter(torch.zeros(1, maxlen, d))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff,
                                               dropout=0.1, activation='gelu',
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, nlayer)
            self.head = nn.Linear(d, 1)

        def forward(self, x, pad_mask):
            B, L, _ = x.shape
            h = self.inp(x) + self.pos[:, :L]
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
            h = self.enc(h, mask=causal, src_key_padding_mask=~pad_mask)
            return self.head(h).squeeze(-1)

    return M()


def main(args):
    t0 = time.time()
    torch, nn = cs2.build_torch()
    cs2._TORCH, cs2._NN = torch, nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = sc.load_dataset(args.data)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part_ids'])
    ex = np.load(args.ex, allow_pickle=True)
    raw_ex = ex['raw'].astype(np.float64)
    ex_sid, ex_par, ex_exr = ex['sid'], ex['part'], ex['exercise']
    dsp, dpart, ddrop = ds['sid'], ds['part_ids'], ds['drop']
    # ---------- 评估口径（三选一）----------
    #   cv          : 按学生分组 5 折（默认，主口径）
    #   cross_part  : 跨段泛化 —— 早段(part1-4)训练→晚段(part5-6)测试，反向再来一次
    #   lopo        : 留一 part（part1..6），跨段泛化的标准口径
    if args.mode == 'cv':
        folds = [(tr, te) for tr, te in
                 sc.group_folds(dsp, ds['final'], n_splits=5, seed=42)]
    elif args.mode == 'cross_part':
        if args.holdout_parts:
            ho = set(args.holdout_parts)
            test = np.array([i for i in range(len(dpart)) if int(dpart[i]) in ho])
            train = np.array([i for i in range(len(dpart)) if int(dpart[i]) not in ho])
            folds = [(train, test)]
        else:
            early = np.where(dpart <= 4)[0]
            late = np.where(dpart >= 5)[0]
            folds = [(early, late), (late, early)]
    elif args.mode == 'lopo':
        folds = [(np.where(dpart != p)[0], np.where(dpart == p)[0])
                 for p in range(1, 7)]
    else:
        raise ValueError(args.mode)
    # 污染标记（提交质量校验结果）：--clean_train 时从**训练侧**剔除污染行
    bad = None
    if args.clean_train:
        bad = np.load(args.quality, allow_pickle=True)['bad_mask'].astype(bool)
        if args.clean_mode == 'random':
            # 等样本量对照：随机剔除同样比例的训练行，用来区分
            # "污染本身有害" 与 "训练样本变少有害"
            rng = np.random.default_rng(20261104)
            n_drop = int(bad.sum())
            idx = rng.choice(len(bad), size=n_drop, replace=False)
            bad = np.zeros(len(bad), dtype=bool)
            bad[idx] = True
            print(f'[口径] --clean_train --clean_mode random：随机剔除 {n_drop} 行'
                  f'（等样本量对照）', flush=True)
        else:
            print(f'[口径] --clean_train：训练侧剔除污染行 {int(bad.sum())} / {len(bad)}',
                  flush=True)
    part_key = {(int(dsp[i]), int(dpart[i])): i for i in range(len(dsp))}
    part_tot = {}
    for p in np.unique(ex_par):
        part_tot[int(p)] = int(ex_exr[ex_par == p].max())
    drop_rows = {i for i in range(len(dsp)) if int(ddrop[i]) == 1}
    meta = build_tokens_and_labels(raw_ex, ex_sid, ex_par, ex_exr, part_key,
                                   part_tot, drop_rows)
    stall_rows = {r: v for r, v in meta.items() if v[4] >= 0}
    print(f'含停点的 (生,段) = {len(stall_rows)} | device={dev}', flush=True)

    os.makedirs(args.out, exist_ok=True)
    results = {}

    # ---------- 启发式基线（不需要训练）----------
    print('\n=== 启发式基线（停点题排序）===', flush=True)
    heur = {}
    for r, (s, p, idxs, last_ex, stall) in stall_rows.items():
        g = raw_ex[np.array(idxs), 1:8]
        n = raw_ex[np.array(idxs), 0]
        heur[r] = {
            'max_events': n,
            'max_edits': g[:, 0],
            'max_focus_churn': g[:, 4] / np.maximum(n, 1),
            'max_paste': g[:, 2],
            'combo': (g[:, 0] / np.maximum(n, 1)) + (g[:, 4] / np.maximum(n, 1))
                     - (g[:, 5] / np.maximum(n, 1)),
        }
    for key in ['max_events', 'max_edits', 'max_focus_churn', 'max_paste', 'combo']:
        hit1 = hit3 = 0
        rr = []
        for r, sc_ in heur.items():
            v = sc_[key]
            order = np.argsort(-v)
            st = meta[r][4]
            if order[0] == st:
                hit1 += 1
            if st in order[:3]:
                hit3 += 1
            pos = int(np.where(order == st)[0][0])
            rr.append(1.0 / (pos + 1))
        print(f'  {key:<18} top1={hit1/len(heur):.3f} top3={hit3/len(heur):.3f} '
              f'MRR={np.mean(rr):.3f}', flush=True)
        results[f'heur_{key}'] = dict(top1=hit1 / len(heur), top3=hit3 / len(heur),
                                      mrr=float(np.mean(rr)))

    # ---------- 深度模型（listwise softmax）----------
    print('\n=== CS2-D 深度模型（段内 listwise 排序）===', flush=True)
    seeds = list(args.seeds)
    hit1 = hit3 = 0
    rrs = []
    oof_scores = {}
    per_split = []
    for fi, (tr, te) in enumerate(folds):
        tr_use = tr if bad is None else tr[~bad[tr]]
        tr_stall = [r for r in stall_rows if r in set(tr_use.tolist())]
        te_stall = [r for r in stall_rows if r in set(te.tolist())]
        if len(tr_stall) < 20 or len(te_stall) == 0:
            continue
        # 训练折的 token 级同伴统计量
        tr_stu = set(dsp[tr_use].tolist())
        tr_idx = [i for i in range(len(ex_sid)) if int(ex_sid[i]) in tr_stu]
        pstats = peer_token_stats(raw_ex, ex_par, ex_exr, tr_idx)

        def build(rows_list):
            Ts, Ps, Ss, Ls = [], [], [], []
            for r in rows_list:
                s, p, idxs, le, st = meta[r]
                ex_ids = ex_exr[np.array(idxs)]
                T, is_end, u, P = token_matrix(raw_ex, idxs, ds['base_ex'][int(p)],
                                               ex_ids, pstats, int(p))
                Ts.append(np.hstack([T, P]))
                Ps.append(P)
                Ss.append(st)
                Ls.append(len(T))
            return Ts, Ps, Ss, Ls

        Ttr, Ptr, Str, Ltr = build(tr_stall)
        Tte, Pte, Ste, Lte = build(te_stall)
        d_in = Ttr[0].shape[1]
        maxlen = max(Ltr + Lte)
        Xtr = np.zeros((len(Ttr), maxlen, d_in), np.float32)
        Mtr = np.zeros((len(Ttr), maxlen), bool)
        ytr = np.zeros(len(Ttr), np.int64)
        ytok = np.zeros((len(Ttr), maxlen), np.float32)
        for i, (T, st) in enumerate(zip(Ttr, Str)):
            L = len(T)
            Xtr[i, :L] = T
            Mtr[i, :L] = True
            ytok[i, st] = 1.0
        Xtr_t = torch.from_numpy(Xtr).to(dev)
        Mtr_t = torch.from_numpy(Mtr).to(dev)
        ytok_t = torch.from_numpy(ytok).to(dev)

        models = []
        for sd in seeds:
            torch.manual_seed(sd)
            np.random.seed(sd)
            m = make_model(torch, nn, d_in, maxlen=maxlen).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            n = len(Ttr)
            for ep in range(args.epochs):
                m.train()
                perm = torch.randperm(n, device=dev)
                for s0 in range(0, n, args.bs):
                    idx = perm[s0:s0 + args.bs]
                    lg = m(Xtr_t[idx], Mtr_t[idx])
                    valid = Mtr_t[idx]
                    yb = ytok_t[idx]
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        lg[valid], yb[valid],
                        pos_weight=torch.tensor(args.pw, device=dev))
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                sched.step()
            models.append(m)

        Xte = np.zeros((len(Tte), maxlen, d_in), np.float32)
        Mte = np.zeros((len(Tte), maxlen), bool)
        for i, T in enumerate(Tte):
            L = len(T)
            Xte[i, :L] = T
            Mte[i, :L] = True
        Xte_t = torch.from_numpy(Xte).to(dev)
        Mte_t = torch.from_numpy(Mte).to(dev)
        with torch.no_grad():
            preds = np.zeros((len(Tte), maxlen))
            for m in models:
                m.eval()
                lg = m(Xte_t, Mte_t)
                preds += torch.sigmoid(lg).cpu().numpy()
        preds /= len(seeds)
        for i, r in enumerate(te_stall):
            L = len(Tte[i])
            sc_ = preds[i, :L]
            st = meta[r][4]
            order = np.argsort(-sc_)
            if order[0] == st:
                hit1 += 1
            if st in order[:3]:
                hit3 += 1
            pos = int(np.where(order == st)[0][0])
            rrs.append(1.0 / (pos + 1))
            oof_scores[r] = sc_.tolist()
        per_split.append(dict(split=fi, n_test=len(te_stall),
                              top1=hit1 / len(rrs), top3=hit3 / len(rrs),
                              mrr=float(np.mean(rrs))))

    print(f'  CS2-D           top1={hit1/len(rrs):.3f} top3={hit3/len(rrs):.3f} '
          f'MRR={np.mean(rrs):.3f}  (n={len(rrs)})', flush=True)
    results['cs2_d'] = dict(top1=hit1 / len(rrs), top3=hit3 / len(rrs),
                            mrr=float(np.mean(rrs)), n=len(rrs))
    results['mode'] = args.mode
    results['clean_train'] = bool(args.clean_train)
    results['per_split_cumulative'] = per_split

    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out, 'oof_scores.json'), 'w') as f:
        json.dump({str(k): v for k, v in oof_scores.items()}, f)
    print(f'\n完成，总耗时 {time.time()-t0:.0f}s → {args.out}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--ex', default='data/exercise_dataset.npz')
    ap.add_argument('--out', default='results/cs2_diag')
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--pw', type=float, default=30.0)
    ap.add_argument('--mode', default='cv', choices=['cv', 'cross_part', 'lopo'],
                    help='评估口径：cv=按学生分组5折；cross_part=早段→晚段；lopo=留一part')
    ap.add_argument('--clean_train', action='store_true',
                    help='训练侧剔除提交质量污染行（数据清洗鲁棒性对照）')
    ap.add_argument('--quality', default='results/quality/quality_flags.npz')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 777])
    ap.add_argument('--clean_mode', default='flags', choices=['flags', 'random'],
                    help='clean_train 的剔除方式：flags=按污染标记；random=等量随机对照')
    ap.add_argument('--holdout_parts', type=int, nargs='*', default=None,
                    help='cross_part 模式下指定留出的 part（如 5 6 = 早段训练→晚段测试）')
    args = ap.parse_args()
    main(args)
