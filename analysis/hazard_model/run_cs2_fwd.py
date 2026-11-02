#!/usr/bin/env python3
"""
CS2-FWD · 段内"中途停下"早警（前瞻窗口任务）+ 因果 Transformer
=================================================================================
背景与依据（全部来自本项目已完成的实测）：

  ① 81.8% 的掉队发生在**段内中途**（最终段完成比例中位 0.474），不是"做完一段不开下一段"
     → 产品需要**段内**就给出预警，而不是等段结束
  ② 现有判别任务的标签是"最终会不会掉队"（part 级，基准率 6–18%，pooled 0.9258）
     本实验换**任务形式**：标签 = "该生会不会在本段中途停下"
     → 标签与"何时停"对齐，且可在段内任意观察点输出
  ③ 习题级"停在哪一题"的经典 hazard 不可学（pooled 0.4920，基准率 0.58%）
     → 本任务在 (生,段) 单元上定义，基准率 ~10%，处于可学区间
  ④ 同伴相对信号主导（peer_z_submit 7/7 分段第 1 重要特征）
     → 显式同伴成对对比损失
  ⑤ 深度模型在 CS1 判别线上打不过 flat RF（CS2-SPC 实测：坐次级 0.79 vs 0.8684）
     → 本实验检验：**换一个任务形式后，深度模型是否能在"段内早警"上赢过 flat**

协议：按学生分组 5 折 × 3 seeds；peer 统计量只用训练折；主指标 part 内 pooled AUC。
跑法（246）：
  python3 analysis/hazard_model/run_cs2_fwd.py --out results/cs2_fwd
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'cs2', os.path.join(_HERE, 'run_cs2_spc.py'))
cs2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs2)


def build_stop_labels(ds, ex_sid, ex_par, ex_exr, part_key):
    """y = 1 ⇔ 该 (生,段) 是掉队生的最终段且**段内中途停下**（停点不是段边界）。

    停点 = 最终段内最后触碰的习题（97.1% 的段触碰连续前缀 → 可作停点）。
    """
    sid = ds['sid']
    part = ds['part_ids']
    maxpart = {}
    for i in range(len(sid)):
        s = int(sid[i])
        maxpart[s] = max(maxpart.get(s, 0), int(part[i]))
    # 每个 (生,段) 触碰的习题数（该段最后触碰的习题）
    last_ex = {}
    for i in range(len(ex_sid)):
        k = (int(ex_sid[i]), int(ex_par[i]))
        e = int(ex_exr[i])
        if k not in last_ex or e > last_ex[k]:
            last_ex[k] = e
    # 每 part 的可用题量基准（最终 part 的题数 = part_tot）
    part_tot = {}
    for p in np.unique(ex_par):
        part_tot[int(p)] = int(ex_exr[ex_par == p].max())
    y = np.zeros(len(sid), dtype=np.int64)
    stop_ex = {}
    for i in range(len(sid)):
        s, p = int(sid[i]), int(part[i])
        if int(ds['drop'][i]) == 1:
            # 该 (生,段) 是掉队事件（最终段且未走完）
            k = (s, p)
            le = last_ex.get(k)
            if le is None:
                continue
            # 段内中途停下：最后触碰的习题不是该段最后一道
            if le < part_tot.get(p, le):
                y[i] = 1
                stop_ex[(s, p)] = le
    return y, stop_ex, part_tot


def make_token_model(torch, nn, d_in, d=64, nlayer=2, heads=4, ff=128, maxlen=64):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d))
            self.pos = nn.Parameter(torch.zeros(1, maxlen, d))
            nn.init.normal_(self.pos, std=0.02)
            self.film = nn.Linear(1, 2 * d)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff,
                                               dropout=0.1, activation='gelu',
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, nlayer)
            self.head = nn.Linear(d, 1)
            self.attn = nn.Linear(d, 1)

        def forward(self, x, u, sit_end, pad_mask):
            B, L, _ = x.shape
            h = self.inp(x) + self.pos[:, :L]
            g, b = self.film(u).chunk(2, dim=-1)
            h = h * (1.0 + g) + b
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
            h = self.enc(h, mask=causal, src_key_padding_mask=~pad_mask)
            tok = self.head(h).squeeze(-1)
            a = self.attn(h).squeeze(-1).masked_fill(~pad_mask, -1e4)
            w = torch.softmax(a, dim=1)
            pooled = torch.bmm(w.unsqueeze(1), h).squeeze(1)
            return tok, self.head(pooled).squeeze(-1), w

    return M()


def main(args):
    t0 = time.time()
    torch, nn = cs2.build_torch()
    cs2._TORCH, cs2._NN = torch, nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = sc.load_dataset(args.data)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part_ids'])
    SELF = sc.self_ref(ds)
    ex = np.load(args.ex, allow_pickle=True)
    raw_ex = ex['raw'].astype(np.float64)
    ex_sid, ex_par, ex_exr = ex['sid'], ex['part'], ex['exercise']
    dsp, dpart = ds['sid'], ds['part_ids']
    seeds = [42, 123, 777]
    folds = sc.group_folds(dsp, ds['final'], n_splits=5, seed=42)
    part_key = {(int(dsp[i]), int(dpart[i])): i for i in range(len(dsp))}
    sittings = cs2.build_sittings(raw_ex, ex_sid, ex_par, ex_exr)

    y_stop, stop_ex, part_tot = build_stop_labels(ds, ex_sid, ex_par, ex_exr, part_key)
    print(f'段内中途停下 正例 {int(y_stop.sum())} / {len(y_stop)} '
          f'（基准率 {y_stop.mean():.3f}） | device={dev}', flush=True)

    # token 序列
    tok = {}
    for (s, p), sids in sittings.items():
        r = part_key.get((s, p))
        if r is None:
            continue
        idxs = sorted([i for sit in sids for i in sit], key=lambda i: ex_exr[i])
        T, is_end, u = cs2.token_features(raw_ex, idxs, ds['base_ex'][int(p)])
        tok[r] = (T, is_end, u)

    os.makedirs(args.out, exist_ok=True)
    results = {}

    # ---------- 截断聚合（flat 基线 + 上下文） ----------
    def trunc_raw(k):
        rows, R = [], []
        for (s, p), sids in sittings.items():
            use = [i for sit in (sids if k is None else sids[:k]) for i in sit]
            if len(use) == 0:
                continue
            if k is not None:
                # 只保留"前 k 道题"口径下题数足够的段（与 trunc_tokens 一致）
                n_ex_all = sum(len(sit) for sit in sids)
                if n_ex_all < k:
                    continue
            sub = np.array(use)
            cnt_ = raw_ex[sub, 1:8].sum(0)
            n = raw_ex[sub, 0].sum()
            span = (raw_ex[sub, 11].max() - raw_ex[sub, 10].min()) / 3600.0
            R.append([cnt_[0], cnt_[1], cnt_[2], cnt_[3], cnt_[4], cnt_[5], cnt_[6],
                      n, float(len(use)), max(span, 1e-3), raw_ex[sub, 8].max(),
                      raw_ex[sub, 9].min()])
            rows.append(part_key[(s, p)])
        return np.array(rows), np.array(R)

    def trunc_tokens(k):
        """前 k 道习题的 token（不是前 k 次坐下）"""
        out = {}
        for r, (T, is_end, u) in tok.items():
            if k is None:
                out[r] = (T, is_end, u)
            elif len(T) >= k:
                out[r] = (T[:k], is_end[:k], u[:k])
        return out

    def deep_run(rows, RAWt, tks, cfg, tag):
        part, lab = ds['part_ids'], y_stop
        pos = {int(r): i for i, r in enumerate(rows)}
        keys = sorted(pos.keys())
        oof = np.zeros(len(keys))
        for tr, te in folds:
            tr_rows = np.array([r for r in tr if int(r) in pos])
            te_rows = np.array([r for r in te if int(r) in pos])
            if len(tr_rows) < 50 or len(te_rows) == 0:
                continue
            it = np.array([pos[int(r)] for r in tr_rows])
            ie = np.array([pos[int(r)] for r in te_rows])
            stats = cs2.peer_stats(RAWt, it, part[rows])
            ztr = cs2.group_peer_z(RAWt, it, part[rows], stats)
            zte = cs2.group_peer_z(RAWt, ie, part[rows], stats)
            Ctr = cs2.stage_context(ds, RAWt, it, rows[it], part[rows][it], ztr, SELF[tr_rows])
            Cte = cs2.stage_context(ds, RAWt, ie, rows[ie], part[rows][ie], zte, SELF[te_rows])
            pz = torch.from_numpy(Ctr[:, cs2.CTX_OFF + cs2.SUB_DIM].astype(np.float32)).to(dev)
            part_tr = torch.from_numpy(part[tr_rows]).to(dev)

            Ttr = [tks[int(rows[i])][0] for i in it]
            SEtr = [tks[int(rows[i])][1] for i in it]
            Utr = [tks[int(rows[i])][2] for i in it]
            X, U, SE, PM = cs2.pad_batch(Ttr, [Ctr[i] for i in range(Ctr.shape[0])],
                                         Utr, SEtr, args.maxlen)
            X, U, SE, PM = [t.to(dev) for t in (X, U, SE, PM)]
            yy = torch.from_numpy(lab[tr_rows].astype(np.float32)).to(dev)
            models = []
            for sd in seeds:
                torch.manual_seed(sd)
                np.random.seed(sd)
                m = make_token_model(torch, nn, X.shape[-1], maxlen=X.shape[1]).to(dev)
                opt = torch.optim.AdamW(m.parameters(), lr=cfg['lr'], weight_decay=1e-4)
                n = X.shape[0]
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'])
                pwt = torch.tensor(cfg['pos_weight'], device=dev)
                for ep in range(cfg['epochs']):
                    m.train()
                    perm = torch.randperm(n, device=dev)
                    for s0 in range(0, n, cfg['bs']):
                        idx = perm[s0:s0 + cfg['bs']]
                        _, pl, _ = m(X[idx], U[idx], SE[idx], PM[idx])
                        loss = nn.functional.binary_cross_entropy_with_logits(
                            pl, yy[idx], pos_weight=pwt)
                        if cfg['peer'] > 0:
                            loss = loss + cfg['peer'] * cs2.peer_contrastive(
                                pl, pz[idx], part_tr[idx])
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                        opt.step()
                    sched.step()
                models.append(m)

            Tte = [tks[int(rows[i])][0] for i in ie]
            SEte = [tks[int(rows[i])][1] for i in ie]
            Ute = [tks[int(rows[i])][2] for i in ie]
            Xt, Ut, SEt, PMt = cs2.pad_batch(Tte, [Cte[i] for i in range(Cte.shape[0])],
                                             Ute, SEte, args.maxlen)
            Xt, Ut, SEt, PMt = [t.to(dev) for t in (Xt, Ut, SEt, PMt)]
            preds = np.zeros(len(ie))
            with torch.no_grad():
                for m in models:
                    m.eval()
                    _, pl, _ = m(Xt, Ut, SEt, PMt)
                    preds += torch.sigmoid(pl).cpu().numpy()
            oof[ie] = preds / len(seeds)
        mm = sc.metrics(lab[keys], oof, part[keys], tag)
        return mm, oof, keys

    for K in args.k_list:
        tt0 = time.time()
        k = None if K <= 0 else K
        rows, RAWt = trunc_raw(k)
        tks = trunc_tokens(k)
        keep = set(int(r) for r in rows)
        tks = {r: v for r, v in tks.items() if r in keep}
        # flat 基线：同截断口径的 31 维上下文 RF
        rf_m, rf_oof = cs2.rf_baseline(ds, RAWt, rows, SELF, folds, seeds, f'rf_fwd_K{K}')
        rf_m = {k2: v for k2, v in rf_m.items() if k2 != 'keys'}
        rf_m['pooled'] = sc.pooled(sc.per_part_auc(y_stop[rows], rf_oof, dpart[rows]))
        rf_m['top5'] = cs2.topk_precision(y_stop[rows], rf_oof, dpart[rows], 0.05)
        print(f'\n=== CS2-FWD · 观察前 {("全部习题" if k is None else str(k)+" 道题")} '
              f'(n={len(rows)}) ===', flush=True)
        print(f'  RF(flat 31维)      pooled={rf_m["pooled"]:.4f} top5%={rf_m["top5"][0]:.3f}',
              flush=True)
        results[f'rf_K{K}'] = dict(rf_m)
        np.savez_compressed(os.path.join(args.out, f'oof_rf_K{K}.npz'),
                            oof=rf_oof, y=y_stop[rows], part=dpart[rows])

        pw = float((len(y_stop) - y_stop.sum()) / max(y_stop.sum(), 1))
        for cname in ['base', 'peer']:
            cfg = dict(lr=args.lr, bs=256, epochs=args.epochs, pos_weight=pw,
                       peer=0.0 if cname == 'base' else args.lam)
            mm, oof, keys = deep_run(rows, RAWt, tks, cfg, f'cs2_fwd_{cname}_K{K}')
            top5 = cs2.topk_precision(y_stop[keys], oof, dpart[keys], 0.05)
            print(f'  CS2-FWD·{cname:<6} pooled={mm["pooled"]:.4f} '
                  f'judged={mm["judged"]:.4f} top5%={top5[0]:.3f} '
                  f'[{time.time()-tt0:.0f}s]', flush=True)
            results[f'cs2_{cname}_K{K}'] = dict(mm, top5=top5[0])
            np.savez_compressed(os.path.join(args.out, f'oof_cs2_{cname}_K{K}.npz'),
                                oof=oof, y=y_stop[keys], part=dpart[keys])

    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n完成，总耗时 {time.time()-t0:.0f}s → {args.out}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--ex', default='data/exercise_dataset.npz')
    ap.add_argument('--out', default='results/cs2_fwd')
    ap.add_argument('--k_list', type=int, nargs='+', default=[1, 3, 10, 0])
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--lam', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--maxlen', type=int, default=64)
    args = ap.parse_args()
    main(args)
