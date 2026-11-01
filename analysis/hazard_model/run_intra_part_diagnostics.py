#!/usr/bin/env python3
"""
IPSD · 段内停滞诊断（Intra-Part Stall Diagnostics）
=================================================================================
核心思路（来自习题级负结果的直接推论）：
  **决策单元留在分段（事件基准率 6–18%，可学），习题只作为"观测分辨率"与"诊断锚点"。**

三步：
  ① 截断增广：把每个 (学生, 分段) 的证据在**第 e 道题处截断**，重算该段的全部
     聚合特征与同伴归一化特征 → 得到 (学生, 分段, e) 训练样本，标签仍是段级 drop。
  ② 段内监控：推理时用截至第 e 题的部分证据输出"该生会在这一段停下吗"的风险，
     得到一条**随题推进上升的风险曲线** + 首次越阈题号。
  ③ 卡点诊断：对被标记学生，逐题与其同伴对照（耗时/编辑/运行/焦点/返工），
     定位"疑似卡在第 k 题"并给出可读的证据与建议。

评估（与项目主协议一致）：5 折按学生分组 × 3 seeds，part 内 pooled AUC。
对照：整段证据的 part 级基线（0.9258）、纯习题前缀模型（0.589–0.649）。
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402

EVENT_TYPES = ['text_insert', 'text_remove', 'text_paste',
               'focus_gained', 'focus_lost', 'run', 'submit']


# ---------------------------------------------------------------- 基础件
def group_peer_z(RAW, rows, group, stats):
    """同 part 同伴 z（中位数/MAD），stats 由训练折行估计"""
    out = np.zeros((len(rows), RAW.shape[1]))
    for j, r in enumerate(rows):
        st = stats.get(int(group[r]))
        if st is None:
            continue
        out[j] = (RAW[r] - st['med']) / st['mad']
    return np.clip(out, -5, 5)


def peer_stats(RAW, rows, group):
    out = {}
    for g in np.unique(group[rows]):
        idx = np.intersect1d(np.where(group == g)[0], rows)
        if len(idx) == 0:
            continue
        Xg = RAW[idx]
        med = np.median(Xg, 0)
        mad = np.maximum(np.median(np.abs(Xg - med), 0), 1e-3)
        out[int(g)] = dict(med=med, mad=mad)
    return out


def fit_rf(Xtr, ytr, Xte, seeds):
    if len(set(np.asarray(ytr).tolist())) < 2:
        return np.full(len(Xte), float(np.mean(ytr)) if len(ytr) else 0.0)
    acc = np.zeros(len(Xte))
    for sd in seeds:
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   max_features='sqrt', class_weight='balanced',
                                   random_state=sd, n_jobs=-1)
        m.fit(Xtr, ytr)
        acc += m.predict_proba(Xte)[:, 1]
    return acc / len(seeds)


def main(args):
    ds = sc.load_dataset(args.data)
    ex = np.load(args.ex, allow_pickle=True)
    raw_ex = ex['raw'].astype(np.float64)     # n,7counts,ttd_max,ttd_min,ts_min,ts_max
    ex_sid, ex_par, ex_exr = ex['sid'], ex['part'], ex['exercise']
    dsp, dpart, ddrop = ds['sid'], ds['part_ids'], ds['drop']
    seeds = [42, 123, 777]
    folds = sc.group_folds(dsp, ds['final'], n_splits=5, seed=42)

    # ---- (学生, part) → 有序习题行 ----
    sp_rows = defaultdict(list)
    for i in range(len(ex_sid)):
        sp_rows[(int(ex_sid[i]), int(ex_par[i]))].append(i)
    for k in sp_rows:
        sp_rows[k].sort(key=lambda i: ex_exr[i])

    part_tot = {int(p): int(ex_exr[ex_par == p].max()) for p in np.unique(ex_par)}
    # part 级标签与索引
    part_key = {}
    for i in range(len(dsp)):
        part_key[(int(dsp[i]), int(dpart[i]))] = i
    stage_idx = ds['stage_idx']

    # ---- 构建"截断到第 e 题"的段级 RAW ----
    def truncated_raw(e):
        """返回 (rows, RAW_trunc)：rows 是 (学生,part) 在 stage 数据中的行号"""
        rows, R = [], []
        for (s, p), idxs in sp_rows.items():
            if len(idxs) < e:
                continue
            sub = idxs[:e]
            cnt = raw_ex[sub, 1:8].sum(0)
            n = raw_ex[sub, 0].sum()
            span = (raw_ex[sub, 11].max() - raw_ex[sub, 10].min()) / 3600.0
            ttd_start = raw_ex[sub, 8].max()
            ttd_end = raw_ex[sub, 9].min()
            r = np.array([cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], cnt[5], cnt[6],
                          n, float(e), max(span, 1e-3), ttd_start, ttd_end])
            rows.append(part_key[(s, p)])
            R.append(r)
        return np.array(rows), np.array(R, dtype=np.float64)

    def features_for(rows, RAWt, tr_mask_idx, te_mask_idx):
        """与项目 rf29 同构：A(4)+B(3)+C(2)+peer_z(10)+SELF(10)"""
        part = dpart[rows]
        # DER 风格派生
        SPAN = np.maximum(RAWt[:, 9], 1e-3)
        INS = np.maximum(RAWt[:, 0], 1e-6)
        n_ex = RAWt[:, 8]
        base = np.array([part_tot[int(p)] for p in part], dtype=float)
        A = np.column_stack([np.log1p(RAWt[:, 7]), np.log1p(n_ex),
                              np.log1p(RAWt[:, 9]), np.log1p(RAWt[:, 7] / SPAN)])
        B = np.column_stack([RAWt[:, 1] / INS, RAWt[:, 4] / SPAN,
                              RAWt[:, 6] / np.maximum(RAWt[:, 7], 1)])
        C = np.column_stack([n_ex / base, stage_idx[rows].astype(float)])
        return A, B, C

    # ---- 评估：不同截断点的段内风险 ----
    print('=== 段内监控曲线：截断到第 e 题时的段级 drop 判别力 ===')
    print(f'{"截断 e":<10}{"覆盖(生,段)行":<16}{"part 内 pooled":<16}{"vs 整段基线"}')
    curves = {}
    oof_store = {}
    for e in args.cutoffs:
        t0 = time.time()
        rows, RAWt = truncated_raw(e)
        if len(rows) == 0:
            continue
        part = dpart[rows]
        lab = ddrop[rows]
        A, B, C = features_for(rows, RAWt, None, None)
        # SELF：该生更早分段的偏离（沿用项目口径，用整段 RAW）
        SELF_all = sc.self_ref(ds)
        oof = np.zeros(len(rows))
        pos_of_row = {int(r): i for i, r in enumerate(rows)}
        for tr, te in folds:
            tr_rows = np.array([r for r in tr if int(r) in pos_of_row])
            te_rows = np.array([r for r in te if int(r) in pos_of_row])
            if len(tr_rows) < 50 or len(te_rows) == 0:
                continue
            it = np.array([pos_of_row[int(r)] for r in tr_rows])
            ie = np.array([pos_of_row[int(r)] for r in te_rows])
            stats = peer_stats(RAWt, it, part)   # part 为截断数组的 part 列
            ztr = group_peer_z(RAWt, it, part, stats)
            zte = group_peer_z(RAWt, ie, part, stats)
            Xtr = np.hstack([A[it], B[it], C[it], ztr, SELF_all[tr_rows]])
            Xte = np.hstack([A[ie], B[ie], C[ie], zte, SELF_all[te_rows]])
            oof[ie] = fit_rf(Xtr, lab[it], Xte, seeds)
        m = sc.metrics(lab, oof, part, f'e{e}')
        curves[e] = m
        oof_store[e] = (rows, oof, part)
        print(f'  前 {e:<7} {len(rows):<16}{m["pooled"]:<16.4f}'
              f'[{time.time()-t0:.0f}s]', flush=True)

    # ---- 诊断：对真实掉队学生逐题与同伴对照 ----
    print('\n=== 卡点诊断示例（真实学生）===')
    e_diag = args.diag_cutoff
    rows, oof, part = oof_store[e_diag]
    pos_of_row = {int(r): i for i, r in enumerate(rows)}
    stall_students = [int(s) for s in np.unique(dsp) if
                      int(dpart[dsp == s].max()) < 7]
    # 选被模型在早期（该段第 e 题）就高分的掉队学生
    cand = []
    for (s, p), idxs in sp_rows.items():
        if s not in stall_students or p != int(dpart[dsp == s].max()):
            continue
        if len(idxs) < e_diag:
            continue
        r = part_key.get((s, p))
        if r is None or int(r) not in pos_of_row:
            continue
        cand.append((float(oof[pos_of_row[int(r)]]), s, p, r))
    cand.sort(reverse=True)
    for rank, (sc_, s, p, r) in enumerate(cand[:args.n_examples]):
        idxs = sp_rows[(s, p)]
        print(f'\n--- 示例 {rank+1}：学生 #{s} · 分段 {p} · '
              f'截断到第 {e_diag} 题的风险 = {sc_:.3f} ---')
        # 风险曲线（用同一模型在各个截断点的分数）
        curve = []
        for e in sorted(oof_store.keys()):
            rr, oo, _ = oof_store[e]
            pm = {int(x): i for i, x in enumerate(rr)}
            if int(r) in pm:
                curve.append((e, float(oo[pm[int(r)]])))
        print('  段内风险曲线（截断题号 → 风险）: ' +
              ' '.join(f'{e}:{v:.2f}' for e, v in curve))
        # 逐题同伴对照
        peer = defaultdict(list)
        for (s2, p2), idxs2 in sp_rows.items():
            if p2 == p and len(idxs2) >= len(idxs):
                peer[len(idxs2)].append(idxs2)
        print(f'  该生本段共完成 {len(idxs)} 题（该段共 {part_tot[p]} 题）')
        print(f'  {"题号":<6}{"事件":<8}{"跨度h":<9}{"同伴中位跨度h":<14}'
              f'{"编辑":<7}{"运行":<7}{"删/插":<9}{"焦点/h":<9}')
        for k, i in enumerate(idxs[-args.diag_tail:], start=len(idxs) - args.diag_tail + 1):
            peers = peer.get(len(idxs), [])
            spans = []
            for j in peers:
                if j is i:
                    continue
                spans.append((raw_ex[j, 11].max() - raw_ex[j, 10].min()) / 3600.0)
            med = float(np.median(spans)) if spans else float('nan')
            span = (raw_ex[i, 11].max() - raw_ex[i, 10].min()) / 3600.0
            ev = raw_ex[i, 1:8]
            ratio = ev[1] / max(ev[0], 1e-6)
            print(f'  {ex_exr[i]:<6}{int(raw_ex[i,0]):<8}{span:<9.2f}{med:<14.2f}'
                  f'{int(ev[0]):<7}{int(ev[5]):<7}{ratio:<9.2f}'
                  f'{ev[3]/max(span,1e-6):<9.2f}')
        # 结论行
        print(f'  → 诊断：该生在本段停于第 {ex_exr[idxs[-1]]} 题；'
              f'最后 {args.diag_tail} 题的累计事件 {int(raw_ex[idxs[-args.diag_tail:],0].sum())}')

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'ipsd_results.npz'),
                        **{f'rows_e{e}': v[0] for e, v in oof_store.items()},
                        **{f'oof_e{e}': v[1] for e, v in oof_store.items()},
                        dsp=dsp, dpart=dpart, ddrop=ddrop)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--ex', default='data/exercise_dataset.npz')
    ap.add_argument('--out', default='results/ipsd')
    ap.add_argument('--cutoffs', type=int, nargs='+', default=[1, 2, 3, 5, 8, 12, 20])
    ap.add_argument('--diag-cutoff', type=int, default=3)
    ap.add_argument('--diag-tail', type=int, default=4)
    ap.add_argument('--n-examples', type=int, default=2)
    main(ap.parse_args())
