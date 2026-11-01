#!/usr/bin/env python3
"""
任务 1 · 会话（sitting）级任务重定义 —— 把多尺度特征从"加进特征"改成"定义决策粒度"
=================================================================================
背景（负结果 → 假设）：
  · 多尺度 14 维会话结构特征**加进**阶段级特征集 → pooled 0.9200（−0.0058，P=97.85% 变差）
  · 但该族**单独**就有 0.8215 → 它不是"更多统计量"，而是**更细的决策粒度**
  · 习题级截断增广（IPSD）已证明：保留段级特征表达力、只改"看到多少证据" → 前 1 题 0.7915

假设：把截断点从"习题"换成"**一次坐下（sitting）**"，并把会话结构特征放在会话边界上注入，
      能进一步提升段内判别力（因为一次坐下才是一次真实的学习投入单元）。

单位定义（无 1.3GB 重扫的近似）：
  用已构建的习题级数据集（每习题 ts_min/ts_max）：
  同一 (学生, 分段) 内，若第 i 题的结束时刻与第 i+1 题的开始时刻间隔 < 30 min，
  则两题属于**同一次坐下**。一次坐下 = 一个"真实投入片段"。

三组对照（同一协议：按学生分组 5 折 × 3 seeds，part 内 pooled AUC）：
  A) 习题截断（复现 IPSD）：第 1/2/3/5/8 题
  B) 坐下截断（不含会话结构特征）：第 1/2/3 次坐下
  C) 坐下截断 + 会话结构前缀特征（14 维族）：第 1/2/3 次坐下
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402

SIT_GAP_SEC = 30 * 60


def group_peer_z(RAW, rows, group, stats):
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
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part_ids'])
    ds['DER'] = sc.derive(ds)
    ds['SELF'] = sc.self_ref(ds)
    ex = np.load(args.ex, allow_pickle=True)
    raw_ex = ex['raw'].astype(np.float64)
    ex_sid, ex_par, ex_exr = ex['sid'], ex['part'], ex['exercise']
    dsp, dpart, ddrop = ds['sid'], ds['part_ids'], ds['drop']
    seeds = [42, 123, 777]
    folds = sc.group_folds(dsp, ds['final'], n_splits=5, seed=42)
    SELF_all = sc.self_ref(ds)
    part_tot = {int(p): int(ex_exr[ex_par == p].max()) for p in np.unique(ex_par)}
    part_key = {}
    for i in range(len(dsp)):
        part_key[(int(dsp[i]), int(dpart[i]))] = i

    # ---------- 归并"一次坐下" ----------
    sp_rows = defaultdict(list)
    for i in range(len(ex_sid)):
        sp_rows[(int(ex_sid[i]), int(ex_par[i]))].append(i)
    sittings = {}      # (s,p) -> list of (exercise_indices, sit_no)
    n_sit_per_sp = {}
    for k, idxs in sp_rows.items():
        idxs = sorted(idxs, key=lambda i: ex_exr[i])
        cur = [idxs[0]]
        sids = []
        for a, b in zip(idxs[:-1], idxs[1:]):
            gap = raw_ex[b, 10] - raw_ex[a, 11]      # 下一题开始 - 上一题结束
            if gap < SIT_GAP_SEC:
                cur.append(b)
            else:
                sids.append(cur)
                cur = [b]
        sids.append(cur)
        sittings[k] = sids
        n_sit_per_sp[k] = len(sids)
    cnt = np.array(list(n_sit_per_sp.values()))
    print(f'坐下数/(生,段)：中位 {int(np.median(cnt))} p90 {int(np.percentile(cnt,90))} '
          f'max {cnt.max()} | 合并后总坐下数 {int(cnt.sum())}', flush=True)

    # ---------- 截断构造 ----------
    def trunc_by_exercise(e):
        rows, R = [], []
        for (s, p), idxs in sp_rows.items():
            if len(idxs) < e:
                continue
            sub = idxs[:e]
            cnt_ = raw_ex[sub, 1:8].sum(0)
            n = raw_ex[sub, 0].sum()
            span = (raw_ex[sub, 11].max() - raw_ex[sub, 10].min()) / 3600.0
            R.append([cnt_[0], cnt_[1], cnt_[2], cnt_[3], cnt_[4], cnt_[5], cnt_[6],
                      n, float(e), max(span, 1e-3), raw_ex[sub, 8].max(),
                      raw_ex[sub, 9].min()])
            rows.append(part_key[(s, p)])
        return np.array(rows), np.array(R)

    def trunc_by_sitting(k, with_sit_feats=True):
        rows, R, S = [], [], []
        for (s, p), sids in sittings.items():
            if len(sids) < k:
                continue
            use = [i for sit in sids[:k] for i in sit]
            sub = np.array(use)
            cnt_ = raw_ex[sub, 1:8].sum(0)
            n = raw_ex[sub, 0].sum()
            span = (raw_ex[sub, 11].max() - raw_ex[sub, 10].min()) / 3600.0
            R.append([cnt_[0], cnt_[1], cnt_[2], cnt_[3], cnt_[4], cnt_[5], cnt_[6],
                      n, float(len(use)), max(span, 1e-3), raw_ex[sub, 8].max(),
                      raw_ex[sub, 9].min()])
            # 坐下结构前缀特征
            sit_spans, sit_ev, sit_sub, gaps = [], [], [], []
            prev_end = None
            for sit in sids[:k]:
                si = np.array(sit)
                sp_h = (raw_ex[si, 11].max() - raw_ex[si, 10].min()) / 3600.0
                sit_spans.append(sp_h)
                sit_ev.append(raw_ex[si, 0].sum())
                sit_sub.append(raw_ex[si, 7].sum())
                if prev_end is not None:
                    gaps.append((raw_ex[si, 10].min() - prev_end) / 3600.0)
                prev_end = raw_ex[si, 11].max()
            ss = np.array(sit_spans)
            se = np.array(sit_ev)
            S.append([
                float(k), float(len(use)), float(np.mean(ss)), float(np.median(ss)),
                float(np.max(ss)), float(np.min(ss)), float(np.std(ss)),
                float(np.mean(se)), float(np.max(se)),
                float(np.sum(sit_sub) / max(n, 1)),
                float(np.mean(gaps)) if gaps else 0.0,
                float(np.max(gaps)) if gaps else 0.0,
                float(len(gaps)) / max(k - 1, 1),
                float(np.sum(se) / max(np.sum(ss), 1e-6)),
            ])
            rows.append(part_key[(s, p)])
        return np.array(rows), np.array(R), np.array(S)

    def evaluate(rows, RAWt, extra=None, tag=''):
        part = dpart[rows]
        lab = ddrop[rows]
        SPAN = np.maximum(RAWt[:, 9], 1e-3)
        INS = np.maximum(RAWt[:, 0], 1e-6)
        n_ex = RAWt[:, 8]
        base = np.array([part_tot[int(p)] for p in part], dtype=float)
        A = np.column_stack([np.log1p(RAWt[:, 7]), np.log1p(n_ex), np.log1p(RAWt[:, 9]),
                             np.log1p(RAWt[:, 7] / SPAN)])
        B = np.column_stack([RAWt[:, 1] / INS, RAWt[:, 4] / SPAN,
                             RAWt[:, 6] / np.maximum(RAWt[:, 7], 1)])
        C = np.column_stack([n_ex / base, ds['stage_idx'][rows].astype(float)])
        pos_of_row = {int(r): i for i, r in enumerate(rows)}
        oof = np.zeros(len(rows))
        for tr, te in folds:
            tr_rows = np.array([r for r in tr if int(r) in pos_of_row])
            te_rows = np.array([r for r in te if int(r) in pos_of_row])
            if len(tr_rows) < 50 or len(te_rows) == 0:
                continue
            it = np.array([pos_of_row[int(r)] for r in tr_rows])
            ie = np.array([pos_of_row[int(r)] for r in te_rows])
            stats = peer_stats(RAWt, it, part)
            ztr = group_peer_z(RAWt, it, part, stats)
            zte = group_peer_z(RAWt, ie, part, stats)
            Xtr = np.hstack([A[it], B[it], C[it], ztr, SELF_all[tr_rows]])
            Xte = np.hstack([A[ie], B[ie], C[ie], zte, SELF_all[te_rows]])
            if extra is not None:
                Xtr = np.hstack([Xtr, extra[it]])
                Xte = np.hstack([Xte, extra[ie]])
            oof[ie] = fit_rf(Xtr, lab[it], Xte, seeds)
        m = sc.metrics(lab, oof, part, tag)
        return m, oof, rows

    print('\n=== A) 习题截断（IPSD 复现）===')
    for e in args.ex_cutoffs:
        t0 = time.time()
        rows, RAWt = trunc_by_exercise(e)
        m, _, _ = evaluate(rows, RAWt, None, f'ex{e}')
        print(f'  前 {e:<4} 题  行数 {len(rows):<6} part 内 pooled = {m["pooled"]:.4f} '
              f'[{time.time()-t0:.0f}s]', flush=True)

    print('\n=== B) 坐下截断（不含会话结构特征）===')
    for k in args.sit_cutoffs:
        t0 = time.time()
        rows, RAWt, S = trunc_by_sitting(k, False)
        m, _, _ = evaluate(rows, RAWt, None, f'sit{k}')
        print(f'  前 {k:<4} 次  行数 {len(rows):<6} part 内 pooled = {m["pooled"]:.4f} '
              f'[{time.time()-t0:.0f}s]', flush=True)

    print('\n=== C) 坐下截断 + 会话结构前缀特征（14 维）===')
    for k in args.sit_cutoffs:
        t0 = time.time()
        rows, RAWt, S = trunc_by_sitting(k, True)
        m, oof, rows_ = evaluate(rows, RAWt, S, f'sit{k}_feat')
        print(f'  前 {k:<4} 次  行数 {len(rows):<6} part 内 pooled = {m["pooled"]:.4f} '
              f'[{time.time()-t0:.0f}s]', flush=True)
        np.savez_compressed(os.path.join(args.out, f'sit_k{k}_feat.npz'),
                            rows=rows, oof=oof, y=ddrop[rows], part=dpart[rows])

    os.makedirs(args.out, exist_ok=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--ex', default='data/exercise_dataset.npz')
    ap.add_argument('--out', default='results/sitting')
    ap.add_argument('--ex-cutoffs', type=int, nargs='+', default=[1, 2, 3, 5, 8])
    ap.add_argument('--sit-cutoffs', type=int, nargs='+', default=[1, 2, 3, 5])
    main(ap.parse_args())
