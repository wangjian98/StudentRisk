#!/usr/bin/env python3
"""
CS1-FSP · FS-D 修正版：坐下级（段内更早）少样本
=================================================================================
为什么单独重跑：
  1) v1 的"全标签 RF 参照"是**同数据拟合同数据**（in-sample）→ AUC=1.0，必须改成 OOF。
  2) v1 的截断特征只保留了计数/跨度，**没有截止压力（timeToDeadline）**；
     而项目里"坐下级 0.8665"那版特征里含 ttd → 口径不同，不可直接对比。
     本版把 ttd 显式建成 12 列（与 agg10 同序 + ttd_max/ttd_min），并用**OOF** 参照。

协议：每段只给 K 个掉队者 + K 个未掉队者，query=该段其余行；12 次重复；
      指标 = part 内 pooled AUC；参照 = 段内全标签 RF（5 折按学生，OOF）。
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402
from analysis.fewshot.run_cs1_fs import (FEATSETS, METHODS, Space, fit_rf,  # noqa: E402
                                         fewshot_curve, label_prop_knn,
                                         pseudo_rf, proto_score, pooled_from_res,
                                         per_part_table, A_COLS, B_COLS)

SEEDS = [42, 123, 777]


def build_sitting_raw12(ex_path, n_sit=1, gap_sec=30 * 60):
    """截断 agg12 = [ins,rem,pas,fg,fl,run,sub, total, distinct_ex, span_h, ttd_max, ttd_min]"""
    ex = np.load(ex_path, allow_pickle=True)
    raw = ex['raw'].astype(np.float64)
    esid, epar, eexr = ex['sid'], ex['part'], ex['exercise']
    sp_rows = defaultdict(list)
    for i in range(len(esid)):
        sp_rows[(int(esid[i]), int(epar[i]))].append(i)
    out = {}
    for (s, p), idxs in sp_rows.items():
        idxs = sorted(idxs, key=lambda i: (eexr[i], raw[i, 10]))
        sittings, cur = [], [idxs[0]]
        for a, b in zip(idxs[:-1], idxs[1:]):
            if raw[b, 10] - raw[a, 11] < gap_sec:
                cur.append(b)
            else:
                sittings.append(cur)
                cur = [b]
        sittings.append(cur)
        if len(sittings) < n_sit:
            continue
        use = np.array([i for sit in sittings[:n_sit] for i in sit])
        cnt = raw[use, 1:8].sum(0)
        n_ev = raw[use, 0].sum()
        span = max((raw[use, 11].max() - raw[use, 10].min()) / 3600.0, 1e-3)
        out[(s, p)] = np.concatenate([cnt, [n_ev], [float(len(use)), span,
                                                   raw[use, 8].max(), raw[use, 9].min()]])
    return out


def sitting_abs_features(RAW12, part, base_ex, stage_idx):
    """由截断 agg12 派生绝对量特征（阶段级同口径 + 截止压力）"""
    SPAN = np.maximum(RAW12[:, 9], 1e-3)
    INS = np.maximum(RAW12[:, 0], 1e-6)
    n_ex = RAW12[:, 8]
    ttd_h = np.maximum(RAW12[:, 10], 0) / 3600.0
    A = np.column_stack([np.log1p(RAW12[:, 7]), np.log1p(n_ex), np.log1p(RAW12[:, 9]),
                         np.log1p(RAW12[:, 7] / SPAN)])
    B = np.column_stack([RAW12[:, 1] / INS, RAW12[:, 4] / SPAN,
                         RAW12[:, 6] / np.maximum(RAW12[:, 7], 1)])
    C = np.column_stack([n_ex / np.array([base_ex[int(p)] for p in part]),
                         stage_idx.astype(float)])
    D = np.column_stack([np.log1p(ttd_h), np.log1p(RAW12[:, 9] /
                                                    np.maximum(ttd_h, 1e-3))])
    return np.hstack([A, B, C, D]).astype(float)     # 12 维绝对量


def ref_oof_rf(X, y, sid, part, n_splits=5, seed=42):
    """段内 5 折按学生分组 OOF（全标签参照）"""
    oof = np.zeros(len(y))
    for p in np.unique(part):
        rows = np.where(part == p)[0]
        if len(rows) < 30 or len(set(y[rows].tolist())) < 2:
            continue
        uniq = np.unique(sid[rows])
        rng = np.random.default_rng(seed)
        buckets = [[] for _ in range(n_splits)]
        for cls in (0, 1):
            ss = [int(s) for s in uniq if int(y[rows][sid[rows] == s][0]) == cls]
            rng.shuffle(ss)
            for i, s in enumerate(ss):
                buckets[i % n_splits].append(s)
        for b in range(n_splits):
            te_s = set(buckets[b])
            te = np.array([r for r in rows if int(sid[r]) in te_s])
            tr = np.array([r for r in rows if int(sid[r]) not in te_s])
            if len(tr) < 20 or len(te) == 0:
                continue
            oof[te] = fit_rf(X[tr], y[tr], X[te])
    return oof


def main(args):
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    ds = sc.load_dataset(args.data)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part_ids'])
    y, part, sid = ds['drop'], ds['part'], ds['sid']

    part_key = {(int(ds['sid'][i]), int(ds['part'][i])): i for i in range(len(sid))}
    out = {}
    for n_sit in [1, 2]:
        sit_raw = build_sitting_raw12(args.ex, n_sit=n_sit)
        rows, RAW12 = [], []
        for (s, p), v in sit_raw.items():
            i = part_key.get((s, p))
            if i is None:
                continue
            rows.append(i)
            RAW12.append(v)
        rows = np.array(rows)
        RAW12 = np.array(RAW12)
        sub_part = part[rows]
        sub_y = y[rows]
        sub_sid = sid[rows]
        stage_idx = ds['stage_idx'][rows]

        absf = sitting_abs_features(RAW12, sub_part, ds['base_ex'], stage_idx)   # 12
        # 同伴相对：在截断 agg12 上算（12 维）
        peer = sc.peer_z({'RAW': RAW12, 'part': sub_part},
                         np.arange(len(rows)), np.arange(len(rows)), sub_part)   # 12
        sp = Space(absf, peer, np.zeros((len(rows), 1)), sub_part, use_self=False)
        dim = absf.shape[1] + peer.shape[1]
        X = np.zeros((len(rows), dim))
        for p in np.unique(sub_part):
            rr, Xp = sp.own_space(int(p))
            X[rr] = Xp

        oof = ref_oof_rf(X, sub_y, sub_sid, sub_part)
        pp = sc.per_part_auc(sub_y, oof, sub_part)
        rec = {'rows': int(len(rows)),
               'ref_full_RF_pooled_oof': round(sc.pooled(pp), 4),
               'ref_per_part': {str(k): round(v['auc'], 4) for k, v in pp.items()}}

        na = absf.shape[1]
        featsets = {'abs+peer': list(range(dim)),
                    'peer_only': list(range(na, dim)),
                    'peer_submit1': [na + 6]}
        for fs, cols in featsets.items():
            res = fewshot_curve(sp, sub_y, sub_part, [1, 2, 4, 8], args.repeats,
                                cols=cols)
            pooled = pooled_from_res(res)
            for m in METHODS:
                for K in [1, 2, 4, 8]:
                    if (m, K) in pooled:
                        rec[f'{fs}|{m}|K={K}'] = round(pooled[(m, K)], 4)
        out[f'sit{n_sit}'] = rec
        print(f'== sit{n_sit}: rows={rec["rows"]} '
              f'OOF 全标签RF pooled={rec["ref_full_RF_pooled_oof"]} '
              f'({time.time()-t0:.0f}s)', flush=True)
        for fs in featsets:
            print(f'  [{fs}]', flush=True)
            for m in METHODS:
                vals = [f'K={K}:{rec.get(f"{fs}|{m}|K={K}", "—")}' for K in [1, 2, 4, 8]]
                print(f'    {m:<9} ' + '  '.join(vals), flush=True)

    with open(os.path.join(args.out, 'cs1_fs_sitting.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'完成 → {args.out}/cs1_fs_sitting.json（{time.time()-t0:.0f}s）', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--ex', default='data/exercise_dataset.npz')
    ap.add_argument('--out', default='results/cs1_fs')
    ap.add_argument('--repeats', type=int, default=12)
    main(ap.parse_args())
