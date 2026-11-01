#!/usr/bin/env python3
"""
LOPO（跨 part 泛化）对比：MetaMamba(四项全上) vs RF（同口径）
==========================================================
每个留出 part 上：train = 其余 part，test = 留出 part。
输出：
  · 逐 part AUC 对照表（MM vs RF vs Δ）
  · n 加权 LOPO AUC
  · 配对 bootstrap（part 内分层重采样）Δ 与 P(Δ≤0)

用法：python3 compare_lopo.py --data data/stage_dataset.npz --out results/stage_experiment
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='results/stage_experiment')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--mm-tag', default='lopo_mm11_np_ctxraw')
    ap.add_argument('--rf-tag', default='lopo_rf')
    args = ap.parse_args()

    D = np.load(args.data, allow_pickle=True)
    y = D['drop'].astype(int)
    part = D['part_ids'].astype(int)

    def load(tag):
        pj = os.path.join(args.out, f'stage_lopo_{tag}.json')
        pn = os.path.join(args.out, f'stage_lopo_oof_{tag}.npz')
        if not (os.path.exists(pj) and os.path.exists(pn)):
            return None, None
        return json.load(open(pj)), np.load(pn, allow_pickle=True)

    mmj, mmz = load(args.mm_tag)
    rfj, rfz = load(args.rf_tag)
    if mmj is None or rfj is None:
        print(f'缺文件：mm={mmj is not None} rf={rfj is not None}', file=sys.stderr)
        return

    mm_name = list(mmj['models'].keys())[0]
    rf_name = list(rfj['models'].keys())[0]
    mm_oof = mmz[mm_name]
    rf_oof = rfz[rf_name]
    parts = mmj['meta']['held_out_parts']

    print('=' * 88)
    print(f'LOPO 跨 part 泛化对照 · {mm_name}  vs  {rf_name}'
          f'（seeds={mmj["meta"]["seeds"]}, epochs={mmj["meta"]["epochs"]}）')
    print('=' * 88)
    print(f'{"留出part":>10}{"n":>7}{"正例":>7}{"MM AUC":>10}{"RF AUC":>10}{"Δ":>9}')
    print('-' * 88)
    per = {}
    for p in parts:
        m = part == p
        yt = y[m]
        if len(set(yt.tolist())) < 2:
            continue
        a_mm = float(roc_auc_score(yt, mm_oof[m]))
        a_rf = float(roc_auc_score(yt, rf_oof[m]))
        per[p] = (m, a_mm, a_rf)
        print(f'{p:>10}{int(m.sum()):>7}{int(yt.sum()):>7}{a_mm:>10.4f}{a_rf:>10.4f}'
              f'{a_mm - a_rf:>+9.4f}')
    w = np.array([per[p][0].sum() for p in per], float)
    a_mm = np.array([per[p][1] for p in per])
    a_rf = np.array([per[p][2] for p in per])
    wm = float(np.average(a_mm, weights=w))
    wr = float(np.average(a_rf, weights=w))
    print('-' * 88)
    print(f'{"加权LOPO AUC":>24}{"":>7}{"":>7}{wm:>10.4f}{wr:>10.4f}{wm - wr:>+9.4f}')
    if 4 in per:
        print(f'{"主口径 part4":>24}{"":>7}{"":>7}{per[4][1]:>10.4f}{per[4][2]:>10.4f}'
              f'{per[4][1] - per[4][2]:>+9.4f}')

    # 配对 bootstrap：每个 part 内重采样 → 加权平均 Δ
    rng = np.random.default_rng(20261031)
    ds_ = np.empty(args.boot)
    for it in range(args.boot):
        num = den = 0.0
        for p in per:
            m, _, _ = per[p]
            idx = np.where(m)[0]
            pick = rng.choice(idx, len(idx), replace=True)
            yt = y[pick]
            if len(set(yt.tolist())) < 2:
                continue
            d = roc_auc_score(yt, mm_oof[pick]) - roc_auc_score(yt, rf_oof[pick])
            num += d * len(idx)
            den += len(idx)
        ds_[it] = num / den if den else np.nan
    ds_ = ds_[~np.isnan(ds_)]
    print()
    print(f'  配对 bootstrap（{args.boot} 次，part 内重采样）：'
          f'Δ加权LOPO AUC={wm - wr:+.4f} '
          f'[{np.percentile(ds_, 2.5):+.4f},{np.percentile(ds_, 97.5):+.4f}] '
          f'P(Δ≤0)={(ds_ <= 0).mean() * 100:.2f}%')

    out = {'mm': mm_name, 'rf': rf_name,
           'per_part': {str(p): dict(n=int(per[p][0].sum()), mm_auc=per[p][1], rf_auc=per[p][2])
                        for p in per},
           'weighted_mm': wm, 'weighted_rf': wr, 'delta': wm - wr,
           'bootstrap': dict(lo=float(np.percentile(ds_, 2.5)),
                             hi=float(np.percentile(ds_, 97.5)),
                             p_le0=float((ds_ <= 0).mean()))}
    with open(os.path.join(args.out, 'stage_lopo_summary.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'\n写出 {args.out}/stage_lopo_summary.json')


if __name__ == '__main__':
    main()
