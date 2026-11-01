#!/usr/bin/env python3
"""
四项全上公平对照 · 汇总与判定
================================
把"旧实验"（stage_oof_*.npz：rf_full / mm11 / mm7 / mm9 / mm11_nofilm / mm11_paperhp）
与"四项全上"（stage_fair_oof_*.npz：rf_full / rf_full_resid / mm11_np_ctx / mm11_np）
合并到一张表，并做决定性配对 bootstrap：

  判定标准（王建）：MetaMamba 必须**显著优于同口径 RF**
  → 判据：paired bootstrap Δpooled 的 P(Δ≤0) < 5%，且 Δ>0

用法：python3 compare_stage_fair.py --data data/stage_dataset.npz --out results/stage_experiment
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc

W_MAIN = 0.25

OLD = ['rf', 'mm11', 'mm7', 'mm9', 'mm11_nofilm', 'mm11_paperhp']
NEW = ['cpu', 'mm11_np_ctx', 'mm11_np', 'mm11_np_ctxraw']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='results/stage_experiment')
    ap.add_argument('--boot', type=int, default=2000)
    args = ap.parse_args()

    D = np.load(args.data, allow_pickle=True)
    y = D['drop'].astype(int)
    part = D['part_ids'].astype(int)

    oof = {}
    for tag in OLD:
        p = os.path.join(args.out, f'stage_oof_{tag}.npz')
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            for k in z.files:
                oof[f'{tag}:{k}'] = z[k]
    for tag in NEW:
        p = os.path.join(args.out, f'stage_fair_oof_{tag}.npz')
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            for k in z.files:
                oof[f'{tag}:{k}'] = z[k]
    if not oof:
        print('没有找到任何 OOF 文件', file=sys.stderr)
        return

    prior = oof.get('cpu:stage_part_prior')
    if prior is None:
        prior = oof.get('rf:stage_part_prior')
    if prior is None:
        folds = sc.group_folds(D['sid'], D['final'], n_splits=5, seed=42)
        prior = sc.stage_prior_oof(y, part, folds)

    # 统一补一份打分层残差化（w=0.25）的版本
    for k in list(oof.keys()):
        if k.endswith(':stage_part_prior'):
            continue
        base = k.split(':', 1)[1]
        if base.endswith('_resid0.25') or base.endswith('_resid0.5') or base.endswith('_resid1.0'):
            continue
        oof[f'{k}|resid{W_MAIN}'] = np.clip(oof[k] - W_MAIN * prior, 1e-6, 1 - 1e-6)

    rows = []
    for k in sorted(oof.keys()):
        v = oof[k]
        if len(v) != len(y):
            continue
        m = sc.metrics(y, v, part, k)
        rows.append(dict(key=k, auc=m['auc'], pooled=m['pooled'], judged=m['judged'],
                         per_part={kk: round(vv['auc'], 4) for kk, vv in m['per_part'].items()}))
    rows.sort(key=lambda r: -r['pooled'])

    print('=' * 104)
    print(f'阶段级掉队预测 · 全部 arm 汇总（N={len(y)} / 学生 {len(np.unique(D["sid"]))} / '
          f'正例 {int(y.sum())}）')
    print('=' * 104)
    print(f'{"arm":<42}{"总体AUC":>9}{"part内pooled":>14}{"judgeable":>11}')
    print('-' * 104)
    for r in rows:
        print(f'{r["key"]:<42}{r["auc"]:>9.4f}{r["pooled"]:>14.4f}{r["judged"]:>11.4f}')

    pairs = [
        ('mm11_np_ctx:mm11_np_ctx|resid0.25', 'cpu:rf_full_resid|resid0.25',
         '主判定：MM(四项全上) − RF(四项全上)'),
        ('mm11_np_ctx:mm11_np_ctx', 'cpu:rf_full_resid', 'MM(四项全上) − RF(特征残差化)'),
        ('mm11_np:mm11_np', 'cpu:rf_full_resid', '纯去混淆序列 − RF(特征残差化)'),
        ('mm11_np_ctxraw:mm11_np_ctxraw', 'cpu:rf_full',
         'MM(上下文不残差化·最强口径) − RF(原始最强)'),
        ('mm11_np_ctxraw:mm11_np_ctxraw|resid0.25', 'cpu:rf_full|resid0.25',
         'MM(上下文不残差化) − RF(原始最强)，带打分残差'),
        ('mm11_np:mm11_np', 'mm11_np_ctx:mm11_np_ctx', '上下文通道净贡献（ctx − 无ctx）'),
        ('mm11_np:mm11_np', 'mm11_np_ctxraw:mm11_np_ctxraw', '上下文通道净贡献（ctxraw − 无ctx）'),
        ('rf:mm11_paperhp', 'rf:rf_full', '旧实验：MM(论文超参) − RF'),
    ]
    print()
    print('=' * 104)
    print('决定性配对 bootstrap（part 内分层重采样，Δ = pooled AUC 差）')
    print('=' * 104)
    boots = {}
    for a, b, label in pairs:
        if a in oof and b in oof:
            d = sc.paired_boot(y, oof[a], oof[b], part, n=args.boot)
            boots[f'{a}_minus_{b}'] = dict(label=label, **d)
            print(f'  {label}')
            print(f'    {a} − {b}: Δpooled={d["obs"]:+.4f} [{d["lo"]:+.4f},{d["hi"]:+.4f}] '
                  f'P(Δ≤0)={d["p_le0"]*100:.2f}%')
        else:
            print(f'  (跳过) {label} —— 缺 {a if a not in oof else b}')

    out = {'rows': rows, 'bootstrap': boots, 'w_prior_main': W_MAIN}
    with open(os.path.join(args.out, 'stage_fair_summary.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'\n写出 {args.out}/stage_fair_summary.json')


if __name__ == '__main__':
    main()
