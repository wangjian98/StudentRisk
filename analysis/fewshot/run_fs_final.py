#!/usr/bin/env python3
"""
CS1-FSP · 收尾实验：FS-E（稀有子类）+ 修正后的显著性（逐行分数配对 bootstrap）
=================================================================================
为什么单独跑：
  · 主脚本里 FS-E 调用块当时没插进去（字符串锚点没匹配上），需要补跑
  · 主脚本的显著性用的是"每段一个常数 AUC"当作 OOF 分数 → 段内排序被抹平，
    算出来的 Δ 是假的（−0.43）。本脚本用**逐行分数**（多次重复取均值）重做。

用法（246）：python3 analysis/fewshot/run_fs_final.py --out results/cs1_fs
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402
from analysis.fewshot.run_cs1_fs import (FEATSETS, METHODS, Space, fit_rf,  # noqa: E402
                                         fewshot_curve, fewshot_rare_class,
                                         pooled_from_res)


def main(args):
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    ds = sc.load_dataset(args.data)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part_ids'])
    ds['DER'] = sc.derive(ds)
    ds['SELF'] = sc.self_ref(ds)
    y, part, sid, final = ds['drop'], ds['part'], ds['sid'], ds['final']
    n = len(y)
    DER = ds['DER']
    abs9 = np.hstack([DER[:, 0:4], DER[:, 4:7], DER[:, [7, 8]]]).astype(float)
    peer = sc.peer_z(ds, np.arange(n), np.arange(n), part)
    selfv = ds['SELF'].astype(float)
    space31 = Space(abs9, peer, selfv, part, use_self=True)
    X31 = np.zeros((n, space31.own_space(1)[1].shape[1]))
    for p in np.unique(part):
        rows, Xp = space31.own_space(int(p))
        X31[rows] = Xp

    out = {}

    # ---------------- FS-E 稀有子类 ----------------
    if not args.only_sig:
        print('=== FS-E 稀有子类：走完全程仍挂科 ===', flush=True)
        rows7 = np.where(part == 7)[0]
        X7, y7, sid7 = X31[rows7], final[rows7], sid[rows7]
        folds7 = sc.group_folds(sid7, y7, n_splits=5, seed=42)
        oof7 = np.zeros(len(y7))
        for tr, te in folds7:
            oof7[te] = fit_rf(X7[tr], y7[tr], X7[te])
        rec = {'n': int(len(y7)), 'pos': int(y7.sum()),
               'ref_full_RF_auc': round(float(roc_auc_score(y7, oof7)), 4)}
        res7 = fewshot_rare_class(X7, y7, [1, 2, 4, 8, 16], args.repeats)
        for (method, K), v in res7.items():
            rec[f'{method}|K={K}'] = dict(auc_mean=round(v['auc_mean'], 4),
                                          auc_std=round(v['auc_std'], 4))
        out['fs_e_rare_class'] = rec
        print(f'  n={rec["n"]} pos={rec["pos"]} 全标签RF AUC={rec["ref_full_RF_auc"]}', flush=True)
        for method in METHODS:
            vals = [f'K={K}:{res7.get((method, K), {}).get("auc_mean", "—")}'
                    for K in [1, 2, 4, 8, 16]]
            print(f'  {method:<9} ' + '  '.join(vals), flush=True)

    # ---------------- 修正后的显著性 ----------------
    print('\n=== 显著性（逐行分数，配对 bootstrap 2000）===', flush=True)
    # 参照：段内全标签 RF（5 折按学生，OOF）
    oof_in = np.zeros(n)
    for p in np.unique(part):
        rows = np.where(part == p)[0]
        if len(rows) < 30 or len(set(y[rows].tolist())) < 2:
            continue
        uniq = np.unique(sid[rows])
        rng = np.random.default_rng(42)
        buckets = [[] for _ in range(5)]
        for cls in (0, 1):
            ss = [int(s) for s in uniq if int(y[rows][sid[rows] == s][0]) == cls]
            rng.shuffle(ss)
            for i, s in enumerate(ss):
                buckets[i % 5].append(s)
        for b in range(5):
            te_s = set(buckets[b])
            te = np.array([r for r in rows if int(sid[r]) in te_s])
            tr = np.array([r for r in rows if int(sid[r]) not in te_s])
            if len(tr) < 20 or len(te) == 0:
                continue
            oof_in[te] = fit_rf(X31[tr], y[tr], X31[te])

    configs = [('all29', 'rf', 8), ('all29', 'rf', 16), ('peer10', 'rf', 8),
               ('peer10', 'logit', 2), ('peer_submit', 'logit', 2),
               ('peer_submit', 'rf', 8), ('all29', 'logit', 8)]
    if args.only_sig:
        configs = [('all29', 'rf', 8), ('all29', 'rf', 16),
                   ('peer10', 'logit', 2), ('peer_submit', 'logit', 2)]
    sig = {}
    for fs_name, method, K in configs:
        cols = FEATSETS[fs_name]
        res = fewshot_curve(space31, y, part, [K], args.repeats, cols=cols,
                            methods=[method])
        oof_fs = np.full(n, np.nan)
        for (p, kk, m), v in res.items():
            for r2, s2 in v.get('row_scores', {}).items():
                oof_fs[r2] = s2
        # 重复级配对 bootstrap：按段重采样 + 按重复重采样
        per_part_aucs = {}
        for (p, kk, m), v in res.items():
            per_part_aucs[p] = np.array(v.get('aucs', []), dtype=float)
        ref_pp = sc.per_part_auc(y, oof_in, part)
        ref_pooled = sc.pooled(ref_pp)
        ref_by_part = {k: v['auc'] for k, v in ref_pp.items()}
        weights = {p: v['n_query'] for (p, kk, m), v in res.items()}
        n_rep = min(len(a) for a in per_part_aucs.values())
        rng = np.random.default_rng(20261105)
        deltas = []
        for _ in range(2000):
            pick = rng.integers(0, n_rep, n_rep)
            num = sum(per_part_aucs[p][pick].mean() * weights[p] for p in per_part_aucs)
            den = sum(weights[p] for p in per_part_aucs)
            deltas.append(num / max(den, 1) - ref_pooled)
        deltas = np.array(deltas)
        num = sum(per_part_aucs[p].mean() * weights[p] for p in per_part_aucs)
        den = sum(weights[p] for p in per_part_aucs)
        obs = float(num / max(den, 1) - ref_pooled)
        key = f'{fs_name}|{method}|K={K} vs 段内全标签RF'
        sig[key] = dict(pooled_fewshot=round(float(num / max(den, 1)), 4),
            pooled_full=round(float(ref_pooled), 4),
            obs=round(obs, 4),
            ci=[round(float(np.percentile(deltas, 2.5)), 4),
                round(float(np.percentile(deltas, 97.5)), 4)],
            p_le0=round(float((deltas <= 0).mean()), 4),
            n_rep=n_rep, n_parts=len(per_part_aucs))
        print(f'  {key}: {sig[key]["pooled_fewshot"]:.4f} vs {ref_pooled:.4f} | '
              f'Δ={obs:+.4f} CI[{sig[key]["ci"][0]:+.4f},{sig[key]["ci"][1]:+.4f}] '
              f'P(Δ≤0)={sig[key]["p_le0"]*100:.1f}%', flush=True)

    out['significance_rowwise'] = sig

    with open(os.path.join(args.out, 'cs1_fs_final.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f'\n完成 → {args.out}/cs1_fs_final.json（{time.time()-t0:.0f}s）', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--out', default='results/cs1_fs')
    ap.add_argument('--repeats', type=int, default=12)
    ap.add_argument('--only-sig', action='store_true')
    main(ap.parse_args())
