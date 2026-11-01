#!/usr/bin/env python3
"""
截止压力特征族是否带来增量？+ 可解释的行为学发现
=================================================================================
A) 判别增量（与 rf29 同协议）：
   rf29 | rf29+deadline9 | deadline9 only
B) 可解释性：对每个阶段级特征做**单变量**风险对照（该特征上/下四分位的掉队率），
   并给出"截止压力"族的行为学发现（可直接写进论文的分析章）。
"""
import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402
from analysis.hazard_model.run_dchm import build_features  # noqa: E402


def fit_rf(Xtr, ytr, Xte, seeds):
    from sklearn.ensemble import RandomForestClassifier
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
    y = ds['drop'].astype(int)
    part, sid = ds['part'], ds['sid']
    n = len(y)
    seeds = [42, 123, 777]
    folds = sc.group_folds(sid, ds['final'], n_splits=5, seed=42)

    D = np.load(args.dl, allow_pickle=True)
    dkey = {(int(D['sid'][i]), int(D['part'][i])): i for i in range(len(D['sid']))}
    rows = np.array([dkey[(int(s), int(p))] for s, p in zip(sid, part)])
    dl = D['feats'][rows].astype(np.float64)
    names = [str(x) for x in D['names']]
    print(f'截止压力特征 dim={dl.shape[1]} {names}', flush=True)

    def z(X, tr, te):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-9] = 1.0
        return (X[tr] - mu) / sd, (X[te] - mu) / sd

    oof = {k: np.zeros(n) for k in ('rf29', 'rf29_dl', 'dl_only')}
    t0 = time.time()
    for tr, te in folds:
        Xtr, Xte = build_features(ds, tr, te, mode='rf29')
        Ztr, Zte = z(dl, tr, te)
        oof['rf29'][te] = fit_rf(Xtr, y[tr], Xte, seeds)
        oof['rf29_dl'][te] = fit_rf(np.hstack([Xtr, Ztr]), y[tr], np.hstack([Xte, Zte]), seeds)
        oof['dl_only'][te] = fit_rf(Ztr, y[tr], Zte, seeds)
    print(f'  5 折完成（{time.time()-t0:.0f}s）', flush=True)

    print('\n=== A) 判别增量（阶段级，part 内 pooled AUC）===')
    for k, v in oof.items():
        m = sc.metrics(y, v, part, k)
        print(f'  {k:<10} overall={m["auc"]:.4f} pooled={m["pooled"]:.4f} '
              f'judged={m["judged"]:.4f}')

    idx_by_g = [np.where(part == g)[0] for g in np.unique(part)]
    rng = np.random.default_rng(20261101)
    boot = [np.concatenate([rng.choice(ii, len(ii), replace=True) for ii in idx_by_g])
            for _ in range(2000)]

    def pooled_of(yy, vv, gg):
        return sc.pooled(sc.per_part_auc(yy, vv, gg))
    base = np.array([pooled_of(y[p], oof['rf29'][p], part[p]) for p in boot])
    print('\n=== vs rf29 基线（配对 bootstrap 2000 次）===')
    for k, v in oof.items():
        if k == 'rf29':
            continue
        obs = pooled_of(y, v, part) - pooled_of(y, oof['rf29'], part)
        bs = np.array([pooled_of(y[p], v[p], part[p]) - b for p, b in zip(boot, base)])
        print(f'  {k:<10} Δpooled={obs:+.4f} '
              f'95%CI[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] '
              f'P(Δ≤0)={float((bs<=0).mean())*100:.2f}%')

    # ---------- B) 可解释的行为学发现 ----------
    print('\n=== B) 截止压力特征的单变量掉队率对照（阶段级，257 个掉队事件）===')
    print(f'{"特征":<16}{"低值四分位":>12}{"高值四分位":>12}{"风险比":>10}{"n(低)":>8}{"n(高)":>8}')
    for j, nm in enumerate(names):
        v = dl[:, j]
        q1, q3 = np.percentile(v, [25, 75])
        lo = v <= q1
        hi = v >= q3
        if lo.sum() < 30 or hi.sum() < 30:
            continue
        r_lo = y[lo].mean()
        r_hi = y[hi].mean()
        rr = (r_hi / r_lo) if r_lo > 0 else float('inf')
        print(f'{nm:<16}{r_lo:>12.3f}{r_hi:>12.3f}{rr:>10.2f}{int(lo.sum()):>8}{int(hi.sum()):>8}')

    # 关键交互：利用率 × 提交时机
    util = dl[:, names.index('utilization')]
    sub_ratio = dl[:, names.index('submit_ratio')]
    print('\n=== 利用率 × 提交时机 的二维表（掉队率 / 样本数）===')
    uq = np.percentile(util, [33.3, 66.7])
    sq = np.percentile(sub_ratio, [33.3, 66.7])
    for i, (u_lo, u_hi, ul) in enumerate([(-1e9, uq[0], '低利用率'),
                                          (uq[0], uq[1], '中利用率'),
                                          (uq[1], 1e9, '高利用率')]):
        row = []
        for j, (s_lo, s_hi, sl) in enumerate([(-1e9, sq[0], '早交'),
                                              (sq[0], sq[1], '中交'),
                                              (sq[1], 1e9, '压哨')]):
            m = (util > u_lo) & (util <= u_hi) & (sub_ratio > s_lo) & (sub_ratio <= s_hi)
            if m.sum() >= 20:
                row.append(f'{sl}: {y[m].mean():.3f} (n={int(m.sum())})')
            else:
                row.append(f'{sl}: —')
        print(f'  {ul:<10}' + ' | '.join(row))

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'oof_deadline.npz'),
                        **{k: v for k, v in oof.items()}, y=y, sid=sid, part=part)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--dl', default='data/deadline_feats.npz')
    ap.add_argument('--out', default='results/deadline')
    main(ap.parse_args())
