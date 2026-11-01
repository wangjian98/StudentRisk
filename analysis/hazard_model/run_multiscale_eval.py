#!/usr/bin/env python3
"""
全轨迹多尺度特征是否带来增量？（rf29 + 14 维会话结构特征）
=================================================================================
背景：S8 显示 max_len 128→1024 仅 +0.4%，且每生事件数跨约 4 个数量级，
      大量学生的序列在 128 处被左填零 → 序列主干实际只看尾部。
      M4 证明"事件间隔/密度类强度统计"无增益 → 但那是对**截断后尾部**做的统计。
      本实验对每个 (学生,part) 的**全部事件**做会话级结构切分（gap>30min），
      检验"结构信息（而非统计量、且不截断）"是否补得上这块缺口。

对照：
  A) RF(rf29)                基线，期望 pooled 0.9258
  B) RF(rf29 + 多尺度 14 维)
  C) LGBM(rf29 + 多尺度 14 维)
  D) RF(仅多尺度 14 维)
判胜：part 内配对 bootstrap（2000 次）
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


def fit_lgbm(Xtr, ytr, Xte, seeds):
    import lightgbm as lgb
    acc = np.zeros(len(Xte))
    for sd in seeds:
        m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.04, num_leaves=15,
                               min_child_samples=20, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, reg_lambda=1.0, class_weight='balanced',
                               random_state=sd, n_jobs=-1, verbose=-1)
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

    M = np.load(args.ms, allow_pickle=True)
    mkey = {}
    for i in range(len(M['sid'])):
        mkey[(int(M['sid'][i]), int(M['part'][i]))] = i
    rows = np.array([mkey[(int(s), int(p))] for s, p in zip(sid, part)])
    ms = M['feats'][rows].astype(np.float64)
    print(f'多尺度特征：dim={ms.shape[1]}  names={list(M["names"])}', flush=True)

    def msz(X, tr, te):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-9] = 1.0
        return (X[tr] - mu) / sd, (X[te] - mu) / sd

    oof = {k: np.zeros(n) for k in ('rf29', 'rf29_ms', 'lgbm29_ms', 'ms_only')}
    t0 = time.time()
    for fi, (tr, te) in enumerate(folds):
        Xtr, Xte = build_features(ds, tr, te, mode='rf29')
        Ztr, Zte = msz(ms, tr, te)
        oof['rf29'][te] = fit_rf(Xtr, y[tr], Xte, seeds)
        oof['rf29_ms'][te] = fit_rf(np.hstack([Xtr, Ztr]), y[tr], np.hstack([Xte, Zte]), seeds)
        oof['lgbm29_ms'][te] = fit_lgbm(np.hstack([Xtr, Ztr]), y[tr], np.hstack([Xte, Zte]), seeds)
        oof['ms_only'][te] = fit_rf(Ztr, y[tr], Zte, seeds)
        print(f'  fold{fi} ({time.time()-t0:.0f}s)', flush=True)

    print('\n=== stage-level（part 内 pooled AUC）===')
    for k, v in oof.items():
        m = sc.metrics(y, v, part, k)
        print(f'  {k:<10} overall={m["auc"]:.4f} pooled={m["pooled"]:.4f} '
              f'judged={m["judged"]:.4f}')

    idx_by_g = [np.where(part == g)[0] for g in np.unique(part)]
    rng = np.random.default_rng(20261101)
    boot = []
    for _ in range(2000):
        boot.append(np.concatenate([rng.choice(ii, len(ii), replace=True)
                                     for ii in idx_by_g]))

    def pooled_of(yy, vv, gg):
        return sc.pooled(sc.per_part_auc(yy, vv, gg))

    base = np.array([pooled_of(y[p], oof['rf29'][p], part[p]) for p in boot])
    print('\n=== vs rf29 基线（配对 bootstrap 2000 次）===')
    for k, v in oof.items():
        if k == 'rf29':
            continue
        obs = pooled_of(y, v, part) - pooled_of(y, oof['rf29'], part)
        bs = np.array([pooled_of(y[p], v[p], part[p]) - b for p, b in zip(boot, base)])
        print(f'  {k:<10} Δpooled={obs:+.4f} 95%CI[{np.percentile(bs,2.5):+.4f},'
              f'{np.percentile(bs,97.5):+.4f}] P(Δ≤0)={float((bs<=0).mean())*100:.2f}%')

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, f'oof_multiscale_{args.tag}.npz'),
                        **{k: v for k, v in oof.items()}, y=y, sid=sid, part=part)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--ms', default='data/multiscale_feats.npz')
    ap.add_argument('--out', default='results/candidates')
    ap.add_argument('--tag', default='run1')
    main(ap.parse_args())
