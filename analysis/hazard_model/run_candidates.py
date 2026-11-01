#!/usr/bin/env python3
"""
候选模型基准（stage-level，统一协议）
=================================================================================
回答「除了 CS1-SHR 的形状，还有哪些模型族适合 StudentRisk/CS1」。

统一协议（与项目主协议一致）：
  · 5 折 **按学生分组** + 课程级标签分层，seeds {42,123,777}
  · 特征：与 RF 基线**逐位同源**的 29 维（rf29）
  · 主指标：**part 内 pooled AUC**（个体判别力口径）

候选：
  A) RF full（基线，期望 pooled 0.9258）
  B) HGB（直方图梯度提升）
  C) LightGBM 分类（GOSS 直方图 GBDT，叶优生长 + L2）
  D) LightGBM **LambdaRank**（组 = part）→ 直接优化 part 内排序 = 主指标本身
  E) **序数/进度深度回归**（预测该生最终到达的 part 深度 → 风险分 = −深度）
  F) **原型混合 MoE**（按轨迹画像 KMeans 分 2 类，各训一个 RF）
  G) **集成**（对各候选 OOF 做秩平均 / 逻辑堆叠）

判负判胜：与 RF 基线做 part 内配对 bootstrap（2000 次）
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402
from analysis.hazard_model.run_dchm import build_features  # noqa: E402


def rank01(v):
    from scipy.stats import rankdata
    return rankdata(v) / len(v)


def fit_predict_rf(Xtr, ytr, Xte, seeds):
    acc = np.zeros(len(Xte))
    for sd in seeds:
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   max_features='sqrt', class_weight='balanced',
                                   random_state=sd, n_jobs=-1)
        m.fit(Xtr, ytr)
        acc += m.predict_proba(Xte)[:, 1]
    return acc / len(seeds)


def fit_predict_hgb(Xtr, ytr, Xte, seeds):
    acc = np.zeros(len(Xte))
    for sd in seeds:
        m = HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.06,
                                            min_samples_leaf=10, l2_regularization=1.0,
                                            class_weight='balanced', random_state=sd)
        m.fit(Xtr, ytr)
        acc += m.predict_proba(Xte)[:, 1]
    return acc / len(seeds)


def fit_predict_lgbm(Xtr, ytr, Xte, seeds):
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


def fit_predict_lgbm_ranker(Xtr, ytr, Xte, seeds, group_tr, group_te):
    """LambdaRank：组 = part（组内排序 = 主指标 part 内 pooled AUC）"""
    import lightgbm as lgb
    acc = np.zeros(len(Xte))
    for sd in seeds:
        m = lgb.LGBMRanker(objective='lambdarank', metric='ndcg', n_estimators=600,
                           learning_rate=0.04, num_leaves=15, min_child_samples=20,
                           colsample_bytree=0.8, reg_lambda=1.0, label_gain=[0, 1],
                           random_state=sd, n_jobs=-1, verbose=-1)
        m.fit(Xtr, ytr, group=group_tr)
        acc += m.predict(Xte)
    return acc / len(seeds)


def main(args):
    import lightgbm  # noqa
    ds = sc.load_dataset(args.data)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part_ids'])
    ds['DER'] = sc.derive(ds)
    ds['SELF'] = sc.self_ref(ds)
    y = ds['drop'].astype(int)
    part, sid, sidx = ds['part'], ds['sid'], ds['stage_idx']
    n = len(y)
    seeds = [42, 123, 777]
    folds = sc.group_folds(sid, ds['final'], n_splits=5, seed=42)

    # 进度深度（该生到达的最远 part）——序数回归目标
    max_part = {}
    for s, p in zip(sid, part):
        max_part[int(s)] = max(max_part.get(int(s), 0), int(p))
    depth = np.array([max_part[int(s)] for s in sid], dtype=float)

    # 学生级画像（用于 MoE 分群）：该生各阶段 peer_z 的均值 + 强度/节奏
    def student_profile(rows, X):
        prof = {}
        for s in sorted(set(int(x) for x in sid[rows])):
            idx = np.where(sid[rows] == s)[0]
            Xs = X[idx]
            prof[s] = np.concatenate([Xs.mean(0), Xs.std(0)])
        return prof

    oof = {k: np.zeros(n) for k in
           ('rf', 'hgb', 'lgbm', 'lgbm_rank', 'ordinal', 'moe')}
    t_all = time.time()

    for fi, (tr, te) in enumerate(folds):
        Xtr, Xte = build_features(ds, tr, te, mode='rf29')
        oof['rf'][te] = fit_predict_rf(Xtr, y[tr], Xte, seeds)
        oof['hgb'][te] = fit_predict_hgb(Xtr, y[tr], Xte, seeds)
        oof['lgbm'][te] = fit_predict_lgbm(Xtr, y[tr], Xte, seeds)

        # D) LambdaRank：组 = part，训练行需按 part 排序（组必须连续）
        order = np.argsort(part[tr], kind='stable')
        ptr = tr[order]
        g_tr = np.bincount(part[ptr]).astype(int)
        order_te = np.argsort(part[te], kind='stable')
        pte = te[order_te]
        g_te = np.bincount(part[pte]).astype(int)
        pred = fit_predict_lgbm_ranker(Xtr[order], y[ptr], Xte[order_te], seeds, g_tr, g_te)
        back = np.zeros(len(te))
        back[order_te] = pred
        oof['lgbm_rank'][te] = back

        # E) 序数/深度回归：目标 = 该生到达的最远 part；风险分 = −深度
        from sklearn.ensemble import HistGradientBoostingRegressor
        acc = np.zeros(len(te))
        for sd in seeds:
            m = HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.06,
                                              min_samples_leaf=10, l2_regularization=1.0,
                                              random_state=sd)
            m.fit(Xtr, depth[tr])
            acc += m.predict(Xte)
        oof['ordinal'][te] = -acc / len(seeds)

        # F) 原型混合 MoE：训练折内 KMeans(k=2) 分原型，各原型一个 RF
        prof_tr = student_profile(tr, Xtr)
        prof_te = student_profile(te, Xte)
        keys_tr = sorted(prof_tr)
        keys_te = sorted(prof_te)
        Ptr = np.array([prof_tr[k] for k in keys_tr])
        Pte = np.array([prof_tr.get(k, prof_tr[keys_tr[0]]) for k in keys_te])
        km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(Ptr)
        c_tr = km.predict(Ptr)
        c_te = km.predict(Pte)
        s_tr = np.array([sid[i] for i in tr])
        s_te = np.array([sid[i] for i in te])
        pos_of = {int(s): i for i, s in enumerate(keys_te)}
        proto_tr = {}
        for s, c in zip(s_tr, c_tr):
            proto_tr.setdefault(int(s), c)
        proto_te = {int(s): int(c) for s, c in zip(s_te, c_te)}
        for c in (0, 1):
            mtr = np.array([i for i, r in enumerate(tr)
                            if proto_tr.get(int(sid[r]), 0) == c], dtype=int)
            mte = np.array([j for j, r in enumerate(te)
                            if proto_te.get(int(sid[r]), 0) == c], dtype=int)
            if len(mtr) < 60 or len(mte) == 0 or len(set(y[tr][mtr].tolist())) < 2:
                if len(mte):
                    oof['moe'][te[mte]] = 0.5
                continue
            oof['moe'][te[mte]] = fit_predict_rf(Xtr[mtr], y[tr][mtr], Xte[mte], seeds)
        print(f'  fold{fi} done ({time.time()-t_all:.0f}s)', flush=True)

    # ---------------- 评估 ----------------
    def metrics(v, tag):
        m = sc.metrics(y, v, part, tag)
        return m

    print('\n=== stage-level 主指标（part 内 pooled AUC）===')
    res = {}
    for k, v in oof.items():
        m = metrics(v, k)
        res[k] = m
        print(f'  {k:<10} overall={m["auc"]:.4f} pooled={m["pooled"]:.4f} '
              f'judged={m["judged"]:.4f}', flush=True)

    # 集成
    ens_rank = np.mean([rank01(oof[k]) for k in ('rf', 'lgbm', 'lgbm_rank')], axis=0)
    res['ens_rank3'] = metrics(ens_rank, 'ens_rank3')
    print(f"  ens_rank3 overall={res['ens_rank3']['auc']:.4f} "
          f"pooled={res['ens_rank3']['pooled']:.4f}")

    # 配对 bootstrap（part 内分层重采样，2000 次）
    print('\n=== vs RF 基线（配对 bootstrap 2000 次，part 内 pooled Δ）===')
    idx_by_g = [np.where(part == g)[0] for g in np.unique(part)]

    def pooled_of(yy, vv, gg):
        return sc.pooled(sc.per_part_auc(yy, vv, gg))

    rng = np.random.default_rng(20261101)
    boot_idx = []
    for _ in range(2000):
        pick = np.concatenate([rng.choice(ii, len(ii), replace=True) for ii in idx_by_g])
        boot_idx.append(pick)
    base_boot = np.array([pooled_of(y[p], oof['rf'][p], part[p]) for p in boot_idx])
    for k, v in oof.items():
        if k == 'rf':
            continue
        obs = pooled_of(y, v, part) - pooled_of(y, oof['rf'], part)
        bs = np.array([pooled_of(y[p], v[p], part[p]) - b for p, b in zip(boot_idx, base_boot)])
        print(f'  {k:<10} Δpooled={obs:+.4f} 95%CI[{np.percentile(bs,2.5):+.4f},'
              f'{np.percentile(bs,97.5):+.4f}] P(Δ≤0)={float((bs<=0).mean())*100:.2f}%',
              flush=True)
    obs = res['ens_rank3']['pooled'] - pooled_of(y, oof['rf'], part)
    bs = np.array([pooled_of(y[p], ens_rank[p], part[p]) - b for p, b in zip(boot_idx, base_boot)])
    print(f'  ens_rank3  Δpooled={obs:+.4f} 95%CI[{np.percentile(bs,2.5):+.4f},'
          f'{np.percentile(bs,97.5):+.4f}] P(Δ≤0)={float((bs<=0).mean())*100:.2f}%')

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f'candidates_{args.tag}.json'), 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'per_part'}
                   for k, v in res.items()}, f, indent=2, ensure_ascii=False, default=float)
    np.savez_compressed(os.path.join(args.out, f'oof_candidates_{args.tag}.npz'),
                        **{k: v for k, v in oof.items()}, ens_rank3=ens_rank,
                        y=y, sid=sid, part=part)
    print(f'\n写出 {args.out}/candidates_{args.tag}.json  总耗时 {time.time()-t_all:.0f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--out', default='results/candidates')
    ap.add_argument('--tag', default='run1')
    main(ap.parse_args())
