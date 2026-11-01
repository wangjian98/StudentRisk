#!/usr/bin/env python3
"""
任务 2 · 解释层工程：per-stage 危险比 + stage-conditioned 重要性 + counterfactual
=================================================================================
(a) **per-stage 危险比**：对每个阶段单独拟合 L2 惩罚 logistic（稳定），exp(系数) 作为
    "该阶段内，行为每 +1 SD 对掉队几率的影响"（HR / OR）；并给出跨阶段的对照表。
(b) **stage-conditioned 重要性**：对每折模型，在该阶段内的测试行上**打乱特征并重新预测**，
    用段内 AUC 的下降量作为该特征在该阶段的重要性（正确的置换定义）。
(c) **counterfactual（M7）**：对被标记学生，找出"把哪个特征改到什么值"能把风险压到阈值以下。

输出：results/explain/*.json + 控制台表格
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402
from analysis.hazard_model.run_dchm import build_features  # noqa: E402

RAW_NAMES = ['ins', 'rem', 'paste', 'focus_gained', 'focus_lost', 'run', 'submit',
             'total', 'n_ex', 'span_h']
FEAT_NAMES = (
    ['log_events', 'log_n_ex', 'log_span', 'log_events_per_span',
     'rem_over_ins', 'focuslost_per_span', 'submit_rate',
     'completion', 'stage_idx']
    + [f'peer_z_{n}' for n in RAW_NAMES]
    + [f'self_z_{n}' for n in RAW_NAMES]
)


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
    os.makedirs(args.out, exist_ok=True)

    X_all, _ = build_features(ds, np.arange(n), np.arange(n), mode='rf29')
    D = X_all.shape[1]

    # ---------------- (a) per-stage 危险比 ----------------
    print('=== (a) per-stage 危险比（每阶段单独拟合 L2 logistic，exp(β)=HR）===')
    hr_rows = []
    for p in range(1, 8):
        rows = np.where(part == p)[0]
        if len(rows) < 60 or len(set(y[rows].tolist())) < 2:
            continue
        Xp = X_all[rows]
        mu, sd = Xp.mean(0), Xp.std(0)
        sd[sd < 1e-9] = 1.0
        Xp = (Xp - mu) / sd
        lr = LogisticRegression(C=1.0 / args.hr_C, max_iter=5000, class_weight='balanced')
        lr.fit(Xp, y[rows])
        co = lr.coef_[0]
        for j, nm in enumerate(FEAT_NAMES):
            hr_rows.append(dict(stage=p, feature=nm, hr=float(np.exp(co[j])),
                                coef=float(co[j]), n=int(len(rows))))
    # 打印：每阶段 |log HR| 最大的 6 个（HR>1 记为升风险）
    for p in range(1, 8):
        sub = [r for r in hr_rows if r['stage'] == p]
        if not sub:
            continue
        sub.sort(key=lambda r: -abs(r['coef']))
        base = np.mean([r['hr'] for r in hr_rows if r['stage'] == p])
        print(f'\n  【第 {p} 段】n={sub[0]["n"]}  前 6（HR>1 = 该特征升高掉队几率）：')
        for r in sub[:6]:
            arrow = '↑' if r['hr'] > 1 else '↓'
            print(f'    {r["feature"]:<24} HR={r["hr"]:.3f} {arrow}')
    with open(os.path.join(args.out, 'hazard_ratios.json'), 'w') as f:
        json.dump(dict(rows=hr_rows, names=FEAT_NAMES), f, indent=2, ensure_ascii=False)

    # ---------------- (b) stage-conditioned 重要性 ----------------
    print('\n=== (b) stage-conditioned 特征重要性（段内打乱特征→重新预测→AUC 下降）===')
    oof = np.zeros(n)
    fold_models = []
    for tr, te in folds:
        Xtr, Xte = build_features(ds, tr, te, mode='rf29')
        acc = np.zeros(len(te))
        m0 = None
        for sd in seeds:
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       max_features='sqrt', class_weight='balanced',
                                       random_state=sd, n_jobs=-1)
            m.fit(Xtr, y[tr])
            acc += m.predict_proba(Xte)[:, 1]
            if sd == seeds[0]:
                m0 = m
        oof[te] = acc / len(seeds)
        fold_models.append(dict(model=m0, Xte=Xte, te=te, Xtr=Xtr))
    rng = np.random.default_rng(7)
    imp_by_stage = {}
    for p in range(1, 8):
        rows_all = np.where(part == p)[0]
        if len(rows_all) < 40 or len(set(y[rows_all].tolist())) < 2:
            continue
        base = roc_auc_score(y[rows_all], oof[rows_all])
        drops = np.zeros(D)
        for j in range(D):
            ds_ = []
            for _ in range(args.perm_reps):
                perm_pred = np.zeros(len(rows_all))
                # 用各折模型对"该阶段行"重新预测（打乱特征 j）
                for fm in fold_models:
                    te = fm['te']
                    m_rows = te[part[te] == p]
                    if len(m_rows) == 0:
                        continue
                    Xp = fm['Xte'][np.searchsorted(te, m_rows)].copy() \
                        if False else fm['Xte'][[i for i, r in enumerate(te) if r in set(m_rows.tolist())]].copy()
                    Xp[:, j] = Xp[np.random.permutation(len(Xp)), j]
                    pr = fm['model'].predict_proba(Xp)[:, 1]
                    for r, v in zip(m_rows, pr):
                        perm_pred[np.searchsorted(rows_all, r)] = v
                ds_.append(base - roc_auc_score(y[rows_all], perm_pred))
            drops[j] = float(np.mean(ds_))
        imp_by_stage[p] = dict(base_auc=float(base), n=int(len(rows_all)),
                                imp=drops.tolist())
        order = np.argsort(-drops)[:6]
        print(f'  第 {p} 段（n={len(rows_all)}，基线 AUC {base:.3f}）前 6：' +
              ' '.join(f'{FEAT_NAMES[j]}={drops[j]:.4f}' for j in order))
    with open(os.path.join(args.out, 'stage_importance.json'), 'w') as f:
        json.dump(dict(importance=imp_by_stage, names=FEAT_NAMES), f, indent=2,
                  ensure_ascii=False)

    # ---------------- (c) counterfactual ----------------
    print('\n=== (c) counterfactual：把哪个特征改到什么值能把风险压到阈值以下 ===')
    fm = fold_models[0]
    m0, Xte, te = fm['model'], fm['Xte'], fm['te']
    peer_med = {}
    for p in np.unique(part[te]):
        idx = te[part[te] == p]
        peer_med[int(p)] = np.median(Xte[[i for i, r in enumerate(te) if r in set(idx.tolist())]], 0)
    cf_rows = []
    shown = 0
    for i in range(len(te)):
        if y[te[i]] != 1:
            continue
        x = Xte[i]
        s = float(m0.predict_proba(x.reshape(1, -1))[0, 1])
        if s < 0.7:
            continue
        pm = peer_med.get(int(part[te[i]]), np.median(Xte, 0))
        best = None
        # (i) 单特征：调到同伴中位 / p25 / p75 / 全局中位
        for j in range(D):
            for tgt in (pm[j], np.percentile(Xte[:, j], 75),
                        np.percentile(Xte[:, j], 25), float(np.median(Xte[:, j]))):
                x2 = x.copy()
                x2[j] = tgt
                s2 = float(m0.predict_proba(x2.reshape(1, -1))[0, 1])
                ok = (s2 < args.cf_threshold) or (s2 <= s - 0.25)
                if ok and (best is None or s2 < best[1]):
                    best = (j, s2, float(tgt), float(x[j]))
        # (ii) 双特征组合：若单特征不够，取"贡献最负"的两个特征同时调到同伴中位
        if best is None:
            contrib = m0.feature_importances_
            cand = np.argsort(contrib)[::-1][:args.cf_topk]
            for j1, j2 in [(cand[a], cand[b]) for a in range(len(cand))
                           for b in range(a + 1, len(cand))]:
                x2 = x.copy()
                x2[j1] = pm[j1]
                x2[j2] = pm[j2]
                s2 = float(m0.predict_proba(x2.reshape(1, -1))[0, 1])
                ok = (s2 < args.cf_threshold) or (s2 <= s - 0.25)
                if ok and (best is None or s2 < best[1]):
                    best = (j1, s2, float(pm[j1]), float(x[j1]),
                            FEAT_NAMES[j2])
        if best is None:
            continue
        if best is None:
            continue
        if len(best) == 5:
            j, s2, tgt, old, second = best
            cf_rows.append(dict(student=int(sid[te[i]]), part=int(part[te[i]]), risk=s,
                                fix_feature=FEAT_NAMES[j], from_value=old,
                                to_value=tgt, new_risk=s2, also=second,
                                mode='pair'))
            print(f'  学生 #{int(sid[te[i]])} · 第 {int(part[te[i]])} 段 · 风险 {s:.3f} → '
                  f'把 {FEAT_NAMES[j]} 由 {old:.3f} 调到 {tgt:.3f}（同伴中位）'
                  f'并同时调 {second} → 风险 {s2:.3f}')
        else:
            j, s2, tgt, old = best
            cf_rows.append(dict(student=int(sid[te[i]]), part=int(part[te[i]]), risk=s,
                                fix_feature=FEAT_NAMES[j], from_value=old,
                                to_value=tgt, new_risk=s2, mode='single'))
            print(f'  学生 #{int(sid[te[i]])} · 第 {int(part[te[i]])} 段 · 风险 {s:.3f} → '
                  f'把 {FEAT_NAMES[j]} 由 {old:.3f} 调到 {tgt:.3f}（同伴中位）→ 风险 {s2:.3f}')
        shown += 1
        if shown >= args.cf_examples:
            break
    with open(os.path.join(args.out, 'counterfactuals.json'), 'w') as f:
        json.dump(cf_rows, f, indent=2, ensure_ascii=False)
    print(f'  共生成 {len(cf_rows)} 条 counterfactual')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--out', default='results/explain')
    ap.add_argument('--hr-C', type=float, default=2.0)
    ap.add_argument('--perm-reps', type=int, default=5)
    ap.add_argument('--cf-threshold', type=float, default=0.3)
    ap.add_argument('--cf-examples', type=int, default=6)
    ap.add_argument('--cf-topk', type=int, default=8)
    main(ap.parse_args())
