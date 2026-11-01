#!/usr/bin/env python3
"""
CS1-SHR 学生级打分器：组合方式定稿实验
=================================================================================
run_early2 的结论（正确设定的双通道 vs 扁平基线）：
  k=1 双通道 0.7359 > 扁平 RF 0.6997 / 扁平 HGB 0.7126   ← 最早时点双通道赢
  k=2/3/4 扁平 0.8252 / 0.8741 / 0.8956 > 双通道 0.7990 / 0.7854 / 0.8201
  → 终局通道（只在完课者上训的 LogReg）太弱（0.60–0.68）→ 把它换成
    **在全部学生上训练、同一套特征扁平模型**（它本身就是最好的"终局/存活"估计），
    再和掉队通道组合，看哪种组合公式最优。

比较的组合：
  A) 扁平模型（RF / HGB）              —— 基线
  B) 掉队通道 P_drop_by_k              —— 早期强、后期弱
  C) 概率式融合 P_drop·q_drop + (1−P_drop)·P_flat
  D) 秩平均 0.5·rank(P_drop) + 0.5·rank(P_flat)
  E) 概率式融合，权重 0.3/0.7（更看重扁平）
输出：全样本 AUC + 仍活跃子集 AUC（干预场景）
"""
import argparse
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402
from analysis.hazard_model.run_dchm import build_features  # noqa: E402


def prefix_snapshot(X, sid, sidx, k):
    n = len(sid)
    out = np.zeros((n, X.shape[1] * 3 + 1))
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        idx = idx[np.argsort(sidx[idx])]
        use = idx[sidx[idx] < k] if (sidx[idx] < k).any() else idx[:1]
        Xs = X[use]
        out[idx] = np.concatenate([Xs[-1], Xs.mean(0), Xs[-1] - Xs[0], [len(use)]])
    return out


def main(args):
    ds = sc.load_dataset(args.data)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part_ids'])
    ds['DER'] = sc.derive(ds)
    ds['SELF'] = sc.self_ref(ds)
    y = ds['drop'].astype(int)
    part, sid, sidx = ds['part'], ds['sid'], ds['stage_idx']
    n = len(y)
    stu = [int(s) for s in np.unique(sid)]
    pos = {s: i for i, s in enumerate(stu)}
    student_final = {int(s): int(ds['final'][sid == s][0]) for s in stu}
    yvec = np.array([student_final[s] for s in stu])
    max_part = {}
    for s, p in zip(sid, part):
        max_part[int(s)] = max(max_part.get(int(s), 0), int(p))
    order_key = sidx * 10 + part
    folds = sc.group_folds(sid, ds['final'], n_splits=5, seed=42)

    Z = np.load(args.oof, allow_pickle=True)
    h = Z['rf_full']

    for k in args.ks:
        oof = {a: np.zeros(len(stu)) for a in ('rf', 'hgb')}
        for tr, te in folds:
            Xtr, Xte = build_features(ds, tr, te, mode='rf29')
            Ptr = prefix_snapshot(Xtr, sid[tr], sidx[tr], k)
            Pte = prefix_snapshot(Xte, sid[te], sidx[te], k)

            def snap(rows, P):
                out, names = [], []
                for s in sorted(set(int(x) for x in sid[rows])):
                    idx = np.where(sid[rows] == s)[0]
                    out.append(P[idx[0]])
                    names.append(s)
                return np.array(out), names
            tr_s = set(int(s) for s in sid[tr])
            Str, s_tr = snap(tr, Ptr)
            Ste, s_te = snap(te, Pte)
            ytr = [student_final[s] for s in s_tr]
            for name, mk in (('rf', lambda: RandomForestClassifier(
                    n_estimators=500, min_samples_leaf=2, class_weight='balanced',
                    random_state=42, n_jobs=-1)),
                             ('hgb', lambda: HistGradientBoostingClassifier(
                    max_depth=3, max_iter=300, learning_rate=0.06,
                    min_samples_leaf=10, l2_regularization=1.0,
                    class_weight='balanced', random_state=42))):
                mdl = mk()
                mdl.fit(Str, ytr)
                for s, p in zip(s_te, mdl.predict_proba(Ste)[:, 1]):
                    oof[name][pos[s]] = p

        # 掉队通道
        pdrop = np.zeros(len(stu))
        alive = np.zeros(len(stu), dtype=bool)
        for i, s in enumerate(stu):
            idx = np.where(sid == s)[0]
            idx = idx[np.argsort(order_key[idx])]
            keep = idx[sidx[idx] < k] if (sidx[idx] < k).any() else idx[:1]
            pdrop[i] = 1 - float(np.prod(np.clip(1 - h[keep], 1e-9, 1)))
            alive[i] = bool((sidx[idx] >= k).any())

        q_drop = 0.988
        flat = oof['hgb']
        comb = {
            'A) 扁平 HGB': flat,
            'B) 掉队通道': pdrop,
            'C) 概率融合 P_drop·q_drop+(1−P_drop)·P_flat':
                pdrop * q_drop + (1 - pdrop) * flat,
            'D) 秩平均 0.5/0.5':
                0.5 * rankdata(pdrop) / len(pdrop) + 0.5 * rankdata(flat) / len(flat),
            'E) 概率融合 0.3/0.7':
                0.3 * pdrop * q_drop + 0.7 * flat,
        }
        print(f'--- cutoff k={k} ---')
        for name, v in comb.items():
            a = roc_auc_score(yvec, v)
            sub = (roc_auc_score(yvec[alive], v[alive])
                   if len(set(yvec[alive].tolist())) > 1 else float('nan'))
            print(f'  {name:<48} 全样本={a:.4f} | 存活子集 n={int(alive.sum())} AUC={sub:.4f}',
                  flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--oof', default='results/dchm/oof_run1.npz')
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 2, 3, 4])
    main(ap.parse_args())
