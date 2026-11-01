#!/usr/bin/env python3
"""
CS1 早期预警 · 正确设定的双通道模型 vs 扁平学生级分类器（v2）
=================================================================================
v1 的错误：把「前 k 段快照」广播到该生**所有**阶段行，再把各阶段 hazard 相乘
→ 连续存活的学生反而被乘出更大的 P(drop)（k=1 时 AUC 0.39 < 随机）。
根因：hazard 必须**随阶段更新**（time-varying covariates），且生存乘积只在
     「已观测阶段」上做。

v2 的正确设定（与提案模型一一对应）：
  掉队通道 Drop channel：
      h_i = P(在阶段 i 掉队 | 已活到阶段 i)，用阶段 i 自己的特征估计
      P_drop_by_k(student) = 1 − Π_{i≤k} (1 − h_i)   ← 只乘已观测阶段
      ⇒ 「到第 k 段为止是否已经掉队」= 早期预警的直接抓手
  终局通道 Terminal channel：
      在「活过第 k 段」的学生上训练 P(最终挂科 | 存活到 k)，特征 = 前 k 段快照
  融合：P(fail) = P_drop_by_k·q_drop + (1 − P_drop_by_k)·P(fail | 存活, 终局通道)

对照基线（同特征、同折）：
  A) 扁平学生级 RF（前 k 段快照）→ 最终挂科
  A2) 扁平学生级 HGB（前 k 段快照）→ 最终挂科
  B) 双通道融合（个体化终局通道）
  C) 双通道融合（常数终局率）

关键指标（两个）：
  ① 全样本 AUC（473 人，最终挂科 314）
  ② **存活子集 AUC**：在第 k 段仍然活跃的学生里，谁会挂科
     ← 这才是真实的干预场景（已经掉队的人不需要预警）
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402
from analysis.hazard_model.run_dchm import build_features  # noqa: E402


def prefix_snapshot(X, sid, sidx, k):
    """前 k 段快照（对每个学生）：[最后一段, 均值, 最后−最初, 用到段数]"""
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
    seeds = [42, 123, 777]
    folds = sc.group_folds(sid, ds['final'], n_splits=5, seed=42)
    stu = np.unique(sid)
    stu_list = [int(s) for s in stu]
    pos = {s: i for i, s in enumerate(stu_list)}
    student_final = {int(s): int(ds['final'][sid == s][0]) for s in stu}
    yvec = np.array([student_final[s] for s in stu_list])
    max_part = {}
    for s, p in zip(sid, part):
        max_part[int(s)] = max(max_part.get(int(s), 0), int(p))
    order_key = sidx * 10 + part

    # ---- 阶段级 hazard 的 OOF（time-varying 特征，每折训练一次，全阶段共用）----
    oof_h = np.zeros(n)
    for tr, te in folds:
        Xtr, Xte = build_features(ds, tr, te, mode='rf29')
        acc = np.zeros(len(te))
        for sd in seeds:
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       max_features='sqrt', class_weight='balanced',
                                       random_state=sd, n_jobs=-1)
            m.fit(Xtr, y[tr])
            acc += m.predict_proba(Xte)[:, 1]
        oof_h[te] = acc / len(seeds)

    for k in args.ks:
        t0 = time.time()
        oof_flat = {a: np.zeros(len(stu)) for a in ('rf', 'hgb')}
        oof_term = np.zeros(len(stu))
        oof_fuse = np.zeros(len(stu))
        oof_fuse_c = np.zeros(len(stu))
        oof_drop = np.zeros(len(stu))
        alive_mask = np.zeros(len(stu), dtype=bool)

        for tr, te in folds:
            Xtr, Xte = build_features(ds, tr, te, mode='rf29')
            Ptr = prefix_snapshot(Xtr, sid[tr], sidx[tr], k)
            Pte = prefix_snapshot(Xte, sid[te], sidx[te], k)
            # 学生级快照矩阵（训练折 / 测试折）
            def snap(rows, P, sset):
                out, names = [], []
                for s in sorted(sset):
                    idx = np.where(sid[rows] == s)[0]
                    if len(idx) == 0:
                        continue
                    out.append(P[idx[0]])
                    names.append(s)
                return np.array(out), names
            tr_s = set(int(s) for s in sid[tr])
            te_s = [s for s in stu_list if s not in tr_s]
            Str, s_tr = snap(tr, Ptr, tr_s)
            Ste, s_te = snap(te, Pte, set(te_s))

            # A) 扁平基线
            for name, mk in (('rf', lambda: RandomForestClassifier(
                    n_estimators=500, min_samples_leaf=2, class_weight='balanced',
                    random_state=42, n_jobs=-1)),
                             ('hgb', lambda: HistGradientBoostingClassifier(
                    max_depth=3, max_iter=300, learning_rate=0.06,
                    min_samples_leaf=10, l2_regularization=1.0,
                    class_weight='balanced', random_state=42))):
                mdl = mk()
                mdl.fit(Str, [student_final[s] for s in s_tr])
                pr = mdl.predict_proba(Ste)[:, 1]
                for s, p in zip(s_te, pr):
                    oof_flat[name][pos[s]] = p

            # 掉队通道：P_drop_by_k = 1 − Π_{i≤k}(1−h_i)（只乘已观测阶段）
            p_drop = {}
            for s in te_s:
                idx = np.where(sid[te] == s)[0]
                idx = idx[np.argsort(order_key[te][idx])]
                keep = idx[sidx[te][idx] < k]
                if len(keep) == 0:
                    keep = idx[:1]
                surv = float(np.prod(np.clip(1 - oof_h[te][keep], 1e-9, 1.0)))
                p_drop[s] = 1.0 - surv
                alive_mask[pos[s]] = len(np.where(sidx[te][idx] >= k)[0]) > 0

            # 终局通道：训练折里「活过第 k 段」的学生（行数 > k）
            surv_tr = []
            for s in s_tr:
                idx = np.where(sid[tr] == s)[0]
                if len(idx) > k:
                    surv_tr.append(s)
            q_drop = float(np.mean([student_final[s] for s in s_tr
                                     if max_part[s] < 7])) if any(
                max_part[s] < 7 for s in s_tr) else 0.99
            comp_tr = [s for s in s_tr if max_part[s] >= 7]
            q_comp = float(np.mean([student_final[s] for s in comp_tr])) if comp_tr else 0.27
            if len(surv_tr) >= 30 and len(set(student_final[s] for s in surv_tr)) > 1:
                idx_c = [i for i, s in enumerate(s_tr) if s in set(surv_tr)]
                lr = LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced')
                lr.fit(Str[idx_c], [student_final[s_tr[i]] for i in idx_c])
                term = lr.predict_proba(Ste)[:, 1]
            else:
                term = np.full(len(s_te), q_comp)
            for s, pt in zip(s_te, term):
                i = pos[s]
                pd_ = p_drop[s]
                oof_term[i] = pt
                oof_fuse[i] = pd_ * q_drop + (1 - pd_) * pt
                oof_fuse_c[i] = pd_ * q_drop + (1 - pd_) * q_comp
                oof_drop[i] = pd_

        def rep(name, v):
            a = float(roc_auc_score(yvec, v))
            sub = roc_auc_score(yvec[alive_mask], v[alive_mask]) \
                if len(set(yvec[alive_mask].tolist())) > 1 else float('nan')
            return f'{name:<34} 全样本 AUC={a:.4f} | 存活子集(n={int(alive_mask.sum())}) AUC={sub:.4f}'

        print(f'--- cutoff k={k} ---')
        print('  ' + rep('A) 扁平学生级 RF', oof_flat['rf']))
        print('  ' + rep('A2) 扁平学生级 HGB', oof_flat['hgb']))
        print('  ' + rep('B) 双通道融合（个体化终局通道）', oof_fuse))
        print('  ' + rep('C) 双通道融合（常数终局率）', oof_fuse_c))
        print('  ' + rep('D) 终局通道单独', oof_term))
        print('  ' + rep('E) 掉队通道 P(drop by k)', oof_drop))
        print(f'   [{time.time()-t0:.0f}s]', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 2, 3])
    main(ap.parse_args())
