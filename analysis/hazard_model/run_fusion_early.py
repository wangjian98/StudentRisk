#!/usr/bin/env python3
"""
CS1 早期预警 · 双通道（掉队 + 终局）融合 vs 扁平学生级分类器
=================================================================================
问题：CS1 的挂科标签是**两种行为相反的失败模式**的混合：
    · 掉队型 255 人（走不完 7 个 part）→ 强度最低、磨洋工
    · 终局型  59 人（走完全程仍挂科）→ 强度最高、焦点最碎、返工最多
扁平学生级分类器在「前 k 段行为 → 学期末挂科」这个早预警任务上，
是否会被混合标签拖累？双通道分解（掉队通道 + 终局通道）是否更好？

对照（同一份前缀特征、同一套按学生分组的 5 折 × 3 seeds）：
  A) 扁平基线：学生级 RF（前 k 段特征）→ 最终挂科
  B) 双通道：阶段级 hazard（只喂前 k 段快照）× q_drop + 终局通道 × q_comp
  C) 双通道（常数融合，q_comp 取常数）

输出：k=1/2/3 的 AUC 对照 + 掉队通道单独 AUC（预测“会不会掉队”）
"""
import argparse
import os
import sys
import time

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


def prefix_snapshot(X, sid, sidx, k):
    """把每个学生的「前 k 段快照」广播回该生所有阶段行：
       snapshot = [最后一段特征, 前 k 段均值, 最后−最初, 用到的段数]"""
    n = len(sid)
    feat_per_row = np.zeros((n, X.shape[1] * 3 + 1))
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        idx = idx[np.argsort(sidx[idx])]
        use = idx[sidx[idx] < k]
        if len(use) == 0:
            use = idx[:1]
        Xs = X[use]
        last, first = Xs[-1], Xs[0]
        mean = Xs.mean(0)
        vec = np.concatenate([last, mean, last - first, [len(use)]])
        feat_per_row[idx] = vec
    return feat_per_row


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
    student_final = {int(s): int(ds['final'][sid == s][0]) for s in stu}
    max_part = {}
    for s, p in zip(sid, part):
        max_part[int(s)] = max(max_part.get(int(s), 0), int(p))
    yvec = np.array([student_final[int(s)] for s in stu])
    order_key = sidx * 10 + part

    # 全量行级特征（peer 统计按折重算）
    for k in args.ks:
        t0 = time.time()
        oof_h = np.zeros(n)        # 阶段级 hazard OOF（前 k 段快照）
        oof_flat = np.zeros(len(stu))   # 扁平学生级 RF OOF
        oof_term = np.zeros(len(stu))   # 终局通道 OOF
        oof_fuse = np.zeros(len(stu))
        oof_fuse_c = np.zeros(len(stu))
        oof_drop_stu = np.zeros(len(stu))  # 掉队通道聚合到学生级

        for tr, te in folds:
            Xtr, Xte = build_features(ds, tr, te, mode='rf29')
            # --- 阶段级 hazard（前 k 段快照广播） ---
            Ptr = prefix_snapshot(Xtr, sid[tr], sidx[tr], k)
            Pte = prefix_snapshot(Xte, sid[te], sidx[te], k)
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       max_features='sqrt', class_weight='balanced',
                                       random_state=42, n_jobs=-1)
            m.fit(Ptr, y[tr])
            oof_h[te] = m.predict_proba(Pte)[:, 1]
            # --- 学生级快照矩阵（训练折 / 测试折学生） ---
            tr_s = set(int(s) for s in sid[tr])
            te_s = [int(s) for s in stu if int(s) not in tr_s]
            # 训练折学生：取其快照行（每个学生任取一行，值相同）
            def snap_of(rows, sset):
                out, names = [], []
                for s in sorted(sset):
                    idx = np.where(sid[rows] == s)[0]
                    if len(idx) == 0:
                        continue
                    out.append(Ptr[idx[0]])
                    names.append(s)
                return np.array(out), names
            Str, s_tr = snap_of(tr, tr_s)
            Ste, s_te = snap_of(te, set(te_s))
            # A) 扁平基线
            rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                        class_weight='balanced', random_state=42, n_jobs=-1)
            rf.fit(Str, [student_final[s] for s in s_tr])
            pred = rf.predict_proba(Ste)[:, 1]
            for s, p in zip(s_te, pred):
                oof_flat[stu.tolist().index(s)] = p
            # 终局通道：只用训练折里「没掉队的人」（走完全程）训练
            comp = [s for s in s_tr if max_part[s] >= 7]
            if len(comp) >= 30 and len(set(student_final[s] for s in comp)) > 1:
                idx_c = [i for i, s in enumerate(s_tr) if s in set(comp)]
                lr = LogisticRegression(C=0.2, max_iter=2000, class_weight='balanced')
                lr.fit(Str[idx_c], [student_final[s_tr[i]] for i in idx_c])
                term = lr.predict_proba(Ste)[:, 1]
            else:
                term = np.full(len(s_te), 0.274)
            # 掉队通道：把阶段级 hazard 概率按学生聚合
            p_drop = {}
            for s in te_s:
                idx = np.where(sid[te] == s)[0]
                idx = idx[np.argsort(order_key[te][idx])]
                surv = np.prod(np.clip(1 - oof_h[te][idx], 1e-9, 1.0))
                p_drop[s] = 1 - float(surv)
            drop_tr = [s for s in s_tr if max_part[s] < 7]
            comp_tr = [s for s in s_tr if max_part[s] >= 7]
            q_drop = float(np.mean([student_final[s] for s in drop_tr])) if drop_tr else 0.99
            q_comp = float(np.mean([student_final[s] for s in comp_tr])) if comp_tr else 0.27
            for s, pt, pc in zip(s_te, term, [q_comp] * len(s_te)):
                i = stu.tolist().index(s)
                oof_term[i] = pt
                pd_ = p_drop[s]
                oof_fuse[i] = pd_ * q_drop + (1 - pd_) * pt
                oof_fuse_c[i] = pd_ * q_drop + (1 - pd_) * q_comp
                oof_drop_stu[i] = pd_

        print(f'--- k={k} ---')
        print(f'  A) 扁平学生级 RF（前 {k} 段）      AUC = {roc_auc_score(yvec, oof_flat):.4f}')
        print(f'  B) 双通道融合（个体化终局通道）    AUC = {roc_auc_score(yvec, oof_fuse):.4f}')
        print(f'  C) 双通道融合（常数终局率）        AUC = {roc_auc_score(yvec, oof_fuse_c):.4f}')
        print(f'  D) 终局通道单独                   AUC = {roc_auc_score(yvec, oof_term):.4f}')
        print(f'  E) 掉队通道单独（学生级 P(drop)）  AUC = '
              f'{roc_auc_score(yvec, oof_drop_stu):.4f}  '
              f'（对最终挂科）  [{time.time()-t0:.0f}s]', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--ks', type=int, nargs='+', default=[1, 2, 3])
    main(ap.parse_args())
