#!/usr/bin/env python3
"""
CS1-SHR 学生级输出：双通道融合 + 累计风险曲线（干预时点）
=================================================================================
输入：阶段级 hazard 的 OOF 概率（results/dchm/oof_run1.npz 的 rf_full，
      与结案报告 pooled 0.9258 逐位一致），直接复用，不再重训。

产出：
  1) 学生级最终挂科风险 = P_drop·q_drop + (1−P_drop)·q_comp
     · P_drop = 1 − Π_i(1 − h_i)（该生各阶段的掉队危险率乘积，删失正确）
     · q_drop = P(挂科 | 掉队) ≈ 0.988；q_comp = P(挂科 | 走完) ≈ 0.274
       或用终局通道（LogReg on 全程轨迹签名，训练于训练折的完课者）个体化
  2) 逐阶段累计风险曲线：P_drop_by_k = 1 − Π_{i≤k}(1 − h_i)
     → 「截至第 k 段是否已经掉队」+ 「活到第 k 段的人里谁会挂科」
  3) 校准：Brier / LogLoss（融合概率 vs 真实标签）
"""
import argparse
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402
from analysis.hazard_model.run_dchm import build_features  # noqa: E402


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
    print(f'阶段级 hazard OOF: {args.oof} | pooled(rep) 见结案报告 0.9258', flush=True)

    # 学生级快照特征（逐折）
    fold_feats = []
    for tr, te in folds:
        Xtr, Xte = build_features(ds, tr, te, mode='rf29')
        def snap(rows, X):
            out, names = [], []
            for s in sorted(set(int(x) for x in sid[rows])):
                idx = np.where(sid[rows] == s)[0]
                idx = idx[np.argsort(sidx[rows][idx])]
                Xs = X[idx]
                out.append(np.concatenate([Xs[-1], Xs.mean(0), Xs[-1] - Xs[0],
                                           [len(idx)]]))
                names.append(s)
            return np.array(out), names
        Str, s_tr = snap(tr, Xtr)
        Ste, s_te = snap(te, Xte)
        fold_feats.append((s_tr, Str, s_te, Ste))

    def p_drop_by_k(k):
        """P(截至第 k 段已掉队)：只乘已观测阶段"""
        out = {}
        for s in stu:
            idx = np.where(sid == s)[0]
            idx = idx[np.argsort(order_key[idx])]
            keep = idx[sidx[idx] < k] if (sidx[idx] < k).any() else idx[:1]
            out[s] = 1.0 - float(np.prod(np.clip(1 - h[keep], 1e-9, 1.0)))
        return out

    def alive_at(k):
        out = np.zeros(len(stu), dtype=bool)
        for i, s in enumerate(stu):
            idx = np.where(sid == s)[0]
            out[i] = bool((sidx[idx] >= k).any())   # 还有第 k 段之后的阶段 → 仍活跃
        return out

    # ---------- 终局通道 + 融合（全期） ----------
    p_full = p_drop_by_k(99)
    fuse, fuse_c = np.zeros(len(stu)), np.zeros(len(stu))
    term_pred = np.zeros(len(stu))
    for fi, (tr, te) in enumerate(folds):
        s_tr, Str, s_te, Ste = fold_feats[fi]
        drop_tr = [s for s in s_tr if max_part[s] < 7]
        comp_tr = [s for s in s_tr if max_part[s] >= 7]
        q_drop = float(np.mean([student_final[s] for s in drop_tr])) if drop_tr else 0.99
        q_comp = float(np.mean([student_final[s] for s in comp_tr])) if comp_tr else 0.27
        qv = {}
        if len(comp_tr) >= 30 and len(set(student_final[s] for s in comp_tr)) > 1:
            idx_c = [i for i, s in enumerate(s_tr) if s in set(comp_tr)]
            lr = LogisticRegression(C=0.1, max_iter=3000, class_weight='balanced')
            lr.fit(Str[idx_c], [student_final[s_tr[i]] for i in idx_c])
            qv = {s: float(lr.predict_proba(Ste[i:i+1])[:, 1][0])
                  for i, s in enumerate(s_te)}
        for s in s_te:
            i = pos[s]
            pd_ = p_full[s]
            pc = qv.get(s, q_comp)
            term_pred[i] = pc
            fuse[i] = pd_ * q_drop + (1 - pd_) * pc
            fuse_c[i] = pd_ * q_drop + (1 - pd_) * q_comp

    def brier(p):
        return float(np.mean((p - yvec) ** 2))

    print('\n=== 学生级最终挂科（473 人 / 314 挂科）===')
    print(f'  掉队通道单独 P_drop(全期)      AUC = {roc_auc_score(yvec, [p_full[s] for s in stu]):.4f}')
    print(f'  终局通道单独（个体化）         AUC = {roc_auc_score(yvec, term_pred):.4f}')
    print(f'  双通道融合（个体化终局）        AUC = {roc_auc_score(yvec, fuse):.4f}  Brier = {brier(fuse):.4f}')
    print(f'  双通道融合（常数终局率 0.274）  AUC = {roc_auc_score(yvec, fuse_c):.4f}  Brier = {brier(fuse_c):.4f}')

    print('\n=== 逐阶段累计风险曲线（干预时点）===')
    for k in range(1, 8):
        pk = p_drop_by_k(k)
        v = np.array([pk[s] for s in stu])
        al = alive_at(k)
        auc_all = roc_auc_score(yvec, v)
        auc_alive = (roc_auc_score(yvec[al], v[al])
                     if len(set(yvec[al].tolist())) > 1 else float('nan'))
        print(f'  k={k}: 全部 AUC={auc_all:.4f} | 仍活跃子集 n={int(al.sum())} '
              f'AUC={auc_alive:.4f} | P_drop 中位={np.median(v):.3f}', flush=True)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'student_fusion.npz'),
                        fuse=fuse, fuse_c=fuse_c, term=term_pred,
                        p_full=np.array([p_full[s] for s in stu]),
                        y=yvec, stu=np.array(stu))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--oof', default='results/dchm/oof_run1.npz')
    ap.add_argument('--out', default='results/shr')
    main(ap.parse_args())
