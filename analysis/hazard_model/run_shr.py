#!/usr/bin/env python3
"""
CS1-SHR — Stage-Hierarchical Risk model（阶段分层收缩危险率模型）
=================================================================================
DCHM v1 的实测结论：纯线性 cloglog 危险率模型在阶段级 pooled AUC 只有
0.63–0.78（远低于 RF 0.9258）→ **生存形式本身不是瓶颈，非线性才是**。

于是把两件事分开：
  (1) 生存/分层形式负责：删失处理、跨阶段可比的概率标尺、累计风险曲线、
      早期干预时点、每阶段的可判别性口径；
  (2) 非线性学习器（RF / HGB）负责：个体判别力。

模型（三个部件）：
  A. 阶段级危险率 h_k(x)：**每 part 单独建模**，向全局模型收缩
        score_k(x) = w_k·pooled(x) + (1−w_k)·part_k(x),  w_k = n_k/(n_k+κ)
     理由：每个 part 的掉队率与行为语义都不同（part1 掉队率 0.176，
     part5 0.060），且 part 内特征效应可能不同；而小 part（part5 n=250、
     正例 15；part6 正例 22）单独训练不稳 → 必须收缩到全局。
  B. 校准与累计风险：p_drop(student) = 1 − Π_k (1 − h_k)，
     这条曲线直接就是产品要的「截至第 k 段的累计风险 / 最早可干预时点」。
  C. 终局通道 Channel B：在走到 part7 的 215 人上建模 P(最终挂科)，
     专门兜住阶段级结构性看不见的 59 人（18.8%）。

严格性：
  · 5 折按学生分组 × 3 seeds（与项目主协议一致）
  · LOPO 留一 part：未见过的 part 自动回退到全局模型（不崩）
  · 所有同伴统计量只用训练折行（无标签变换）

用法：
  python3 run_shr.py --data data/stage_dataset.npz --out results/shr
"""
import argparse
import json
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
from analysis.hazard_model.run_dchm import (build_features,  # noqa: E402
                                            student_risk_from_stage)


def student_matrix(Xrows, row_pos):
    """学生级特征矩阵：最后一段 + 段间均值 + 段间斜差 + 段数
    row_pos: {student_id: 该生在本折矩阵中的行索引}"""
    out = []
    for s in sorted(row_pos):
        idx = np.array(sorted(row_pos[s]))
        Xs = Xrows[idx]
        out.append(np.concatenate([Xs[-1], Xs.mean(0), Xs[-1] - Xs[0],
                                   np.array([len(idx)])]))
    return np.array(out), sorted(row_pos)


def make_model(kind, seed):
    if kind == 'rf':
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      max_features='sqrt', class_weight='balanced',
                                      random_state=seed, n_jobs=-1)
    if kind == 'hgb':
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=300, learning_rate=0.06,
            min_samples_leaf=10, l2_regularization=1.0,
            class_weight='balanced', random_state=seed)
    raise ValueError(kind)


def fit_predict(kind, Xtr, ytr, Xte, seeds):
    acc = np.zeros(len(Xte))
    for sd in seeds:
        m = make_model(kind, sd)
        m.fit(Xtr, ytr)
        acc += m.predict_proba(Xte)[:, 1]
    return acc / len(seeds)


def main(args):
    ds = sc.load_dataset(args.data)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part_ids'])
    ds['DER'] = sc.derive(ds)
    ds['SELF'] = sc.self_ref(ds)
    y = ds['drop'].astype(int)
    part = ds['part']
    sid = ds['sid']
    sidx = ds['stage_idx']
    n = len(y)
    seeds = [42, 123, 777]
    folds = sc.group_folds(sid, ds['final'], n_splits=5, seed=42)
    print(f'N={n} students={len(np.unique(sid))} drop_events={int(y.sum())}', flush=True)

    res = {}

    for kind in args.models:
        # ---------- pooled（全局）模型 ----------
        oof_pool = np.zeros(n)
        t0 = time.time()
        for tr, te in folds:
            Xtr, Xte = build_features(ds, tr, te, mode='rf29')
            oof_pool[te] = fit_predict(kind, Xtr, y[tr], Xte, seeds)
        m = sc.metrics(y, oof_pool, part, f'{kind}_pooled')
        m['brier'] = float(np.mean((oof_pool - y) ** 2))
        res[f'{kind}_pooled'] = m
        print(f"  {kind}_pooled      overall={m['auc']:.4f} pooled={m['pooled']:.4f} "
              f"judged={m['judged']:.4f} ({time.time()-t0:.0f}s)", flush=True)

        # ---------- 每 part 模型（只在训练折里该 part 的行上训练） ----------
        oof_part = np.zeros(n)
        cnt = {}
        for tr, te in folds:
            Xtr, Xte = build_features(ds, tr, te, mode='rf29')
            for p in np.unique(part):
                ptr = tr[part[tr] == p]
                pte = te[part[te] == p]
                if len(ptr) < 30 or len(pte) == 0:
                    continue
                if len(set(y[ptr].tolist())) < 2:
                    continue
                oof_part[pte] = fit_predict(kind, Xtr[part[tr] == p], y[ptr],
                                            Xte[part[te] == p], seeds)
                cnt[int(p)] = (len(ptr), int(y[ptr].sum()))
        mp = sc.metrics(y, oof_part, part, f'{kind}_part')
        res[f'{kind}_part'] = mp
        print(f"  {kind}_part        overall={mp['auc']:.4f} pooled={mp['pooled']:.4f} "
              f"judged={mp['judged']:.4f}", flush=True)

        # ---------- 收缩混合 ----------
        for kappa in args.kappas:
            oof = np.zeros(n)
            for tr, te in folds:
                nk = {int(p): int((part[tr] == p).sum()) for p in np.unique(part)}
                for p in np.unique(part[te]):
                    mte = te[part[te] == p]
                    k = nk.get(int(p), 0)
                    w = k / (k + kappa)
                    src = oof_part[mte] if oof_part[mte].std() > 0 else oof_pool[mte]
                    oof[mte] = w * oof_pool[mte] + (1 - w) * src
            key = f'{kind}_shr_k{kappa}'
            mm = sc.metrics(y, oof, part, key)
            mm['brier'] = float(np.mean((oof - y) ** 2))
            res[key] = mm
            print(f"  {key:<16} overall={mm['auc']:.4f} pooled={mm['pooled']:.4f} "
                  f"judged={mm['judged']:.4f} brier={mm['brier']:.4f}", flush=True)
            print('     per-part: ' + ' '.join(
                f"p{k}={v['auc']:.3f}" for k, v in sorted(mm['per_part'].items(),
                                                         key=lambda kv: int(kv[0]))), flush=True)

    # ---------- LOPO：留一 part（未见 part 自动回退全局模型） ----------
    lopo_parts = [int(p) for p in np.unique(part)
                  if y[part == p].sum() > 0 and (1 - y[part == p]).sum() > 0]
    lopo = {}
    for kind in args.models:
        oof = np.zeros(n)
        per_part = {}
        for p in lopo_parts:
            te = np.where(part == p)[0]
            tr = np.where(part != p)[0]
            Xtr, Xte = build_features(ds, tr, te, mode='rf29', lopo=True)
            oof[te] = fit_predict(kind, Xtr, y[tr], Xte, seeds)
            per_part[p] = float(roc_auc_score(y[te], oof[te]))
        w = float(np.average(list(per_part.values()),
                             weights=[int((part == p).sum()) for p in per_part]))
        lopo[kind] = dict(per_part=per_part, weighted=w)
        print(f"  LOPO {kind:<12} weighted AUC = {w:.4f}  "
              + ' '.join(f'p{k}={v:.3f}' for k, v in per_part.items()), flush=True)

    # ---------- 学生级：融合 + 累计风险曲线（干预时点） ----------
    max_part = {}
    for s, p in zip(sid, part):
        max_part[int(s)] = max(max_part.get(int(s), 0), int(p))
    student_final = {int(s): int(ds['final'][sid == s][0]) for s in np.unique(sid)}
    stu = np.unique(sid)
    yvec = np.array([student_final[int(s)] for s in stu])
    order_key = sidx * 10 + part

    # 学生级特征（逐折构建）
    fold_feats = []
    for tr, te in folds:
        Xtr, Xte = build_features(ds, tr, te, mode='rf29')
        pos_tr, pos_te = {}, {}
        for j, r in enumerate(tr):
            pos_tr.setdefault(int(sid[r]), []).append(j)
        for j, r in enumerate(te):
            pos_te.setdefault(int(sid[r]), []).append(j)
        Str, s_tr = student_matrix(Xtr, pos_tr)
        Ste, s_te = student_matrix(Xte, pos_te)
        fold_feats.append((s_tr, Str, s_te, Ste))

    oof_s = np.zeros(len(stu))
    for (s_tr, Str, s_te, Ste) in fold_feats:
        tri = [i for i, s in enumerate(stu) if int(s) in set(s_tr)]
        tei = [i for i, s in enumerate(stu) if int(s) in set(s_te)]
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
        m.fit(Str, yvec[tri])
        oof_s[tei] = m.predict_proba(Ste)[:, 1]
    res['student_rf_fulltraj'] = dict(auc=float(roc_auc_score(yvec, oof_s)))
    print(f"  [student] RF(全程轨迹特征) AUC = {roc_auc_score(yvec, oof_s):.4f} "
          f"← 含进度信息，仅作上限参考", flush=True)

    # 融合：P(drop) 由阶段级 hazard 累乘得到；q_drop/q_comp 训练折内估计
    for key in [k for k in res if k.endswith('_pooled') or '_shr_k' in k]:
        oof_stage = res[key].get('_oof')
        if oof_stage is None:
            continue

    # 重算每个阶段级配置的 OOF（保存起来供融合/曲线用）
    stage_oofs = {}
    for kind in args.models:
        oof_pool = np.zeros(n)
        oof_part = np.zeros(n)
        for tr, te in folds:
            Xtr, Xte = build_features(ds, tr, te, mode='rf29')
            oof_pool[te] = fit_predict(kind, Xtr, y[tr], Xte, seeds)
            for p in np.unique(part):
                ptr, pte = tr[part[tr] == p], te[part[te] == p]
                if len(ptr) < 30 or len(pte) == 0 or len(set(y[ptr].tolist())) < 2:
                    continue
                oof_part[pte] = fit_predict(kind, Xtr[part[tr] == p], y[ptr],
                                            Xte[part[te] == p], seeds)
        stage_oofs[f'{kind}_pooled'] = oof_pool
        stage_oofs[f'{kind}_part'] = oof_part
        for kappa in args.kappas:
            oof = np.zeros(n)
            for tr, te in folds:
                nk = {int(p): int((part[tr] == p).sum()) for p in np.unique(part)}
                for p in np.unique(part[te]):
                    mte = te[part[te] == p]
                    k = nk.get(int(p), 0)
                    w = k / (k + kappa)
                    src = oof_part[mte] if oof_part[mte].std() > 0 else oof_pool[mte]
                    oof[mte] = w * oof_pool[mte] + (1 - w) * src
            stage_oofs[f'{kind}_shr_k{kappa}'] = oof

    for key, oof in stage_oofs.items():
        p_drop = student_risk_from_stage(oof, sid, sidx, order_key)
        fmap, cmap = {}, {}
        for fi, (tr, te) in enumerate(folds):
            s_tr, Str, s_te, Ste = fold_feats[fi]
            tr_s = set(int(s) for s in s_tr)
            drop_tr = [s for s in tr_s if max_part[s] < 7]
            comp_tr = [s for s in tr_s if max_part[s] >= 7]
            q_drop = float(np.mean([student_final[s] for s in drop_tr])) if drop_tr else 0.99
            q_comp = float(np.mean([student_final[s] for s in comp_tr])) if comp_tr else 0.27
            q_comp_vec = {s: q_comp for s in s_tr}
            if len(comp_tr) >= 30 and len(set(student_final[s] for s in comp_tr)) > 1:
                idx_c = [i for i, s in enumerate(s_tr) if s in set(comp_tr)]
                lr = LogisticRegression(C=0.2, max_iter=2000, class_weight='balanced')
                lr.fit(Str[idx_c], [student_final[s_tr[i]] for i in idx_c])
                q_comp_vec = {s: float(lr.predict_proba(Ste[i:i+1])[:, 1][0])
                              for i, s in enumerate(s_te)}
            for s in s_te:
                pd_ = float(p_drop.get(int(s), 0.0))
                fmap[int(s)] = pd_ * q_drop + (1 - pd_) * q_comp_vec.get(int(s), q_comp)
                cmap[int(s)] = pd_ * q_drop + (1 - pd_) * q_comp
        pvec = np.array([fmap[int(s)] for s in stu])
        pc = np.array([cmap[int(s)] for s in stu])
        res[f'{key}_fusion'] = dict(auc=float(roc_auc_score(yvec, pvec)),
                                    auc_const=float(roc_auc_score(yvec, pc)))
        print(f"  [fusion {key:<22}] student final-fail AUC = "
              f"{roc_auc_score(yvec, pvec):.4f} (常数 {roc_auc_score(yvec, pc):.4f})", flush=True)

    # 累计风险曲线（干预时点）：P(掉队 by 第 k 段)
    curve = {}
    for key, oof in stage_oofs.items():
        rows = []
        for k in range(1, 8):
            alive = {}
            for s in stu:
                idx = np.where(sid == s)[0]
                idx = idx[np.argsort(order_key[idx])]
                idx = idx[idx <= k - 1]          # 只用前 k 段
                if len(idx) == 0:
                    continue
                alive[int(s)] = 1.0 - float(np.prod(np.clip(1 - oof[idx], 1e-9, 1)))
            ys = [student_final[int(s)] for s in alive]
            ps = [alive[int(s)] for s in alive]
            auc = float(roc_auc_score(ys, ps)) if len(set(ys)) > 1 else float('nan')
            rows.append((k, len(alive), auc))
        curve[key] = rows
        print(f"  [curve {key}] " + ' '.join(f'k{k}:n{n_},AUC{a:.3f}'
                                           for k, n_, a in rows), flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f'shr_{args.tag}.json'), 'w') as f:
        json.dump(dict(results=res, lopo=lopo, curve=curve), f, indent=2,
                  ensure_ascii=False, default=float)
    np.savez_compressed(os.path.join(args.out, f'oof_{args.tag}.npz'),
                        **{k: v for k, v in stage_oofs.items()},
                        y=y, sid=sid, part=part)
    print(f'写出 {args.out}/shr_{args.tag}.json', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--out', default='results/shr')
    ap.add_argument('--tag', default='run1')
    ap.add_argument('--models', nargs='+', default=['rf'])
    ap.add_argument('--kappas', type=float, nargs='+', default=[10.0, 30.0, 100.0])
    main(ap.parse_args())
