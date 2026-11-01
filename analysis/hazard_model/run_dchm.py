#!/usr/bin/env python3
"""
DCHM — Dual-Channel discrete-time Hazard Model for CS1 stage-level dropout
=================================================================================
针对 CS1（n=473 学生 / 2171 阶段样本 / 28.58M IDE 事件）提出的新预测模型原型。

模型动机（全部来自本项目已复现的实测证据，见 docs/IMPROVEMENT_PROPOSAL_v2.md
与 docs/STAGE_TASK_REPORT_20261031.md）：

  E1  CS1 的标签结构本质是「离散时间生存」问题：
      258 名学生在某个 part 掉队（事件），215 名学生走到 part7（右删失），
      其中 59 人走完全程仍挂科。单一 flat 分类器无法同时服务两类人：
      掉队型（强度最低）vs 终局型（强度最高、焦点最碎、返工最多）。
  E2  part 级先验在学生级主导一切（AUC 0.9135，MetaMamba 只高 +0.0184），
      但在阶段级 part-prior 只有 0.4164（低于随机）→ 阶段级任务天然无捷径。
  E3  深度模型在 2171 样本 / 258 事件上不占优：MetaMamba 全部变体
      pooled 0.85–0.88 < RF 0.9258（Δ −0.1302，P=100%）。
  E4  跨 part 泛化是硬墙（学生级 LOPO = 0.5000 随机），阶段级 RF LOPO = 0.9128。
  E5  特征层残差化在阶段级是负收益（pooled −0.027），M4 强度特征也是负结果。
  E6  阶段级第 1 段即可预警（pooled 0.9423）→ 产品价值在「早」而不在「准」。

模型结构（三层）：

  Channel A（掉队 hazard，阶段级）：
      离散时间 cloglog 危险率模型
          log(-log(1-h_i)) = α(stage_i) + x_i'β
      · α 用 stage 序号的三次多项式基（可外推到未见 part → 抗 LOPO）
      · x = part 不变特征（同伴分位/收缩 z/自参照/完成度/节奏比）
      · L2 惩罚（学生数 473、事件 258 → 强正则）
      → 天然输出 P(该阶段之后掉队)，并可逐步累乘得到「截至第 k 段的累计风险」
        = 产品要的「最早可干预时点」曲线。

  Channel B（终局挂科，学生级）：
      在「走到 part7 的人」上建模 P(最终挂科)，
      特征 = 学期全程轨迹签名（强度/焦点碎片度/返工率/推进比）。
      这一通道专门兜住 Channel A 结构性看不见的 18.8%（59/215）。

  Fusion（学生级最终挂科风险）：
      P(fail) = P(drop)·q_drop + (1-P(drop))·q_comp
      q_drop = P(fail | 掉队) ≈ 0.988（训练折内估计）
      q_comp = P(fail | 走完) ≈ 0.274 或 Channel B 个体化取值

评估协议（与项目既有主协议一致，可直接对比）：
  · 5 折按学生分组 × 3 seeds，OOF
  · 主指标 = part 内 pooled AUC（个体判别力口径）
  · LOPO（留一 part）= 跨 part 泛化
  · 附加：Brier / LogLoss（校准）—— hazard 模型按构造输出概率，
    这是它相对 RF 点估计的核心增量价值（可干预阈值需要校准概率）

用法：
  python3 run_dchm.py --data data/stage_dataset.npz --out results/dchm
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402


# ===========================================================================
# 1. 特征层
# ===========================================================================
def peer_stats(RAW, rows, group):
    """按 part 计算同伴统计量（只用给定行 = 训练折行，无标签变换）。"""
    out = {}
    for g in np.unique(group[rows]):
        idx = np.where(group == g)[0]
        idx = np.intersect1d(idx, rows)
        if len(idx) == 0:
            continue
        Xg = RAW[idx]
        med = np.median(Xg, 0)
        mad = np.maximum(np.median(np.abs(Xg - med), 0), 1e-3)
        out[int(g)] = dict(med=med, mad=mad, n=len(idx),
                           sorted=np.sort(Xg, axis=0))
    return out


def peer_z_train(RAW, tr, group, stats):
    """训练行：z = (x - med_part)/MAD_part"""
    out = np.zeros((len(tr), RAW.shape[1]))
    for j, r in enumerate(tr):
        st = stats.get(int(group[r]))
        if st is None:
            continue
        out[j] = (RAW[r] - st['med']) / st['mad']
    return np.clip(out, -5, 5)


def peer_z_apply(RAW, rows, group, stats, fallback=None):
    """测试行：用训练折统计量映射；该 part 训练集为空时用 fallback（全局统计量）"""
    out = np.zeros((len(rows), RAW.shape[1]))
    for j, r in enumerate(rows):
        st = stats.get(int(group[r]))
        if st is None:
            st = fallback
        if st is None:
            continue
        out[j] = (RAW[r] - st['med']) / st['mad']
    return np.clip(out, -5, 5)


def peer_rank(RAW, rows, group, stats):
    """同伴分位（经验分布位置）：跨 part 可比 → 抗 LOPO 的 part 不变特征"""
    out = np.zeros((len(rows), RAW.shape[1]))
    for j, r in enumerate(rows):
        st = stats.get(int(group[r]))
        if st is None:
            continue
        srt = st['sorted']
        n = st['n']
        for d in range(RAW.shape[1]):
            k = np.searchsorted(srt[:, d], RAW[r, d], side='left')
            out[j, d] = (k + 0.5) / (n + 1.0)
    return out


def global_stats(RAW, rows):
    med = np.median(RAW[rows], 0)
    mad = np.maximum(np.median(np.abs(RAW[rows] - med), 0), 1e-3)
    return dict(med=med, mad=mad, n=len(rows),
                sorted=np.sort(RAW[rows], axis=0))


def shrunk_z(RAW, rows, group, stats, gstats, kappa=20.0):
    """经验贝叶斯收缩 z：小 part（part5 n=14 / part6 n=9）统计不稳 → 向全局收缩
        z_shrunk = w·z_part + (1-w)·z_global,  w = n_g/(n_g+κ)"""
    zp = peer_z_apply(RAW, rows, group, stats, fallback=gstats)
    zg = np.clip((RAW[rows] - gstats['med']) / gstats['mad'], -5, 5)
    w = np.zeros(len(rows))
    for j, r in enumerate(rows):
        st = stats.get(int(group[r]))
        n = st['n'] if st is not None else 0
        w[j] = n / (n + kappa)
    return w[:, None] * zp + (1 - w[:, None]) * zg


def build_features(ds, tr, te, mode='inv', kappa=20.0, lopo=False):
    """返回 (Xtr, Xte)。mode:
        'rf29' : 与 RF 基线**逐位同源**的 29 维（A4+B3+C2+peer_z10+SELF10）
        'inv'  : 29 维 + 同伴分位 10 + 收缩 z 10（part 不变增强）
    """
    RAW, part, DER, SELF = ds['RAW'], ds['part'], ds['DER'], ds['SELF']
    if lopo:
        # 留出的 part 在训练集里没有行 → 用该 part 自己的全部行算统计量
        # （纯无标签变换，不构成泄漏），两个 arm 口径一致
        stats = peer_stats(RAW, np.arange(len(RAW)), part)
        gstats = global_stats(RAW, np.arange(len(RAW)))
    else:
        stats = peer_stats(RAW, tr, part)
        gstats = global_stats(RAW, tr)

    A, B = DER[:, [0, 1, 2, 3]], DER[:, [4, 5, 6]]
    C = np.column_stack([DER[:, 7], DER[:, 8]])
    ztr = sc.peer_z(ds, tr, tr, part) if not lopo else peer_z_train(RAW, tr, part, stats)
    zte = sc.peer_z(ds, tr, te, part) if not lopo else peer_z_apply(RAW, te, part, stats, gstats)
    Xtr = np.hstack([A[tr], B[tr], C[tr], ztr, SELF[tr]])
    Xte = np.hstack([A[te], B[te], C[te], zte, SELF[te]])
    if mode == 'inv':
        rtr = peer_rank(RAW, tr, part, stats)
        rte = peer_rank(RAW, te, part, stats)
        str_ = shrunk_z(RAW, tr, part, stats, gstats, kappa)
        ste = shrunk_z(RAW, te, part, stats, gstats, kappa)
        Xtr = np.hstack([Xtr, rtr, str_])
        Xte = np.hstack([Xte, rte, ste])
    return Xtr.astype(np.float64), Xte.astype(np.float64)


# ===========================================================================
# 2. Channel A：离散时间 cloglog 危险率模型（L2 惩罚，IRLS）
# ===========================================================================
def poly_basis(stage_idx, deg=3):
    s = np.asarray(stage_idx, dtype=np.float64) / 6.0
    cols = [np.ones_like(s)]
    for k in range(1, deg + 1):
        cols.append(s ** k)
    return np.column_stack(cols)


def fit_cloglog(X, y, stage_idx, lam=1.0, lam_stage=0.1, deg=3,
                max_iter=100, tol=1e-9):
    """惩罚离散时间 cloglog：
         log(-log(1-p)) = α(stage) + x'β ，  α 用 stage 序号多项式基
    返回 dict(beta, alpha_coef, mu, sd, predict)
    """
    n, d = X.shape
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    B = poly_basis(stage_idx, deg)
    D = np.hstack([B, Xs])
    # 惩罚：stage 基项弱惩罚（相当于偏置），特征项强惩罚
    P = np.eye(D.shape[1]) * lam
    P[:B.shape[1], :B.shape[1]] = lam_stage

    theta = np.zeros(D.shape[1])
    theta[0] = np.log(-np.log(1 - np.clip(y.mean(), 1e-3, 1 - 1e-3)))
    for _ in range(max_iter):
        eta = np.clip(D @ theta, -20, 20)
        E = np.exp(eta)
        p = 1.0 - np.exp(-E)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        dp = (1 - p) * E
        z = eta + (y - p) / dp
        W = dp ** 2 / (p * (1 - p))
        W = np.clip(W, 1e-8, 1e8)
        A = D.T @ (D * W[:, None]) + P
        b = D.T @ (W * z) - P @ theta
        try:
            new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            new = np.linalg.lstsq(A, b, rcond=None)[0]
        if np.max(np.abs(new - theta)) < tol:
            theta = new
            break
        theta = new

    def predict(Xn, stage_idx_n):
        Xn_s = (Xn - mu) / sd
        Dn = np.hstack([poly_basis(stage_idx_n, deg), Xn_s])
        eta = np.clip(Dn @ theta, -20, 20)
        return 1.0 - np.exp(-np.exp(eta))

    return dict(theta=theta, beta=theta[B.shape[1]:], mu=mu, sd=sd,
                deg=deg, predict=predict)


# ===========================================================================
# 3. 指标
# ===========================================================================
def brier(y, p):
    return float(np.mean((p - y) ** 2))


def logloss(y, p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def eval_scores(y, oof, group, tag=''):
    m = sc.metrics(y, oof, group, tag)
    m['brier'] = brier(y, oof)
    m['logloss'] = logloss(y, oof)
    return m


def student_risk_from_stage(prob_drop_stage, sid, stage_idx, order_key):
    """由阶段级 hazard 概率聚合成学生级累计风险 P(掉队)：
       P_drop(student) = 1 - Π_i (1 - h_i)"""
    out = {}
    for s in np.unique(sid):
        idx = np.where(sid == s)[0]
        idx = idx[np.argsort(order_key[idx])]
        surv = np.prod(np.clip(1.0 - prob_drop_stage[idx], 1e-9, 1.0))
        out[int(s)] = 1.0 - surv
    return out


# ===========================================================================
# 4. 主流程
# ===========================================================================
def run(args):
    ds = sc.load_dataset(args.data)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part_ids'])
    ds['DER'] = sc.derive(ds)
    ds['SELF'] = sc.self_ref(ds)
    y = ds['drop'].astype(int)
    part = ds['part']
    sid = ds['sid']
    sidx = ds['stage_idx']
    n = len(y)
    print(f'[{args.tag}] N={n} students={len(np.unique(sid))} drop_events={int(y.sum())}', flush=True)

    seeds = [42, 123, 777]
    folds = sc.group_folds(sid, ds['final'], n_splits=5, seed=42)

    # ---------- RF 基线（复刻 run_stage_fair.py 的 rf_full 口径） ----------
    from sklearn.ensemble import RandomForestClassifier
    oof_rf = np.zeros(n)
    for tr, te in folds:
        Xtr, Xte = build_features(ds, tr, te, mode='rf29')
        acc = np.zeros(len(te))
        for sd in seeds:
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       max_features='sqrt', class_weight='balanced',
                                       random_state=sd, n_jobs=-1)
            m.fit(Xtr, y[tr])
            acc += m.predict_proba(Xte)[:, 1]
        oof_rf[te] = acc / len(seeds)
    m_rf = eval_scores(y, oof_rf, part, 'rf_full')
    print(f"  RF-full      overall={m_rf['auc']:.4f} pooled={m_rf['pooled']:.4f} "
          f"judged={m_rf['judged']:.4f} brier={m_rf['brier']:.4f} ll={m_rf['logloss']:.4f}",
          flush=True)

    # ---------- Channel A：DCHM（不同特征集 / 不同惩罚） ----------
    results = {'rf_full': m_rf}
    oof_by_cfg = {}
    for mode in (['rf29', 'inv'] if args.mode == 'both' else [args.mode]):
        for lam in args.lams:
            t0 = time.time()
            oof = np.zeros(n)
            coefs = []
            for tr, te in folds:
                Xtr, Xte = build_features(ds, tr, te, mode=mode)
                mdl = fit_cloglog(Xtr, y[tr], sidx[tr], lam=lam)
                oof[te] = mdl['predict'](Xte, sidx[te])
                coefs.append(mdl['beta'])
            key = f'dchm_{mode}_lam{lam}'
            m = eval_scores(y, oof, part, key)
            results[key] = m
            oof_by_cfg[key] = oof
            print(f"  {key:<26} overall={m['auc']:.4f} pooled={m['pooled']:.4f} "
                  f"judged={m['judged']:.4f} brier={m['brier']:.4f} ll={m['logloss']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            print('     per-part: ' + ' '.join(
                f"p{k}={v['auc']:.3f}" for k, v in sorted(m['per_part'].items(),
                                                         key=lambda kv: int(kv[0]))), flush=True)

    # ---------- LOPO（跨 part 泛化） ----------
    lopo_parts = [int(p) for p in np.unique(part)
                  if y[part == p].sum() > 0 and (1 - y[part == p]).sum() > 0]
    lopo = {}
    oof_rf_lopo = np.zeros(n)
    per_part_rf = {}
    for p in lopo_parts:
        te = np.where(part == p)[0]
        tr = np.where(part != p)[0]
        Xtr, Xte = build_features(ds, tr, te, mode='rf29', lopo=True)
        acc = np.zeros(len(te))
        for sd in seeds:
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       max_features='sqrt', class_weight='balanced',
                                       random_state=sd, n_jobs=-1)
            m.fit(Xtr, y[tr])
            acc += m.predict_proba(Xte)[:, 1]
        oof_rf_lopo[te] = acc / len(seeds)
        per_part_rf[p] = float(roc_auc_score(y[te], oof_rf_lopo[te]))
    lopo['rf_full'] = dict(per_part=per_part_rf,
                           weighted=float(np.average(list(per_part_rf.values()),
                                                    weights=[int((part == p).sum())
                                                             for p in per_part_rf])))
    print(f"  LOPO rf_full weighted AUC = {lopo['rf_full']['weighted']:.4f}  "
          + ' '.join(f'p{k}={v:.3f}' for k, v in per_part_rf.items()), flush=True)

    for mode in (['rf29', 'inv'] if args.mode == 'both' else [args.mode]):
        for lam in args.lams:
            oof = np.zeros(n)
            per_part = {}
            for p in lopo_parts:
                te = np.where(part == p)[0]
                tr = np.where(part != p)[0]
                Xtr, Xte = build_features(ds, tr, te, mode=mode, lopo=True)
                mdl = fit_cloglog(Xtr, y[tr], sidx[tr], lam=lam)
                oof[te] = mdl['predict'](Xte, sidx[te])
                per_part[p] = float(roc_auc_score(y[te], oof[te]))
            w = float(np.average(list(per_part.values()),
                                 weights=[int((part == p).sum()) for p in per_part]))
            key = f'dchm_{mode}_lam{lam}'
            lopo[key] = dict(per_part=per_part, weighted=w)
            print(f"  LOPO {key:<26} weighted AUC = {w:.4f}  "
                  + ' '.join(f'p{k}={v:.3f}' for k, v in per_part.items()), flush=True)

    # ---------- 学生级最终挂科：融合 ----------
    # Channel B：在「走到 part7」的学生上建模 P(最终挂科)
    max_part = {}
    for s, p in zip(sid, part):
        max_part[int(s)] = max(max_part.get(int(s), 0), int(p))
    reached7 = np.array([s for s in np.unique(sid) if max_part[int(s)] >= 7])
    student_final = {int(s): int(ds['final'][sid == s][0]) for s in np.unique(sid)}

    # 学生级特征：最后一段特征 + 段间均值 + 段间斜率（按折构建，避免任何折间泄漏）
    def student_matrix(Xrows, row_pos):
        """row_pos: {student_id: 该生在本折矩阵中的行索引}"""
        out = []
        for s in sorted(row_pos):
            idx = np.array(sorted(row_pos[s]))
            Xs = Xrows[idx]
            out.append(np.concatenate([Xs[-1], Xs.mean(0), Xs[-1] - Xs[0],
                                       np.array([len(idx)])]))
        return np.array(out), sorted(row_pos)

    # 阶段级 OOF → 学生级 P(drop)
    order_key = sidx * 10 + part
    stu = np.unique(sid)
    yvec = np.array([student_final[int(s)] for s in stu])

    # 学生级 RF 基线（同一套学生级特征，按学生分组 CV，逐折构建特征）
    from sklearn.ensemble import RandomForestClassifier
    oof_s = np.zeros(len(stu))
    pos_of = {int(s): i for i, s in enumerate(stu)}
    student_fold_feats = []
    for tr, te in folds:
        Xtr, Xte = build_features(ds, tr, te, mode='rf29')
        pos_tr = {}
        for j, r in enumerate(tr):
            pos_tr.setdefault(int(sid[r]), []).append(j)
        pos_te = {}
        for j, r in enumerate(te):
            pos_te.setdefault(int(sid[r]), []).append(j)
        Str, s_tr = student_matrix(Xtr, pos_tr)
        Ste, s_te = student_matrix(Xte, pos_te)
        student_fold_feats.append((s_tr, Str, s_te, Ste))
        tri = [i for i, s in enumerate(stu) if int(s) in set(s_tr)]
        tei = [i for i, s in enumerate(stu) if int(s) in set(s_te)]
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   class_weight='balanced', random_state=42, n_jobs=-1)
        m.fit(Str, yvec[tri])
        oof_s[tei] = m.predict_proba(Ste)[:, 1]
    auc_rf_s = float(roc_auc_score(yvec, oof_s))
    results['student_rf_baseline'] = dict(auc=auc_rf_s)
    print(f"  [student-level] RF baseline (同特征) AUC = {auc_rf_s:.4f}", flush=True)

    for key, oof in oof_by_cfg.items():
        p_drop = student_risk_from_stage(oof, sid, sidx, order_key)
        fmap, cmap = {}, {}
        for fi, (tr, te) in enumerate(folds):
            s_tr, Str, s_te, Ste = student_fold_feats[fi]
            tr_s = set(int(s) for s in s_tr)
            # q_drop = P(fail | dropped)，训练折内估计
            drop_tr = [s for s in tr_s if max_part[s] < 7]
            q_drop = (float(np.mean([student_final[s] for s in drop_tr]))
                      if drop_tr else 0.99)
            comp_tr = [s for s in tr_s if max_part[s] >= 7]
            q_comp = (float(np.mean([student_final[s] for s in comp_tr]))
                      if comp_tr else 0.27)
            # Channel B 个体化（只用训练折的完课者训练）
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
        auc = float(roc_auc_score(yvec, pvec))
        auc_c = float(roc_auc_score(yvec, pc))
        results[key + '_student_fusion'] = dict(auc=auc, auc_const=auc_c,
                                                brier=brier(yvec, pvec))
        print(f"  [fusion {key}] student-level final-fail AUC = {auc:.4f} "
              f"(常数融合 {auc_c:.4f})", flush=True)

    # ---------- 落盘 ----------
    out = dict(tag=args.tag, results=results, lopo=lopo,
               dataset=dict(n=int(n), students=int(len(np.unique(sid))),
                            drop_events=int(y.sum())))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f'dchm_{args.tag}.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(args.out, f'oof_{args.tag}.npz'),
                        **{k: v for k, v in oof_by_cfg.items()},
                        rf_full=oof_rf, y=y, sid=sid, part=part)
    print(f'写出 {args.out}/dchm_{args.tag}.json', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--out', default='results/dchm')
    ap.add_argument('--tag', default='run1')
    ap.add_argument('--mode', default='both', choices=['rf29', 'inv', 'both'])
    ap.add_argument('--lams', type=float, nargs='+', default=[1.0, 10.0, 100.0])
    args = ap.parse_args()
    run(args)
