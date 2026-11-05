#!/usr/bin/env python3
"""
CS1-FSP · 段锚定少样本学习（Part-Anchored Few-Shot Learning）
=================================================================================
命题（王建，2026-11-05）：「根据 studentrisk 目前的实验结果，详细归纳现有模型的不足，
并提出一种新的少样本模型，提升预测的准确性或早期预测」。

前置事实（本次新增，重要）：
  仓库里原有的 per-part 5-shot FOMAML 评估**是跑坏的**
  （适配后权重未加载=没适配 / 跨段状态泄漏 / 段索引错位 / "accuracy=1.0" 来自单类别 query）
  → 「少样本在 CS1 失败」这个旧结论不可信；少样本在本项目上还没被正经检验过。

设计要点：
  ① 段锚定转导标准化：每段的绝对量特征（A强度/B比率/C进度）用**该段自己的**无标签行
     中位数/MAD 标准化 → 消掉段间分布漂移（这是跨段迁移的关键工程对策）。
  ② 标签预算 K：每段只给 K 个掉队者 + K 个未掉队者（K=1/2/4/8/16/32）。
  ③ 模型族对比 —— 本实验真正要回答的问题是「**小标签预算下哪种归纳偏置还站得住**」：
       proto    双原型距离比（等价于线性判别方向）
       nn       1-NN（局部邻域）
       logit    逻辑回归（线性对照）
       tree2/3  浅层决策树（少量样本即可表达交互规则）
       rf       随机森林（K 标签直接训练）
       ls       LabelSpreading 转导传播（RBF 图，K 个种子）
       ls_rf    LS 伪标签 → RF 重训（本提案主模型）
       proto_rf 原型伪标签 → RF 重训
       metric   度量学习原型网（MLP + 原型损失，对照用）
  ④ 特征子集对比：all29 / peer10 / peer_submit（peer_z_submit 是全项目第 1 重要特征，
     领先第 2 名一个数量级）→ 检验「小标签预算下，特征少而准是否胜过多而全」。

协议（全新，不与已有 5 折按学生分组协议混用）：
  FS-A 段内冷启动学习曲线（K 正 + K 负标注，R 次重复，query=该段其余行）
  FS-B 弱段（part5/6，正例 15/22）
  FS-C 跨段迁移（support 只来自其它段）
  FS-D 坐下级（早期）少样本（第 1/2 次坐下的截断特征）

参照系：
  REF-global-main   主协议 RF（原始空间）→ 应复现 pooled 0.9258（管线 sanity check）
  REF-inpart-full   段内全标签 RF（少样本同一特征空间）→ 标签充足时的上限
  REF-crosspart     跨段全标签 RF → 段间可迁移性

用法（246）：
  python3 analysis/fewshot/run_cs1_fs.py --data data/stage_dataset.npz \
      --ex data/exercise_dataset.npz --out results/cs1_fs
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402

A_COLS = [0, 1, 2, 3]
B_COLS = [4, 5, 6]
SEEDS = [42, 123, 777]
PEER_SUBMIT = 6          # agg10 列序：[ins,rem,pas,fg,fl,run,sub,total,distinct,span]
FEATSETS = {
    'all29': list(range(29)),
    'peer10': list(range(9, 19)),
    'peer_submit': [9 + PEER_SUBMIT],
}
METHODS = ['proto', 'nn', 'logit', 'tree2', 'tree3', 'rf', 'ls', 'ls_rf', 'proto_rf']
FS_RF_TREES = 100          # 少样本内部 RF 的树数（参照系仍用 300）


# ---------------------------------------------------------------- 特征空间
def part_median_mad(X, part):
    out = {}
    for p in np.unique(part):
        m = part == p
        Z = X[m].astype(float)
        med = np.median(Z, 0)
        mad = np.maximum(np.median(np.abs(Z - med), 0), 1e-3)
        out[int(p)] = (med, mad)
    return out


def std_vec(v, stats):
    med, mad = stats
    return (v - med) / mad


class Space:
    """abs9(9, 绝对量) + peer10(10, 同伴相对) + self10(10, 自参照)

    own_space(p)      该段用自己统计量标准化后的特征（段内原型打分用）
    donor_proto(idx, target_part)  其它段样本的原型映射到 target_part 坐标
    """

    def __init__(self, abs9, peer, selfv, part, use_self=True):
        self.abs9 = abs9.astype(float)
        self.peer = peer.astype(float)
        self.selfv = selfv.astype(float)
        self.part = np.asarray(part)
        self.use_self = use_self
        self.stats = part_median_mad(self.abs9, self.part)
        self._own = {}

    def dim(self):
        return 9 + self.peer.shape[1] + (self.selfv.shape[1] if self.use_self else 0)

    def own_space(self, p):
        if p not in self._own:
            rows = np.where(self.part == p)[0]
            z = std_vec(self.abs9[rows], self.stats[int(p)])
            cols = [z, self.peer[rows]]
            if self.use_self:
                cols.append(self.selfv[rows])
            self._own[p] = (rows, np.clip(np.hstack(cols), -10, 10))
        return self._own[p]

    def donor_proto(self, idx, target_part):
        a = self.abs9[idx].mean(0)
        z = std_vec(a, self.stats[int(target_part)])
        cols = [z, self.peer[idx].mean(0)]
        if self.use_self:
            cols.append(self.selfv[idx].mean(0))
        return np.hstack(cols)


def proto_score(X, cp, cn, qidx):
    dp = np.linalg.norm(X[qidx] - cp, axis=1)
    dn = np.linalg.norm(X[qidx] - cn, axis=1)
    return (dn - dp) / (dn + dp + 1e-9)


def nn_score(X, sp, sn, qidx):
    dp = np.min(np.linalg.norm(X[qidx][:, None, :] - X[sp][None, :, :], axis=2), axis=1)
    dn = np.min(np.linalg.norm(X[qidx][:, None, :] - X[sn][None, :, :], axis=2), axis=1)
    return (dn - dp) / (dn + dp + 1e-9)


def fit_rf(Xtr, ytr, Xte, seeds=SEEDS, n_estimators=300):
    ytr = np.asarray(ytr)
    if len(set(ytr.tolist())) < 2:
        return np.full(len(Xte), float(np.mean(ytr)) if len(ytr) else 0.0)
    acc = np.zeros(len(Xte))
    for sd in seeds:
        m = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2,
                                   max_features='sqrt', class_weight='balanced',
                                   random_state=sd, n_jobs=-1)
        m.fit(Xtr, ytr)
        acc += m.predict_proba(Xte)[:, 1]
    return acc / len(seeds)


def label_prop_knn(X, sp_l, sn_l, alpha=0.9, k=15, iters=20):
    """稀疏 kNN 图上的标签传播（替代 sklearn LabelSpreading，快 1~2 个数量级）

    · 图：每个点连 k 个最近邻，权重 exp(-d^2/2sigma^2)，对称化后行归一化
    · 传播：F <- alpha * S F + (1-alpha) * Y0，每轮把种子点夹回真值
    · 返回：全体行在"掉队类"上的软标签
    """
    n = X.shape[0]
    Y0 = np.full((n, 2), 0.5)
    Y0[sp_l] = [1.0, 0.0]        # 类 0 = 掉队
    Y0[sn_l] = [0.0, 1.0]        # 类 1 = 未掉队
    if n <= k + 2:
        return Y0[:, 0]
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argpartition(d2, k, axis=1)[:, :k]
    sigma2 = np.median(np.take_along_axis(d2, nn_idx, 1)) + 1e-6
    W = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k)
    cols = nn_idx.ravel()
    vals = np.exp(-d2[rows, cols] / (2 * sigma2))
    W[rows, cols] = vals
    W = np.maximum(W, W.T)
    deg = W.sum(1, keepdims=True)
    S = W / np.maximum(deg, 1e-9)
    F = Y0.copy()
    for _ in range(iters):
        F = alpha * (S @ F) + (1 - alpha) * Y0
        F[sp_l] = Y0[sp_l]
        F[sn_l] = Y0[sn_l]
    return F[:, 0]          # 类 0 = 掉队 的软标签（分数越大越危险）


def pseudo_rf(X, y_local, sup_l, q_l, base_scores, hi_q=0.70, lo_q=0.30):
    """用 base_scores 生成伪标签（top/bottom 分位）+ 真标签 → RF 重训"""
    thr_hi = np.quantile(base_scores, hi_q)
    thr_lo = np.quantile(base_scores, lo_q)
    pp = q_l[base_scores >= thr_hi]
    pn = q_l[base_scores <= thr_lo]
    tr = np.concatenate([sup_l, pp, pn])
    y_tr = np.concatenate([y_local[sup_l], np.ones(len(pp)), np.zeros(len(pn))])
    if len(set(y_tr.tolist())) < 2:
        return None
    return fit_rf(X[tr], y_tr, X[q_l], seeds=[42], n_estimators=FS_RF_TREES)


# ---------------------------------------------------------------- 少样本打分器
def score_method(method, X, y_local, sp_l, sn_l, q_l, rng):
    """返回 (scores_on_q, ok_flag)"""
    sup_l = np.concatenate([sp_l, sn_l])
    y_sup = np.concatenate([np.ones(len(sp_l)), np.zeros(len(sn_l))])

    if method == 'proto':
        cp = X[sp_l].mean(0)
        cn = X[sn_l].mean(0)
        return proto_score(X, cp, cn, q_l), True

    if method == 'nn':
        return nn_score(X, sp_l, sn_l, q_l), True

    if method == 'logit':
        if len(sp_l) < 1 or len(sn_l) < 1:
            return None, False
        m = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')
        m.fit(X[sup_l], y_sup)
        return m.predict_proba(X[q_l])[:, 1], True

    if method in ('tree2', 'tree3'):
        depth = 2 if method == 'tree2' else 3
        m = DecisionTreeClassifier(max_depth=depth, random_state=int(rng.integers(1e6)))
        m.fit(X[sup_l], y_sup)
        return m.predict_proba(X[q_l])[:, 1], True

    if method == 'rf':
        if len(sp_l) < 2 or len(sn_l) < 2:
            return None, False
        return fit_rf(X[sup_l], y_sup, X[q_l], seeds=[42],
                      n_estimators=FS_RF_TREES), True

    if method in ('ls', 'ls_rf'):
        if len(sp_l) < 1 or len(sn_l) < 1:
            return None, False
        base = label_prop_knn(X, sp_l, sn_l)[q_l]
        if method == 'ls':
            return base, True
        s2 = pseudo_rf(X, y_local, sup_l, q_l, base)
        return (s2, True) if s2 is not None else (base, True)

    if method == 'proto_rf':
        cp = X[sp_l].mean(0)
        cn = X[sn_l].mean(0)
        base = proto_score(X, cp, cn, q_l)
        s2 = pseudo_rf(X, y_local, sup_l, q_l, base)
        return (s2, True) if s2 is not None else (base, True)

    raise ValueError(method)


def fewshot_curve(space, y, part, shots, repeats, cols=None, methods=None,
                  donors=None, tag=''):
    """FS-A/B/C：少样本学习曲线，返回 {(part, K, method): dict(...)}
    支撑集在 (part, K, repeat) 上只抽一次、所有方法共用 → 方法间是配对比较"""
    methods = methods or METHODS
    res = {}
    for p in np.unique(part):
        rows, Xfull = space.own_space(int(p))
        X = Xfull[:, cols] if cols is not None else Xfull
        y_local = y[rows]
        pos_l = np.where(y_local == 1)[0]
        neg_l = np.where(y_local == 0)[0]
        kmax = min(len(pos_l), len(neg_l)) - 1
        for K in shots:
            if K > kmax:
                continue
            acc = {m: [] for m in methods}
            row_acc = {m: {} for m in methods}
            for r in range(repeats):
                rng = np.random.default_rng(1000 * p + 17 * K + r)
                sp_l = rng.choice(pos_l, K, replace=False)
                sn_l = rng.choice(neg_l, K, replace=False)
                mask = np.zeros(len(rows), bool)
                mask[sp_l] = True
                mask[sn_l] = True
                q_l = np.where(~mask)[0]
                if len(set(y_local[q_l].tolist())) < 2:
                    continue
                for method in methods:
                    if donors is None:
                        s, ok = score_method(method, X, y_local, sp_l, sn_l, q_l, rng)
                    else:
                        s, ok = score_method_donor(space, cols, method, y, part,
                                                   p, donors, K, q_l, rng)
                    if not ok or s is None:
                        continue
                    try:
                        acc[method].append(roc_auc_score(y_local[q_l], s))
                    except ValueError:
                        continue
                    for j, sc_ in zip(q_l, s):
                        row_acc[method][int(rows[j])] = \
                            row_acc[method].get(int(rows[j]), 0.0) + float(sc_)
            for method in methods:
                aucs = acc[method]
                if aucs:
                    nrep = max(1, len(aucs))
                    rs = {k2: v / nrep for k2, v in row_acc[method].items()}
                    res[(int(p), int(K), method)] = dict(
                        aucs=[float(a) for a in aucs],
                        auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                        n_pos=int(len(pos_l)), n_neg=int(len(neg_l)),
                        n_query=int(len(rows) - 2 * K), n_rep=len(aucs),
                        row_scores=rs)
    return res


def score_method_donor(space, cols, method, y, part, target_p, donors, K, q_l, rng):
    """跨段：原型从 donor 段借来（映射到目标段坐标），其余方法沿用 donor 的判别方向"""
    rows_t, Xt_full = space.own_space(int(target_p))
    Xt = Xt_full[:, cols] if cols is not None else Xt_full
    cps, cns = [], []
    for dp_ in donors:
        drows = np.where(part == dp_)[0]
        dpos = drows[y[drows] == 1]
        dneg = drows[y[drows] == 0]
        if len(dpos) < K or len(dneg) < K:
            continue
        dps = rng.choice(dpos, K, replace=False)
        dns = rng.choice(dneg, K, replace=False)
        cps.append(space.donor_proto(dps, target_p))
        cns.append(space.donor_proto(dns, target_p))
    if not cps:
        return None, False
    cp = np.mean(cps, axis=0)
    cn = np.mean(cns, axis=0)
    if cols is not None:
        cp = cp[cols]
        cn = cn[cols]
    return proto_score(Xt, cp, cn, q_l), True


# ---------------------------------------------------------------- 汇总
def pooled_from_res(res, methods=None, parts=None):
    out = {}
    if not res:
        return out
    for method in (methods or sorted({k[2] for k in res})):
        for K in sorted({k[1] for k in res if k[2] == method}):
            aucs, ws = [], []
            for (p, kk, m), v in res.items():
                if kk != K or m != method:
                    continue
                if parts is not None and p not in parts:
                    continue
                aucs.append(v['auc_mean'])
                ws.append(v['n_query'])
            if aucs:
                out[(method, K)] = float(np.average(aucs, weights=ws))
    return out


def per_part_table(res, K, method):
    parts = sorted({p for (p, k, m) in res if k == K and m == method})
    return {str(p): round(res[(p, K, method)]['auc_mean'], 4)
            for p in parts if (p, K, method) in res}


# ---------------------------------------------------------------- M5 度量学习对照
def metric_proto_net(X, y, part, train_parts, K=4, epochs=200, seed=0,
                     d_h=64, d_emb=32, lr=1e-3, device='cpu'):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    dev = torch.device(device)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(nn.Linear(X.shape[1], d_h), nn.ReLU(),
                                     nn.Linear(d_h, d_emb))

        def forward(self, x):
            return self.enc(x)

    pos_by_p, neg_by_p = {}, {}
    for p in train_parts:
        rows = np.where(part == p)[0]
        pos_by_p[p] = rows[y[rows] == 1]
        neg_by_p[p] = rows[y[rows] == 0]
    usable = [int(p) for p in train_parts
              if len(pos_by_p[p]) >= K + 1 and len(neg_by_p[p]) >= K + 1]
    if not usable:
        return None
    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        p = rng.choice(usable)
        rows = np.where(part == p)[0]
        sp = rng.choice(pos_by_p[p], K, replace=False)
        sn = rng.choice(neg_by_p[p], K, replace=False)
        rest = np.setdiff1d(rows, np.concatenate([sp, sn]))
        if len(rest) < 8:
            continue
        q = rng.choice(rest, min(16, len(rest)), replace=False)
        emb = net(Xt)
        cp = emb[sp].mean(0, keepdim=True)
        cn = emb[sn].mean(0, keepdim=True)
        d_p = torch.cdist(emb[q], cp).squeeze(1)
        d_n = torch.cdist(emb[q], cn).squeeze(1)
        logits = torch.stack([-d_p, -d_n], dim=1)
        loss = nn.functional.cross_entropy(
            logits, torch.zeros(len(q), dtype=torch.long, device=dev))
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return net(Xt).cpu().numpy()


def fewshot_curve_emb(E, y, part, shots, repeats):
    """嵌入空间里的原型打分（对照用）"""
    res = {}
    for p in np.unique(part):
        rows = np.where(part == p)[0]
        y_local = y[rows]
        pos_l = np.where(y_local == 1)[0]
        neg_l = np.where(y_local == 0)[0]
        kmax = min(len(pos_l), len(neg_l)) - 1
        for K in shots:
            if K > kmax:
                continue
            aucs = []
            for r in range(repeats):
                rng = np.random.default_rng(1000 * p + 17 * K + r)
                sp_l = rng.choice(pos_l, K, replace=False)
                sn_l = rng.choice(neg_l, K, replace=False)
                mask = np.zeros(len(rows), bool)
                mask[sp_l] = True
                mask[sn_l] = True
                q_l = np.where(~mask)[0]
                if len(set(y_local[q_l].tolist())) < 2:
                    continue
                cp = E[rows][sp_l].mean(0)
                cn = E[rows][sn_l].mean(0)
                s = proto_score(E[rows], cp, cn, q_l)
                aucs.append(roc_auc_score(y_local[q_l], s))
            if aucs:
                res[(int(p), int(K), 'metric_proto')] = dict(
                    auc_mean=float(np.mean(aucs)), auc_std=float(np.std(aucs)),
                    n_pos=int(len(pos_l)), n_neg=int(len(neg_l)),
                    n_query=int(len(rows) - 2 * K), n_rep=len(aucs))
    return res


# ---------------------------------------------------------------- 参照系
def ref_inpart_full(X, y, sid, part, n_splits=5, seed=42):
    oof = np.zeros(len(y))
    for p in np.unique(part):
        rows = np.where(part == p)[0]
        if len(rows) < 30 or len(set(y[rows].tolist())) < 2:
            continue
        uniq = np.unique(sid[rows])
        rng = np.random.default_rng(seed)
        buckets = [[] for _ in range(n_splits)]
        for cls in (0, 1):
            ss = [int(s) for s in uniq if int(y[rows][sid[rows] == s][0]) == cls]
            rng.shuffle(ss)
            for i, s in enumerate(ss):
                buckets[i % n_splits].append(s)
        for b in range(n_splits):
            te_s = set(buckets[b])
            te = np.array([r for r in rows if int(sid[r]) in te_s])
            tr = np.array([r for r in rows if int(sid[r]) not in te_s])
            if len(tr) < 20 or len(te) == 0:
                continue
            oof[te] = fit_rf(X[tr], y[tr], X[te])
    return oof


def ref_crosspart(X, y, part):
    oof = np.zeros(len(y))
    for p in np.unique(part):
        te = np.where(part == p)[0]
        tr = np.where(part != p)[0]
        if len(tr) < 50 or len(set(y[te].tolist())) < 2:
            continue
        oof[te] = fit_rf(X[tr], y[tr], X[te])
    return oof


def ref_global(X, y, sid, part, final):
    folds = sc.group_folds(sid, final, n_splits=5, seed=42)
    oof = np.zeros(len(y))
    for tr, te in folds:
        oof[te] = fit_rf(X[tr], y[tr], X[te])
    return oof


# ---------------------------------------------------------------- 坐下级
def build_sitting_raw(ex_path, n_sit=1, gap_sec=30 * 60):
    ex = np.load(ex_path, allow_pickle=True)
    raw = ex['raw'].astype(np.float64)
    esid, epar, eexr = ex['sid'], ex['part'], ex['exercise']
    sp_rows = defaultdict(list)
    for i in range(len(esid)):
        sp_rows[(int(esid[i]), int(epar[i]))].append(i)
    out = {}
    for (s, p), idxs in sp_rows.items():
        idxs = sorted(idxs, key=lambda i: (eexr[i], raw[i, 10]))
        sittings, cur = [], [idxs[0]]
        for a, b in zip(idxs[:-1], idxs[1:]):
            if raw[b, 10] - raw[a, 11] < gap_sec:
                cur.append(b)
            else:
                sittings.append(cur)
                cur = [b]
        sittings.append(cur)
        if len(sittings) < n_sit:
            continue
        use = np.array([i for sit in sittings[:n_sit] for i in sit])
        cnt = raw[use, 1:8].sum(0)
        n_ev = raw[use, 0].sum()
        span = max((raw[use, 11].max() - raw[use, 10].min()) / 3600.0, 1e-3)
        out[(s, p)] = np.concatenate([cnt, [n_ev], [float(len(use)), span]])
    return out


def sitting_features(sit_raw, ds):
    part_key = {}
    for i in range(len(ds['sid'])):
        part_key[(int(ds['sid'][i]), int(ds['part'][i]))] = i
    rows, RAW = [], []
    for (s, p), v in sit_raw.items():
        i = part_key.get((s, p))
        if i is None:
            continue
        rows.append(i)
        RAW.append(v)
    rows = np.array(rows)
    RAW = np.array(RAW)
    sub = dict(ds)
    sub['RAW'] = RAW
    sub['part'] = ds['part'][rows]
    sub['sid'] = ds['sid'][rows]
    sub['stage_idx'] = ds['stage_idx'][rows]
    DER = sc.derive(sub)
    abs9 = np.hstack([DER[:, A_COLS], DER[:, B_COLS], DER[:, [7, 8]]]).astype(float)
    n = len(rows)
    z = sc.peer_z(sub, np.arange(n), np.arange(n), sub['part'])
    return rows, abs9, z



# ---------------------------------------------------------------- FS-E 稀有子类
def fewshot_rare_class(X, y_rare, shots, repeats, cols=None, methods=None):
    """稀有子类少样本：'走完全程仍挂科'（正例 ~59 人）vs '完课通过'

    指标 = 学生级 AUC（单一群体，无 part 内加权）。
    """
    methods = methods or METHODS
    X = X if cols is None else X[:, cols]
    pos = np.where(y_rare == 1)[0]
    neg = np.where(y_rare == 0)[0]
    res = {}
    for K in shots:
        if K > min(len(pos), len(neg)) - 1:
            continue
        acc = {m: [] for m in methods}
        for r in range(repeats):
            rng = np.random.default_rng(7000 + 13 * K + r)
            sp = rng.choice(pos, K, replace=False)
            sn = rng.choice(neg, K, replace=False)
            mask = np.zeros(len(y_rare), bool)
            mask[sp] = True
            mask[sn] = True
            q = np.where(~mask)[0]
            if len(set(y_rare[q].tolist())) < 2:
                continue
            for method in methods:
                s, ok = score_method(method, X, y_rare, sp, sn, q, rng)
                if not ok or s is None:
                    continue
                try:
                    acc[method].append(roc_auc_score(y_rare[q], s))
                except ValueError:
                    continue
        for method in methods:
            aucs = acc[method]
            if aucs:
                res[(method, K)] = dict(auc_mean=float(np.mean(aucs)),
                                        auc_std=float(np.std(aucs)),
                                        n_rep=len(aucs))
    return res


# ---------------------------------------------------------------- main
def main(args):
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)
    ds = sc.load_dataset(args.data)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part_ids'])
    ds['DER'] = sc.derive(ds)
    ds['SELF'] = sc.self_ref(ds)
    y, part, sid, final = ds['drop'], ds['part'], ds['sid'], ds['final']
    n = len(y)

    DER = ds['DER']
    abs9 = np.hstack([DER[:, A_COLS], DER[:, B_COLS], DER[:, [7, 8]]]).astype(float)
    peer = sc.peer_z(ds, np.arange(n), np.arange(n), part)
    selfv = ds['SELF'].astype(float)

    space21 = Space(abs9, peer, selfv, part, use_self=False)
    space31 = Space(abs9, peer, selfv, part, use_self=True)
    X_main = np.clip(np.hstack([abs9, peer, selfv]), -10, 10)

    out = {'config': vars(args), 'per_part_pos_neg': {}}
    for p in np.unique(part):
        rows = part == p
        out['per_part_pos_neg'][str(int(p))] = dict(
            n=int(rows.sum()), pos=int(y[rows].sum()),
            neg=int((rows & (y == 0)).sum()),
            base_rate=round(float(y[rows].mean()), 4))
    print(f'== 阶段级 {n} 行 / {len(np.unique(sid))} 生；abs9={abs9.shape} '
          f'peer={peer.shape} self={selfv.shape}', flush=True)

    # ---------- 参照系 ----------
    print('\n=== 参照系 ===', flush=True)
    ref = {}
    oof_main = ref_global(X_main, y, sid, part, final)
    pp = sc.per_part_auc(y, oof_main, part)
    ref['REF-global-main (主协议原始空间 RF)'] = dict(
        pooled=round(sc.pooled(pp), 4),
        per_part={str(k): round(v['auc'], 4) for k, v in pp.items()})
    print(f'  REF-global-main: pooled={ref["REF-global-main (主协议原始空间 RF)"]["pooled"]}',
          flush=True)

    X31_full = np.zeros((n, space31.own_space(1)[1].shape[1]))
    for p in np.unique(part):
        rows, Xp = space31.own_space(int(p))
        X31_full[rows] = Xp
    for name, oof in [('REF-inpart-full (段内全标签 RF, 少样本同空间)',
                       ref_inpart_full(X31_full, y, sid, part)),
                      ('REF-crosspart (跨段全标签 RF)',
                       ref_crosspart(X31_full, y, part))]:
        pp = sc.per_part_auc(y, oof, part)
        ref[name] = dict(pooled=round(sc.pooled(pp), 4),
                         per_part={str(k): round(v['auc'], 4) for k, v in pp.items()})
        print(f'  {name}: pooled={ref[name]["pooled"]}', flush=True)
    out['reference'] = ref

    # ---------- FS-A ----------
    shots = [1, 2, 4, 8, 16]
    print('\n=== FS-A 段内冷启动学习曲线（K 正 + K 负）===', flush=True)
    fsA = {}
    for fs_name, cols in FEATSETS.items():
        res = fewshot_curve(space31, y, part, shots, args.repeats, cols=cols)
        pooled = pooled_from_res(res)
        for method in METHODS:
            for K in shots:
                key = f'{fs_name}|{method}|K={K}'
                if (method, K) not in pooled:
                    continue
                fsA[key] = dict(pooled=round(pooled[(method, K)], 4),
                                per_part=per_part_table(res, K, method))
        print(f'  --- 特征集 {fs_name} 完成（{time.time()-t0:.0f}s）---', flush=True)
        for method in METHODS:
            vals = [f'{pooled[(method, K)]:.4f}' if (method, K) in pooled else '—'
                    for K in shots]
            print(f'  [{fs_name}] {method:<9} ' + '  '.join(vals), flush=True)
        print('  ' + ' ' * 19 + '  '.join(f'K={k}' for k in shots), flush=True)
    out['fs_a_inpart_coldstart'] = fsA

    # ---------- FS-B 弱段 ----------
    print('\n=== FS-B 弱段（part5/6）===', flush=True)
    weak = [5, 6]
    fsB = {}
    res = fewshot_curve(space31, y, part, [1, 2, 4, 8], args.repeats,
                        cols=FEATSETS['all29'])
    for method in METHODS:
        row = []
        for K in [1, 2, 4, 8]:
            p_all = pooled_from_res(res, methods=[method]).get((method, K))
            p_wk = pooled_from_res(res, methods=[method], parts=weak).get((method, K))
            row.append((K, p_all, p_wk))
        fsB[method] = {str(K): dict(pooled_all=round(a, 4) if a else None,
                                    weak_pooled=round(b, 4) if b else None)
                       for K, a, b in row}
        print(f'  {method:<9} ' + '  '.join(
            f'K={K}:{a if a else float("nan"):.3f}/{b if b else float("nan"):.3f}'
            for K, a, b in row), flush=True)
    out['fs_b_weak_parts'] = fsB

    # ---------- FS-C 跨段迁移 ----------
    print('\n=== FS-C 跨段原型迁移 ===', flush=True)
    fsC = {}
    for K in [2, 4, 8]:
        res = {}
        for tgt in np.unique(part):
            donors = [int(p) for p in np.unique(part) if int(p) != int(tgt)]
            res.update(fewshot_curve(space31, y, part, [K], args.repeats,
                                     cols=FEATSETS['all29'], methods=['proto'],
                                     donors=donors))
        pooled = pooled_from_res(res)
        fsC[f'K={K}'] = dict(pooled=round(pooled.get(('proto', K), float('nan')), 4),
                             per_part=per_part_table(res, K, 'proto'))
        print(f'  K={K}: pooled={fsC[f"K={K}"]["pooled"]} 逐段={fsC[f"K={K}"]["per_part"]}',
              flush=True)
    out['fs_c_crosspart_transfer'] = fsC

    # ---------- M5 度量学习原型网 ----------
    print('\n=== M5 度量学习原型网（MLP + 原型损失，训练段 1-6）===', flush=True)
    fsM5 = {}
    for K in [1, 2, 4, 8]:
        E = metric_proto_net(X31_full, y, part, [1, 2, 3, 4, 5, 6], K=K,
                              epochs=args.epochs, seed=42, device=args.device)
        if E is None:
            continue
        res = fewshot_curve_emb(E, y, part, [K], args.repeats)
        pooled = pooled_from_res(res)
        fsM5[f'K={K}'] = dict(pooled=round(pooled.get(('metric_proto', K), float('nan')), 4),
                              per_part=per_part_table(res, K, 'metric_proto'))
        print(f'  K={K}: pooled={fsM5[f"K={K}"]["pooled"]}', flush=True)
    out['m5_metric_proto_net'] = fsM5

    # ---------- FS-D 坐下级（早期） ----------
    if args.ex and os.path.exists(args.ex):
        print('\n=== FS-D 坐下级（早期）少样本 ===', flush=True)
        fsD = {}
        for n_sit in [1, 2]:
            sit_raw = build_sitting_raw(args.ex, n_sit=n_sit)
            rows, abs9s, peers = sitting_features(sit_raw, ds)
            sub_part, sub_y = part[rows], y[rows]
            sp_sit = Space(abs9s, peers, np.zeros((len(rows), 10)), sub_part,
                           use_self=False)
            Xs = np.zeros((len(rows), sp_sit.own_space(1)[1].shape[1]))
            for p in np.unique(sub_part):
                rr, Xp = sp_sit.own_space(int(p))
                Xs[rr] = Xp
            oof = fit_rf(Xs, sub_y, Xs)
            pp = sc.per_part_auc(sub_y, oof, sub_part)
            rec = {'rows': int(len(rows)), 'ref_full_RF_pooled': round(sc.pooled(pp), 4)}
            res = fewshot_curve(sp_sit, sub_y, sub_part, [2, 4, 8], args.repeats,
                                cols=list(range(19)))
            for method in METHODS:
                for K in [2, 4, 8]:
                    pv = pooled_from_res(res, methods=[method]).get((method, K))
                    if pv is not None:
                        rec[f'{method}|K={K}'] = round(pv, 4)
            fsD[f'sit{n_sit}'] = rec
            print(f'  sit{n_sit}: rows={rec["rows"]} 全标签RF={rec["ref_full_RF_pooled"]}',
                  flush=True)
            for method in METHODS:
                vals = [f'{method}@{K}={rec.get(f"{method}|K={K}", "—")}' for K in [2, 4, 8]]
                print('    ' + '  '.join(vals), flush=True)
        out['fs_d_sitting_early'] = fsD

    # ---------- FS-E 稀有子类（完课仍挂科）----------
    print('\n=== FS-E 稀有子类少样本：走完全程仍挂科 ===', flush=True)
    rows7 = np.where(part == 7)[0]
    X7 = X31_full[rows7]
    y7 = final[rows7]
    sid7 = sid[rows7]
    folds7 = sc.group_folds(sid7, y7, n_splits=5, seed=42)
    oof7 = np.zeros(len(y7))
    for tr, te in folds7:
        oof7[te] = fit_rf(X7[tr], y7[tr], X7[te])
    rec7 = {'n': int(len(y7)), 'pos': int(y7.sum()),
            'ref_full_RF_auc': round(float(roc_auc_score(y7, oof7)), 4)}
    res7 = fewshot_rare_class(X7, y7, [1, 2, 4, 8, 16], args.repeats)
    for (method, K), v in res7.items():
        rec7[f'{method}|K={K}'] = dict(auc_mean=round(v['auc_mean'], 4),
                                       auc_std=round(v['auc_std'], 4))
    out['fs_e_rare_class'] = rec7
    print(f'  n={rec7["n"]} pos={rec7["pos"]} 全标签RF AUC={rec7["ref_full_RF_auc"]}',
          flush=True)
    for method in METHODS:
        vals = [f'K={K}:{res7.get((method, K), {}).get("auc_mean", "—")}'
                for K in [1, 2, 4, 8, 16]]
        print(f'  {method:<9} ' + '  '.join(vals), flush=True)

    # ---------- 显著性：最佳少样本 vs 段内全标签 ----------
    print('\n=== 显著性（配对 bootstrap 2000）===', flush=True)
    sig = {}
    best_key = None
    for fs_name in FEATSETS:
        for method in METHODS:
            for K in [4, 8]:
                v = fsA.get(f'{fs_name}|{method}|K={K}')
                if v and (best_key is None or v['pooled'] > best_key[1]['pooled']):
                    best_key = (f'{fs_name}|{method}|K={K}', v)
    if best_key:
        fs_name, method = best_key[0].split('|')[0], best_key[0].split('|')[1]
        K = int(best_key[0].split('K=')[1])
        cols = FEATSETS[fs_name]
        res = fewshot_curve(space31, y, part, [K], args.repeats, cols=cols,
                            methods=[method])
        oof_fs = np.full(len(y), np.nan)
        for (p, kk, m), v in res.items():
            for r2, s2 in v.get('row_scores', {}).items():
                oof_fs[r2] = s2
        oof_in = ref_inpart_full(X31_full, y, sid, part)
        d = sc.paired_boot(y, oof_fs, oof_in, part, n=2000)
        sig[f'{best_key[0]} − REF-inpart-full'] = dict(
            obs=round(float(d['obs']), 4),
            ci=[round(float(d['lo']), 4), round(float(d['hi']), 4)],
            p_le0=round(float(d['p_le0']), 4))
        print(f'  最佳少样本 {best_key[0]}: pooled={best_key[1]["pooled"]}；'
              f'Δ vs inpart-full={d["obs"]:+.4f} CI[{d["lo"]:+.4f},{d["hi"]:+.4f}] '
              f'P(Δ≤0)={d["p_le0"]*100:.1f}%', flush=True)
        out['best_fewshot'] = dict(key=best_key[0], pooled=best_key[1]['pooled'])
    out['significance'] = sig

    with open(os.path.join(args.out, 'cs1_fs.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(args.out, 'ctx.npz'),
                        X31=X31_full, y=y, part=part, sid=sid)
    write_md(out, os.path.join(args.out, 'cs1_fs.md'))
    print(f'\n完成 → {args.out}/cs1_fs.md（用时 {time.time()-t0:.0f}s）', flush=True)


def write_md(out, path):
    shots = [1, 2, 4, 8, 16]
    L = ['# CS1-FSP · 段锚定少样本学习（实测结果）', '',
         f"配置：repeats={out['config'].get('repeats')}，device={out['config'].get('device')}", '']
    L += ['## 每段正负例分布', '', '| part | n | pos | neg | 基准率 |', '|---:|---:|---:|---:|---:|']
    for p, v in out['per_part_pos_neg'].items():
        L.append(f'| {p} | {v["n"]} | {v["pos"]} | {v["neg"]} | {v["base_rate"]} |')
    L += ['', '## 参照系', '', '| 模型 | pooled | 逐段 |', '|---|---:|---|']
    for k, v in out['reference'].items():
        pp = ' '.join(f'p{a}={b}' for a, b in v['per_part'].items())
        L.append(f'| {k} | {v["pooled"]} | {pp} |')
    L += ['', '## FS-A 段内冷启动学习曲线（K 正 + K 负标注，pooled AUC）', '']
    for fs_name in FEATSETS:
        L += [f'### 特征集 {fs_name}', '',
              '| 方法 | ' + ' | '.join(f'K={k}' for k in shots) + ' |',
              '|---|' + '---:|' * len(shots)]
        for method in METHODS:
            vals = []
            for K in shots:
                r = out['fs_a_inpart_coldstart'].get(f'{fs_name}|{method}|K={K}')
                vals.append(str(r['pooled']) if r else '—')
            L.append(f'| {method} | ' + ' | '.join(vals) + ' |')
        L.append('')
    L += ['## FS-B 弱段（part5/6）：全段 pooled / 弱段 pooled', '',
          '| 方法 | ' + ' | '.join(f'K={k}' for k in [1, 2, 4, 8]) + ' |',
          '|---|' + '---:|' * 4]
    for method, rec in out.get('fs_b_weak_parts', {}).items():
        vals = []
        for K in ['1', '2', '4', '8']:
            r = rec.get(K, {})
            vals.append(f'{r.get("pooled_all")} / {r.get("weak_pooled")}')
        L.append(f'| {method} | ' + ' | '.join(vals) + ' |')
    L += ['', '## FS-C 跨段原型迁移', '', '| K | pooled | 逐段 |', '|---:|---:|---|']
    for K, v in out.get('fs_c_crosspart_transfer', {}).items():
        L.append(f'| {K} | {v["pooled"]} | ' + ' '.join(
            f'p{a}={b}' for a, b in v['per_part'].items()) + ' |')
    L += ['', '## M5 度量学习原型网', '', '| K | pooled |', '|---:|---:|']
    for K, v in out.get('m5_metric_proto_net', {}).items():
        L.append(f'| {K} | {v["pooled"]} |')
    if 'fs_d_sitting_early' in out:
        L += ['', '## FS-D 坐下级（早期）少样本', '']
        for k, v in out['fs_d_sitting_early'].items():
            L.append(f'### {k}（rows={v["rows"]}，全标签 RF pooled={v["ref_full_RF_pooled"]}）')
            L += ['', '| 方法 | K=2 | K=4 | K=8 |', '|---|---:|---:|---:|']
            for method in METHODS:
                L.append(f'| {method} | {v.get(f"{method}|K=2", "—")} | '
                         f'{v.get(f"{method}|K=4", "—")} | {v.get(f"{method}|K=8", "—")} |')
            L.append('')
    if 'fs_e_rare_class' in out:
        rec = out['fs_e_rare_class']
        L += ['', '## FS-E 稀有子类（完课仍挂科）', '',
              f'n={rec.get("n")}，正例={rec.get("pos")}，全标签 RF AUC={rec.get("ref_full_RF_auc")}',
              '', '| 方法 | ' + ' | '.join(f'K={k}' for k in [1, 2, 4, 8, 16]) + ' |',
              '|---|' + '---:|' * 5]
        for method in METHODS:
            vals = []
            for K in [1, 2, 4, 8, 16]:
                r = rec.get(f'{method}|K={K}')
                vals.append(str(r['auc_mean']) if r else '—')
            L.append(f'| {method} | ' + ' | '.join(vals) + ' |')
    if 'significance' in out:
        L += ['', '## 显著性', '', '| 对比 | Δ | 95% CI | P(Δ≤0) |', '|---|---:|---|---:|']
        for k, v in out['significance'].items():
            L.append(f'| {k} | {v["obs"]} | [{v["ci"][0]}, {v["ci"][1]}] | {v["p_le0"]} |')
    with open(path, 'w') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--ex', default='data/exercise_dataset.npz')
    ap.add_argument('--out', default='results/cs1_fs')
    ap.add_argument('--repeats', type=int, default=30)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--device', default='cuda')
    main(ap.parse_args())
