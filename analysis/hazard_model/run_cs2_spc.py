#!/usr/bin/env python3
"""
CS2-SPC · Sitting-anchored Peer-contrastive Hazard Transformer
=================================================================================
CS1-SHR 的后继模型原型（2026-11-02）。设计依据全部来自本项目已完成的实测：

  ① 决策单元 = "坐下"（sitting），不是"习题"、不是"整段"
     - 习题级 hazard pooled 0.4920（低于随机）→ 粒度与事件基准率不匹配
     - 习题截断 第1题 0.7915 / 坐下截断 第1次 0.8665（+0.075）/ 整段 0.9258
     - 会话结构 14 维**加进特征集**无增量（0.8644 vs 0.8665）
       → 结构的价值在"定义观测单位"，不在"提供特征"
  ② 同伴相对信号是主导特征
     - peer_z_submit 在 7/7 分段都是第 1 重要特征（0.057–0.127，领先第 2 名一个数量级）
     - counterfactual：5 个真实学生唯一需改动的特征都是 peer_z_submit
     - 现有实现把 peer-z 当 31 维里的一个标量交给 tree 切分，没有"相对同伴"的归纳偏置
  ③ 卡住表征随进度变化
     - rem_over_ins 在第2-3段 HR 2.5–2.7；peer_z_paste 第4-5段 HR 2.2–5.2
     - 但"part 身份入模"历史上 LOPO=0.5000（随机）→ **不能**用 part identity
       → 改用**连续进度** u = 题序/基准题量 做 FiLM 调制（可跨 part 泛化）
  ④ 已被证伪、不要再试
     - 换学习器（LGBM/HGB/MoE/LambdaRank/序数回归）全部 ≤ RF
     - 加特征（多尺度 14 维 −0.0058；dl9 判别增量不显著 P=25.2%）
     - 直接优化主指标（LambdaRank −0.0229）

模型 = 因果 Transformer（token = 习题，坐次为观测锚点）
      + 进度条件化 FiLM（连续量，非 part 身份）
      + 离散时间 hazard 头（坐次末输出）
      + 同伴成对对比损失（可选）
      + 阶段级上下文广播（与 RF 基线完全同源、同折统计量、早警不泄漏）

协议与既有实验一致：按学生分组 5 折 × 3 seeds；peer 统计量只用训练折；
主指标 = part 内 pooled AUC。

跑法（246 /home/ubuntu/studentrisk_stage）：
  python3 analysis/hazard_model/run_cs2_spc.py --out results/cs2_spc
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402

SIT_GAP_SEC = 30 * 60
SUB_DIM = 6      # 截断聚合 12 列布局: [ins,rem,paste,fg,fl,run,submit,total,n_ex,span,ttd_max,ttd_min]
CTX_OFF = 9     # 阶段级上下文里 peer-z 的起始列

_TORCH = None
_NN = None


# ----------------------------------------------------------------------------
# 数据准备
# ----------------------------------------------------------------------------
def build_sittings(raw_ex, ex_sid, ex_par, ex_exr):
    """(sid, part) -> 坐次列表（每坐次 = 习题索引列表）"""
    sp_rows = defaultdict(list)
    for i in range(len(ex_sid)):
        sp_rows[(int(ex_sid[i]), int(ex_par[i]))].append(i)
    sittings = {}
    for k, idxs in sp_rows.items():
        idxs = sorted(idxs, key=lambda i: ex_exr[i])
        cur, out = [idxs[0]], []
        for a, b in zip(idxs[:-1], idxs[1:]):
            gap = raw_ex[b, 10] - raw_ex[a, 11]        # 下一题开始 - 上一题结束
            if gap < SIT_GAP_SEC:
                cur.append(b)
            else:
                out.append(cur)
                cur = [b]
        out.append(cur)
        sittings[k] = out
    return sittings


def token_features(raw_ex, idxs, base_ex):
    """每习题 token 特征（22 维）+ 坐次末标记 + 连续进度 u"""
    L = len(idxs)
    if L == 0:
        return np.zeros((0, 22), np.float32), np.zeros(0, bool), np.zeros(0, np.float32)
    g = raw_ex[idxs]
    n, ins, rem, pas = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
    fg, fl, run, sub = g[:, 4], g[:, 5], g[:, 6], g[:, 7]
    ttd_max, ttd_min, ts_min, ts_max = g[:, 8], g[:, 9], g[:, 10], g[:, 11]

    T = np.zeros((L, 22), np.float32)
    T[:, 0] = np.log1p(n)
    T[:, 1] = np.log1p(ins)
    T[:, 2] = np.log1p(rem)
    T[:, 3] = np.log1p(pas)
    T[:, 4] = np.log1p(fg)
    T[:, 5] = np.log1p(fl)
    T[:, 6] = np.log1p(run)
    T[:, 7] = np.log1p(sub)
    T[:, 8] = run / np.maximum(sub, 1.0)
    T[:, 9] = rem / np.maximum(ins, 1.0)
    T[:, 10] = fl / np.maximum(n, 1.0)
    T[:, 11] = ins / np.maximum(n, 1.0)
    T[:, 12] = fg / np.maximum(n, 1.0)
    T[:, 13] = np.log1p((ts_min - ts_min.min()) / 3600.0)
    gap_prev = np.zeros(L)
    gap_prev[1:] = np.maximum(ts_min[1:] - ts_max[:-1], 0.0) / 60.0
    T[:, 14] = np.log1p(gap_prev)
    T[:, 15] = np.log1p((ts_max - ts_min) / 3600.0)
    T[:, 16] = np.arange(L) / max(base_ex, 1)
    T[:, 17] = ttd_max / 3600.0
    T[:, 18] = (ttd_max - ttd_min) / 3600.0

    # 坐次边界
    bounds = [0] + [i for i in range(1, L)
                    if (ts_min[i] - ts_max[i - 1]) >= SIT_GAP_SEC] + [L]
    is_end = np.zeros(L, bool)
    in_sit = np.zeros(L, np.float32)
    sit_id = np.zeros(L, np.int64)
    for si, (b0, b1) in enumerate(zip(bounds[:-1], bounds[1:])):
        sz = max(b1 - b0, 1)
        is_end[b1 - 1] = True
        for i in range(b0, b1):
            in_sit[i] = (i - b0) / sz
            sit_id[i] = si
    T[:, 19] = in_sit
    T[:, 20] = sit_id / 5.0
    T[:, 21] = is_end.astype(np.float32)
    u = T[:, 16].astype(np.float32)
    return T, is_end, u


def peer_stats(RAW, rows, group):
    out = {}
    for g in np.unique(group[rows]):
        idx = np.intersect1d(np.where(group == g)[0], rows)
        if len(idx) == 0:
            continue
        Xg = RAW[idx]
        med = np.median(Xg, 0)
        mad = np.maximum(np.median(np.abs(Xg - med), 0), 1e-3)
        out[int(g)] = dict(med=med, mad=mad)
    return out


def group_peer_z(RAW, rows, group, stats):
    out = np.zeros((len(rows), RAW.shape[1]))
    for j, r in enumerate(rows):
        st = stats.get(int(group[r]))
        if st is None:
            continue
        out[j] = (RAW[r] - st['med']) / st['mad']
    return np.clip(out, -5, 5)


def stage_context(ds, RAWt, pos, gids, part_sel, peer_z, SELF_sel):
    """与 RF 基线完全同源的 31 维阶段级上下文

    pos     : RAWt 中被选行的位置（长度 n）
    gids    : 对应的全局阶段行号（长度 n，用于 stage_idx）
    part_sel: 对应的 part（长度 n）
    peer_z  : 同折统计量算出的 peer z（长度 n）
    SELF_sel: 自参照特征（长度 n）
    """
    R = RAWt[pos]
    SPAN = np.maximum(R[:, 9], 1e-3)
    INS = np.maximum(R[:, 0], 1e-6)
    n_ex = R[:, 8]
    base = np.array([ds['base_ex'][int(p)] for p in part_sel], dtype=float)
    A = np.column_stack([np.log1p(R[:, 7]), np.log1p(n_ex), np.log1p(R[:, 9]),
                         np.log1p(R[:, 7] / SPAN)])
    B = np.column_stack([R[:, 1] / INS, R[:, 4] / SPAN,
                         R[:, 6] / np.maximum(R[:, 7], 1)])
    C = np.column_stack([n_ex / base, ds['stage_idx'][gids].astype(float)])
    return np.hstack([A, B, C, peer_z, SELF_sel])


# ----------------------------------------------------------------------------
# 截断聚合（与 run_sitting_truncation.py 同口径）
# ----------------------------------------------------------------------------
def trunc_agg(sittings, part_key, raw_ex, k):
    """前 k 次坐下的 12 列聚合；k=None → 全段"""
    rows, R = [], []
    for (s, p), sids in sittings.items():
        if k is not None and len(sids) < k:
            continue
        use = [i for sit in (sids if k is None else sids[:k]) for i in sit]
        sub = np.array(use)
        cnt_ = raw_ex[sub, 1:8].sum(0)
        n = raw_ex[sub, 0].sum()
        span = (raw_ex[sub, 11].max() - raw_ex[sub, 10].min()) / 3600.0
        R.append([cnt_[0], cnt_[1], cnt_[2], cnt_[3], cnt_[4], cnt_[5], cnt_[6],
                  n, float(len(use)), max(span, 1e-3), raw_ex[sub, 8].max(),
                  raw_ex[sub, 9].min()])
        rows.append(part_key[(s, p)])
    return np.array(rows), np.array(R)


# ----------------------------------------------------------------------------
# RF 基线
# ----------------------------------------------------------------------------
def fit_rf(Xtr, ytr, Xte, seeds):
    from sklearn.ensemble import RandomForestClassifier
    if len(set(np.asarray(ytr).tolist())) < 2:
        return np.full(len(Xte), float(np.mean(ytr)) if len(ytr) else 0.0)
    acc = np.zeros(len(Xte))
    for sd in seeds:
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   max_features='sqrt', class_weight='balanced',
                                   random_state=sd, n_jobs=-1)
        m.fit(Xtr, ytr)
        acc += m.predict_proba(Xte)[:, 1]
    return acc / len(seeds)


def rf_baseline(ds, RAWt, rows, SELF, folds, seeds, tag):
    part = ds['part_ids']
    lab = ds['drop']
    pos = {int(r): i for i, r in enumerate(rows)}
    oof = np.zeros(len(rows))
    for tr, te in folds:
        tr_rows = np.array([r for r in tr if int(r) in pos])
        te_rows = np.array([r for r in te if int(r) in pos])
        if len(tr_rows) < 50 or len(te_rows) == 0:
            continue
        it = np.array([pos[int(r)] for r in tr_rows])
        ie = np.array([pos[int(r)] for r in te_rows])
        stats = peer_stats(RAWt, it, part[rows])
        ztr = group_peer_z(RAWt, it, part[rows], stats)
        zte = group_peer_z(RAWt, ie, part[rows], stats)
        Xtr = stage_context(ds, RAWt, it, rows[it], part[rows][it], ztr, SELF[tr_rows])
        Xte = stage_context(ds, RAWt, ie, rows[ie], part[rows][ie], zte, SELF[te_rows])
        oof[ie] = fit_rf(Xtr, lab[rows][it], Xte, seeds)
    return sc.metrics(lab[rows], oof, part[rows], tag), oof


# ----------------------------------------------------------------------------
# 深度模型
# ----------------------------------------------------------------------------
def build_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def make_model(torch, nn, d_in, use_film, d=64, nlayer=2, heads=4, ff=128, maxlen=64):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d))
            self.pos = nn.Parameter(torch.zeros(1, maxlen, d))
            nn.init.normal_(self.pos, std=0.02)
            self.use_film = use_film
            if use_film:
                self.film = nn.Linear(1, 2 * d)
                nn.init.zeros_(self.film.weight)
                nn.init.zeros_(self.film.bias)
            layer = nn.TransformerEncoderLayer(d_model=d, nhead=heads, dim_feedforward=ff,
                                               dropout=0.1, activation='gelu',
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, nlayer)
            self.sit_head = nn.Linear(d, 1)
            self.attn = nn.Linear(d, 1)
            self.part_head = nn.Linear(d, 1)

        def forward(self, x, u, sit_end, pad_mask):
            B, L, _ = x.shape
            h = self.inp(x) + self.pos[:, :L]
            if self.use_film:
                g, b = self.film(u).chunk(2, dim=-1)
                h = h * (1.0 + g) + b
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), 1)
            h = self.enc(h, mask=causal, src_key_padding_mask=~pad_mask)
            sit_logits = self.sit_head(h).squeeze(-1).masked_fill(~sit_end, -1e4)
            a = self.attn(h).squeeze(-1).masked_fill(~sit_end, -1e4)
            w = torch.softmax(a, dim=1)
            pooled = torch.bmm(w.unsqueeze(1), h).squeeze(1)
            part_logit = self.part_head(pooled).squeeze(-1)
            return sit_logits, part_logit, w

    return M()


def peer_contrastive(part_logit, pz, part):
    """同段内：相对同伴提交越少 → 风险应越高（成对 logistic，组 = part）"""
    torch = _TORCH
    loss, cnt = 0.0, 0
    for g in torch.unique(part):
        m = (part == g).nonzero(as_tuple=True)[0]
        if len(m) < 2:
            continue
        lg = part_logit[m]
        pzg = pz[m]
        d_pz = pzg[:, None] - pzg[None, :]
        d_lg = lg[:, None] - lg[None, :]
        mask = (d_pz < -0.5)
        if int(mask.sum()) == 0:
            continue
        loss = loss + torch.nn.functional.softplus(-d_lg[mask]).mean()
        cnt += 1
    if cnt == 0:
        return torch.zeros((), device=part_logit.device)
    return loss / cnt


def make_mlp(torch, nn, d_in, d=64):
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(d, d), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(d, 1))

        def forward(self, x, u, sit_end, pad_mask):
            # x: [B, L, D] → 取第一个有效 token 的上下文（token 部分已在外部置零）
            v = x[:, 0, :]
            return None, self.net(v).squeeze(-1), None
    return M()


def train_deep(X, U, SE, PM, y, pz, part, cfg, seed, dev):
    if cfg.get('kind') == 'mlp':
        torch, nn = _TORCH, _NN
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = make_mlp(torch, nn, X.shape[-1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
        n = X.shape[0]
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'])
        pw = torch.tensor(float(cfg.get('pos_weight', 1.0)), device=dev)
        for ep in range(cfg['epochs']):
            model.train()
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, cfg['bs']):
                idx = perm[s:s + cfg['bs']]
                _, pl, _ = model(X[idx], U[idx], SE[idx], PM[idx])
                yy = y[idx]
                loss = nn.functional.binary_cross_entropy_with_logits(pl, yy, pos_weight=pw)
                if cfg.get('peer', 0) > 0:
                    loss = loss + cfg['peer'] * peer_contrastive(
                        pl, pz[idx], part[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
        return model

    pw = float(cfg.get('pos_weight', 1.0))
    torch, nn = _TORCH, _NN
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model(torch, nn, X.shape[-1], cfg['film'], maxlen=X.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    n = X.shape[0]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['epochs'])
    for ep in range(cfg['epochs']):
        model.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, cfg['bs']):
            idx = perm[s:s + cfg['bs']]
            sit_logits, part_logit, _ = model(X[idx], U[idx], SE[idx], PM[idx])
            p = torch.sigmoid(sit_logits) * SE[idx]
            logS = torch.log(torch.clamp(1.0 - p, 1e-6, 1.0)).sum(1)
            yy = y[idx]
            loss_surv = -(yy * torch.log(torch.clamp(-logS, 1e-6, 1.0))
                          + (1 - yy) * torch.log(torch.clamp(logS, 1e-6, 1.0))).mean()
            loss = loss_surv + nn.functional.binary_cross_entropy_with_logits(
                part_logit, yy, pos_weight=torch.tensor(pw, device=dev))
            if cfg['peer'] > 0:
                loss = loss + cfg['peer'] * peer_contrastive(part_logit, pz[idx], part[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return model


def pad_batch(T_list, ctx_list, u_list, se_list, maxlen):
    torch = _TORCH
    d_tok = T_list[0].shape[1]
    d_ctx = ctx_list[0].shape[0]
    B = len(T_list)
    X = np.zeros((B, maxlen, d_tok + d_ctx), np.float32)
    U = np.zeros((B, maxlen, 1), np.float32)
    SE = np.zeros((B, maxlen), bool)
    PM = np.zeros((B, maxlen), bool)
    for i in range(B):
        T = T_list[i]
        L = min(len(T), maxlen)
        X[i, :L, :d_tok] = T[:L]
        X[i, :L, d_tok:] = ctx_list[i][None, :]
        U[i, :L, 0] = u_list[i][:L]
        SE[i, :L] = se_list[i][:L]
        PM[i, :L] = True
    return (torch.from_numpy(X), torch.from_numpy(U), torch.from_numpy(SE),
            torch.from_numpy(PM))


def deep_eval(ds, rows, RAWt, toks, SELF, folds, seeds, cfg, maxlen=64, tag='', dev='cuda'):
    """按折重算 peer 统计量 → 阶段级上下文 → 训练/预测"""
    torch = _TORCH
    part = ds['part_ids']
    lab = ds['drop']
    pos = {int(r): i for i, r in enumerate(rows)}
    keys = sorted(pos.keys())
    oof = np.zeros(len(keys))
    for tr, te in folds:
        tr_rows = np.array([r for r in tr if int(r) in pos])
        te_rows = np.array([r for r in te if int(r) in pos])
        if len(tr_rows) < 50 or len(te_rows) == 0:
            continue
        it = np.array([pos[int(r)] for r in tr_rows])
        ie = np.array([pos[int(r)] for r in te_rows])
        stats = peer_stats(RAWt, it, part[rows])
        ztr = group_peer_z(RAWt, it, part[rows], stats)
        zte = group_peer_z(RAWt, ie, part[rows], stats)
        Ctr = stage_context(ds, RAWt, it, rows[it], part[rows][it], ztr, SELF[tr_rows])
        Cte = stage_context(ds, RAWt, ie, rows[ie], part[rows][ie], zte, SELF[te_rows])
        pz_tr = torch.from_numpy(Ctr[:, CTX_OFF + SUB_DIM].astype(np.float32)).to(dev)
        part_tr = torch.from_numpy(part[tr_rows]).to(dev)

        Ttr = [toks[int(rows[i])][0] for i in it]
        SEtr = [toks[int(rows[i])][1] for i in it]
        Utr = [toks[int(rows[i])][2] for i in it]
        if cfg.get('kind') == 'mlp':
            _td = Ttr[0].shape[1]
            Ttr = [np.zeros((1, _td), np.float32) for _ in Ttr]
            Utr = [np.zeros(1, np.float32) for _ in Utr]
            SEtr = [np.ones(1, bool) for _ in SEtr]
        X, U, SE, PM = pad_batch(Ttr, [Ctr[i] for i in range(Ctr.shape[0])], Utr, SEtr, maxlen)
        X, U, SE, PM = [t.to(dev) for t in (X, U, SE, PM)]
        y = torch.from_numpy(lab[tr_rows].astype(np.float32)).to(dev)
        models = [train_deep(X, U, SE, PM, y, pz_tr, part_tr, cfg, sd, dev) for sd in seeds]

        Tte = [toks[int(rows[i])][0] for i in ie]
        SEte = [toks[int(rows[i])][1] for i in ie]
        Ute = [toks[int(rows[i])][2] for i in ie]
        if cfg.get('kind') == 'mlp':
            _td2 = Tte[0].shape[1]
            Tte = [np.zeros((1, _td2), np.float32) for _ in Tte]
            Ute = [np.zeros(1, np.float32) for _ in Ute]
            SEte = [np.ones(1, bool) for _ in SEte]
        Xt, Ut, SEt, PMt = pad_batch(Tte, [Cte[i] for i in range(Cte.shape[0])], Ute, SEte, maxlen)
        Xt, Ut, SEt, PMt = [t.to(dev) for t in (Xt, Ut, SEt, PMt)]
        preds = np.zeros(len(ie))
        with torch.no_grad():
            for m in models:
                m.eval()
                _, pl, _ = m(Xt, Ut, SEt, PMt)
                preds += torch.sigmoid(pl).cpu().numpy()
        oof[ie] = preds / len(seeds)
    return sc.metrics(lab[keys], oof, part[keys], tag), oof, keys


def topk_precision(y, oof, part, frac=0.05):
    """每段内取最危险 frac 的名单，计算精确率（与周报口径一致）"""
    tot_n, tot_hit, num, den = 0, 0, 0.0, 0
    for g in np.unique(part):
        m = np.where(part == g)[0]
        if len(m) < 10:
            continue
        k = max(int(round(len(m) * frac)), 1)
        sel = m[np.argsort(-oof[m])[:k]]
        tot_n += len(sel)
        tot_hit += int(y[sel].sum())
        num += y[sel].sum() / len(sel)
        den += 1
    return (tot_hit / tot_n if tot_n else float('nan')), (num / den if den else float('nan'))


def main(args):
    global _TORCH, _NN
    t0 = time.time()
    torch, nn = build_torch()
    _TORCH, _NN = torch, nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ds = sc.load_dataset(args.data)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part_ids'])
    SELF = sc.self_ref(ds)
    ex = np.load(args.ex, allow_pickle=True)
    raw_ex = ex['raw'].astype(np.float64)
    ex_sid, ex_par, ex_exr = ex['sid'], ex['part'], ex['exercise']
    dsp, dpart, ddrop = ds['sid'], ds['part_ids'], ds['drop']
    seeds = [42, 123, 777]
    folds = sc.group_folds(dsp, ds['final'], n_splits=5, seed=42)
    part_key = {(int(dsp[i]), int(dpart[i])): i for i in range(len(dsp))}
    sittings = build_sittings(raw_ex, ex_sid, ex_par, ex_exr)
    print(f'坐次总数 {sum(len(v) for v in sittings.values())} | (生,段) {len(sittings)} '
          f'| device={dev}', flush=True)
    os.makedirs(args.out, exist_ok=True)
    results = {}

    # ---------------- RF 基线 ----------------
    print('\n=== 0) RF 基线复现（同协议）===', flush=True)
    for k in [1, 2, 3]:
        rows, RAWt = trunc_agg(sittings, part_key, raw_ex, k)
        m, oof = rf_baseline(ds, RAWt, rows, SELF, folds, seeds, f'rf_sit{k}')
        p5, p5m = topk_precision(ddrop[rows], oof, dpart[rows], 0.05)
        print(f'  RF 坐下第{k}次  n={len(rows):<5} pooled={m["pooled"]:.4f} '
              f'auc={m["auc"]:.4f} judged={m["judged"]:.4f} top5%={p5:.3f}', flush=True)
        results[f'rf_sit{k}'] = dict(m, top5=p5, top5_perpart=p5m)
        np.savez_compressed(os.path.join(args.out, f'oof_rf_sit{k}.npz'),
                            oof=oof, y=ddrop[rows], part=dpart[rows])
    rows_full, RAWfull = trunc_agg(sittings, part_key, raw_ex, None)
    m, oof = rf_baseline(ds, RAWfull, rows_full, SELF, folds, seeds, 'rf_full')
    p5, p5m = topk_precision(ddrop[rows_full], oof, dpart[rows_full], 0.05)
    print(f'  RF 整段      n={len(rows_full):<5} pooled={m["pooled"]:.4f} '
          f'auc={m["auc"]:.4f} judged={m["judged"]:.4f} top5%={p5:.3f}', flush=True)
    results['rf_full'] = dict(m, top5=p5, top5_perpart=p5m)
    np.savez_compressed(os.path.join(args.out, 'oof_rf_full.npz'),
                        oof=oof, y=ddrop[rows_full], part=dpart[rows_full])

    # ---------------- token 构造 ----------------
    print('\n=== 1) token 序列 ===', flush=True)
    tok_full = {}
    for (s, p), sids in sittings.items():
        r = part_key.get((s, p))
        if r is None:
            continue
        idxs = sorted([i for sit in sids for i in sit], key=lambda i: ex_exr[i])
        T, is_end, u = token_features(raw_ex, idxs, ds['base_ex'][int(p)])
        tok_full[r] = (T, is_end, u)
    lens = np.array([len(v[0]) for v in tok_full.values()])
    print(f'  序列数 {len(tok_full)}  长度 中位 {int(np.median(lens))} '
          f'p90 {int(np.percentile(lens,90))} max {lens.max()}', flush=True)

    def token_subset(K):
        out = {}
        for r, (T, is_end, u) in tok_full.items():
            if K is None:
                out[r] = (T, is_end, u)
                continue
            ends = np.where(is_end)[0]
            if len(ends) < K:
                continue
            cut = ends[K - 1] + 1
            out[r] = (T[:cut], is_end[:cut], u[:cut])
        return out

    # ---------------- 深度模型 ----------------
    npos = float(ddrop.sum())
    pw = float((len(ddrop) - npos) / max(npos, 1))
    print(f'类别不平衡：正例 {int(npos)} / {len(ddrop)}  → pos_weight={pw:.2f}', flush=True)
    cfgs = {
        'base':      dict(film=False, peer=0.0, lr=args.lr, bs=256, epochs=args.epochs,
                           pos_weight=pw),
        'film':      dict(film=True, peer=0.0, lr=args.lr, bs=256, epochs=args.epochs,
                          pos_weight=pw),
        'peer':      dict(film=False, peer=args.lam, lr=args.lr, bs=256, epochs=args.epochs,
                          pos_weight=pw),
        'film_peer': dict(film=True, peer=args.lam, lr=args.lr, bs=256, epochs=args.epochs,
                          pos_weight=pw),
        'mlp_ctx': dict(kind='mlp', film=False, peer=0.0, lr=args.lr, bs=256,
                        epochs=args.epochs, pos_weight=pw),
    }
    for K in args.k_list:
        kk = None if K <= 0 else K
        tks = token_subset(kk)
        rows, RAWt = trunc_agg(sittings, part_key, raw_ex, kk)
        keep = set(int(r) for r in rows)
        tks = {r: v for r, v in tks.items() if r in keep}
        if not tks:
            continue
        print(f'\n=== 2) CS2-SPC · 观察前 {"全部坐次" if kk is None else str(kk)+" 次坐下"} '
              f'(n={len(rows)}) ===', flush=True)
        for cname in args.configs:
            cfg = cfgs[cname]
            tt = time.time()
            m, oof, keys = deep_eval(ds, rows, RAWt, tks, SELF, folds, seeds, cfg,
                                      maxlen=args.maxlen, tag=f'cs2_{cname}_K{K}', dev=dev)
            p5, p5m = topk_precision(ddrop[keys], oof, dpart[keys], 0.05)
            print(f'  {cname:<10} pooled={m["pooled"]:.4f} auc={m["auc"]:.4f} '
                  f'judged={m["judged"]:.4f} top5%={p5:.3f} [{time.time()-tt:.0f}s]', flush=True)
            results[f'cs2_{cname}_K{K}'] = dict(m, top5=p5, top5_perpart=p5m)
            np.savez_compressed(os.path.join(args.out, f'oof_cs2_{cname}_K{K}.npz'),
                                oof=oof, y=ddrop[keys], part=dpart[keys])

    with open(os.path.join(args.out, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\n完成，总耗时 {time.time()-t0:.0f}s → {args.out}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--ex', default='data/exercise_dataset.npz')
    ap.add_argument('--out', default='results/cs2_spc')
    ap.add_argument('--k_list', type=int, nargs='+', default=[1, 2, 0])
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--lam', type=float, default=0.3)
    ap.add_argument('--maxlen', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--configs', nargs='+',
                    default=['base', 'film', 'peer', 'film_peer', 'mlp_ctx'])
    args = ap.parse_args()
    main(args)
