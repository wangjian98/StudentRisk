"""
实验 H（重写）：PCDH-CPU 原型 + 泛化套件
======================================
修正上一版两个 bug：
 1) 同伴归一化必须**按 part 块分别**计算：49 维向量里第 p 块（part p 的 7 维事件计数）
    要用"主 part = p 的学生"的第 p 块统计量来归一化，不是整条 49 维共用一个统计量
 2) fold 先验要**逐折累加**，不能整数组覆盖
另修：自参照特征维度 10、lead-time 块重写

输出：
 H1 复现 V0/V1/V3（验证 pipeline 可信）
 H2 V4 = V3 + 同伴背景标量 + 自参照锚定 + 进度特征
 H3 阶段级模型：预测"在此 part 之后不再推进"（硬观测）
 H4 泛化：LOPO / 时间外推 / lead-time 提前预警
 H5 置换检验（打乱标签 / 打乱同伴分组）
"""
import json
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

D = np.load('results/F_stage_probe/stage_matrix.npz', allow_pickle=True)
FEAT, SID, PID, FIRST, LAST = D['feat'], D['sid'], D['pid'], D['first'], D['last']
Y_STU, STUDENTS = D['y'], D['students']
EVN, PARTS = 7, list(range(1, 8))

sidx = {int(s): i for i, s in enumerate(STUDENTS)}
per_stu = defaultdict(dict)
for j in range(len(SID)):
    per_stu[int(SID[j])][int(PID[j])] = (FEAT[j].astype(float), FIRST[j], LAST[j])

info, STU = {}, sorted(per_stu)
for s in STU:
    parts = per_stu[s]
    x49 = np.zeros(49)
    for p, (x, f, l) in parts.items():
        x49[(p - 1) * EVN:(p - 1) * EVN + EVN] = x[:EVN]
    info[s] = dict(main_part=max(parts, key=lambda p: parts[p][0][7]),
                   max_part=max(parts), x49=x49,
                   first_ts=min(v[1] for v in parts.values()),
                   last_ts=max(v[2] for v in parts.values()),
                   label=int(Y_STU[sidx[s]]))
n_stu = len(STU)
y = np.array([info[s]['label'] for s in STU])
main_parts = np.array([info[s]['main_part'] for s in STU])
max_parts = np.array([info[s]['max_part'] for s in STU])
X49 = np.array([info[s]['x49'] for s in STU])
s2row = {s: i for i, s in enumerate(STU)}

# 阶段样本
st_s, st_p, st_x, st_y, st_i = [], [], [], [], []
for s in STU:
    parts = per_stu[s]
    ks = sorted(parts)
    for j, k in enumerate(ks):
        st_s.append(s); st_p.append(k); st_x.append(parts[k][0])
        st_y.append(1 if (k == max(ks) and max(ks) < 7) else 0)
        st_i.append(j)
st_x = np.array(st_x); st_y = np.array(st_y); st_p = np.array(st_p); st_i = np.array(st_i)
st_row = np.array([s2row[s] for s in st_s])
print(f'学生 {n_stu} / failed {y.sum()} / 阶段样本 {len(st_y)} / 掉队正例 {st_y.sum()} ({st_y.mean():.1%})')
print(f'未达 part7 学生数 = {(max_parts < 7).sum()}，其中挂科 {y[max_parts < 7].sum()} '
      f'({y[max_parts<7].mean():.1%})；达 part7 {n_stu-(max_parts<7).sum()} 人，其中挂科 {y[max_parts==7].sum()}')

rng = np.random.default_rng(0)
fold_id = np.zeros(n_stu, int)
for i, r in enumerate(rng.permutation(n_stu)):
    fold_id[r] = i % 5


# ---------------------------------------------------------------- 工具
def peer_blocks_49(main_p, X49, train_rows, target_rows):
    """按 part 块分别做 leave-one-out 同伴归一化（M2 的忠实实现）"""
    out = np.zeros((len(target_rows), 49))
    bg = np.zeros((len(target_rows), 3))
    for p in PARTS:
        tr_p = np.where(train_rows & (main_p == p))[0]
        if len(tr_p) == 0:
            continue
        blk = X49[tr_p][:, (p - 1) * EVN:p * EVN]
        med = np.median(blk, 0)
        mad = np.median(np.abs(blk - med), 0) * 1.4826 + 1e-9
        tot_tr = blk.sum(1)
        med_tot = float(np.median(tot_tr)); mad_tot = float(np.median(np.abs(tot_tr - med_tot)) * 1.4826 + 1e-9)
        pos_in_tr = {r: i for i, r in enumerate(tr_p)}
        for i, r in enumerate(target_rows):
            if main_p[r] != p:
                continue
            b = X49[r][(p - 1) * EVN:p * EVN]
            if train_rows[r] and len(tr_p) > 1:      # leave-one-out
                k = pos_in_tr[r]
                b2 = np.delete(blk, k, axis=0)
                m2 = np.median(b2, 0); d2 = np.median(np.abs(b2 - m2), 0) * 1.4826 + 1e-9
                out[i][(p - 1) * EVN:p * EVN] = (b - m2) / d2
            else:
                out[i][(p - 1) * EVN:p * EVN] = (b - med) / mad
            bg[i] = [np.log1p(mad_tot), np.log1p(len(tr_p)), np.log1p(med_tot)]
    return out, bg


def peer_stage(st_p, st_x, train_rows, target_idx):
    """阶段级同伴归一化：同伴 = 主 part == 该 part 的学生，在该 part 的行为"""
    out = np.zeros((len(target_idx), st_x.shape[1]))
    bg = np.zeros((len(target_idx), 3))
    main_of_stage = np.array([main_parts[s2row[s]] for s in st_s])
    for p in PARTS:
        tr_p = np.where(train_rows & (main_of_stage == p))[0]
        if len(tr_p) == 0:
            continue
        Xp = st_x[tr_p]
        med = np.median(Xp, 0); mad = np.median(np.abs(Xp - med), 0) * 1.4826 + 1e-9
        pos = {r: i for i, r in enumerate(tr_p)}
        for i, j in enumerate(target_idx):
            if st_p[j] != p:
                continue
            b = st_x[j]
            if train_rows[j] and len(tr_p) > 1:
                k = pos[j]
                X2 = np.delete(Xp, k, axis=0)
                m2 = np.median(X2, 0); d2 = np.median(np.abs(X2 - m2), 0) * 1.4826 + 1e-9
                out[i] = (b - m2) / d2
            else:
                out[i] = (b - med) / mad
            bg[i] = [np.log1p(mad[7]), np.log1p(len(tr_p)), np.log1p(med[7])]
    return out, bg


def self_anchor(X, rows):
    """自参照锚定：相对该生自己此前阶段的偏离"""
    z = np.zeros_like(X)
    by = defaultdict(list)
    for j, (s, i) in enumerate(zip(st_s, st_i)):
        by[s].append(j)
    for s, js in by.items():
        js = sorted(js, key=lambda j: st_i[j])
        for pos, j in enumerate(js):
            if pos:
                h = X[js[:pos]]
                z[j] = (X[j] - h.mean(0)) / (h.std(0) + 1e-9)
    return z


def prior_fold(main_p, y, tr_mask, te_mask, out):
    for p in np.unique(main_p):
        m_tr = tr_mask & (main_p == p); m_te = te_mask & (main_p == p)
        out[m_te] = y[m_tr].mean() if m_tr.sum() else y[tr_mask].mean()
    return out


def grp_bal(p, groups, yv):
    a, w = [], []
    for g in np.unique(groups):
        m = groups == g
        if len(set(yv[m])) > 1:
            a.append(roc_auc_score(yv[m], p[m])); w.append(m.sum())
    return float(np.average(a, weights=w)) if a else float('nan')


def rf_fit(X, yv, seed=0, n=300):
    rf = RandomForestClassifier(n_estimators=n, min_samples_leaf=2,
                                class_weight='balanced', random_state=seed, n_jobs=-1)
    rf.fit(X, yv)
    return rf


def cv_oof(build, yv, groups, n_splits=5):
    oof = np.zeros(len(yv))
    for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0).split(
            np.zeros((len(yv), 1)), yv, groups):
        Xtr, Xte = build(tr, te)
        oof[te] = rf_fit(Xtr, yv[tr]).predict_proba(Xte)[:, 1]
    return oof


def self_prog_feats():
    sf = np.zeros((n_stu, 10)); pf = np.zeros((n_stu, 4))
    for i, s in enumerate(STU):
        ks = sorted(per_stu[s])
        Xs = np.array([per_stu[s][k][0] for k in ks])
        if len(Xs) > 1:
            sf[i] = (Xs[-1] - Xs[:-1].mean(0)) / (Xs[:-1].std(0) + 1e-9)
        pf[i] = [len(ks), max(ks), max(ks) - min(ks), Xs[:, 7].mean()]
    return sf, pf


SF, PF = self_prog_feats()
R = {}

print('\n' + '=' * 92)
print('H1) 复现既有基线（pipeline 可信性检验）')
print('=' * 92)
prior_oof = np.zeros(n_stu)
for f in range(5):
    tr = fold_id != f; te = fold_id == f
    prior_oof = prior_fold(main_parts, y, tr, te, prior_oof)
print(f'  part-prior baseline (5折无泄漏)      AUC={roc_auc_score(y, prior_oof):.4f}')

for name, mode in [('V0_raw_49d', 'raw'), ('V1_deconf_49d', 'dec'), ('V3_deconf+part1h', 'v3')]:
    oof = np.zeros(n_stu)
    for f in range(5):
        tr = fold_id != f; te = fold_id == f
        if mode == 'raw':
            Xtr, Xte = X49[tr], X49[te]
        else:
            zr, bgr = peer_blocks_49(main_parts, X49, tr, np.where(tr)[0])
            zt, bgt = peer_blocks_49(main_parts, X49, tr, np.where(te)[0])
            Xtr, Xte = np.hstack([zr, bgr]), np.hstack([zt, bgt])
            if mode == 'v3':
                Xtr = np.hstack([Xtr, np.eye(8)[main_parts[tr]]])
                Xte = np.hstack([Xte, np.eye(8)[main_parts[te]]])
        oof[te] = rf_fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
    a, g = roc_auc_score(y, oof), grp_bal(oof, main_parts, y)
    R[name] = dict(auc=float(a), grp_bal=float(g))
    print(f'  {name:22s} AUC={a:.4f}  grp_bal={g:.4f}')
print('  历史参照：V0 0.9503/0.7988 · V1 0.9439/0.8403 · V3 0.9533/0.8420')

print('\n' + '=' * 92)
print('H2) V4 = V3 + 同伴背景标量 + 自参照锚定 + 进度特征')
print('=' * 92)
oof_v4 = np.zeros(n_stu)
for f in range(5):
    tr = fold_id != f; te = fold_id == f
    zr, bgr = peer_blocks_49(main_parts, X49, tr, np.where(tr)[0])
    zt, bgt = peer_blocks_49(main_parts, X49, tr, np.where(te)[0])
    Xtr = np.hstack([zr, bgr, np.eye(8)[main_parts[tr]], SF[tr], PF[tr]])
    Xte = np.hstack([zt, bgt, np.eye(8)[main_parts[te]], SF[te], PF[te]])
    oof_v4[te] = rf_fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
a4, g4 = roc_auc_score(y, oof_v4), grp_bal(oof_v4, main_parts, y)
print(f'  V4  AUC={a4:.4f}  grp_bal={g4:.4f}   （vs V3 grp_bal 0.8420 / 论文主模型 0.7234）')
R['V4'] = dict(auc=float(a4), grp_bal=float(g4))

print('\n' + '=' * 92)
print('H3) 阶段级模型：预测"在此 part 之后不再推进"')
print('=' * 92)
tr_all = np.ones(len(st_y), bool)
zs, bgs = peer_stage(st_p, st_x, tr_all, np.arange(len(st_y)))
zself = self_anchor(st_x, np.arange(len(st_y)))
Xs_stage = np.hstack([zs, bgs, zself, st_i.reshape(-1, 1).astype(float)])
oof_stage = np.zeros(len(st_y))
for f in range(5):
    tr = st_row[fold_id[st_row]] != f; te = ~tr
    oof_stage[te] = rf_fit(Xs_stage[tr], st_y[tr]).predict_proba(Xs_stage[te])[:, 1]
pa, pw = [], []
per_p = []
for p in PARTS:
    m = st_p == p
    if m.sum() >= 10 and len(set(st_y[m])) > 1:
        a = roc_auc_score(st_y[m], oof_stage[m]); pa.append(a); pw.append(m.sum())
        per_p.append((int(p), int(m.sum()), int(st_y[m].sum()), float(a)))
        print(f'    part{p}: n={m.sum():4d} 掉队率={st_y[m].mean():.3f}  AUC={a:.4f}')
pooled = float(np.average(pa, weights=pw))
print(f'  阶段级 pooled within-part AUC = {pooled:.4f}  （0.5=无个体信号）')
print(f'  阶段级 总体 AUC               = {roc_auc_score(st_y, oof_stage):.4f}')
R['stage'] = dict(pooled=pooled, per_part=per_p)

print('\n' + '=' * 92)
print('H4) 泛化套件')
print('=' * 92)


def lopo(build, tag):
    per = []
    for p in PARTS:
        te = main_parts == p; tr = ~te
        if te.sum() < 10 or len(set(y[te])) < 2:
            continue
        Xtr, Xte = build(tr, te)
        pr = rf_fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        per.append((p, int(te.sum()), float(roc_auc_score(y[te], pr))))
    w = np.array([x[1] for x in per], float); w /= w.sum()
    m = float(np.average([x[2] for x in per], weights=w))
    print(f'  {tag:22s} 加权 LOPO AUC = {m:.4f}  ' + ' '.join(f'p{p}={a:.3f}' for p, _, a in per))
    return m, per


def b_raw(tr, te):
    return X49[tr], X49[te]


def b_dec(tr, te):
    zr, bgr = peer_blocks_49(main_parts, X49, tr, np.where(tr)[0])
    zt, bgt = peer_blocks_49(main_parts, X49, tr, np.where(te)[0])
    return np.hstack([zr, bgr]), np.hstack([zt, bgt])


def b_full(tr, te):
    zr, bgr = peer_blocks_49(main_parts, X49, tr, np.where(tr)[0])
    zt, bgt = peer_blocks_49(main_parts, X49, tr, np.where(te)[0])
    return (np.hstack([zr, bgr, np.eye(8)[main_parts[tr]], SF[tr], PF[tr]]),
            np.hstack([zt, bgt, np.eye(8)[main_parts[te]], SF[te], PF[te]]))


m_raw, p_raw = lopo(b_raw, 'LOPO raw 49d')
m_dec, p_dec = lopo(b_dec, 'LOPO deconf')
m_full, p_full = lopo(b_full, 'LOPO full V4')
print('  历史参照：MetaMamba-7d LOPO 加权 0.6134（part7 ROC 0.221）')
R['lopo'] = dict(raw=m_raw, deconf=m_dec, full=m_full)

order = np.argsort([info[s]['first_ts'] for s in STU])
cut = int(0.6 * n_stu)
tr_idx, te_idx = order[:cut], order[cut:]
print(f'\n  时间外推：前 {len(tr_idx)} 人训练 → 后 {len(te_idx)} 人测试')
mt = np.zeros(n_stu, bool); mt[tr_idx] = True
me = np.zeros(n_stu, bool); me[te_idx] = True
for nm, bld in [('raw 49d', b_raw), ('deconf', b_dec), ('full V4', b_full)]:
    Xtr, Xte = bld(mt, me)
    pr = rf_fit(Xtr, y[mt]).predict_proba(Xte)[:, 1]
    pri = prior_fold(main_parts, y, mt, me, np.zeros(n_stu))
    yte = y[me]
    print(f'    {nm:10s} 测试 AUC={roc_auc_score(yte,pr):.4f}  grp_bal={grp_bal(pr,main_parts[me],yte):.4f}'
          f'  (同口径 prior={roc_auc_score(yte,pri[me]):.4f})')
    R[f'timesplit_{nm}'] = dict(auc=float(roc_auc_score(yte, pr)))

print('\n  lead-time（仅用前 k 个 stage 预测最终挂科）')
lead = []
for kmax in [1, 2, 3]:
    rows = sorted({s for s in STU if st_i[st_s.index(s)] < kmax} if False else
                  {s for s in STU if any(st_s[j] == s and st_i[j] < kmax for j in range(len(st_s)))})
    idx_map = {s: i for i, s in enumerate(rows)}
    yy = np.array([info[s]['label'] for s in rows])
    Zk = np.zeros((len(rows), 10)); Bk = np.zeros((len(rows), 3))
    for j in range(len(st_y)):
        if st_i[j] >= kmax:
            continue
        r = idx_map[st_s[j]]
        Zk[r] += zs[j]; Bk[r] += bgs[j]
    cnt = defaultdict(int)
    for j in range(len(st_y)):
        if st_i[j] < kmax:
            cnt[st_s[j]] += 1
    for s, c in cnt.items():
        Zk[idx_map[s]] /= c; Bk[idx_map[s]] /= c
    Zk = np.hstack([Zk, Bk])
    oof = np.zeros(len(yy))
    for f in range(5):
        tr = fold_id[[s2row[s] for s in rows]] != f; te = ~tr
        oof[te] = rf_fit(Zk[tr], yy[tr]).predict_proba(Zk[te])[:, 1]
    a = roc_auc_score(yy, oof)
    lead.append((kmax, float(a), len(rows)))
    print(f'    前 {kmax} 个 stage：n={len(rows):3d}  AUC={a:.4f}  覆盖率 {len(rows)/n_stu:.0%}')
R['lead'] = lead

print('\n' + '=' * 92)
print('H5) 置换检验')
print('=' * 92)
yp = rng.permutation(y)
oof_p = cv_oof(b_raw, yp, main_parts)
print(f'  打乱标签 → AUC={roc_auc_score(yp,oof_p):.4f}（应≈0.5）')
rng2 = np.random.default_rng(7)
fake = rng2.permutation(main_parts)
oof_f = np.zeros(n_stu)
for f in range(5):
    tr = fold_id != f; te = fold_id == f
    zr, bgr = peer_blocks_49(fake, X49, tr, np.where(tr)[0])
    zt, bgt = peer_blocks_49(fake, X49, tr, np.where(te)[0])
    oof_f[te] = rf_fit(np.hstack([zr, bgr]), y[tr]).predict_proba(np.hstack([zt, bgt]))[:, 1]
print(f'  打乱同伴分组 → grp_bal={grp_bal(oof_f,main_parts,y):.4f}（真实分组 {g4:.4f}）')

json.dump(R, open('results/F_stage_probe/experiment_H_results.json', 'w'),
          indent=2, ensure_ascii=False, default=float)
print('\n  → results/F_stage_probe/experiment_H_results.json')
