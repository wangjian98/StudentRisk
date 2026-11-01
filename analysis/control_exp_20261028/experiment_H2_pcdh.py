"""
实验 H2（定稿）：PCDH-CPU 原型 + 泛化套件 + 稳健指标
=====================================================
对齐历史协议（可直接与 V0 0.9503/0.7988、V1 0.9439/0.8403、V3 0.9533/0.8420 对比）：
  · 折：StratifiedKFold(5, shuffle=True, random_state=42)；RF(300, min_leaf=2, sqrt, balanced)；3 seed 平均
  · 去混淆：整条 49 维行，按"主 part 同伴组"的中位数/MAD 归一化（M2 的忠实实现）
  · 本版比历史更严：训练行用 leave-one-out 同伴统计（消除自归一化偏置），统计量只在训练折内算

新增稳健指标（关键）：
  · per-part AUC 全量（含退化 part）
  · judgeable = 正负各 ≥10 的 part 上的平均 AUC（part4 / part7）
  · legacy weighted-all（与历史口径一致，用于对照）
  · 中位数（抗退化 part）
"""
import json
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

D = np.load('results/F_stage_probe/stage_matrix.npz', allow_pickle=True)
FEAT, SID, PID, FIRST, LAST = D['feat'], D['sid'], D['pid'], D['first'], D['last']
Y_STU, STUDENTS = D['y'], D['students']
EVN, PARTS, SEEDS = 7, list(range(1, 8)), [42, 123, 777]

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
n = len(STU)
y = np.array([info[s]['label'] for s in STU])
main_parts = np.array([info[s]['main_part'] for s in STU])
X49 = np.array([info[s]['x49'] for s in STU])
s2row = {s: i for i, s in enumerate(STU)}

st_s, st_p, st_x, st_y, st_i = [], [], [], [], []
for s in STU:
    parts = per_stu[s]; ks = sorted(parts)
    for j, k in enumerate(ks):
        st_s.append(s); st_p.append(k); st_x.append(parts[k][0])
        st_y.append(1 if (k == max(ks) and max(ks) < 7) else 0); st_i.append(j)
st_x = np.array(st_x); st_y = np.array(st_y); st_p = np.array(st_p); st_i = np.array(st_i)
st_row = np.array([s2row[s] for s in st_s])

print(f'学生 {n} / failed {y.sum()} / 阶段样本 {len(st_y)} / 掉队正例 {st_y.sum()}')
print('各主 part 的标签分布（注意退化 part）：')
for p in PARTS:
    m = main_parts == p
    print(f'   part{p}: n={m.sum():3d}  负样本(通过)={int((y[m]==0).sum()):3d}  正样本(挂科)={int(y[m].sum()):3d}')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(X49, y))
rng = np.random.default_rng(7)


def deconf_rows(main_p, X, tr_rows, target_rows, loo=True):
    """整行归一化：按主 part 同伴组的中位数/MAD（历史 M2 口径）"""
    out = np.zeros((len(target_rows), X.shape[1]))
    bg = np.zeros((len(target_rows), 3))
    for p in np.unique(main_p):
        grp = np.where(main_p == p)[0]
        tr_grp = np.intersect1d(grp, tr_rows)
        if len(tr_grp) == 0:
            continue
        Xg = X[tr_grp]
        med = np.median(Xg, 0); mad = np.maximum(np.median(np.abs(Xg - med), 0), 1e-3)
        pos = {r: i for i, r in enumerate(tr_grp)}
        for i, r in enumerate(target_rows):
            if main_p[r] != p:
                continue
            if loo and r in pos and len(tr_grp) > 1:
                X2 = np.delete(Xg, pos[r], axis=0)
                m2 = np.median(X2, 0); d2 = np.maximum(np.median(np.abs(X2 - m2), 0), 1e-3)
                out[i] = (X[r] - m2) / d2
            else:
                out[i] = (X[r] - med) / mad
            bg[i] = [np.log1p(mad.sum()), np.log1p(len(tr_grp)), np.log1p(med.sum())]
    return np.clip(out, -5, 5), bg


def rf_oof(X, yv, folds):
    oof = np.zeros(len(yv))
    for seed in SEEDS:
        acc = np.zeros(len(yv))
        for tr, te in folds:
            m = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                       max_features='sqrt', class_weight='balanced',
                                       random_state=seed, n_jobs=-1)
            m.fit(X[tr], yv[tr])
            acc[te] += m.predict_proba(X[te])[:, 1]
        oof += acc / len(SEEDS)
    return oof / len(SEEDS)


def prior_oof(main_p, yv, folds):
    pr = np.zeros(len(yv))
    for tr, te in folds:
        for p in np.unique(main_p):
            mtr = tr[main_p[tr] == p]; mte = te[main_p[te] == p]
            if len(mtr):
                pr[mte] = yv[mtr].mean()
            elif len(mte):
                pr[mte] = yv[tr].mean()
    return pr


def metrics(oof, groups, yv, tag):
    per = {}
    for p in np.unique(groups):
        m = groups == p
        if len(set(yv[m])) > 1:
            per[int(p)] = dict(n=int(m.sum()), neg=int((yv[m] == 0).sum()),
                               auc=float(roc_auc_score(yv[m], oof[m])))
    judged = [v['auc'] for v in per.values() if v['neg'] >= 10 and v['n'] - v['neg'] >= 10]
    w = np.array([v['n'] for v in per.values()], float)
    legacy = float(np.average(list(per.values()), weights=w,
                              ))if False else None
    aucs = np.array([v['auc'] for v in per.values()]); ns = np.array([v['n'] for v in per.values()])
    legacy = float(np.average(aucs, weights=ns))
    med = float(np.median(aucs))
    j = float(np.mean(judged)) if judged else float('nan')
    print(f'  {tag:26s} AUC={roc_auc_score(yv,oof):.4f} | judgeable(neg>=10)={j:.4f} '
          f'| legacy加权={legacy:.4f} | 中位={med:.4f}')
    for p, v in per.items():
        print(f'      part{p}: n={v["n"]:3d} neg={v["neg"]:3d}  AUC={v["auc"]:.4f}')
    return dict(auc=float(roc_auc_score(yv, oof)), judgeable=j, legacy=legacy, median=med, per=per)


R = {}
print('\n' + '=' * 96)
print('H1) 复现历史基线（协议对齐：same folds / same RF / same deconf 口径）')
print('=' * 96)
prior = prior_oof(main_parts, y, folds)
print(f'  part-prior baseline  AUC={roc_auc_score(y, prior):.4f}   （历史 0.9135）')

sets = {}
for name in ['V0_raw_49d', 'V1_deconf_49d', 'V3_deconf+part1h']:
    oof = np.zeros(n)
    for tr, te in folds:
        if name == 'V0_raw_49d':
            Xtr, Xte = X49[tr], X49[te]
        else:
            zr, _ = deconf_rows(main_parts, X49, tr, tr)
            zt, _ = deconf_rows(main_parts, X49, tr, te)
            Xtr, Xte = zr, zt
            if name == 'V3_deconf+part1h':
                oh = np.eye(8)
                Xtr = np.hstack([Xtr, oh[main_parts[tr]]]); Xte = np.hstack([Xte, oh[main_parts[te]]])
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                                   class_weight='balanced', random_state=42, n_jobs=-1)
        m.fit(Xtr, y[tr]); oof[te] = m.predict_proba(Xte)[:, 1]
    sets[name] = oof
    R[name] = metrics(oof, main_parts, y, name)
print('  历史参照：V0 0.9503/legacy0.7988 · V1 0.9439/0.8403 · V3 0.9533/0.8420（legacy 口径）')

print('\n' + '=' * 96)
print('H2) V4 = V3 + 同伴背景标量 + 自参照锚定 + 进度特征')
print('=' * 96)
SF = np.zeros((n, 10)); PF = np.zeros((n, 4))
for i, s in enumerate(STU):
    ks = sorted(per_stu[s]); Xs = np.array([per_stu[s][k][0] for k in ks])
    if len(Xs) > 1:
        SF[i] = (Xs[-1] - Xs[:-1].mean(0)) / (Xs[:-1].std(0) + 1e-9)
    PF[i] = [len(ks), max(ks), max(ks) - min(ks), Xs[:, 7].mean()]
oof_v4 = np.zeros(n)
for tr, te in folds:
    zr, bgr = deconf_rows(main_parts, X49, tr, tr)
    zt, bgt = deconf_rows(main_parts, X49, tr, te)
    oh = np.eye(8)
    Xtr = np.hstack([zr, bgr, oh[main_parts[tr]], SF[tr], PF[tr]])
    Xte = np.hstack([zt, bgt, oh[main_parts[te]], SF[te], PF[te]])
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                               class_weight='balanced', random_state=42, n_jobs=-1)
    m.fit(Xtr, y[tr]); oof_v4[te] = m.predict_proba(Xte)[:, 1]
R['V4'] = metrics(oof_v4, main_parts, y, 'V4(V3+自参照+进度)')
R['V4']['auc_delta_vs_V3'] = float(roc_auc_score(y, oof_v4) - roc_auc_score(y, sets['V3_deconf+part1h']))

print('\n' + '=' * 96)
print('H3) 阶段级模型：预测"在此 part 之后不再推进"（硬观测标签）')
print('=' * 96)
# 阶段级特征：主 part 组同伴归一化（用学生的 main part 作分组；行=阶段样本）
main_of_stage = np.array([main_parts[s2row[s]] for s in st_s])
zs = np.zeros((len(st_y), st_x.shape[1])); bgs = np.zeros((len(st_y), 3))
for p in np.unique(main_of_stage):
    grp = np.where(main_of_stage == p)[0]
    Xg = st_x[grp]
    med = np.median(Xg, 0); mad = np.maximum(np.median(np.abs(Xg - med), 0), 1e-3)
    for j in grp:
        zs[j] = (st_x[j] - med) / mad
        bgs[j] = [np.log1p(mad[7]), np.log1p(len(grp)), np.log1p(med[7])]
zs = np.clip(zs, -5, 5)
# 自参照
by = defaultdict(list)
for j, (s, i) in enumerate(zip(st_s, st_i)):
    by[s].append(j)
zself = np.zeros_like(zs)
for s, js in by.items():
    js = sorted(js, key=lambda j: st_i[j])
    for pos, j in enumerate(js):
        if pos:
            h = zs[js[:pos]]
            zself[j] = (zs[j] - h.mean(0)) / (h.std(0) + 1e-9)
Xs_stage = np.hstack([zs, bgs, zself, st_i.reshape(-1, 1).astype(float)])
oof_stage = np.zeros(len(st_y))
stage_folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xs_stage, st_y))
for tr, te in stage_folds:
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                               class_weight='balanced', random_state=42, n_jobs=-1)
    m.fit(Xs_stage[tr], st_y[tr]); oof_stage[te] = m.predict_proba(Xs_stage[te])[:, 1]
pa, pw, pdet = [], [], []
for p in PARTS:
    m = st_p == p
    if m.sum() >= 10 and len(set(st_y[m])) > 1:
        a = roc_auc_score(st_y[m], oof_stage[m]); pa.append(a); pw.append(m.sum())
        pdet.append((int(p), int(m.sum()), int(st_y[m].sum()), float(a)))
        print(f'    part{p}: n={m.sum():4d} 掉队率={st_y[m].mean():.3f}  AUC={a:.4f}')
pooled = float(np.average(pa, weights=pw))
print(f'  阶段级 pooled within-part AUC = {pooled:.4f}   总体 AUC = {roc_auc_score(st_y,oof_stage):.4f}')
R['stage'] = dict(pooled=pooled, per_part=pdet)

print('\n' + '=' * 96)
print('H4) 泛化：LOPO / 时间外推 / lead-time')
print('=' * 96)


def lopo(build, tag):
    per = []
    for p in PARTS:
        te = np.where(main_parts == p)[0]; tr = np.setdiff1d(np.arange(n), te)
        if len(te) < 10 or len(set(y[te])) < 2:
            continue
        Xtr, Xte = build(tr, te)
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                                   class_weight='balanced', random_state=42, n_jobs=-1)
        m.fit(Xtr, y[tr]); pr = m.predict_proba(Xte)[:, 1]
        per.append((p, len(te), float(roc_auc_score(y[te], pr))))
    ns = np.array([x[1] for x in per], float); ns /= ns.sum()
    mm = float(np.average([x[2] for x in per], weights=ns))
    print(f'  {tag:20s} 加权 LOPO AUC = {mm:.4f}  ' + ' '.join(f'p{p}={a:.3f}' for p, _, a in per))
    return mm, per


def b_raw(tr, te):
    return X49[tr], X49[te]


def b_v3(tr, te):
    zr, _ = deconf_rows(main_parts, X49, tr, tr); zt, _ = deconf_rows(main_parts, X49, tr, te)
    oh = np.eye(8)
    return np.hstack([zr, oh[main_parts[tr]]]), np.hstack([zt, oh[main_parts[te]]])


def b_v4(tr, te):
    zr, bgr = deconf_rows(main_parts, X49, tr, tr); zt, bgt = deconf_rows(main_parts, X49, tr, te)
    oh = np.eye(8)
    return (np.hstack([zr, bgr, oh[main_parts[tr]], SF[tr], PF[tr]]),
            np.hstack([zt, bgt, oh[main_parts[te]], SF[te], PF[te]]))


m_raw, _ = lopo(b_raw, 'LOPO raw')
m_v3, _ = lopo(b_v3, 'LOPO V3')
m_v4, _ = lopo(b_v4, 'LOPO V4')
print('  历史参照：MetaMamba-7d LOPO 加权 0.6134（part7 ROC 0.221）')
R['lopo'] = dict(raw=m_raw, v3=m_v3, v4=m_v4)

order = np.argsort([info[s]['first_ts'] for s in STU])
cut = int(0.6 * n)
tr_idx, te_idx = order[:cut], order[cut:]
mt = np.zeros(n, bool); mt[tr_idx] = True
me = np.zeros(n, bool); me[te_idx] = True
print(f'\n  时间外推：前 {len(tr_idx)} 人训练 → 后 {len(te_idx)} 人测试')
for nm, bld in [('raw', b_raw), ('V3', b_v3), ('V4', b_v4)]:
    Xtr, Xte = bld(np.where(mt)[0], np.where(me)[0])
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                               class_weight='balanced', random_state=42, n_jobs=-1)
    m.fit(Xtr, y[mt]); pr = m.predict_proba(Xte)[:, 1]
    pri = prior_oof(main_parts, y, [(np.where(mt)[0], np.where(me)[0])])[me]
    yte = y[me]
    jud = []
    for p in np.unique(main_parts[me]):
        mm = (main_parts[me] == p)
        if len(set(yte[mm])) > 1 and (yte[mm] == 0).sum() >= 10:
            jud.append(roc_auc_score(yte[mm], pr[mm]))
    print(f'    {nm:4s} 测试 AUC={roc_auc_score(yte,pr):.4f}  judgeable={np.mean(jud) if jud else float("nan"):.4f}'
          f'  (同口径 prior={roc_auc_score(yte,pri):.4f})')
    R[f'timesplit_{nm}'] = dict(auc=float(roc_auc_score(yte, pr)))

print('\n  lead-time（仅用前 k 个 stage 的行为预测最终挂科）')
lead = []
for kmax in [1, 2, 3]:
    Xk = np.zeros((n, 49))
    for j in range(len(st_y)):
        if st_i[j] < kmax:
            Xk[st_row[j], (st_p[j] - 1) * EVN:(st_p[j] - 1) * EVN + EVN] += st_x[j][:EVN]
    used = np.array([any(st_i[j] < kmax and st_s[j] == STU[i] for j in range(len(st_y)))
                     for i in range(n)])
    oof = np.zeros(n)
    for tr, te in folds:
        zr, _ = deconf_rows(main_parts, Xk, tr, tr); zt, _ = deconf_rows(main_parts, Xk, tr, te)
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                                   class_weight='balanced', random_state=42, n_jobs=-1)
        m.fit(zr[tr], y[tr]); oof[te] = m.predict_proba(zt[te])[:, 1]
    yy = y[used]; oo = oof[used]
    a = roc_auc_score(yy, oo)
    lead.append((kmax, float(a), int(used.sum())))
    print(f'    前 {kmax} 个 stage：n={used.sum():3d}  AUC={a:.4f}  覆盖率 {used.sum()/n:.0%}')
R['lead'] = lead

print('\n' + '=' * 96)
print('H5) 置换检验')
print('=' * 96)
yp = rng.permutation(y)
oof_p = np.zeros(n)
for tr, te in folds:
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                               class_weight='balanced', random_state=42, n_jobs=-1)
    m.fit(X49[tr], yp[tr]); oof_p[te] = m.predict_proba(X49[te])[:, 1]
print(f'  打乱标签 → AUC={roc_auc_score(yp,oof_p):.4f}（应≈0.5）')
fake = rng.permutation(main_parts)
oof_f = np.zeros(n)
for tr, te in folds:
    zr, _ = deconf_rows(fake, X49, tr, tr); zt, _ = deconf_rows(fake, X49, tr, te)
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                               class_weight='balanced', random_state=42, n_jobs=-1)
    m.fit(zr[tr], y[tr]); oof_f[te] = m.predict_proba(zt[te])[:, 1]
print(f'  打乱同伴分组 → judgeable AUC = '
      f'{metrics(oof_f, main_parts, y, "fake-peers")["judgeable"]:.4f}（真实分组见上）')

json.dump(R, open('results/F_stage_probe/experiment_H2_results.json', 'w'),
          indent=2, ensure_ascii=False, default=float)
print('\n  → results/F_stage_probe/experiment_H2_results.json')
