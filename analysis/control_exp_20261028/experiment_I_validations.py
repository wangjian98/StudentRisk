#!/usr/bin/env python3
"""
实验 I：三项判决性验证（2026-10-30 王建拍板）
===========================================
V1 去活跃度对照：阶段级判别力是不是"只会数活跃度（敲了多少行/做了多少题/花了多久）"？
V2 覆盖能力：阶段级掉队标签能否覆盖"走完全程仍挂科"的学生？（不做这件事 = 漏报 ~19%）
V3 早期性曲线：只用前 k 段行为，最早能在第几段发出预警？

数据：results/F_stage_probe/stage_matrix.npz（2171 个 (student,part) 阶段样本 / 473 学生）
      每行 10 维 = 7 事件计数 + total + distinct_exercises + span_hours
运行：纯 CPU，无需 GPU。  输出：results/I_validations/
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

BASE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(BASE, 'results/F_stage_probe/stage_matrix.npz')
OUT = os.path.join(BASE, 'results/I_validations')
os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 123, 777]
PARTS = list(range(1, 8))
NFOLD = 5

# ----------------------------------------------------------------- 加载
D = np.load(NPZ, allow_pickle=True)
FEAT, SID, PID = D['feat'], D['sid'], D['pid']
Y, STU = D['y'], D['students']
sidx = {int(s): i for i, s in enumerate(STU)}

per = defaultdict(dict)
for j in range(len(SID)):
    per[int(SID[j])][int(PID[j])] = FEAT[j].astype(float)

# 每 part 题量基准（完成度分母）= 全体 distinct_exercises 众数
base_ex = {}
for p in PARTS:
    c = Counter(int(per[s][p][8]) for s in per if p in per[s])
    base_ex[p] = c.most_common(1)[0][0]

# ---- 学生级分组：掉队型挂科 / 走完全程仍挂科 / 通过
stu_grp = {}
for s in per:
    mx = max(per[s])
    lab = int(Y[sidx[s]])
    stu_grp[s] = ('dropout' if (mx < 7 and lab == 1)
                  else 'full_fail' if (mx == 7 and lab == 1)
                  else 'pass')
grp_cnt = Counter(stu_grp.values())

# 阶段记录
st = []
for s in sorted(per):
    ks = sorted(per[s])
    for i, k in enumerate(ks):
        st.append(dict(s=s, p=k, i=i, f=per[s][k], maxp=max(ks),
                       drop=1 if (i == len(ks) - 1 and max(ks) < 7) else 0,
                       final=int(Y[sidx[s]]), grp=stu_grp[s]))
N = len(st)
st_p = np.array([r['p'] for r in st])
st_i = np.array([r['i'] for r in st])
st_drop = np.array([r['drop'] for r in st])
st_final = np.array([r['final'] for r in st])
st_s = np.array([r['s'] for r in st])
st_grp = np.array([r['grp'] for r in st])
main_part = {s: max(per[s], key=lambda p: per[s][p][7]) for s in per}
st_main = np.array([main_part[s] for s in st_s])

print('=' * 92)
print(f'阶段样本 {N} / 学生 {len(per)} / 掉队正例 {int(st_drop.sum())} / 题量基准 {base_ex}')
print(f"学生三类：掉队型挂科 {grp_cnt['dropout']} / 走完全程仍挂科 {grp_cnt['full_fail']} "
      f"/ 通过 {grp_cnt['pass']}")


# ----------------------------------------------------------------- 特征
def raw_vec(f):
    return np.array([f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9]])


RAW = np.array([raw_vec(r['f']) for r in st])
SPAN = np.maximum(RAW[:, 9], 1e-3)
INS = np.maximum(RAW[:, 0], 1e-3)

DER = np.column_stack([
    np.log1p(RAW[:, 7]),           # 0 log_total
    np.log1p(RAW[:, 8]),           # 1 log_exercises
    np.log1p(RAW[:, 9]),           # 2 log_span
    np.log1p(RAW[:, 7] / SPAN),    # 3 log_intensity
    RAW[:, 1] / INS,               # 4 del_rate
    RAW[:, 4] / SPAN,              # 5 focus_lost per hour
    RAW[:, 6] / np.maximum(RAW[:, 7], 1),          # 6 submit_rate
    np.array([RAW[j, 8] / base_ex[st_p[j]] for j in range(N)]),   # 7 completion
    st_i.astype(float),            # 8 stage_idx
])

SELF = np.zeros((N, 10))
_by = defaultdict(list)
for j in range(N):
    _by[st_s[j]].append(j)
for s, js in _by.items():
    js = sorted(js, key=lambda j: st_i[j])
    for pos, j in enumerate(js):
        if pos:
            h = RAW[js[:pos]]
            SELF[j] = (RAW[j] - h.mean(0)) / (h.std(0) + 1e-9)


def peer_z(tr_rows, tgt_rows, group_full):
    """按 group 分组做同伴 z-score（中位数/MAD），统计量只用训练折"""
    out = np.zeros((len(tgt_rows), RAW.shape[1]))
    for g in np.unique(group_full[tr_rows]):
        tr_g = np.intersect1d(np.where(group_full == g)[0], tr_rows)
        if len(tr_g) == 0:
            continue
        Xg = RAW[tr_g]
        med = np.median(Xg, 0)
        mad = np.maximum(np.median(np.abs(Xg - med), 0), 1e-3)
        for i, r in enumerate(tgt_rows):
            if group_full[r] == g:
                out[i] = (RAW[r] - med) / mad
    return np.clip(out, -5, 5)


def per_part_auc(y, oof, group):
    out = {}
    for g in np.unique(group):
        m = group == g
        if len(set(y[m])) > 1:
            out[int(g)] = dict(n=int(m.sum()), pos=int(y[m].sum()),
                               auc=float(roc_auc_score(y[m], oof[m])))
    return out


def pooled(pp):
    if not pp:
        return float('nan')
    a = np.array([v['auc'] for v in pp.values()])
    n = np.array([v['n'] for v in pp.values()], float)
    return float(np.average(a, weights=n))


def judged(pp):
    j = [v['auc'] for v in pp.values() if v['pos'] >= 10 and v['n'] - v['pos'] >= 10]
    return float(np.mean(j)) if j else float('nan')


def report(tag, y, oof, group):
    pp = per_part_auc(y, oof, group)
    r = dict(tag=tag, auc=float(roc_auc_score(y, oof)), pooled=pooled(pp),
             judged=judged(pp), per_part={str(k): v for k, v in pp.items()},
             oof=[float(x) for x in oof])
    print(f'  {tag:36s} 总体AUC={r["auc"]:.4f} | part内pooled={r["pooled"]:.4f} | '
          f'judgeable={r["judged"]:.4f}')
    for k, v in sorted(pp.items()):
        print(f'      part{k}: n={v["n"]:4d} pos={v["pos"]:4d} AUC={v["auc"]:.4f}')
    return r


def part_prior_oof(y, group, folds):
    pr = np.zeros(len(y))
    for tr, te in folds:
        for g in np.unique(group[tr]):
            mtr = tr[group[tr] == g]
            mte = te[group[te] == g]
            if len(mtr):
                pr[mte] = y[mtr].mean()
            elif len(mte):
                pr[mte] = y[tr].mean()
    return pr


def train_oof(Xtr_list, Xte_list, y, tr, te):
    acc = np.zeros(len(te))
    for sd in SEEDS:
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                                   class_weight='balanced', random_state=sd, n_jobs=-1)
        m.fit(Xtr_list, y[tr])
        acc += m.predict_proba(Xte_list)[:, 1]
    return acc / len(SEEDS)


def paired_boot(y, oa, ob, group, n=3000):
    rng = np.random.default_rng(20261030)
    gs = np.unique(group)
    idx_by_g = {g: np.where(group == g)[0] for g in gs}
    obs = pooled(per_part_auc(y, oa, group)) - pooled(per_part_auc(y, ob, group))
    diffs = np.empty(n)
    for it in range(n):
        oa2, ob2 = oa.copy(), ob.copy()
        for g in gs:
            ii = idx_by_g[g]
            pick = rng.choice(ii, size=len(ii), replace=True)
            oa2[pick] = oa[ii]
            ob2[pick] = ob[ii]
        diffs[it] = pooled(per_part_auc(y, oa2, group)) - pooled(per_part_auc(y, ob2, group))
    return dict(obs=float(obs), lo=float(np.percentile(diffs, 2.5)),
                hi=float(np.percentile(diffs, 97.5)), p_le0=float((diffs <= 0).mean()))


# ================================================================= V1
print('\n' + '=' * 92)
print('V1) 去活跃度对照：阶段级判别力是不是"只会数活跃度"？')
print('=' * 92)

folds = list(StratifiedKFold(NFOLD, shuffle=True, random_state=42).split(np.zeros((N, 1)), st_drop))
prior_stage = part_prior_oof(st_drop, st_p, folds)

A_COL = [0, 1, 2, 3]
VOL_COL = [0, 1]
B_COL = [4, 5, 6]
ACT_DIMS = [7, 8, 9]      # RAW 里对应 total / exercises / span


def build(kind, tr, te):
    g = st_main if kind == 'mainpart' else st_p
    ztr = peer_z(tr, tr, g)
    zte = peer_z(tr, te, g)
    A, VOL, B = DER[:, A_COL], DER[:, VOL_COL], DER[:, B_COL]
    C = np.column_stack([DER[:, 7], DER[:, 8]])
    Dna_tr = np.delete(ztr, ACT_DIMS, axis=1)
    Dna_te = np.delete(zte, ACT_DIMS, axis=1)
    Ena_tr = np.delete(SELF[tr], ACT_DIMS, axis=1)
    Ena_te = np.delete(SELF[te], ACT_DIMS, axis=1)
    spec = {
        'active_only': (A[tr], A[te]),
        'volume_only': (VOL[tr], VOL[te]),
        'completion_only': (C[tr], C[te]),
        'peer_all': (ztr, zte),
        'full': (np.hstack([A[tr], B[tr], C[tr], ztr, SELF[tr]]),
                 np.hstack([A[te], B[te], C[te], zte, SELF[te]])),
        'no_active': (np.hstack([B[tr], C[tr], Dna_tr, Ena_tr]),
                      np.hstack([B[te], C[te], Dna_te, Ena_te])),
    }
    return spec['full' if kind == 'mainpart' else kind]


def run_variant(kind, y=None, fold_list=None):
    y = st_drop if y is None else y
    fold_list = folds if fold_list is None else fold_list
    oof = np.zeros(N)
    for tr, te in fold_list:
        Xtr, Xte = build(kind, tr, te)
        oof[te] = train_oof(Xtr, Xte, y, tr, te)
    return oof


R1 = {}
for kind in ['active_only', 'volume_only', 'completion_only', 'peer_all', 'full', 'no_active']:
    R1[kind] = report(f'V1 {kind}', st_drop, run_variant(kind), st_p)

oof_main = np.zeros(N)
for tr, te in folds:
    Xtr, Xte = build('mainpart', tr, te)
    oof_main[te] = train_oof(Xtr, Xte, st_drop, tr, te)
R1['full_mainpart_replication'] = report('V1 full(main-part分组,H3口径)', st_drop, oof_main, st_p)
R1['part_prior_stage'] = report('V1 阶段级 part-prior 基线', st_drop, prior_stage, st_p)


def residualize(Xtr, Xte, Atr_in, Ate_in):
    Atr = np.hstack([np.ones((len(Atr_in), 1)), Atr_in])
    beta, *_ = np.linalg.lstsq(Atr, Xtr, rcond=None)
    Ate = np.hstack([np.ones((len(Ate_in), 1)), Ate_in])
    return Xtr - Atr @ beta, Xte - Ate @ beta


def run_residual():
    oof = np.zeros(N)
    for tr, te in folds:
        ztr = peer_z(tr, tr, st_p)
        zte = peer_z(tr, te, st_p)
        Xtr_raw = np.hstack([DER[tr][:, 4:9], ztr, SELF[tr]])
        Xte_raw = np.hstack([DER[te][:, 4:9], zte, SELF[te]])
        Rtr, Rte = residualize(Xtr_raw, Xte_raw, DER[tr][:, A_COL], DER[te][:, A_COL])
        oof[te] = train_oof(Rtr, Rte, st_drop, tr, te)
    return oof


R1['residual_on_active'] = report('V1 残差化(剔除活跃度方向)', st_drop, run_residual(), st_p)

print('\n  [配对 bootstrap · pooled within-part AUC 差 · 3000 次]')
boots = {}
for a, b in [('full', 'active_only'), ('full', 'no_active'), ('no_active', 'active_only'),
             ('full', 'completion_only'), ('residual_on_active', 'active_only'),
             ('no_active', 'part_prior_stage'), ('full', 'part_prior_stage'),
             ('completion_only', 'active_only')]:
    d = paired_boot(st_drop, np.array(R1[a]['oof']), np.array(R1[b]['oof']), st_p)
    boots[f'{a}_minus_{b}'] = d
    print(f'    {a} − {b}: Δpooled={d["obs"]:+.4f} [{d["lo"]:+.4f},{d["hi"]:+.4f}] P(≤0)={d["p_le0"]*100:.2f}%')

# ================================================================= V2
print('\n' + '=' * 92)
print('V2) 覆盖能力：阶段级掉队标签能否覆盖"走完全程仍挂科"的学生？')
print('=' * 92)

is_ff = (st_grp == 'full_fail').astype(int)
is_do = (st_grp == 'dropout').astype(int)

mask_nd = np.where(st_grp != 'dropout')[0]
y_ff = is_ff[mask_nd]
folds_ff = list(StratifiedKFold(NFOLD, shuffle=True, random_state=42).split(np.zeros((len(mask_nd), 1)), y_ff))
oof_ff = np.zeros(N)
for tr, te in folds_ff:
    tr_g, te_g = mask_nd[tr], mask_nd[te]
    ztr = peer_z(tr_g, tr_g, st_p)
    zte = peer_z(tr_g, te_g, st_p)
    Xtr = np.hstack([DER[tr_g][:, A_COL], DER[tr_g][:, 4:9], ztr, SELF[tr_g]])
    Xte = np.hstack([DER[te_g][:, A_COL], DER[te_g][:, 4:9], zte, SELF[te_g]])
    oof_ff[te_g] = train_oof(Xtr, Xte, y_ff, tr, te)
R2 = {'fullfail_vs_pass': report('V2a full_fail vs pass（part内）', y_ff, oof_ff[mask_nd], st_p[mask_nd])}
R2['fullfail_vs_pass']['n_pos'] = int(y_ff.sum())
R2['fullfail_vs_pass']['n_neg'] = int((1 - y_ff).sum())

oof_final = np.zeros(N)
for tr, te in folds:
    ztr = peer_z(tr, tr, st_p)
    zte = peer_z(tr, te, st_p)
    Xtr = np.hstack([DER[tr][:, 4:9], ztr, SELF[tr]])
    Xte = np.hstack([DER[te][:, 4:9], zte, SELF[te]])
    oof_final[te] = train_oof(Xtr, Xte, st_final, tr, te)
R2['final_all'] = report('V2b 全部学生·最终挂科（part内）', st_final, oof_final, st_p)

print('\n  [画像对比]')
prof = {}
for nm in ['dropout', 'full_fail', 'pass']:
    sel = np.where(st_grp == nm)[0]
    prof[nm] = dict(n_students=int(len(set(st_s[sel].tolist()))),
                    n_stage_rows=int(len(sel)),
                    log_intensity_med=float(np.median(DER[sel, 3])),
                    del_rate_med=float(np.median(DER[sel, 4])),
                    focus_rate_med=float(np.median(DER[sel, 5])),
                    completion_med=float(np.median(DER[sel, 7])),
                    span_h_med=float(np.median(np.expm1(DER[sel, 2]))))
    p = prof[nm]
    print(f'    {nm:10s} 学生 {p["n_students"]:3d} | 强度log中位 {p["log_intensity_med"]:.2f} '
          f'| 删除率 {p["del_rate_med"]*100:.1f}% | 焦点/h {p["focus_rate_med"]:.1f} '
          f'| 完成度 {p["completion_med"]:.2f} | 跨度 {p["span_h_med"]:.1f}h')
R2['profile'] = prof
R2['fullfail_part_dist'] = {str(k): v for k, v in sorted(Counter(st_p[is_ff == 1].tolist()).items())}
print(f'    full_fail 在各 part 的样本分布: {R2["fullfail_part_dist"]}')

# ================================================================= V3
print('\n' + '=' * 92)
print('V3) 早期性曲线：只用前 k 段行为，最早第几段能预警？')
print('=' * 92)


def prefix_raw(student, k):
    ks = sorted(per[student])[:k]
    return np.sum([raw_vec(per[student][p]) for p in ks], axis=0)


def mk_prefix(raw, rows, k):
    """前缀累计特征；完成度分母 = 该生前 k 个 part 的题量之和"""
    def denom(a):
        s = st_s[rows[a]]
        ks = sorted(per[s])[:k]
        return max(sum(base_ex[p] for p in ks), 1)
    d = np.column_stack([
        np.log1p(raw[:, 7]), np.log1p(raw[:, 8]), np.log1p(raw[:, 9]),
        np.log1p(raw[:, 7] / np.maximum(raw[:, 9], 1e-3)),
        raw[:, 1] / np.maximum(raw[:, 0], 1e-3),
        raw[:, 4] / np.maximum(raw[:, 9], 1e-3),
        raw[:, 6] / np.maximum(raw[:, 7], 1),
        np.array([raw[a, 8] / denom(a) for a in range(len(rows))]),
        np.full(len(rows), float(k - 1)),
    ])
    return d


R3 = {'curve': []}
for k in range(1, 7):
    pref = np.zeros((N, 10))
    ok = np.zeros(N, bool)
    for j, r in enumerate(st):
        if r['i'] + 1 >= k:
            pref[j] = prefix_raw(r['s'], k)
            ok[j] = True
    idx = np.where(ok)[0]
    yk = st_drop[idx]
    folds_k = list(StratifiedKFold(NFOLD, shuffle=True, random_state=42).split(np.zeros((len(idx), 1)), yk))
    oof_k = np.zeros(len(idx))
    for tr, te in folds_k:
        tr_g, te_g = idx[tr], idx[te]
        ztr = peer_z(tr_g, tr_g, st_p)
        zte = peer_z(tr_g, te_g, st_p)
        Xtr = np.hstack([mk_prefix(pref[tr_g], tr_g, k), ztr])
        Xte = np.hstack([mk_prefix(pref[te_g], te_g, k), zte])
        oof_k[te] = train_oof(Xtr, Xte, yk, tr, te)
    prior_k = part_prior_oof(yk, st_p[idx], folds_k)
    pp = per_part_auc(yk, oof_k, st_p[idx])
    pp_prior = per_part_auc(yk, prior_k, st_p[idx])
    row = dict(k=k, n=int(len(idx)), pos=int(yk.sum()),
               auc=float(roc_auc_score(yk, oof_k)), pooled=pooled(pp),
               prior_auc=float(roc_auc_score(yk, prior_k)), prior_pooled=pooled(pp_prior),
               per_part={str(a): b for a, b in pp.items()})
    R3['curve'].append(row)
    print(f'    k={k}: n={row["n"]:5d} 总体AUC={row["auc"]:.4f} | part内pooled={row["pooled"]:.4f} '
          f'| part-prior pooled={row["prior_pooled"]:.4f} | 领先先验={row["auc"]-row["prior_auc"]:+.4f}')

# ================================================================= 落盘
out = dict(meta=dict(N=N, students=len(per), base_ex=base_ex,
                     student_groups=dict(grp_cnt)),
           V1={k: {kk: vv for kk, vv in v.items() if kk != 'oof'} for k, v in R1.items()},
           V1_bootstrap=boots,
           V2={k: ({kk: vv for kk, vv in v.items() if kk != 'oof'} if isinstance(v, dict) and 'oof' in v else v)
               for k, v in R2.items()},
           V3=R3)
with open(os.path.join(OUT, 'validation_results.json'), 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
np.savez_compressed(os.path.join(OUT, 'oof_all.npz'),
                    **{f'oof_{k}': np.array(v['oof']) for k, v in R1.items() if 'oof' in v},
                    oof_ff=oof_ff, oof_final=oof_final)
print(f'\n写出: {os.path.join(OUT, "validation_results.json")}')
print('=' * 92)
