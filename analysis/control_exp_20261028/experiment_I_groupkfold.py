#!/usr/bin/env python3
"""
实验 I-补充：学生级泄漏排查（GroupKFold by student）
=====================================================
问题：阶段级任务的样本是 (学生, part) 行，同一学生的多个阶段行在 StratifiedKFold(按行) 下
      会同时进入训练折和测试折 → 模型可能"记住学生身份" → 阶段级指标被系统性抬高。
      本实验用 GroupKFold(student) 复算所有关键数字，与按行划分的口径对照。

若 GroupKFold 下数字基本不变 → 之前的 0.9718 成立；
若明显下降 → 阶段级高指标部分来自泄漏，M8 需要重新表述。
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'results/I_validations')
os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 123, 777]
PARTS = list(range(1, 8))

D = np.load(os.path.join(BASE, 'results/F_stage_probe/stage_matrix.npz'), allow_pickle=True)
FEAT, SID, PID = D['feat'], D['sid'], D['pid']
Y, STU = D['y'], D['students']
sidx = {int(s): i for i, s in enumerate(STU)}

per = defaultdict(dict)
for j in range(len(SID)):
    per[int(SID[j])][int(PID[j])] = FEAT[j].astype(float)

base_ex = {}
for p in PARTS:
    c = Counter(int(per[s][p][8]) for s in per if p in per[s])
    base_ex[p] = c.most_common(1)[0][0]

st = []
for s in sorted(per):
    ks = sorted(per[s])
    for i, k in enumerate(ks):
        st.append(dict(s=s, p=k, i=i, f=per[s][k], maxp=max(ks),
                       drop=1 if (i == len(ks) - 1 and max(ks) < 7) else 0,
                       final=int(Y[sidx[s]])))
N = len(st)
st_p = np.array([r['p'] for r in st])
st_i = np.array([r['i'] for r in st])
st_drop = np.array([r['drop'] for r in st])
st_final = np.array([r['final'] for r in st])
st_s = np.array([r['s'] for r in st])
main_part = {s: max(per[s], key=lambda p: per[s][p][7]) for s in per}
st_main = np.array([main_part[s] for s in st_s])

RAW = np.array([np.array([r['f'][0], r['f'][1], r['f'][2], r['f'][3], r['f'][4], r['f'][5],
                           r['f'][6], r['f'][7], r['f'][8], r['f'][9]]) for r in st])
SPAN = np.maximum(RAW[:, 9], 1e-3)
INS = np.maximum(RAW[:, 0], 1e-3)
DER = np.column_stack([
    np.log1p(RAW[:, 7]), np.log1p(RAW[:, 8]), np.log1p(RAW[:, 9]),
    np.log1p(RAW[:, 7] / SPAN), RAW[:, 1] / INS, RAW[:, 4] / SPAN,
    RAW[:, 6] / np.maximum(RAW[:, 7], 1),
    np.array([RAW[j, 8] / base_ex[st_p[j]] for j in range(N)]),
    st_i.astype(float),
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
    a = np.array([v['auc'] for v in pp.values()])
    n = np.array([v['n'] for v in pp.values()], float)
    return float(np.average(a, weights=n))


def judged(pp):
    j = [v['auc'] for v in pp.values() if v['pos'] >= 10 and v['n'] - v['pos'] >= 10]
    return float(np.mean(j)) if j else float('nan')


def show(tag, y, oof, group):
    pp = per_part_auc(y, oof, group)
    print(f'  {tag:44s} 总体={roc_auc_score(y,oof):.4f} | part内pooled={pooled(pp):.4f} '
          f'| judgeable={judged(pp):.4f}')
    return dict(auc=float(roc_auc_score(y, oof)), pooled=pooled(pp), judged=judged(pp),
                per_part={str(k): v for k, v in pp.items()})


def full_X(kind, tr, te):
    g = st_main if kind == 'mainpart' else st_p
    ztr = peer_z(tr, tr, g)
    zte = peer_z(tr, te, g)
    A, B = DER[:, [0, 1, 2, 3]], DER[:, [4, 5, 6]]
    C = np.column_stack([DER[:, 7], DER[:, 8]])
    return (np.hstack([A[tr], B[tr], C[tr], ztr, SELF[tr]]),
            np.hstack([A[te], B[te], C[te], zte, SELF[te]]))


def train(Xtr, Xte, y, tr, te):
    acc = np.zeros(len(te))
    for sd in SEEDS:
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                                   class_weight='balanced', random_state=sd, n_jobs=-1)
        m.fit(Xtr, y[tr])
        acc += m.predict_proba(Xte)[:, 1]
    return acc / len(SEEDS)


def prefix_raw(student, k):
    return np.sum([np.array([per[student][p][0], per[student][p][1], per[student][p][2],
                              per[student][p][3], per[student][p][4], per[student][p][5],
                              per[student][p][6], per[student][p][7], per[student][p][8],
                              per[student][p][9]]) for p in sorted(per[student])[:k]], axis=0)


def mk_prefix(raw, rows, k):
    def denom(a):
        s = st_s[rows[a]]
        return max(sum(base_ex[p] for p in sorted(per[s])[:k]), 1)
    return np.column_stack([
        np.log1p(raw[:, 7]), np.log1p(raw[:, 8]), np.log1p(raw[:, 9]),
        np.log1p(raw[:, 7] / np.maximum(raw[:, 9], 1e-3)),
        raw[:, 1] / np.maximum(raw[:, 0], 1e-3),
        raw[:, 4] / np.maximum(raw[:, 9], 1e-3),
        raw[:, 6] / np.maximum(raw[:, 7], 1),
        np.array([raw[a, 8] / denom(a) for a in range(len(rows))]),
        np.full(len(rows), float(k - 1)),
    ])


def folds_row(y):
    return list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros((len(y), 1)), y))


def folds_group(y, groups):
    """按学生分组 + 分层：先按标签分层把学生分到 5 份，再取其全部阶段行"""
    uniq = np.unique(groups)
    lab = {s: int(y[groups == s][0]) for s in uniq}
    rng = np.random.default_rng(42)
    buckets = [[] for _ in range(5)]
    for cls in [0, 1]:
        ss = [s for s in uniq if lab[s] == cls]
        rng.shuffle(ss)
        for i, s in enumerate(ss):
            buckets[i % 5].append(s)
    out = []
    for b in range(5):
        te_s = set(buckets[b])
        te = np.array([j for j in range(len(y)) if groups[j] in te_s])
        tr = np.array([j for j in range(len(y)) if groups[j] not in te_s])
        out.append((tr, te))
    return out


RES = {}
print('=' * 100)
print('A) 阶段级掉队预测：按行 StratifiedKFold  vs  按学生 GroupKFold')
print('=' * 100)
for name, kind in [('full(current-part分组)', 'cur'), ('full(main-part分组,H3口径)', 'mainpart')]:
    oof_r = np.zeros(N)
    for tr, te in folds_row(st_drop):
        Xtr, Xte = full_X(kind, tr, te)
        oof_r[te] = train(Xtr, Xte, st_drop, tr, te)
    rr = show(f'{name} [按行]', st_drop, oof_r, st_p)
    oof_g = np.zeros(N)
    for tr, te in folds_group(st_drop, st_s):
        Xtr, Xte = full_X(kind, tr, te)
        oof_g[te] = train(Xtr, Xte, st_drop, tr, te)
    rg = show(f'{name} [按学生分组]', st_drop, oof_g, st_p)
    RES[name] = dict(row=rr, group=rg)
    print(f'    → Δpooled(分组 − 按行) = {rg["pooled"]-rr["pooled"]:+.4f}')

print('\n' + '=' * 100)
print('B) 学生级最终挂科（part 内）：按行 vs 按学生分组')
print('=' * 100)
oof_r = np.zeros(N)
for tr, te in folds_row(st_final):
    ztr = peer_z(tr, tr, st_p); zte = peer_z(tr, te, st_p)
    Xtr = np.hstack([DER[tr][:, 4:9], ztr, SELF[tr]])
    Xte = np.hstack([DER[te][:, 4:9], zte, SELF[te]])
    oof_r[te] = train(Xtr, Xte, st_final, tr, te)
RES['final_row'] = show('学生级最终挂科 [按行]', st_final, oof_r, st_p)
oof_g = np.zeros(N)
for tr, te in folds_group(st_final, st_s):
    ztr = peer_z(tr, tr, st_p); zte = peer_z(tr, te, st_p)
    Xtr = np.hstack([DER[tr][:, 4:9], ztr, SELF[tr]])
    Xte = np.hstack([DER[te][:, 4:9], zte, SELF[te]])
    oof_g[te] = train(Xtr, Xte, st_final, tr, te)
RES['final_group'] = show('学生级最终挂科 [按学生分组]', st_final, oof_g, st_p)

print('\n' + '=' * 100)
print('C) 早期性曲线（仅前 k 段）：按行 vs 按学生分组')
print('=' * 100)
RES['leadtime'] = []
for k in [1, 2, 3]:
    pref = np.zeros((N, 10))
    ok = np.zeros(N, bool)
    for j, r in enumerate(st):
        if r['i'] + 1 >= k:
            pref[j] = prefix_raw(r['s'], k)
            ok[j] = True
    idx = np.where(ok)[0]
    yk = st_drop[idx]
    for mode in ['row', 'group']:
        fl = folds_row(yk) if mode == 'row' else folds_group(yk, st_s[idx])
        oof_k = np.zeros(len(idx))
        for tr, te in fl:
            tr_g, te_g = idx[tr], idx[te]
            ztr = peer_z(tr_g, tr_g, st_p); zte = peer_z(tr_g, te_g, st_p)
            Xtr = np.hstack([mk_prefix(pref[tr_g], tr_g, k), ztr])
            Xte = np.hstack([mk_prefix(pref[te_g], te_g, k), zte])
            oof_k[te] = train(Xtr, Xte, yk, tr, te)
        pp = per_part_auc(yk, oof_k, st_p[idx])
        r = dict(k=k, mode=mode, n=int(len(idx)), auc=float(roc_auc_score(yk, oof_k)),
                 pooled=pooled(pp), judged=judged(pp))
        RES['leadtime'].append(r)
        print(f'    k={k} [{mode:5s}] n={r["n"]:5d} 总体={r["auc"]:.4f} | part内pooled={r["pooled"]:.4f} '
              f'| judgeable={r["judged"]:.4f}')

with open(os.path.join(OUT, 'groupkfold_check.json'), 'w') as f:
    json.dump(RES, f, ensure_ascii=False, indent=1)
print('\n写出 groupkfold_check.json')
