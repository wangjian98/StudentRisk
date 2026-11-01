#!/usr/bin/env python3
"""
实验 I 后置：干净版配对 bootstrap（修正 experiment_I_validations.py 内联 bootstrap 的就地覆写偏差）
用 rows 重采样构造新数组（标准做法），而不是在原数组上就地覆写。
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'results/I_validations')
Z = np.load(os.path.join(OUT, 'oof_all.npz'))

D = np.load(os.path.join(BASE, 'results/F_stage_probe/stage_matrix.npz'), allow_pickle=True)
SID, PID = D['sid'], D['pid']
st_p = PID.astype(int)
# 阶段级掉队标签：从每生最大 part 推导
per_max = {}
for j in range(len(SID)):
    s = int(SID[j]); p = int(PID[j])
    per_max[s] = max(per_max.get(s, 0), p)
# drop=1 当且仅当该 (学生,part) 是其最远 part 且未到 part7
st_drop = np.zeros(len(SID), dtype=int)
seen = set()
for j in range(len(SID)):
    s = int(SID[j]); p = int(PID[j])
    if (s, p) in seen:
        continue
    seen.add((s, p))
    if p == per_max[s] and p < 7:
        st_drop[j] = 1
# 只保留每个 (s,p) 的首行（与主实验一致）
mask_first = np.zeros(len(SID), bool)
for j in range(len(SID)):
    key = (int(SID[j]), int(PID[j]))
    if key not in seen:
        pass
# 直接重建：主脚本用 dict 覆盖，顺序与 stage_matrix 一致
per = {}
order = []
for j in range(len(SID)):
    k = (int(SID[j]), int(PID[j]))
    if k not in per:
        per[k] = j
        order.append(j)
sel = np.array(order)
st_p = st_p[sel]
st_drop = st_drop[sel]


def pooled_auc(y, oof, group):
    aucs, ns = [], []
    for g in np.unique(group):
        m = group == g
        if len(set(y[m])) > 1:
            aucs.append(roc_auc_score(y[m], oof[m]))
            ns.append(m.sum())
    return float(np.average(aucs, weights=ns))


def boot_diff(y, oa, ob, group, n=4000, seed=20261030):
    rng = np.random.default_rng(seed)
    idx_by_g = [np.where(group == g)[0] for g in np.unique(group)]
    obs = pooled_auc(y, oa, group) - pooled_auc(y, ob, group)
    ds = np.empty(n)
    for it in range(n):
        pick = np.concatenate([rng.choice(ii, len(ii), replace=True) for ii in idx_by_g])
        ds[it] = pooled_auc(y[pick], oa[pick], group[pick]) - pooled_auc(y[pick], ob[pick], group[pick])
    return dict(obs=float(obs), lo=float(np.percentile(ds, 2.5)), hi=float(np.percentile(ds, 97.5)),
                p_le0=float((ds <= 0).mean()), mean=float(ds.mean()))


R = {}
print(f'{"对比":52s} {"Δpooled":>9s} {"95%CI":>22s} {"P(≤0)":>8s}')
for a, b in [('oof_full', 'oof_active_only'), ('oof_full', 'oof_no_active'),
             ('oof_no_active', 'oof_active_only'), ('oof_full', 'oof_completion_only'),
             ('oof_residual_on_active', 'oof_active_only'), ('oof_full', 'oof_volume_only'),
             ('oof_peer_all', 'oof_active_only'),
             ('oof_full', 'oof_part_prior_stage'), ('oof_no_active', 'oof_part_prior_stage'),
             ('oof_full', 'oof_full_mainpart_replication')]:
    if a not in Z.files or b not in Z.files:
        continue
    d = boot_diff(st_drop, Z[a], Z[b], st_p)
    R[f'{a}_minus_{b}'] = d
    print(f'{a.replace("oof_","")+" − "+b.replace("oof_",""):52s} {d["obs"]:+9.4f} '
          f'[{d["lo"]:+7.4f},{d["hi"]:+7.4f}] {d["p_le0"]*100:7.2f}%')

with open(os.path.join(OUT, 'bootstrap_clean.json'), 'w') as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print('\n写出 bootstrap_clean.json')
