#!/usr/bin/env python3
"""
阶段级主结果头对头分析（本地，同折同协议）
============================================
输入：246 上跑出的 OOF 概率（同一 5 折按学生分组划分 + 3 seeds）
输出：
  1. 主表（总体 AUC / part 内 pooled / judgeable / ΔAUC vs 阶段级 prior）
  2. MetaMamba vs RF 的配对 bootstrap（判决性）
  3. 双层聚合：阶段级告警 → 学生级终局风险 + 覆盖率
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, 'results')
D = np.load(os.path.join(BASE, '..', '..', 'data', 'stage_matrix.npz')) \
    if os.path.exists(os.path.join(BASE, '..', '..', 'data', 'stage_matrix.npz')) \
    else np.load('/root/.openclaw/workspace-staging/pc-ceo_assistant/StudentRisk/'
                 'analysis/control_exp_20261028/results/F_stage_probe/stage_matrix.npz',
                 allow_pickle=True)

FEAT, SID, PID = D['feat'], D['sid'], D['pid']
Y, STU = D['y'], D['students']
sidx = {int(s): i for i, s in enumerate(STU)}
per_max = {}
for j in range(len(SID)):
    s, p = int(SID[j]), int(PID[j])
    per_max[s] = max(per_max.get(s, 0), p)
seen, order = set(), []
for j in range(len(SID)):
    k = (int(SID[j]), int(PID[j]))
    if k not in seen:
        seen.add(k)
        order.append(j)
sel = np.array(order)
part = PID[sel].astype(int)
sid = SID[sel].astype(int)
drop = np.array([1 if (part[j] == per_max[sid[j]] and part[j] < 7) else 0 for j in range(len(sel))])
final = np.array([int(Y[sidx[sid[j]]]) for j in range(len(sel))])
students = np.array(sorted(set(sid.tolist())))
final_stu = np.array([int(Y[sidx[s]]) for s in students])

Z = np.load(os.path.join(RES, 'stage_oof_rf.npz'))
OOF = {k: Z[k] for k in Z.files}


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


def boot(y, oa, ob, group, n=4000, seed=20261031):
    rng = np.random.default_rng(seed)
    idx_by_g = [np.where(group == g)[0] for g in np.unique(group)]
    obs = pooled(per_part_auc(y, oa, group)) - pooled(per_part_auc(y, ob, group))
    ds = np.empty(n)
    for it in range(n):
        pick = np.concatenate([rng.choice(ii, len(ii), replace=True) for ii in idx_by_g])
        ds[it] = (pooled(per_part_auc(y[pick], oa[pick], group[pick]))
                  - pooled(per_part_auc(y[pick], ob[pick], group[pick])))
    return dict(obs=float(obs), lo=float(np.percentile(ds, 2.5)),
                hi=float(np.percentile(ds, 97.5)), p_le0=float((ds <= 0).mean()))


print('=' * 96)
print('表 8 · 阶段级推进决策主结果（5 折按学生分组 × 3 seeds，n=2171 阶段样本 / 473 学生）')
print('=' * 96)
rows = {}
pp_prior = per_part_auc(drop, OOF['stage_part_prior'], part)
prior_pool = pooled(pp_prior)
rows['stage_part_prior'] = dict(auc=float(roc_auc_score(drop, OOF['stage_part_prior'])),
                                 pooled=prior_pool, judged=judged(pp_prior))
for k in ['rf_completion_only', 'rf_active_only', 'rf_peer_all', 'rf_full', 'rf_no_active']:
    pp = per_part_auc(drop, OOF[k], part)
    rows[k] = dict(auc=float(roc_auc_score(drop, OOF[k])), pooled=pooled(pp), judged=judged(pp),
                   per_part={str(a): b for a, b in pp.items()})
for tag, k in [('mm11', 'stage_oof_mm11'), ('mm9', 'stage_oof_mm9'), ('mm7', 'stage_oof_mm7')]:
    pass
MM = {}
for f, tag in [('stage_oof_mm11.npz', 'mm11'), ('stage_oof_mm9.npz', 'mm9'),
               ('stage_oof_mm7.npz', 'mm7')]:
    ZZ = np.load(os.path.join(RES, f))
    MM[tag] = ZZ[tag]

print(f'{"模型":38s} {"总体AUC":>8s} {"part内pooled":>12s} {"judgeable":>10s} {"Δpooled vs prior":>16s}')
for k in ['stage_part_prior', 'rf_completion_only', 'rf_active_only', 'rf_peer_all',
          'rf_full', 'rf_no_active']:
    v = rows[k]
    print(f'{k:38s} {v["auc"]:8.4f} {v["pooled"]:12.4f} {v["judged"]:10.4f} '
          f'{v["pooled"]-prior_pool:+16.4f}')
for tag in ['mm11', 'mm9', 'mm7']:
    o = MM[tag]
    pp = per_part_auc(drop, o, part)
    v = dict(auc=float(roc_auc_score(drop, o)), pooled=pooled(pp), judged=judged(pp),
             per_part={str(a): b for a, b in pp.items()})
    rows[tag] = v
    print(f'{"MetaMamba-" + tag:38s} {v["auc"]:8.4f} {v["pooled"]:12.4f} {v["judged"]:10.4f} '
          f'{v["pooled"]-prior_pool:+16.4f}')

print('\n' + '=' * 96)
print('判决性对比（配对 bootstrap · part 内 pooled AUC 差 · 4000 次）')
print('=' * 96)
boots = {}
pairs = [('mm11', 'rf_full'), ('mm11', 'rf_no_active'), ('mm9', 'rf_full'),
         ('mm7', 'rf_full'), ('rf_full', 'rf_no_active'), ('rf_full', 'rf_active_only'),
         ('mm11', 'rf_peer_all')]
for a, b in pairs:
    d = boot(drop, MM[a], OOF[b], part)
    boots[f'{a}_minus_{b}'] = d
    verdict = 'A 更好' if d['p_le0'] < 0.05 else ('B 更好' if d['p_le0'] > 0.95 else '无显著差异')
    print(f'  {a:12s} − {b:16s} Δpooled={d["obs"]:+.4f} [{d["lo"]:+.4f},{d["hi"]:+.4f}] '
          f'P(≤0)={d["p_le0"]*100:5.2f}%  → {verdict}')

print('\n' + '=' * 96)
print('双层聚合：阶段级告警 → 学生级终局风险')
print('=' * 96)
res = {}
for tag, o in list(MM.items()) + [('rf_full', OOF['rf_full']), ('rf_no_active', OOF['rf_no_active'])]:
    # 每个学生 = 其所有阶段分数的 max / mean / last
    stu_max = np.array([o[sid == s].max() for s in students])
    stu_mean = np.array([o[sid == s].mean() for s in students])
    # 学生级"是否有任一阶段高风险"（阈值扫描）
    res[tag] = dict(
        max_auc=float(roc_auc_score(final_stu, stu_max)),
        mean_auc=float(roc_auc_score(final_stu, stu_mean)),
    )
    print(f'  {tag:14s} 学生级 AUC：max聚合={res[tag]["max_auc"]:.4f}  mean聚合={res[tag]["mean_auc"]:.4f}'
          f'   （对照：学生级 part-prior 0.9135 / 学生级 RF part内 0.7690）')

# 覆盖率：阶段级在某个阈值下抓到多少挂科学生
print('\n  覆盖率（第一层抓到 / 第二层补上）：')
for tag in ['mm11', 'rf_full']:
    o = MM[tag] if tag != 'rf_full' else OOF['rf_full']
    stu_max = np.array([o[sid == s].max() for s in students])
    for thr in [0.3, 0.5, 0.7]:
        alerted = stu_max >= thr
        tp = int((alerted & (final_stu == 1)).sum())
        fp = int((alerted & (final_stu == 0)).sum())
        cov = tp / max(int((final_stu == 1).sum()), 1)
        prec = tp / max(tp + fp, 1)
        print(f'    {tag:8s} 阈值{thr}: 告警{alerted.sum():3d}人 命中挂科{tp:3d}/'
              f'{int((final_stu==1).sum())}（召回{cov*100:.1f}%） 精确率{prec*100:.1f}%')

out = dict(main_table={k: {kk: vv for kk, vv in v.items() if kk != 'per_part'}
                       for k, v in rows.items()},
           bootstrap=boots, two_layer=res,
           per_part={k: rows[k].get('per_part') for k in rows})
with open(os.path.join(RES, 'stage_headtohead.json'), 'w') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f'\n写出 {os.path.join(RES, "stage_headtohead.json")}')
