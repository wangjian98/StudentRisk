"""
实验 F：Stage-Conditional Hazard（进程条件化风险）可行性探针
=============================================================
目的：验证"把整课挂科重构为分阶段生存/流失风险"这一新模型路线的前提条件。

要回答的 4 个问题：
F1 事实层：part 序列在时间上是否单调（= "进度" 可从事件流直接推导）？
F2 事实层：failed vs passed 的进度分布差异有多大？（"进度"是否 ≈ 标签）
F3 关键：在"停下来的那个 part"内部（真正发生掉队的现场），个体行为是否仍有判别力？
     → 若 within-stop-part AUC ≈ 0.5，说明阶段标签里没有个体信号 → 新模型路线应砍
     → 若明显 > 0.5，说明阶段标签带了个体信号 → 新模型可做
F4 工程：per-(student, part) 聚合量是否够支撑"同伴参照系"（每 part 同伴数是否够）

只读本地数据，纯 CPU，无 GPU 依赖。
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

IDE = os.environ.get('STUDENTRISK_IDE_LOGS', '/root/.openclaw/data-backups/IDE_logs.csv')
PASS = os.environ.get('STUDENTRISK_PASSED', '/root/.openclaw/data-backups/passed.csv')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'F_stage_probe')
os.makedirs(OUT, exist_ok=True)

EV = ['text_insert', 'text_remove', 'text_paste',
      'focus_gained', 'focus_lost', 'run', 'submit']

print('=' * 100)
print('F0) 读取原始日志（28.5M 行，分块聚合）')
print('=' * 100, flush=True)

# (student, part) -> eventType 计数 / 时间跨度 / 练习数
cnt = defaultdict(lambda: defaultdict(int))
ts_min = {}
ts_max = {}
ex_n = {}
part_seq = defaultdict(list)   # student -> [(ts, part)] 的 part 序列（抽样用于单调性检查）
students = set()
raw_parts = set()
chunk_i = 0
for chunk in pd.read_csv(IDE, usecols=['student', 'part', 'exercise', 'eventType', 'timestamp'],
                         chunksize=2_000_000):
    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
    students.update(chunk['student'].unique().tolist())
    raw_parts.update(chunk['part'].unique().tolist())
    # 事件类型计数
    g = chunk.groupby(['student', 'part', 'eventType']).size()
    for (s, p, e), v in g.items():
        cnt[(int(s), int(p))][e] += int(v)
    # 时间跨度
    g2 = chunk.groupby(['student', 'part'])['timestamp'].agg(['min', 'max'])
    for (s, p), r in g2.iterrows():
        k = (int(s), int(p))
        if k not in ts_min or r['min'] < ts_min[k]:
            ts_min[k] = r['min']
        if k not in ts_max or r['max'] > ts_max[k]:
            ts_max[k] = r['max']
    # 练习数
    g3 = chunk.groupby(['student', 'part'])['exercise'].nunique()
    for (s, p), v in g3.items():
        k = (int(s), int(p))
        ex_n[k] = ex_n.get(k, 0) + int(v)
    # 单调性抽样：每块取前 2000 行有序
    sub = chunk[['student', 'part', 'timestamp']].head(2000)
    for s, p, t in sub.itertuples(index=False):
        part_seq[int(s)].append((t, int(p)))
    chunk_i += 1
    print(f'  chunk {chunk_i} done, students so far={len(students)}', flush=True)

print(f'  原始 part 取值: {sorted(raw_parts)}')
print(f'  学生数(日志): {len(students)}')

# ---------------------------------------------------------------- 标签
pdf = pd.read_csv(PASS)
pdf['student'] = pdf['student'].astype(int)
pdf['failed'] = (~pdf['passed'].astype(bool)).astype(int)
label_map = dict(zip(pdf['student'], pdf['failed']))
print(f'  passed.csv: {len(pdf)} 人, failed=1 {int(pdf["failed"].sum())}')

# ---------------------------------------------------------------- 每人每 part 特征
# feats[(s,p)] = 7 维事件计数 + total + ex_n + span_hours
feats = {}
for k, d in cnt.items():
    v = np.array([d.get(e, 0) for e in EV], dtype=float)
    span = (ts_max[k] - ts_min[k]).total_seconds() / 3600.0
    feats[k] = np.concatenate([v, [v.sum(), float(ex_n.get(k, 0)), span]])

# 每个学生的 part 集合 / 主 part（argmax 事件数）/ 最大 part / 事件序列（按时间）
by_student = defaultdict(list)
for (s, p) in feats:
    by_student[s].append(p)

seq_ok = None
mono_bad = 0
for s, seq in part_seq.items():
    parts = [p for _, p in sorted(seq)]
    # 检查是否存在回退（时间上 part 变小）
    dec = sum(1 for a, b in zip(parts, parts[1:]) if b < a)
    if dec > 0:
        mono_bad += 1

rows = []
for s, ps in by_student.items():
    ps = sorted(ps)
    main_p = max(ps, key=lambda p: feats[(s, p)][7])  # argmax total
    max_p = max(ps)
    tot = sum(feats[(s, p)][7] for p in ps)
    rows.append(dict(student=s, main_part=main_p, max_part=max_p, n_parts=len(ps),
                     total=tot, label=label_map.get(s, -1)))
S = pd.DataFrame(rows).sort_values('student').reset_index(drop=True)
print(f'  有标签学生: {(S.label >= 0).sum()} / {len(S)}')

# ---------------------------------------------------------------- F1/F2
print()
print('=' * 100)
print('F1) part 时间序列单调性（"进度"是否可从事件流直接推导）')
print('=' * 100)
print(f'  抽样学生中存在 part 回退(时间上变小)的比例: {mono_bad}/{len(part_seq)}')
print('  → 回退普遍存在则"最大 part"= 进度上界（不是严格单调），仍可作进度代理')

print()
print('=' * 100)
print('F2) 进度 vs 标签（failed/passed 在 main_part 与 max_part 上的分布）')
print('=' * 100)
ct_main = pd.crosstab(S['main_part'], S['label'])
ct_main.columns = ['passed(0)', 'failed(1)']
ct_main['fail_rate'] = ct_main['failed(1)'] / ct_main.sum(axis=1)
print('  [main_part]'); print(ct_main.to_string())
ct_max = pd.crosstab(S['max_part'], S['label'])
ct_max.columns = ['passed(0)', 'failed(1)']
ct_max['fail_rate'] = ct_max['failed(1)'] / ct_max.sum(axis=1)
print('\n  [max_part 达到的最远 part]'); print(ct_max.to_string())
S.to_csv(f'{OUT}/student_progress.csv', index=False)

# ---------------------------------------------------------------- 同伴归一化工具
def deconf_matrix(p):
    """给定 part p，返回 (学生索引, 归一化后特征) —— 只保留该 part 有活动的学生"""
    idx, X = [], []
    for s, r in S.iterrows():
        if r['main_part'] != p:
            continue
        if (int(s), int(p)) not in feats:
            continue
        idx.append(s)
        X.append(feats[(int(s), int(p))])
    if not X:
        return None, None
    X = np.array(X)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826 + 1e-9
    Z = (X - med) / mad
    return idx, Z


# ---------------------------------------------------------------- F3：阶段现场判别力
print()
print('=' * 100)
print('F3) 【判决点】在"掉队现场 part"内部，个体行为是否还有判别力？')
print('=' * 100)
print('  设定：每个学生取其 max_part（他最后停留的 part）作为"现场"')
print('  对每个样本量足够的 part，用同伴归一化特征预测 failed → within-part AUC')
print()

rows_auc = []
for p in sorted(S['max_part'].unique()):
    sub = S[S['max_part'] == p]
    if len(sub) < 12 or sub['label'].nunique() < 2:
        print(f'  part {p}: n={len(sub)}, 跳过（样本不足/单一标签）')
        continue
    idx, Z = deconf_matrix(p)
    if idx is None:
        continue
    lab = np.array([S.loc[i, 'label'] for i in idx])
    keep = np.array([S.loc[i, 'max_part'] == p for i in idx])
    idx = np.array(idx)[keep]
    Z, lab = Z[keep], lab[keep]
    if len(set(lab)) < 2 or len(lab) < 12:
        continue
    # 5-fold OOF RF
    from sklearn.model_selection import StratifiedKFold
    oof = np.zeros(len(lab))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Z, lab):
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                    class_weight='balanced', random_state=0, n_jobs=-1)
        rf.fit(Z[tr], lab[tr])
        oof[va] = rf.predict_proba(Z[va])[:, 1]
    auc = roc_auc_score(lab, oof)
    rows_auc.append(dict(part=int(p), n=int(len(lab)), pos=int(lab.sum()),
                         fail_rate=float(lab.mean()), within_part_auc=float(auc)))
    print(f'  part {p}: n={len(lab):4d}  挂科率={lab.mean():.3f}  **within-part AUC = {auc:.4f}**')

F3 = pd.DataFrame(rows_auc)
print()
if len(F3):
    w = F3['n'] / F3['n'].sum()
    pooled = float((F3['within_part_auc'] * w).sum())
    print(f'  加权 pooled within-part AUC = {pooled:.4f}   (0.5 = 无个体信号)')
    print(F3.to_string(index=False))
else:
    pooled = float('nan')
    print('  无可判别 part')

# ---------------------------------------------------------------- F4：同伴数
print()
print('=' * 100)
print('F4) 每 part 同伴数（同伴参照系的可行性）')
print('=' * 100)
pcnt = S['main_part'].value_counts().sort_index()
print(pcnt.to_string())
print('  → n<20 的 part 做同伴归一化会不稳定，需收缩或回退')

# ---------------------------------------------------------------- 落盘
res = dict(f1_mono_back_ratio=mono_bad / max(len(part_seq), 1),
           f3_pooled_within_part_auc=pooled,
           f3_rows=rows_auc,
           f4_part_peers={int(k): int(v) for k, v in pcnt.items()},
           label_dist_by_main_part=ct_main.reset_index().to_dict('records'),
           label_dist_by_max_part=ct_max.reset_index().to_dict('records'))
json.dump(res, open(f'{OUT}/probe_results.json', 'w'), indent=2, ensure_ascii=False)
np.save(f'{OUT}/S.npy', S.to_numpy())
print(f'\n  → {OUT}/probe_results.json')
