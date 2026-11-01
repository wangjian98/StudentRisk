"""实验 B：验证改进方法 —— 「part 去混淆特征 + part 内排序」能否真正提升个体判别力

动机（实验 A 已证）：
  · part-prior baseline AUC=0.9135，MetaMamba-7d 的个体信号净贡献 ΔAUC=+0.0050 (P(≤0)=23.8%, 不显著)
  · 组内 AUC(grp_bal) 排名与总体 AUC 排名 15/15 完全反转 → RF 第一，MetaMamba-7d 垫底
  · 7d 特征只有 event-type one-hot；11d 里 part_id 既进输入特征又进 FiLM（双重注入）

本实验检验的改进（不需要 GPU，纯 CPU + sklearn）：
  V0  baseline  : 原始 per-student 事件计数（复现 RF 类特征）
  V1  去 part 混淆: 每个特征减去「同 part 同伴中位数」再除以 MAD（相对同伴的偏离度）
  V2  V1 + 原始计数
  V3  V1 + part one-hot（显式给 part 信息，让模型自己权衡）

评估三件套（对应改进方法 1）：
  ① 总体 AUC / F1-macro（论文现有口径）
  ② grp_bal = part 内 AUC 加权平均（个体判别力的诚实度量）
  ③ ΔAUC vs part-prior baseline（个体信号净贡献）+ bootstrap CI

判定标准：若 V1/V2/V3 的 grp_bal 显著高于 V0 且 ΔAUC 显著为正 → 改进方法成立。
"""
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

IDE_CSV = '/root/.openclaw/data-backups/IDE_logs.csv'
PASSED_CSV = '/root/.openclaw/data-backups/passed.csv'
OUT = 'experiment_B_results.json'

EVENT_TYPES = ['text_insert', 'text_remove', 'text_paste',
               'focus_gained', 'focus_lost', 'run', 'submit']
ET_IDX = {et: i for i, et in enumerate(EVENT_TYPES)}

SEEDS = [42, 123, 777]
N_SPLITS = 5
usecols = ['student', 'part', 'eventType']

t0 = time.time()
print("=" * 100)
print("B0) 加载 IDE_logs（chunked，1.3GB）...")
print("=" * 100)

passed = pd.read_csv(PASSED_CSV)
student_ids = passed['student'].values
y = (~passed['passed'].values).astype(int)   # Failed=1
sid_to_row = {s: i for i, s in enumerate(student_ids)}
n = len(student_ids)
print(f"  学生数 {n}, 挂科率 {y.mean():.4f}")

# 聚合：per (student, part) 的事件计数
agg = defaultdict(lambda: np.zeros(len(EVENT_TYPES), dtype=np.int64))
# 每个 (student, part) 的事件总数，用于取众数 part（与官方 task_ids.npy 口径一致：
# int(ev['part'].mode().iloc[0]) - 1）
part_counts = defaultdict(lambda: defaultdict(int))
rows_read = 0
for chunk in pd.read_csv(IDE_CSV, usecols=usecols, chunksize=2_000_000):
    chunk = chunk[chunk['student'].isin(sid_to_row)]
    for sid, pt, et in zip(chunk['student'].values, chunk['part'].values,
                           chunk['eventType'].values):
        r = sid_to_row[sid]
        key = (r, int(pt))
        agg[key][ET_IDX.get(et, 0)] += 1
        part_counts[r][int(pt)] += 1
    rows_read += len(chunk)
    del chunk
print(f"  聚合完成，用时 {time.time()-t0:.0f}s，命中行 {rows_read:,}，"
      f"(student,part) 组合 {len(agg)}")

# 构建 per-student 特征矩阵（把所有 part 的事件计数拼起来）
all_parts = sorted({p for _, p in agg.keys()})
print(f"  part 范围: {all_parts} (1-indexed)")
P = len(all_parts)
D = P * len(EVENT_TYPES)
# part 1..7 -> 列块索引 0..6
part_to_blk = {p: i for i, p in enumerate(all_parts)}

X_raw = np.zeros((n, D), dtype=np.float32)
for (r, p), v in agg.items():
    b = part_to_blk[p]
    X_raw[r, b * len(EVENT_TYPES): (b + 1) * len(EVENT_TYPES)] = v
X_raw = np.log1p(X_raw)

# 🔴 part_idx = 该学生**众数 part**（与官方 task_ids.npy 完全同口径）
#    修正前一版 bug：用最后一次迭代到的 part 覆盖，学生跨 part 时会取错
part_idx = np.zeros(n, dtype=np.int64)
for r, cnt in part_counts.items():
    part_idx[r] = part_to_blk[max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]]
# 与官方 task_ids 交叉校验
try:
    ref = np.load('/root/.openclaw/workspace-staging/pc-ceo_assistant/StudentRisk/'
                  'analysis/control_exp_20261028/oof/meta_mamba_7d/task_ids.npy').ravel()
    print(f"  part_idx 与官方 task_ids 一致率: {(part_idx == ref).mean()*100:.2f}% "
          f"({(part_idx == ref).sum()}/{n})")
except Exception as e:
    print(f"  (无法交叉校验 task_ids: {e})")

# part 统计（用于去混淆）：同 part 同伴的逐特征中位数 / MAD
print()
print("=" * 100)
print("B1) 构建去 part 混淆特征（V1/V2/V3）")
print("=" * 100)
X_median = np.zeros_like(X_raw)
X_mad = np.ones_like(X_raw)
for b in range(P):                     # 🔴 用**块索引** 0..P-1，与 part_idx 同一坐标系
    rows = np.where(part_idx == b)[0]
    if len(rows) == 0:
        continue
    block = X_raw[rows]
    med = np.median(block, axis=0)
    mad = np.median(np.abs(block - med), axis=0)
    X_median[rows] = med
    X_mad[rows] = np.maximum(mad, 1e-3)

X_rel = (X_raw - X_median) / X_mad          # 相对同伴的偏离度
X_rel = np.clip(X_rel, -5, 5)

part_onehot = np.zeros((n, P), dtype=np.float32)
part_onehot[np.arange(n), part_idx] = 1.0

FEATSETS = {
    'V0_raw': X_raw,
    'V1_deconf': X_rel,
    'V2_deconf+raw': np.hstack([X_rel, X_raw]),
    'V3_deconf+part1h': np.hstack([X_rel, part_onehot]),
}
for k, v in FEATSETS.items():
    print(f"  {k:18s} 维度 {v.shape[1]}")

# part-prior baseline（按 fold 无泄漏）
fold_idx = np.zeros(n, dtype=np.int64)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
for i, (_, te) in enumerate(skf.split(np.zeros(n), y)):
    fold_idx[te] = i

prior_fold = np.zeros(n)
for f_ in range(N_SPLITS):
    tr, va = fold_idx != f_, fold_idx == f_
    for b in range(P):                 # 🔴 同样用块索引
        m = tr & (part_idx == b)
        va_m = va & (part_idx == b)
        if va_m.sum() == 0:
            continue
        prior_fold[va_m] = y[m].mean() if m.sum() > 0 else y[tr].mean()
prior_fold = np.clip(prior_fold, 1e-6, 1 - 1e-6)
a_prior = roc_auc_score(y, prior_fold)
print(f"\n  part-prior baseline (fold 无泄漏) AUC={a_prior:.4f}")

valid_parts = [p for p in all_parts if y[part_idx == p].sum() >= 5 and (1 - y[part_idx == p]).sum() >= 5]
print(f"  可判别 part（正负各≥5）: {valid_parts}")


def grp_bal_auc(p, y_, part_):
    aucs, ws = [], []
    for pt in np.unique(part_):
        m = part_ == pt
        if len(set(y_[m])) > 1:
            aucs.append(roc_auc_score(y_[m], p[m]))
            ws.append(m.sum())
    if not aucs:
        return np.nan
    return float(np.average(aucs, weights=ws))


def run_featset(name, X):
    oof = np.zeros(n)
    for seed in SEEDS:
        for tr, te in skf.split(X, y):
            m = RandomForestClassifier(n_estimators=300, max_depth=None,
                                       min_samples_leaf=2, max_features='sqrt',
                                       class_weight='balanced', random_state=seed,
                                       n_jobs=-1)
            m.fit(X[tr], y[tr])
            oof[te] += m.predict_proba(X[te])[:, 1]
    oof /= len(SEEDS)
    yp = (oof >= 0.5).astype(int)
    # bootstrap ΔAUC
    rng = np.random.default_rng(7)
    d = []
    for _ in range(1500):
        b = rng.integers(0, n, n)
        if len(set(y[b])) < 2:
            continue
        d.append(roc_auc_score(y[b], oof[b]) - roc_auc_score(y[b], prior_fold[b]))
    d = np.array(d)
    return dict(featset=name, dim=int(X.shape[1]),
                auc=float(roc_auc_score(y, oof)),
                acc=float(accuracy_score(y, yp)),
                f1_macro=float(f1_score(y, yp, average='macro', zero_division=0)),
                grp_bal_auc=grp_bal_auc(oof, y, part_idx),
                delta_auc=float(a_prior and roc_auc_score(y, oof) - a_prior),
                delta_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                p_delta_le0=float((d <= 0).mean()),
                part_auc={int(p): float(roc_auc_score(y[part_idx == p], oof[part_idx == p]))
                          for p in valid_parts},
                _oof=oof)


print()
print("=" * 100)
print("B2) 训练与评估（5-fold × 3 seeds RF，各特征集）")
print("=" * 100)
results = []
for name, X in FEATSETS.items():
    r = run_featset(name, X)
    print(f"\n  [{name}] dim={r['dim']}")
    print(f"    AUC={r['auc']:.4f}  acc={r['acc']:.4f}  f1_macro={r['f1_macro']:.4f}")
    print(f"    grp_bal_auc={r['grp_bal_auc']:.4f}   ← 个体判别力")
    print(f"    ΔAUC vs part-prior = {r['delta_auc']:+.4f}  95%CI [{r['delta_ci'][0]:+.4f}, {r['delta_ci'][1]:+.4f}]"
          f"  P(≤0)={r['p_delta_le0']*100:.2f}%")
    print(f"    part 内 AUC: {r['part_auc']}")
    results.append(r)
    np.save(f'expB_oof_{name}.npy', r.pop('_oof'))

print()
print("=" * 100)
print("B3) 判定：去 part 混淆是否真的提升个体判别力？")
print("=" * 100)
base = [r for r in results if r['featset'] == 'V0_raw'][0]
print(f"  {'featset':20s} {'AUC':>8s} {'ΔAUC':>9s} {'grp_bal':>9s} {'vs V0 grp_bal':>15s}")
for r in results:
    print(f"  {r['featset']:20s} {r['auc']:8.4f} {r['delta_auc']:+9.4f} {r['grp_bal_auc']:9.4f} "
          f"{r['grp_bal_auc']-base['grp_bal_auc']:+15.4f}")
print()
best = max(results, key=lambda r: r['grp_bal_auc'])
print(f"  → 个体判别力最优: {best['featset']} (grp_bal={best['grp_bal_auc']:.4f}, "
      f"vs V0 {best['grp_bal_auc']-base['grp_bal_auc']:+.4f})")
print(f"  → 对照实验 A 的模型: rf7 grp_bal=0.7785 / meta_mamba_7d grp_bal=0.4829")

json.dump(dict(part_prior_auc=a_prior, results=results,
               valid_parts=[int(p) for p in valid_parts]),
          open(OUT, 'w'), indent=2, ensure_ascii=False)
print(f"\n  → 已写入 {OUT}，总用时 {time.time()-t0:.0f}s")
