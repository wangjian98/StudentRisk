"""实验 D：验证两项尚未验证的改进方法

D1 (M1) Two-Stage Prior + Residual Scoring：
    显式把 p(y|part) 作为基线，模型只学残差。
    做法（用已有 OOF 概率，不重训）：score_resid = p_model − p_prior（对数几率空间的减法等价形式）
    看：个体判别力(grp_bal) 是否因为"显式扣除先验"而提升。

D2 (M4) 表征升级：加"行为强度"通道
    原始 7 维只有事件类型 one-hot。新增：事件间隔分箱、单位时间密度、最近k事件滑动计数。
    看：强度特征能否在去 part 混淆基础上继续提升 grp_bal。

D3 (M3 的可验证部分) FiLM 位置/结构的替代方案 —— 用"part 内偏离度"作为唯一 part 信息的载体
    比较：FiLM 注入 part id  vs  直接给 part one-hot  vs  只给去混淆特征(不给 part 身份)
    看"是否需要单独的任务调制通路"。

数据：本地 IDE_logs.csv（1.3GB）
"""
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

IDE_CSV = '/root/.openclaw/data-backups/IDE_logs.csv'
PASSED_CSV = '/root/.openclaw/data-backups/passed.csv'
EVENT_TYPES = ['text_insert', 'text_remove', 'text_paste',
               'focus_gained', 'focus_lost', 'run', 'submit']
ET_IDX = {et: i for i, et in enumerate(EVENT_TYPES)}
SEEDS = [42, 123, 777]
N_SPLITS = 5
t0 = time.time()

passed = pd.read_csv(PASSED_CSV)
student_ids = passed['student'].values
y = (~passed['passed'].values).astype(int)
sid_to_row = {s: i for i, s in enumerate(student_ids)}
n = len(student_ids)

print("=" * 100)
print("D0) 聚合 IDE_logs：事件计数 + 时间戳序列（为强度特征）")
print("=" * 100)
agg = defaultdict(lambda: np.zeros(len(EVENT_TYPES), dtype=np.int64))
part_counts = defaultdict(lambda: defaultdict(int))
ts_seq = defaultdict(list)
rows_read = 0
for chunk in pd.read_csv(IDE_CSV, usecols=['student', 'part', 'eventType', 'timestamp'],
                         chunksize=2_000_000):
    chunk = chunk[chunk['student'].isin(sid_to_row)]
    for sid, pt, et, ts in zip(chunk['student'].values, chunk['part'].values,
                               chunk['eventType'].values, chunk['timestamp'].values):
        r = sid_to_row[sid]
        agg[(r, int(pt))][ET_IDX.get(et, 0)] += 1
        part_counts[r][int(pt)] += 1
        ts_seq[r].append(ts)
    rows_read += len(chunk)
    del chunk
print(f"  聚合完成 {time.time()-t0:.0f}s，命中行 {rows_read:,}")

all_parts = sorted({p for _, p in agg.keys()})
P = len(all_parts)
part_to_blk = {p: i for i, p in enumerate(all_parts)}
D = P * len(EVENT_TYPES)
X_raw = np.zeros((n, D), dtype=np.float32)
for (r, p), v in agg.items():
    b = part_to_blk[p]
    X_raw[r, b * len(EVENT_TYPES): (b + 1) * len(EVENT_TYPES)] = v
X_raw = np.log1p(X_raw)

part_idx = np.zeros(n, dtype=np.int64)
for r, cnt in part_counts.items():
    part_idx[r] = part_to_blk[max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]]
ref = np.load('oof/meta_mamba_7d/task_ids.npy').ravel()
print(f"  part_idx 与官方 task_ids 一致率 {(part_idx==ref).mean()*100:.2f}%")

# 强度特征（M4）：每个学生的强度统计
print()
print("=" * 100)
print("D0b) 构造 M4 强度特征（事件间隔 / 密度 / 局部活跃度 / 错误-修正模式）")
print("=" * 100)
X_strong = np.zeros((n, 8), dtype=np.float32)
for r in range(n):
    ts = ts_seq.get(r)
    if not ts:
        continue
    ts = np.sort(np.array(ts, dtype='datetime64[s]').astype('int64'))
    if len(ts) < 3:
        continue
    d = np.diff(ts).astype(float)
    d = np.maximum(d, 1.0)
    span = max(ts[-1] - ts[0], 1.0)
    X_strong[r, 0] = np.log1p(d.mean())          # 平均事件间隔
    X_strong[r, 1] = np.log1p(np.median(d))       # 中位间隔
    X_strong[r, 2] = np.log1p(d.std())            # 间隔波动（不规律程度）
    X_strong[r, 3] = len(ts) / (span / 3600.0)   # 每小时事件密度
    X_strong[r, 4] = np.log1p(len(ts))            # 总事件量
    X_strong[r, 5] = np.log1p((d < 60).sum())     # 短间隔事件数（连续工作）
    X_strong[r, 6] = np.log1p((d > 1800).sum())   # 长间隔数（走神/中断）
    X_strong[r, 7] = np.log1p((d > 3600).sum())   # 超长中断
X_strong = np.log1p(np.maximum(X_strong, 0))
print(f"  强度特征维度 {X_strong.shape[1]}，样例 student0 = {np.round(X_strong[0],3)}")

# 同伴归一化（对全部数值特征统一做）
X_median = np.zeros_like(X_raw)
X_mad = np.ones_like(X_raw)
for b in range(P):
    rows = np.where(part_idx == b)[0]
    if len(rows) == 0:
        continue
    blk = X_raw[rows]
    med = np.median(blk, axis=0)
    mad = np.median(np.abs(blk - med), axis=0)
    X_median[rows] = med
    X_mad[rows] = np.maximum(mad, 1e-3)
X_rel = np.clip((X_raw - X_median) / X_mad, -5, 5)

# 强度特征也做同伴归一化
S_med = np.zeros_like(X_strong)
S_mad = np.ones_like(X_strong)
for b in range(P):
    rows = np.where(part_idx == b)[0]
    if len(rows) == 0:
        continue
    blk = X_strong[rows]
    med = np.median(blk, axis=0)
    mad = np.median(np.abs(blk - med), axis=0)
    S_med[rows] = med
    S_mad[rows] = np.maximum(mad, 1e-3)
S_rel = np.clip((X_strong - S_med) / S_mad, -5, 5)

part_onehot = np.zeros((n, P), dtype=np.float32)
part_onehot[np.arange(n), part_idx] = 1.0

FEATSETS = {
    'V3_deconf+part1h (基线)': np.hstack([X_rel, part_onehot]),
    'D2_deconf+strong+part1h': np.hstack([X_rel, S_rel, part_onehot]),
    'D3_deconf+strong (无part身份)': np.hstack([X_rel, S_rel]),
    'D3b_deconf+part1h (无强度)': np.hstack([X_rel, part_onehot]),
}

fold_idx = np.zeros(n, dtype=np.int64)
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
for i, (_, te) in enumerate(skf.split(np.zeros(n), y)):
    fold_idx[te] = i
prior_fold = np.zeros(n)
for f_ in range(N_SPLITS):
    tr, va = fold_idx != f_, fold_idx == f_
    for b in range(P):
        m = tr & (part_idx == b)
        va_m = va & (part_idx == b)
        if va_m.sum() == 0:
            continue
        prior_fold[va_m] = y[m].mean() if m.sum() > 0 else y[tr].mean()
prior_fold = np.clip(prior_fold, 1e-6, 1 - 1e-6)
a_prior = roc_auc_score(y, prior_fold)
valid = [b for b in range(P) if y[part_idx == b].sum() >= 5 and (1 - y[part_idx == b]).sum() >= 5]
print(f"\n  part-prior (fold 无泄漏) AUC={a_prior:.4f}   可判别 part={valid}")


def grp(p):
    a, w = [], []
    for b in np.unique(part_idx):
        m = part_idx == b
        if len(set(y[m])) > 1:
            a.append(roc_auc_score(y[m], p[m]))
            w.append(m.sum())
    return float(np.average(a, weights=w))


def run(name, X):
    oof = np.zeros(n)
    for seed in SEEDS:
        for tr, te in skf.split(X, y):
            m = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                       max_features='sqrt', class_weight='balanced',
                                       random_state=seed, n_jobs=-1)
            m.fit(X[tr], y[tr])
            oof[te] += m.predict_proba(X[te])[:, 1]
    oof /= len(SEEDS)
    return oof


print()
print("=" * 100)
print("D1+D2+D3) 各改进变体的表现")
print("=" * 100)
res = {}
for name, X in FEATSETS.items():
    oof = run(name, X)
    res[name] = oof
    yp = (oof >= 0.5).astype(int)
    print(f"  {name:32s} dim={X.shape[1]:3d} AUC={roc_auc_score(y,oof):.4f} "
          f"f1m={f1_score(y,yp,average='macro',zero_division=0):.4f} "
          f"grp_bal={grp(oof):.4f} Δ={roc_auc_score(y,oof)-a_prior:+.4f}")

# ---- D1: 显式两阶段残差打分 ----
print()
print("=" * 100)
print("D1 (M1) 显式扣除 part 先验后重打分（Two-Stage Residual Scoring）")
print("=" * 100)
base = res['V3_deconf+part1h (基线)']
lp = np.log(np.clip(base, 1e-6, 1 - 1e-6) / np.clip(1 - base, 1e-6, 1))
lprior = np.log(np.clip(prior_fold, 1e-6, 1 - 1e-6) / np.clip(1 - prior_fold, 1e-6, 1))
print(f"  基线 grp_bal={grp(base):.4f}  AUC={roc_auc_score(y,base):.4f}")
for w in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
    s = 1 / (1 + np.exp(-(lp - w * lprior)))
    print(f"    w={w:4.2f} (减去 w×先验logit)  AUC={roc_auc_score(y,s):.4f}  grp_bal={grp(s):.4f}")
print("  解读：grp_bal 随 w 上升 = 显式扣除先验后，个体判别力提升；总体 AUC 下降 = 牺牲总体精度换个体分辨力")
rng_d1 = np.random.default_rng(3)
for w in [0.25, 0.5, 1.0]:
    s = 1 / (1 + np.exp(-(lp - w * lprior)))
    dd1 = []
    for _ in range(1500):
        b = rng_d1.integers(0, n, n)
        if len(set(y[b])) < 2:
            continue
        dd1.append(roc_auc_score(y[b], s[b]) - roc_auc_score(y[b], base[b]))
    dd1 = np.array(dd1)
    print(f"    w={w}: ΔAUC vs 基线 = {dd1.mean():+.4f}  95%CI [{np.percentile(dd1,2.5):+.4f}, {np.percentile(dd1,97.5):+.4f}]"
          f"  P(Δ<=0)={(dd1<=0).mean()*100:.2f}%")

# ---- D2: 强度特征是否带来提升 ----
print()
print("=" * 100)
print("D2 (M4) 强度特征的增量（配对 bootstrap，grp_bal + part 内 AUC）")
print("=" * 100)
d2 = res['D2_deconf+strong+part1h']
b3 = res['D3b_deconf+part1h (无强度)']
rng = np.random.default_rng(11)
dd = []
for _ in range(2000):
    b = rng.integers(0, n, n)
    if len(set(y[b])) < 2:
        continue
    dd.append(roc_auc_score(y[b], d2[b]) - roc_auc_score(y[b], b3[b]))
dd = np.array(dd)
print(f"  D2 − D3b (总体AUC): Δ={dd.mean():+.4f} 95%CI [{np.percentile(dd,2.5):+.4f}, {np.percentile(dd,97.5):+.4f}] "
      f"P(≤0)={(dd<=0).mean()*100:.2f}%")
print(f"  grp_bal: D2={grp(d2):.4f} vs D3b={grp(b3):.4f}  Δ={grp(d2)-grp(b3):+.4f}")
for b_ in valid:
    m = part_idx == b_
    print(f"    part{b_} 内 AUC: D2={roc_auc_score(y[m],d2[m]):.4f} vs D3b={roc_auc_score(y[m],b3[m]):.4f}")

# ---- D3: 是否还需要单独的任务调制通路 ----
print()
print("=" * 100)
print("D3 (M3) 是否还需要'单独的任务调制通路'?")
print("=" * 100)
d3n = res['D3_deconf+strong (无part身份)']
print(f"  去掉 part one-hot 后: AUC={roc_auc_score(y,d3n):.4f} grp_bal={grp(d3n):.4f}")
print(f"  保留 part one-hot   : AUC={roc_auc_score(y,d2):.4f} grp_bal={grp(d2):.4f}")
dd3 = []
for _ in range(2000):
    b = rng.integers(0, n, n)
    if len(set(y[b])) < 2:
        continue
    dd3.append(roc_auc_score(y[b], d2[b]) - roc_auc_score(y[b], d3n[b]))
dd3 = np.array(dd3)
print(f"  ΔAUC(有part身份 − 无part身份) = {dd3.mean():+.4f} 95%CI [{np.percentile(dd3,2.5):+.4f}, {np.percentile(dd3,97.5):+.4f}] P(≤0)={(dd3<=0).mean()*100:.2f}%")
print(f"  grp_bal Δ = {grp(d2)-grp(d3n):+.4f}")

np.save('expD_oof_D2.npy', d2)
np.save('expD_oof_D3n.npy', d3n)
json.dump({k: dict(auc=float(roc_auc_score(y, v)), grp_bal=grp(v),
                   f1m=float(f1_score(y, (v >= 0.5).astype(int), average='macro', zero_division=0)))
           for k, v in res.items()}, open('experiment_D_results.json', 'w'),
          indent=2, ensure_ascii=False)
print(f"\n  总用时 {time.time()-t0:.0f}s → experiment_D_results.json")
