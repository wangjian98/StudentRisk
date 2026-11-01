"""跨模型检验: 各基线的 AUC 里, 有多少来自 part 级先验, 有多少来自个体事件序列?

若所有模型 (含 RF/LSTM 基线) 的 AUC 都远高于 part-prior baseline,
则 "MetaMamba 显著优于基线" 这一结论可能是**基线共享的 part 先验红利**,
而非 MetaMamba 架构本身的优势。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

BASE = 'oof'
DIRS = ['meta_mamba_7d', 'rf7', 'lstm_7d', 'bilstm_7d', 'attention_7d', 'meta_mamba']

# part 定义: 用 meta_mamba_7d 的 task_ids 作为权威 part 映射
task = np.load(f'{BASE}/meta_mamba_7d/task_ids.npy')
sids = np.load(f'{BASE}/meta_mamba_7d/student_ids.npy')
labels = np.load(f'{BASE}/meta_mamba_7d/labels.npy')
task = np.asarray(task).ravel()
y = np.asarray(labels).ravel()
print(f"n={len(y)}  task_ids shape={np.asarray(task).shape}  unique tasks={np.unique(task)}")
print(f"label distribution: {np.bincount(y.astype(int))}  fail_rate={y.mean():.4f}\n")


def part_prior_loo(y, part):
    n = len(y)
    prior = np.zeros(n)
    for i in range(n):
        same = np.where(part == part[i])[0]
        same = same[same != i]
        prior[i] = y[same].mean() if len(same) > 0 else y.mean()
    return np.clip(prior, 1e-6, 1 - 1e-6)


prior = part_prior_loo(y, task)
a_prior = roc_auc_score(y, prior)
print(f"part-prior baseline (LOO, 无个体信息): AUC={a_prior:.4f}  acc={accuracy_score(y,(prior>=0.5).astype(int)):.4f}  "
      f"f1m={f1_score(y,(prior>=0.5).astype(int),average='macro',zero_division=0):.4f}")
print(f"  part 挂科率先验: { {int(p): round(y[task==p].mean(),4) for p in np.unique(task)} }\n")

rows = []
for d in DIRS:
    try:
        p = np.load(f'{BASE}/{d}/oof_probs.npy').ravel()
        yy = np.load(f'{BASE}/{d}/labels.npy').ravel()
    except Exception as e:
        print(f"{d:16s} 跳过: {e}")
        continue
    if len(p) != len(y):
        print(f"{d:16s} 跳过: 长度不一致 {len(p)} vs {len(y)}")
        continue
    yp = (p >= 0.5).astype(int)
    a = roc_auc_score(yy, p)
    rows.append(dict(model=d, auc=a, acc=accuracy_score(yy, yp),
                     f1m=f1_score(yy, yp, average='macro', zero_division=0),
                     f1_pass=f1_score(yy, yp, pos_label=0, zero_division=0),
                     f1_fail=f1_score(yy, yp, pos_label=1, zero_division=0)))

R = pd.DataFrame(rows).sort_values('auc', ascending=False)
print("=" * 118)
print("N) 跨模型: AUC 中多少来自 part 先验 (a_prior = %.4f)" % a_prior)
print("=" * 118)
R['auc_gain_vs_prior'] = R['auc'] - a_prior
R['pct_of_above_random'] = (R['auc_gain_vs_prior'] / (R['auc'] - 0.5) * 100).round(1)
print(R.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print()
print("=" * 118)
print("O) 关键: 基线之间 AUC 差距 是否显著大于 '超出 part 先验后' 的差距")
print("=" * 118)
best_base = R[R.model != 'meta_mamba_7d'].iloc[0]
mm = R[R.model == 'meta_mamba_7d'].iloc[0]
print(f"  最强基线: {best_base['model']:16s} AUC={best_base['auc']:.4f}  超出 part 先验 {best_base['auc_gain_vs_prior']:+.4f}")
print(f"  MetaMamba-7d            AUC={mm['auc']:.4f}  超出 part 先验 {mm['auc_gain_vs_prior']:+.4f}")
print(f"  两者总差距 {mm['auc']-best_base['auc']:+.4f}  →  扣除 part 先验后 {mm['auc_gain_vs_prior']-best_base['auc_gain_vs_prior']:+.4f}")
print(f"  → part 先验贡献了总差距的 "
      f"{1 - (mm['auc_gain_vs_prior']-best_base['auc_gain_vs_prior'])/(mm['auc']-best_base['auc']):.1%}"
      if abs(mm['auc'] - best_base['auc']) > 1e-9 else "")

# 配对 bootstrap: MetaMamba-7d vs 最强基线, 以及各自 vs part prior
rng = np.random.default_rng(11)
pm = np.load(f'{BASE}/meta_mamba_7d/oof_probs.npy').ravel()
pb = np.load(f'{BASE}/{best_base["model"]}/oof_probs.npy').ravel()
d_mm_prior, d_base_prior, d_mm_base = [], [], []
for _ in range(3000):
    b = rng.integers(0, len(y), len(y))
    if len(set(y[b])) < 2:
        continue
    a_mm = roc_auc_score(y[b], pm[b])
    a_bs = roc_auc_score(y[b], pb[b])
    a_pr = roc_auc_score(y[b], prior[b])
    d_mm_prior.append(a_mm - a_pr)
    d_base_prior.append(a_bs - a_pr)
    d_mm_base.append(a_mm - a_bs)
for nm, arr in [('MetaMamba-7d − part_prior', d_mm_prior),
                (f'{best_base["model"]} − part_prior', d_base_prior),
                ('MetaMamba-7d − 最强基线', d_mm_base)]:
    a = np.array(arr)
    lo, hi = np.percentile(a, [2.5, 97.5])
    print(f"  {nm:38s} 均值 {a.mean():+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  P(<=0)={(a<=0).mean()*100:.2f}%")

print()
print("=" * 118)
print("P) 各模型在 part 内的个体判别力 (只取正负样本各>=5 的 part)")
print("=" * 118)
valid_parts = [p for p in np.unique(task) if y[task == p].sum() >= 5 and (1 - y[task == p]).sum() >= 5]
print(f"  可判别 part: {valid_parts}")
for d in DIRS:
    try:
        p = np.load(f'{BASE}/{d}/oof_probs.npy').ravel()
    except Exception:
        continue
    aucs = []
    for pt in valid_parts:
        m = task == pt
        if len(set(y[m])) > 1:
            aucs.append(roc_auc_score(y[m], p[m]))
    if aucs:
        print(f"  {d:16s} part 内 AUC: {[round(a,4) for a in aucs]}  均值 {np.mean(aucs):.4f}")
