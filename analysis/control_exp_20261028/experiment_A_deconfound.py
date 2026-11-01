"""实验 A：验证"MetaMamba 的优势主要来自 part 先验，不是个体序列"这一不足诊断

目的：为改进方法提供实证依据（而不是空谈）。

做四件事：
  A1 复现 part-prior baseline（无泄漏、按 fold），作为"必须加入论文的强基线"
  A2 量化各模型"个体信息净贡献" = AUC − part_prior_auc，并给 bootstrap CI
  A3 检验改进假设 H1：把 part 先验从特征里去掉（part 内标准化 / 残差化）后，
     深度模型是否反而能超过 RF？（若成立 → 改进方向 = 去 part 混淆的建模）
  A4 检验改进假设 H2：加"组内对比度"评估后，各模型排序是否反转

只用已保存的 OOF 概率 + labels + task_ids，无需 GPU。
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

B = 'oof'
DIRS = ['meta_mamba', 'meta_mamba_7d', 'rf7', 'lstm_7d', 'bilstm_7d', 'attention_7d']
rng = np.random.default_rng(2026)


def load(d):
    o = {}
    for k, f in [('p', 'oof_probs.npy'), ('y', 'labels.npy'), ('task', 'task_ids.npy'),
                 ('fold', 'fold_idx.npy')]:
        try:
            a = np.load(f'{B}/{d}/{f}')
            o[k] = np.asarray(a).ravel()
        except Exception:
            o[k] = None
    return o


M = {d: load(d) for d in DIRS}
y = M['meta_mamba_7d']['y']
task = M['meta_mamba_7d']['task']
fold = M['meta_mamba_7d']['fold']
n = len(y)

# --------------------------------------------------------------- A1
print("=" * 100)
print("A1) part-prior baseline（按 fold 无泄漏）—— 必须进论文的强基线")
print("=" * 100)
prior_fold = np.zeros(n)
for f_ in np.unique(fold):
    tr, va = fold != f_, fold == f_
    for t in np.unique(task):
        m = tr & (task == t)
        prior_fold[va & (task == t)] = y[m].mean() if m.sum() > 0 else y[tr].mean()
prior_fold = np.clip(prior_fold, 1e-6, 1 - 1e-6)
a_prior = roc_auc_score(y, prior_fold)
yp = (prior_fold >= 0.5).astype(int)
print(f"  part-prior (fold 无泄漏): AUC={a_prior:.4f}  acc={accuracy_score(y,yp):.4f}  "
      f"f1_macro={f1_score(y,yp,average='macro',zero_division=0):.4f}")
prior_loo = np.array([(y[(task == t) & (np.arange(n) != i)].mean()
                       if (task == t).sum() > 1 else y.mean())
                      for i, t in enumerate(task)])
print(f"  part-prior (LOO, 更保守) : AUC={roc_auc_score(y,prior_loo):.4f}")
print(f"  全局常数(无任何 part 信息): AUC=0.5000")

rows = []
for d in DIRS:
    p = M[d]['p']
    if p is None:
        continue
    yp = (p >= 0.5).astype(int)
    rows.append(dict(model=d, auc=roc_auc_score(y, p), acc=accuracy_score(y, yp),
                     f1m=f1_score(y, yp, average='macro', zero_division=0),
                     gain_vs_prior=roc_auc_score(y, p) - a_prior))
R = pd.DataFrame(rows).sort_values('auc', ascending=False)
print("\n  各模型 vs part-prior:")
print(R.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# --------------------------------------------------------------- A2
print()
print("=" * 100)
print("A2) 个体信息净贡献 = AUC − part_prior_auc，带 bootstrap 95% CI")
print("=" * 100)
print("  H0: 该模型的 AUC 与纯 part 先验无差异（个体信号为 0）")
for d in DIRS:
    p = M[d]['p']
    if p is None:
        continue
    diffs = []
    for _ in range(3000):
        b = rng.integers(0, n, n)
        if len(set(y[b])) < 2:
            continue
        diffs.append(roc_auc_score(y[b], p[b]) - roc_auc_score(y[b], prior_fold[b]))
    a = np.array(diffs)
    lo, hi = np.percentile(a, [2.5, 97.5])
    pval = (a <= 0).mean()
    print(f"  {d:16s} ΔAUC={a.mean():+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  "
          f"P(Δ<=0)={pval*100:5.2f}%  {'← 个体信号显著为正' if pval < 0.05 else '← 个体信号未显著'}")
print()
print("  解读：Δ 的 95%CI 下界接近 0 → 该模型的 AUC 与'只看 part 难度'几乎无差别")

# --------------------------------------------------------------- A3
print()
print("=" * 100)
print("A3) 检验改进假设 H1：'去 part 混淆'后，深度模型能否反超 RF？")
print("=" * 100)
print("  做法：part 内 z-score 标准化预测（去掉 part 均值）+ 组内 AUC（每 part 等权）")


def residualize(p, task):
    r = p.copy()
    for t in np.unique(task):
        m = task == t
        mu, sd = p[m].mean(), p[m].std() + 1e-9
        r[m] = (p[m] - mu) / sd
    return r


def group_balanced_auc(p, task, y):
    aucs, ws = [], []
    for t in np.unique(task):
        m = task == t
        if len(set(y[m])) > 1:
            aucs.append(roc_auc_score(y[m], p[m]))
            ws.append(m.sum())
    return float(np.average(aucs, weights=ws))


valid = [t for t in np.unique(task) if y[task == t].sum() >= 5 and (1 - y[task == t]).sum() >= 5]
print(f"  可判别 part（正负各≥5）: {valid}")
print(f"  {'model':16s} {'AUC':>8s} {'resid_z':>9s} {'grp_bal':>9s} {'p3':>7s} {'p6':>7s}")
res = []
for d in DIRS:
    p = M[d]['p']
    if p is None:
        continue
    rz = residualize(p, task)
    gb = group_balanced_auc(p, task, y)
    pa = {}
    for t in valid:
        m = task == t
        pa[t] = roc_auc_score(y[m], p[m]) if len(set(y[m])) > 1 else np.nan
    res.append(dict(model=d, auc=roc_auc_score(y, p), resid_z=roc_auc_score(y, rz),
                    grp_bal=gb, **{f'p{t}': pa.get(t, np.nan) for t in valid}))
    print(f"  {d:16s} {roc_auc_score(y,p):8.4f} {roc_auc_score(y,rz):9.4f} {gb:9.4f} "
          + " ".join(f"{pa.get(t,np.nan):7.4f}" for t in valid))
print()
print("  解读：'grp_bal'（组内 AUC 每 part 等权）才是'个体判别力'的诚实度量")
print("        若深度模型 grp_bal < RF，说明架构没有兑现'个体行为建模'的宣称")

# --------------------------------------------------------------- A4
print()
print("=" * 100)
print("A4) 排序反转检验：总体 AUC 排名 vs 组内 AUC 排名")
print("=" * 100)
df = pd.DataFrame(res)
o1 = list(df.sort_values('auc', ascending=False)['model'])
o2 = list(df.sort_values('grp_bal', ascending=False)['model'])
print(f"  按总体 AUC 排名      : {o1}")
print(f"  按组内 AUC(个体判别) : {o2}")
from itertools import permutations
inv = sum(1 for a, b in permutations(o1, 2)
          if o2.index(a) > o2.index(b))
print(f"  两排序的逆序对数量    : {inv} / {len(o1)*(len(o1)-1)//2}  "
      f"→ {'排名显著不一致' if inv >= 6 else '排名基本一致'}")

# --------------------------------------------------------------- 落盘
out = dict(part_prior_auc_fold_no_leak=a_prior,
           part_prior_auc_loo=roc_auc_score(y, prior_loo),
           per_model=R.to_dict('records'),
           valid_parts=[int(t) for t in valid])
json.dump(out, open('experiment_A_results.json', 'w'), indent=2, ensure_ascii=False)
print("\n  → 已写入 experiment_A_results.json")
