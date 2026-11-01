"""验证 part 先验发现: 排除实现错误 / 对齐错误 / 折叠泄漏等可能解释。

检查项:
  V1 OOF probs / labels / task_ids / student_ids 的对齐是否正确
  V2 fold_idx 是否真的覆盖每个学生一次 (OOF 合法性)
  V3 part 先验 baseline 是否泄漏 (LOO 复核 + 用 fold 划分复核)
  V4 part 内 AUC 是否因样本量太小而不稳定 (bootstrap)
  V5 MetaMamba vs RF 在 part 内 (个体判别力) 的配对比较 + CI
  V6 换成 '校准后' 的阈值重算 acc/f1, 看结论是否依赖阈值 0.5
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

B = 'oof'


def load(d):
    out = {}
    for k, f in [('p', 'oof_probs.npy'), ('y', 'labels.npy'), ('sid', 'student_ids.npy'),
                 ('task', 'task_ids.npy'), ('fold', 'fold_idx.npy')]:
        try:
            a = np.load(f'{B}/{d}/{f}')
            out[k] = np.asarray(a).ravel() if a.ndim > 1 else np.asarray(a)
        except Exception:
            out[k] = None
    return out


mm7 = load('meta_mamba_7d')
rf = load('rf7')
y = mm7['y']
task = mm7['task']
n = len(y)

print("=" * 100)
print("V1) 对齐检查")
print("=" * 100)
print(f"  meta_mamba_7d: p={mm7['p'].shape} y={y.shape} sid={mm7['sid'].shape} task={task.shape} fold={mm7['fold'].shape}")
print(f"  rf7          : p={rf['p'].shape} y={rf['y'].shape} sid={rf['sid'].shape}")
print(f"  student_ids 相等? {np.array_equal(mm7['sid'], rf['sid'])}")
print(f"  labels 相等?       {np.array_equal(y, rf['y'])}")
print(f"  task_ids 取值:     {np.unique(task)}")
print(f"  sid 是否已排序/唯一? unique={len(np.unique(mm7['sid']))} n={n}")

print()
print("=" * 100)
print("V2) OOF 合法性: 每个学生恰好被预测一次 (fold_idx)")
print("=" * 100)
f = mm7['fold']
print(f"  fold 取值: {np.unique(f, return_counts=True)}")
if f is not None and len(f) == n:
    print(f"  fold_idx 与 student 顺序一致? (前20): {f[:20]}")

print()
print("=" * 100)
print("V3) part 先验 baseline 的泄漏排查")
print("=" * 100)
prior_loo = np.zeros(n)
for i in range(n):
    same = np.where(task == task[i])[0]
    same = same[same != i]
    prior_loo[i] = y[same].mean() if len(same) > 0 else y.mean()
prior_loo = np.clip(prior_loo, 1e-6, 1 - 1e-6)
print(f"  LOO part 先验 AUC = {roc_auc_score(y, prior_loo):.4f}")
# 完全不留一的 (理论上会更高 = 泄漏上界)
prior_full = np.array([y[task == t].mean() for t in task])
print(f"  不留一(有泄漏上界) AUC = {roc_auc_score(y, prior_full):.4f}")
# 全局常数
print(f"  全局常数(无任何 part 信息) AUC = {roc_auc_score(y, np.full(n, y.mean())):.4f}")
# 用 fold 划分做无泄漏的 part 先验: 用训练折的 part 先验预测验证折
if f is not None and len(f) == n:
    prior_fold = np.zeros(n)
    for fold in np.unique(f):
        tr = f != fold
        va = f == fold
        for t in np.unique(task):
            m = tr & (task == t)
            prior_fold[va & (task == t)] = y[m].mean() if m.sum() > 0 else y[tr].mean()
    print(f"  按 fold 无泄漏 part 先验 AUC = {roc_auc_score(y, prior_fold):.4f}   <-- 诚实上界")

print()
print("=" * 100)
print("V4) part 内 AUC 的稳定性 (bootstrap, 每 part)")
print("=" * 100)
valid_parts = [t for t in np.unique(task) if y[task == t].sum() >= 5 and (1 - y[task == t]).sum() >= 5]
rng = np.random.default_rng(3)
for name, M in [('meta_mamba_7d', mm7), ('rf7', rf)]:
    p = M['p']
    for t in valid_parts:
        m = task == t
        yy, pp = y[m], p[m]
        a = roc_auc_score(yy, pp)
        bs = []
        for _ in range(2000):
            b = rng.integers(0, len(yy), len(yy))
            if len(set(yy[b])) > 1:
                bs.append(roc_auc_score(yy[b], pp[b]))
        bs = np.array(bs)
        print(f"  {name:14s} part={t} n={m.sum()} fail={int(yy.sum())} AUC={a:.4f} "
              f"95%CI [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]")

print()
print("=" * 100)
print("V5) part 内个体判别力: MetaMamba-7d vs RF-7d (配对 bootstrap, 跨 2 个可判别 part)")
print("=" * 100)
diffs = []
for _ in range(3000):
    accs = []
    ok = True
    for t in valid_parts:
        m = np.where(task == t)[0]
        b = rng.choice(m, size=len(m), replace=True)
        yy = y[b]
        if len(set(yy)) < 2:
            ok = False
            break
        accs.append(roc_auc_score(yy, mm7['p'][b]) - roc_auc_score(yy, rf['p'][b]))
    if ok:
        diffs.append(np.mean(accs))
diffs = np.array(diffs)
print(f"  MetaMamba-7d − RF-7d (part 内 AUC 差, 越大越差) 均值 {diffs.mean():+.4f} "
      f"95%CI [{np.percentile(diffs,2.5):+.4f}, {np.percentile(diffs,97.5):+.4f}] "
      f"P(<=0)={(diffs<=0).mean()*100:.2f}%")

print()
print("=" * 100)
print("V6) 结论是否依赖阈值 0.5? 用最优阈值 (per-part 留一选阈值) 重算")
print("=" * 100)
for name, M in [('meta_mamba_7d', mm7), ('rf7', rf), ('meta_mamba', load('meta_mamba'))]:
    p = M['p']
    # 全局最优阈值 (事后, 仅作参考)
    ths = np.linspace(0.1, 0.9, 33)
    f1s = [f1_score(y, (p >= t).astype(int), average='macro', zero_division=0) for t in ths]
    bt = ths[int(np.argmax(f1s))]
    yp = (p >= bt).astype(int)
    print(f"  {name:14s} 最优thr={bt:.2f} f1_macro={max(f1s):.4f} acc={accuracy_score(y,yp):.4f} "
          f"AUC={roc_auc_score(y,p):.4f}")
print()
print("  注: 阈值调整不改变 AUC; AUC 才是与先验无关的判别力度量。")
