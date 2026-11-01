"""决定性检验: MetaMamba 的判别力到底来自"学生个体的 IDE 行为序列"还是"part 级先验"?

方法 (全部 leave-one-out, 无泄漏):
  1) part-prior baseline: 只用每个 part 的挂科率先验 (留一法), 不看任何个体事件
  2) part-mean baseline : 模型输出的 part 内均值 (去掉个体信息)
  3) part-residual       : 模型输出 - part 内均值 (只保留个体信息)
  4) 与真实模型 AUC 对比, 看性能主要来自哪一部分

判定: 若 part-prior baseline 的 AUC 已接近模型 AUC, 说明性能主要来自 part 级先验。
"""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

pred = pd.read_csv('results/concat_dummy_task_id/per_student_predictions.csv')
t = pred[(pred.conditioning == 'film') & (pred.task_id_used == 'true')].sort_values('student_idx').reset_index(drop=True)
y = t['true_label'].values
p_model = t['pred_prob'].values
part = t['true_part'].values
n = len(y)
print(f"n={n}, 挂科率={y.mean():.4f}, part 数={len(np.unique(part))}\n")

# ---------- 1) part-prior baseline (leave-one-out) ----------
prior = np.zeros(n)
for i in range(n):
    mask = part != part[i]          # 留一: 用其他 part 的学生? -> 不, 应该同 part 其他学生
    # 同 part 内留一
    same = (part == part[i])
    same[i] = False
    prior[i] = y[same].mean() if same.sum() > 0 else y.mean()
prior = np.clip(prior, 1e-6, 1 - 1e-6)

# ---------- 2) part-mean of model output ----------
pm = np.zeros(n)
for i in range(n):
    same = np.where(part == part[i])[0]
    same = same[same != i]
    pm[i] = p_model[same].mean() if len(same) > 0 else p_model.mean()
residual = p_model - pm

def rep(name, p):
    yp = (p >= 0.5).astype(int)
    print(f"{name:34s} AUC={roc_auc_score(y, p):.4f}  acc={accuracy_score(y, yp):.4f}  "
          f"f1_macro={f1_score(y, yp, average='macro', zero_division=0):.4f}  "
          f"f1_pass={f1_score(y, yp, pos_label=0, zero_division=0):.4f}  "
          f"f1_fail={f1_score(y, yp, pos_label=1, zero_division=0):.4f}")

print("=" * 105)
print("K) 决定性检验: 性能来自个体序列, 还是 part 级先验?")
print("=" * 105)
rep("① 模型完整输出", p_model)
rep("② 仅 part 先验 (留一, 无个体信息)", prior)
rep("③ 模型 part 内均值 (去个体)", pm)
rep("④ 模型个体残差 (去 part 均值)", residual + 0.5)
print()
print("解释:")
print("  ② 若 AUC 已很高 -> 性能主要来自 part 级挂科先验, 个体事件序列贡献有限")
print("  ④ 若 AUC 接近 0.5 -> 模型去掉 part 信息后就没有个体判别力")

print()
print("=" * 105)
print("L) 各 part 的可判别性体检 (正负样本对比度)")
print("=" * 105)
rows = []
for pt in sorted(np.unique(part)):
    m = part == pt
    yy, pp = y[m], p_model[m]
    npos, nneg = int(yy.sum()), int((1 - yy).sum())
    auc = roc_auc_score(yy, pp) if len(set(yy)) > 1 else np.nan
    yp = (pp >= 0.5).astype(int)
    # 全判多数类的准确率 = 基线
    maj_acc = max(yy.mean(), 1 - yy.mean())
    rows.append(dict(part=pt, n=len(yy), n_fail=npos, n_pass=nneg,
                     base_rate=round(yy.mean(), 4), model_acc=round((yp == yy).mean(), 4),
                     trivial_acc=round(maj_acc, 4),
                     gain=round((yp == yy).mean() - maj_acc, 4),
                     auc_in_part=round(auc, 4) if not np.isnan(auc) else None,
                     pred_pos=round(yp.mean(), 4)))
R = pd.DataFrame(rows)
print(R.to_string(index=False))
print()
valid = R[R.n_fail >= 5][R.n_pass >= 5]
print(f"只统计正负样本各≥5 的 part (可判别): part {list(valid.part)}")
print(f"  这些 part 的 part 内 AUC: {list(valid.auc_in_part)}  → 均值 {valid.auc_in_part.mean():.4f}")
print(f"  这些 part 的 acc 增益 (模型 - 多数类基线): {list(valid.gain)}  → 均值 {valid.gain.mean():+.4f}")

print()
print("=" * 105)
print("M) 结论量化: 模型 vs part-prior baseline")
print("=" * 105)
a_m = roc_auc_score(y, p_model)
a_p = roc_auc_score(y, prior)
print(f"  模型 AUC            = {a_m:.4f}")
print(f"  part-prior AUC      = {a_p:.4f}")
print(f"  差值 (个体信息的净贡献) = {a_m - a_p:+.4f}")
print(f"  → 个体信息解释了 {(a_m - a_p) / (a_m - 0.5) * 100:.1f}% 的超越随机的空间")

# bootstrap 差值置信区间
rng = np.random.default_rng(7)
diffs = []
for _ in range(2000):
    b = rng.integers(0, n, n)
    if len(set(y[b])) < 2:
        continue
    diffs.append(roc_auc_score(y[b], p_model[b]) - roc_auc_score(y[b], prior[b]))
diffs = np.array(diffs)
print(f"  bootstrap 差值 95% CI = [{np.percentile(diffs,2.5):.4f}, {np.percentile(diffs,97.5):.4f}]")
print(f"  差值<=0 的比例 = {(diffs <= 0).mean()*100:.2f}%")
