"""实验 E：M1 残差打分 vs 论文模型 —— 残差化后的个体判别力能否超过论文主模型

D1 已显示：显式扣除 part 先验，grp_bal 单调上升（AUC 降、个体判别升）。
本实验把同一套残差化处理施加到**论文的三个模型**上，看残差化后的个体判别力排名，
并与 M2（去混淆特征）方案直接对比 —— 确定"最优路线"到底是 M1 还是 M2 还是两者叠加。

输出：
  E1 每个模型 残差化前/后 的 AUC 与 grp_bal
  E2 残差化后 vs 未残差化的 grp_bal 提升（配对 bootstrap）
  E3 M2 方案（去混淆特征）与 M1 方案（残差打分）叠加的效果
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

B = 'oof'
y = np.load(f'{B}/meta_mamba_7d/labels.npy').ravel()
task = np.load(f'{B}/meta_mamba_7d/task_ids.npy').ravel()
n = len(y)
rng = np.random.default_rng(5)

# part-prior（按 fold 无泄漏）
fold = np.load(f'{B}/meta_mamba_7d/fold_idx.npy').ravel()
prior = np.zeros(n)
for f_ in np.unique(fold):
    tr, va = fold != f_, fold == f_
    for t in np.unique(task):
        m = tr & (task == t)
        va_m = va & (task == t)
        if va_m.sum() == 0:
            continue
        prior[va_m] = y[m].mean() if m.sum() > 0 else y[tr].mean()
prior = np.clip(prior, 1e-6, 1 - 1e-6)
lprior = np.log(prior / (1 - prior))

MODELS = {
    'meta_mamba (11d 论文主模型)': f'{B}/meta_mamba/oof_probs.npy',
    'meta_mamba_7d': f'{B}/meta_mamba_7d/oof_probs.npy',
    'rf7 (论文基线)': f'{B}/rf7/oof_probs.npy',
    'attention_7d': f'{B}/attention_7d/oof_probs.npy',
    'bilstm_7d': f'{B}/bilstm_7d/oof_probs.npy',
    'lstm_7d': f'{B}/lstm_7d/oof_probs.npy',
    'M2_V3_deconf+part1h': 'expB_oof_V3_deconf+part1h.npy',
}


def grp(p):
    a, w = [], []
    for t in np.unique(task):
        m = task == t
        if len(set(y[m])) > 1:
            a.append(roc_auc_score(y[m], p[m]))
            w.append(m.sum())
    return float(np.average(a, weights=w))


def resid(p, w):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    return 1 / (1 + np.exp(-(lp - w * lprior)))


print("=" * 118)
print("E1) 各模型在「残差化」前后的表现 (w=1.0 = 完全扣除 part 先验)")
print("=" * 118)
print(f"  {'模型':30s} {'AUC前':>8s} {'grp前':>8s} | {'AUC后':>8s} {'grp后':>8s} | {'Δgrp':>8s}")
store = {}
for name, path in MODELS.items():
    p = np.load(path).ravel()
    g0 = grp(p)
    store[name] = (p, g0)
    row = [name, roc_auc_score(y, p), g0]
    best = None
    for w in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        s = resid(p, w)
        g = grp(s)
        if best is None or g > best[1]:
            best = (w, g, s)
    row += [roc_auc_score(y, best[2]), best[1], best[1] - g0]
    store[name] = (p, g0, best[0], best[1], best[2])
    print(f"  {name:30s} {row[1]:8.4f} {row[2]:8.4f} | {row[3]:8.4f} {row[4]:8.4f} | {row[5]:+8.4f}"
          f"   (最优 w={best[0]})")

print()
print("=" * 118)
print("E2) 残差化对个体判别力的提升，是否显著？(配对 bootstrap，grp_bal 用 part 内 AUC 差)")
print("=" * 118)


def grp_bootstrap_delta(pA, pB, nboot=1200):
    """比较两个模型在组内(每 part 等权)的 AUC 差，bootstrap 重采样学生"""
    diffs = []
    for _ in range(nboot):
        tot = []
        ok = True
        for t in np.unique(task):
            idx = np.where(task == t)[0]
            if len(set(y[idx])) < 2:
                continue
            b = rng.choice(idx, size=len(idx), replace=True)
            yy = y[b]
            if len(set(yy)) < 2:
                ok = False
                break
            tot.append(roc_auc_score(yy, pA[b]) - roc_auc_score(yy, pB[b]))
        if ok and tot:
            diffs.append(np.mean(tot))
    d = np.array(diffs)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5), (d <= 0).mean()


for name in ['meta_mamba (11d 论文主模型)', 'meta_mamba_7d', 'rf7 (论文基线)']:
    p, g0, w, g1, s = store[name]
    m, lo, hi, pv = grp_bootstrap_delta(s, p)
    print(f"  {name:30s} w={w}: Δgrp_bal = {m:+.4f} 95%CI [{lo:+.4f}, {hi:+.4f}] P(Δ<=0)={pv*100:5.2f}%")

print()
print("=" * 118)
print("E3) 残差化(M1) 与 去混淆特征(M2) 叠加 —— 最优路线是哪个？")
print("=" * 118)
p_m2 = np.load('expB_oof_V3_deconf+part1h.npy').ravel()
print(f"  M2 特征方案 (V3):          AUC={roc_auc_score(y,p_m2):.4f}  grp_bal={grp(p_m2):.4f}")
for w in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]:
    s = resid(p_m2, w)
    print(f"  M2 + 残差化 w={w:4.2f}:      AUC={roc_auc_score(y,s):.4f}  grp_bal={grp(s):.4f}")

print()
print("=" * 118)
print("E4) 最终对比：论文主模型 vs 各改进路线（总体 AUC / 个体判别力）")
print("=" * 118)
p_mm, g_mm0, w_mm, g_mm1, s_mm = store['meta_mamba (11d 论文主模型)']
rows = [
    ('论文主模型 MetaMamba-11d（原样）', p_mm),
    ('论文主模型 + M1 残差化', s_mm),
    ('M2 去混淆特征 (V3)', p_m2),
    ('M2 + M1 残差化 (w=1.0)', resid(p_m2, 1.0)),
]
print(f"  {'方案':34s} {'AUC':>8s} {'F1macro':>8s} {'grp_bal':>9s}")
for name, p in rows:
    yp = (p >= 0.5).astype(int)
    print(f"  {name:34s} {roc_auc_score(y,p):8.4f} {f1_score(y,yp,average='macro',zero_division=0):8.4f} {grp(p):9.4f}")

best = max(rows, key=lambda r: grp(r[1]))
print(f"\n  → 个体判别力最优: {best[0]}  (grp_bal={grp(best[1]):.4f})")
print(f"  → 论文主模型原样 grp_bal={grp(p_mm):.4f}，改进后提升 {grp(best[1])-grp(p_mm):+.4f}")

json.dump({n: dict(auc=float(roc_auc_score(y, store[n][4])),
                    grp_bal=float(store[n][3]), w=store[n][2])
           for n in store if len(store[n]) == 5}, open('experiment_E_results.json', 'w'),
          indent=2, ensure_ascii=False)
print("\n  → experiment_E_results.json")
