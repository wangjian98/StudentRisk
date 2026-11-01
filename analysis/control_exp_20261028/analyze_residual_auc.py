"""最终验证: 个体判别力 (part 内 / 残差) 的跨模型公平比较 + 与论文声称数字的核对"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

B = 'oof'
DIRS = ['meta_mamba', 'meta_mamba_7d', 'rf7', 'lstm_7d', 'bilstm_7d', 'attention_7d']


def load(d):
    o = {}
    for k, f in [('p', 'oof_probs.npy'), ('y', 'labels.npy'), ('sid', 'student_ids.npy'),
                 ('task', 'task_ids.npy')]:
        try:
            a = np.load(f'{B}/{d}/{f}')
            o[k] = np.asarray(a).ravel()
        except Exception:
            o[k] = None
    return o


M = {d: load(d) for d in DIRS}
y = M['meta_mamba_7d']['y']
task = M['meta_mamba_7d']['task']
n = len(y)

print("=" * 105)
print("Q) 与各模型 results.json 声称数字的核对 (确认 oof_probs.npy 就是论文用的那批)")
print("=" * 105)
for d in DIRS:
    try:
        r = json.load(open(f'{B}/{d}/results.json'))
    except Exception:
        continue
    p = M[d]['p']
    yp = (p >= 0.5).astype(int)
    mine_auc = roc_auc_score(y, p)
    mine_f1 = f1_score(y, yp, average='macro', zero_division=0)
    keys = [k for k in r.keys() if any(s in k.lower() for s in ['auc', 'f1', 'acc'])]
    print(f"  --- {d} ---")
    print(f"    重算: AUC={mine_auc:.4f} f1_macro={mine_f1:.4f} acc={accuracy_score(y,yp):.4f}")
    for k in sorted(keys)[:8]:
        v = r[k]
        if isinstance(v, (int, float)):
            print(f"      {k} = {v:.4f}" if isinstance(v, float) else f"      {k} = {v}")

print()
print("=" * 105)
print("R) 个体判别力公平比较: 残差 AUC (减去 part 内均值 -> 纯个体信号)")
print("=" * 105)


def residual_auc(p, task, y):
    """先减去每个 part 的均值, 再算 AUC (pooled)"""
    r = p.copy()
    for t in np.unique(task):
        m = task == t
        r[m] = p[m] - p[m].mean()
    return roc_auc_score(y, r)


def group_balanced_auc(p, task, y):
    """组内 AUC 的加权平均 (每个 part 等权)"""
    aucs, ws = [], []
    for t in np.unique(task):
        m = task == t
        if len(set(y[m])) > 1:
            aucs.append(roc_auc_score(y[m], p[m]))
            ws.append(m.sum())
    return float(np.average(aucs, weights=ws)), aucs


rows = []
for d in DIRS:
    p = M[d]['p']
    if p is None:
        continue
    gauc, per = group_balanced_auc(p, task, y)
    rows.append(dict(model=d, auc=roc_auc_score(y, p), resid_auc=residual_auc(p, task, y),
                     group_bal_auc=gauc,
                     part3_auc=round(per[np.unique(task).tolist().index(3)], 4) if 3 in np.unique(task) else None,
                     part6_auc=round(per[np.unique(task).tolist().index(6)], 4) if 6 in np.unique(task) else None))
R = pd.DataFrame(rows).sort_values('auc', ascending=False)
print(R.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print()
print("解读:")
print("  auc            = 总体 AUC (含 part 先验红利)")
print("  resid_auc      = 去掉 part 均值后的 AUC (纯个体信号)")
print("  group_bal_auc  = part 内 AUC 加权平均 (每个 part 等权, 不给大 part 主导)")
print("  part3/part6    = 唯二正负样本各>=5 的 part 的 part 内 AUC")

print()
print("=" * 105)
print("S) part 先验 baseline 的诚实上界 (按 fold 无泄漏) 与各模型对比")
print("=" * 105)
f = np.load(f'{B}/meta_mamba_7d/fold_idx.npy')
prior_fold = np.zeros(n)
for fold in np.unique(f):
    tr, va = f != fold, f == fold
    for t in np.unique(task):
        m = tr & (task == t)
        prior_fold[va & (task == t)] = y[m].mean() if m.sum() > 0 else y[tr].mean()
a_prior = roc_auc_score(y, prior_fold)
print(f"  part-prior (fold 无泄漏) AUC = {a_prior:.4f}")
for _, r in R.iterrows():
    print(f"  {r['model']:16s} AUC={r['auc']:.4f}  超出 part 先验 {r['auc']-a_prior:+.4f}  "
          f"个体信号(resid) AUC={r['resid_auc']:.4f}")

print()
print("=" * 105)
print("T) 复现检验: 新实验 (10-28, 独立训练的 OOF) 是否复现同一现象")
print("=" * 105)
pred = pd.read_csv('results/concat_dummy_task_id/per_student_predictions.csv')
t2 = pred[(pred.conditioning == 'film') & (pred.task_id_used == 'true')].sort_values('student_idx')
p2 = t2['pred_prob'].values
pt2 = t2['true_part'].values
y2 = t2['true_label'].values
print(f"  新 OOF: AUC={roc_auc_score(y2,p2):.4f}  resid_auc={residual_auc(p2,pt2,y2):.4f}  "
      f"group_bal_auc={group_balanced_auc(p2,pt2,y2)[0]:.4f}")
print("  → 若同样呈现 总体高 / 个体低, 则两批独立 OOF 一致, 非实现偶发")
