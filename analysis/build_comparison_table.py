"""Generate full comparison table across all 6 StudentRisk models.

Output: outputs/comparison_full.md (Markdown) + outputs/comparison_full.csv
"""
import os
import json

ROOT = '/home/ubuntu/StudentRisk'
RES  = os.path.join(ROOT, 'results')
OUT  = os.path.join(ROOT, 'outputs', 'plots')

ORDER = ['meta_mamba', 'meta_mamba_7d', 'rf7', 'attention_7d', 'bilstm_7d', 'lstm_7d']

rows = []
for m in ORDER:
    with open(os.path.join(RES, m, 'results.json')) as f:
        d = json.load(f)
    cu = json.load(open(os.path.join(RES, m, 'config_used.json')))
    o = d['overall']; pf = d['per_fold_summary']
    rows.append({
        'model': m,
        'display': d.get('model', m),
        'n_params': cu.get('n_params', '—'),
        'elapsed_sec': round(d.get('elapsed_seconds', 0), 1),
        'acc': o['accuracy'],
        'macro_f1': o['macro_f1'],
        'macro_f1_std': pf['macro_f1_std'],
        'w_f1': o['weighted_f1'],
        'roc_auc': o['roc_auc'],
        'roc_auc_std': pf['roc_auc_std'],
        'pr_auc': o['pr_auc'],
        'p_pass': o['precision_class_0'],
        'r_pass': o['recall_class_0'],
        'f1_pass': o['f1_class_0'],
        'p_fail': o['precision_class_1'],
        'r_fail': o['recall_class_1'],
        'f1_fail': o['f1_class_1'],
        'f1_fail_std': pf['f1_class_1_std'],
        'threshold': d.get('threshold', 0.5),
    })

# Sort by macro_f1 descending
rows.sort(key=lambda r: -r['macro_f1'])

def fmt_pct(x): return f'{x*100:.2f}%'

md = ['# StudentRisk — 6 模型评测指标对比表',
      '',
      f'数据集：n_students = 473, 故障率 fail_rate = 0.6638, n_seeds = 3, n_splits = 5, threshold = 0.5  ',
      f'来源：`results/<model>/results.json`（overall + per_fold_summary）  ',
      f'排序：按 Macro-F1 降序',
      '',
      '## 1. 主指标（OOF）',
      '',
      '| # | 模型 | 参数量 | Acc | Macro-F1 | Macro-F1 ±std | Weighted-F1 | ROC-AUC | ROC-AUC ±std | PR-AUC |',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
      ]
for i, r in enumerate(rows, 1):
    np_disp = f"{r['n_params']:,}" if isinstance(r['n_params'], int) else str(r['n_params'])
    md.append(
        f"| {i} | {r['display']} | {np_disp} | {fmt_pct(r['acc'])} | **{fmt_pct(r['macro_f1'])}** | "
        f"±{r['macro_f1_std']*100:.2f}% | {fmt_pct(r['w_f1'])} | **{fmt_pct(r['roc_auc'])}** | "
        f"±{r['roc_auc_std']*100:.2f}% | {fmt_pct(r['pr_auc'])} |"
    )

md += [
    '',
    '## 2. 分类明细（Precision / Recall / F1）',
    '',
    '| 模型 | P(通过) | R(通过) | F1(通过) | P(挂科) | R(挂科) | F1(挂科) | F1(挂科) ±std |',
    '|---|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    md.append(
        f"| {r['display']} | {fmt_pct(r['p_pass'])} | {fmt_pct(r['r_pass'])} | {fmt_pct(r['f1_pass'])} | "
        f"{fmt_pct(r['p_fail'])} | {fmt_pct(r['r_fail'])} | **{fmt_pct(r['f1_fail'])}** | "
        f"±{r['f1_fail_std']*100:.2f}% |"
    )

md += [
    '',
    '## 3. 效率对比（参数量 / 训练耗时）',
    '',
    '| 模型 | 参数量 | 训练耗时 (s) | 训练耗时 (min) |',
    '|---|---:|---:|---:|',
]
for r in rows:
    np_disp = f"{r['n_params']:,}" if isinstance(r['n_params'], int) else str(r['n_params'])
    md.append(
        f"| {r['display']} | {np_disp} | {r['elapsed_sec']:.1f} | {r['elapsed_sec']/60:.1f} |"
    )

md += [
    '',
    '## 4. 显著性（MetaMamba-7d 作为基线，Holm 校正 t 检验）',
    '',
    '来自 `outputs/significance.csv`（paired t-test，3 seeds × 5 folds = 15 配对观测）：',
    '',
    '| 基线模型 | ΔMacro-F1 (mean) | ΔROC-AUC (mean) | ΔF1(挂科) (mean) | 显著? |',
    '|---|---:|---:|---:|---|',
    '| MetaMamba-7d | −0.0013 (ns) | **+0.0166*** | −0.0013 (ns) | ROC-AUC 显著 |',
    '| LSTM-7d | **+0.4689\*\*\*** | **+0.3349\*\*\*** | **+0.1187\*\*\*** | 全部显著 |',
    '| BiLSTM-7d | **+0.3865\*\*\*** | **+0.3015\*\*\*** | **+0.1124\*\*\*** | 全部显著 |',
    '| Attention-7d | **+0.3425\*\*\*** | **+0.2567\*\*\*** | **+0.1204\*\*\*** | 全部显著 |',
    '',
    '`*` p<0.05, `\*\*` p<0.01, `\*\*\*` p<0.001  ',
    '> MetaMamba-7d vs MetaMamba 仅在 ROC-AUC 上显著（+1.7pp），其它主指标差异很小（<0.2pp），属于同一水平。',
    '',
    '## 5. 关键解读',
    '',
    '1. **MetaMamba 系列绝对领先**：F1(挂科) ≈ 0.913，比第二名 RF-7d 高 +2.7pp，比序列基线 LSTM/BiLSTM/Attention 高 +11pp 以上。',
    '2. **MetaMamba 与 MetaMamba-7d 几乎并列**：11 维扩展（时间间隔/截止距离等连续变量）带来的提升在 F1 上微乎其微（<0.2pp），但 ROC-AUC 显著占优（+1.7pp）。',
    '3. **三个 7d 循环 / Attention 模型没有真正学到判别**：Macro-F1 在 0.41-0.51，召回挂科接近 1.0（漏报极少）但精确率 < 0.69（误报太多），属于"懒预测多数类"模式。',
    '4. **RF-7d 是非序列最强基线**：F1=0.886，且训练只需 5.3 秒，比所有序列模型快 1-3 个数量级——适合作为工程落地的轻量 fallback。',
    '5. **稳定性**：MetaMamba-7d 的 Macro-F1 std = 1.91%（最低），Attention-7d 高达 10.35%（最差），序列模型稳定性也明显差于 Mamba 系列。',
    '',
    '## 6. 引用与产物',
    '',
    '- 原始指标：`results/<model>/results.json`',
    '- 折指标：`results/<model>/fold_metrics.csv`',
    '- 显著性：`outputs/significance.csv`',
    '- 训练参数：`outputs/training_params.md`',
    '- 完整 Markdown：`outputs/comparison_full.md`',
    '- CSV：`outputs/comparison_full.csv`',
]

md_path = os.path.join(OUT, 'comparison_full.md')
os.makedirs(OUT, exist_ok=True)
with open(md_path, 'w') as f:
    f.write('\n'.join(md))

# CSV
csv_path = os.path.join(OUT, 'comparison_full.csv')
with open(csv_path, 'w') as f:
    keys = list(rows[0].keys())
    f.write(','.join(keys) + '\n')
    for r in rows:
        f.write(','.join(str(r[k]) for k in keys) + '\n')

print(f'Saved {md_path}')
print(f'Saved {csv_path}')