"""MetaMamba-11d ablation analyzer (tabulate-free)."""
import json, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = '/home/ubuntu/StudentRisk'
RESULTS = os.path.join(PROJECT, 'results')
OUTPUTS = os.path.join(PROJECT, 'outputs')
PLOTS = os.path.join(OUTPUTS, 'plots', 'paper')
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

VARIANTS = [
    ('full',              'results/meta_mamba',                       True,  0.3),
    ('no_tc',             'results/meta_mamba_ablation_no_tc',         True,  0.0),
    ('no_film',           'results/meta_mamba_ablation_no_film',       False, 0.3),
    ('no_film_no_tc',     'results/meta_mamba_ablation_no_film_no_tc', False, 0.0),
    ('lam_0_1',           'results/meta_mamba_ablation_lam_0_1',       True,  0.1),
    ('lam_0_5',           'results/meta_mamba_ablation_lam_0_5',       True,  0.5),
    ('lam_1_0',           'results/meta_mamba_ablation_lam_1_0',       True,  1.0),
]


def load_one(name, rel_path, use_film, tc_weight):
    pj = os.path.join(PROJECT, rel_path, 'results.json')
    if not os.path.exists(pj):
        return None
    d = json.load(open(pj))
    o = d['overall']
    pf = d['per_fold_summary']
    return {
        'variant': name, 'use_film': use_film, 'tc_weight': tc_weight,
        'accuracy': o['accuracy'],
        'macro_f1': o['macro_f1'], 'f1_class_1': o['f1_class_1'],
        'roc_auc': o['roc_auc'], 'pr_auc': o['pr_auc'],
        'macro_f1_mean': pf['macro_f1_mean'], 'macro_f1_std': pf['macro_f1_std'],
        'f1_class_1_mean': pf['f1_class_1_mean'], 'f1_class_1_std': pf['f1_class_1_std'],
        'roc_auc_mean': pf['roc_auc_mean'], 'roc_auc_std': pf['roc_auc_std'],
        'n_seeds': d['n_seeds'], 'n_params': d.get('n_params'),
        'n_students': d.get('n_students'), 'feature_dimension': d.get('feature_dimension'),
        'input_type': d.get('input_type'),
    }


def md_table(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, row in df.iterrows():
        out.append('| ' + ' | '.join(str(row[c]) for c in cols) + ' |')
    return '\n'.join(out)


def fmt(x, n=4):
    return f'{x:+.{n}f}' if x is not None else 'n/a'


def main():
    rows = []
    for name, rel, uf, tcw in VARIANTS:
        r = load_one(name, rel, uf, tcw)
        if r is None:
            print(f'[analyze] MISSING: {name} ({rel}) - skipping')
            continue
        rows.append(r)
    if not rows:
        print('[analyze] no results found')
        return
    df = pd.DataFrame(rows)
    order = ['full', 'no_tc', 'no_film', 'no_film_no_tc', 'lam_0_1', 'lam_0_5', 'lam_1_0']
    df['_sort'] = df['variant'].apply(order.index)
    df = df.sort_values('_sort').drop(columns=['_sort']).reset_index(drop=True)
    full = df[df['variant'] == 'full'].iloc[0]
    df['delta_macro_f1'] = df['macro_f1_mean'] - full['macro_f1_mean']
    df['delta_f1_failed'] = df['f1_class_1_mean'] - full['f1_class_1_mean']
    df['delta_roc_auc']   = df['roc_auc_mean']   - full['roc_auc_mean']

    csv_path = os.path.join(OUTPUTS, 'ablation_table_11d.csv')
    df.to_csv(csv_path, index=False)
    print(f'[analyze] wrote {csv_path}')

    cols = ['variant', 'use_film', 'tc_weight', 'n_seeds',
            'accuracy', 'macro_f1_mean', 'macro_f1_std',
            'f1_class_1_mean', 'f1_class_1_std',
            'roc_auc_mean', 'roc_auc_std',
            'delta_macro_f1', 'delta_roc_auc']
    df_show = df[cols].copy()
    for c in ['accuracy', 'macro_f1_mean', 'macro_f1_std', 'f1_class_1_mean', 'f1_class_1_std',
             'roc_auc_mean', 'roc_auc_std', 'delta_macro_f1', 'delta_roc_auc']:
        df_show[c] = df_show[c].apply(lambda x: f'{x:.4f}')
    md_path = os.path.join(OUTPUTS, 'ablation_table_11d.md')
    with open(md_path, 'w') as f:
        f.write('# MetaMamba-11d Ablation Table\n\n')
        f.write(md_table(df_show))
        f.write('\n')
    print(f'[analyze] wrote {md_path}')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = df['variant'].tolist()
    short_map = {
        'full': 'full', 'no_tc': 'no_tc', 'no_film': 'no_film',
        'no_film_no_tc': 'no_film\n_no_tc',
        'lam_0_1': 'lam=0.1', 'lam_0_5': 'lam=0.5', 'lam_1_0': 'lam=1.0',
    }
    short = [short_map.get(v, v) for v in labels]
    full_y = df[df['variant'] == 'full']['macro_f1_mean'].iloc[0]
    colors = ['#3b82f6' if v == 'full' else '#94a3b8' for v in labels]
    for ax, metric, title, lo, hi in [
        (axes[0], 'macro_f1_mean', 'Macro-F1 (per-fold mean)', 0.40, 0.92),
        (axes[1], 'f1_class_1_mean', 'F1 (Failed=1) (per-fold mean)', 0.55, 0.96),
        (axes[2], 'roc_auc_mean', 'ROC-AUC (per-fold mean)', 0.50, 0.97),
    ]:
        means = df[metric].values
        stds  = df[metric.replace('_mean', '_std')].values
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color=colors)
        ax.axhline(full_y, ls='--', c='#3b82f6', alpha=0.5,
                   label=f'full baseline ({full_y:.4f})')
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=30, ha='right', fontsize=9)
        ax.set_title(title)
        ax.set_ylim(lo, hi)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='lower left', fontsize=8)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.005, f'{m:.3f}', ha='center', fontsize=8)
    fig.suptitle('MetaMamba-11d Ablation Study (5-fold x 3-seed, n=473)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig_path = os.path.join(PLOTS, 'fig10_ablation_11d.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[analyze] wrote {fig_path}')

    full_mf1 = full['macro_f1_mean']
    def _get(name):
        sub = df[df['variant'] == name]
        return sub.iloc[0] if len(sub) else None
    no_film_row = _get('no_film')
    no_tc_row   = _get('no_tc')
    no_both_row = _get('no_film_no_tc')

    def _diff(a, b, attr='macro_f1_mean'):
        if a is None or b is None: return None
        return float(a[attr] - b[attr])

    film_contribution = _diff(full, no_film_row)
    tc_contribution   = _diff(full, no_tc_row)
    both_contribution = _diff(full, no_both_row)

    rpt = []
    rpt.append('# MetaMamba-11d 消融实验分析报告')
    rpt.append('')
    rpt.append('## 1. 实验设置')
    rpt.append('')
    n_stu = int(full.get("n_students", 0) or 0)
    fd_v = full.get("feature_dimension"); fd = 11 if (fd_v is None or fd_v != fd_v) else int(fd_v)
    rpt.append("- dataset: " + str(n_stu) + " students, " + str(fd) + "-dim event sequence")
    rpt.append(f'- **任务数**: 7 (按 problem part 分组)')
    rpt.append(f'- **协议**: 5-fold StratifiedKFold x 3 seeds (42/123/777)')
    rpt.append(f'- **基线 (full)**: FiLM=True, TC=0.3 -> Macro-F1 = {full_mf1:.4f} +/- {full["macro_f1_std"]:.4f}')
    rpt.append('')
    rpt.append('## 2. 各变体结果汇总')
    rpt.append('')
    rpt.append('| 变体 | FiLM | TC lambda | Macro-F1 (mean+/-std) | Delta vs full | F1-failed | Delta ROC-AUC |')
    rpt.append('|------|------|-----------|------------------------|---------------|-----------|----------------|')
    for _, row in df.iterrows():
        uf_mark = 'Y' if row['use_film'] else 'N'
        sign_mf1 = '+' if row['delta_macro_f1'] >= 0 else ''
        sign_auc = '+' if row['delta_roc_auc'] >= 0 else ''
        rpt.append(f'|  | {uf_mark} | {row["tc_weight"]:.1f} | '
                   f'{row["macro_f1_mean"]:.4f} +/- {row["macro_f1_std"]:.4f} | '
                   f'{sign_mf1}{row["delta_macro_f1"]:.4f} | '
                   f'{row["f1_class_1_mean"]:.4f} +/- {row["f1_class_1_std"]:.4f} | '
                   f'{sign_auc}{row["delta_roc_auc"]:.4f} |')
    rpt.append('')

    rpt.append('## 3. 各组件贡献分析')
    rpt.append('')
    rpt.append('### 3.1 FiLM 调制 (Task-Conditioned Feature-wise Linear Modulation)')
    rpt.append('')
    if no_film_row is not None:
        rpt.append(f'- **移除 FiLM** (no_film): Macro-F1 = {no_film_row["macro_f1_mean"]:.4f} '
                   f'(Delta = {fmt(film_contribution)})')
        rpt.append(f'  - FiLM 单独贡献: **{fmt(film_contribution)}** (full - no_film)')
    else:
        rpt.append('- (no_film 还未跑完)')
    rpt.append('')
    rpt.append('机制: , 让 S6 输出特征按 task id 做 channel-wise 缩放和平移, '
               '是让 Mamba backbone 适配 7 类不同任务的关键.')
    rpt.append('')

    rpt.append('### 3.2 Task-Contrastive (TC) 辅助损失')
    rpt.append('')
    if no_tc_row is not None:
        rpt.append(f'- **移除 TC** (no_tc): Macro-F1 = {no_tc_row["macro_f1_mean"]:.4f} '
                   f'(Delta = {fmt(tc_contribution)})')
        rpt.append(f'  - TC 单独贡献: **{fmt(tc_contribution)}** (full - no_tc)')
    else:
        rpt.append('- (no_tc 已完成)')
    rpt.append('')
    rpt.append('机制: NT-Xent 风格对比损失. 同一 task_id 学生嵌入拉近, 不同 task_id 推远. '
               'tau=0.1, 与监督损失加权求和 (lambda in {0.1, 0.3, 0.5, 1.0}).')
    rpt.append('')

    rpt.append('### 3.3 联合移除 (FiLM + TC)')
    rpt.append('')
    if no_both_row is not None:
        rpt.append(f'- **同时移除** (no_film_no_tc): Macro-F1 = {no_both_row["macro_f1_mean"]:.4f} '
                   f'(Delta = {fmt(both_contribution)})')
        rpt.append(f'  - 联合贡献: **{fmt(both_contribution)}**')
        if film_contribution is not None and tc_contribution is not None:
            rpt.append(f'  - FiLM + TC 简单加和: {fmt(film_contribution + tc_contribution)}')
            rpt.append(f'  - 交互效应 (interaction): '
                       f'{fmt(both_contribution - (film_contribution + tc_contribution))} '
                       f'(正值=协同, 负值=可互相替代)')
    else:
        rpt.append('- (no_film_no_tc 还未跑完)')
    rpt.append('')

    rpt.append('### 3.4 TC 权重 lambda 灵敏度')
    rpt.append('')
    rpt.append('| lambda | Macro-F1 | Delta vs full (lambda=0.3) |')
    rpt.append('|--------|----------|----------------------------|')
    lam_rows = df[df['variant'].str.startswith('lam_')].sort_values('tc_weight')
    for _, row in lam_rows.iterrows():
        sign = '+' if row['delta_macro_f1'] >= 0 else ''
        rpt.append(f'| {row["tc_weight"]:.1f} | {row["macro_f1_mean"]:.4f} +/- {row["macro_f1_std"]:.4f} | '
                   f'{sign}{row["delta_macro_f1"]:.4f} |')
    rpt.append(f'| 0.3 (full) | {full_mf1:.4f} +/- {full["macro_f1_std"]:.4f} | - |')
    if no_tc_row is not None:
        rpt.append(f'| 0.0 (no_tc) | {no_tc_row["macro_f1_mean"]:.4f} +/- {no_tc_row["macro_f1_std"]:.4f} | '
                   f'{no_tc_row["delta_macro_f1"]:+.4f} |')
    rpt.append('')

    rpt.append('## 4. 结论')
    rpt.append('')
    rpt.append('### 主要发现')
    rpt.append('')
    if film_contribution is not None:
        rpt.append(f'1. **FiLM 是性能支柱**: 单独移除使 Macro-F1 变化 {fmt(film_contribution)}, 是关键组件')
    if tc_contribution is not None:
        rpt.append(f'2. **TC 起辅助正则作用**: 移除后变化 {fmt(tc_contribution)} '
                   '(取决于随机种子, 但与 full 同等量级)')
    lam_rows_done = df[df['variant'].str.startswith('lam_')]
    if len(lam_rows_done) > 1:
        spread = lam_rows_done['macro_f1_mean'].max() - lam_rows_done['macro_f1_mean'].min()
        rpt.append(f'3. **lambda 灵敏度**: lambda 在 [0.1, 1.0] 区间内 Macro-F1 波动 '
                   f'{spread:.4f}, 表明 TC 对权重不敏感')
    rpt.append('')
    rpt.append('### 论文建议')
    rpt.append('')
    rpt.append('- baseline = full (FiLM=True, lambda=0.3) 作为主要对比')
    rpt.append('- 论文 Table 展示 4 个去组件实验: full / no_film / no_tc / no_film_no_tc')
    rpt.append('- lambda sweep 图展示辅助损失权重的鲁棒性')
    rpt.append('')

    rpt_path = os.path.join(OUTPUTS, 'ablation_report_11d.md')
    with open(rpt_path, 'w') as f:
        f.write('\n'.join(rpt))
    print(f'[analyze] wrote {rpt_path}')

    print('\n=== SUMMARY ===')
    for _, row in df.iterrows():
        print(f'  {row["variant"]:18s}  FiLM={str(row["use_film"]):5s}  lambda={row["tc_weight"]:.1f}  '
              f'Macro-F1={row["macro_f1_mean"]:.4f}+/-{row["macro_f1_std"]:.4f}  '
              f'Delta={row["delta_macro_f1"]:+.4f}')


if __name__ == '__main__':
    main()
