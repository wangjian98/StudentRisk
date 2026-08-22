#!/usr/bin/env python3
"""significance.py — Paired T-Test + Holm-Bonferroni for fold-level metric comparison.

Compares MetaMamba vs every other baseline on 8-fold × 3-seed = 24 paired samples,
per metric (macro_f1 / roc_auc / f1_class_1). Multiple-comparison correction is
applied within each metric family using Holm-Bonferroni (less conservative than
Bonferroni while still controlling FWER).
"""
import os
import csv
from scipy import stats

ROOT = '/home/ubuntu/StudentRisk'
RESULTS_DIR = os.path.join(ROOT, 'results')
OUT_DIR = os.path.join(ROOT, 'outputs')

MODELS = ['meta_mamba', 'meta_mamba_7d', 'lstm_7d', 'bilstm_7d', 'attention_7d']
DISPLAY = {
    'meta_mamba':    'MetaMamba',
    'meta_mamba_7d': 'MetaMamba-7d',
    'lstm_7d':       'LSTM-7d',
    'bilstm_7d':     'BiLSTM-7d',
    'attention_7d':  'Attention-7d',
}
METRICS = ['macro_f1', 'roc_auc', 'f1_class_1']
REF = 'meta_mamba'


def load_fold_metrics(model):
    path = os.path.join(RESULTS_DIR, model, 'fold_metrics.csv')
    with open(path) as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for r in rows:
        parsed.append({
            'seed': int(r['seed']),
            'fold': int(r['fold']),
            'macro_f1':   float(r['macro_f1']),
            'roc_auc':    float(r['roc_auc']),
            'f1_class_1': float(r['f1_class_1']),
        })
    return parsed


def holm_bonferroni(p_values):
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        scaled = p * (n - rank)
        running_max = max(running_max, scaled)
        adj[orig_idx] = min(running_max, 1.0)
    return adj


def sig_marker(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def main():
    data = {m: load_fold_metrics(m) for m in MODELS}

    # alignment check
    ref_keys = [(r['seed'], r['fold']) for r in data[REF]]
    for m in MODELS:
        keys = [(r['seed'], r['fold']) for r in data[m]]
        if keys != ref_keys:
            raise RuntimeError(f'{m} fold order does not match {REF}')
    n_pairs = len(ref_keys)
    print(f'n_pairs = {n_pairs} (3 seeds × 8 folds), reference = {DISPLAY[REF]}')

    out = []
    out.append('# StudentRisk — Statistical Significance Analysis')
    out.append('')
    out.append(f'> **Paired t-test** on fold-level metrics, n={n_pairs} pairs (3 seeds × 8 folds, paired by (seed, fold))')
    out.append(f'> Reference model: **{DISPLAY[REF]}**')
    out.append(f'> Multiple-comparison correction: **Holm-Bonferroni** within each metric family (4 comparisons / family)')
    out.append(f'> Significance levels: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` not significant')
    out.append(f'> CI = 95% confidence interval for mean difference (MetaMamba − baseline)')
    out.append('')
    out.append('---')
    out.append('')

    csv_rows = [['metric', 'baseline', 'mean_meta_mamba', 'mean_baseline',
                 'mean_diff', 't_statistic', 'p_raw', 'p_holm', 'ci_low', 'ci_high', 'significant']]

    for metric in METRICS:
        ref_vals = [r[metric] for r in data[REF]]
        rows = []
        for baseline in MODELS:
            if baseline == REF:
                continue
            base_vals = [r[metric] for r in data[baseline]]
            assert len(base_vals) == n_pairs
            t_res = stats.ttest_rel(ref_vals, base_vals)
            t_stat = float(t_res.statistic)
            p_raw = float(t_res.pvalue)
            mean_d = sum(a - b for a, b in zip(ref_vals, base_vals)) / n_pairs
            se = (sum((a - b - mean_d) ** 2 for a, b in zip(ref_vals, base_vals)) / (n_pairs - 1)) ** 0.5 / (n_pairs ** 0.5)
            t_crit = stats.t.ppf(0.975, df=n_pairs - 1)
            ci_low  = mean_d - t_crit * se
            ci_high = mean_d + t_crit * se
            rows.append((baseline, sum(ref_vals)/n_pairs, sum(base_vals)/n_pairs,
                         mean_d, t_stat, p_raw, ci_low, ci_high))

        adj_ps = holm_bonferroni([r[5] for r in rows])

        out.append(f'## Metric: `{metric}`')
        out.append('')
        out.append('| Baseline | Mean (MetaMamba) | Mean (Baseline) | Δ (Meta − Base) | t | p (raw) | p (Holm) | 95% CI of Δ | Sig |')
        out.append('|---|---|---|---|---|---|---|---|---|')
        for (baseline, mm, bm, md, t_stat, p_raw, ci_low, ci_high), adj_p in zip(rows, adj_ps):
            sig = sig_marker(adj_p)
            out.append(f'| {DISPLAY[baseline]} | {mm:.4f} | {bm:.4f} | '
                       f'{md:+.4f} | {t_stat:+.3f} | {p_raw:.2e} | {adj_p:.2e} | '
                       f'[{ci_low:+.4f}, {ci_high:+.4f}] | {sig} |')
            csv_rows.append([metric, baseline, f'{mm:.6f}', f'{bm:.6f}',
                             f'{md:+.6f}', f'{t_stat:+.6f}', f'{p_raw:.6e}',
                             f'{adj_p:.6e}', f'{ci_low:+.6f}', f'{ci_high:+.6f}', sig])
        out.append('')

    # per-fold detailed dump for MetaMamba (so reviewer can verify the pairs)
    out.append('---')
    out.append('')
    out.append('## Per-(seed, fold) raw values (for verification)')
    out.append('')
    header = '| seed | fold | MetaMamba | MetaMamba-7d | LSTM-7d | BiLSTM-7d | Attention-7d |'
    out.append(header)
    out.append('|---|---|---|---|---|---|---|')
    for i in range(n_pairs):
        s = data[REF][i]['seed']; f_ = data[REF][i]['fold']
        cells = [f'{s}', f'{f_}']
        for m in MODELS:
            for met in ['macro_f1']:
                cells.append(f"{data[m][i][met]:.4f}")
        out.append('| ' + ' | '.join(cells) + ' |')

    md_path = os.path.join(OUT_DIR, 'significance.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(out))
    csv_path = os.path.join(OUT_DIR, 'significance.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerows(csv_rows)

    print(f'Wrote {md_path}')
    print(f'Wrote {csv_path}')
    print()
    print('--- Compact summary ---')
    for line in out:
        if 'p (raw)' in line or 'Sig |' in line or 'Δ (Meta' in line:
            print(line)
    # print data rows
    in_table = False
    for line in out:
        if line.startswith('## Metric:'):
            in_table = True; print(); print(line); continue
        if in_table and line.startswith('| '):
            print(line)
        if line.startswith('---'):
            in_table = False


if __name__ == '__main__':
    main()
