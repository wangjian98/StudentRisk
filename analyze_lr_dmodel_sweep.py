"""Analyze lr x d_model grid sweep for MetaMamba-11d."""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/home/ubuntu/StudentRisk'
RES = os.path.join(ROOT, 'results')
OUT = os.path.join(ROOT, 'outputs')
PLOTS = os.path.join(OUT, 'plots', 'paper')
os.makedirs(OUT, exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)

LRS = [0.0003, 0.001, 0.003]
DMS = [32, 64, 128]


def load_one(lr, dm):
    rel = f'sweep_lr{lr:.4f}_d{dm}'
    pj = os.path.join(RES, rel, 'results.json')
    if not os.path.exists(pj):
        return None
    d = json.load(open(pj))
    pf = d['per_fold_summary']
    cu = json.load(open(os.path.join(RES, rel, 'config_used.json')))
    return {
        'lr': lr, 'd_model': dm,
        'rel': rel,
        'macro_f1': pf['macro_f1_mean'], 'macro_f1_std': pf['macro_f1_std'],
        'f1_class_1': pf['f1_class_1_mean'], 'f1_class_1_std': pf['f1_class_1_std'],
        'roc_auc': pf['roc_auc_mean'], 'roc_auc_std': pf['roc_auc_std'],
        'n_params': cu.get('n_params'),
        'elapsed_sec': d.get('elapsed_seconds'),
        'n_seeds': d.get('n_seeds'),
    }


def main():
    rows = []
    for lr in LRS:
        for dm in DMS:
            r = load_one(lr, dm)
            if r is None:
                print(f'[analyze] MISSING: lr={lr} d_model={dm}')
                continue
            rows.append(r)

    if not rows:
        print('[analyze] No sweep results found.')
        return
    df = pd.DataFrame(rows)
    df = df.sort_values(['lr', 'd_model']).reset_index(drop=True)

    csv_path = os.path.join(OUT, 'lr_dmodel_sweep_11d.csv')
    df.to_csv(csv_path, index=False)
    print(f'[analyze] wrote {csv_path}')

    print('\n=== Grid table ===')
    pivot_mf1 = df.pivot(index='lr', columns='d_model', values='macro_f1')
    pivot_roc = df.pivot(index='lr', columns='d_model', values='roc_auc')
    pivot_f1c = df.pivot(index='lr', columns='d_model', values='f1_class_1')
    pivot_params = df.pivot(index='lr', columns='d_model', values='n_params')

    print('\nMacro-F1 (mean per-fold):')
    print(pivot_mf1.to_string())
    print('\nF1 (Failed=1):')
    print(pivot_f1c.to_string())
    print('\nROC-AUC:')
    print(pivot_roc.to_string())
    print('\nParams:')
    print(pivot_params.to_string())

    best_idx = df['macro_f1'].idxmax()
    best = df.iloc[best_idx]
    print(f'\n=== Best by Macro-F1 ===')
    print(f'  lr={best["lr"]}, d_model={best["d_model"]}, n_params={best["n_params"]}, '
          f'macro_f1={best["macro_f1"]:.4f}+/-{best["macro_f1_std"]:.4f}, '
          f'f1_fail={best["f1_class_1"]:.4f}, '
          f'roc_auc={best["roc_auc"]:.4f}, '
          f'elapsed={best["elapsed_sec"]:.0f}s')

    # Heatmap
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = [('Macro-F1 (mean)', pivot_mf1, True),
              ('F1 (Failed=1)', pivot_f1c, False),
              ('ROC-AUC', pivot_roc, False)]
    for ax, (title, pivot, annotate) in zip(axes, titles):
        im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(idx) for idx in pivot.index])
        ax.set_xlabel('d_model')
        ax.set_ylabel('learning rate')
        ax.set_title(title)
        if annotate:
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    v = pivot.values[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, f'{v:.4f}', ha='center', va='center',
                                color='white' if v < pivot.values.mean() else 'black',
                                fontsize=10, fontweight='bold')
        fig.colorbar(im, ax=ax)
    fig.suptitle('MetaMamba-11d lr x d_model grid sweep (5-fold x 3 seeds)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig_path = os.path.join(PLOTS, 'fig_lr_dmodel_sweep_11d.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[analyze] wrote {fig_path}')

    md = ['# MetaMamba-11d lr x d_model grid sweep', '']
    md.append('## Grid (Macro-F1 mean per fold)')
    md.append('')
    md.append('| lr \\ d_model | 32 | 64 | 128 |')
    md.append('|---|---|---|---|')
    for lr in LRS:
        cells = []
        for dm in DMS:
            v = pivot_mf1.loc[lr, dm] if (lr, dm) in [(r['lr'], r['d_model']) for _, r in df.iterrows()] else float('nan')
            cells.append(f'{v:.4f}' if not np.isnan(v) else '—')
        md.append(f'| {lr} | ' + ' | '.join(cells) + ' |')
    md.append('')
    md.append('## Best combination')
    md.append('')
    md.append(f'- **lr** = ')
    md.append(f'- **d_model** = ')
    md.append(f'- **n_params** = {best["n_params"]}')
    md.append(f'- **Macro-F1** = ')
    md.append(f'- **F1 (Failed=1)** = ')
    md.append(f'- **ROC-AUC** = ')
    md.append(f'- **Training time** = ')
    md_path = os.path.join(OUT, 'lr_dmodel_sweep_11d.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))
    print(f'[analyze] wrote {md_path}')

    rpt_path = os.path.join(OUT, 'lr_dmodel_sweep_report_11d.md')
    rpt = []
    rpt.append('# MetaMamba-11d lr x d_model Grid Search Analysis\n')
    rpt.append('## 1. Setup\n')
    rpt.append('- **Model**: MetaMamba-11d (11-dim event sequence input, max_len=256)')
    rpt.append('- **Protocol**: 5-fold StratifiedKFold x 3 seeds (42, 123, 777)')
    rpt.append('- **Fixed components**: FiLM=ON, TC lambda=0.3 (paper default), no FOMAML eval')
    rpt.append('- **Grid**:')
    rpt.append('  - learning rate: {3e-4, 1e-3, 3e-3}')
    rpt.append('  - d_model: {32, 64, 128}')
    rpt.append('  - Total: 3 x 3 = 9 runs')
    rpt.append('')
    rpt.append('## 2. Results\n')
    rpt.append('### 2.1 Macro-F1 (mean per fold)')
    rpt.append('')
    rpt.append('| lr \\ d_model | 32 | 64 | 128 |')
    rpt.append('|---|---|---|---|')
    for lr in LRS:
        cells = []
        for dm in DMS:
            v = pivot_mf1.loc[lr, dm] if (lr, dm) in [(r['lr'], r['d_model']) for _, r in df.iterrows()] else float('nan')
            cells.append(f'{v:.4f}' if not np.isnan(v) else 'missing')
        rpt.append(f'| {lr} | ' + ' | '.join(cells) + ' |')
    rpt.append('')
    rpt.append('### 2.2 F1 (Failed=1)')
    rpt.append('')
    rpt.append('| lr \\ d_model | 32 | 64 | 128 |')
    rpt.append('|---|---|---|---|')
    for lr in LRS:
        cells = []
        for dm in DMS:
            v = pivot_f1c.loc[lr, dm] if (lr, dm) in [(r['lr'], r['d_model']) for _, r in df.iterrows()] else float('nan')
            cells.append(f'{v:.4f}' if not np.isnan(v) else 'missing')
        rpt.append(f'| {lr} | ' + ' | '.join(cells) + ' |')
    rpt.append('')
    rpt.append('### 2.3 ROC-AUC')
    rpt.append('')
    rpt.append('| lr \\ d_model | 32 | 64 | 128 |')
    rpt.append('|---|---|---|---|')
    for lr in LRS:
        cells = []
        for dm in DMS:
            v = pivot_roc.loc[lr, dm] if (lr, dm) in [(r['lr'], r['d_model']) for _, r in df.iterrows()] else float('nan')
            cells.append(f'{v:.4f}' if not np.isnan(v) else 'missing')
        rpt.append(f'| {lr} | ' + ' | '.join(cells) + ' |')
    rpt.append('')
    rpt.append('### 2.4 Params & Training Time')
    rpt.append('')
    rpt.append('| lr \\ d_model | 32 | 64 | 128 |')
    rpt.append('|---|---|---|---|')
    rpt.append('| **Params (K)** |')
    for lr in LRS:
        cells = []
        for dm in DMS:
            r = next((r for _, r in df.iterrows() if r['lr'] == lr and r['d_model'] == dm), None)
            cells.append(f'{int(r["n_params"]/1000)}K ({r["elapsed_sec"]/60:.0f}m)' if r else '—')
        rpt.append(f'| lr={lr} | ' + ' | '.join(cells) + ' |')
    rpt.append('')
    rpt.append('## 3. Best Combination\n')
    rpt.append(f'**lr = {best["lr"]}, d_model = {best["d_model"]}**\n')
    rpt.append(f'- Macro-F1 = **{best["macro_f1"]:.4f} +/- {best["macro_f1_std"]:.4f}** (best in grid)')
    rpt.append(f'- F1 (Failed=1) = {best["f1_class_1"]:.4f} +/- {best["f1_class_1_std"]:.4f}')
    rpt.append(f'- ROC-AUC = {best["roc_auc"]:.4f} +/- {best["roc_auc_std"]:.4f}')
    rpt.append(f'- n_params = {best["n_params"]} ({int(best["n_params"]/1000)}K)')
    rpt.append(f'- training time = {best["elapsed_sec"]:.0f}s ({best["elapsed_sec"]/60:.1f} min)')
    rpt.append('')
    rpt.append('## 4. Sensitivity Analysis\n')
    rpt.append('')
    if len(df) >= 4:
        mf1_range = df['macro_f1'].max() - df['macro_f1'].min()
        mf1_std = df['macro_f1'].std()
        best_minus_default_d64 = None
        # Compare with current default (lr=1e-3, d_model=64)
        default_run = next((r for _, r in df.iterrows() if r['lr'] == 0.001 and r['d_model'] == 64), None)
        if default_run is not None:
            best_minus_default_d64 = best['macro_f1'] - default_run['macro_f1']
        rpt.append(f'- Macro-F1 range across grid: **{mf1_range:.4f}** (max - min)')
        rpt.append(f'- Macro-F1 std across grid: {mf1_std:.4f}')
        if best_minus_default_d64 is not None:
            rpt.append(f'- Best vs default (lr=1e-3, d_model=64): delta = **{best_minus_default_d64:+.4f}**')
        rpt.append('')
        # LR sensitivity per d_model
        rpt.append('### lr sensitivity (per d_model)')
        rpt.append('')
        for dm in DMS:
            sub = df[df['d_model'] == dm].sort_values('lr')
            if len(sub) >= 2:
                lr_range = sub['macro_f1'].max() - sub['macro_f1'].min()
                rpt.append(f'- d_model={dm}: lr range **{lr_range:.4f}** '
                           f'(best lr={sub.loc[sub["macro_f1"].idxmax(), "lr"]})')
        rpt.append('')
        rpt.append('### d_model sensitivity (per lr)')
        rpt.append('')
        for lr in LRS:
            sub = df[df['lr'] == lr].sort_values('d_model')
            if len(sub) >= 2:
                dm_range = sub['macro_f1'].max() - sub['macro_f1'].min()
                rpt.append(f'- lr={lr}: d_model range **{dm_range:.4f}** '
                           f'(best d_model={sub.loc[sub["macro_f1"].idxmax(), "d_model"]})')
        rpt.append('')
    rpt.append('## 5. Conclusions\n')
    rpt.append('')
    if best['d_model'] == 32:
        rpt.append('- **d_model=32 wins the grid**: smaller model generalizes better on n=473 dataset (less overfitting)')
    elif best['d_model'] == 128:
        rpt.append('- **d_model=128 wins the grid**: bigger capacity helps with the 11-dim event sequence')
    else:
        rpt.append('- **d_model=64 (default) is competitive**: confirms original hyperparameter choice was reasonable')
    rpt.append('')
    with open(rpt_path, 'w') as f:
        f.write('\n'.join(rpt))
    print(f'[analyze] wrote {rpt_path}')


if __name__ == '__main__':
    main()
