"""Visualization for max_len sweep results.

Generates a 2-panel figure:
  - Top: F1(FAIL) mean ± std vs max_len (log-scale x-axis)
  - Bottom: Average fold training time vs max_len (cost overlay)
"""
from __future__ import annotations

import os
import sys
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_max_len_curve(csv_path, out_path):
    df = pd.read_csv(csv_path)
    # Aggregate
    grouped = df.groupby('max_len').agg(
        f1_mean=('f1_class_1', 'mean'),
        f1_std=('f1_class_1', 'std'),
        macro_mean=('macro_f1', 'mean'),
        macro_std=('macro_f1', 'std'),
        time_mean=('fold_seconds', 'mean'),
        time_std=('fold_seconds', 'std'),
    ).reset_index()

    # Best max_len (highest F1 mean)
    best_idx = grouped['f1_mean'].idxmax()
    best_max_len = int(grouped.loc[best_idx, 'max_len'])
    best_f1 = float(grouped.loc[best_idx, 'f1_mean'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    # Top: F1(FAIL)
    ax1.errorbar(grouped['max_len'], grouped['f1_mean'],
                 yerr=grouped['f1_std'], marker='o', capsize=4,
                 linewidth=2, color='#1f77b4', label='F1(FAIL)')
    ax1.axhline(y=best_f1, color='gray', linestyle='--', alpha=0.4)
    ax1.scatter([best_max_len], [best_f1], color='red', s=140,
                marker='*', zorder=5,
                label=f'best: max_len={best_max_len}, F1={best_f1:.4f}')
    ax1.set_ylabel('F1 (FAILED) — mean ± std')
    ax1.set_title('MetaMamba max_len sweep on CS1 (5-fold × N seeds)')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(grouped['max_len'])
    ax1.set_xticklabels([str(m) for m in grouped['max_len']])
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim(bottom=max(0.0, grouped['f1_mean'].min() - 0.05))

    # Bottom: training time
    ax2.errorbar(grouped['max_len'], grouped['time_mean'],
                 yerr=grouped['time_std'], marker='s', capsize=4,
                 linewidth=2, color='#ff7f0e', label='fold time (s)')
    ax2.set_xlabel('max_len (sequence length)')
    ax2.set_ylabel('Seconds per fold (mean ± std)')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(grouped['max_len'])
    ax2.set_xticklabels([str(m) for m in grouped['max_len']])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=9)

    # Annotation: speedup annotation
    baseline_time = grouped.loc[grouped['max_len'] == 256, 'time_mean'].values
    if len(baseline_time) > 0:
        baseline_time = float(baseline_time[0])
        for _, row in grouped.iterrows():
            ratio = row['time_mean'] / baseline_time
            ax2.annotate(f"{ratio:.1f}x",
                         xy=(row['max_len'], row['time_mean']),
                         xytext=(0, 8), textcoords='offset points',
                         ha='center', fontsize=8, color='#cc6600')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[max_len_plot] saved figure to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    plot_max_len_curve(args.csv, args.out)