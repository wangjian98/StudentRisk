"""Generate Figure 11: Cross-Dimension Comparison (v3 exclusive).

Three subfigures:
  - fig11_dimension_comparison.png : 5 architectures × 3 feature dimensions F1(FAIL) bar chart
  - fig11b_dimension_delta.png    : Dimension upgrade marginal-gain paths (waterfall-like)
  - fig11c_efficiency_scatter.png : Parameters vs F1 scatter plot

Run:
    python -m analysis.generate_fig11
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 必须先 setup font，否则中文乱码
from analysis.setup_fonts import setup_chinese_font
setup_chinese_font()

# 数据（来自 paper_v3 主结果表）
ARCHS = ['RF', 'LSTM', 'BiLSTM', 'Attention', 'MetaMamba']
DIM7 = [0.8911, 0.7985, 0.8086, 0.7995, 0.9111]
DIM11 = [None, None, None, None, 0.9144]
DIM46 = [0.8795, 0.8805, 0.8797, 0.8840, None]

ARCH_COLORS = {
    'RF':         '#1f77b4',
    'LSTM':       '#ff7f0e',
    'BiLSTM':     '#2ca02c',
    'Attention':  '#d62728',
    'MetaMamba':  '#9467bd',
}

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'docs', 'plots', 'paper'
)
os.makedirs(OUT_DIR, exist_ok=True)


def fig11_dimension_comparison():
    """5 architectures × 3 dimensions F1(FAIL) bar chart."""
    fig, ax = plt.subplots(figsize=(12, 6.5))

    x = np.arange(len(ARCHS))
    width = 0.27

    bars7 = ax.bar(x - width, DIM7, width,
                   color=[ARCH_COLORS[a] for a in ARCHS],
                   alpha=0.85, edgecolor='black', linewidth=0.8,
                   label='7-dim raw (counts/sequence)')
    dim11_vals = [v if v is not None else 0 for v in DIM11]
    bars11 = ax.bar(x, dim11_vals, width,
                    color=[ARCH_COLORS['MetaMamba'] if v > 0 else '#cccccc' for v in dim11_vals],
                    alpha=0.85, edgecolor='black', linewidth=0.8,
                    label='11-dim temporal (only MetaMamba)')
    dim46_vals = [v if v is not None else 0 for v in DIM46]
    bars46 = ax.bar(x + width, dim46_vals, width,
                    color=[ARCH_COLORS[a] if v > 0 else '#cccccc' for a, v in zip(ARCHS, dim46_vals)],
                    alpha=0.85, edgecolor='black', linewidth=0.8,
                    label='46-dim aggregate')

    for bars, vals in [(bars7, DIM7), (bars11, DIM11), (bars46, DIM46)]:
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.005, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    for i, arch in enumerate(ARCHS):
        if arch == 'MetaMamba':
            ax.annotate('** MetaMamba 7d\nF1=0.9111',
                        xy=(i - width, DIM7[i]), xytext=(i - width - 0.4, 0.95),
                        fontsize=9, color='#9467bd', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.5))
            ax.annotate('** MetaMamba 11d\nF1=0.9144 (SOTA)',
                        xy=(i, DIM11[i]), xytext=(i + 0.3, 0.97),
                        fontsize=9, color='#9467bd', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='#9467bd', lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(ARCHS, fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 (FAILED)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 11 (v3). Cross-Dimension Comparison: 7-dim raw vs 11-dim temporal vs 46-dim aggregate\n'
                 'MetaMamba achieves SOTA across all available dimensions; 7-dim version is only -0.33% below 11-dim',
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc='upper left', fontsize=10, frameon=True, edgecolor='black')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    ax.text(0.5, -0.18,
            'Core Insight: Architecture difference (MetaMamba-7d vs LSTM-7d: +11.26%) >> Feature dimension difference (MetaMamba-7d vs MetaMamba-11d: +0.33%)',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=10, color='#9467bd', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f4ecf7', edgecolor='#9467bd', lw=1.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    out = os.path.join(OUT_DIR, 'fig11_dimension_comparison.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'[fig11] saved {out}')


def fig11b_dimension_delta():
    """Dimension upgrade marginal-gain paths."""
    deltas = [
        ('RF-7d counts -> RF-46d aggregate', -1.16),
        ('LSTM-7d sequence -> LSTM-46d aggregate', +8.20),
        ('BiLSTM-7d sequence -> BiLSTM-46d aggregate', +7.11),
        ('Attention-7d sequence -> Attention-46d aggregate', +8.45),
        ('MetaMamba-7d -> MetaMamba-11d (+4 continuous)', +0.33),
        ('Attention-46d -> MetaMamba-11d (upgrade arch)', +3.04),
        ('LSTM-46d -> MetaMamba-11d', +3.39),
    ]
    labels = [d[0] for d in deltas]
    values = [d[1] for d in deltas]
    colors = ['#27ae60' if v > 0 else '#e74c3c' for v in values]

    fig, ax = plt.subplots(figsize=(11, 5))
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, alpha=0.85, edgecolor='black')
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel('Delta F1(FAIL) [%]', fontsize=11, fontweight='bold')
    ax.set_title('Figure 11b (v3). Marginal Gain of Dimension Upgrade Paths\n(green = positive, red = negative)',
                 fontsize=12, fontweight='bold', pad=15)
    for bar, v in zip(bars, values):
        ax.text(v + (0.2 if v > 0 else -0.2), bar.get_y() + bar.get_height()/2,
                f'{v:+.2f}%', va='center', ha='left' if v > 0 else 'right',
                fontsize=10, fontweight='bold')
    ax.set_xlim(-3, 10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    ax.text(0.5, -0.20,
            'Key Insights: (1) Weak architectures gain large from 7->46 dim (+7~8%); (2) Strong architecture MetaMamba only +0.33% from adding 4 continuous; (3) Strong architecture compensates feature gap',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9, color='black',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fafafa', edgecolor='gray', lw=1))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    out = os.path.join(OUT_DIR, 'fig11b_dimension_delta.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'[fig11b] saved {out}')


def fig11c_efficiency_scatter():
    """Parameters vs F1(FAIL) scatter plot."""
    params = {
        'RF-7d':           (None,    0.8911),
        'RF-46d':          (None,    0.8795),
        'LSTM-46d':        (36353,   0.8805),
        'BiLSTM-46d':      (69697,   0.8797),
        'Attention-46d':   (70209,   0.8840),
        'LSTM-7d':         (33857,   0.7985),
        'BiLSTM-7d':       (67201,   0.8086),
        'Attention-7d':    (67713,   0.7995),
        'MetaMamba-7d':    (21809,   0.9111),
        'MetaMamba':       (22065,   0.9144),
    }
    colors_map = {
        'RF-7d':          '#17becf', 'RF-46d':         '#1f77b4',
        'LSTM-46d':       '#ff7f0e', 'BiLSTM-46d':     '#2ca02c',
        'Attention-46d':  '#d62728',
        'LSTM-7d':        '#ff7f0e', 'BiLSTM-7d':      '#2ca02c',
        'Attention-7d':   '#d62728',
        'MetaMamba-7d':   '#9467bd', 'MetaMamba':      '#9467bd',
    }
    sizes_map = {
        'MetaMamba': 350, 'MetaMamba-7d': 350,
    }

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for name, (p, f1) in params.items():
        if p is None:
            ax.scatter(1000, f1, s=200, c=colors_map[name], alpha=0.85,
                       edgecolor='black', linewidth=1.2, marker='^')
            ax.annotate(name, xy=(1000, f1), xytext=(1500, f1),
                        fontsize=9, fontweight='bold', color=colors_map[name])
        else:
            sz = sizes_map.get(name, 200)
            ax.scatter(p, f1, s=sz, c=colors_map[name], alpha=0.85,
                       edgecolor='black', linewidth=1.2, marker='o')
            offset_y = 0.005 if 'MetaMamba' not in name else 0.012
            offset_x = p * 1.3
            ax.annotate(name, xy=(p, f1), xytext=(offset_x, f1 + offset_y),
                        fontsize=10 if 'MetaMamba' in name else 9,
                        fontweight='bold' if 'MetaMamba' in name else 'normal',
                        color=colors_map[name])

    ax.set_xscale('log')
    ax.set_xlabel('Parameter Count (log scale)', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 (FAILED)', fontsize=11, fontweight='bold')
    ax.set_title('Figure 11c (v3). Efficiency Scatter: Parameters vs F1\n'
                 '** MetaMamba achieves highest F1 with smallest parameters',
                 fontsize=12, fontweight='bold', pad=15)
    ax.set_ylim(0.78, 0.93)
    ax.set_xlim(700, 200000)
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.annotate('', xy=(22065, 0.9144), xytext=(50000, 0.925),
                arrowprops=dict(arrowstyle='->', color='#9467bd', lw=2))
    ax.text(50000, 0.927, '** MetaMamba\n22K params\nF1=0.9144',
            fontsize=10, color='#9467bd', fontweight='bold')

    ax.axhline(y=0.8795, color='gray', linestyle=':', lw=1, alpha=0.5)
    ax.text(1000, 0.881, 'Best 46-dim baseline (Attention, F1=0.884)', fontsize=8, color='gray')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'fig11c_efficiency_scatter.png')
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'[fig11c] saved {out}')


if __name__ == '__main__':
    print(f'Generating v3 figures into {OUT_DIR} ...')
    fig11_dimension_comparison()
    fig11b_dimension_delta()
    fig11c_efficiency_scatter()
    print('Done.')
