"""Generate a single combined confusion-matrix figure for all 6 StudentRisk models.

Loads each model's OOF probabilities + labels (threshold = 0.5 from results.json)
and lays the 6 confusion matrices out in a 2x3 grid, with Chinese-friendly labels
("通过 / 挂科") and per-cell value annotations.

Usage (from inside StudentRisk/):
    python -m analysis.plot_confusion_grid
or
    python3 analysis/plot_confusion_grid.py

Output:
    outputs/plots/confusion_matrices_6models.png
"""
import os
import sys
import json

# --- Chinese font setup BEFORE pyplot ---
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from setup_fonts import setup_chinese_font  # noqa: E402
setup_chinese_font()

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

# Reuse the project's loader + model list to stay consistent with the rest of the suite.
from visualize import (  # noqa: E402
    _load_oof, _read_results, MODEL_ORDER, MODEL_NAMES, RESULTS_DIR,
)

OUTPUTS_DIR = os.path.abspath(os.path.join(_HERE, '..', 'outputs'))
PLOTS_DIR = os.path.join(OUTPUTS_DIR, 'plots')


def _metrics_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    acc = (tp + tn) / total if total else 0.0
    # class-1 (FAILED) precision/recall/f1
    prec1 = tp / (tp + fp) if (tp + fp) else 0.0
    rec1  = tp / (tp + fn) if (tp + fn) else 0.0
    f1_1  = 2 * prec1 * rec1 / (prec1 + rec1) if (prec1 + rec1) else 0.0
    return acc, prec1, rec1, f1_1


def plot_confusion_grid(save_path: str = None):
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'confusion_matrices_6models.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n = len(MODEL_ORDER)
    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()

    rows_data = []  # for side summary
    for i, m in enumerate(MODEL_ORDER):
        ax = axes[i]
        oof = _load_oof(m)
        d = _read_results(m)
        title_extra = ''
        if oof is None or d is None:
            ax.set_title(f'{MODEL_NAMES[m]} (无 OOF)')
            ax.axis('off')
            continue
        probs, y, _ = oof
        threshold = d.get('threshold', 0.5)
        y_pred = (probs >= threshold).astype(int)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        acc, prec1, rec1, f1_1 = _metrics_from_cm(cm)
        rows_data.append({
            'model': m,
            'display': MODEL_NAMES[m],
            'acc': acc,
            'prec_挂科': prec1,
            'rec_挂科': rec1,
            'f1_挂科': f1_1,
            'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]),
            'FN': int(cm[1, 0]), 'TP': int(cm[1, 1]),
        })

        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(
            f'{MODEL_NAMES[m]}\n'
            f'threshold={threshold}  Acc={acc:.3f}  F1(挂科)={f1_1:.3f}',
            fontsize=11
        )
        ax.set_xlabel('预测', fontsize=10)
        ax.set_ylabel('真实', fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['通过', '挂科'])
        ax.set_yticklabels(['通过', '挂科'])

        # 在每格写计数 + 比例
        total = cm.sum()
        for ii in range(2):
            for jj in range(2):
                cnt = int(cm[ii, jj])
                pct = cnt / total * 100 if total else 0.0
                color = 'white' if cm[ii, jj] > cm.max() / 2 else 'black'
                ax.text(jj, ii, f'{cnt}\n({pct:.1f}%)',
                        ha='center', va='center',
                        color=color, fontsize=12, fontweight='bold')

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide any leftover axes
    for j in range(len(MODEL_ORDER), len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        'StudentRisk — 6 个模型的混淆矩阵（OOF 概率，threshold=0.5）',
        fontsize=15, fontweight='bold', y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Saved {save_path}")

    # 同时保存每张矩阵的数值摘要（CSV + JSON），便于后续对比
    csv_path = os.path.join(PLOTS_DIR, 'confusion_matrices_6models.csv')
    json_path = os.path.join(PLOTS_DIR, 'confusion_matrices_6models.json')
    with open(csv_path, 'w') as f:
        f.write('model,display,acc,prec_挂科,rec_挂科,f1_挂科,TN,FP,FN,TP\n')
        for r in rows_data:
            f.write(f'{r["model"]},{r["display"]},{r["acc"]:.4f},{r["prec_挂科"]:.4f},'
                    f'{r["rec_挂科"]:.4f},{r["f1_挂科"]:.4f},'
                    f'{r["TN"]},{r["FP"]},{r["FN"]},{r["TP"]}\n')
    with open(json_path, 'w') as f:
        json.dump(rows_data, f, ensure_ascii=False, indent=2)
    print(f"[viz] Saved {csv_path}")
    print(f"[viz] Saved {json_path}")

    return save_path


if __name__ == '__main__':
    plot_confusion_grid()