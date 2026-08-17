"""Visualization utilities for StudentRisk evaluation.

Generates PNG plots from per-model OOF probabilities:
  - per-model confusion matrix
  - per-model ROC curve
  - per-model PR curve
  - all-models metric comparison bar chart
  - all-models ROC overlays
  - all-models PR overlays
  - per-fold F1 stability box plot
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, auc as sk_auc,
)
import pandas as pd


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
RESULTS_DIR = os.path.join(_ROOT, 'results')
OUTPUTS_DIR = os.path.join(_ROOT, 'outputs')
PLOTS_DIR    = os.path.join(OUTPUTS_DIR, 'plots')

MODEL_ORDER = ['rf7', 'meta_mamba', 'lstm_7d', 'bilstm_7d', 'attention_7d', 'meta_mamba_7d']
MODEL_NAMES = {
    'rf7':           'RF-7d (raw event counts)',
    'meta_mamba':    'MetaMamba',
    'lstm_7d':       'LSTM-7d',
    'bilstm_7d':     'BiLSTM-7d',
    'attention_7d':  'Attention-7d',
    'meta_mamba_7d': 'MetaMamba-7d',
}
COLORS = {
    'rf7':           '#17becf',
    'meta_mamba':    '#9467bd',
    'lstm_7d':       '#ff7f0e',
    'bilstm_7d':     '#2ca02c',
    'attention_7d':  '#d62728',
    'meta_mamba_7d': '#8c564b',
}


def _read_results(model_name: str):
    path = os.path.join(RESULTS_DIR, model_name, 'results.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _load_oof(model_name: str):
    """Load OOF probs + labels + fold_idx if available."""
    base = os.path.join(RESULTS_DIR, model_name)
    p_p = os.path.join(base, 'oof_probs.npy')
    p_l = os.path.join(base, 'labels.npy')
    p_f = os.path.join(base, 'fold_idx.npy')
    if not (os.path.exists(p_p) and os.path.exists(p_l) and os.path.exists(p_f)):
        return None
    return (np.load(p_p), np.load(p_l), np.load(p_f))


def plot_confusion_matrices(save_path: str = None):
    """Plot confusion matrices in a 2x2 grid."""
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'confusion_matrices.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, m in zip(axes.flatten(), MODEL_ORDER):
        oof = _load_oof(m)
        if oof is None:
            ax.set_title(f'{MODEL_NAMES[m]} (no OOF)')
            ax.axis('off')
            continue
        probs, y, _ = oof
        d = _read_results(m)
        threshold = d.get('threshold', 0.5) if d else 0.5
        y_pred = (probs >= threshold).astype(int)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        im = ax.imshow(cm, cmap='Blues')
        ax.set_title(f'{MODEL_NAMES[m]} (th={threshold})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['PASSED', 'FAILED'])
        ax.set_yticklabels(['PASSED', 'FAILED'])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black',
                        fontsize=14)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Saved {save_path}", flush=True)
    return save_path


def plot_roc_curves(save_path: str = None):
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'roc_curves_all.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in MODEL_ORDER:
        oof = _load_oof(m)
        if oof is None:
            continue
        probs, y, _ = oof
        fpr, tpr, _ = roc_curve(y, probs)
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[m], lw=2,
                label=f'{MODEL_NAMES[m]} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — StudentRisk Models (OOF)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Saved {save_path}", flush=True)
    return save_path


def plot_pr_curves(save_path: str = None):
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'pr_curves_all.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in MODEL_ORDER:
        oof = _load_oof(m)
        if oof is None:
            continue
        probs, y, _ = oof
        precision, recall, _ = precision_recall_curve(y, probs)
        pr_auc = sk_auc(recall, precision)
        ax.plot(recall, precision, color=COLORS[m], lw=2,
                label=f'{MODEL_NAMES[m]} (AP={pr_auc:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves — StudentRisk Models (OOF)')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Saved {save_path}", flush=True)
    return save_path


def plot_metric_comparison(csv_path: str = None, save_path: str = None):
    """Bar chart comparing Accuracy / Macro-F1 / ROC-AUC across models."""
    if csv_path is None:
        csv_path = os.path.join(OUTPUTS_DIR, 'comparison.csv')
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'metric_comparison.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.read_csv(csv_path)
    metrics = ['accuracy', 'macro_f1', 'weighted_f1', 'roc_auc']
    labels  = ['Accuracy', 'Macro-F1', 'Weighted-F1', 'ROC-AUC']
    x = np.arange(len(df))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        ax.bar(x + (i - 1.5) * width, df[m].astype(float), width,
               label=lab, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'].tolist(), rotation=15)
    ax.set_ylim([0.5, 1.0])
    ax.set_ylabel('Score')
    ax.set_title('Per-Model Comparison — StudentRisk')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Saved {save_path}", flush=True)
    return save_path


def plot_per_fold_stability(save_path: str = None):
    """Box plot of per-fold Macro-F1 across all models."""
    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, 'per_fold_stability.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data, labels, colors = [], [], []
    for m in MODEL_ORDER:
        d = _read_results(m)
        if d is None:
            continue
        folds = d.get('per_fold', [])
        if not folds:
            continue
        data.append([f['macro_f1'] for f in folds])
        labels.append(MODEL_NAMES[m])
        colors.append(COLORS[m])
    fig, ax = plt.subplots(figsize=(8, 6))
    # matplotlib >=3.9 uses 'tick_labels', older uses 'labels'
    try:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel('Macro-F1')
    ax.set_title('Per-Fold Macro-F1 Stability')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Saved {save_path}", flush=True)
    return save_path


def generate_all_visualizations():
    """Generate all plots and save them under outputs/plots/."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_confusion_matrices()
    plot_roc_curves()
    plot_pr_curves()
    if os.path.exists(os.path.join(OUTPUTS_DIR, 'comparison.csv')):
        plot_metric_comparison()
    plot_per_fold_stability()


if __name__ == '__main__':
    generate_all_visualizations()