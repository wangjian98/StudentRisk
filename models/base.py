"""Base utilities for all models in StudentRisk.

Conventions:
  - y=1 = FAILED (positive class)
  - y=0 = PASSED (negative class)
"""
import os
import json
import time
import random
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)


def set_seed(seed: int):
    """Set seeds for Python, NumPy, PyTorch (CPU + GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None):
    """Compute per-class and overall metrics.

    Returns dict with:
      - accuracy: overall accuracy
      - precision_class_0/1, recall_class_0/1, f1_class_0/1, support_class_0/1
      - macro_f1, weighted_f1
      - roc_auc, pr_auc (if y_prob provided)
      - confusion_matrix
    """
    out = {}
    out['accuracy'] = float(accuracy_score(y_true, y_pred))

    # Per-class metrics (class 0 = passed, class 1 = failed)
    for cls in [0, 1]:
        mask_true = (y_true == cls)
        mask_pred = (y_pred == cls)
        support = int(mask_true.sum())
        tp = int((mask_true & mask_pred).sum())
        fp = int((~mask_true & mask_pred).sum())
        fn = int((mask_true & ~mask_pred).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        out[f'precision_class_{cls}'] = float(precision)
        out[f'recall_class_{cls}']    = float(recall)
        out[f'f1_class_{cls}']        = float(f1)
        out[f'support_class_{cls}']   = support

    out['macro_f1']    = float(f1_score(y_true, y_pred, average='macro',   zero_division=0))
    out['weighted_f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))

    if y_prob is not None:
        try:
            out['roc_auc'] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            out['roc_auc'] = float('nan')
        try:
            out['pr_auc'] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            out['pr_auc'] = float('nan')

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out['confusion_matrix'] = {
        'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]),
        'FN': int(cm[1, 0]), 'TP': int(cm[1, 1]),
    }
    return out


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    """Evaluate at a specific threshold. Returns per-class + overall metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    return compute_per_class_metrics(y_true, y_pred, y_prob)


def save_results(out_dir: str, payload: dict):
    """Save results to a JSON file. Creates dir if needed."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'results.json')
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    return path


class BaseModel:
    """Base class interface for all models.

    Subclasses should implement:
      - fit(X_tr, y_tr, X_va=None, y_va=None, **kwargs)
      - predict_proba(X)  →  np.ndarray of shape (n,)
    """
    name = 'base'

    def fit(self, X_tr, y_tr, **kwargs):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError