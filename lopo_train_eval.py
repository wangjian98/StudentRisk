"""Leave-One-Part-Out (LOPO) cross-task generalization evaluation.

For each part p ∈ {1..7}:
  - Test students = those whose dominant part == p
  - Train students = remaining students (473 - test_size)
  - Run 5-fold CV on train students (MetaMamba-7d, 1 seed, 40 epochs)
  - OOF predictions on test students
  - Compute Acc, Macro F1, ROC-AUC, PR-AUC

Aggregate:
  - Per-part metrics
  - Mean ± std across 7 parts
  - Comparison with standard 5-fold CV (Acc=0.886, AUC=0.93)

Outputs:
  results/lopo/per_part_metrics.csv
  results/lopo/per_part_summary.csv
  results/lopo/summary.json
"""
import os
import sys
import json
import time
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, precision_score, recall_score,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import load_dataset
from models.base import set_seed, evaluate_predictions
from models.meta_mamba.model import MetaMambaClassifier
from models.meta_mamba.train import train_one_fold
from models.meta_mamba_7d.data import build_event_sequences_7d


def load_config(path=None):
    if path is None:
        path = os.path.join(_ROOT, 'configs', 'default.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    seeds = (42,)
    n_splits = 5
    threshold = 0.5
    max_len = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_dir = os.path.join(_ROOT, 'results', 'lopo')
    os.makedirs(out_dir, exist_ok=True)
    print(f"[LOPO] device={device}, out_dir={out_dir}", flush=True)

    # Load data
    ide_logs, labels_df, y, student_ids = load_dataset()
    out = build_event_sequences_7d(ide_logs, student_ids, max_len=max_len)
    if len(out) == 4:
        sequences, masks, task_ids, _ = out
    else:
        sequences, masks, task_ids = out
    n_tasks = int(task_ids.max()) + 1
    print(f"[LOPO] sequences={sequences.shape}, n_tasks={n_tasks}", flush=True)

    config = load_config()
    mcfg = config.get('meta_mamba', config.get('attention', {}))

    n = len(y)
    parts = sorted(np.unique(task_ids).tolist())
    print(f"[LOPO] LOPO parts: {parts} (0-indexed, +1 = 1-indexed)", flush=True)

    # Per-part test sizes
    for p in parts:
        n_in_p = int((task_ids == p).sum())
        print(f"  part {p} (1-idx {p+1}): {n_in_p} test students", flush=True)

    lopo_records = []
    t_lopo = time.time()

    for held_out in parts:
        t_part = time.time()
        test_mask = (task_ids == held_out)
        train_mask = ~test_mask
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(train_mask)[0]
        y_test = y[test_idx]
        y_train = y[train_idx]

        n_test_pass = int((y_test == 0).sum())
        n_test_fail = int((y_test == 1).sum())
        n_train_pass = int((y_train == 0).sum())
        n_train_fail = int((y_train == 1).sum())
        print(f"\n=== Held-out part {held_out+1} (1-indexed) ===", flush=True)
        print(f"  train: {len(train_idx)} ({n_train_pass} pass / {n_train_fail} fail)", flush=True)
        print(f"  test:  {len(test_idx)} ({n_test_pass} pass / {n_test_fail} fail)", flush=True)

        # 5-fold CV on train set
        oof = np.zeros(len(train_idx))
        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold_idx_, (tr_rel, va_rel) in enumerate(skf.split(np.zeros(len(train_idx)), y_train)):
                set_seed(seed * 1000 + fold_idx_)
                tr_abs = train_idx[tr_rel]
                va_abs = train_idx[va_rel]
                model = MetaMambaClassifier(
                    d_event=sequences.shape[-1],
                    d_model=mcfg.get('d_model', 64),
                    d_state=mcfg.get('d_state', 16),
                    n_layers=mcfg.get('n_layers', 2),
                    n_tasks=n_tasks,
                    dropout=mcfg.get('dropout', 0.2),
                ).to(device)
                model, _ = train_one_fold(
                    model,
                    sequences[tr_abs], masks[tr_abs], task_ids[tr_abs], y[tr_abs],
                    sequences[va_abs], masks[va_abs], task_ids[va_abs], y[va_abs],
                    epochs=mcfg.get('epochs', 40),
                    lr=1e-3, weight_decay=1e-3,
                    batch_size=mcfg.get('batch_size', 16), patience=10,
                    contrastive_weight=mcfg.get('contrastive_weight', 0.3),
                    use_film=True,
                    device=device,
                )
                # OOF prediction on this fold's val set
                model.eval()
                with torch.no_grad():
                    p = torch.sigmoid(model(
                        torch.from_numpy(sequences[va_abs]).float().to(device),
                        torch.from_numpy(masks[va_abs]).float().to(device),
                        torch.from_numpy(task_ids[va_abs]).long().to(device),
                    )).cpu().numpy()
                oof[va_rel] += p
        oof /= len(seeds)

        # Train final model on FULL train set, predict on test
        final_model = MetaMambaClassifier(
            d_event=sequences.shape[-1],
            d_model=mcfg.get('d_model', 64),
            d_state=mcfg.get('d_state', 16),
            n_layers=mcfg.get('n_layers', 2),
            n_tasks=n_tasks,
            dropout=mcfg.get('dropout', 0.2),
        ).to(device)
        # Split train_idx into 90/10 for final training (val for early stop)
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(train_idx))
        n_val = max(int(len(train_idx) * 0.1), 10)
        val_rel = perm[:n_val]
        tr_rel = perm[n_val:]
        tr_abs_final = train_idx[tr_rel]
        va_abs_final = train_idx[val_rel]
        final_model, _ = train_one_fold(
            final_model,
            sequences[tr_abs_final], masks[tr_abs_final], task_ids[tr_abs_final], y[tr_abs_final],
            sequences[va_abs_final], masks[va_abs_final], task_ids[va_abs_final], y[va_abs_final],
            epochs=mcfg.get('epochs', 40),
            lr=1e-3, weight_decay=1e-3,
            batch_size=mcfg.get('batch_size', 16), patience=10,
            contrastive_weight=mcfg.get('contrastive_weight', 0.3),
            use_film=True,
            device=device,
        )
        final_model.eval()
        with torch.no_grad():
            test_prob = torch.sigmoid(final_model(
                torch.from_numpy(sequences[test_idx]).float().to(device),
                torch.from_numpy(masks[test_idx]).float().to(device),
                torch.from_numpy(task_ids[test_idx]).long().to(device),
            )).cpu().numpy()

        # Compute metrics
        y_pred = (test_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_failed = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_passed = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
        roc_auc = roc_auc_score(y_test, test_prob) if len(np.unique(y_test)) == 2 else float('nan')
        pr_auc = average_precision_score(y_test, test_prob) if len(np.unique(y_test)) == 2 else float('nan')

        elapsed = time.time() - t_part
        rec = {
            'held_out_part_1idx': int(held_out + 1),
            'held_out_part_0idx': int(held_out),
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'n_test_passed': n_test_pass,
            'n_test_failed': n_test_fail,
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),
            'f1_passed': float(f1_passed),
            'f1_failed': float(f1_failed),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
            'elapsed_seconds': float(elapsed),
        }
        lopo_records.append(rec)
        print(f"  → test acc={acc:.3f} macro_f1={macro_f1:.3f} "
              f"f1_fail={f1_failed:.3f} roc_auc={roc_auc:.3f} "
              f"pr_auc={pr_auc:.3f} ({elapsed:.0f}s)", flush=True)

    total_elapsed = time.time() - t_lopo
    print(f"\n[LOPO] All 7 parts done in {total_elapsed:.0f}s", flush=True)

    # Save
    df = pd.DataFrame(lopo_records)
    df.to_csv(os.path.join(out_dir, 'per_part_metrics.csv'), index=False)
    summary = {
        'model': 'MetaMamba-7d',
        'eval': 'Leave-One-Part-Out (LOPO)',
        'n_splits_inner': n_splits,
        'seeds': list(seeds),
        'max_len': max_len,
        'elapsed_seconds': total_elapsed,
        'mean_metrics': {
            'accuracy': float(df['accuracy'].mean()),
            'accuracy_std': float(df['accuracy'].std()),
            'macro_f1': float(df['macro_f1'].mean()),
            'macro_f1_std': float(df['macro_f1'].std()),
            'roc_auc': float(df['roc_auc'].mean()),
            'roc_auc_std': float(df['roc_auc'].std()),
            'pr_auc': float(df['pr_auc'].mean()),
        },
        'reference_within_task_5fold_CV': {
            'accuracy': 0.886, 'macro_f1': 0.874, 'roc_auc': 0.932,
        },
        'per_part': lopo_records,
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[LOPO] Per-part summary:", flush=True)
    print(df[['held_out_part_1idx', 'n_test', 'n_test_passed', 'n_test_failed',
              'accuracy', 'macro_f1', 'f1_failed', 'roc_auc', 'pr_auc']].to_string(), flush=True)
    print(f"\nMean across 7 parts:", flush=True)
    print(f"  Accuracy:  {summary['mean_metrics']['accuracy']:.3f} ± {summary['mean_metrics']['accuracy_std']:.3f}", flush=True)
    print(f"  Macro F1:  {summary['mean_metrics']['macro_f1']:.3f} ± {summary['mean_metrics']['macro_f1_std']:.3f}", flush=True)
    print(f"  ROC-AUC:   {summary['mean_metrics']['roc_auc']:.3f} ± {summary['mean_metrics']['roc_auc_std']:.3f}", flush=True)
    print(f"  PR-AUC:    {summary['mean_metrics']['pr_auc']:.3f}", flush=True)
    print(f"\n[LOPO] Done. Results in {out_dir}", flush=True)


if __name__ == '__main__':
    main()
