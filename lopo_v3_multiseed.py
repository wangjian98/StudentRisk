"""LOPO with 3 random seeds for stable mean ± std estimation.

For each part p ∈ {1..7} × seed ∈ {42, 123, 777}:
  - Test students = those whose dominant part == p
  - Train students = remaining (held in 90/10 for early-stop val)
  - Train ONE MetaMamba-7d model
  - Predict on test, compute metrics

Aggregates per-part mean ± std across 3 seeds.
"""
import os
import sys
import json
import time
import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import load_dataset
from models.base import set_seed
from models.meta_mamba.model import MetaMambaClassifier
from models.meta_mamba.train import train_one_fold
from models.meta_mamba_7d.data import build_event_sequences_7d


def load_config(path=None):
    if path is None:
        path = os.path.join(_ROOT, 'configs', 'default.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    seeds = (42, 123, 777)
    epochs = 30
    batch_size = 16
    max_len = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_dir = os.path.join(_ROOT, 'results', 'lopo_3seed')
    os.makedirs(out_dir, exist_ok=True)
    print(f"[LOPO-3seed] device={device}, out_dir={out_dir}, seeds={seeds}, epochs={epochs}", flush=True)

    # Load data
    ide_logs, labels_df, y, student_ids = load_dataset()
    out = build_event_sequences_7d(ide_logs, student_ids, max_len=max_len)
    if len(out) == 4:
        sequences, masks, task_ids, _ = out
    else:
        sequences, masks, task_ids = out
    n_tasks = int(task_ids.max()) + 1
    print(f"[LOPO-3seed] sequences={sequences.shape}, n_tasks={n_tasks}", flush=True)

    config = load_config()
    mcfg = config.get('meta_mamba', config.get('attention', {}))

    parts = sorted(np.unique(task_ids).tolist())
    all_records = []
    t_lopo = time.time()

    for held_out in parts:
        test_idx = np.where(task_ids == held_out)[0]
        train_pool = np.where(task_ids != held_out)[0]
        y_test = y[test_idx]
        n_test = len(test_idx)

        print(f"\n=== Held-out part {held_out+1} (1-indexed) | test={n_test} ===", flush=True)

        for seed in seeds:
            # Stratified 90/10 split on train pool
            tr_abs, va_abs = train_test_split(
                train_pool, test_size=0.1, stratify=y[train_pool], random_state=seed
            )

            set_seed(seed)
            model = MetaMambaClassifier(
                d_event=sequences.shape[-1],
                d_model=mcfg.get('d_model', 64),
                d_state=mcfg.get('d_state', 16),
                n_layers=mcfg.get('n_layers', 2),
                n_tasks=n_tasks,
                dropout=mcfg.get('dropout', 0.2),
            ).to(device)
            t0 = time.time()
            model, _ = train_one_fold(
                model,
                sequences[tr_abs], masks[tr_abs], task_ids[tr_abs], y[tr_abs],
                sequences[va_abs], masks[va_abs], task_ids[va_abs], y[va_abs],
                epochs=epochs, lr=1e-3, weight_decay=1e-3,
                batch_size=batch_size, patience=8,
                contrastive_weight=mcfg.get('contrastive_weight', 0.3),
                use_film=True,
                device=device,
            )
            train_time = time.time() - t0

            # Predict on held-out test
            model.eval()
            with torch.no_grad():
                test_prob = torch.sigmoid(model(
                    torch.from_numpy(sequences[test_idx]).float().to(device),
                    torch.from_numpy(masks[test_idx]).float().to(device),
                    torch.from_numpy(task_ids[test_idx]).long().to(device),
                )).cpu().numpy()

            y_pred = (test_prob >= 0.5).astype(int)
            try:
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            except Exception:
                tn = fp = fn = tp = 0
            acc = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_failed = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1_passed = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
            if len(np.unique(y_test)) == 2:
                roc_auc = float(roc_auc_score(y_test, test_prob))
                pr_auc = float(average_precision_score(y_test, test_prob))
            else:
                roc_auc = float('nan')
                pr_auc = float('nan')

            rec = {
                'held_out_part_1idx': int(held_out + 1),
                'held_out_part_0idx': int(held_out),
                'seed': int(seed),
                'n_train': int(len(tr_abs)),
                'n_val': int(len(va_abs)),
                'n_test': int(n_test),
                'n_test_passed': int((y_test == 0).sum()),
                'n_test_failed': int((y_test == 1).sum()),
                'train_time_s': float(train_time),
                'accuracy': float(acc),
                'macro_f1': float(macro_f1),
                'f1_passed': float(f1_passed),
                'f1_failed': float(f1_failed),
                'roc_auc': float(roc_auc),
                'pr_auc': float(pr_auc),
            }
            all_records.append(rec)
            print(f"  seed={seed} ({train_time:.0f}s): acc={acc:.3f} macro_f1={macro_f1:.3f} "
                  f"f1_fail={f1_failed:.3f} roc_auc={roc_auc:.3f} pr_auc={pr_auc:.3f}", flush=True)

    total_elapsed = time.time() - t_lopo
    print(f"\n[LOPO-3seed] All done in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)

    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(out_dir, 'per_part_metrics.csv'), index=False)

    # Aggregate per part
    agg = df.groupby('held_out_part_1idx').agg(
        n_seeds=('seed', 'count'),
        n_test=('n_test', 'first'),
        n_test_passed=('n_test_passed', 'first'),
        n_test_failed=('n_test_failed', 'first'),
        accuracy_mean=('accuracy', 'mean'),
        accuracy_std=('accuracy', 'std'),
        macro_f1_mean=('macro_f1', 'mean'),
        macro_f1_std=('macro_f1', 'std'),
        f1_passed_mean=('f1_passed', 'mean'),
        f1_failed_mean=('f1_failed', 'mean'),
        f1_failed_std=('f1_failed', 'std'),
        roc_auc_mean=('roc_auc', 'mean'),
        roc_auc_std=('roc_auc', 'std'),
        pr_auc_mean=('pr_auc', 'mean'),
        pr_auc_std=('pr_auc', 'std'),
    ).reset_index()
    agg.to_csv(os.path.join(out_dir, 'per_part_summary.csv'), index=False)

    summary = {
        'model': 'MetaMamba-7d',
        'eval': 'Leave-One-Part-Out (LOPO)',
        'seeds': list(seeds),
        'epochs': epochs,
        'max_len': max_len,
        'elapsed_seconds': total_elapsed,
        'mean_metrics': {
            'accuracy': float(agg['accuracy_mean'].mean()),
            'accuracy_std_of_means': float(agg['accuracy_mean'].std()),
            'macro_f1': float(agg['macro_f1_mean'].mean()),
            'macro_f1_std_of_means': float(agg['macro_f1_mean'].std()),
            'roc_auc': float(agg['roc_auc_mean'].mean()),
            'roc_auc_std_of_means': float(agg['roc_auc_mean'].std()),
            'pr_auc': float(agg['pr_auc_mean'].mean()),
        },
        'reference_within_task_5fold_CV': {
            'accuracy': 0.886, 'macro_f1': 0.874, 'roc_auc': 0.932,
        },
        'per_part_aggregated': agg.to_dict(orient='records'),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[LOPO-3seed] Per-part mean ± std across 3 seeds:", flush=True)
    print(agg.to_string(), flush=True)
    print(f"\nGrand mean across 7 parts:", flush=True)
    print(f"  Accuracy:  {summary['mean_metrics']['accuracy']:.3f}", flush=True)
    print(f"  Macro F1:  {summary['mean_metrics']['macro_f1']:.3f}", flush=True)
    print(f"  ROC-AUC:   {summary['mean_metrics']['roc_auc']:.3f} ± {summary['mean_metrics']['roc_auc_std_of_means']:.3f}", flush=True)
    print(f"  PR-AUC:    {summary['mean_metrics']['pr_auc']:.3f}", flush=True)
    print(f"\nReference (within-task 5-fold CV): Acc=0.886, Macro F1=0.874, ROC-AUC=0.932", flush=True)
    print(f"\n[LOPO-3seed] Done. Results in {out_dir}", flush=True)


if __name__ == '__main__':
    main()
