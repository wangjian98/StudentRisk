"""Reproduce the exact 5 passed + 5 failed student IDs used in each (seed, part) of the per-part balanced 5-shot eval.

Same sampling logic as eval_per_part_5shot.py:
  np.random.seed(seed * 1000 + part)
  sup_p = np.random.choice(idx_pass, size=k_per_class, replace=False)
  sup_f = np.random.choice(idx_fail, size=k_per_class, replace=False)
"""
import os
import sys
import json
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import load_dataset
from models.meta_mamba_7d.data import build_event_sequences_7d


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'results/per_part_5shot_full'
    k_per_class = 5

    ide_logs, labels_df, y, student_ids = load_dataset()
    out = build_event_sequences_7d(ide_logs, student_ids, max_len=128)
    if len(out) == 4:
        sequences, masks, task_ids, _ = out
    else:
        sequences, masks, task_ids = out

    # passed=0, failed=1
    print("student_ids shape:", student_ids.shape, "y shape:", y.shape)

    metrics = pd.read_csv(os.path.join(out_dir, 'per_part_metrics.csv'))
    print("metrics rows:", len(metrics), "parts:", sorted(metrics['part'].unique()))

    sample_records = []
    for _, row in metrics.iterrows():
        part = int(row['part'])
        seed = int(row['seed'])
        np.random.seed(seed * 1000 + part)
        idx_part = np.where(task_ids == part)[0]
        y_part = y[idx_part]
        idx_pass = idx_part[y_part == 0]
        idx_fail = idx_part[y_part == 1]
        n_pass = len(idx_pass)
        n_fail = len(idx_fail)
        eff_k = min(k_per_class, n_pass, n_fail)

        sup_p = np.random.choice(idx_pass, size=eff_k, replace=False)
        sup_f = np.random.choice(idx_fail, size=eff_k, replace=False)

        sup_p_sids = sorted(student_ids[sup_p].tolist())
        sup_f_sids = sorted(student_ids[sup_f].tolist())

        sample_records.append({
            'part': part,
            'seed': seed,
            'k_per_class': int(eff_k),
            'support_passed_count': int(eff_k),
            'support_failed_count': int(eff_k),
            'support_passed_sids': sup_p_sids,
            'support_failed_sids': sup_f_sids,
            'support_all_sids': sorted(student_ids[np.concatenate([sup_p, sup_f])].tolist()),
            'qry_n': int(row['n_query']),
            'qry_n_passed': int(row['qry_n_passed']),
            'qry_n_failed': int(row['qry_n_failed']),
            'accuracy': float(row['accuracy']),
            'f1_failed': float(row['f1_failed']),
            'f1_passed': float(row['f1_passed']),
            'macro_f1': float(row['macro_f1']),
            'roc_auc': float(row['roc_auc']),
            'pr_auc': float(row['pr_auc']),
        })

    # Save full JSON
    full_path = os.path.join(out_dir, 'per_part_5plus5_samples.json')
    with open(full_path, 'w') as f:
        json.dump(sample_records, f, indent=2)
    print(f"Saved full sample list to {full_path}")

    # Save CSV summary
    rows = []
    for r in sample_records:
        rows.append({
            'part': r['part'],
            'seed': r['seed'],
            'k': r['k_per_class'],
            'support_passed_sids': ' '.join(map(str, r['support_passed_sids'])),
            'support_failed_sids': ' '.join(map(str, r['support_failed_sids'])),
            'qry_n': r['qry_n'],
            'acc': r['accuracy'],
            'f1_failed': r['f1_failed'],
            'f1_passed': r['f1_passed'],
            'macro_f1': r['macro_f1'],
            'roc_auc': r['roc_auc'],
            'pr_auc': r['pr_auc'],
        })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'per_part_5plus5_samples.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV summary to {csv_path}")

    # Print readable summary
    print("\n=========== Per-part 5+5 Balanced Samples ===========\n")
    for part in sorted(metrics['part'].unique()):
        part_records = [r for r in sample_records if r['part'] == part]
        print(f"\n=== Part {part} ===")
        print(f"  Pool: {part_records[0]['qry_n'] + len(part_records[0]['support_all_sids'])} students total")
        print(f"  Per seed: {part_records[0]['k_per_class']} passed + {part_records[0]['k_per_class']} failed = {2*part_records[0]['k_per_class']} support, query={part_records[0]['qry_n']}")
        for r in part_records:
            print(f"  seed={r['seed']}: passed={r['support_passed_sids']}, failed={r['support_failed_sids']}, "
                  f"acc={r['accuracy']:.3f} f1_fail={r['f1_failed']:.3f} auc={r['roc_auc']:.3f}")


if __name__ == '__main__':
    main()
