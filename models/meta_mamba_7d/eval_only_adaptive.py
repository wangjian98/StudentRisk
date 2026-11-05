"""Re-run per-part balanced 5-shot FOMAML on an existing checkpoint with adaptive k_per_class.

Usage:
  python3 eval_only_adaptive.py --ckpt results/per_part_5shot_full/final_model.ckpt \
    --out-dir results/per_part_5shot_adaptive --k-per-class 5 --adaptive-k
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import load_dataset
from models.meta_mamba.model import MetaMambaClassifier
from models.meta_mamba_7d.data import build_event_sequences_7d
from models.meta_mamba_7d.eval_per_part_5shot import run_per_part_balanced_5shot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--max-len', type=int, default=128)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--d-state', type=int, default=16)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--k-per-class', type=int, default=5)
    p.add_argument('--adaptive-k', action='store_true')
    p.add_argument('--n-seeds', type=int, default=10)
    p.add_argument('--inner-lr', type=float, default=0.01)
    p.add_argument('--inner-steps', type=int, default=3)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--n-query-max', type=int, default=50)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    ide_logs, labels_df, y, student_ids = load_dataset()
    out = build_event_sequences_7d(ide_logs, student_ids, max_len=args.max_len)
    if len(out) == 4:
        sequences, masks, task_ids, _ = out
    else:
        sequences, masks, task_ids = out
    n_tasks = int(task_ids.max()) + 1

    # Load model
    model = MetaMambaClassifier(
        d_event=sequences.shape[-1],
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        n_tasks=n_tasks,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()
    print(f"[eval-only] ckpt={args.ckpt}, device={device}", flush=True)
    print(f"[eval-only] adaptive_k={args.adaptive_k}, k_per_class={args.k_per_class}", flush=True)

    jsonl_path = os.path.join(args.out_dir, 'per_part_runs.jsonl')
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    t0 = time.time()
    result = run_per_part_balanced_5shot(
        model, sequences, masks, task_ids, y,
        parts=tuple(range(1, n_tasks + 1)),
        k_per_class=args.k_per_class,
        adaptive_k=args.adaptive_k,
        n_query_max=args.n_query_max,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        n_seeds=args.n_seeds,
        threshold=args.threshold,
        device=device,
        out_jsonl=jsonl_path,
    )
    elapsed = time.time() - t0

    # Save
    df = pd.DataFrame([r for r in result['runs'] if 'accuracy' in r])
    if len(df) > 0:
        df.to_csv(os.path.join(args.out_dir, 'per_part_metrics.csv'), index=False)
        agg = df.groupby('part').agg(
            n_runs=('seed', 'count'),
            n_query_mean=('n_query', 'mean'),
            acc_mean=('accuracy', 'mean'),
            acc_std=('accuracy', 'std'),
            f1_passed_mean=('f1_passed', 'mean'),
            f1_failed_mean=('f1_failed', 'mean'),
            f1_failed_std=('f1_failed', 'std'),
            macro_f1_mean=('macro_f1', 'mean'),
            macro_f1_std=('macro_f1', 'std'),
            prec_failed_mean=('precision_failed', 'mean'),
            rec_failed_mean=('recall_failed', 'mean'),
            roc_auc_mean=('roc_auc', 'mean'),
            roc_auc_std=('roc_auc', 'std'),
            pr_auc_mean=('pr_auc', 'mean'),
            k_used_mean=('k_per_class', 'mean'),
        ).reset_index()
        agg.to_csv(os.path.join(args.out_dir, 'per_part_summary.csv'), index=False)
        print(f"\n[eval-only] Per-part summary:", flush=True)
        print(agg.to_string(), flush=True)

    summary = {
        'eval': 'per-part balanced 5-shot FOMAML (adaptive k)',
        'k_per_class_max': args.k_per_class,
        'adaptive_k': args.adaptive_k,
        'n_seeds': args.n_seeds,
        'ckpt': args.ckpt,
        'elapsed_seconds': elapsed,
        'per_part_aggregated': result['summary'].get('per_part', []),
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[eval-only] DONE in {elapsed:.1f}s → {args.out_dir}", flush=True)


if __name__ == '__main__':
    main()