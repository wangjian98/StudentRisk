"""Train MetaMamba-7d with 5-fold CV (single seed for speed) then do:
  (a) NEW: Per-part BALANCED 5-shot FOMAML with full metrics

Output: results/per_part_5shot_*/summary.json + per_part_*.csv
"""
import os
import sys
import json
import time
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import load_dataset
from models.base import set_seed, evaluate_predictions, save_results
from models.meta_mamba.model import MetaMambaClassifier
from models.meta_mamba.train import train_one_fold
from models.meta_mamba_7d.data import build_event_sequences_7d
from models.meta_mamba_7d.eval_per_part_5shot import run_per_part_balanced_5shot


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(_ROOT, 'configs', 'default.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 777])
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--max-len', type=int, default=128)
    p.add_argument('--out-dir', type=str, default=None)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--d-state', type=int, default=16)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--contrastive-weight', type=float, default=0.3)
    # Per-part eval params
    p.add_argument('--k-per-class', type=int, default=5)
    p.add_argument('--adaptive-k', action='store_true', help='Use min(k_per_class, min(n_pass, n_fail)) per part')
    p.add_argument('--n-seeds-eval', type=int, default=10)
    p.add_argument('--inner-lr', type=float, default=0.01)
    p.add_argument('--inner-steps', type=int, default=3)
    p.add_argument('--n-query-max', type=int, default=50)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = load_config()
    mcfg = config.get('meta_mamba', config.get('attention', {}))
    for k in ['d_model','d_state','n_layers','dropout','epochs','batch_size','contrastive_weight']:
        mcfg[k] = getattr(args, k)

    if args.out_dir is None:
        ts = time.strftime('%Y%m%d_%H%M%S')
        args.out_dir = os.path.join(_ROOT, 'results', f'per_part_5shot_{ts}')
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[train+eval] device={device}, out_dir={args.out_dir}", flush=True)
    print(f"[train+eval] args={vars(args)}", flush=True)

    # ===== 1. Load data =====
    ide_logs, labels_df, y, student_ids = load_dataset()
    out = build_event_sequences_7d(
        ide_logs, student_ids, max_len=args.max_len,
    )
    # Some builds return 3, some 4 (with event_counts)
    if len(out) == 4:
        sequences, masks, task_ids, _ = out
    else:
        sequences, masks, task_ids = out
    n_tasks = int(task_ids.max()) + 1
    print(f"[train+eval] sequences={sequences.shape}, n_tasks={n_tasks}", flush=True)

    n = len(y)

    # ===== 2. Train with CV =====
    t_train = time.time()
    final_model = None
    for seed in args.seeds:
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)
        for fold_idx_, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y)):
            set_seed(seed * 1000 + fold_idx_)
            model = MetaMambaClassifier(
                d_event=sequences.shape[-1],
                d_model=mcfg['d_model'],
                d_state=mcfg['d_state'],
                n_layers=mcfg['n_layers'],
                n_tasks=n_tasks,
                dropout=mcfg['dropout'],
            ).to(device)
            model, _ = train_one_fold(
                model,
                sequences[tr_idx], masks[tr_idx], task_ids[tr_idx], y[tr_idx],
                sequences[va_idx], masks[va_idx], task_ids[va_idx], y[va_idx],
                epochs=mcfg['epochs'],
                lr=1e-3, weight_decay=1e-3,
                batch_size=mcfg['batch_size'], patience=10,
                contrastive_weight=mcfg['contrastive_weight'],
                use_film=True,
                device=device,
            )
            final_model = model
            elapsed = time.time() - t_train
            print(f"[train+eval]   seed={seed} fold={fold_idx_} done at {elapsed:.1f}s", flush=True)

    train_elapsed = time.time() - t_train
    print(f"[train+eval] Training done in {train_elapsed:.1f}s", flush=True)

    # Save model checkpoint
    ckpt_path = os.path.join(args.out_dir, 'final_model.ckpt')
    torch.save(final_model.state_dict(), ckpt_path)
    print(f"[train+eval] Saved model checkpoint to {ckpt_path}", flush=True)

    # ===== 3. Per-part BALANCED 5-shot FOMAML =====
    print(f"\n[train+eval] === Per-part BALANCED 5-shot FOMAML ===", flush=True)
    jsonl_path = os.path.join(args.out_dir, 'per_part_runs.jsonl')
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    t_eval = time.time()
    pp_result = run_per_part_balanced_5shot(
        final_model, sequences, masks, task_ids, y,
        parts=tuple(range(1, n_tasks + 1)),
        k_per_class=args.k_per_class,
        adaptive_k=args.adaptive_k,
        n_query_max=args.n_query_max,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        n_seeds=args.n_seeds_eval,
        threshold=args.threshold,
        device=device,
        out_jsonl=jsonl_path,
    )
    eval_elapsed = time.time() - t_eval
    print(f"[train+eval] Per-part eval done in {eval_elapsed:.1f}s", flush=True)

    # ===== 4. Save =====
    runs = pp_result['runs']
    df = pd.DataFrame([r for r in runs if 'accuracy' in r])
    if len(df) > 0:
        csv_path = os.path.join(args.out_dir, 'per_part_metrics.csv')
        df.to_csv(csv_path, index=False)
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
        ).reset_index()
        agg_csv = os.path.join(args.out_dir, 'per_part_summary.csv')
        agg.to_csv(agg_csv, index=False)
        print(f"\n[train+eval] Per-part summary:", flush=True)
        print(agg.to_string(), flush=True)

    summary = {
        'model': 'MetaMamba-7d',
        'train_config': mcfg,
        'seeds': args.seeds,
        'n_splits': args.n_splits,
        'n_tasks': n_tasks,
        'max_len': args.max_len,
        'threshold': args.threshold,
        'train_elapsed_seconds': train_elapsed,
        'eval_elapsed_seconds': eval_elapsed,
        'eval': {
            'method': 'Per-part BALANCED 5-shot FOMAML',
            'k_per_class': args.k_per_class,
            'total_support': args.k_per_class * 2,
            'n_seeds': args.n_seeds_eval,
            'inner_lr': args.inner_lr,
            'inner_steps': args.inner_steps,
            'n_query_max': args.n_query_max,
        },
        'per_part_aggregated': pp_result['summary'].get('per_part', []),
    }
    summary_path = os.path.join(args.out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[train+eval] DONE. Results saved to {args.out_dir}", flush=True)


if __name__ == '__main__':
    main()
