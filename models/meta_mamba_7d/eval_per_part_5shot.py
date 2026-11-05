"""Per-part balanced 5-shot FOMAML evaluation for MetaMamba-7d.

Per-part balanced 5-shot:
  - For each part p ∈ {1..7}:
    - Filter students whose task_id == p
    - Sample 5 passed + 5 failed as support (CLASS-BALANCED)
    - Sample remaining students in this part as query
    - FOMAML inner-loop adapt on support (3 steps)
    - Predict on query, compute full metrics
  - Repeat with N sample seeds for stability

Compared to original `run_maml_fewshot`:
  - Original: k_shot=5 random, no class balance, only F1, aggregated across all parts
  - This script: k_shot_per_class=5 (i.e. 10 total), class-balanced, all metrics, per-part breakdown

Outputs:
  - results/per_part_5shot/per_part_metrics.csv  (per-part mean ± std)
  - results/per_part_5shot/per_part_runs.jsonl   (each individual run for transparency)
  - results/per_part_5shot/summary.json          (overall summary)
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import load_dataset
from models.meta_mamba.model import MetaMambaClassifier
from models.meta_mamba_7d.data import build_event_sequences_7d


def full_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute full classification metrics. Failed=1 convention."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'n': int(len(y_true)),
        'n_passed': int((y_true == 0).sum()),
        'n_failed': int((y_true == 1).sum()),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_passed': float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'recall_passed':    float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'f1_passed':        float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'precision_failed': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'recall_failed':    float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'f1_failed':        float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'macro_f1':         float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'roc_auc':          float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float('nan'),
        'pr_auc':           float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float('nan'),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
    }


def fomaml_inner_adapt(model, seq_sup, mask_sup, task_sup, y_sup,
                        criterion, inner_lr=0.01, inner_steps=3, device='cpu'):
    """FOMAML first-order inner loop. Returns the adapted state_dict (params)."""
    theta = {k: v.clone() for k, v in model.state_dict().items()}
    for _ in range(inner_steps):
        model.train()
        seq_t   = torch.from_numpy(seq_sup).float().to(device)
        mask_t  = torch.from_numpy(mask_sup).float().to(device)
        task_t  = torch.from_numpy(task_sup).long().to(device)
        y_t     = torch.from_numpy(y_sup.astype(np.float32)).to(device)
        logits  = model(seq_t, mask_t, task_t)
        loss    = criterion(logits, y_t)
        grads   = torch.autograd.grad(loss, model.parameters(), create_graph=False)
        with torch.no_grad():
            for p, g in zip(model.parameters(), grads):
                p.sub_(inner_lr * g)
    return theta


def predict_with_state(model, state, seq, mask, task, device='cpu') -> np.ndarray:
    """Restore state, predict, restore theta."""
    theta = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(
            torch.from_numpy(seq).float().to(device),
            torch.from_numpy(mask).float().to(device),
            torch.from_numpy(task).long().to(device),
        )).cpu().numpy()
    model.load_state_dict(theta)  # restore
    return p


def run_per_part_balanced_5shot(model, sequences, masks, task_ids, y,
                                 parts=(1, 2, 3, 4, 5, 6, 7),
                                 k_per_class: int = 5,
                                 adaptive_k: bool = False,
                                 n_query_max: int = 50,
                                 inner_lr: float = 0.01,
                                 inner_steps: int = 3,
                                 n_seeds: int = 10,
                                 threshold: float = 0.5,
                                 device: str = 'cpu',
                                 out_jsonl: str = None,
                                 ) -> dict:
    """Per-part balanced 5-shot FOMAML.

    For each part:
      - If adaptive_k=True: use k = min(k_per_class, min(n_pass, n_fail))
        so parts with limited data still get evaluated
      - If adaptive_k=False: skip parts with < k_per_class in either class
      - Sample 2*k students (balanced) for support
      - Remaining students in this part → query (capped at n_query_max)
    """
    criterion = nn.BCEWithLogitsLoss()
    runs = []

    for part in parts:
        part_mask = (task_ids == part)
        idx_part = np.where(part_mask)[0]
        y_part = y[idx_part]
        n_part = len(idx_part)
        n_pass = int((y_part == 0).sum())
        n_fail = int((y_part == 1).sum())
        print(f"\n[Part {part}] n={n_part}, passed={n_pass}, failed={n_fail}", flush=True)

        # Determine effective k for this part
        if adaptive_k:
            eff_k = min(k_per_class, n_pass, n_fail)
        else:
            eff_k = k_per_class

        if n_pass < eff_k or n_fail < eff_k or eff_k < 2:
            print(f"  → SKIP (need ≥{eff_k} per class, have passed={n_pass} failed={n_fail})", flush=True)
            runs.append({
                'part': int(part), 'seed': -1, 'skip': True,
                'reason': f'need >= {eff_k} per class, got passed={n_pass} failed={n_fail}',
                'n_pass': n_pass, 'n_fail': n_fail,
                'effective_k': eff_k,
            })
            continue

        idx_pass = idx_part[y_part == 0]
        idx_fail = idx_part[y_part == 1]

        print(f"  effective k_per_class = {eff_k}", flush=True)

        for seed in range(n_seeds):
            np.random.seed(seed * 1000 + part)
            sup_p = np.random.choice(idx_pass, size=eff_k, replace=False)
            sup_f = np.random.choice(idx_fail, size=eff_k, replace=False)
            sup_idx = np.concatenate([sup_p, sup_f])
            # Query = remaining in this part
            mask_used = np.zeros(len(idx_part), dtype=bool)
            used_set = set()
            for i in sup_idx:
                # find position in idx_part
                pos = np.where(idx_part == i)[0][0]
                mask_used[pos] = True
            qry_idx = idx_part[~mask_used]
            if len(qry_idx) > n_query_max:
                np.random.shuffle(qry_idx)
                qry_idx = qry_idx[:n_query_max]

            # Build support arrays
            seq_sup = sequences[sup_idx]
            mask_sup = masks[sup_idx]
            task_sup = task_ids[sup_idx]
            y_sup = y[sup_idx]

            # Inner adapt
            adapted_state = fomaml_inner_adapt(
                model, seq_sup, mask_sup, task_sup, y_sup,
                criterion, inner_lr=inner_lr, inner_steps=inner_steps, device=device,
            )

            # Predict on query
            q_prob = predict_with_state(
                model, adapted_state,
                sequences[qry_idx], masks[qry_idx], task_ids[qry_idx],
                device=device,
            )

            # Metrics
            m = full_metrics(y[qry_idx], q_prob, threshold=threshold)
            m.update({
                'part': int(part), 'seed': seed,
                'k_per_class': eff_k,
                'k_per_class_requested': k_per_class,
                'adaptive_k': adaptive_k,
                'n_support': len(sup_idx),
                'n_query': int(len(qry_idx)),
                'qry_n_passed': int((y[qry_idx] == 0).sum()),
                'qry_n_failed': int((y[qry_idx] == 1).sum()),
                'inner_lr': inner_lr, 'inner_steps': inner_steps,
            })
            runs.append(m)
            if out_jsonl:
                with open(out_jsonl, 'a') as f:
                    f.write(json.dumps(m) + '\n')

            print(f"  seed={seed}: qry_acc={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f} "
                  f"f1_failed={m['f1_failed']:.3f} roc_auc={m['roc_auc']:.3f}", flush=True)

    # Aggregate per part
    df = pd.DataFrame([r for r in runs if 'accuracy' in r])
    summary = {}
    if len(df) > 0:
        agg = df.groupby('part').agg(
            n_runs=('seed', 'count'),
            acc_mean=('accuracy', 'mean'),  acc_std=('accuracy', 'std'),
            f1_passed_mean=('f1_passed', 'mean'), f1_passed_std=('f1_passed', 'std'),
            f1_failed_mean=('f1_failed', 'mean'), f1_failed_std=('f1_failed', 'std'),
            macro_f1_mean=('macro_f1', 'mean'), macro_f1_std=('macro_f1', 'std'),
            prec_failed_mean=('precision_failed', 'mean'),
            rec_failed_mean=('recall_failed', 'mean'),
            roc_auc_mean=('roc_auc', 'mean'), roc_auc_std=('roc_auc', 'std'),
            pr_auc_mean=('pr_auc', 'mean'),
            n_query_mean=('n_query', 'mean'),
        ).reset_index()
        summary['per_part'] = agg.to_dict(orient='records')
    return {'runs': runs, 'summary': summary}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-len', type=int, default=128)
    p.add_argument('--k-per-class', type=int, default=5, help='Support shots per class (5=balanced 5-shot)')
    p.add_argument('--adaptive-k', action='store_true', help='Use min(k_per_class, min(n_pass, n_fail)) per part')
    p.add_argument('--n-seeds', type=int, default=10)
    p.add_argument('--inner-lr', type=float, default=0.01)
    p.add_argument('--inner-steps', type=int, default=3)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--out-dir', type=str, default=None)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--d-state', type=int, default=16)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--n-query-max', type=int, default=50)
    p.add_argument('--ckpt-in', type=str, default=None,
                   help='Optional: load model from a checkpoint instead of retraining')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = args.out_dir or os.path.join(_ROOT, 'results', 'per_part_5shot')
    os.makedirs(out_dir, exist_ok=True)

    print(f"[per-part 5shot] device={device}, out_dir={out_dir}", flush=True)

    # 1) Load data
    ide_logs, labels_df, y, student_ids = load_dataset()
    print(f"[per-part 5shot] Building 7-dim event sequences (max_len={args.max_len}) ...", flush=True)
    sequences, masks, task_ids, event_counts = build_event_sequences_7d(
        ide_logs, student_ids, max_len=args.max_len,
    )
    n_tasks = int(task_ids.max()) + 1
    print(f"[per-part 5shot] sequences={sequences.shape}, n_tasks={n_tasks}", flush=True)

    # 2) Build / load model
    model = MetaMambaClassifier(
        d_event=sequences.shape[-1],
        d_model=args.d_model,
        d_state=args.d_state,
        n_layers=args.n_layers,
        n_tasks=n_tasks,
        dropout=args.dropout,
    ).to(device)
    if args.ckpt_in and os.path.exists(args.ckpt_in):
        model.load_state_dict(torch.load(args.ckpt_in, map_location=device))
        print(f"[per-part 5shot] Loaded model from {args.ckpt_in}", flush=True)
    else:
        print(f"[per-part 5shot] Using random-init model (no checkpoint provided)", flush=True)

    # 3) Run per-part balanced 5-shot FOMAML
    jsonl_path = os.path.join(out_dir, 'per_part_runs.jsonl')
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

    # 4) Save
    df = pd.DataFrame([r for r in result['runs'] if 'accuracy' in r])
    if len(df) > 0:
        csv_path = os.path.join(out_dir, 'per_part_metrics.csv')
        df.to_csv(csv_path, index=False)
        agg = df.groupby('part').agg(
            n_runs=('seed', 'count'),
            acc_mean=('accuracy', 'mean'),  acc_std=('accuracy', 'std'),
            f1_passed_mean=('f1_passed', 'mean'),
            f1_failed_mean=('f1_failed', 'mean'),
            macro_f1_mean=('macro_f1', 'mean'),
            prec_failed_mean=('precision_failed', 'mean'),
            rec_failed_mean=('recall_failed', 'mean'),
            roc_auc_mean=('roc_auc', 'mean'),
        ).reset_index()
        agg_csv = os.path.join(out_dir, 'per_part_summary.csv')
        agg.to_csv(agg_csv, index=False)

    summary = {
        'model': 'MetaMamba-7d',
        'eval': 'per-part balanced 5-shot FOMAML',
        'k_per_class': args.k_per_class,
        'n_seeds': args.n_seeds,
        'inner_lr': args.inner_lr,
        'inner_steps': args.inner_steps,
        'threshold': args.threshold,
        'max_len': args.max_len,
        'd_model': args.d_model,
        'n_tasks': n_tasks,
        'elapsed_seconds': elapsed,
        'per_part_aggregated': result['summary'].get('per_part', []),
    }
    summary_path = os.path.join(out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n[per-part 5shot] DONE in {elapsed:.1f}s", flush=True)
    print(f"  results: {out_dir}", flush=True)


if __name__ == '__main__':
    main()
