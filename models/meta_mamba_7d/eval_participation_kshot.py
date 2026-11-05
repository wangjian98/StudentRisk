"""Per-part balanced K-shot FOMAML using PARTICIPATION-based task definition (Plan B).

For each part p ∈ {1..7}:
  - Pool = students with ≥1 event in part p
  - For K ∈ {5, 10, 15, 20}:
    - Filter pool by label, sample K passed + K failed as support (or skip if not enough)
    - Query = remaining pool members
    - Override task_id → (p-1) for both support and query (consistent FiLM conditioning)
    - FOMAML adapt 3 steps, predict on query, compute full metrics
  - Repeat with N sample seeds for stability

Key difference from Plan A (dominant part):
  - Pool much larger (215-471 vs 0-162) → 7/7 parts feasible at K=5,10,15,20
  - Each student's task_id is forced to current part (consistent FiLM input)

Outputs:
  - results/per_part_kshot_participation/per_part_metrics.csv
  - results/per_part_kshot_participation/per_part_summary.csv
  - results/per_part_kshot_participation/summary.json
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


def full_metrics(y_true, y_prob, threshold=0.5):
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


def fomaml_inner_adapt(model, seq, mask, task, y, criterion, inner_lr=0.01, inner_steps=3, device='cpu'):
    """FOMAML inner loop. Returns adapted state_dict."""
    for _ in range(inner_steps):
        model.train()
        s = torch.from_numpy(seq).float().to(device)
        m = torch.from_numpy(mask).float().to(device)
        t = torch.from_numpy(task).long().to(device)
        y_t = torch.from_numpy(y.astype(np.float32)).to(device)
        logits = model(s, m, t)
        loss = criterion(logits, y_t)
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
        with torch.no_grad():
            for p, g in zip(model.parameters(), grads):
                p.sub_(inner_lr * g)


def predict(model, seq, mask, task, device='cpu'):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(
            torch.from_numpy(seq).float().to(device),
            torch.from_numpy(mask).float().to(device),
            torch.from_numpy(task).long().to(device),
        )).cpu().numpy()
    return p


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--max-len', type=int, default=128)
    p.add_argument('--d-model', type=int, default=64)
    p.add_argument('--d-state', type=int, default=16)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--k-values', nargs='+', type=int, default=[5, 10, 15, 20])
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
    sequences, masks, original_task_ids, _ = out if len(out) == 4 else (*out, None)

    # Build participation matrix: did student i participate in part p (1-indexed)?
    print(f"[B-plan] Building participation matrix ...", flush=True)
    part_per_event = ide_logs[['student', 'part']].drop_duplicates()
    part_per_event['part'] = part_per_event['part'].astype(int)
    sid_to_idx = {sid: i for i, sid in enumerate(student_ids)}
    participation = np.zeros((len(student_ids), 7), dtype=bool)
    for _, row in part_per_event.iterrows():
        if row['student'] in sid_to_idx:
            i = sid_to_idx[row['student']]
            p_idx = int(row['part']) - 1  # 0-indexed
            if 0 <= p_idx < 7:
                participation[i, p_idx] = True

    n_tasks = 7
    print(f"[B-plan] sequences={sequences.shape}, n_tasks={n_tasks}", flush=True)
    print(f"[B-plan] Per-part participation counts:", flush=True)
    for p_idx in range(7):
        n_p = int(participation[:, p_idx].sum())
        n_pass = int(((y == 0) & participation[:, p_idx]).sum())
        n_fail = int(((y == 1) & participation[:, p_idx]).sum())
        print(f"  part {p_idx+1}: n={n_p}, passed={n_pass}, failed={n_fail}, min_class={min(n_pass, n_fail)}", flush=True)

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
    print(f"[B-plan] ckpt loaded, device={device}", flush=True)

    jsonl_path = os.path.join(args.out_dir, 'per_part_runs.jsonl')
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    criterion = nn.BCEWithLogitsLoss()
    all_runs = []

    t0 = time.time()
    for part in range(1, n_tasks + 1):
        p_idx = part - 1
        pool_idx = np.where(participation[:, p_idx])[0]
        y_pool = y[pool_idx]
        n_pass = int((y_pool == 0).sum())
        n_fail = int((y_pool == 1).sum())
        print(f"\n=== Part {part} (pool={len(pool_idx)}, passed={n_pass}, failed={n_fail}) ===", flush=True)

        for K in args.k_values:
            print(f"\n  -- K={K} --", flush=True)
            if n_pass < K or n_fail < K:
                print(f"    SKIP (need ≥{K} per class, have passed={n_pass} failed={n_fail})", flush=True)
                all_runs.append({
                    'part': part, 'K': K, 'seed': -1, 'skip': True,
                    'reason': f'need >= {K} per class',
                    'n_pass': n_pass, 'n_fail': n_fail,
                })
                continue

            idx_pass = pool_idx[y_pool == 0]
            idx_fail = pool_idx[y_pool == 1]

            for seed in range(args.n_seeds):
                np.random.seed(seed * 1000 + part * 100 + K)
                sup_p = np.random.choice(idx_pass, size=K, replace=False)
                sup_f = np.random.choice(idx_fail, size=K, replace=False)
                sup_idx = np.concatenate([sup_p, sup_f])
                used = np.isin(pool_idx, sup_idx)
                qry_idx = pool_idx[~used]
                if len(qry_idx) > args.n_query_max:
                    np.random.shuffle(qry_idx)
                    qry_idx = qry_idx[:args.n_query_max]

                # Override task_id to current part (consistent FiLM input)
                task_sup = np.full(len(sup_idx), p_idx, dtype=np.int64)
                task_qry = np.full(len(qry_idx), p_idx, dtype=np.int64)

                # Backup θ
                theta = {k: v.clone() for k, v in model.state_dict().items()}

                # FOMAML adapt
                fomaml_inner_adapt(
                    model,
                    sequences[sup_idx], masks[sup_idx], task_sup, y[sup_idx],
                    criterion, inner_lr=args.inner_lr, inner_steps=args.inner_steps,
                    device=device,
                )

                # Predict query
                q_prob = predict(
                    model, sequences[qry_idx], masks[qry_idx], task_qry,
                    device=device,
                )

                # Restore θ
                model.load_state_dict(theta)

                # Metrics
                m = full_metrics(y[qry_idx], q_prob, threshold=args.threshold)
                m.update({
                    'part': part, 'K': K, 'seed': seed,
                    'n_support': len(sup_idx),
                    'n_query': int(len(qry_idx)),
                    'qry_n_passed': int((y[qry_idx] == 0).sum()),
                    'qry_n_failed': int((y[qry_idx] == 1).sum()),
                    'inner_lr': args.inner_lr, 'inner_steps': args.inner_steps,
                })
                all_runs.append(m)
                with open(jsonl_path, 'a') as f:
                    f.write(json.dumps(m) + '\n')

                print(f"    seed={seed}: qry_n={m['n_query']}, "
                      f"acc={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f} "
                      f"f1_fail={m['f1_failed']:.3f} roc_auc={m['roc_auc']:.3f}", flush=True)

    elapsed = time.time() - t0
    print(f"\n[B-plan] DONE in {elapsed:.1f}s", flush=True)

    # Save
    df = pd.DataFrame([r for r in all_runs if 'accuracy' in r])
    if len(df) > 0:
        df.to_csv(os.path.join(args.out_dir, 'per_part_metrics.csv'), index=False)
        agg = df.groupby(['part', 'K']).agg(
            n_runs=('seed', 'count'),
            n_query_mean=('n_query', 'mean'),
            qry_n_passed_mean=('qry_n_passed', 'mean'),
            qry_n_failed_mean=('qry_n_failed', 'mean'),
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
        agg.to_csv(os.path.join(args.out_dir, 'per_part_summary.csv'), index=False)
        print(f"\n[B-plan] Per-part × K summary:", flush=True)
        print(agg.to_string(), flush=True)

    summary = {
        'plan': 'B (participation-based task definition)',
        'k_values': args.k_values,
        'n_seeds': args.n_seeds,
        'inner_lr': args.inner_lr,
        'inner_steps': args.inner_steps,
        'threshold': args.threshold,
        'elapsed_seconds': elapsed,
        'n_tasks': n_tasks,
        'task_id_override': 'all students in part p forced to task_id=p-1 during eval',
        'note': 'Model weights unchanged (reused from per_part_5shot_full training)',
        'per_part_k_aggregated': (
            agg.to_dict(orient='records') if len(df) > 0 else []
        ),
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[B-plan] Results saved to {args.out_dir}", flush=True)


if __name__ == '__main__':
    main()
