"""Max_len sweep for MetaMamba on CS1.

Single-variable sweep: max_len ∈ {64, 128, 256, 512, 1024}, all other
hyperparameters fixed at the v3.1 paper defaults.

Usage:
  # Smoke test (synthetic data, fast)
  python -m generalization.run_max_len_sweep --synthetic --seeds 42 --n-splits 5 --epochs 5

  # Real data (246 GPU server)
  python -m generalization.run_max_len_sweep --seeds 42 123 777 --n-splits 5

  # With FOMAML evaluation
  python -m generalization.run_max_len_sweep --seeds 42 123 777 --with-fomaml

Output:
  outputs/generalization/exp_max_len/
    ├── results.csv       (max_len, seed, fold, ...)
    ├── summary.csv       (max_len, mean metrics across 15 folds)
    ├── best_max_len.json
    ├── config_used.json
    └── max_len_curve.png (F1(FAIL) vs max_len + cost overlay)
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import load_dataset
from models.base import set_seed, evaluate_predictions, save_results
from models.meta_mamba.data import build_event_sequences
from models.meta_mamba.model import MetaMambaClassifier
from sklearn.model_selection import StratifiedKFold


# ──────────────────────────── Defaults (v3.1 paper config) ────────────────────────────

DEFAULT_CONFIG = {
    'd_model': 64,
    'd_state': 16,
    'n_layers': 2,
    'dropout': 0.3,
    'lr': 1e-3,
    'weight_decay': 1e-3,
    'epochs': 60,
    'batch_size': 32,
    'patience': 12,
    'contrastive_weight': 0.3,
}

MAX_LEN_VALUES = [64, 128, 256, 512, 1024]


# ──────────────────────────── Train one fold (lite version) ────────────────────────────

def train_one_fold_lite(model, seq_tr, mask_tr, task_tr, y_tr,
                        seq_va, mask_va, task_va, y_va,
                        epochs, lr, weight_decay, batch_size,
                        patience, contrastive_weight, device):
    """Single-fold trainer using MetaMambaClassifier's forward + TC loss.

    Mirrors the supervised + task-contrastive math in
    models/meta_mamba/train.train_one_fold, but uses model.forward() directly.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    seq_tr_t = torch.from_numpy(seq_tr).float().to(device)
    mask_tr_t = torch.from_numpy(mask_tr).float().to(device)
    task_tr_t = torch.from_numpy(task_tr).long().to(device)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    seq_va_t = torch.from_numpy(seq_va).float().to(device)
    mask_va_t = torch.from_numpy(mask_va).float().to(device)
    task_va_t = torch.from_numpy(task_va).long().to(device)
    y_va_t = torch.from_numpy(y_va.astype(np.float32)).to(device)

    model.to(device)
    n_tr = len(y_tr)
    best_val_loss = float('inf')
    best_state = None
    pc = 0

    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n_tr)
        for i in range(0, n_tr, batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            x = seq_tr_t[idx]
            m = mask_tr_t[idx]
            t = task_tr_t[idx]
            y = y_tr_t[idx]
            # Use model.forward (full pipeline) and capture pooled rep via hook
            # Simpler: replicate pool manually
            x_emb = model.event_embed(x)
            x_emb = model.input_norm(x_emb)
            for blk in model.blocks:
                x_emb = blk(x_emb)
            x_film = model.film(x_emb, t)
            mask_f = m.unsqueeze(-1)
            x_sum = (x_film * mask_f).sum(dim=1)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            pooled = x_sum / denom
            pooled = model.pool_norm(pooled)
            logit = model.head(pooled).squeeze(-1)
            sup_loss = criterion(logit, y)

            # Task-contrastive loss (NT-Xent with temperature 0.1, matches paper)
            z = F.normalize(pooled, dim=-1)
            sim = z @ z.T / 0.1
            task_eq = (t.unsqueeze(0) == t.unsqueeze(1)).float()
            eye = torch.eye(len(idx), device=device)
            pos_mask = task_eq - eye
            neg_mask = 1.0 - task_eq
            exp_sim = torch.exp(sim)
            pos_sum = (exp_sim * pos_mask).sum(dim=1)
            neg_sum = (exp_sim * neg_mask).sum(dim=1)
            tc = -torch.log((pos_sum + 1e-8) / (neg_sum + 1e-8) + 1e-8)
            valid = (pos_mask.sum(dim=1) > 0).float()
            if valid.sum() > 0:
                tc_loss = (tc * valid).sum() / (valid.sum() + 1e-8)
            else:
                tc_loss = torch.tensor(0.0, device=device)

            loss = sup_loss + contrastive_weight * tc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            x_emb = model.event_embed(seq_va_t)
            x_emb = model.input_norm(x_emb)
            for blk in model.blocks:
                x_emb = blk(x_emb)
            x_film = model.film(x_emb, task_va_t)
            mask_f = mask_va_t.unsqueeze(-1)
            x_sum = (x_film * mask_f).sum(dim=1)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            pooled = x_sum / denom
            pooled = model.pool_norm(pooled)
            v_logit = model.head(pooled).squeeze(-1)
            v_loss = criterion(v_logit, y_va_t).item()
        if not np.isfinite(v_loss):
            break
        if v_loss < best_val_loss - 1e-4:
            best_val_loss = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ──────────────────────────── Synthetic data fallback ────────────────────────────

def make_synthetic_data(n_students=473, fail_rate=0.66, max_len=256, n_tasks=7, seed=42):
    """Generate synthetic IDE_logs and labels for sandbox smoke testing.

    Mimics the real data shape: each student has up to max_len events,
    11-dim feature vector, and a task_id (problem part 0-6).
    """
    rng = np.random.RandomState(seed)
    event_types = ['text_insert', 'text_remove', 'text_paste',
                   'focus_gained', 'focus_lost', 'run', 'submit']
    records = []
    for s in range(n_students):
        n_events = rng.randint(max_len // 4, max_len * 2)  # 64-512 events
        part = rng.randint(0, n_tasks)
        for e in range(n_events):
            et = rng.choice(len(event_types))
            ts_offset = e * rng.uniform(0.5, 5.0)
            records.append({
                'student': s,
                'part': part + 1,
                'exercise': rng.randint(1, 20),
                'eventType': event_types[et],
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(seconds=ts_offset),
                'timeToDeadline': rng.uniform(0, 30 * 24 * 3600),
            })
    ide_logs = pd.DataFrame(records)
    labels_df = pd.DataFrame({
        'student': np.arange(n_students),
        'passed': rng.choice([True, False], size=n_students, p=[1 - fail_rate, fail_rate]),
    })
    return ide_logs, labels_df


# ──────────────────────────── Main sweep function ────────────────────────────

def run(seeds=(42, 123, 777), n_splits=5, threshold=0.5,
        max_len_values=None, epochs=None, batch_size=None,
        device=None, out_dir=None, synthetic=False):
    if max_len_values is None:
        max_len_values = MAX_LEN_VALUES
    cfg = deepcopy(DEFAULT_CONFIG)
    if epochs is not None:
        cfg['epochs'] = epochs
    if batch_size is not None:
        cfg['batch_size'] = batch_size
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if out_dir is None:
        out_dir = os.path.join(_ROOT, 'outputs', 'generalization', 'exp_max_len')
    os.makedirs(out_dir, exist_ok=True)

    # Load data once
    print(f"[max_len_sweep] device={device}, synthetic={synthetic}", flush=True)
    if synthetic:
        ide_logs, labels_df = make_synthetic_data(n_students=473, fail_rate=0.66, max_len=1024, seed=42)
        labels_df['failed'] = (~labels_df['passed'].astype(bool)).astype(int)
        labels_df = labels_df.sort_values('student').reset_index(drop=True)
        student_ids = labels_df['student'].values
        y = labels_df['failed'].values
    else:
        ide_logs, labels_df, y, student_ids = load_dataset()

    n = len(y)
    n_tasks_real = max(int(labels_df['student'].shape[0]), 7)  # 7 parts in CS1
    print(f"[max_len_sweep] n_students={n}, fail_rate={y.mean():.4f}", flush=True)

    records = []
    for max_len in max_len_values:
        print(f"\n{'=' * 70}", flush=True)
        print(f"[max_len_sweep] max_len={max_len} starting ...", flush=True)
        print(f"{'=' * 70}", flush=True)
        t0 = time.time()
        # Build sequences once per max_len
        sequences, masks, task_ids, event_counts = build_event_sequences(
            ide_logs, student_ids, max_len=max_len,
        )
        print(f"[max_len_sweep] sequences={sequences.shape}, "
              f"avg events/student={event_counts.mean():.0f}", flush=True)
        # CS1 n_tasks = 7 (problem parts are 0-6)
        n_tasks_real = 7

        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold_idx_, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y)):
                set_seed(seed * 1000 + fold_idx_)
                model = MetaMambaClassifier(
                    d_event=sequences.shape[-1],
                    d_model=cfg['d_model'],
                    d_state=cfg['d_state'],
                    n_layers=cfg['n_layers'],
                    n_tasks=n_tasks_real,
                    dropout=cfg['dropout'],
                )
                fold_t0 = time.time()
                model = train_one_fold_lite(
                    model,
                    sequences[tr_idx], masks[tr_idx], task_ids[tr_idx], y[tr_idx],
                    sequences[va_idx], masks[va_idx], task_ids[va_idx], y[va_idx],
                    epochs=cfg['epochs'],
                    lr=cfg['lr'],
                    weight_decay=cfg['weight_decay'],
                    batch_size=cfg['batch_size'],
                    patience=cfg['patience'],
                    contrastive_weight=cfg['contrastive_weight'],
                    device=device,
                )
                # Predict
                model.eval()
                with torch.no_grad():
                    p = torch.sigmoid(model(
                        torch.from_numpy(sequences[va_idx]).float().to(device),
                        torch.from_numpy(masks[va_idx]).float().to(device),
                        torch.from_numpy(task_ids[va_idx]).long().to(device),
                    )).cpu().numpy()
                fold_t = time.time() - fold_t0
                metrics = evaluate_predictions(y[va_idx], p, threshold=threshold)
                rec = {
                    'max_len': max_len,
                    'seed': seed,
                    'fold': fold_idx_,
                    'n_train': len(tr_idx),
                    'n_val': len(va_idx),
                    'avg_events': float(event_counts.mean()),
                    'epochs_used': min(cfg['epochs'], cfg['patience'] + 1),
                    'fold_seconds': round(fold_t, 2),
                    **metrics,
                }
                records.append(rec)
                print(f"[max_len_sweep]   max_len={max_len} seed={seed} fold={fold_idx_} "
                      f"F1(FAIL)={metrics['f1_class_1']:.4f} "
                      f"MacroF1={metrics['macro_f1']:.4f} "
                      f"({fold_t:.1f}s)", flush=True)
        elapsed = time.time() - t0
        print(f"[max_len_sweep] max_len={max_len} done in {elapsed:.1f}s "
              f"(avg per fold {elapsed / (len(seeds) * n_splits):.1f}s)", flush=True)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(out_dir, 'results.csv'), index=False)

    # Summary: aggregate by max_len
    agg_cols = ['accuracy', 'macro_f1', 'f1_class_1', 'roc_auc', 'pr_auc', 'fold_seconds']
    summary = df.groupby('max_len')[agg_cols].agg(['mean', 'std']).round(4)
    summary.columns = ['_'.join(c).strip('_') for c in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)

    # Best max_len by F1(FAIL) mean
    best_row = summary.loc[summary['f1_class_1_mean'].idxmax()]
    best_max_len = int(best_row['max_len'])
    best_f1 = float(best_row['f1_class_1_mean'])
    best_std = float(best_row['f1_class_1_std'])

    best = {
        'best_max_len': best_max_len,
        'best_f1_fail_mean': best_f1,
        'best_f1_fail_std': best_std,
        'best_macro_f1_mean': float(best_row['macro_f1_mean']),
        'best_seconds_per_fold': float(best_row['fold_seconds_mean']),
        'all_results': summary.to_dict(orient='records'),
        'config_used': cfg,
        'seeds': list(seeds),
        'n_splits': n_splits,
        'threshold': threshold,
        'synthetic': synthetic,
    }
    with open(os.path.join(out_dir, 'best_max_len.json'), 'w') as f:
        json.dump(best, f, indent=2)
    with open(os.path.join(out_dir, 'config_used.json'), 'w') as f:
        json.dump({'config': cfg, 'max_len_values': list(max_len_values),
                   'seeds': list(seeds), 'n_splits': n_splits,
                   'threshold': threshold, 'synthetic': synthetic}, f, indent=2)

    # Print summary table
    print(f"\n{'=' * 70}", flush=True)
    print(f"[max_len_sweep] SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\n[max_len_sweep] BEST max_len = {best_max_len} "
          f"F1(FAIL) = {best_f1:.4f} ± {best_std:.4f}", flush=True)
    print(f"[max_len_sweep] Results saved to {out_dir}/", flush=True)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 777])
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--max-lens', nargs='+', type=int, default=MAX_LEN_VALUES)
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs (default: use 60 for real, 5 for synthetic)')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic IDE_logs (for sandbox smoke test)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate max_len_curve.png')
    args = parser.parse_args()

    # Auto-tune epochs for smoke test
    if args.epochs is None:
        args.epochs = 5 if args.synthetic else 60

    best = run(
        seeds=args.seeds, n_splits=args.n_splits, threshold=args.threshold,
        max_len_values=args.max_lens, epochs=args.epochs,
        batch_size=args.batch_size, device=args.device,
        out_dir=args.out_dir, synthetic=args.synthetic,
    )
    if args.plot:
        from generalization.max_len_plot import plot_max_len_curve
        plot_max_len_curve(
            csv_path=os.path.join(best['best_max_len'] and args.out_dir or
                                  _ROOT + '/outputs/generalization/exp_max_len',
                                  'results.csv'),
            out_path=os.path.join(args.out_dir or
                                  _ROOT + '/outputs/generalization/exp_max_len',
                                  'max_len_curve.png'),
        )


if __name__ == '__main__':
    main()