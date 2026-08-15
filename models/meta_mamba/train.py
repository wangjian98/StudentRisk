"""Meta-Mamba training script.

Implements:
  1) Standard supervised training with task-aware FiLM modulation
  2) Task-contrastive auxiliary loss (proxy for temporal contrastive pretraining)
  3) MAML-style K-shot few-shot adaptation evaluation (proxy for cross-curriculum
     generalization — uses problem part as task grouping)

Label convention: Failed=1 (positive class).
"""
import os
import sys
import json
import argparse
import time
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import load_dataset
from models.base import set_seed, evaluate_predictions, save_results
from models.meta_mamba.model import MetaMambaClassifier
from models.meta_mamba.data import build_event_sequences


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(_ROOT, 'configs', 'default.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def task_contrastive_loss(features: torch.Tensor, task_ids: torch.Tensor, temperature: float = 0.1):
    """NT-Xent style task-contrastive loss (proxy for temporal contrastive).

    Pulls together embeddings of students in the same task (problem part),
    pushes apart embeddings of students in different tasks.

    Args:
      features: (B, d_model) pooled embeddings
      task_ids:  (B,) integer task ids
      temperature: τ for NT-Xent
    """
    B = features.size(0)
    if B < 2:
        return torch.tensor(0.0, device=features.device)
    z = F.normalize(features, dim=-1)
    sim = z @ z.T / temperature                                # (B, B)
    # Mask: positives = same task
    task_eq = (task_ids.unsqueeze(0) == task_ids.unsqueeze(1)).float()
    # Exclude self-similarity
    eye = torch.eye(B, device=features.device)
    pos_mask = task_eq - eye
    neg_mask = 1.0 - task_eq
    # For each anchor i, loss = -log(sum_pos exp(sim) / sum_neg exp(sim))
    exp_sim = torch.exp(sim)
    eps = 1e-8
    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    neg_sum = (exp_sim * neg_mask).sum(dim=1)
    loss = -torch.log(pos_sum / (neg_sum + eps) + eps)
    # Only count anchors with at least one positive
    valid = (pos_mask.sum(dim=1) > 0).float()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)
    return (loss * valid).sum() / (valid.sum() + eps)


def train_one_fold(model, seq_tr, mask_tr, task_tr, y_tr,
                   seq_va, mask_va, task_va, y_va,
                   epochs=40, lr=1e-3, weight_decay=1e-3,
                   batch_size=16, patience=10,
                   contrastive_weight=0.3,
                   device='cpu'):
    """Train Meta-Mamba with supervised + task-contrastive loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    seq_tr_t   = torch.from_numpy(seq_tr).float().to(device)
    mask_tr_t  = torch.from_numpy(mask_tr).float().to(device)
    task_tr_t  = torch.from_numpy(task_tr).long().to(device)
    y_tr_t     = torch.from_numpy(y_tr.astype(np.float32)).to(device)

    seq_va_t   = torch.from_numpy(seq_va).float().to(device)
    mask_va_t  = torch.from_numpy(mask_va).float().to(device)
    task_va_t  = torch.from_numpy(task_va).long().to(device)
    y_va_t     = torch.from_numpy(y_va.astype(np.float32)).to(device)

    model.to(device)
    n_tr = len(y_tr)
    best_val_loss = float('inf')
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    pc = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_tr)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n_tr, batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()

            # Forward pass through Mamba backbone to get pooled embedding (for contrastive)
            x = model.event_embed(seq_tr_t[idx])
            x = model.input_norm(x)
            for blk in model.blocks:
                x = blk(x)
            x_film = model.film(x, task_tr_t[idx])
            mask_f = mask_tr_t[idx].unsqueeze(-1)
            x_sum = (x_film * mask_f).sum(dim=1)
            denom = mask_f.sum(dim=1).clamp(min=1.0)
            pooled = x_sum / denom
            pooled = model.pool_norm(pooled)

            # Supervised head
            logit = model.head(pooled).squeeze(-1)
            sup_loss = criterion(logit, y_tr_t[idx])

            # Task-contrastive auxiliary loss
            tc_loss = task_contrastive_loss(pooled, task_tr_t[idx])

            loss = sup_loss + contrastive_weight * tc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = model(seq_va_t, mask_va_t, task_va_t)
            v_loss = criterion(v_logits, y_va_t).item()
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

    model.load_state_dict(best_state)
    return model, best_val_loss


def run_maml_fewshot(model, sequences, masks, task_ids, y,
                     n_way: int = 4, k_shot: int = 5, n_query: int = 10,
                     inner_lr: float = 0.01, inner_steps: int = 3,
                     seeds=(42, 123, 777), device='cpu') -> dict:
    """MAML-style K-shot few-shot evaluation.

    Simulates cross-curriculum generalization: each "task" = a problem part.
    For each task:
      - Sample K students as support set
      - Sample N query students
      - Inner loop: θ' = θ - inner_lr * ∇_θ L_support (first-order, no second derivative)
      - Outer eval: loss on query set
    Reports mean / std of query-set F1 across all tasks × seeds.

    Note: This is a simplified first-order MAML (FOMAML) for evaluation only.
    """
    print(f"[meta_mamba] Few-shot eval: n_way={n_way}, k_shot={k_shot}, "
          f"n_query={n_query}, inner_steps={inner_steps}", flush=True)
    criterion = nn.BCEWithLogitsLoss()

    unique_tasks = np.unique(task_ids)
    all_f1 = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        for tid in unique_tasks:
            mask_t = (task_ids == tid)
            idx_t = np.where(mask_t)[0]
            if len(idx_t) < k_shot + n_query:
                continue
            # K support + N query
            np.random.shuffle(idx_t)
            sup_idx = idx_t[:k_shot]
            qry_idx = idx_t[k_shot:k_shot + n_query]

            # Save θ
            theta = {k: v.clone() for k, v in model.state_dict().items()}

            # Inner-loop: adapt on support
            for _ in range(inner_steps):
                model.train()
                # Forward support
                logits = model(
                    torch.from_numpy(sequences[sup_idx]).float().to(device),
                    torch.from_numpy(masks[sup_idx]).float().to(device),
                    torch.from_numpy(task_ids[sup_idx]).long().to(device),
                )
                loss = criterion(logits, torch.from_numpy(y[sup_idx].astype(np.float32)).to(device))
                grads = torch.autograd.grad(loss, model.parameters(), create_graph=False)
                # Manual SGD update (first-order)
                with torch.no_grad():
                    for p, g in zip(model.parameters(), grads):
                        p.sub_(inner_lr * g)

            # Evaluate on query
            model.eval()
            with torch.no_grad():
                q_logits = model(
                    torch.from_numpy(sequences[qry_idx]).float().to(device),
                    torch.from_numpy(masks[qry_idx]).float().to(device),
                    torch.from_numpy(task_ids[qry_idx]).long().to(device),
                )
                q_prob = torch.sigmoid(q_logits).cpu().numpy()
            y_q = y[qry_idx]
            # F1 at threshold 0.5
            y_pred = (q_prob >= 0.5).astype(int)
            from sklearn.metrics import f1_score
            f1 = f1_score(y_q, y_pred, zero_division=0)
            all_f1.append(f1)

            # Restore θ
            model.load_state_dict(theta)

    if not all_f1:
        return {'mean_f1': float('nan'), 'std_f1': float('nan'), 'n_tasks_evaluated': 0}
    return {
        'mean_f1':  float(np.mean(all_f1)),
        'std_f1':   float(np.std(all_f1)),
        'n_tasks_evaluated': len(all_f1),
        'k_shot': k_shot,
        'n_query': n_query,
        'inner_steps': inner_steps,
        'method': 'FOMAML (first-order)',
    }


def run(seeds=(42, 123, 777), n_splits: int = 5, threshold: float = 0.5,
        out_dir: str = None, save_oof: bool = True, config: dict = None,
        device: str = None, max_len: int = 256, run_fewshot: bool = True):
    if config is None:
        config = load_config()
    mcfg = config.get('meta_mamba', config.get('attention', {}))
    # Use attention's defaults as fallback
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if out_dir is None:
        out_dir = os.path.join(_ROOT, 'results', 'meta_mamba')
    os.makedirs(out_dir, exist_ok=True)

    print(f"[MetaMamba] device={device}", flush=True)
    print(f"[MetaMamba] Loading data ...", flush=True)
    ide_logs, labels_df, y, student_ids = load_dataset()

    print(f"[MetaMamba] Building event sequences (max_len={max_len}) ...", flush=True)
    sequences, masks, task_ids, event_counts = build_event_sequences(
        ide_logs, student_ids, max_len=max_len,
    )
    n_tasks = int(task_ids.max()) + 1
    print(f"[MetaMamba] sequences={sequences.shape}, n_tasks={n_tasks}, "
          f"avg events={event_counts.mean():.0f}", flush=True)

    n = len(y)
    oof = np.zeros(n)
    fold_idx = np.zeros(n, dtype=int)
    fold_records = []

    t0 = time.time()
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx_, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y)):
            set_seed(seed * 1000 + fold_idx_)
            model = MetaMambaClassifier(
                d_event=sequences.shape[-1],
                d_model=mcfg.get('d_model', 64),
                d_state=mcfg.get('d_state', 16),
                n_layers=mcfg.get('n_layers', 2),
                n_tasks=n_tasks,
                dropout=mcfg.get('dropout', 0.2),
            )
            model, _ = train_one_fold(
                model,
                sequences[tr_idx], masks[tr_idx], task_ids[tr_idx], y[tr_idx],
                sequences[va_idx], masks[va_idx], task_ids[va_idx], y[va_idx],
                epochs=mcfg.get('epochs', 40),
                lr=mcfg.get('lr', 1e-3),
                weight_decay=mcfg.get('weight_decay', 1e-3),
                batch_size=mcfg.get('batch_size', 16),
                patience=mcfg.get('patience', 10),
                contrastive_weight=mcfg.get('contrastive_weight', 0.3),
                device=device,
            )
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(
                    torch.from_numpy(sequences[va_idx]).float().to(device),
                    torch.from_numpy(masks[va_idx]).float().to(device),
                    torch.from_numpy(task_ids[va_idx]).long().to(device),
                )).cpu().numpy()
            oof[va_idx] += p
            fold_idx[va_idx] = fold_idx_
            fold_m = evaluate_predictions(y[va_idx], p, threshold=threshold)
            fold_records.append({'seed': seed, 'fold': fold_idx_, **fold_m})
        print(f"[MetaMamba]   seed={seed} done at {time.time()-t0:.1f}s", flush=True)

    oof /= len(seeds)
    elapsed = time.time() - t0
    overall = evaluate_predictions(y, oof, threshold=threshold)
    fold_df = pd.DataFrame(fold_records)
    fold_summary = {
        'macro_f1_mean': float(fold_df['macro_f1'].mean()),
        'macro_f1_std':  float(fold_df['macro_f1'].std()),
        'f1_class_1_mean': float(fold_df['f1_class_1'].mean()),
        'f1_class_1_std':  float(fold_df['f1_class_1'].std()),
        'roc_auc_mean':   float(fold_df['roc_auc'].mean()),
        'roc_auc_std':    float(fold_df['roc_auc'].std()),
    }
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Few-shot evaluation (FOMAML) on the last fold's model
    fewshot = None
    if run_fewshot:
        try:
            fewshot = run_maml_fewshot(
                model, sequences, masks, task_ids, y,
                n_way=min(n_tasks, 4), k_shot=5, n_query=10,
                inner_lr=0.01, inner_steps=3, seeds=(42,), device=device,
            )
            print(f"[MetaMamba] FOMAML few-shot F1 = "
                  f"{fewshot['mean_f1']:.4f} ± {fewshot['std_f1']:.4f} "
                  f"on {fewshot['n_tasks_evaluated']} tasks", flush=True)
        except Exception as e:
            print(f"[MetaMamba] Few-shot eval failed: {e}", flush=True)
            fewshot = {'error': str(e)}

    payload = {
        'model': 'MetaMamba',
        'config': mcfg,
        'n_params': n_params,
        'threshold': threshold,
        'n_seeds': len(seeds),
        'n_splits': n_splits,
        'n_students': n,
        'fail_rate': float(y.mean()),
        'n_tasks': n_tasks,
        'label_convention': 'Failed=1, Passed=0',
        'overall': overall,
        'per_fold_summary': fold_summary,
        'per_fold': fold_records,
        'fewshot_fomaml': fewshot,
        'elapsed_seconds': elapsed,
    }
    save_results(out_dir, payload)
    fold_df.to_csv(os.path.join(out_dir, 'fold_metrics.csv'), index=False)
    if save_oof:
        np.save(os.path.join(out_dir, 'oof_probs.npy'), oof)
        np.save(os.path.join(out_dir, 'labels.npy'), y)
        np.save(os.path.join(out_dir, 'fold_idx.npy'), fold_idx)
        np.save(os.path.join(out_dir, 'student_ids.npy'), student_ids)
        np.save(os.path.join(out_dir, 'task_ids.npy'), task_ids)
    with open(os.path.join(out_dir, 'config_used.json'), 'w') as f:
        json.dump({'seeds': list(seeds), 'n_splits': n_splits, 'threshold': threshold,
                   'meta_mamba_cfg': mcfg, 'n_params': n_params,
                   'max_len': max_len, 'n_tasks': n_tasks}, f, indent=2)

    print(f"\n[MetaMamba] DONE in {elapsed:.1f}s. Overall: "
          f"acc={overall['accuracy']:.4f}, macro_f1={overall['macro_f1']:.4f}, "
          f"f1_failed={overall['f1_class_1']:.4f}, roc_auc={overall['roc_auc']:.4f}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description='Train Meta-Mamba on CS1')
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--max-len', type=int, default=256)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--no-oof', action='store_true')
    parser.add_argument('--no-fewshot', action='store_true')
    args = parser.parse_args()
    config = load_config()
    seeds = args.seeds if args.seeds else config.get('cv', {}).get('seeds', [42, 123, 777])
    n_splits = args.n_splits or config.get('cv', {}).get('n_splits', 5)
    run(seeds=seeds, n_splits=n_splits, threshold=args.threshold,
        out_dir=args.out_dir, save_oof=not args.no_oof, config=config,
        max_len=args.max_len, run_fewshot=not args.no_fewshot)


if __name__ == '__main__':
    main()