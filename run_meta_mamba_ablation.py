"""Ablation runner for Meta-Mamba-11d.

Drives the same 5-fold × 3-seed protocol as the v1 paper but with
explicit ablation flags (use_film, contrastive_weight) and writes
results into a per-variant directory under results/.

Usage:
    python run_meta_mamba_ablation.py --variant full
    python run_meta_mamba_ablation.py --variant no_tc
    python run_meta_mamba_ablation.py --variant no_film
    python run_meta_mamba_ablation.py --variant no_film_no_tc
    python run_meta_mamba_ablation.py --variant lam_0_0 --tc-weight 0.0
    python run_meta_mamba_ablation.py --variant lam_0_1 --tc-weight 0.1
    python run_meta_mamba_ablation.py --variant lam_0_5 --tc-weight 0.5
    python run_meta_mamba_ablation.py --variant lam_1_0 --tc-weight 1.0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

_PROJECT = os.path.abspath(os.path.dirname(__file__))
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from models.meta_mamba.model import MetaMambaClassifier
from models.meta_mamba.data import build_event_sequences
from models.meta_mamba.train import train_one_fold, task_contrastive_loss, run_maml_fewshot
from data.data_loader import load_dataset


def load_yaml_config(path: str = None) -> dict:
    """Load config from configs/default.yaml (the one used by meta_mamba/train.py)."""
    try:
        import yaml
    except Exception:
        return {}
    if path is None:
        path = os.path.join(_PROJECT, 'configs', 'default.yaml')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f)
from models.meta_mamba.data import build_event_sequences
from models.meta_mamba.train import train_one_fold, task_contrastive_loss, run_maml_fewshot
from data.data_loader import load_dataset


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    from sklearn.metrics import (
        f1_score, roc_auc_score, average_precision_score,
        accuracy_score, confusion_matrix,
    )
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision_class_0': float(TN / max(TN + FP, 1)),
        'recall_class_0': float(TN / max(TN + FN, 1)),
        'f1_class_0': float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'support_class_0': int((y_true == 0).sum()),
        'precision_class_1': float(TP / max(TP + FP, 1)),
        'recall_class_1': float(TP / max(TP + FN, 1)),
        'f1_class_1': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'support_class_1': int((y_true == 1).sum()),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'pr_auc': float(average_precision_score(y_true, y_prob)),
        'confusion_matrix': {'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP},
    }


def run_ablation(variant, use_film, contrastive_weight, seeds, n_splits, max_len, threshold, out_dir, run_fewshot=False):
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Ablation-{variant}] device={device} use_film={use_film} tc_weight={contrastive_weight}", flush=True)

    # load config
    config = load_yaml_config()
    mcfg = config.get('meta_mamba', config.get('attention', {}))

    print(f"[Ablation-{variant}] Loading data ...", flush=True)
    ide_logs, labels_df, y, student_ids = load_dataset()
    sequences, masks, task_ids, event_counts = build_event_sequences(
        ide_logs, student_ids, max_len=max_len,
    )
    n_tasks = int(task_ids.max()) + 1
    print(f"[Ablation-{variant}] sequences={sequences.shape} n_tasks={n_tasks}", flush=True)

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
                contrastive_weight=contrastive_weight,
                use_film=use_film,
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
        print(f"[Ablation-{variant}]   seed={seed} done at {time.time()-t0:.1f}s", flush=True)

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
    n_params = sum(p.numel() for model in [model] if True for p in model.parameters() if p.requires_grad)

    fewshot = None
    if run_fewshot:
        try:
            fewshot = run_maml_fewshot(
                model, sequences, masks, task_ids, y,
                n_way=min(n_tasks, 4), k_shot=5, n_query=10,
                inner_lr=0.01, inner_steps=3, seeds=(42,), device=device,
            )
            print(f"[Ablation-{variant}] FOMAML few-shot F1 = "
                  f"{fewshot['mean_f1']:.4f} ± {fewshot['std_f1']:.4f}", flush=True)
        except Exception as e:
            print(f"[Ablation-{variant}] Few-shot eval failed: {e}", flush=True)
            fewshot = {'error': str(e)}

    payload = {
        'model': 'MetaMamba',
        'ablation': {'no_film': not use_film, 'no_tc': contrastive_weight == 0.0},
        'feature_dimension': sequences.shape[-1],
        'config': mcfg,
        'tc_weight': contrastive_weight,
        'use_film': use_film,
        'n_params': n_params,
        'threshold': threshold,
        'n_seeds': len(seeds),
        'n_splits': n_splits,
        'n_students': n,
        'fail_rate': float(y.mean()),
        'n_tasks': n_tasks,
        'label_convention': 'Failed=1, Passed=0',
        'input_type': f'event_sequence_{sequences.shape[-1]}d',
        'overall': overall,
        'per_fold_summary': fold_summary,
        'per_fold': fold_records,
        'fewshot_fomaml': fewshot,
        'elapsed_seconds': elapsed,
    }

    os.makedirs(out_dir, exist_ok=True)
    json.dump(payload, open(os.path.join(out_dir, 'results.json'), 'w'), indent=2)
    np.save(os.path.join(out_dir, 'oof_probs.npy'), oof)
    np.save(os.path.join(out_dir, 'fold_idx.npy'), fold_idx)
    np.save(os.path.join(out_dir, 'labels.npy'), y)
    np.save(os.path.join(out_dir, 'task_ids.npy'), task_ids)
    pd.DataFrame(fold_records).to_csv(os.path.join(out_dir, 'fold_metrics.csv'), index=False)
    json.dump({'seeds': list(seeds), 'n_splits': n_splits, 'threshold': threshold,
               'meta_mamba_cfg': mcfg, 'n_params': n_params,
               'max_len': max_len, 'n_tasks': n_tasks,
               'ablation': payload['ablation'],
               'tc_weight': contrastive_weight,
               'use_film': use_film},
              open(os.path.join(out_dir, 'config_used.json'), 'w'), indent=2)

    print(f"\n[Ablation-{variant}] DONE in {elapsed:.1f}s. Overall: "
          f"acc={overall['accuracy']:.4f} macro_f1={overall['macro_f1']:.4f} "
          f"f1_failed={overall['f1_class_1']:.4f} auc={overall['roc_auc']:.4f}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True,
                        choices=['full', 'no_tc', 'no_film', 'no_film_no_tc',
                                 'lam_0_0', 'lam_0_1', 'lam_0_3', 'lam_0_5', 'lam_1_0'])
    parser.add_argument('--tc-weight', type=float, default=None)
    parser.add_argument('--no-film', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 777])
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--max-len', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--no-fewshot', action='store_true')
    args = parser.parse_args()

    if args.variant == 'full':
        use_film, tc_weight = True, 0.3
    elif args.variant == 'no_tc':
        use_film, tc_weight = True, 0.0
    elif args.variant == 'no_film':
        use_film, tc_weight = False, 0.3
    elif args.variant == 'no_film_no_tc':
        use_film, tc_weight = False, 0.0
    else:
        use_film = True
        tc_weight = float(args.variant.split('_')[1] + '.' + args.variant.split('_')[2])
    if args.tc_weight is not None:
        tc_weight = args.tc_weight
    if args.no_film:
        use_film = False

    out_dir = args.out_dir or f'results/meta_mamba_ablation_{args.variant}'

    run_ablation(
        variant=args.variant,
        use_film=use_film,
        contrastive_weight=tc_weight,
        seeds=args.seeds,
        n_splits=args.n_splits,
        max_len=args.max_len,
        threshold=args.threshold,
        out_dir=out_dir,
        run_fewshot=not args.no_fewshot,
    )


if __name__ == '__main__':
    main()