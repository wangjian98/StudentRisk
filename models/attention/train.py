"""Attention (Transformer encoder) training script: 5-fold × 3 seeds OOF on CS1 dataset."""
import os
import sys
import json
import argparse
import time
import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import load_dataset, build_features
from models.base import set_seed, evaluate_predictions, save_results
from models.attention.model import AttentionClassifier
from models.lstm.train import train_one_fold  # shared training loop


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(_ROOT, 'configs', 'default.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def run(seeds=(42, 123, 777), n_splits: int = 5, threshold: float = 0.5,
        out_dir: str = None, save_oof: bool = True, config: dict = None,
        device: str = None):
    if config is None:
        config = load_config()
    mcfg = config.get('attention', {})
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if out_dir is None:
        out_dir = os.path.join(_ROOT, 'results', 'attention')
    os.makedirs(out_dir, exist_ok=True)

    print(f"[ATTN] device={device}", flush=True)
    print(f"[ATTN] Loading data + building features ...", flush=True)
    ide_logs, labels_df, y, student_ids = load_dataset()
    X, feat_names = build_features(ide_logs, student_ids)
    print(f"[ATTN] X.shape={X.shape}", flush=True)

    n = len(y)
    oof = np.zeros(n)
    fold_idx = np.zeros(n, dtype=int)
    fold_records = []

    def san(X_):
        X_ = np.nan_to_num(X_, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.clip(X_, -1e6, 1e6)

    t0 = time.time()
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx_, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y)):
            set_seed(seed * 1000 + fold_idx_)
            sc = StandardScaler().fit(san(X[tr_idx]))
            Xs_tr = sc.transform(san(X[tr_idx]))
            Xs_va = sc.transform(san(X[va_idx]))

            model = AttentionClassifier(
                input_dim=X.shape[1],
                d_model=mcfg.get('d_model', 64),
                n_heads=mcfg.get('n_heads', 4),
                n_layers=mcfg.get('n_layers', 2),
                dim_feedforward=mcfg.get('dim_feedforward', 128),
                dropout=mcfg.get('dropout', 0.3),
            )
            model, _ = train_one_fold(
                model, Xs_tr, y[tr_idx], Xs_va, y[va_idx],
                lr=mcfg.get('lr', 1e-3),
                weight_decay=mcfg.get('weight_decay', 1e-3),
                epochs=mcfg.get('epochs', 60),
                batch_size=mcfg.get('batch_size', 32),
                patience=mcfg.get('patience', 12),
                device=device,
            )
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(torch.from_numpy(Xs_va).float().to(device))).cpu().numpy()
            oof[va_idx] += p
            fold_idx[va_idx] = fold_idx_
            fold_m = evaluate_predictions(y[va_idx], p, threshold=threshold)
            fold_records.append({'seed': seed, 'fold': fold_idx_, **fold_m})
        print(f"[ATTN]   seed={seed} done at {time.time()-t0:.1f}s", flush=True)

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

    payload = {
        'model': 'Attention',
        'config': mcfg,
        'n_params': n_params,
        'threshold': threshold,
        'n_seeds': len(seeds),
        'n_splits': n_splits,
        'n_students': n,
        'fail_rate': float(y.mean()),
        'label_convention': 'Failed=1, Passed=0',
        'overall': overall,
        'per_fold_summary': fold_summary,
        'per_fold': fold_records,
        'elapsed_seconds': elapsed,
    }
    save_results(out_dir, payload)
    fold_df.to_csv(os.path.join(out_dir, 'fold_metrics.csv'), index=False)
    if save_oof:
        np.save(os.path.join(out_dir, 'oof_probs.npy'), oof)
        np.save(os.path.join(out_dir, 'labels.npy'), y)
        np.save(os.path.join(out_dir, 'fold_idx.npy'), fold_idx)
        np.save(os.path.join(out_dir, 'student_ids.npy'), student_ids)
    with open(os.path.join(out_dir, 'config_used.json'), 'w') as f:
        json.dump({'seeds': list(seeds), 'n_splits': n_splits, 'threshold': threshold,
                   'attention_cfg': mcfg, 'n_params': n_params}, f, indent=2)

    print(f"\n[ATTN] DONE in {elapsed:.1f}s. Overall: "
          f"acc={overall['accuracy']:.4f}, macro_f1={overall['macro_f1']:.4f}, "
          f"f1_failed={overall['f1_class_1']:.4f}, roc_auc={overall['roc_auc']:.4f}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--no-oof', action='store_true')
    args = parser.parse_args()
    config = load_config()
    seeds = args.seeds if args.seeds else config.get('cv', {}).get('seeds', [42, 123, 777])
    n_splits = args.n_splits or config.get('cv', {}).get('n_splits', 5)
    run(seeds=seeds, n_splits=n_splits, threshold=args.threshold,
        out_dir=args.out_dir, save_oof=not args.no_oof, config=config)


if __name__ == '__main__':
    main()