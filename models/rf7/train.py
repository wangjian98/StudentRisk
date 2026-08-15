"""RF-7dim training script: 5-fold × 3 seeds OOF on CS1 (7 raw event-count features)."""
import os
import sys
import json
import argparse
import time
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import load_dataset
from models.base import set_seed, evaluate_predictions, save_results
from models.rf.model import RFModel
from models.rf7.data import build_7dim_features, EVENT_TYPES_7


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(_ROOT, 'configs', 'default.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def run(seeds=(42, 123, 777), n_splits: int = 5, threshold: float = 0.5,
        out_dir: str = None, save_oof: bool = True, config: dict = None):
    if config is None:
        config = load_config()
    rf_cfg = config.get('random_forest', {})

    if out_dir is None:
        out_dir = os.path.join(_ROOT, 'results', 'rf7')
    os.makedirs(out_dir, exist_ok=True)

    print(f"[RF-7d] Loading data ...", flush=True)
    ide_logs, labels_df, y, student_ids = load_dataset()
    print(f"[RF-7d] Building 7-dim features (event counts) ...", flush=True)
    X, feat_names = build_7dim_features(ide_logs, student_ids)
    print(f"[RF-7d] X.shape={X.shape}, y.shape={y.shape}, "
          f"fail_rate={y.mean():.4f}", flush=True)

    n = len(y)
    oof = np.zeros(n)
    fold_idx = np.zeros(n, dtype=int)
    fold_records = []

    t0 = time.time()
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx_, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y)):
            set_seed(seed * 1000 + fold_idx_)
            rf = RFModel(
                n_estimators=rf_cfg.get('n_estimators', 200),
                max_depth=rf_cfg.get('max_depth', 10),
                min_samples_split=rf_cfg.get('min_samples_split', 5),
                class_weight=rf_cfg.get('class_weight', 'balanced'),
                random_state=seed * 1000 + fold_idx_,
            )
            rf.fit(X[tr_idx], y[tr_idx])
            p = rf.predict_proba(X[va_idx])
            oof[va_idx] += p
            fold_idx[va_idx] = fold_idx_
            fold_m = evaluate_predictions(y[va_idx], p, threshold=threshold)
            fold_records.append({'seed': seed, 'fold': fold_idx_, **fold_m})
        print(f"[RF-7d]   seed={seed} done at {time.time()-t0:.1f}s", flush=True)

    oof /= len(seeds)
    elapsed = time.time() - t0
    overall = evaluate_predictions(y, oof, threshold=threshold)

    # Feature importance (last fold's model — approximation)
    rf_final = RFModel(
        n_estimators=rf_cfg.get('n_estimators', 200),
        max_depth=rf_cfg.get('max_depth', 10),
        class_weight=rf_cfg.get('class_weight', 'balanced'),
        random_state=42,
    )
    rf_final.fit(X, y)
    fi = rf_final.model.feature_importances_.tolist()
    feature_importance = dict(zip(feat_names, [round(float(x), 4) for x in fi]))

    fold_df = pd.DataFrame(fold_records)
    fold_summary = {
        'macro_f1_mean': float(fold_df['macro_f1'].mean()),
        'macro_f1_std':  float(fold_df['macro_f1'].std()),
        'f1_class_1_mean': float(fold_df['f1_class_1'].mean()),
        'f1_class_1_std':  float(fold_df['f1_class_1'].std()),
        'roc_auc_mean':   float(fold_df['roc_auc'].mean()),
        'roc_auc_std':    float(fold_df['roc_auc'].std()),
    }

    payload = {
        'model': 'RandomForest-7d',
        'config': rf_cfg,
        'feature_dimension': 7,
        'feature_names': feat_names,
        'feature_importance': feature_importance,
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
                   'rf_cfg': rf_cfg, 'feature_dimension': 7}, f, indent=2)

    print(f"\n[RF-7d] DONE in {elapsed:.1f}s. Overall: "
          f"acc={overall['accuracy']:.4f}, macro_f1={overall['macro_f1']:.4f}, "
          f"f1_failed={overall['f1_class_1']:.4f}, roc_auc={overall['roc_auc']:.4f}", flush=True)
    print(f"[RF-7d] Top-3 features by importance: "
          f"{sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]}", flush=True)
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