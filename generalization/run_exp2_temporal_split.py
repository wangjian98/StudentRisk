"""EXP-2: Temporal-split generalization experiment for StudentRisk v3.1.

Chronologically partitions the student pool by each student's *last event*
timestamp: the earliest ``split_pct`` of students become the training set,
the latest ``1 - split_pct`` become the test set. This simulates the
realistic deployment scenario where a model is trained on past cohorts and
deployed on future cohorts — *no* future-label leakage.

Compared to EXP-1 (random splits), EXP-2 answers the question:
  *"Does the model degrade when tested on students from a later cohort?"*

Design:
  - For each ``split_pct ∈ (0.50, 0.60, 0.70, 0.80)``, we get ONE deterministic
    (train, test) split from ``data_splits.temporal_split()``.
  - Each model is then re-trained with 3 random seeds (42, 123, 777) for
    variance estimation. The split itself does not depend on the seed —
    that variance comes purely from model stochasticity (init, dropout,
    batch shuffling).
  - Total runs = 10 models × 4 ratios × 3 seeds = 120 runs.

Outputs (under ``outputs/generalization/exp2_temporal/``):
  - results.csv       one row per (model, train_ratio, seed)
  - summary.csv       one row per (model, train_ratio) with mean/std across seeds
  - config.json       snapshot of the experiment configuration
  - errors.csv        one row per failed run (graceful degradation)

Aligned with the existing OOF protocol:
  - Failed=1, Passed=0
  - Threshold = 0.5
  - Same set of 3 seeds (42, 123, 777) as OOF, for variance comparability

The smoke-test invocation (must complete in < 5 minutes):
  python -m generalization.run_exp2_temporal_split \\
      --models rf7 meta_mamba_7d \\
      --seeds 42 \\
      --ratios 0.80 \\
      --epochs 10 \\
      --synthetic
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
import traceback
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Project path bootstrap (same convention as run_exp1_learning_curve.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Reuse Day 1's helpers: model registry, train/predict dispatch, config loader,
# synthetic-data fallback. This avoids duplicating ~250 lines of plumbing and
# guarantees that EXP-2 uses the *exact same* training/inference code paths
# as EXP-1 (the paper must compare apples to apples).
from generalization.run_exp1_learning_curve import (
    MODEL_REGISTRY,
    load_config,
    pick_device,
    _load_or_synthesize,
    _build_features_for_model,
    _train_and_predict_one,
)
from generalization.data_splits import temporal_split
from models.base import set_seed, evaluate_predictions


# ────────────────────────── defaults ───────────────────────────


DEFAULT_SEEDS: Tuple[int, ...] = (42, 123, 777)
# Four training-set sizes — the smallest (0.50) is the most stressful,
# the largest (0.80) is closest to standard 80/20.
DEFAULT_SPLIT_RATIOS: Tuple[float, ...] = (0.50, 0.60, 0.70, 0.80)


# ────────────────────────── core runner ───────────────────────────


def run(seeds: Tuple[int, ...] = DEFAULT_SEEDS,
        split_ratios: Tuple[float, ...] = DEFAULT_SPLIT_RATIOS,
        models_to_test: Optional[List[str]] = None,
        out_dir: Optional[str] = None,
        config: Optional[dict] = None,
        device: Optional[str] = None,
        epochs_override: Optional[int] = None,
        use_synthetic: bool = False,
        threshold: float = 0.5) -> dict:
    """Run EXP-2 temporal-split experiment.

    Args:
      seeds: random seeds for model re-training (split itself is deterministic).
      split_ratios: tuple of fractions in (0, 1) for the training portion.
      models_to_test: subset of MODEL_REGISTRY keys to evaluate. None = all 10.
      out_dir: directory to save CSV/JSON outputs. None = default under outputs/.
      config: full config dict (loaded from configs/default.yaml if None).
      device: 'cpu' or 'cuda'. None = auto-pick.
      epochs_override: if not None, override epochs in training (used by smoke test).
      use_synthetic: if True, use synthetic data instead of real CS1 dataset.
      threshold: decision threshold for F1 metrics.

    Returns:
      payload dict with overall results + per-run records.
    """
    if config is None:
        config = load_config()
    if device is None:
        device = pick_device()

    if out_dir is None:
        out_dir = os.path.join(_ROOT, 'outputs', 'generalization', 'exp2_temporal')
    os.makedirs(out_dir, exist_ok=True)

    if models_to_test is None:
        models_to_test = list(MODEL_REGISTRY.keys())

    # Sanity-check ratios
    for r in split_ratios:
        if not (0.0 < r < 1.0):
            raise ValueError(f"split_ratios must be in (0, 1), got {r!r}")

    print(f"\n[EXP-2] === START ===", flush=True)
    print(f"[EXP-2] out_dir={out_dir}", flush=True)
    print(f"[EXP-2] device={device}", flush=True)
    print(f"[EXP-2] models={models_to_test}", flush=True)
    print(f"[EXP-2] seeds={seeds}, split_ratios={split_ratios}", flush=True)
    if epochs_override is not None:
        print(f"[EXP-2] epochs_override={epochs_override} (smoke-test mode)", flush=True)

    # Load (or synthesize) dataset
    ide_logs, labels_df, y, student_ids, is_synthetic = _load_or_synthesize(use_synthetic)
    n = len(y)
    print(f"[EXP-2] data: n={n}, fail_rate={y.mean():.4f}, "
          f"n_events={len(ide_logs)}, synthetic={is_synthetic}", flush=True)

    # Pre-compute the (deterministic) temporal splits ONCE per ratio.
    # Each split is then reused across all 3 seeds (only model training varies).
    splits_by_ratio: Dict[float, Tuple[np.ndarray, np.ndarray, dict]] = {}
    for ratio in split_ratios:
        tr_idx, te_idx, meta = temporal_split(ide_logs, labels_df, split_pct=ratio)
        # Sanity: cover all students (no overlap, no leakage)
        assert len(set(tr_idx.tolist()) & set(te_idx.tolist())) == 0, \
            f"train/test overlap for ratio={ratio}"
        assert len(tr_idx) + len(te_idx) == n, \
            f"split does not cover all students (ratio={ratio})"
        # Update meta with dataset-side info
        meta.update({
            'n_total': int(n),
            'y_train_fail_rate': float(y[tr_idx].mean()),
            'y_test_fail_rate':  float(y[te_idx].mean()),
        })
        splits_by_ratio[ratio] = (tr_idx, te_idx, meta)
        print(f"[EXP-2] split ratio={ratio:.2f}: "
              f"n_train={len(tr_idx)}, n_test={len(te_idx)}, "
              f"cutoff={meta.get('cutoff')!s}, "
              f"fail_train={meta['y_train_fail_rate']:.3f}, "
              f"fail_test={meta['y_test_fail_rate']:.3f}", flush=True)

    records: List[dict] = []
    errors: List[dict] = []
    t_global = time.time()

    for model_name in models_to_test:
        if model_name not in MODEL_REGISTRY:
            print(f"[EXP-2] WARNING: unknown model={model_name!r}, skipping", flush=True)
            continue
        model_spec = MODEL_REGISTRY[model_name]

        for ratio in split_ratios:
            tr_idx, te_idx, split_meta = splits_by_ratio[ratio]

            for seed in seeds:
                t0 = time.time()
                rec = {
                    'model': model_name,
                    'train_ratio': float(ratio),
                    'seed': int(seed),
                    'n_train': int(len(tr_idx)),
                    'n_test': int(len(te_idx)),
                    'method': str(split_meta.get('method', 'temporal')),
                }
                err_msg = None
                try:
                    # Build features ONCE per (model, ratio): same train/test
                    # student set across the 3 seeds, so features are identical.
                    feats = _build_features_for_model(
                        model_name, model_spec,
                        ide_logs, student_ids,
                        tr_idx, te_idx,
                    )
                    y_tr = y[tr_idx]
                    y_te = y[te_idx]
                    if len(np.unique(y_tr)) < 2:
                        err_msg = (f"y_tr has only 1 unique value "
                                   f"(n_train={len(tr_idx)}, n_fail={int(y_tr.sum())}). "
                                   f"Skipping.")
                        raise ValueError(err_msg)

                    p_te = _train_and_predict_one(
                        model_name, model_spec,
                        feats, y_tr, y_te,
                        config, seed, epochs_override, device,
                    )
                    metrics = evaluate_predictions(y_te, p_te, threshold=threshold)
                    rec.update({
                        'accuracy':      metrics['accuracy'],
                        'precision_c1':  metrics['precision_class_1'],
                        'recall_c1':     metrics['recall_class_1'],
                        'f1_class_1':    metrics['f1_class_1'],
                        'macro_f1':      metrics['macro_f1'],
                        'roc_auc':       metrics['roc_auc'],
                        'pr_auc':        metrics['pr_auc'],
                        'TN': metrics['confusion_matrix']['TN'],
                        'FP': metrics['confusion_matrix']['FP'],
                        'FN': metrics['confusion_matrix']['FN'],
                        'TP': metrics['confusion_matrix']['TP'],
                        'elapsed_seconds': round(time.time() - t0, 2),
                        'is_synthetic': bool(is_synthetic),
                        'device': device,
                        'feature_dim': int(model_spec['feature_dim']),
                        'model_kind': model_spec['kind'],
                    })
                except Exception as e:
                    err_msg = f"{type(e).__name__}: {e}"
                    rec.update({
                        'accuracy': float('nan'),
                        'precision_c1': float('nan'),
                        'recall_c1': float('nan'),
                        'f1_class_1': float('nan'),
                        'macro_f1': float('nan'),
                        'roc_auc': float('nan'),
                        'pr_auc': float('nan'),
                        'TN': 0, 'FP': 0, 'FN': 0, 'TP': 0,
                        'elapsed_seconds': round(time.time() - t0, 2),
                        'is_synthetic': bool(is_synthetic),
                        'device': device,
                        'feature_dim': int(model_spec['feature_dim']),
                        'model_kind': model_spec['kind'],
                    })
                    errors.append({
                        'model': model_name, 'train_ratio': float(ratio),
                        'seed': int(seed),
                        'n_train': int(len(tr_idx)), 'n_test': int(len(te_idx)),
                        'error': err_msg,
                        'traceback': traceback.format_exc(limit=4),
                    })
                records.append(rec)
                # progress line
                if not np.isnan(rec['f1_class_1']):
                    print(f"[EXP-2] model={model_name:14s} ratio={ratio:.2f} "
                          f"seed={seed} F1(FAIL)={rec['f1_class_1']:.3f} "
                          f"({rec['elapsed_seconds']:.1f}s)", flush=True)
                else:
                    print(f"[EXP-2] model={model_name:14s} ratio={ratio:.2f} "
                          f"seed={seed} FAILED ({err_msg})", flush=True)

    # ─── aggregate ───
    df = pd.DataFrame(records)
    df_err = pd.DataFrame(errors)
    print(f"\n[EXP-2] Total runs: {len(df)} (errors: {len(df_err)})", flush=True)

    # save results.csv
    results_csv = os.path.join(out_dir, 'results.csv')
    df.to_csv(results_csv, index=False)
    print(f"[EXP-2] Saved {results_csv} ({len(df)} rows)", flush=True)

    if not df_err.empty:
        err_csv = os.path.join(out_dir, 'errors.csv')
        df_err.to_csv(err_csv, index=False)
        print(f"[EXP-2] Saved {err_csv} ({len(df_err)} rows)", flush=True)

    # build summary.csv: mean/std across seeds for each (model, train_ratio)
    summary = (
        df.groupby(['model', 'train_ratio'])
        .agg(
            n_runs=('f1_class_1', 'size'),
            acc_mean=('accuracy', 'mean'),
            acc_std=('accuracy', 'std'),
            f1_fail_mean=('f1_class_1', 'mean'),
            f1_fail_std=('f1_class_1', 'std'),
            macro_f1_mean=('macro_f1', 'mean'),
            macro_f1_std=('macro_f1', 'std'),
            roc_auc_mean=('roc_auc', 'mean'),
            roc_auc_std=('roc_auc', 'std'),
            pr_auc_mean=('pr_auc', 'mean'),
            pr_auc_std=('pr_auc', 'std'),
            avg_n_train=('n_train', 'mean'),
            avg_n_test=('n_test', 'mean'),
            avg_elapsed=('elapsed_seconds', 'mean'),
        )
        .reset_index()
    )
    summary_csv = os.path.join(out_dir, 'summary.csv')
    summary.to_csv(summary_csv, index=False)
    print(f"[EXP-2] Saved {summary_csv}", flush=True)

    # save config.json
    cfg_snapshot = {
        'experiment': 'EXP-2 temporal split',
        'seeds': list(seeds),
        'split_ratios': [float(r) for r in split_ratios],
        'models': list(models_to_test),
        'threshold': float(threshold),
        'device': device,
        'epochs_override': epochs_override,
        'use_synthetic': use_synthetic,
        'is_synthetic_data': bool(is_synthetic),
        'n_students': int(n),
        'fail_rate': float(y.mean()),
        'feature_dim_per_model': {k: v['feature_dim'] for k, v in MODEL_REGISTRY.items()},
        'model_kinds': {k: v['kind'] for k, v in MODEL_REGISTRY.items()},
        'split_summaries': {
            f'{r:.2f}': {k: (int(v) if isinstance(v, (int, np.integer)) else
                              float(v) if isinstance(v, (float, np.floating)) else str(v))
                         for k, v in splits_by_ratio[r][2].items()
                         if k not in {'method'}}
            for r in split_ratios
        },
        'split_methods': {f'{r:.2f}': splits_by_ratio[r][2]['method'] for r in split_ratios},
    }
    cfg_path = os.path.join(out_dir, 'config.json')
    with open(cfg_path, 'w') as f:
        json.dump(cfg_snapshot, f, indent=2, default=str)
    print(f"[EXP-2] Saved {cfg_path}", flush=True)

    elapsed_total = time.time() - t_global
    print(f"\n[EXP-2] DONE in {elapsed_total:.1f}s. "
          f"{len(df)} runs, {len(df_err)} errors.", flush=True)

    payload = {
        'config': cfg_snapshot,
        'records': records,
        'errors': errors,
        'summary': summary.to_dict('records'),
        'elapsed_seconds': elapsed_total,
    }
    return payload


# ────────────────────────── CLI ───────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='EXP-2 temporal-split generalization experiment')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Random seeds for model re-training '
                             '(default: 42 123 777)')
    parser.add_argument('--ratios', '--split-ratios', nargs='+', type=float, default=None,
                        dest='ratios',
                        help='Training-set ratios in (0,1) (default: 0.50 0.60 0.70 0.80)')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help=f'Subset of models to run. Available: {list(MODEL_REGISTRY.keys())}')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: outputs/generalization/exp2_temporal)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default=None,
                        help="Force 'cpu' or 'cuda' (default: auto)")
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override training epochs (used for smoke tests)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (when real IDE_logs.csv is unavailable)')
    args = parser.parse_args()

    config = load_config()
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    split_ratios = tuple(args.ratios) if args.ratios else DEFAULT_SPLIT_RATIOS

    run(
        seeds=seeds, split_ratios=split_ratios,
        models_to_test=args.models, out_dir=args.out_dir,
        config=config, device=args.device,
        epochs_override=args.epochs, use_synthetic=args.synthetic,
        threshold=args.threshold,
    )


if __name__ == '__main__':
    main()