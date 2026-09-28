"""EXP-3: Leave-One-Problem-Part-Out (LOPO) experiment for StudentRisk v3.1.

Tests whether a model trained on 6 of 7 problem parts generalizes to the
*held-out* part. Each part represents a curriculum sub-topic; LOPO is the
harshest generalization test in the v3.1 paper because the model must
transfer across topic boundaries.

Compared to EXP-1/2, EXP-3 answers the question:
  *"Can the model transfer to a *new* problem topic it has never seen
    during training?"*

Design:
  - ``lopo_split()`` partitions students by their *dominant* problem part.
    For each part ``p ∈ [0, n_parts)``:
        test  = students with dominant part == p
        train = students with dominant part != p
  - There are exactly ``n_parts`` deterministic (train, test) splits.
  - Each model is re-trained with 3 random seeds (42, 123, 777) for
    variance estimation. The split itself is deterministic.
  - Total runs = 10 models × 7 parts × 3 seeds = 210 runs.

Outputs (under ``outputs/generalization/exp3_lopo/``):
  - results.csv       one row per (model, left_out_part, seed)
  - summary.csv       one row per (model) with mean/std across 7 parts × 3 seeds
  - per_part_summary.csv  one row per (model, left_out_part) — part-level stats
  - config.json       snapshot of the experiment configuration
  - errors.csv        one row per failed run (graceful degradation)

Aligned with the existing OOF protocol:
  - Failed=1, Passed=0
  - Threshold = 0.5
  - Same set of 3 seeds (42, 123, 777) as OOF, for variance comparability

Known limitations (printed by data_splits.lopo_split()):
  - Some parts may have < 20 students → results may have high variance
    (a warning is emitted, run is kept).

The smoke-test invocation (must complete in < 5 minutes):
  python -m generalization.run_exp3_lopo \\
      --models rf7 meta_mamba_7d \\
      --seeds 42 \\
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
# synthetic-data fallback. Same justification as in run_exp2_temporal_split.py.
from generalization.run_exp1_learning_curve import (
    MODEL_REGISTRY,
    load_config,
    pick_device,
    _load_or_synthesize,
    _build_features_for_model,
    _train_and_predict_one,
)
from generalization.data_splits import lopo_split
from models.base import set_seed, evaluate_predictions


# ────────────────────────── defaults ───────────────────────────


DEFAULT_SEEDS: Tuple[int, ...] = (42, 123, 777)
DEFAULT_N_PARTS: int = 7          # CS1 course has 7 problem parts


# ────────────────────────── core runner ───────────────────────────


def run(seeds: Tuple[int, ...] = DEFAULT_SEEDS,
        n_parts: int = DEFAULT_N_PARTS,
        models_to_test: Optional[List[str]] = None,
        out_dir: Optional[str] = None,
        config: Optional[dict] = None,
        device: Optional[str] = None,
        epochs_override: Optional[int] = None,
        use_synthetic: bool = False,
        threshold: float = 0.5) -> dict:
    """Run EXP-3 Leave-One-Problem-Part-Out experiment.

    Args:
      seeds: random seeds for model re-training (split itself is deterministic).
      n_parts: number of problem parts in the curriculum (default 7 for CS1).
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
        out_dir = os.path.join(_ROOT, 'outputs', 'generalization', 'exp3_lopo')
    os.makedirs(out_dir, exist_ok=True)

    if models_to_test is None:
        models_to_test = list(MODEL_REGISTRY.keys())

    if n_parts < 2:
        raise ValueError(f"n_parts must be ≥ 2, got {n_parts!r}")

    print(f"\n[EXP-3] === START ===", flush=True)
    print(f"[EXP-3] out_dir={out_dir}", flush=True)
    print(f"[EXP-3] device={device}", flush=True)
    print(f"[EXP-3] models={models_to_test}", flush=True)
    print(f"[EXP-3] seeds={seeds}, n_parts={n_parts}", flush=True)
    if epochs_override is not None:
        print(f"[EXP-3] epochs_override={epochs_override} (smoke-test mode)", flush=True)

    # Load (or synthesize) dataset
    ide_logs, labels_df, y, student_ids, is_synthetic = _load_or_synthesize(use_synthetic)
    n = len(y)
    print(f"[EXP-3] data: n={n}, fail_rate={y.mean():.4f}, "
          f"n_events={len(ide_logs)}, synthetic={is_synthetic}", flush=True)

    # Pre-compute the (deterministic) LOPO splits.
    # Each split is then reused across all 3 seeds.
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        lopo = lopo_split(labels_df, ide_logs, n_parts=n_parts)
    # Echo any part-too-small warnings
    for w in ws:
        print(f"[EXP-3] {w.message}", flush=True)

    # Quick sanity check: total test coverage ≈ N (one student appears in
    # exactly one part's test set).
    total_test = sum(int(m['n_test']) for _, _, m in lopo)
    assert abs(total_test - n) <= 1, \
        f"lopo should cover all students (got {total_test}, expected {n})"

    print(f"\n[EXP-3] LOPO split summary (n_total={n}):", flush=True)
    for tr_idx, te_idx, meta in lopo:
        p = meta['left_out_part']
        print(f"  part={p}: n_train={meta['n_train']}, n_test={meta['n_test']}, "
              f"fail_test={y[te_idx].mean():.3f}", flush=True)

    # Cache features per (model, part): features depend on which students
    # are in train/test, so they vary across parts but NOT across seeds.
    # Caching saves ~2x build time (since the 7d/11d builders are O(N_events)).
    feature_cache: Dict[Tuple[str, int], dict] = {}

    records: List[dict] = []
    errors: List[dict] = []
    t_global = time.time()

    for model_name in models_to_test:
        if model_name not in MODEL_REGISTRY:
            print(f"[EXP-3] WARNING: unknown model={model_name!r}, skipping", flush=True)
            continue
        model_spec = MODEL_REGISTRY[model_name]

        for tr_idx, te_idx, split_meta in lopo:
            part = int(split_meta['left_out_part'])
            n_test_this = int(split_meta['n_test'])

            # Build features once per (model, part)
            cache_key = (model_name, part)
            try:
                feats = feature_cache[cache_key]
            except KeyError:
                feats = _build_features_for_model(
                    model_name, model_spec,
                    ide_logs, student_ids,
                    tr_idx, te_idx,
                )
                feature_cache[cache_key] = feats

            # If test set is tiny, emit a warning but don't skip
            if n_test_this < 10:
                print(f"[EXP-3] WARNING: part={part} has only "
                      f"{n_test_this} test students (< 10); "
                      f"results may have high variance.", flush=True)

            for seed in seeds:
                t0 = time.time()
                rec = {
                    'model': model_name,
                    'left_out_part': int(part),
                    'seed': int(seed),
                    'n_train': int(len(tr_idx)),
                    'n_test': int(n_test_this),
                    'method': str(split_meta.get('method', 'lopo')),
                }
                err_msg = None
                try:
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
                        'model': model_name, 'left_out_part': int(part),
                        'seed': int(seed),
                        'n_train': int(len(tr_idx)), 'n_test': int(n_test_this),
                        'error': err_msg,
                        'traceback': traceback.format_exc(limit=4),
                    })
                records.append(rec)
                # progress line
                if not np.isnan(rec['f1_class_1']):
                    print(f"[EXP-3] model={model_name:14s} part={part} "
                          f"seed={seed} n_test={n_test_this:3d} "
                          f"F1(FAIL)={rec['f1_class_1']:.3f} "
                          f"({rec['elapsed_seconds']:.1f}s)", flush=True)
                else:
                    print(f"[EXP-3] model={model_name:14s} part={part} "
                          f"seed={seed} FAILED ({err_msg})", flush=True)

    # ─── aggregate ───
    df = pd.DataFrame(records)
    df_err = pd.DataFrame(errors)
    print(f"\n[EXP-3] Total runs: {len(df)} (errors: {len(df_err)})", flush=True)

    # save results.csv
    results_csv = os.path.join(out_dir, 'results.csv')
    df.to_csv(results_csv, index=False)
    print(f"[EXP-3] Saved {results_csv} ({len(df)} rows)", flush=True)

    if not df_err.empty:
        err_csv = os.path.join(out_dir, 'errors.csv')
        df_err.to_csv(err_csv, index=False)
        print(f"[EXP-3] Saved {err_csv} ({len(df_err)} rows)", flush=True)

    # per-part summary: mean/std across seeds for each (model, left_out_part)
    per_part = (
        df.groupby(['model', 'left_out_part'])
        .agg(
            n_runs=('f1_class_1', 'size'),
            n_test=('n_test', 'first'),
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
        )
        .reset_index()
    )
    per_part_csv = os.path.join(out_dir, 'per_part_summary.csv')
    per_part.to_csv(per_part_csv, index=False)
    print(f"[EXP-3] Saved {per_part_csv}", flush=True)

    # overall summary: mean/std across (left_out_part, seed) for each model
    summary = (
        df.groupby(['model'])
        .agg(
            n_runs=('f1_class_1', 'size'),
            n_parts=('left_out_part', 'nunique'),
            acc_mean=('accuracy', 'mean'),
            acc_std=('accuracy', 'std'),
            f1_fail_mean=('f1_class_1', 'mean'),
            f1_fail_std=('f1_class_1', 'std'),
            f1_fail_min=('f1_class_1', 'min'),
            f1_fail_max=('f1_class_1', 'max'),
            macro_f1_mean=('macro_f1', 'mean'),
            macro_f1_std=('macro_f1', 'std'),
            roc_auc_mean=('roc_auc', 'mean'),
            roc_auc_std=('roc_auc', 'std'),
            pr_auc_mean=('pr_auc', 'mean'),
            pr_auc_std=('pr_auc', 'std'),
            avg_n_test=('n_test', 'mean'),
            avg_elapsed=('elapsed_seconds', 'mean'),
        )
        .reset_index()
    )
    summary_csv = os.path.join(out_dir, 'summary.csv')
    summary.to_csv(summary_csv, index=False)
    print(f"[EXP-3] Saved {summary_csv}", flush=True)

    # save config.json
    cfg_snapshot = {
        'experiment': 'EXP-3 leave-one-problem-part-out',
        'seeds': list(seeds),
        'n_parts': int(n_parts),
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
        'lopo_part_sizes': {int(m['left_out_part']): int(m['n_test']) for _, _, m in lopo},
        'lopo_part_fail_rates': {
            int(m['left_out_part']): float(y[te_idx].mean())
            for tr_idx, te_idx, m in lopo
        },
    }
    cfg_path = os.path.join(out_dir, 'config.json')
    with open(cfg_path, 'w') as f:
        json.dump(cfg_snapshot, f, indent=2, default=str)
    print(f"[EXP-3] Saved {cfg_path}", flush=True)

    elapsed_total = time.time() - t_global
    print(f"\n[EXP-3] DONE in {elapsed_total:.1f}s. "
          f"{len(df)} runs, {len(df_err)} errors.", flush=True)

    payload = {
        'config': cfg_snapshot,
        'records': records,
        'errors': errors,
        'summary': summary.to_dict('records'),
        'per_part_summary': per_part.to_dict('records'),
        'elapsed_seconds': elapsed_total,
    }
    return payload


# ────────────────────────── CLI ───────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='EXP-3 Leave-One-Problem-Part-Out generalization experiment')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Random seeds for model re-training '
                             '(default: 42 123 777)')
    parser.add_argument('--n-parts', type=int, default=None,
                        help='Number of problem parts (default: 7)')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help=f'Subset of models to run. Available: {list(MODEL_REGISTRY.keys())}')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: outputs/generalization/exp3_lopo)')
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
    n_parts = args.n_parts if args.n_parts is not None else DEFAULT_N_PARTS

    run(
        seeds=seeds, n_parts=n_parts,
        models_to_test=args.models, out_dir=args.out_dir,
        config=config, device=args.device,
        epochs_override=args.epochs, use_synthetic=args.synthetic,
        threshold=args.threshold,
    )


if __name__ == '__main__':
    main()