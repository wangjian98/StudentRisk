"""EXP-3-calibrated: EXP-3 LOPO + per-(model, part, seed) threshold sweep.

This script re-runs EXP-3 LOPO (10 models × 7 parts × 3 seeds = 210 runs)
and for each run searches the threshold that MAXIMIZES Macro F1 on the
test set. This is a calibration SURROGATE: instead of fitting a Platt
scaling model (which would require an internal validation split), we
ask "what would the best constant threshold have given the same model?"

Caveats:
  - This is a UPPER-BOUND on what threshold-tuning could achieve; it
    is honest only if we report it as such.
  - F1(FAIL) at t=0.5 is preserved for direct comparison with EXP-3.

Output columns added on top of EXP-3:
  - best_threshold  : the t* that maximizes Macro F1
  - macro_f1_at_t   : Macro F1 at t*
  - macro_f1_t05    : Macro F1 at t=0.5 (for EXP-3 comparison)
"""
from __future__ import annotations
import os, sys, time, json, argparse, warnings, traceback
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from generalization.run_exp1_learning_curve import (
    MODEL_REGISTRY, _build_features_for_model, _train_and_predict_one,
)
from generalization.data_splits import lopo_split
from models.base import set_seed

warnings.filterwarnings('ignore')


def _best_threshold(p: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Search threshold maximizing Macro F1 on (p, y)."""
    if len(np.unique(y)) < 2 or len(y) < 5:
        return 0.5, float('nan')
    best_t, best_mf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        y_pred = (p > t).astype(int)
        f1_0 = ((y_pred == 0) & (y == 0)).sum() / max(((y == 0).sum()), 1)
        f1_1 = ((y_pred == 1) & (y == 1)).sum() / max(((y == 1).sum()), 1)
        mf = (f1_0 + f1_1) / 2
        if mf > best_mf:
            best_mf, best_t = mf, t
    return float(best_t), float(best_mf)


def run(seeds: Tuple[int, ...] = (42, 123, 777),
        models_to_test: Optional[List[str]] = None,
        out_dir: Optional[str] = None,
        config: Optional[dict] = None,
        device: Optional[str] = None,
        epochs_override: Optional[int] = None,
        use_synthetic: bool = False) -> pd.DataFrame:
    """Re-run EXP-3 LOPO with threshold sweep."""
    import torch
    t0_overall = time.time()
    if models_to_test is None:
        models_to_test = list(MODEL_REGISTRY.keys())
    if out_dir is None:
        out_dir = os.path.join(_ROOT, 'outputs', 'generalization', 'exp3_lopo_calibrated')
    os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'[EXP-3-CAL] === START ===', flush=True)
    print(f'[EXP-3-CAL] out_dir={out_dir}', flush=True)
    print(f'[EXP-3-CAL] device={device}', flush=True)
    print(f'[EXP-3-CAL] models={models_to_test}', flush=True)
    print(f'[EXP-3-CAL] seeds={seeds}', flush=True)
    print(f'[EXP-3-CAL] use_synthetic={use_synthetic}', flush=True)

    # Load data
    if use_synthetic:
        from generalization.data_splits import _make_synthetic_dataset
        ide_logs, labels_df, _, _ = _make_synthetic_dataset(n_students=473, fail_rate=0.66)
    else:
        from data import load_dataset
        ide_logs, labels_df, _, _ = load_dataset()
    if config is None:
        config = {}

    results = []
    errors = []
    total_runs = len(models_to_test) * 7 * len(seeds)
    done = 0

    for model_name in models_to_test:
        if model_name not in MODEL_REGISTRY:
            continue
        model_spec = MODEL_REGISTRY[model_name]
        for part_id in range(7):
            for seed in seeds:
                done += 1
                t0 = time.time()
                try:
                    set_seed(seed)
                    splits = lopo_split(labels_df, ide_logs, n_parts=7)
                    if part_id >= len(splits):
                        raise IndexError(f'part_id={part_id} out of range, splits={len(splits)}')
                    tr_idx, te_idx, info = splits[part_id]
                    n_train = len(tr_idx); n_test = len(te_idx)
                    if n_train < 20 or n_test < 1:
                        raise ValueError(f'Too few samples: n_train={n_train}, n_test={n_test}')
                    y_tr = labels_df.iloc[tr_idx]['failed'].values.astype(np.int32)
                    y_te = labels_df.iloc[te_idx]['failed'].values.astype(np.int32)

                    # Build features
                    student_ids = labels_df['student'].values
                    features = _build_features_for_model(
                        model_name, model_spec,
                        ide_logs, student_ids, tr_idx, te_idx,
                    )

                    # Train + predict
                    p_te_raw = _train_and_predict_one(
                        model_name, model_spec, features, y_tr, y_te,
                        config, seed, epochs_override, device,
                    )

                    # Threshold sweep for Macro F1 upper bound
                    best_t, best_mf = _best_threshold(p_te_raw, y_te)

                    # At t=0.5
                    y_pred_05 = (p_te_raw > 0.5).astype(int)
                    tp = int(((y_pred_05 == 1) & (y_te == 1)).sum())
                    fp = int(((y_pred_05 == 1) & (y_te == 0)).sum())
                    fn = int(((y_pred_05 == 0) & (y_te == 1)).sum())
                    tn = int(((y_pred_05 == 0) & (y_te == 0)).sum())
                    f1_1_05 = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) > 0 else 0
                    f1_0_05 = tn / (tn + 0.5 * (fp + fn)) if (tn + fp + fn) > 0 else 0
                    mf_05 = (f1_1_05 + f1_0_05) / 2
                    accuracy = (tp + tn) / len(y_te)
                    if len(np.unique(y_te)) == 2:
                        from sklearn.metrics import roc_auc_score, average_precision_score
                        roc = float(roc_auc_score(y_te, p_te_raw))
                        pr  = float(average_precision_score(y_te, p_te_raw))
                    else:
                        roc, pr = 0.5, float(y_te.mean())

                    elapsed = time.time() - t0
                    results.append({
                        'model': model_name,
                        'left_out_part': part_id,
                        'seed': seed,
                        'n_train': n_train,
                        'n_test': n_test,
                        'best_threshold': best_t,
                        'macro_f1_calibrated': best_mf,
                        'macro_f1_t05': mf_05,
                        'accuracy': accuracy,
                        'f1_class_1_t05': f1_1_05,
                        'f1_class_0_t05': f1_0_05,
                        'roc_auc': roc,
                        'pr_auc': pr,
                        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
                        'elapsed_seconds': elapsed,
                    })
                    print(f'[EXP-3-CAL] [{done}/{total_runs}] model={model_name:18s} part={part_id} seed={seed} '
                          f'MacroF1_calib={best_mf:.3f} (t*={best_t:.2f}) | MacroF1_t05={mf_05:.3f} F1(FAIL)={f1_1_05:.3f} ({elapsed:.1f}s)',
                          flush=True)
                except Exception as e:
                    elapsed = time.time() - t0
                    errors.append({
                        'model': model_name, 'left_out_part': part_id, 'seed': seed,
                        'n_train': n_train if 'n_train' in locals() else 0,
                        'n_test': n_test if 'n_test' in locals() else 0,
                        'error': f'{type(e).__name__}: {e}',
                        'traceback': traceback.format_exc()[-1500:],
                    })
                    print(f'[EXP-3-CAL] [{done}/{total_runs}] model={model_name:18s} part={part_id} seed={seed} '
                          f'FAILED ({type(e).__name__}: {e})', flush=True)

    df_res = pd.DataFrame(results)
    df_err = pd.DataFrame(errors)
    df_res.to_csv(f'{out_dir}/results.csv', index=False)
    df_err.to_csv(f'{out_dir}/errors.csv', index=False)

    # Summary
    if not df_res.empty:
        metric_cols = ['accuracy', 'macro_f1_calibrated', 'macro_f1_t05',
                       'f1_class_1_t05', 'roc_auc', 'pr_auc']
        summary_rows = []
        per_part_rows = []
        for (m, p), sub in df_res.groupby(['model', 'left_out_part']):
            row = {'model': m, 'left_out_part': p, 'n_runs': len(sub)}
            for c in metric_cols:
                row[f'{c}_mean'] = sub[c].mean()
                row[f'{c}_std']  = sub[c].std()
            row['best_t_mean'] = sub['best_threshold'].mean()
            row['f1_fail_min'] = sub['f1_class_1_t05'].min()
            row['f1_fail_max'] = sub['f1_class_1_t05'].max()
            row['n_test'] = sub['n_test'].iloc[0]
            per_part_rows.append(row)
        for m, sub in df_res.groupby('model'):
            row = {'model': m, 'n_runs': len(sub)}
            for c in metric_cols:
                row[f'{c}_mean'] = sub[c].mean()
                row[f'{c}_std']  = sub[c].std()
            row['best_t_mean'] = sub['best_threshold'].mean()
            row['f1_fail_min'] = sub['f1_class_1_t05'].min()
            row['f1_fail_max'] = sub['f1_class_1_t05'].max()
            row['n_parts'] = sub['left_out_part'].nunique()
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(f'{out_dir}/summary.csv', index=False)
        pd.DataFrame(per_part_rows).to_csv(f'{out_dir}/per_part_summary.csv', index=False)

    cfg_dump = {
        'seeds': list(seeds), 'models': list(models_to_test), 'n_parts': 7,
        'epochs_override': epochs_override, 'use_synthetic': use_synthetic,
        'note': 'best_threshold = argmax_t MacroF1; macro_f1_calibrated is an UPPER BOUND on threshold-tuning.',
    }
    with open(f'{out_dir}/config.json', 'w') as f:
        json.dump(cfg_dump, f, indent=2)

    print(f'[EXP-3-CAL] DONE in {time.time()-t0_overall:.1f}s. '
          f'{len(results)} runs, {len(errors)} errors.', flush=True)
    return df_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 777])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()
    run(
        seeds=tuple(args.seeds),
        models_to_test=args.models,
        epochs_override=args.epochs,
        use_synthetic=args.synthetic,
        out_dir=args.out_dir,
    )