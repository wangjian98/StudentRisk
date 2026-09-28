"""EXP-4: Hyperparameter sweep for MetaMamba (StudentRisk v3.1).

This script runs the hyperparameter-search evidence required by the paper's
"Hyperparameter Selection Justification" appendix (v3.1).  It complements
EXP-1/2/3 (which compare *fixed-config* models) by **scanning** the four
core architectural and three core training hyperparameters of MetaMamba
and producing a defensible "best config" with full variance estimates.

Two-stage design (avoids combinatorial explosion of 81 × 27 = 2,187 configs):

  Stage 1 · ARCHITECTURE  (d_model × d_state × n_layers × dropout = 81 configs)
        ├── quick-screen mode  : 1 seed × 1 fold per config (81 runs, ~4h CPU)
        │     then rank → keep top-K (default 10) by F1(FAIL)
        └── full mode         : 3 seeds × 5 folds per top-K config (15·K runs)

  Stage 2 · TRAINING      (lr × contrastive_weight × batch_size = 27 configs)
        └── full mode         : 3 seeds × 5 folds per config (405 runs)
        (Stage 2 anchors architecture on the best from Stage 1.)

Metric selection rules (for picking "the best" config):
  1. Higher mean F1(FAIL)   (primary; matches paper's headline metric)
  2. Higher mean Macro-F1   (secondary; balances the two classes)
  3. Fewer trainable parameters (tiebreak; favours small-model story)

Outputs (under ``outputs/generalization/exp4_param_sweep/``):
  - stage1_screen_results.csv   one row per (config, seed, fold) of the quick screen
  - stage1_top_configs.csv      one row per surviving top-K config (full-eval aggregate)
  - stage1_full_results.csv     one row per (config, seed, fold) of full-eval top-K
  - stage2_results.csv          one row per (config, seed, fold) of Stage 2
  - best_config.json            winning config + metric summary
  - summary_table.md            paper-ready markdown table
  - config_used.json            snapshot of the run-time configuration

Smoke-test invocation (must complete in < 5 minutes):
    python -m generalization.run_param_sweep_meta_mamba \\
        --stage arch --quick-screen --top-k 3 \\
        --models meta_mamba_7d \\
        --seeds 42 --n-splits 5 \\
        --epochs 5 \\
        --synthetic

Convention: Failed=1 (positive class), Passed=0. Threshold = 0.5.
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
import itertools
import traceback
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

# Project path bootstrap (same convention as run_exp1/2/3.py)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Reuse Day-1 helpers: config loader, device picker, synthetic data fallback,
# event-sequence builders, MetaMamba model + train_one_fold.
from generalization.run_exp1_learning_curve import (
    load_config,
    pick_device,
)
from generalization.data_splits import _make_synthetic_dataset
from generalization.hparam_grid import (
    ARCH_HPARAM_GRID,
    TRAIN_HPARAM_GRID,
    DEFAULT_CONFIG,
    count_configs,
    iter_grid,
    build_config,
)
from data import load_dataset
from models.base import set_seed, evaluate_predictions
from models.meta_mamba.model import MetaMambaClassifier
from models.meta_mamba.train import train_one_fold as meta_train_one_fold
from models.meta_mamba.data import build_event_sequences as build_event_sequences_11d
from models.meta_mamba_7d.data import build_event_sequences_7d


# ────────────────────────── defaults & dispatch tables ──────────────────────────


# Two model "flavours" — both share the MetaMamba architecture and hparam
# interface, only the input event dimension differs.
MODEL_KINDS: Dict[str, Dict] = {
    'meta_mamba': {
        'feature_dim': 11,
        'build_seq':   build_event_sequences_11d,    # returns (seq, mask, task, counts)
        'max_len':     256,
    },
    'meta_mamba_7d': {
        'feature_dim': 7,
        'build_seq':   build_event_sequences_7d,      # returns (seq, mask, task)
        'max_len':     128,
    },
}

DEFAULT_SEEDS: Tuple[int, ...] = (42, 123, 777)
DEFAULT_N_SPLITS: int = 5
DEFAULT_TOP_K: int = 10
DEFAULT_THRESHOLD: float = 0.5


# ────────────────────────── data loading ──────────────────────────


def _load_or_synthesize(use_synthetic: bool = False):
    """Load the real CS1 dataset, or fall back to small synthetic data.

    Mirrors the helper in run_exp1_learning_curve.py so behaviour is
    identical to the rest of the generalization scripts.
    """
    if use_synthetic:
        ide_logs, labels_df, y, student_ids = _make_synthetic_dataset()
        return ide_logs, labels_df, y, student_ids, True
    try:
        ide_logs, labels_df, y, student_ids = load_dataset()
        return ide_logs, labels_df, y, student_ids, False
    except Exception as e:
        warnings.warn(
            f"[PARAM-SWEEP] Real dataset unavailable ({type(e).__name__}: {e}). "
            f"Falling back to SYNTHETIC data.",
            RuntimeWarning,
        )
        ide_logs, labels_df, y, student_ids = _make_synthetic_dataset()
        return ide_logs, labels_df, y, student_ids, True


def _build_sequences_for_model(model_name: str, model_kind: Dict,
                               ide_logs, student_ids, max_len: int):
    """Build event sequences once per (model, max_len).

    Returns ``(sequences, masks, task_ids)`` — the ``build_seq`` callable's
    arity is 11-d (4-tuple) for ``meta_mamba`` and 7-d (3-tuple) for
    ``meta_mamba_7d``, so we strip the trailing entry if present.
    """
    out = model_kind['build_seq'](ide_logs, student_ids, max_len=max_len)
    if len(out) == 4:
        seq, mask, task, _counts = out
    else:
        seq, mask, task = out
    return seq, mask, task


# ────────────────────────── single-config trainer ──────────────────────────


def _train_one_config(model_name: str, model_kind: Dict,
                      sequences: np.ndarray, masks: np.ndarray, task_ids: np.ndarray,
                      y: np.ndarray,
                      config: Dict[str, object],
                      seeds: Tuple[int, ...],
                      n_splits: int,
                      threshold: float,
                      device: str,
                      epochs_override: Optional[int],
                      ) -> List[Dict]:
    """Train+evaluate one hparam config across all (seed, fold) pairs.

    Returns one record per (seed, fold) with full metric block.
    """
    records: List[Dict] = []
    n = len(y)
    feature_dim = model_kind['feature_dim']
    n_tasks = int(task_ids.max()) + 1

    # extract architecture + training hparams
    arch = {
        'd_event':  feature_dim,
        'd_model':  config['d_model'],
        'd_state':  config['d_state'],
        'n_layers': config['n_layers'],
        'n_tasks':  n_tasks,
        'dropout':  config['dropout'],
    }
    epochs = int(epochs_override) if epochs_override is not None else int(config['epochs'])
    train_kwargs = dict(
        lr                 = float(config['lr']),
        weight_decay       = float(config['weight_decay']),
        batch_size         = int(config['batch_size']),
        patience           = int(config['patience']),
        contrastive_weight = float(config['contrastive_weight']),
        epochs             = epochs,
        device             = device,
    )

    # Static (config-level) descriptor
    cfg_id = {k: config[k] for k in (
        'd_model', 'd_state', 'n_layers', 'dropout',
        'lr', 'contrastive_weight', 'batch_size',
        'weight_decay', 'patience', 'max_len',
    )}

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y)):
            t0 = time.time()
            set_seed(seed * 1000 + fold_idx)
            try:
                model = MetaMambaClassifier(**arch).to(device)
                model, _v_loss = meta_train_one_fold(
                    model,
                    sequences[tr_idx], masks[tr_idx], task_ids[tr_idx], y[tr_idx],
                    sequences[va_idx], masks[va_idx], task_ids[va_idx], y[va_idx],
                    **train_kwargs,
                )
                model.eval()
                with torch.no_grad():
                    p = torch.sigmoid(model(
                        torch.from_numpy(sequences[va_idx]).float().to(device),
                        torch.from_numpy(masks[va_idx]).float().to(device),
                        torch.from_numpy(task_ids[va_idx]).long().to(device),
                    )).cpu().numpy()
                metrics = evaluate_predictions(y[va_idx], p, threshold=threshold)
                # parameter count (re-instantiate so we count exactly what this
                # config produces)
                n_params = sum(pp.numel() for pp in MetaMambaClassifier(**arch).parameters()
                               if pp.requires_grad)
                rec = {
                    'model':           model_name,
                    'seed':            int(seed),
                    'fold':            int(fold_idx),
                    'n_train':         int(len(tr_idx)),
                    'n_test':          int(len(va_idx)),
                    'feature_dim':     int(feature_dim),
                    'n_params':        int(n_params),
                    'accuracy':        float(metrics['accuracy']),
                    'precision_c1':    float(metrics['precision_class_1']),
                    'recall_c1':       float(metrics['recall_class_1']),
                    'f1_class_1':      float(metrics['f1_class_1']),
                    'macro_f1':        float(metrics['macro_f1']),
                    'roc_auc':         float(metrics['roc_auc']),
                    'pr_auc':          float(metrics['pr_auc']),
                    'TP': metrics['confusion_matrix']['TP'],
                    'FP': metrics['confusion_matrix']['FP'],
                    'TN': metrics['confusion_matrix']['TN'],
                    'FN': metrics['confusion_matrix']['FN'],
                    'elapsed_seconds': round(time.time() - t0, 2),
                    'v_loss':          float(_v_loss),
                }
                rec.update(cfg_id)
                records.append(rec)
            except Exception as e:
                tb = traceback.format_exc(limit=3)
                print(f"[PARAM-SWEEP] FAILED config=({config}) seed={seed} fold={fold_idx}: "
                      f"{type(e).__name__}: {e}", flush=True)
                rec = {
                    'model':           model_name,
                    'seed':            int(seed),
                    'fold':            int(fold_idx),
                    'n_train':         int(len(tr_idx)),
                    'n_test':          int(len(va_idx)),
                    'feature_dim':     int(feature_dim),
                    'n_params':        0,
                    'accuracy':        float('nan'),
                    'precision_c1':    float('nan'),
                    'recall_c1':       float('nan'),
                    'f1_class_1':      float('nan'),
                    'macro_f1':        float('nan'),
                    'roc_auc':         float('nan'),
                    'pr_auc':          float('nan'),
                    'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0,
                    'elapsed_seconds': round(time.time() - t0, 2),
                    'v_loss':          float('nan'),
                    'error':           f"{type(e).__name__}: {e}",
                }
                rec.update(cfg_id)
                records.append(rec)
                # swallow — keep sweep going on subsequent configs
                del tb
    return records


# ────────────────────────── stage runner ──────────────────────────


def _aggregate(records: List[Dict], config_keys) -> pd.DataFrame:
    """Group records by config_keys and compute mean/std for each metric.

    ``config_keys`` may be a tuple or list of column names.
    """
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if isinstance(config_keys, tuple):
        config_keys = list(config_keys)
    keep = list(config_keys) + [
        'n_params', 'feature_dim', 'model',
        'f1_class_1', 'macro_f1', 'accuracy', 'roc_auc', 'pr_auc',
        'precision_c1', 'recall_c1',
        'elapsed_seconds', 'n_train', 'n_test',
    ]
    keep = [c for c in keep if c in df.columns]
    agg = df.groupby(list(config_keys), dropna=False).agg(
        n_runs       = ('f1_class_1', 'size'),
        n_params     = ('n_params', 'first'),
        feature_dim  = ('feature_dim', 'first'),
        f1_fail_mean = ('f1_class_1', 'mean'),
        f1_fail_std  = ('f1_class_1', 'std'),
        f1_fail_min  = ('f1_class_1', 'min'),
        f1_fail_max  = ('f1_class_1', 'max'),
        macro_f1_mean= ('macro_f1',   'mean'),
        macro_f1_std = ('macro_f1',   'std'),
        acc_mean     = ('accuracy',   'mean'),
        roc_auc_mean = ('roc_auc',    'mean'),
        pr_auc_mean  = ('pr_auc',     'mean'),
        avg_elapsed  = ('elapsed_seconds', 'mean'),
    ).reset_index()
    return agg


def _select_best(agg: pd.DataFrame) -> Dict:
    """Apply selection rules: F1(FAIL) > Macro-F1 > smaller n_params."""
    if agg.empty:
        return {}
    df = agg.copy()
    # NaN-safe sort: smaller primary key first (descending for metrics)
    df = df.sort_values(
        by=['f1_fail_mean', 'macro_f1_mean', 'n_params'],
        ascending=[False, False, True],
        na_position='last',
    )
    row = df.iloc[0]
    return row.to_dict()


def _format_config_id(cfg: Dict[str, object]) -> str:
    """Pretty-print the hparams that define a config (skips missing keys)."""
    KEYS = (
        'd_model', 'd_state', 'n_layers', 'dropout',
        'lr', 'contrastive_weight', 'batch_size',
    )
    parts = []
    for k in KEYS:
        if k in cfg:
            v = cfg[k]
            if isinstance(v, float):
                parts.append(f"{k}={v:g}")
            else:
                parts.append(f"{k}={v}")
    return ", ".join(parts)


def _progress_msg(stage: str, idx: int, total: int, cfg: Dict, metric: Optional[float],
                  elapsed: Optional[float]) -> str:
    body = f"config {idx}/{total}: {_format_config_id(cfg)}"
    if metric is not None:
        body += f" | F1(FAIL)={metric:.4f}"
    if elapsed is not None:
        body += f" ({elapsed:.1f}s)"
    return f"[PARAM-SWEEP {stage}] {body}"


def run(stage: str = 'arch',
        models_to_test: Optional[List[str]] = None,
        seeds: Tuple[int, ...] = DEFAULT_SEEDS,
        n_splits: int = DEFAULT_N_SPLITS,
        quick_screen: bool = True,
        top_k: int = DEFAULT_TOP_K,
        out_dir: Optional[str] = None,
        config: Optional[dict] = None,
        device: Optional[str] = None,
        epochs_override: Optional[int] = None,
        use_synthetic: bool = False,
        threshold: float = DEFAULT_THRESHOLD,
        max_len_override: Optional[int] = None,
        ) -> dict:
    """Run the two-stage MetaMamba hyperparameter sweep.

    Args:
      stage: 'arch' for Stage 1 (architecture), 'train' for Stage 2 (training).
      models_to_test: subset of MODEL_KINDS keys. None = run all.
      seeds: tuple of random seeds for StratifiedKFold.
      n_splits: number of CV folds per seed.
      quick_screen:
          - Stage 1: if True, evaluate only the FIRST ``top_k`` configs of the
            grid (1 seed × 1 fold each).  Set ``top_k`` to the full grid size
            (81 by default) to cover every config.  If False, run the full
            two-phase protocol internally: first a quick screen over ALL
            configs (1 × 1), then full evaluation (3 × 5) on the top-K.
          - Stage 2: ignored — always evaluates all 27 configs at 3 × 5.
      top_k: see ``quick_screen``.  For Stage 2, must be ≥ 27 to cover the grid.
      out_dir: directory to save CSV/JSON/MD outputs. None = default.
      config: full config dict (loaded from configs/default.yaml if None).
      device: 'cpu' or 'cuda'. None = auto-pick.
      epochs_override: if not None, override training epochs (smoke test).
      use_synthetic: if True, use synthetic data instead of real CS1 dataset.
      threshold: decision threshold for F1 metrics.
      max_len_override: if not None, override event-sequence length.

    Returns:
      payload dict with all stage results + best_config.
    """
    if config is None:
        config = load_config()
    if device is None:
        device = pick_device()
    if models_to_test is None:
        models_to_test = list(MODEL_KINDS.keys())

    if stage not in ('arch', 'train'):
        raise ValueError(f"stage must be 'arch' or 'train', got {stage!r}")

    if out_dir is None:
        out_dir = os.path.join(_ROOT, 'outputs', 'generalization', 'exp4_param_sweep')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[PARAM-SWEEP] === START stage={stage} ===", flush=True)
    print(f"[PARAM-SWEEP] out_dir={out_dir}", flush=True)
    print(f"[PARAM-SWEEP] device={device}", flush=True)
    print(f"[PARAM-SWEEP] models={models_to_test}", flush=True)
    print(f"[PARAM-SWEEP] seeds={seeds}, n_splits={n_splits}, "
          f"quick_screen={quick_screen}, top_k={top_k}", flush=True)
    if epochs_override is not None:
        print(f"[PARAM-SWEEP] epochs_override={epochs_override} (smoke-test mode)", flush=True)

    # ---- data ----
    ide_logs, labels_df, y, student_ids, is_synthetic = _load_or_synthesize(use_synthetic)
    n = len(y)
    print(f"[PARAM-SWEEP] data: n={n}, fail_rate={y.mean():.4f}, "
          f"n_events={len(ide_logs)}, synthetic={is_synthetic}", flush=True)

    payload: Dict = {
        'config': {
            'experiment': 'EXP-4 MetaMamba hyperparameter sweep',
            'stage': stage,
            'seeds': list(seeds),
            'n_splits': int(n_splits),
            'quick_screen': bool(quick_screen),
            'top_k': int(top_k),
            'threshold': float(threshold),
            'device': device,
            'epochs_override': epochs_override,
            'use_synthetic': bool(use_synthetic),
            'is_synthetic_data': bool(is_synthetic),
            'n_students': int(n),
            'fail_rate': float(y.mean()),
            'models': list(models_to_test),
            'max_len_override': max_len_override,
        },
        'stage1_screen': [],
        'stage1_top':    [],
        'stage1_full':   [],
        'stage2':        [],
        'best_config':   None,
    }

    best_cfg: Optional[Dict] = None

    # ───── Stage 1: ARCHITECTURE ─────
    if stage == 'arch':
        all_arch_cfgs = list(iter_grid(ARCH_HPARAM_GRID))
        full_grid_size = len(all_arch_cfgs)
        # quick_screen=True  ⇒ only enumerate the first top_k configs
        # quick_screen=False ⇒ still need quick screen over the entire grid first
        quick_cfgs = all_arch_cfgs if not quick_screen else all_arch_cfgs[:int(top_k)]
        print(f"\n[PARAM-SWEEP Stage 1] arch grid: {full_grid_size} configs total; "
              f"quick-screen running {len(quick_cfgs)} configs", flush=True)

        # Phase 1a — quick screen (always 1 seed × 1 fold; cheap)
        screen_seeds  = (seeds[0],)
        screen_splits = n_splits

        for model_name in models_to_test:
            model_kind = MODEL_KINDS[model_name]
            max_len = int(max_len_override) if max_len_override is not None else model_kind['max_len']
            print(f"\n[PARAM-SWEEP] Building sequences for {model_name} "
                  f"(max_len={max_len}) ...", flush=True)
            t_build = time.time()
            sequences, masks, task_ids = _build_sequences_for_model(
                model_name, model_kind, ide_logs, student_ids, max_len,
            )
            print(f"[PARAM-SWEEP]   sequences={sequences.shape}, "
                  f"n_tasks={int(task_ids.max())+1}, "
                  f"build_time={time.time()-t_build:.1f}s", flush=True)

            screen_records: List[Dict] = []
            t_global = time.time()
            for idx, override in enumerate(quick_cfgs, start=1):
                cfg = build_config(override)
                cfg['max_len'] = max_len
                recs = _train_one_config(
                    model_name, model_kind,
                    sequences, masks, task_ids, y,
                    cfg,
                    seeds=screen_seeds,
                    n_splits=screen_splits,
                    threshold=threshold, device=device,
                    epochs_override=epochs_override,
                )
                # quick-screen: keep only fold 0
                recs = [r for r in recs if r['fold'] == 0]
                screen_records.extend(recs)
                m = recs[0]['f1_class_1'] if recs else float('nan')
                e = recs[0]['elapsed_seconds'] if recs else None
                print(_progress_msg("Stage 1 quick", idx, len(quick_cfgs),
                                    cfg, m, e), flush=True)
            payload['stage1_screen'].extend(screen_records)

            # save screen CSV
            screen_df = pd.DataFrame(screen_records)
            screen_csv = os.path.join(out_dir, 'stage1_screen_results.csv')
            screen_df.to_csv(screen_csv, index=False)
            print(f"[PARAM-SWEEP] Saved {screen_csv} ({len(screen_df)} rows, "
                  f"{time.time()-t_global:.1f}s)", flush=True)

            # rank by f1_fail_mean across the 1 quick run per config
            agg_screen = _aggregate(screen_records, (
                'd_model', 'd_state', 'n_layers', 'dropout'))
            agg_screen = agg_screen.sort_values(
                'f1_fail_mean', ascending=False, na_position='last').reset_index(drop=True)
            top_rows = agg_screen.head(int(top_k))
            print(f"\n[PARAM-SWEEP Stage 1 quick] Top {top_k} configs by F1(FAIL):",
                  flush=True)
            for i, row in top_rows.iterrows():
                print(f"  {i+1:2d}. {_format_config_id(row.to_dict())} | "
                      f"F1(FAIL)={row['f1_fail_mean']:.4f} ± "
                      f"{row['f1_fail_std']:.4f} | n_params={int(row['n_params'])}",
                      flush=True)

            top_csv = os.path.join(out_dir, 'stage1_top_configs.csv')
            top_rows.to_csv(top_csv, index=False)
            print(f"[PARAM-SWEEP] Saved {top_csv}", flush=True)

            # If quick_screen only, skip full eval and stop
            if quick_screen:
                print(f"\n[PARAM-SWEEP] quick_screen=True — skipping full Stage 1 "
                      f"evaluation. Re-run with --no-quick-screen for full eval.",
                      flush=True)
                # For smoke test: still select "best" from quick screen
                best = _select_best(top_rows)
                best['source_stage']  = 'stage1_quick_screen'
                best['config_keys']   = ('d_model', 'd_state', 'n_layers', 'dropout')
                payload['best_config'] = best
            else:
                # Phase 1b — full evaluation of top-K configs (all seeds × all folds)
                full_records: List[Dict] = []
                t_global = time.time()
                for idx, row in top_rows.iterrows():
                    override = {
                        'd_model':  row['d_model'],
                        'd_state':  row['d_state'],
                        'n_layers': row['n_layers'],
                        'dropout':  row['dropout'],
                    }
                    cfg = build_config(override)
                    cfg['max_len'] = max_len
                    recs = _train_one_config(
                        model_name, model_kind,
                        sequences, masks, task_ids, y,
                        cfg, seeds=seeds, n_splits=n_splits,
                        threshold=threshold, device=device,
                        epochs_override=epochs_override,
                    )
                    full_records.extend(recs)
                    agg_one = _aggregate(recs, (
                        'd_model', 'd_state', 'n_layers', 'dropout'))
                    if not agg_one.empty:
                        r = agg_one.iloc[0]
                        print(f"[PARAM-SWEEP Stage 1 full] top-{idx+1}/{len(top_rows)}: "
                              f"{_format_config_id(r.to_dict())} | "
                              f"F1(FAIL)={r['f1_fail_mean']:.4f} ± "
                              f"{r['f1_fail_std']:.4f}", flush=True)
                payload['stage1_full'].extend(full_records)
                full_df = pd.DataFrame(full_records)
                full_csv = os.path.join(out_dir, 'stage1_full_results.csv')
                full_df.to_csv(full_csv, index=False)
                print(f"[PARAM-SWEEP] Saved {full_csv} ({len(full_df)} rows, "
                      f"{time.time()-t_global:.1f}s)", flush=True)

                agg_full = _aggregate(full_records, (
                    'd_model', 'd_state', 'n_layers', 'dropout'))
                best = _select_best(agg_full)
                best['source_stage'] = 'stage1_full'
                best['config_keys']  = ('d_model', 'd_state', 'n_layers', 'dropout')
                payload['best_config'] = best

    # ───── Stage 2: TRAINING ─────
    elif stage == 'train':
        all_train_cfgs = list(iter_grid(TRAIN_HPARAM_GRID))
        print(f"\n[PARAM-SWEEP Stage 2] train grid: {len(all_train_cfgs)} configs "
              f"(= {count_configs(TRAIN_HPARAM_GRID)})", flush=True)

        for model_name in models_to_test:
            model_kind = MODEL_KINDS[model_name]
            max_len = int(max_len_override) if max_len_override is not None else model_kind['max_len']
            print(f"\n[PARAM-SWEEP] Building sequences for {model_name} "
                  f"(max_len={max_len}) ...", flush=True)
            t_build = time.time()
            sequences, masks, task_ids = _build_sequences_for_model(
                model_name, model_kind, ide_logs, student_ids, max_len,
            )
            print(f"[PARAM-SWEEP]   sequences={sequences.shape}, "
                  f"build_time={time.time()-t_build:.1f}s", flush=True)

            records: List[Dict] = []
            t_global = time.time()
            for idx, override in enumerate(all_train_cfgs, start=1):
                cfg = build_config(override)
                cfg['max_len'] = max_len
                # Stage 2 always uses full eval (3 seeds × 5 folds) — there's
                # no quick-screen for Stage 2 (only 27 configs).
                recs = _train_one_config(
                    model_name, model_kind,
                    sequences, masks, task_ids, y,
                    cfg, seeds=seeds, n_splits=n_splits,
                    threshold=threshold, device=device,
                    epochs_override=epochs_override,
                )
                records.extend(recs)
                agg_one = _aggregate(recs, (
                    'lr', 'contrastive_weight', 'batch_size'))
                if not agg_one.empty:
                    r = agg_one.iloc[0]
                    print(f"[PARAM-SWEEP Stage 2] config {idx}/{len(all_train_cfgs)}: "
                          f"{_format_config_id(r.to_dict())} | "
                          f"F1(FAIL)={r['f1_fail_mean']:.4f} ± "
                          f"{r['f1_fail_std']:.4f}", flush=True)
            payload['stage2'].extend(records)
            df = pd.DataFrame(records)
            csv = os.path.join(out_dir, 'stage2_results.csv')
            df.to_csv(csv, index=False)
            print(f"[PARAM-SWEEP] Saved {csv} ({len(df)} rows, "
                  f"{time.time()-t_global:.1f}s)", flush=True)

            agg = _aggregate(records, ('lr', 'contrastive_weight', 'batch_size'))
            best = _select_best(agg)
            best['source_stage'] = 'stage2'
            best['config_keys']  = ('lr', 'contrastive_weight', 'batch_size')
            payload['best_config'] = best

    # ───── emit outputs ─────
    def _clean(o):
        """Recursively replace NaN/Inf with None for JSON compatibility."""
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            v = float(o)
            if not np.isfinite(v):
                return None
            return v
        if isinstance(o, np.ndarray):
            return _clean(o.tolist())
        return o

    best_cfg_path = os.path.join(out_dir, 'best_config.json')
    with open(best_cfg_path, 'w') as f:
        json.dump(_clean(payload['best_config'] or {}), f, indent=2)
    print(f"[PARAM-SWEEP] Saved {best_cfg_path}", flush=True)

    cfg_used_path = os.path.join(out_dir, 'config_used.json')
    with open(cfg_used_path, 'w') as f:
        json.dump(payload['config'], f, indent=2, default=str)
    print(f"[PARAM-SWEEP] Saved {cfg_used_path}", flush=True)

    # markdown summary table
    md_path = os.path.join(out_dir, 'summary_table.md')
    _write_summary_table(md_path, payload, models_to_test)
    print(f"[PARAM-SWEEP] Saved {md_path}", flush=True)

    print(f"\n[PARAM-SWEEP] DONE.", flush=True)
    return payload


# ────────────────────────── markdown summary ──────────────────────────


def _write_summary_table(path: str, payload: Dict, models_to_test: List[str]) -> None:
    """Write a paper-ready markdown table summarising the sweep results."""
    lines: List[str] = []
    lines.append("# EXP-4 MetaMamba Hyperparameter Sweep — Summary\n")
    cfg = payload['config']
    lines.append("## Run configuration\n")
    lines.append(f"- Stage: **{cfg['stage']}**")
    lines.append(f"- Models: `{', '.join(models_to_test)}`")
    lines.append(f"- Seeds: `{cfg['seeds']}`")
    lines.append(f"- n_splits: `{cfg['n_splits']}`")
    lines.append(f"- Quick-screen: `{cfg['quick_screen']}`")
    lines.append(f"- Top-K (Stage 1): `{cfg['top_k']}`")
    lines.append(f"- Epochs override: `{cfg['epochs_override']}` (None ⇒ use config default)")
    lines.append(f"- Synthetic data: `{cfg['is_synthetic_data']}`")
    lines.append(f"- n_students: `{cfg['n_students']}`, fail_rate: `{cfg['fail_rate']:.4f}`")
    lines.append(f"- Device: `{cfg['device']}`")
    lines.append("")

    if payload.get('best_config'):
        bc = payload['best_config']
        lines.append("## Best configuration\n")
        lines.append(f"- Source stage: **{bc.get('source_stage', '?')}**")
        if 'f1_fail_mean' in bc and bc['f1_fail_mean'] is not None and np.isfinite(bc['f1_fail_mean']):
            std_v = bc.get('f1_fail_std', None)
            if std_v is None or not np.isfinite(std_v):
                std_str = "(n=1, std undefined)"
            else:
                std_str = f"± {std_v:.4f}"
            lines.append(f"- F1(FAIL) mean = **{bc['f1_fail_mean']:.4f}** {std_str}")
        if 'macro_f1_mean' in bc and bc['macro_f1_mean'] is not None and np.isfinite(bc['macro_f1_mean']):
            lines.append(f"- Macro-F1 mean = **{bc['macro_f1_mean']:.4f}**")
        if 'n_params' in bc and bc['n_params'] is not None:
            lines.append(f"- n_params = **{int(bc['n_params']):,}**")
        keys = bc.get('config_keys', ())
        if keys:
            lines.append("")
            lines.append("| Hyperparameter | Value |")
            lines.append("|---|---|")
            for k in keys:
                v = bc.get(k, '?')
                if isinstance(v, float):
                    lines.append(f"| `{k}` | `{v:g}` |")
                else:
                    lines.append(f"| `{k}` | `{v}` |")
        lines.append("")

    # Stage 1 quick screen
    if payload.get('stage1_screen'):
        agg = _aggregate(payload['stage1_screen'],
                         ('model', 'd_model', 'd_state', 'n_layers', 'dropout'))
        agg = agg.sort_values('f1_fail_mean', ascending=False, na_position='last')
        lines.append("## Stage 1 · Architecture quick-screen (1 seed × 1 fold)\n")
        lines.append("| Rank | Model | d_model | d_state | n_layers | dropout | "
                     "F1(FAIL) | n_params |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, row in agg.head(20).iterrows():
            lines.append(
                f"| {i+1} | `{row['model']}` | {int(row['d_model'])} | "
                f"{int(row['d_state'])} | {int(row['n_layers'])} | "
                f"{row['dropout']:.2f} | {row['f1_fail_mean']:.4f} | "
                f"{int(row['n_params']):,} |"
            )
        lines.append("")

    # Stage 1 full
    if payload.get('stage1_full'):
        agg = _aggregate(payload['stage1_full'],
                         ('model', 'd_model', 'd_state', 'n_layers', 'dropout'))
        agg = agg.sort_values('f1_fail_mean', ascending=False, na_position='last')
        lines.append("## Stage 1 · Architecture full evaluation (3 seeds × 5 folds)\n")
        lines.append("| Rank | Model | d_model | d_state | n_layers | dropout | "
                     "F1(FAIL) mean | F1(FAIL) std | Macro-F1 | n_params |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for i, row in agg.iterrows():
            lines.append(
                f"| {i+1} | `{row['model']}` | {int(row['d_model'])} | "
                f"{int(row['d_state'])} | {int(row['n_layers'])} | "
                f"{row['dropout']:.2f} | {row['f1_fail_mean']:.4f} | "
                f"{row['f1_fail_std']:.4f} | {row['macro_f1_mean']:.4f} | "
                f"{int(row['n_params']):,} |"
            )
        lines.append("")

    # Stage 2
    if payload.get('stage2'):
        agg = _aggregate(payload['stage2'],
                         ('model', 'lr', 'contrastive_weight', 'batch_size'))
        agg = agg.sort_values('f1_fail_mean', ascending=False, na_position='last')
        lines.append("## Stage 2 · Training hyperparameters (3 seeds × 5 folds)\n")
        lines.append("| Rank | Model | lr | contrastive_weight | batch_size | "
                     "F1(FAIL) mean | F1(FAIL) std | Macro-F1 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for i, row in agg.iterrows():
            lines.append(
                f"| {i+1} | `{row['model']}` | {row['lr']:g} | "
                f"{row['contrastive_weight']:.2f} | {int(row['batch_size'])} | "
                f"{row['f1_fail_mean']:.4f} | {row['f1_fail_std']:.4f} | "
                f"{row['macro_f1_mean']:.4f} |"
            )
        lines.append("")

    with open(path, 'w') as f:
        f.write("\n".join(lines))


# ────────────────────────── CLI ──────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='EXP-4 MetaMamba hyperparameter sweep (Stage 1 / Stage 2)')
    parser.add_argument('--stage', choices=['arch', 'train'], default='arch',
                        help='Which stage to run: arch (architecture params) '
                             'or train (training params). Default: arch.')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help=f'Subset of models to sweep. Available: {list(MODEL_KINDS.keys())}')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Random seeds for StratifiedKFold. Default: 42 123 777.')
    parser.add_argument('--n-splits', type=int, default=None,
                        help='Number of folds per seed. Default: 5.')
    parser.add_argument('--quick-screen', dest='quick_screen', action='store_true',
                        default=True,
                        help='Stage 1: 1 seed × 1 fold per config (default).')
    parser.add_argument('--no-quick-screen', dest='quick_screen', action='store_false',
                        help='Stage 1: full 3 seeds × 5 folds on the top-K configs.')
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K,
                        help='Stage 1: how many top configs to advance to full eval. '
                             f'Default: {DEFAULT_TOP_K}.')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: outputs/generalization/exp4_param_sweep)')
    parser.add_argument('--device', type=str, default=None,
                        help="Force 'cpu' or 'cuda' (default: auto).")
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override training epochs (used by smoke tests).')
    parser.add_argument('--max-len', type=int, default=None,
                        help='Override event-sequence length (default: '
                             '256 for meta_mamba, 128 for meta_mamba_7d).')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help=f'Decision threshold. Default: {DEFAULT_THRESHOLD}.')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (when real IDE_logs.csv is unavailable)')
    args = parser.parse_args()

    config = load_config()
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    n_splits = args.n_splits if args.n_splits is not None else DEFAULT_N_SPLITS

    run(
        stage=args.stage,
        models_to_test=args.models,
        seeds=seeds, n_splits=n_splits,
        quick_screen=args.quick_screen, top_k=args.top_k,
        out_dir=args.out_dir,
        config=config, device=args.device,
        epochs_override=args.epochs,
        use_synthetic=args.synthetic,
        threshold=args.threshold,
        max_len_override=args.max_len,
    )


if __name__ == '__main__':
    main()