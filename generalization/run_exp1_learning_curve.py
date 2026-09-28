"""EXP-1: Learning-curve generalization experiment for StudentRisk v3.1.

Trains each model on progressively-larger stratified subsets of the training
fold and reports F1(FAIL), Macro-F1, ROC-AUC and PR-AUC on the held-out fold
test set. The output is a model × ratio × seed × fold table that visualizes
*how quickly* each model reaches its asymptote.

Aligned with the existing OOF protocol:
  - 5-fold StratifiedKFold (shuffle=True, seed ∈ {42, 123, 777})
  - Failed=1, Passed=0
  - Threshold = 0.5

Outputs (under ``outputs/generalization/exp1_learning_curve/``):
  - results.csv       one row per (model, ratio, seed, fold)
  - summary.csv       one row per (model, ratio) with mean/std across seeds×folds
  - config.json       snapshot of the experiment configuration
  - errors.csv        one row per failed run (graceful degradation)

The smoke-test invocation (must complete in < 5 minutes):
  python -m generalization.run_exp1_learning_curve \\
      --models rf7 meta_mamba_7d \\
      --seeds 42 \\
      --ratios 0.50 \\
      --n-splits 5
"""
from __future__ import annotations

import os
import sys
import time
import json
import argparse
import traceback
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Project path bootstrap
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Project utilities
from data import load_dataset
from data.features import build_features as build_46d_features
from models.rf7.data import build_7dim_features
from models.lstm_7d.data import build_event_sequences_7d
from models.meta_mamba.data import build_event_sequences as build_event_sequences_11d
from models.base import set_seed, evaluate_predictions
from models.rf.model import RFModel
from models.lstm.model import LSTMClassifier
from models.bilstm.model import BiLSTMClassifier
from models.attention.model import AttentionClassifier
from models.lstm_7d.model import LSTM7DClassifier
from models.bilstm_7d.model import BiLSTM7DClassifier
from models.attention_7d.model import Attention7DClassifier
from models.meta_mamba.model import MetaMambaClassifier
# reuse LSTM training loop for 46-d and 7-d tabular nn models
from models.lstm.train import train_one_fold as lstm_train_one_fold
from models.lstm_7d.train import train_one_fold_7d as lstm7d_train_one_fold
from models.meta_mamba.train import train_one_fold as meta_mamba_train_one_fold

from generalization.data_splits import learning_curve_split, _make_synthetic_dataset


# (MODEL_REGISTRY moved below train helpers — see end of train_helpers section)




def _sanitize(X: np.ndarray) -> np.ndarray:
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip(X, -1e6, 1e6).astype(np.float32)


def _train_tabular_sklearn(model_name: str, model_spec: dict, model_kwargs: dict,
                           X_tr, y_tr, X_te, y_te, cfg: dict, seed: int,
                           epochs_override: Optional[int] = None,
                           device: str = 'cpu', **kw) -> np.ndarray:
    """Train tabular sklearn model (RF) on a subsample and return predicted probs on test."""
    set_seed(seed)
    model = model_spec['model_cls'](**model_kwargs)
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)


def _train_tabular_nn(model_name: str, model_spec: dict, model_kwargs: dict,
                      X_tr, y_tr, X_te, y_te, cfg: dict, seed: int,
                      epochs_override: Optional[int] = None,
                      device: str = 'cpu', **kw) -> np.ndarray:
    """Train tabular NN (LSTM / BiLSTM / Attention with 46-dim input) and predict on test."""
    set_seed(seed)
    sc = StandardScaler().fit(_sanitize(X_tr))
    Xs_tr = sc.transform(_sanitize(X_tr))
    Xs_te = sc.transform(_sanitize(X_te))

    model = model_spec['model_cls'](**model_kwargs)
    # resolve training cfg from any of train_cfg_keys
    train_cfg = {}
    for k in model_spec.get('train_cfg_keys', ()):
        if k in cfg:
            train_cfg = cfg[k]
            break
    epochs = epochs_override if epochs_override is not None else train_cfg.get('epochs', 60)
    model, _ = lstm_train_one_fold(
        model, Xs_tr, y_tr, Xs_te, y_te,
        lr=train_cfg.get('lr', 1e-3),
        weight_decay=train_cfg.get('weight_decay', 1e-3),
        epochs=epochs,
        batch_size=train_cfg.get('batch_size', 32),
        patience=train_cfg.get('patience', 12),
        device=device,
    )
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(Xs_te).float().to(device))).cpu().numpy()
    return p


def _train_sequence_7d(model_name: str, model_spec: dict, model_kwargs: dict,
                        seq_tr, mask_tr, y_tr, seq_te, mask_te, y_te,
                        cfg: dict, seed: int,
                        epochs_override: Optional[int] = None,
                        device: str = 'cpu', **kw) -> np.ndarray:
    """Train sequence (LSTM-7d / BiLSTM-7d / Attention-7d) and predict on test."""
    set_seed(seed)
    train_cfg = {}
    for k in model_spec.get('train_cfg_keys', ()):
        if k in cfg:
            train_cfg = cfg[k]
            break
    epochs = epochs_override if epochs_override is not None else train_cfg.get('epochs', 40)
    model = model_spec['model_cls'](**model_kwargs)
    model, _ = lstm7d_train_one_fold(
        model, seq_tr, mask_tr, y_tr, seq_te, mask_te, y_te,
        lr=train_cfg.get('lr', 1e-3),
        weight_decay=train_cfg.get('weight_decay', 1e-3),
        epochs=epochs,
        batch_size=train_cfg.get('batch_size', 32),
        patience=train_cfg.get('patience', 10),
        device=device,
    )
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(
            torch.from_numpy(seq_te).float().to(device),
            torch.from_numpy(mask_te).float().to(device),
        )).cpu().numpy()
    return p


def _train_meta_mamba(model_name: str, model_spec: dict, model_kwargs: dict,
                       seq_tr, mask_tr, task_tr, y_tr,
                       seq_te, mask_te, task_te, y_te,
                       cfg: dict, seed: int,
                       epochs_override: Optional[int] = None,
                       device: str = 'cpu', **kw) -> np.ndarray:
    """Train MetaMamba (11-dim sequence + FiLM(task)) and predict on test."""
    set_seed(seed)
    train_cfg = {}
    for k in model_spec.get('train_cfg_keys', ('meta_mamba', 'attention')):
        if k in cfg:
            train_cfg = cfg[k]
            break
    epochs = epochs_override if epochs_override is not None else train_cfg.get('epochs', 40)
    n_tasks = model_kwargs.get('n_tasks', 7)
    model = model_spec['model_cls'](**model_kwargs)
    model, _ = meta_mamba_train_one_fold(
        model, seq_tr, mask_tr, task_tr, y_tr, seq_te, mask_te, task_te, y_te,
        epochs=epochs,
        lr=train_cfg.get('lr', 1e-3),
        weight_decay=train_cfg.get('weight_decay', 1e-3),
        batch_size=train_cfg.get('batch_size', 16),
        patience=train_cfg.get('patience', 10),
        contrastive_weight=train_cfg.get('contrastive_weight', 0.3),
        device=device,
    )
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(
            torch.from_numpy(seq_te).float().to(device),
            torch.from_numpy(mask_te).float().to(device),
            torch.from_numpy(task_te).long().to(device),
        )).cpu().numpy()
    return p


def _train_meta_mamba_7d(model_name: str, model_spec: dict, model_kwargs: dict,
                          seq_tr, mask_tr, task_tr, y_tr,
                          seq_te, mask_te, task_te, y_te,
                          cfg: dict, seed: int,
                          epochs_override: Optional[int] = None,
                          device: str = 'cpu', **kw) -> np.ndarray:
    """Train MetaMamba-7d (7-dim event sequences + FiLM(task)) and predict on test."""
    # The 7d input is identical to a 11d model but with d_event=7;
    # We call meta_mamba.train_one_fold with the smaller feature dim.
    return _train_meta_mamba(
        model_name, model_spec, model_kwargs,
        seq_tr, mask_tr, task_tr, y_tr,
        seq_te, mask_te, task_te, y_te,
        cfg, seed,
        epochs_override=epochs_override,
        device=device,
    )


# ────────────────────── model registry ──────────────────────


# All 10 models. Each entry maps model-name → dict with metadata used by the
# runner to build features, instantiate the model, and train/predict.
MODEL_REGISTRY: Dict[str, dict] = {
    # --- 46-dim tabular (aggregate) ---
    'rf': {
        'kind': 'tabular',
        'feature_dim': 46,
        'build_X': lambda df, sids: build_46d_features(df, sids),
        'train_fn': _train_tabular_sklearn,
        'model_kwargs_fn': lambda cfg: dict(
            n_estimators=cfg.get('random_forest', {}).get('n_estimators', 200),
            max_depth=cfg.get('random_forest', {}).get('max_depth', 10),
            min_samples_split=cfg.get('random_forest', {}).get('min_samples_split', 5),
            class_weight=cfg.get('random_forest', {}).get('class_weight', 'balanced'),
        ),
        'model_cls': RFModel,
        'needs_task_ids': False,
        'epochs': 0,  # sklearn: no epochs
    },
    # --- 7-dim tabular (raw event counts) ---
    'rf7': {
        'kind': 'tabular',
        'feature_dim': 7,
        'build_X': lambda df, sids: build_7dim_features(df, sids),
        'train_fn': _train_tabular_sklearn,
        'model_kwargs_fn': lambda cfg: dict(
            n_estimators=cfg.get('random_forest', {}).get('n_estimators', 200),
            max_depth=cfg.get('random_forest', {}).get('max_depth', 10),
            min_samples_split=cfg.get('random_forest', {}).get('min_samples_split', 5),
            class_weight=cfg.get('random_forest', {}).get('class_weight', 'balanced'),
        ),
        'model_cls': RFModel,
        'needs_task_ids': False,
        'epochs': 0,
    },
    # --- 46-dim tabular NN models (LSTM, BiLSTM, Attention) ---
    'lstm': {
        'kind': 'tabular_nn',
        'feature_dim': 46,
        'build_X': lambda df, sids: build_46d_features(df, sids),
        'train_fn': _train_tabular_nn,
        'model_kwargs_fn': lambda cfg: dict(
            input_dim=46,
            d_model=cfg.get('lstm', {}).get('hidden_dim', 64),
            hidden_dim=cfg.get('lstm', {}).get('hidden_dim', 64),
            num_layers=cfg.get('lstm', {}).get('num_layers', 1),
            dropout=cfg.get('lstm', {}).get('dropout', 0.3),
        ),
        'model_cls': LSTMClassifier,
        'train_cfg_keys': ('lstm',),
        'needs_task_ids': False,
        'epochs_key': 'lstm',
    },
    'bilstm': {
        'kind': 'tabular_nn',
        'feature_dim': 46,
        'build_X': lambda df, sids: build_46d_features(df, sids),
        'train_fn': _train_tabular_nn,
        'model_kwargs_fn': lambda cfg: dict(
            input_dim=46,
            d_model=cfg.get('bilstm', {}).get('hidden_dim', 64),
            hidden_dim=cfg.get('bilstm', {}).get('hidden_dim', 64),
            num_layers=cfg.get('bilstm', {}).get('num_layers', 1),
            dropout=cfg.get('bilstm', {}).get('dropout', 0.3),
        ),
        'model_cls': BiLSTMClassifier,
        'train_cfg_keys': ('bilstm',),
        'needs_task_ids': False,
        'epochs_key': 'bilstm',
    },
    'attention': {
        'kind': 'tabular_nn',
        'feature_dim': 46,
        'build_X': lambda df, sids: build_46d_features(df, sids),
        'train_fn': _train_tabular_nn,
        'model_kwargs_fn': lambda cfg: dict(
            input_dim=46,
            d_model=cfg.get('attention', {}).get('d_model', 64),
            n_heads=cfg.get('attention', {}).get('n_heads', 4),
            n_layers=cfg.get('attention', {}).get('n_layers', 2),
            dim_feedforward=cfg.get('attention', {}).get('dim_feedforward', 128),
            dropout=cfg.get('attention', {}).get('dropout', 0.3),
        ),
        'model_cls': AttentionClassifier,
        'train_cfg_keys': ('attention',),
        'needs_task_ids': False,
        'epochs_key': 'attention',
    },
    # --- 7-dim sequence (event-type only) ---
    'lstm_7d': {
        'kind': 'sequence7',
        'feature_dim': 7,
        'build_X': lambda df, sids: build_event_sequences_7d(df, sids, max_len=128),
        'train_fn': _train_sequence_7d,
        'model_kwargs_fn': lambda cfg: dict(
            n_event_dims=7,
            d_model=cfg.get('lstm', {}).get('hidden_dim', 64),
            hidden_dim=cfg.get('lstm', {}).get('hidden_dim', 64),
            num_layers=cfg.get('lstm', {}).get('num_layers', 1),
            dropout=cfg.get('lstm', {}).get('dropout', 0.3),
        ),
        'model_cls': LSTM7DClassifier,
        'train_cfg_keys': ('lstm',),
        'needs_task_ids': False,
        'epochs_key': 'lstm',
        'max_len': 128,
    },
    'bilstm_7d': {
        'kind': 'sequence7',
        'feature_dim': 7,
        'build_X': lambda df, sids: build_event_sequences_7d(df, sids, max_len=128),
        'train_fn': _train_sequence_7d,
        'model_kwargs_fn': lambda cfg: dict(
            n_event_dims=7,
            d_model=cfg.get('bilstm', {}).get('hidden_dim', 64),
            hidden_dim=cfg.get('bilstm', {}).get('hidden_dim', 64),
            num_layers=cfg.get('bilstm', {}).get('num_layers', 1),
            dropout=cfg.get('bilstm', {}).get('dropout', 0.3),
        ),
        'model_cls': BiLSTM7DClassifier,
        'train_cfg_keys': ('bilstm',),
        'needs_task_ids': False,
        'epochs_key': 'bilstm',
        'max_len': 128,
    },
    'attention_7d': {
        'kind': 'sequence7',
        'feature_dim': 7,
        'build_X': lambda df, sids: build_event_sequences_7d(df, sids, max_len=128),
        'train_fn': _train_sequence_7d,
        'model_kwargs_fn': lambda cfg: dict(
            n_event_dims=7,
            d_model=cfg.get('attention', {}).get('d_model', 64),
            n_heads=cfg.get('attention', {}).get('n_heads', 4),
            n_layers=cfg.get('attention', {}).get('n_layers', 2),
            dim_feedforward=cfg.get('attention', {}).get('dim_feedforward', 128),
            dropout=cfg.get('attention', {}).get('dropout', 0.3),
        ),
        'model_cls': Attention7DClassifier,
        'train_cfg_keys': ('attention',),
        'needs_task_ids': False,
        'epochs_key': 'attention',
        'max_len': 128,
    },
    # --- 11-dim sequence with task ids (meta-mamba) ---
    'meta_mamba': {
        'kind': 'sequence11',
        'feature_dim': 11,
        'build_X': lambda df, sids: build_event_sequences_11d(df, sids, max_len=128),
        'train_fn': _train_meta_mamba,
        'model_kwargs_fn': lambda cfg: dict(
            d_event=11,
            d_model=cfg.get('meta_mamba', cfg.get('attention', {})).get('d_model', 64),
            d_state=cfg.get('meta_mamba', cfg.get('attention', {})).get('d_state', 16),
            n_layers=cfg.get('meta_mamba', cfg.get('attention', {})).get('n_layers', 2),
            n_tasks=7,
            dropout=cfg.get('meta_mamba', cfg.get('attention', {})).get('dropout', 0.2),
        ),
        'model_cls': MetaMambaClassifier,
        'train_cfg_keys': ('meta_mamba', 'attention'),
        'needs_task_ids': True,
        'epochs_key': 'meta_mamba',
        'max_len': 128,
    },
    'meta_mamba_7d': {
        'kind': 'sequence7',
        'feature_dim': 7,
        'build_X': lambda df, sids: build_event_sequences_7d(df, sids, max_len=128),
        'train_fn': _train_meta_mamba_7d,
        'model_kwargs_fn': lambda cfg: dict(
            d_event=7,
            d_model=cfg.get('meta_mamba', cfg.get('attention', {})).get('d_model', 64),
            d_state=cfg.get('meta_mamba', cfg.get('attention', {})).get('d_state', 16),
            n_layers=cfg.get('meta_mamba', cfg.get('attention', {})).get('n_layers', 2),
            n_tasks=7,
            dropout=cfg.get('meta_mamba', cfg.get('attention', {})).get('dropout', 0.2),
        ),
        'model_cls': MetaMambaClassifier,
        'train_cfg_keys': ('meta_mamba', 'attention'),
        'needs_task_ids': True,
        'epochs_key': 'meta_mamba',
        'max_len': 128,
    },
}


# ────────────────────── config / device helpers ──────────────────────


def load_config(path: Optional[str] = None) -> dict:
    if path is None:
        path = os.path.join(_ROOT, 'configs', 'default.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device(prefer: Optional[str] = None) -> str:
    if prefer:
        return prefer
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _load_or_synthesize(use_synthetic: bool = False):
    """Load real dataset, or synthesize a small fallback when unavailable."""
    if use_synthetic:
        ide_logs, labels_df, y, student_ids = _make_synthetic_dataset()
        return ide_logs, labels_df, y, student_ids, True
    try:
        ide_logs, labels_df, y, student_ids = load_dataset()
        return ide_logs, labels_df, y, student_ids, False
    except Exception as e:
        warnings.warn(
            f"[run_exp1] Real dataset unavailable ({type(e).__name__}: {e}). "
            f"Falling back to SYNTHETIC data.",
            RuntimeWarning,
        )
        ide_logs, labels_df, y, student_ids = _make_synthetic_dataset()
        return ide_logs, labels_df, y, student_ids, True


# ────────────────────── core runner ──────────────────────


def _build_features_for_model(model_name: str, model_spec: dict,
                               ide_logs: pd.DataFrame, student_ids_all: np.ndarray,
                               sub_idx: np.ndarray, te_idx: np.ndarray,
                               ratios_max: float = 1.0):
    """Build features for a sub_train sample and a fixed test fold.

    For sequence models, also returns the task_ids tensor (dominant problem
    part per student).

    Args:
      model_name: key in MODEL_REGISTRY
      model_spec: entry from MODEL_REGISTRY
      ide_logs: full event log
      student_ids_all: full ordered student ids array
      sub_idx: indices into y for the sub_train students
      te_idx: indices into y for the test students

    Returns:
      dict with keys:
        'X_tr' (or 'seq_tr','task_tr'), 'X_te' (or 'seq_te','task_te'), 'mask_tr','mask_te'
    """
    sub_sids = student_ids_all[sub_idx]
    te_sids = student_ids_all[te_idx]
    kind = model_spec['kind']

    if kind in ('tabular', 'tabular_nn'):
        X_tr, _ = model_spec['build_X'](ide_logs, sub_sids)
        X_te, _ = model_spec['build_X'](ide_logs, te_sids)
        return {'X_tr': X_tr, 'X_te': X_te}

    if kind == 'sequence7':
        seq_tr, mask_tr, task_tr = model_spec['build_X'](ide_logs, sub_sids)
        seq_te, mask_te, task_te = model_spec['build_X'](ide_logs, te_sids)
        return {'seq_tr': seq_tr, 'mask_tr': mask_tr, 'task_tr': task_tr,
                'seq_te': seq_te, 'mask_te': mask_te, 'task_te': task_te}

    if kind == 'sequence11':
        seq_tr, mask_tr, task_tr, _ = model_spec['build_X'](ide_logs, sub_sids)
        seq_te, mask_te, task_te, _ = model_spec['build_X'](ide_logs, te_sids)
        return {'seq_tr': seq_tr, 'mask_tr': mask_tr, 'task_tr': task_tr,
                'seq_te': seq_te, 'mask_te': mask_te, 'task_te': task_te}

    raise ValueError(f"Unknown model kind: {kind}")


def _train_and_predict_one(model_name: str, model_spec: dict,
                            features: dict, y_tr: np.ndarray, y_te: np.ndarray,
                            cfg: dict, seed: int, epochs_override: Optional[int],
                            device: str) -> np.ndarray:
    """Dispatch to the model's training helper. Returns probabilities on test."""
    kind = model_spec['kind']
    train_fn = model_spec['train_fn']
    model_kwargs = model_spec['model_kwargs_fn'](cfg)

    if kind == 'tabular':
        return train_fn(
            model_name, model_spec, model_kwargs,
            features['X_tr'], y_tr, features['X_te'], y_te,
            cfg, seed, epochs_override=epochs_override, device=device,
        )
    if kind == 'tabular_nn':
        return train_fn(
            model_name, model_spec, model_kwargs,
            features['X_tr'], y_tr, features['X_te'], y_te,
            cfg, seed, epochs_override=epochs_override, device=device,
        )
    if model_spec.get('needs_task_ids', False):
        # Models with FiLM (meta_mamba / meta_mamba_7d) consume task ids.
        return train_fn(
            model_name, model_spec, model_kwargs,
            features['seq_tr'], features['mask_tr'], features.get('task_tr'),
            y_tr, features['seq_te'], features['mask_te'], features.get('task_te'),
            y_te, cfg, seed, epochs_override=epochs_override, device=device,
        )
    if kind in ('sequence7', 'sequence11'):
        # Other sequence models (lstm_7d / bilstm_7d / attention_7d) do not consume task ids.
        return train_fn(
            model_name, model_spec, model_kwargs,
            features['seq_tr'], features['mask_tr'], y_tr,
            features['seq_te'], features['mask_te'], y_te,
            cfg, seed, epochs_override=epochs_override, device=device,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def run(seeds: Tuple[int, ...] = (42, 123, 777),
        n_splits: int = 5,
        ratios: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
        models_to_test: Optional[List[str]] = None,
        out_dir: Optional[str] = None,
        config: Optional[dict] = None,
        device: Optional[str] = None,
        epochs_override: Optional[int] = None,
        use_synthetic: bool = False,
        threshold: float = 0.5) -> dict:
    """Run EXP-1 learning curve experiment.

    Args:
      seeds: random seeds for StratifiedKFold.
      n_splits: number of folds per seed.
      ratios: training-set subsampling fractions.
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
        out_dir = os.path.join(_ROOT, 'outputs', 'generalization', 'exp1_learning_curve')
    os.makedirs(out_dir, exist_ok=True)

    if models_to_test is None:
        models_to_test = list(MODEL_REGISTRY.keys())

    print(f"\n[EXP-1] === START ===", flush=True)
    print(f"[EXP-1] out_dir={out_dir}", flush=True)
    print(f"[EXP-1] device={device}", flush=True)
    print(f"[EXP-1] models={models_to_test}", flush=True)
    print(f"[EXP-1] seeds={seeds}, n_splits={n_splits}, ratios={ratios}", flush=True)
    if epochs_override is not None:
        print(f"[EXP-1] epochs_override={epochs_override} (smoke-test mode)", flush=True)

    # Load (or synthesize) dataset
    ide_logs, labels_df, y, student_ids, is_synthetic = _load_or_synthesize(use_synthetic)
    n = len(y)
    print(f"[EXP-1] data: n={n}, fail_rate={y.mean():.4f}, "
          f"n_events={len(ide_logs)}, synthetic={is_synthetic}", flush=True)

    records: List[dict] = []
    errors: List[dict] = []
    t_global = time.time()

    for model_name in models_to_test:
        if model_name not in MODEL_REGISTRY:
            print(f"[EXP-1] WARNING: unknown model={model_name!r}, skipping", flush=True)
            continue
        model_spec = MODEL_REGISTRY[model_name]

        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold_idx, (tr_idx_full, te_idx) in enumerate(skf.split(np.zeros(n), y)):
                # Compute learning-curve subsamples ONCE per (model, seed, fold)
                splits = learning_curve_split(y, tr_idx_full, ratios=ratios, seed=seed)
                # iterates ratios × (sub_idx, full_train_idx, meta)

                for ratio, (sub_idx, _orig_tr, _meta) in zip(ratios, splits):
                    t0 = time.time()
                    rec = {
                        'model': model_name,
                        'ratio': float(ratio),
                        'seed': int(seed),
                        'fold': int(fold_idx),
                        'n_train': int(len(sub_idx)),
                        'n_test': int(len(te_idx)),
                    }
                    err_msg = None
                    try:
                        # Build features only for the subsample + test set
                        feats = _build_features_for_model(
                            model_name, model_spec,
                            ide_logs, student_ids,
                            sub_idx, te_idx,
                        )
                        # Train and predict
                        y_tr = y[sub_idx]
                        y_te = y[te_idx]
                        if len(np.unique(y_tr)) < 2:
                            err_msg = (f"y_tr has only 1 unique value "
                                       f"(n_train={len(sub_idx)}, n_fail={int(y_tr.sum())}). "
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
                            'model': model_name, 'ratio': float(ratio),
                            'seed': int(seed), 'fold': int(fold_idx),
                            'n_train': int(len(sub_idx)),
                            'error': err_msg,
                            'traceback': traceback.format_exc(limit=4),
                        })
                    records.append(rec)
                    # progress line
                    if not np.isnan(rec['f1_class_1']):
                        print(f"[EXP-1] model={model_name:14s} ratio={ratio:.2f} "
                              f"seed={seed} fold={fold_idx} F1(FAIL)={rec['f1_class_1']:.3f} "
                              f"({rec['elapsed_seconds']:.1f}s)", flush=True)
                    else:
                        print(f"[EXP-1] model={model_name:14s} ratio={ratio:.2f} "
                              f"seed={seed} fold={fold_idx} FAILED ({err_msg})", flush=True)

    # ─── aggregate ───
    df = pd.DataFrame(records)
    df_err = pd.DataFrame(errors)
    print(f"\n[EXP-1] Total runs: {len(df)} (errors: {len(df_err)})", flush=True)

    # save results.csv
    results_csv = os.path.join(out_dir, 'results.csv')
    df.to_csv(results_csv, index=False)
    print(f"[EXP-1] Saved {results_csv} ({len(df)} rows)", flush=True)

    if not df_err.empty:
        err_csv = os.path.join(out_dir, 'errors.csv')
        df_err.to_csv(err_csv, index=False)
        print(f"[EXP-1] Saved {err_csv} ({len(df_err)} rows)", flush=True)

    # build summary.csv: mean/std across (seed, fold) for each (model, ratio)
    summary = (
        df.groupby(['model', 'ratio'])
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
            avg_elapsed=('elapsed_seconds', 'mean'),
        )
        .reset_index()
    )
    summary_csv = os.path.join(out_dir, 'summary.csv')
    summary.to_csv(summary_csv, index=False)
    print(f"[EXP-1] Saved {summary_csv}", flush=True)

    # save config.json
    cfg_snapshot = {
        'experiment': 'EXP-1 learning curve',
        'seeds': list(seeds),
        'n_splits': int(n_splits),
        'ratios': [float(r) for r in ratios],
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
    }
    cfg_path = os.path.join(out_dir, 'config.json')
    with open(cfg_path, 'w') as f:
        json.dump(cfg_snapshot, f, indent=2, default=str)
    print(f"[EXP-1] Saved {cfg_path}", flush=True)

    elapsed_total = time.time() - t_global
    print(f"\n[EXP-1] DONE in {elapsed_total:.1f}s. "
          f"{len(df)} runs, {len(df_err)} errors.", flush=True)

    payload = {
        'config': cfg_snapshot,
        'records': records,
        'errors': errors,
        'summary': summary.to_dict('records'),
        'elapsed_seconds': elapsed_total,
    }
    return payload


# ────────────────────── CLI ──────────────────────


def main():
    parser = argparse.ArgumentParser(description='EXP-1 learning-curve experiment')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Random seeds for StratifiedKFold (default: from config)')
    parser.add_argument('--n-splits', type=int, default=None,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--ratios', nargs='+', type=float, default=None,
                        help='Training-set subsampling ratios (default: 0.25 0.50 0.75 1.00)')
    parser.add_argument('--models', nargs='+', type=str, default=None,
                        help=f'Subset of models to run. Available: {list(MODEL_REGISTRY.keys())}')
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory (default: outputs/generalization/exp1_learning_curve)')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--device', type=str, default=None,
                        help="Force 'cpu' or 'cuda' (default: auto)")
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override training epochs (used for smoke tests)')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data (when real IDE_logs.csv is unavailable)')
    args = parser.parse_args()

    config = load_config()
    seeds = tuple(args.seeds) if args.seeds else tuple(config.get('cv', {}).get('seeds', [42, 123, 777]))
    n_splits = args.n_splits if args.n_splits is not None else config.get('cv', {}).get('n_splits', 5)
    ratios = tuple(args.ratios) if args.ratios else (0.25, 0.50, 0.75, 1.00)

    run(
        seeds=seeds, n_splits=n_splits, ratios=ratios,
        models_to_test=args.models, out_dir=args.out_dir,
        config=config, device=args.device,
        epochs_override=args.epochs, use_synthetic=args.synthetic,
        threshold=args.threshold,
    )


if __name__ == '__main__':
    main()