"""Hyperparameter search grids for the MetaMamba hyperparameter sweep (EXP-4).

This module centralises every hyperparameter search space for the StudentRisk
v3.1 paper "Hyperparameter Selection Justification" appendix.  Keeping the
grids in one file lets reviewers reproduce the sweep without re-reading the
runner source, and lets the runner cite these values verbatim in
``summary_table.md``.

Conventions
-----------
* Failed = 1 (positive class), Passed = 0.
* All event-sequence lengths are bucketed: ``max_len=256`` is the production
  default, ``max_len=128`` is the smoke-test default.
* Search-space sizes reported below drive the budget:
    - ARCH_HPARAM_GRID  : 3 × 3 × 3 × 3 = **81** configs
    - TRAIN_HPARAM_GRID : 3 × 3 × 3     = **27** configs

Total (full evaluation at 3 seeds × 5 folds):
    - Stage 1 quick screen  :  81 × 1 × 1 =    81 runs
    - Stage 1 full (top-K)  :  K × 3 × 5 =  15 K runs
    - Stage 2 full          :  27 × 3 × 5 =  405 runs
"""
from __future__ import annotations

from typing import Dict, Iterable, List


# ────────────────────── architecture hyperparameters ──────────────────────
# Vary first; they determine model capacity (number of parameters).
ARCH_HPARAM_GRID: Dict[str, List] = {
    'd_model':  [32, 64, 128],          # hidden state dim of every block
    'd_state':  [8, 16, 32],            # SSM inner state size
    'n_layers': [1, 2, 4],              # depth of the Mamba stack
    'dropout':  [0.1, 0.2, 0.3],        # regularisation strength
}


# ────────────────────── training hyperparameters ──────────────────────
# Vary second; they determine the optimisation trajectory only.
TRAIN_HPARAM_GRID: Dict[str, List] = {
    'lr':                 [3e-4, 1e-3, 3e-3],   # AdamW learning rate
    'contrastive_weight': [0.0,  0.3,  0.5 ],   # weight of the task-contrastive auxiliary loss
    'batch_size':         [8,   16,   32  ],   # mini-batch size
}


# ────────────────────── default config (matches models.meta_mamba defaults) ──────────────────────
# Used as the starting point: deepcopy this dict, then ``.update(override)``
# to build each per-config variant.  All keys must be accepted by
# ``models.meta_mamba.train.run()`` via its ``config`` argument.
DEFAULT_CONFIG: Dict[str, object] = {
    # architecture (kept as the unchanged "anchor" for Stage 2)
    'd_model':  64,
    'd_state':  16,
    'n_layers':  2,
    'dropout': 0.2,
    # training
    'lr':                 1e-3,
    'weight_decay':       1e-3,
    'batch_size':         16,
    'epochs':             40,
    'patience':           10,
    'contrastive_weight': 0.3,
    # event-sequence length (model-agnostic; meta_mamba uses 256, meta_mamba_7d uses 128)
    'max_len':            256,
}


# ────────────────────── helpers ──────────────────────

def count_configs(grid: Dict[str, List]) -> int:
    """Cartesian-product size of a hyperparameter grid."""
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


def iter_grid(grid: Dict[str, List]):
    """Yield every Cartesian-product config as a flat ``{name: value}`` dict."""
    keys: List[str] = list(grid.keys())
    lists: List[List] = [grid[k] for k in keys]

    def _recurse(i: int, current: Dict[str, object]):
        if i == len(keys):
            yield dict(current)
            return
        for v in lists[i]:
            current[keys[i]] = v
            yield from _recurse(i + 1, current)
            current.pop(keys[i], None)

    yield from _recurse(0, {})


def build_config(overrides: Dict[str, object]) -> Dict[str, object]:
    """Deep-clone DEFAULT_CONFIG and apply a flat override dict."""
    from copy import deepcopy
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg.update(overrides)
    return cfg


def configs_varying(grid: Dict[str, List],
                    exclude: Iterable[str] = ()) -> List[Dict[str, object]]:
    """Return every override dict that varies only keys in ``grid``.

    ``exclude`` lets the caller hold some keys constant (e.g. while running
    Stage 2 with the Stage-1 best architecture).
    """
    exclude_set = set(exclude)
    sub = {k: v for k, v in grid.items() if k not in exclude_set}
    return [c for c in iter_grid(sub)]


__all__ = [
    'ARCH_HPARAM_GRID',
    'TRAIN_HPARAM_GRID',
    'DEFAULT_CONFIG',
    'count_configs',
    'iter_grid',
    'build_config',
    'configs_varying',
]