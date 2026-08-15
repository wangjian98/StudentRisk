"""Meta-Mamba model for student risk prediction.

Components:
  - Mamba-style selective SSM backbone (self-implemented, no mamba-ssm dependency)
  - Task-aware FiLM modulation (conditioned on problem part)
  - MAML-style outer loop (student-level K-shot adaptation)
  - Task-contrastive pretraining loss (proxy for temporal contrastive)

Convention: Failed=1 (positive class), Passed=0
"""
from .model import S6Block, MambaBlock, TaskFiLM, MetaMambaClassifier
from .data import build_event_sequences
from .train import run, run_maml_fewshot

__all__ = [
    'S6Block', 'MambaBlock', 'TaskFiLM', 'MetaMambaClassifier',
    'build_event_sequences', 'run', 'run_maml_fewshot',
]