"""Models for student risk prediction.

Each subdirectory contains a self-contained model:
  - model.py: model architecture definition
  - train.py:  training script (runnable as `python -m models.<name>.train`)
"""
from .base import BaseModel, set_seed, compute_per_class_metrics, evaluate_predictions

__all__ = ['BaseModel', 'set_seed', 'compute_per_class_metrics', 'evaluate_predictions']