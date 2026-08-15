"""Random Forest model trained on 7-dim raw event-count features.

The 7-dim features = total count per event type for each student.
Event types: text_insert, text_remove, text_paste, focus_gained, focus_lost, run, submit

Convention: Failed=1 (positive class).
"""
from .data import build_7dim_features
from .train import run

__all__ = ['build_7dim_features', 'run']