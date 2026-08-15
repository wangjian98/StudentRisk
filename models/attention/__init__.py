"""Attention (Transformer encoder) model for student risk prediction."""
from .model import AttentionClassifier
from .train import run

__all__ = ['AttentionClassifier', 'run']