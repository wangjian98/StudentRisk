"""BiLSTM model for student risk prediction."""
from .model import BiLSTMClassifier
from .train import run

__all__ = ['BiLSTMClassifier', 'run']