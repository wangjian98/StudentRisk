"""Random Forest model for student risk prediction."""
from .model import RFModel
from .train import run

__all__ = ['RFModel', 'run']