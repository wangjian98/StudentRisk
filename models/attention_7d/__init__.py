"""Attention (Transformer) model on 7-dim raw event sequences."""
from .data import build_event_sequences_7d
from .train import run

__all__ = ['build_event_sequences_7d', 'run']