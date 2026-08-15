"""LSTM model on 7-dim raw event sequences (only event-type one-hot, no 46-dim aggregate).

Each event is encoded as a 7-dim one-hot over event types.
Sequence length: max_len=128 (most recent events).

Convention: Failed=1 (positive class).
"""
from .data import build_event_sequences_7d
from .train import run

__all__ = ['build_event_sequences_7d', 'run']