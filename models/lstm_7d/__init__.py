"""LSTM model on 7-dim raw event sequences (only event-type one-hot, no 46-dim aggregate).

Each event is encoded as a 7-dim one-hot over event types.
Sequence length: max_len=128 (most recent events).

Convention: Failed=1 (positive class).

NOTE: Only the data builder is re-exported here. The 7d LSTM trainer
(models.lstm_7d.train) is no longer transitively imported because it
references the removed 46-dim module (models.lstm). Use the trainer
directly via `python -m models.lstm_7d.train` if needed.
"""
from .data import build_event_sequences_7d

__all__ = ['build_event_sequences_7d']
