"""MetaMamba-7d: Meta-Mamba on 7-dim event sequences (only event-type one-hot).

Same architecture as Meta-Mamba (S6 + FiLM + Task-Contrastive + FOMAML)
but the input event features are ONLY 7-dim (event-type one-hot) — NO 4 continuous features
(time interval, deadline distance, problem part, exercise number).
This enables fair comparison with other 7-dim models (RF-7d, LSTM-7d, BiLSTM-7d, Attention-7d).

Convention: Failed=1 (positive class).
"""
from .data import build_event_sequences_7d
from .train import run

__all__ = ['build_event_sequences_7d', 'run']