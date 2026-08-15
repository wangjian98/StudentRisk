"""7-dim event sequence builder for MetaMamba-7d.

Identical to lstm_7d.data — only event-type one-hot, no continuous features.
"""
import numpy as np
import pandas as pd

# Same as lstm_7d
from models.lstm_7d.data import build_event_sequences_7d

__all__ = ['build_event_sequences_7d']