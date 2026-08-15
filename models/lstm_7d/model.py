"""LSTM classifier on 7-dim event sequences (Failed=1 convention)."""
import torch
import torch.nn as nn


class LSTM7DClassifier(nn.Module):
    """LSTM on 7-dim event-type one-hot sequences.

    Architecture:
      Input (B, L, 7) → Linear(7→64) → GELU → Dropout
                    → LSTM(64→hidden) → masked mean pool
                    → Dropout → Linear(hidden→1) → logit
    """

    def __init__(self, n_event_dims: int = 7, d_model: int = 64,
                 hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.n_event_dims = n_event_dims
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.event_embed = nn.Sequential(
            nn.Linear(n_event_dims, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, sequences, mask):
        """
        sequences: (B, L, 7)
        mask:      (B, L) — 1=real, 0=pad
        Returns:   logit (B,)
        """
        x = self.event_embed(sequences)              # (B, L, d_model)
        # Zero out padded positions
        x = x * mask.unsqueeze(-1)
        out, _ = self.lstm(x)                        # (B, L, hidden_dim)
        # Masked mean pool
        mask_f = mask.unsqueeze(-1)
        x_sum = (out * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = x_sum / denom                      # (B, hidden_dim)
        logit = self.head(pooled).squeeze(-1)
        return logit

    def predict_proba(self, sequences, mask):
        with torch.no_grad():
            return torch.sigmoid(self.forward(sequences, mask))