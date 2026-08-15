"""LSTM classifier for student risk prediction (Failed=1 convention)."""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Single-direction LSTM on a synthetic 1-step sequence of 46-dim features.

    Architecture:
      Input (B, 46) → MLP projection (46 → d_model) → unsqueeze to (B, 1, d_model)
                    → LSTM(d_model → hidden) → last hidden (B, hidden)
                    → Dropout → Linear(hidden → 1) → logit
    """
    def __init__(self, input_dim: int = 46, d_model: int = 64,
                 hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.input_dim  = input_dim
        self.d_model    = d_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
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

    def forward(self, x):
        # x: (B, input_dim)
        h = self.projection(x)               # (B, d_model)
        h = h.unsqueeze(1)                   # (B, 1, d_model)  — sequence length 1
        out, _ = self.lstm(h)                # out: (B, 1, hidden_dim)
        last = out[:, -1, :]                 # (B, hidden_dim)
        logit = self.head(last).squeeze(-1)  # (B,)
        return logit

    def predict_proba(self, x):
        """Return P(failed=1)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)