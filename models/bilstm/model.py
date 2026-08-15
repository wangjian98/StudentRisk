"""BiLSTM classifier for student risk prediction (Failed=1 convention)."""
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM on a synthetic 1-step sequence of 46-dim features.

    Architecture:
      Input (B, 46) → MLP projection (46 → d_model) → unsqueeze to (B, 1, d_model)
                    → BiLSTM(d_model → hidden) → concat forward+backward (B, 2*hidden)
                    → Dropout → Linear(2*hidden → 1) → logit
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
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 1),
        )

    def forward(self, x):
        h = self.projection(x)
        h = h.unsqueeze(1)
        out, _ = self.lstm(h)
        last = out[:, -1, :]                 # (B, 2*hidden_dim) — concat forward+backward
        logit = self.head(last).squeeze(-1)
        return logit

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)