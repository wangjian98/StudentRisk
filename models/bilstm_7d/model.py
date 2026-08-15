"""BiLSTM classifier on 7-dim event sequences."""
import torch
import torch.nn as nn


class BiLSTM7DClassifier(nn.Module):
    """Bidirectional LSTM on 7-dim event-type one-hot sequences."""

    def __init__(self, n_event_dims: int = 7, d_model: int = 64,
                 hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.event_embed = nn.Sequential(
            nn.Linear(n_event_dims, d_model),
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

    def forward(self, sequences, mask):
        x = self.event_embed(sequences)
        x = x * mask.unsqueeze(-1)
        out, _ = self.lstm(x)
        mask_f = mask.unsqueeze(-1)
        x_sum = (out * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        pooled = x_sum / denom
        logit = self.head(pooled).squeeze(-1)
        return logit

    def predict_proba(self, sequences, mask):
        with torch.no_grad():
            return torch.sigmoid(self.forward(sequences, mask))