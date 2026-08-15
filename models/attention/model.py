"""Multi-head self-attention (Transformer encoder) classifier for student risk prediction.

Convention: Failed=1, Passed=0.

Architecture:
  Input (B, 46) → Linear projection (46 → d_model) → reshape to (B, 1, d_model)
                → [CLS]-style token prepend → (B, 2, d_model)
                → TransformerEncoder(n_layers, n_heads, dim_feedforward)
                → take CLS token → LayerNorm → Dropout → Linear → logit
"""
import torch
import torch.nn as nn


class AttentionClassifier(nn.Module):
    def __init__(self, input_dim: int = 46, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.d_model   = d_model
        self.n_heads   = n_heads
        self.n_layers  = n_layers

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        # Learnable CLS-like token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        h = self.input_proj(x)                          # (B, d_model)
        h = h.unsqueeze(1)                              # (B, 1, d_model)
        cls = self.cls_token.expand(h.size(0), -1, -1)  # (B, 1, d_model)
        h = torch.cat([cls, h], dim=1)                  # (B, 2, d_model)
        h = self.encoder(h)
        cls_out = self.norm(h[:, 0, :])                 # (B, d_model)
        logit = self.head(cls_out).squeeze(-1)          # (B,)
        return logit

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)