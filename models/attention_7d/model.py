"""Transformer-encoder Attention classifier on 7-dim event sequences."""
import torch
import torch.nn as nn


class Attention7DClassifier(nn.Module):
    """Multi-head self-attention (Transformer Encoder) on 7-dim event-type sequences.

    Same as 46-dim Attention but input is 7-dim event one-hot (no aggregate features).
    """

    def __init__(self, n_event_dims: int = 7, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 dim_feedforward: int = 128, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_event_dims, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
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

    def forward(self, sequences, mask):
        """sequences: (B, L, 7), mask: (B, L)"""
        h = self.input_proj(sequences)
        h = h * mask.unsqueeze(-1)
        cls = self.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        # Extend mask for CLS token (CLS always visible)
        cls_mask = torch.ones(h.size(0), 1, device=h.device, dtype=mask.dtype)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        # nn.TransformerEncoder uses src_key_padding_mask: True = ignore
        padding_mask = ~full_mask.bool()
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        cls_out = self.norm(h[:, 0, :])
        logit = self.head(cls_out).squeeze(-1)
        return logit

    def predict_proba(self, sequences, mask):
        with torch.no_grad():
            return torch.sigmoid(self.forward(sequences, mask))