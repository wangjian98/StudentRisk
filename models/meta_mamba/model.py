"""Meta-Mamba architecture: S6 backbone + FiLM + classifier.

Components:
  - S6Block: self-implemented Selective State Space (simplified, no mamba-ssm dependency)
  - MambaBlock: residual wrapper with normalization
  - TaskFiLM: feature-wise linear modulation conditioned on task id
  - MetaMambaClassifier: full model = Event Embedding → Mamba → FiLM → Pool → Classifier

Convention: Failed=1 (positive class).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────── Selective SSM (S6) ────────────────────────────

class S6Block(nn.Module):
    """Simplified S6 (Selective SSM) block.

    Reference: Mamba paper (Gu & Dao, 2023). We self-implement the core
    selective-scan idea without depending on the broken mamba-ssm package.

    Input  shape: (B, L, d_inner)
    Output shape: (B, L, d_inner)

    State equation (per channel): h_k = A * h_{k-1} + B * x_k
    Output:                          y_k = C * h_k
    A, B, C, dt are all computed from the input (selective).
    """

    def __init__(self, d_inner: int, d_state: int = 16, d_conv: int = 4, dt_rank: int = None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv  = d_conv
        self.dt_rank = dt_rank or max(d_inner // 16, 1)

        # Causal conv1d for local pattern
        self.conv1d = nn.Conv1d(
            in_channels=d_inner, out_channels=d_inner,
            kernel_size=d_conv, padding=d_conv - 1, groups=d_inner,
        )

        # Projection: x → (dt, B, C)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + d_state * 2, bias=False)

        # dt projection
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)

        # A log (state transition, learned)
        self.A_log = nn.Parameter(torch.log(torch.empty(d_inner, d_state).uniform_(1, 16)))

        # D skip (residual)
        self.D = nn.Parameter(torch.ones(d_inner))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_inner) → y: (B, L, d_inner)"""
        B, L, D = x.shape

        # 1) Causal conv1d on inner dimension (across time)
        x_t = x.transpose(1, 2)                    # (B, d_inner, L)
        x_conv = self.conv1d(x_t)[..., :L]          # causal: trim right padding
        x_conv = x_conv.transpose(1, 2)             # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # 2) Compute dt, B, C from input (selective)
        x_proj = self.x_proj(x_conv)                 # (B, L, dt_rank + 2*d_state)
        dt_x = x_proj[:, :, :self.dt_rank]            # (B, L, dt_rank)
        B_x  = x_proj[:, :, self.dt_rank:self.dt_rank + self.d_state]  # (B, L, d_state)
        C_x  = x_proj[:, :, self.dt_rank + self.d_state:]              # (B, L, d_state)

        dt = F.softplus(self.dt_proj(dt_x))           # (B, L, d_inner), positive

        # A: (d_inner, d_state) — negative diag (continuous → discrete)
        A = -torch.exp(self.A_log)                   # (d_inner, d_state)

        # 3) Selective scan — vectorized over batch
        # We use the closed-form discrete SSM:
        #   dA = exp(dt * A)  (per channel)         — (B, L, d_inner, d_state)
        #   dB = dt * B_x.unsqueeze(-1)             — (B, L, d_inner, d_state)
        # Then run recurrence: h_k = dA_k * h_{k-1} + dB_k * x_k (broadcast x over state dim)
        #
        # For efficiency we use the parallel scan trick:
        # y = (C ⊙ h), which we compute via the "logcumsumexp" trick in fp32.
        #
        # We approximate by computing h_t sequentially (Python loop over L).
        # Given L=256 and d_inner<=64 on GPU this is acceptable.

        x_for_scan = x_conv                          # (B, L, d_inner)
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            dA_t = torch.exp(dt[:, t, :].unsqueeze(-1) * A)            # (B, d_inner, d_state)
            dB_t = dt[:, t, :].unsqueeze(-1) * B_x[:, t, :].unsqueeze(1)  # (B, d_inner, d_state)
            h = dA_t * h + dB_t * x_for_scan[:, t, :].unsqueeze(-1)
            y_t = (h * C_x[:, t, :].unsqueeze(1)).sum(-1)              # (B, d_inner)
            ys.append(y_t)

        y_seq = torch.stack(ys, dim=1)               # (B, L, d_inner)

        # 4) Skip connection + output proj
        y_seq = y_seq + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        y_seq = self.out_proj(y_seq)
        return y_seq


class MambaBlock(nn.Module):
    """Residual Mamba block: x → S6 → SiLU → dropout → +x."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s6 = S6Block(d_inner=d_model, d_state=d_state, d_conv=d_conv)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm residual
        return x + self.dropout(self.s6(self.norm(x)))


# ──────────────────────────────── Task FiLM ────────────────────────────────

class TaskFiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on task id (problem part).

    h' = γ(task) ⊙ h + β(task)
    γ, β = MLP(task_embedding)
    """

    def __init__(self, d_model: int, n_tasks: int, task_emb_dim: int = 16):
        super().__init__()
        self.task_emb = nn.Embedding(n_tasks, task_emb_dim)
        self.gamma_mlp = nn.Sequential(
            nn.Linear(task_emb_dim, d_model),
            nn.Sigmoid(),  # γ in [0, 1] for stable modulation
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(task_emb_dim, d_model),
        )

    def forward(self, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model), task_ids: (B,)"""
        t = self.task_emb(task_ids)               # (B, task_emb_dim)
        gamma = self.gamma_mlp(t).unsqueeze(1)    # (B, 1, d_model)
        beta  = self.beta_mlp(t).unsqueeze(1)    # (B, 1, d_model)
        return gamma * x + beta


# ─────────────────────── Meta-Mamba full classifier ───────────────────────

class MetaMambaClassifier(nn.Module):
    """Meta-Mamba: Event embedding → N × MambaBlock → FiLM(task) → Pool → Classifier.

    Inputs:
      sequences: (B, L, D_event=11)
      mask:      (B, L)               1=real, 0=pad
      task_ids:  (B,)                 problem part id (0-indexed)

    Output:
      logit:     (B,)                  P(failed)
    """

    def __init__(self, d_event: int = 11, d_model: int = 64, d_state: int = 16,
                 n_layers: int = 2, n_tasks: int = 7, dropout: float = 0.2):
        super().__init__()
        self.event_embed = nn.Sequential(
            nn.Linear(d_event, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.input_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.film = TaskFiLM(d_model=d_model, n_tasks=n_tasks)
        self.pool_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor, task_ids: torch.Tensor):
        """Returns logit (B,)."""
        x = self.event_embed(sequences)            # (B, L, d_model)
        x = self.input_norm(x)
        for blk in self.blocks:
            x = blk(x)
        # FiLM modulation (task-aware)
        x = self.film(x, task_ids)
        # Masked mean pool over time
        mask_f = mask.unsqueeze(-1)                # (B, L, 1)
        x_sum = (x * mask_f).sum(dim=1)            # (B, d_model)
        denom = mask_f.sum(dim=1).clamp(min=1.0)   # (B, 1)
        pooled = x_sum / denom                     # (B, d_model)
        pooled = self.pool_norm(pooled)
        logit = self.head(pooled).squeeze(-1)      # (B,)
        return logit

    def predict_proba(self, sequences, mask, task_ids):
        with torch.no_grad():
            return torch.sigmoid(self.forward(sequences, mask, task_ids))