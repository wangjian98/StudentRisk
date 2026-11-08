"""
RegimeMamba — 行为 regime 切换的选择性状态空间模型（CS1 风险预测）

方法设计见 `StudentRisk/docs/REGIMEMAMBA_DESIGN_20261108.md`。

要点（与 MetaMamba 的唯一结构差异）：
  MetaMamba : 单一选择性 SSM（全局一套动力学）+ 任务感知 FiLM
  RegimeMamba: **离散行为 regime 前向滤波**（因果、可微）+ **regime 条件化的 SSM 动力学**
                + **regime 直通特征**（占用直方图 / 末步后验 / 切换率）
                + **per-regime 风险偏置**（可解释）

regime 发射网络只看行为（x_t），**不接 part / exercise 身份** → regime 不得走身份捷径。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.meta_mamba.model import TaskFiLM


# ─────────────────────────── 行为 regime 前向滤波 ───────────────────────────

class RegimeGate(nn.Module):
    """因果离散 regime 滤波器（软化的前向算法）。

    r_t = softmax( log( r_{t-1} A ) + g(x_t) )
    · A: (K,K) 行随机（softmax 参数化），A[k,j] = 从 regime k 转到 j 的概率
    · g(x_t): 发射 logit（只依赖当前事件表征，不含身份）
    · padding 步不更新后验（carry forward），保证不把 padding 当证据
    """

    def __init__(self, d_model: int, n_regime: int = 4, d_hidden: int = 64):
        super().__init__()
        self.n = n_regime
        self.emit = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Linear(d_hidden, n_regime))
        self.trans_logit = nn.Parameter(torch.zeros(n_regime, n_regime))

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """x: (B,L,D)  mask: (B,L)  →  R: (B,L,K)"""
        B, L, D = x.shape
        g = self.emit(x)                                   # (B,L,K)
        A = F.softmax(self.trans_logit, dim=-1)            # (K,K)
        r = torch.full((B, self.n), 1.0 / self.n, device=x.device, dtype=x.dtype)
        rs = []
        for t in range(L):
            prior = r @ A                                  # (B,K)  Σ_k r_k A[k,j]
            log_post = torch.log(prior + 1e-9) + g[:, t, :]
            new_r = F.softmax(log_post, dim=-1)
            m = mask[:, t].unsqueeze(-1)                   # (B,1)
            r = m * new_r + (1.0 - m) * r                  # padding 步保持
            rs.append(r)
        return torch.stack(rs, dim=1)                      # (B,L,K)


# ──────────────────────── regime 条件化的选择性 SSM ────────────────────────

class RegimeS6Block(nn.Module):
    """S6 + regime 条件化参数（dt / B / C / A 都按 regime 混合）。

    与 S6Block 的差别：所有选择性参数不再是单一来源，而是
        p_t = Σ_k r_{t,k} · p_k(x_t)
    其中 p_k 由 regime k 的仿射调制产生。**只跑一次扫描**（参数混合在线性算子前完成），
    计算量与 MetaMamba 同量级（×K 的小张量运算）。
    """

    def __init__(self, d_inner: int, d_state: int = 16, d_conv: int = 4, n_regime: int = 4):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = max(d_inner // 16, 1)
        self.K = n_regime

        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        self.A_log = nn.Parameter(torch.log(torch.empty(d_inner, d_state).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_inner)

        # ── regime 调制参数 ──
        self.dt_scale = nn.Parameter(torch.ones(n_regime, d_inner))
        self.dt_shift = nn.Parameter(torch.zeros(n_regime, d_inner))
        self.B_scale = nn.Parameter(torch.ones(n_regime, d_state))
        self.C_scale = nn.Parameter(torch.ones(n_regime, d_state))
        self.A_off = nn.Parameter(torch.zeros(n_regime, d_inner, d_state))

    def forward(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """x: (B,L,D)  R: (B,L,K) → (B,L,D)"""
        B, L, D = x.shape
        x_t = x.transpose(1, 2)
        x_conv = F.silu(self.conv1d(x_t)[..., :L].transpose(1, 2))        # (B,L,D)

        x_proj = self.x_proj(x_conv)
        dt_x = x_proj[:, :, :self.dt_rank]
        B_x = x_proj[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C_x = x_proj[:, :, self.dt_rank + self.d_state:]

        dt_raw = F.softplus(self.dt_proj(dt_x))                            # (B,L,D)

        # regime 混合（一次性）
        dt_k = F.softplus(dt_raw.unsqueeze(2) * self.dt_scale.unsqueeze(0).unsqueeze(0)
                          + self.dt_shift.unsqueeze(0).unsqueeze(0))       # (B,L,K,D)
        dt_eff = (R.unsqueeze(-1) * dt_k).sum(dim=2)                       # (B,L,D)

        B_k = B_x.unsqueeze(2) * self.B_scale.unsqueeze(0).unsqueeze(0)   # (B,L,K,S)
        B_eff = (R.unsqueeze(-1) * B_k).sum(dim=2)                         # (B,L,S)
        C_k = C_x.unsqueeze(2) * self.C_scale.unsqueeze(0).unsqueeze(0)
        C_eff = (R.unsqueeze(-1) * C_k).sum(dim=2)                         # (B,L,S)

        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            a_mix = torch.einsum('bk,kds->bds', R[:, t, :], self.A_off)   # (B,D,S)
            A_t = -torch.exp(self.A_log.unsqueeze(0) + a_mix)              # (B,D,S)
            dA_t = torch.exp(dt_eff[:, t, :].unsqueeze(-1) * A_t)         # (B,D,S)
            dB_t = dt_eff[:, t, :].unsqueeze(-1) * B_eff[:, t, :].unsqueeze(1)
            h = dA_t * h + dB_t * x_conv[:, t, :].unsqueeze(-1)
            ys.append((h * C_eff[:, t, :].unsqueeze(1)).sum(-1))          # (B,D)
        y_seq = torch.stack(ys, dim=1)
        y_seq = y_seq + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        return self.out_proj(y_seq)


class RegimeMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, dropout=0.1, n_regime=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s6 = RegimeS6Block(d_inner=d_model, d_state=d_state, n_regime=n_regime)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, R):
        return x + self.dropout(self.s6(self.norm(x), R))


# ─────────────────────────────── 主模型 ───────────────────────────────

class RegimeMambaClassifier(nn.Module):
    """RegimeMamba: 事件嵌入 → 行为 regime 滤波 → regime 条件化 S6 栈 → 直通特征 → 判别头。

    直通特征（direct features）：
      usage  (K)  真实步上的 regime 平均占用
      last   (K)  最后有效步的后验
      switch (1)  平均切换幅度（||r_t − r_{t−1}||_1）
    per-regime 风险偏置：logit += Σ_k usage_k · b_k（可解释：该生有多少时间处于高风险 regime）
    """

    def __init__(self, d_event: int = 28, d_model: int = 64, d_state: int = 16,
                 n_layers: int = 2, n_tasks: int = 7, dropout: float = 0.2,
                 n_regime: int = 4, use_film: bool = True, use_direct: bool = True):
        super().__init__()
        self.K = n_regime
        self.use_direct = use_direct
        self.event_embed = nn.Sequential(
            nn.Linear(d_event, d_model), nn.GELU(), nn.Dropout(dropout))
        self.input_norm = nn.LayerNorm(d_model)
        self.gate = RegimeGate(d_model, n_regime=n_regime)
        self.blocks = nn.ModuleList([
            RegimeMambaBlock(d_model=d_model, d_state=d_state, dropout=dropout,
                             n_regime=n_regime) for _ in range(n_layers)])
        self.use_film = use_film
        if use_film:
            self.film = TaskFiLM(d_model=d_model, n_tasks=n_tasks)
        self.pool_norm = nn.LayerNorm(d_model)
        n_extra = (n_regime * 2 + 1) if use_direct else 0
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model + n_extra, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        self.regime_bias = nn.Parameter(torch.zeros(n_regime))

    # ── 辅助信息 ──
    def regime_stats(self, R, mask):
        mf = mask.unsqueeze(-1)                                  # (B,L,1)
        denom = mf.sum(dim=1).clamp(min=1.0)
        usage = (R * mf).sum(dim=1) / denom                      # (B,K)
        idx = (mask.sum(dim=1).long() - 1).clamp(min=0)          # 最后一个有效步索引（左 padding）
        last = R[torch.arange(R.size(0), device=R.device), idx]  # (B,K)
        d = (R[:, 1:, :] - R[:, :-1, :]).abs().sum(-1)            # (B,L-1)
        m2 = mask[:, 1:] * mask[:, :-1]
        switch = (d * m2).sum(dim=1) / m2.sum(dim=1).clamp(min=1.0)   # (B,)
        return usage, last, switch

    def forward(self, sequences, mask, task_ids, return_regime: bool = False):
        x = self.event_embed(sequences)
        x = self.input_norm(x)
        R = self.gate(x, mask)                                    # (B,L,K)
        for blk in self.blocks:
            x = blk(x, R)
        if self.use_film:
            x = self.film(x, task_ids)
        mf = mask.unsqueeze(-1)
        pooled = (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
        pooled = self.pool_norm(pooled)
        usage, last, switch = self.regime_stats(R, mask)
        if self.use_direct:
            feat = torch.cat([pooled, usage, last, switch.unsqueeze(-1)], dim=-1)
        else:
            feat = pooled
        logit = self.head(feat).squeeze(-1) + (usage * self.regime_bias).sum(-1)
        if return_regime:
            return logit, R, usage, last, switch
        return logit


# ───────────────────── 训练用辅助损失（regime 质量约束） ─────────────────────

def regime_losses(R, mask, usage, switch, lam_usage=0.05, lam_switch=0.01, lam_sharp=0.01):
    """返回 (usage_entropy_loss, switch_loss, sharpness_loss)。

    · usage_entropy：鼓励各 regime 被用到（防止塌缩到单一 regime）→ 最大化熵
    · switch_loss  ：鼓励 regime 持续（少抖动），让 regime 成为"状态"而非噪声
    · sharpness    ：鼓励每步后验尖锐（每步明确属于某个 regime）

    ⚠️ K=1 时 log(K)=0 会导致除零 → 训练发散（AUC=NaN）。
    这里显式对 K=1 短路：单 regime 情况下不存在"占用熵/切换/锐化"的意义，
    三个损失一律返回 0（K=1 是"无切换"对照组，必须与 K=4 保持同样的监督强度口径）。
    """
    K = R.size(-1)
    if K <= 1:
        zero = R.sum() * 0.0
        return zero, zero, zero
    p = usage.mean(dim=0)                                   # (K,) batch 平均占用
    ent = -(p * torch.log(p + 1e-9)).sum()
    usage_loss = -ent / math.log(K)                        # 最小化 → 最大化熵

    mf = mask.unsqueeze(-1)
    d = (R[:, 1:, :] - R[:, :-1, :]).abs().sum(-1) * (mask[:, 1:] * mask[:, :-1])
    switch_loss = d.sum() / (mf.sum() * K).clamp(min=1.0)

    p_t = R.clamp(min=1e-9)
    step_ent = -(p_t * torch.log(p_t)).sum(-1)              # (B,L)
    sharp_loss = (step_ent * mf.squeeze(-1)).sum() / mf.sum().clamp(min=1.0) / math.log(K)

    return usage_loss, switch_loss, sharp_loss
