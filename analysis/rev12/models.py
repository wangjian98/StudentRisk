"""
rev12.models — 统一的架构实现与训练循环（等预算对比）

设计要点（对应审稿意见"调整3：基线对称化"）：
* LSTM / BiLSTM / Attention / MetaMamba 共用**同一套训练循环**（AdamW + cosine +
  早停 + 内层验证集），同一 max_epochs（200）、同一 patience（20）、同一 batch 规则。
* 每个架构有**同样大小（4 个）的超参搜索空间**；超参在固定 dev 折上按内层验证
  macro-F1 选择，随后冻结并用于全部 15 个外层折 —— 对所有架构完全对称。
* 早停用的内层验证集来自**训练折内部**（15%），评估折从不参与模型选择。

RF 作为表格基线走 sklearn（同样在训练折内做内层调参：n_estimators / max_depth /
min_samples_leaf / class_weight），搜索空间大小与深度模型一致。
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.base import set_seed
from models.lstm_7d.model import LSTM7DClassifier
from models.bilstm_7d.model import BiLSTM7DClassifier
from models.attention_7d.model import Attention7DClassifier
from models.meta_mamba.model import MetaMambaClassifier

DEEP_ARCHS = ['lstm', 'bilstm', 'attention', 'metamamba']
ALL_ARCHS = DEEP_ARCHS + ['rf']

# ── 对称的超参搜索空间：每个架构 4 个候选 ──────────────────────────────
HP_SPACE = {
    'lstm': [
        {'hidden_dim': 64, 'num_layers': 1, 'dropout': 0.3, 'lr': 1e-3},
        {'hidden_dim': 128, 'num_layers': 1, 'dropout': 0.3, 'lr': 1e-3},
        {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.2, 'lr': 1e-3},
        {'hidden_dim': 64, 'num_layers': 1, 'dropout': 0.3, 'lr': 3e-4},
    ],
    'bilstm': [
        {'hidden_dim': 64, 'num_layers': 1, 'dropout': 0.3, 'lr': 1e-3},
        {'hidden_dim': 128, 'num_layers': 1, 'dropout': 0.3, 'lr': 1e-3},
        {'hidden_dim': 64, 'num_layers': 2, 'dropout': 0.2, 'lr': 1e-3},
        {'hidden_dim': 64, 'num_layers': 1, 'dropout': 0.3, 'lr': 3e-4},
    ],
    'attention': [
        {'d_model': 64, 'n_heads': 4, 'n_layers': 2, 'dim_feedforward': 128, 'dropout': 0.3, 'lr': 1e-3},
        {'d_model': 128, 'n_heads': 4, 'n_layers': 2, 'dim_feedforward': 256, 'dropout': 0.3, 'lr': 1e-3},
        {'d_model': 64, 'n_heads': 4, 'n_layers': 3, 'dim_feedforward': 128, 'dropout': 0.2, 'lr': 1e-3},
        {'d_model': 64, 'n_heads': 4, 'n_layers': 2, 'dim_feedforward': 128, 'dropout': 0.3, 'lr': 3e-4},
    ],
    'metamamba': [
        {'d_model': 64, 'd_state': 16, 'n_layers': 2, 'dropout': 0.2, 'lr': 1e-3, 'contrastive_weight': 0.3},
        {'d_model': 128, 'd_state': 16, 'n_layers': 2, 'dropout': 0.2, 'lr': 1e-3, 'contrastive_weight': 0.3},
        {'d_model': 64, 'd_state': 16, 'n_layers': 2, 'dropout': 0.2, 'lr': 1e-3, 'contrastive_weight': 0.0},
        {'d_model': 64, 'd_state': 16, 'n_layers': 2, 'dropout': 0.2, 'lr': 3e-4, 'contrastive_weight': 0.3},
    ],
    'rf': [
        {'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 1},
        {'n_estimators': 500, 'max_depth': None, 'min_samples_leaf': 1},
        {'n_estimators': 300, 'max_depth': 6, 'min_samples_leaf': 2},
        {'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 1},
    ],
}

TRAIN_COMMON = {
    'max_epochs': 200,
    'patience': 20,
    'batch_size': 32,
    'weight_decay': 1e-3,
    'grad_clip': 1.0,
}


# ───────────────────────────── 模型构建 ─────────────────────────────

def build_model(arch: str, d_event: int, hp: dict, n_tasks: int = 7, use_film: bool = True):
    if arch == 'lstm':
        return LSTM7DClassifier(n_event_dims=d_event, d_model=hp['hidden_dim'],
                                hidden_dim=hp['hidden_dim'], num_layers=hp['num_layers'],
                                dropout=hp['dropout'])
    if arch == 'bilstm':
        return BiLSTM7DClassifier(n_event_dims=d_event, d_model=hp['hidden_dim'],
                                  hidden_dim=hp['hidden_dim'], num_layers=hp['num_layers'],
                                  dropout=hp['dropout'])
    if arch == 'attention':
        return Attention7DClassifier(n_event_dims=d_event, d_model=hp['d_model'],
                                     n_heads=hp['n_heads'], n_layers=hp['n_layers'],
                                     dim_feedforward=hp['dim_feedforward'],
                                     dropout=hp['dropout'])
    if arch == 'metamamba':
        return MetaMambaClassifier(d_event=d_event, d_model=hp['d_model'],
                                   d_state=hp['d_state'], n_layers=hp['n_layers'],
                                   n_tasks=n_tasks, dropout=hp['dropout'])
    raise ValueError(f'unknown arch: {arch}')


def _forward(arch, model, seq, mask, task, use_film=True):
    if arch == 'metamamba':
        if use_film:
            return model(seq, mask, task)
        # FiLM-off：跳过调制（其余完全相同）
        x = model.event_embed(seq)
        x = model.input_norm(x)
        for blk in model.blocks:
            x = blk(x)
        mask_f = mask.unsqueeze(-1)
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        pooled = model.pool_norm(pooled)
        return model.head(pooled).squeeze(-1)
    return model(seq, mask)


def _task_contrastive_loss(features, task_ids, temperature=0.1):
    import torch.nn.functional as F
    B = features.size(0)
    if B < 2:
        return torch.tensor(0.0, device=features.device)
    z = F.normalize(features, dim=-1)
    sim = z @ z.T / temperature
    task_eq = (task_ids.unsqueeze(0) == task_ids.unsqueeze(1)).float()
    eye = torch.eye(B, device=features.device)
    pos_mask = task_eq - eye
    neg_mask = 1.0 - task_eq
    exp_sim = torch.exp(sim)
    eps = 1e-8
    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    neg_sum = (exp_sim * neg_mask).sum(dim=1)
    loss = -torch.log(pos_sum / (neg_sum + eps) + eps)
    valid = (pos_mask.sum(dim=1) > 0).float()
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)
    return (loss * valid).sum() / (valid.sum() + eps)


def train_torch_model(arch: str, hp: dict, data: dict, epochs: int = None,
                       patience: int = None, device: str = 'cuda',
                       use_film: bool = True, seed: int = 0, verbose: bool = False):
    """统一训练循环：早停用 data['va']（内层验证集），返回训练好的 model。"""
    epochs = epochs or TRAIN_COMMON['max_epochs']
    patience = patience or TRAIN_COMMON['patience']
    bs = TRAIN_COMMON['batch_size']

    set_seed(seed)
    model = build_model(arch, data['d_event'], hp, n_tasks=data['n_tasks'], use_film=use_film).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'],
                             weight_decay=TRAIN_COMMON['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit = nn.BCEWithLogitsLoss()

    seq_tr = torch.from_numpy(data['seq_tr']).float().to(device)
    mask_tr = torch.from_numpy(data['mask_tr']).float().to(device)
    task_tr = torch.from_numpy(data['task_tr']).long().to(device)
    y_tr = torch.from_numpy(data['y_tr'].astype(np.float32)).to(device)
    seq_va = torch.from_numpy(data['seq_va']).float().to(device)
    mask_va = torch.from_numpy(data['mask_va']).float().to(device)
    task_va = torch.from_numpy(data['task_va']).long().to(device)
    y_va = torch.from_numpy(data['y_va'].astype(np.float32)).to(device)

    cw = hp.get('contrastive_weight', 0.0) if arch == 'metamamba' else 0.0
    best_val, best_state, pc = float('inf'), None, 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y_tr), device=device)
        for i in range(0, len(y_tr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            if arch == 'metamamba':
                x = model.event_embed(seq_tr[idx])
                x = model.input_norm(x)
                for blk in model.blocks:
                    x = blk(x)
                if use_film:
                    x = model.film(x, task_tr[idx])
                mf = mask_tr[idx].unsqueeze(-1)
                pooled = (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1.0)
                pooled = model.pool_norm(pooled)
                logit = model.head(pooled).squeeze(-1)
                sup = crit(logit, y_tr[idx])
                loss = sup + cw * _task_contrastive_loss(pooled, task_tr[idx])
            else:
                logit = _forward(arch, model, seq_tr[idx], mask_tr[idx], None, use_film)
                loss = crit(logit, y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN_COMMON['grad_clip'])
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            v_logit = _forward(arch, model, seq_va, mask_va, task_va, use_film)
            v_loss = crit(v_logit, y_va).item()
        if not np.isfinite(v_loss):
            break
        if v_loss < best_val - 1e-4:
            best_val = v_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            pc = 0
        else:
            pc += 1
            if pc >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model._best_val_loss = best_val
    model._epochs_run = ep + 1
    return model


@torch.no_grad()
def predict_torch(arch, model, seq, mask, task, device='cuda', use_film=True):
    model.eval()
    out = []
    for i in range(0, len(seq), 128):
        s = torch.from_numpy(seq[i:i + 128]).float().to(device)
        m = torch.from_numpy(mask[i:i + 128]).float().to(device)
        t = torch.from_numpy(task[i:i + 128]).long().to(device)
        logit = _forward(arch, model, s, m, t, use_film)
        out.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(out)


def film_parameters(model) -> dict:
    """从训练好的 MetaMamba 中取出 FiLM 的 γ / β（按 task 展开）。

    返回: {'gamma': (n_tasks, d_model), 'beta': (n_tasks, d_model), 'task_emb': ...}
    """
    film = model.film
    with torch.no_grad():
        t = film.task_emb.weight.detach()             # (n_tasks, emb)
        gamma = torch.sigmoid(film.gamma_mlp[0](t))   # (n_tasks, d_model)
        beta = film.beta_mlp[0](t)                    # (n_tasks, d_model)
    return {
        'gamma': gamma.detach().cpu().numpy(),
        'beta': beta.detach().cpu().numpy(),
        'task_emb': t.detach().cpu().numpy(),
    }
