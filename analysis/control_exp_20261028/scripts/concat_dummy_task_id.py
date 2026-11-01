#!/usr/bin/env python3
"""concat_dummy_task_id.py —— task 信息注入方式 × dummy task id 的严格同口径对照

背景:
  FiLM (乘性调制) 已有 dummy 结果, 但 summary.csv 里 film_on+true 那行取自
  results/meta_mamba_7d/oof_probs.npy (7维 OOF), 而 zero/mean/random/disabled
  四行用的是 results/film_inference_ablation 的 ckpt (11维) -> 基线与被测不同源。

本实验统一口径: 同一个模型、同一个 fold 划分、同一套 4 种 task_id 喂法,
  对比三种注入方式:
    film   - 乘性调制  x * gamma + beta
    concat - 加性注入  x + concat_proj(t)
    none   - 无条件化

评估: 5-fold x 3 seeds = 15 个模型, per-student OOF (与 dummy_task_id_lopo 同口径)
"""
import os, sys, time, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_HERE = '/home/ubuntu/StudentRisk'
sys.path.insert(0, _HERE)

from data import load_dataset
from models.base import set_seed
from models.meta_mamba.model import MambaBlock
from models.meta_mamba_7d.data import build_event_sequences_7d
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             accuracy_score, confusion_matrix)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = os.path.join(_HERE, 'results', 'concat_dummy_task_id')
os.makedirs(OUT_DIR, exist_ok=True)

EPOCHS, BATCH, LR, WD, PATIENCE = 60, 64, 3e-4, 1e-2, 12
SEEDS = [42, 123, 777]
N_SPLITS = 5
MAX_LEN = 128


class MetaMambaCond(nn.Module):
    """MetaMamba with pluggable task conditioning: film / concat / none"""

    def __init__(self, d_event=7, d_model=64, d_state=16, n_layers=2,
                 n_tasks=7, dropout=0.2, conditioning='film', task_emb_dim=16):
        super().__init__()
        self.d_model = d_model
        self.conditioning = conditioning
        self.event_embed = nn.Sequential(
            nn.Linear(d_event, d_model), nn.GELU(), nn.Dropout(dropout))
        self.input_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)])
        self.pool_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, 1))
        if conditioning == 'film':
            self.task_emb = nn.Embedding(n_tasks, task_emb_dim)
            self.gamma_mlp = nn.Sequential(nn.Linear(task_emb_dim, d_model), nn.Sigmoid())
            self.beta_mlp = nn.Sequential(nn.Linear(task_emb_dim, d_model))
        elif conditioning == 'concat':
            self.task_emb = nn.Embedding(n_tasks, task_emb_dim)
            self.concat_proj = nn.Linear(task_emb_dim, d_model)
        elif conditioning == 'none':
            pass
        else:
            raise ValueError(conditioning)

    def apply_condition(self, x, task_ids, mask):
        if self.conditioning == 'film':
            t = self.task_emb(task_ids)
            gamma = self.gamma_mlp(t).unsqueeze(1)
            beta = self.beta_mlp(t).unsqueeze(1)
            return x * gamma + beta, mask
        if self.conditioning == 'concat':
            t = self.task_emb(task_ids)
            ctx = self.concat_proj(t).unsqueeze(1)
            return x + ctx, mask
        return x, mask

    def forward(self, sequences, mask, task_ids):
        x = self.event_embed(sequences)
        x = self.input_norm(x)
        for blk in self.blocks:
            x = blk(x)
        x, mask = self.apply_condition(x, task_ids, mask)
        mask_f = mask.unsqueeze(-1)
        x_sum = (x * mask_f).sum(dim=1)
        pooled = x_sum / mask_f.sum(dim=1).clamp(min=1.0)
        pooled = self.pool_norm(pooled)
        return self.head(pooled).squeeze(-1)


class EmbOverrideWrapper(nn.Module):
    """把 task_emb lookup 换成原始向量 (对 film / concat 都适用)"""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, sequences, mask, task_ids, override=None):
        if override is None:
            return self.base(sequences, mask, task_ids)
        d = self.base.task_emb.embedding_dim
        B = sequences.shape[0]
        ovr = override.to(sequences.device)
        if ovr.dim() == 1:
            ovr = ovr.unsqueeze(0).expand(B, -1)
        orig = self.base.task_emb.forward
        self.base.task_emb.forward = lambda ids: ovr
        try:
            return self.base(sequences, mask, task_ids)
        finally:
            self.base.task_emb.forward = orig


def train_one_fold(model, seq, mask, task, y, tr, va, device=DEVICE):
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=1e-6)
    crit = nn.BCEWithLogitsLoss()
    s_tr = torch.from_numpy(seq[tr]).float().to(device)
    m_tr = torch.from_numpy(mask[tr]).float().to(device)
    t_tr = torch.from_numpy(task[tr]).long().to(device)
    y_tr = torch.from_numpy(y[tr].astype(np.float32)).to(device)
    s_va = torch.from_numpy(seq[va]).float().to(device)
    m_va = torch.from_numpy(mask[va]).float().to(device)
    t_va = torch.from_numpy(task[va]).long().to(device)
    y_va = torch.from_numpy(y[va].astype(np.float32)).to(device)
    best_vl, best_state, pc = float('inf'), {k: v.clone() for k, v in model.state_dict().items()}, 0
    n_tr = len(tr)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_tr)
        for i in range(0, n_tr, BATCH):
            idx = perm[i:i + BATCH]
            optim.zero_grad()
            logit = model(s_tr[idx], m_tr[idx], t_tr[idx])
            loss = crit(logit, y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v_loss = crit(model(s_va, m_va, t_va), y_va).item()
        if not np.isfinite(v_loss):
            break
        if v_loss < best_vl - 1e-4:
            best_vl, best_state, pc = v_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, seq, mask, task, override=None, batch_size=64):
    model.eval()
    n = len(task)
    out = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch_size):
        s = torch.from_numpy(seq[i:i + batch_size]).float().to(DEVICE)
        m = torch.from_numpy(mask[i:i + batch_size]).float().to(DEVICE)
        t = torch.from_numpy(task[i:i + batch_size]).long().to(DEVICE)
        if override is None:
            logit = model(s, m, t)
        else:
            logit = EmbOverrideWrapper(model)(s, m, t, override=override)
        out[i:i + batch_size] = torch.sigmoid(logit.squeeze(-1).ravel()).cpu().numpy()
    return out


def metrics_of(y, p, thr=0.5):
    yp = (p >= thr).astype(int)
    cm = confusion_matrix(y, yp, labels=[0, 1])
    TN, FP, FN, TP = [int(v) for v in (cm[0][0], cm[0][1], cm[1][0], cm[1][1])]
    return dict(
        accuracy=float(accuracy_score(y, yp)),
        f1_macro=float(f1_score(y, yp, average='macro', zero_division=0)),
        f1_class_0=float(f1_score(y, yp, pos_label=0, zero_division=0)),
        f1_class_1=float(f1_score(y, yp, pos_label=1, zero_division=0)),
        roc_auc=float(roc_auc_score(y, p)) if len(set(y)) > 1 else float('nan'),
        pr_auc=float(average_precision_score(y, p)),
        mean_prob=float(p.mean()), std_prob=float(p.std()),
        TN=TN, FP=FP, FN=FN, TP=TP)


def main():
    t_start = time.time()
    print('=' * 78)
    print('concat_dummy_task_id: 注入方式 x dummy task_id 严格同口径对照')
    print('=' * 78, flush=True)
    print(f'device={DEVICE}  seeds={SEEDS}  folds={N_SPLITS}  max_len={MAX_LEN}', flush=True)

    ide, labels_df, y, student_ids = load_dataset()
    sequences, masks, task_ids = build_event_sequences_7d(ide, student_ids, max_len=MAX_LEN)
    n_tasks = int(task_ids.max()) + 1
    print(f'sequences={sequences.shape}  n_tasks={n_tasks}  fail_rate={y.mean():.3f}', flush=True)

    y = np.asarray(y).astype(int)
    n = len(y)
    rows = []
    per_student = []

    for cond in ['film', 'concat']:
        print(f'\n===== conditioning = {cond} =====', flush=True)
        # 训练 15 个模型, 逐折保存 state_dict + 该折的 val 索引
        fold_models = []   # (seed, fold_i, state_dict)
        fold_splits = []
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            for fold_i, (tr, va) in enumerate(skf.split(np.zeros(n), y)):
                set_seed(seed * 1000 + fold_i)
                model = MetaMambaCond(d_event=sequences.shape[-1], d_model=64, d_state=16,
                                      n_layers=2, n_tasks=n_tasks, dropout=0.2,
                                      conditioning=cond).to(DEVICE)
                model = train_one_fold(model, sequences, masks, task_ids, y, tr, va)
                fold_models.append((seed, fold_i, {k: v.detach().cpu().clone()
                                                   for k, v in model.state_dict().items()}))
                fold_splits.append((seed, fold_i, va))
                print(f'  trained seed={seed} fold={fold_i} (elapsed {time.time()-t_start:.0f}s)', flush=True)

        # 载入模型
        models = []
        for seed, fold_i, sd in fold_models:
            m = MetaMambaCond(d_event=sequences.shape[-1], d_model=64, d_state=16,
                              n_layers=2, n_tasks=n_tasks, dropout=0.2, conditioning=cond).to(DEVICE)
            m.load_state_dict(sd)
            m.eval()
            models.append(m)

        # 该 conditioning 的 task embedding (用于 mean / random)
        emb = models[0].task_emb.weight.detach().cpu()
        mean_emb = emb.mean(dim=0)
        rand_emb = torch.rand_like(mean_emb)

        # 4 种 task_id 喂法, 逐折用对应 fold 的模型 ensemble
        feeds = {
            'true': None,
            'zero(0)': 'zero',
            'mean': mean_emb,
            'random': rand_emb,
        }
        for feed_name, override in feeds.items():
            probs = np.zeros(n, dtype=np.float32)
            for (seed, fold_i, va), (s2, f2, sd) in zip(fold_splits, fold_models):
                # 同一 fold 的 3 个 seed ensemble
                idxs = [i for i, (sd_, fi, _) in enumerate(fold_models) if fi == fold_i]
                ps = []
                for i in idxs:
                    m = models[i]
                    if feed_name == 'true':
                        p = predict(m, sequences[va], masks[va], task_ids[va])
                    elif feed_name == 'zero(0)':
                        p = predict(m, sequences[va], masks[va], task_ids[va],
                                    override=torch.zeros(models[0].task_emb.embedding_dim))
                    else:
                        p = predict(m, sequences[va], masks[va], task_ids[va], override=override)
                    ps.append(p)
                probs[va] = np.mean(ps, axis=0)
            mm = metrics_of(y, probs)
            rows.append(dict(conditioning=cond, task_id_used=feed_name, **mm))
            print(f'  [{cond:6s}] {feed_name:8s} macro_f1={mm["f1_macro"]:.4f} '
                  f'f1_1={mm["f1_class_1"]:.4f} f1_0={mm["f1_class_0"]:.4f} '
                 'auc={mm["roc_auc"]:.4f} mean_p={mm["mean_prob"]:.4f} std_p={mm["std_prob"]:.4f}', flush=True)
            for stu in range(n):
                per_student.append(dict(student_idx=stu, true_part=int(task_ids[stu]),
                                        true_label=int(y[stu]), conditioning=cond,
                                        task_id_used=feed_name, pred_prob=float(probs[stu])))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'summary.csv'), index=False)
    pd.DataFrame(per_student).to_csv(os.path.join(OUT_DIR, 'per_student_predictions.csv'), index=False)

    # 汇总对比表
    piv = df.pivot(index='task_id_used', columns='conditioning', values='f1_macro')
    piv2 = df.pivot(index='task_id_used', columns='conditioning', values='roc_auc')
    piv3 = df.pivot(index='task_id_used', columns='conditioning', values='std_prob')
    piv.to_csv(os.path.join(OUT_DIR, 'pivot_macro_f1.csv'))
    print('\n' + '=' * 78)
    print('Macro-F1 对照')
    print('=' * 78)
    print(piv.round(4).to_string())
    print('\nROC-AUC 对照')
    print(piv2.round(4).to_string())
    print('\nstd_prob 对照 (越分散=模型对不同学生给出越有区分的预测)')
    print(piv3.round(4).to_string())
    print(f'\n[done] 用时 {time.time()-t_start:.0f}s -> {OUT_DIR}')


if __name__ == '__main__':
    main()
