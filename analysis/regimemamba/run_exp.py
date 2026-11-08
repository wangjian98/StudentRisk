"""
regimemamba.run_exp — RegimeMamba 实验运行器（协议与 rev12 完全对齐）

协议（与 rev12 exp1/exp2 一致，保证可比）：
  · 外层 5 折 × 3 seeds（按学生分组、标签分层）
  · 训练折内再切 15% 内层验证集（早停 + 超参选择）；评估折不参与任何选择
  · 超参在 dev 折（seed=42, fold=0）按内层 macro-F1 选优，随后冻结用于全部 15 折
  · 每架构同样 4 个候选超参；max_epochs=200、patience=20、batch=32、AdamW+cosine
  · 指标：accuracy / balanced accuracy / macro-F1 / F1(FAIL) / ROC-AUC

输出：results/regimemamba/<tag>/{results.json, fold_metrics.csv, oof_probs.npy, regime_stats.npz}
"""
import os
import sys
import json
import time
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from analysis.rev12.common import (load_cache, build_space, make_folds, save_json,
                                   save_fold_csv, compute_metrics, inner_split,
                                   deconfound_seq)
from models.base import set_seed
from analysis.regimemamba.model import (RegimeMambaClassifier, regime_losses)

OUT_ROOT = os.path.join(_ROOT, 'results', 'regimemamba')

HP_SPACE = [
    {'d_model': 64, 'd_state': 16, 'n_layers': 2, 'dropout': 0.2, 'lr': 1e-3},
    {'d_model': 128, 'd_state': 16, 'n_layers': 2, 'dropout': 0.2, 'lr': 1e-3},
    {'d_model': 64, 'd_state': 16, 'n_layers': 2, 'dropout': 0.2, 'lr': 3e-4},
    {'d_model': 64, 'd_state': 32, 'n_layers': 2, 'dropout': 0.2, 'lr': 1e-3},
]

TRAIN = {'max_epochs': 200, 'patience': 20, 'batch_size': 32,
         'weight_decay': 1e-3, 'grad_clip': 1.0}


def build(hp, d_event, n_tasks, n_regime, use_film, use_direct):
    return RegimeMambaClassifier(
        d_event=d_event, d_model=hp['d_model'], d_state=hp['d_state'],
        n_layers=hp['n_layers'], n_tasks=n_tasks, dropout=hp['dropout'],
        n_regime=n_regime, use_film=use_film, use_direct=use_direct)


def _prepare(data, device):
    return (torch.from_numpy(data['seq_tr']).float().to(device),
            torch.from_numpy(data['mask_tr']).float().to(device),
            torch.from_numpy(data['task_tr']).long().to(device),
            torch.from_numpy(data['y_tr'].astype(np.float32)).to(device),
            torch.from_numpy(data['seq_va']).float().to(device),
            torch.from_numpy(data['mask_va']).float().to(device),
            torch.from_numpy(data['task_va']).long().to(device),
            torch.from_numpy(data['y_va'].astype(np.float32)).to(device))


def train_regime(hp, data, n_regime=4, use_film=True, use_direct=True,
                 lam_usage=0.05, lam_switch=0.01, lam_sharp=0.01,
                 epochs=None, patience=None, device='cuda', seed=0, log=print):
    epochs = epochs or TRAIN['max_epochs']
    patience = patience or TRAIN['patience']
    bs = TRAIN['batch_size']
    set_seed(seed)
    model = build(hp, data['d_event'], data['n_tasks'], n_regime, use_film, use_direct).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=TRAIN['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    crit = nn.BCEWithLogitsLoss()
    (seq_tr, mask_tr, task_tr, y_tr, seq_va, mask_va, task_va, y_va) = _prepare(data, device)

    best_val, best_state, pc = float('inf'), None, 0
    ep = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y_tr), device=device)
        for i in range(0, len(y_tr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            logit, R, usage, last, switch = model(seq_tr[idx], mask_tr[idx], task_tr[idx],
                                                   return_regime=True)
            sup = crit(logit, y_tr[idx])
            ul, sl, pl = regime_losses(R, mask_tr[idx], usage, switch,
                                        lam_usage, lam_switch, lam_sharp)
            loss = sup + lam_usage * ul + lam_switch * sl + lam_sharp * pl
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN['grad_clip'])
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            v_logit = model(seq_va, mask_va, task_va)
            v_loss = crit(v_logit, y_va).item()
        if not np.isfinite(v_loss):
            break
        if v_loss < best_val - 1e-4:
            best_val, pc = v_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
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
def predict_regime(model, seq, mask, task, device='cuda', want_stats=False):
    model.eval()
    probs, stats = [], {'usage': [], 'last': [], 'switch': [], 'post_mean': []}
    for i in range(0, len(seq), 128):
        s = torch.from_numpy(seq[i:i + 128]).float().to(device)
        m = torch.from_numpy(mask[i:i + 128]).float().to(device)
        t = torch.from_numpy(task[i:i + 128]).long().to(device)
        if want_stats:
            logit, R, usage, last, switch = model(s, m, t, return_regime=True)
            stats['usage'].append(usage.cpu().numpy())
            stats['last'].append(last.cpu().numpy())
            stats['switch'].append(switch.cpu().numpy())
            mf = m.unsqueeze(-1)
            stats['post_mean'].append(((R * mf).sum(1) / mf.sum(1).clamp(min=1.0)).cpu().numpy())
        else:
            logit = model(s, m, t)
        probs.append(torch.sigmoid(logit).cpu().numpy())
    out = np.concatenate(probs)
    if want_stats:
        for k in stats:
            stats[k] = np.concatenate(stats[k])
        return out, stats
    return out, None


def fold_data(seq, mask, task, y, fold, inner_frac=0.15):
    tr, va = fold['tr'], fold['va']
    itr, iva = inner_split(y[tr], inner_frac, seed=fold['seed'] * 97 + fold['fold'] * 7 + 1)
    return {
        'seq_tr': seq[tr[itr]], 'mask_tr': mask[tr[itr]], 'task_tr': task[tr[itr]], 'y_tr': y[tr[itr]],
        'seq_va': seq[tr[iva]], 'mask_va': mask[tr[iva]], 'task_va': task[tr[iva]], 'y_va': y[tr[iva]],
        'd_event': seq.shape[-1], 'n_tasks': int(task.max()) + 1,
    }, tr[itr], tr[iva], va


def select_hp(seq, mask, task, y, dev_fold, hp_space, n_regime, use_film, use_direct,
              lam_usage, lam_switch, lam_sharp, device, epochs, patience, log):
    best, best_score = None, -1
    for hp in hp_space:
        d, _, _, _ = fold_data(seq, mask, task, y, dev_fold)
        m = train_regime(hp, d, n_regime, use_film, use_direct, lam_usage, lam_switch,
                         lam_sharp, epochs=epochs, patience=patience, device=device,
                         seed=dev_fold['seed'] * 1000 + dev_fold['fold'])
        p, _ = predict_regime(m, d['seq_va'], d['mask_va'], d['task_va'], device)
        score = compute_metrics(d['y_va'], p)['macro_f1']
        log(f"    [hp] cand {hp} -> inner macroF1={score:.4f}")
        if score > best_score:
            best, best_score = hp, score
    return best, best_score


def run_cell(tag, dim, prefix, deconfound, n_regime, use_film, use_direct,
             lam_usage, lam_switch, lam_sharp, hp_fixed, n_seeds, n_folds,
             max_len=128, device='cuda', log=print):
    t0 = time.time()
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    cache = load_cache()
    y = cache['y'].astype(int)
    seeds = (42, 123, 777)[:n_seeds]
    folds = make_folds(y, seeds, n_folds)
    seq, mask, task = build_space(cache, dim, prefix, max_len)
    if deconfound:
        seq = deconfound_seq(seq)
        task = np.zeros_like(task)

    dev_fold = folds[0]
    if hp_fixed:
        hp, hp_score = hp_fixed, None
        log(f"  [hp] fixed: {hp}")
    else:
        hp, hp_score = select_hp(seq, mask, task, y, dev_fold, HP_SPACE, n_regime,
                                 use_film, use_direct, lam_usage, lam_switch, lam_sharp,
                                 device, TRAIN['max_epochs'], TRAIN['patience'], log)
        log(f"  [hp] chosen: {hp} (dev inner macroF1={hp_score:.4f})")

    crit = nn.BCEWithLogitsLoss()
    fold_records, oof = [], np.zeros(len(y))
    usage_all, last_all, switch_all = np.zeros((len(y), n_regime)), np.zeros((len(y), n_regime)), np.zeros(len(y))
    post_mean_all = []
    bias_all, trans_all = [], []
    for f in folds:
        d, _, _, va = fold_data(seq, mask, task, y, f)
        model = train_regime(hp, d, n_regime, use_film, use_direct, lam_usage, lam_switch,
                             lam_sharp, device=device, seed=f['seed'] * 1000 + f['fold'])
        p, stats = predict_regime(model, seq[va], mask[va], task[va], device, want_stats=True)
        oof[va] += p
        usage_all[va] += stats['usage'] / len(seeds)
        last_all[va] += stats['last'] / len(seeds)
        switch_all[va] += stats['switch'] / len(seeds)
        post_mean_all.append(stats['post_mean'].mean(axis=0))    # (L,K) fold 平均后验轨迹
        with torch.no_grad():
            bias_all.append(model.regime_bias.detach().cpu().numpy())
            trans_all.append(torch.softmax(model.gate.trans_logit, dim=-1).detach().cpu().numpy())
        m = compute_metrics(y[va], p, 0.5)
        fold_records.append({'seed': f['seed'], 'fold': f['fold'], **m})
        log(f"    [regime] seed={f['seed']} fold={f['fold']} macroF1={m['macro_f1']:.4f} "
            f"balAcc={m['balanced_accuracy']:.4f} auc={m['roc_auc']:.4f}")
    oof = oof / len(seeds)

    import pandas as pd
    df = pd.DataFrame(fold_records)
    per_fold_mean = {k: float(df[k].mean()) for k in
                     ['macro_f1', 'accuracy', 'balanced_accuracy', 'roc_auc', 'f1_class_1', 'f1_class_0']}
    per_fold_std = {k: float(df[k].std()) for k in ['macro_f1', 'roc_auc', 'balanced_accuracy']}
    payload = {
        'tag': tag, 'arch': 'regimemamba', 'dim': dim, 'prefix': prefix,
        'deconfound': deconfound, 'n_regime': n_regime, 'use_film': use_film,
        'use_direct': use_direct, 'lam_usage': lam_usage, 'lam_switch': lam_switch,
        'lam_sharp': lam_sharp, 'hp': hp, 'hp_dev_score': hp_score,
        'n_folds': len(folds), 'seeds': list(seeds), 'max_len': max_len,
        'per_fold_mean': per_fold_mean, 'per_fold_std': per_fold_std,
        'pooled_oof': compute_metrics(y, oof, 0.5),
        'elapsed_s': time.time() - t0,
        'protocol': '5fold x 3seed stratified-by-student; inner 15% val; max_epochs=200; patience=20',
    }
    save_json(os.path.join(out_dir, 'results.json'), payload)
    save_fold_csv(os.path.join(out_dir, 'fold_metrics.csv'), df)
    np.save(os.path.join(out_dir, 'oof_probs.npy'), oof)
    np.savez(os.path.join(out_dir, 'regime_stats.npz'), usage=usage_all, last=last_all,
             switch=switch_all, post_mean=np.array(post_mean_all), y=y,
             regime_bias=np.array(bias_all), trans=np.array(trans_all))
    log(f"  -> {tag}: macroF1={per_fold_mean['macro_f1']:.4f} balAcc={per_fold_mean['balanced_accuracy']:.4f} "
        f"auc(pooled)={payload['pooled_oof']['roc_auc']:.4f} auc(fold-mean)={per_fold_mean['roc_auc']:.4f} "
        f"[{time.time()-t0:.0f}s]")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--dim', type=int, default=28)
    ap.add_argument('--prefix', type=float, default=1.0)
    ap.add_argument('--deconfound', action='store_true')
    ap.add_argument('--n_regime', type=int, default=4)
    ap.add_argument('--no_film', action='store_true')
    ap.add_argument('--no_direct', action='store_true')
    ap.add_argument('--lam_usage', type=float, default=0.05)
    ap.add_argument('--lam_switch', type=float, default=0.01)
    ap.add_argument('--lam_sharp', type=float, default=0.01)
    ap.add_argument('--hp_fixed', type=str, default='')       # json
    ap.add_argument('--n_seeds', type=int, default=3)
    ap.add_argument('--n_folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=0)           # 0 = 协议默认 200
    ap.add_argument('--patience', type=int, default=0)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()

    hp_fixed = json.loads(args.hp_fixed) if args.hp_fixed else None
    if args.epochs:
        TRAIN['max_epochs'] = args.epochs
    if args.patience:
        TRAIN['patience'] = args.patience

    run_cell(args.tag, args.dim, args.prefix, args.deconfound, args.n_regime,
             not args.no_film, not args.no_direct,
             args.lam_usage, args.lam_switch, args.lam_sharp, hp_fixed,
             args.n_seeds, args.n_folds, device=args.device,
             log=lambda s: print(s, flush=True))


if __name__ == '__main__':
    main()
