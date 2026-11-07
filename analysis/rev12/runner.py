"""
rev12.runner — 各模型的统一"折级"运行器（RF / 深度序列模型 / HMM 家族）

所有模型共享同一套折划分与超参选择协议：
  * 外层：5 折 × 3 seeds（按学生分组，标签分层）
  * 内层：训练折内再切 15%（分层）作验证集，用于早停 + 超参选择
  * 超参搜索：在 dev 折（seed=42, fold=0）上跑完整候选空间，按内层 macro-F1 选优，
    之后冻结用于全部 15 个外层折 —— 对所有架构对称
"""
import os
import sys
import time
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.rev12.common import inner_split, compute_metrics
from analysis.rev12.models import (HP_SPACE, TRAIN_COMMON, train_torch_model,
                                   predict_torch, film_parameters)


def fold_data(seq, mask, task, y, fold, inner_frac=0.15):
    tr, va = fold['tr'], fold['va']
    itr, iva = inner_split(y[tr], inner_frac, seed=fold['seed'] * 97 + fold['fold'] * 7 + 1)
    tr_idx, iva_idx = tr[itr], tr[iva]
    d = {
        'seq_tr': seq[tr_idx], 'mask_tr': mask[tr_idx], 'task_tr': task[tr_idx], 'y_tr': y[tr_idx],
        'seq_va': seq[iva_idx], 'mask_va': mask[iva_idx], 'task_va': task[iva_idx], 'y_va': y[iva_idx],
        'd_event': seq.shape[-1],
        'n_tasks': int(task.max()) + 1,
    }
    return d, tr_idx, iva_idx, va


# ─────────────────────────── RF（表格基线） ───────────────────────────

def run_rf(X, y, folds, hp, threshold=0.5, seed_base=1000):
    from sklearn.ensemble import RandomForestClassifier
    from analysis.rev12.common import inner_split
    fold_records, oof = [], np.zeros(len(y))
    for f in folds:
        tr, va = f['tr'], f['va']
        rf = RandomForestClassifier(
            n_estimators=hp['n_estimators'], max_depth=hp['max_depth'],
            min_samples_leaf=hp['min_samples_leaf'], class_weight='balanced',
            random_state=f['seed'] * seed_base + f['fold'], n_jobs=4)
        rf.fit(X[tr], y[tr])
        p = rf.predict_proba(X[va])[:, 1]
        oof[va] += p
        m = compute_metrics(y[va], p, threshold)
        fold_records.append({'seed': f['seed'], 'fold': f['fold'], **m})
    return fold_records, oof


def select_rf_hp(X, y, dev_fold, hp_space):
    """dev 折上按内层验证 macro-F1 选超参（与深度模型同预算：4 个候选）。"""
    tr = dev_fold['tr']
    itr, iva = inner_split(y[tr], 0.15, seed=1234)
    best, best_score = None, -1
    for hp in hp_space:
        rf = __import__('sklearn.ensemble', fromlist=['RandomForestClassifier']).RandomForestClassifier(
            n_estimators=hp['n_estimators'], max_depth=hp['max_depth'],
            min_samples_leaf=hp['min_samples_leaf'], class_weight='balanced',
            random_state=42, n_jobs=4)
        rf.fit(X[tr[itr]], y[tr[itr]])
        p = rf.predict_proba(X[tr[iva]])[:, 1]
        score = compute_metrics(y[tr[iva]], p)['macro_f1']
        if score > best_score:
            best, best_score = hp, score
    return best, best_score


# ─────────────────────── 深度序列模型（LSTM/BiLSTM/Att/MetaMamba） ───────────────────────

def select_deep_hp(arch, seq, mask, task, y, dev_fold, hp_space, use_film=True, device='cuda'):
    best, best_score = None, -1
    for hp in hp_space:
        d, _, iva, _ = fold_data(seq, mask, task, y, dev_fold)
        model = train_torch_model(arch, hp, d, device=device, use_film=use_film,
                                  seed=dev_fold['seed'] * 1000 + dev_fold['fold'])
        p = predict_torch(arch, model, d['seq_va'], d['mask_va'], d['task_va'],
                          device=device, use_film=use_film)
        score = compute_metrics(d['y_va'], p)['macro_f1']
        if score > best_score:
            best, best_score = hp, score
    return best, best_score


def run_deep(arch, seq, mask, task, y, folds, hp, threshold=0.5, use_film=True,
             device='cuda', collect_film=False, log=print):
    fold_records, oof = [], np.zeros(len(y))
    film_dump = []
    for f in folds:
        d, tr_idx, iva_idx, va = fold_data(seq, mask, task, y, f)
        model = train_torch_model(arch, hp, d, device=device, use_film=use_film,
                                  seed=f['seed'] * 1000 + f['fold'])
        p = predict_torch(arch, model, seq[va], mask[va], task[va], device=device, use_film=use_film)
        oof[va] += p
        m = compute_metrics(y[va], p, threshold)
        fold_records.append({'seed': f['seed'], 'fold': f['fold'], **m})
        if collect_film and arch == 'metamamba':
            film_dump.append({'seed': f['seed'], 'fold': f['fold'],
                              **film_parameters(model)})
        log(f"    [{arch}] seed={f['seed']} fold={f['fold']} "
            f"macroF1={m['macro_f1']:.4f} acc={m['accuracy']:.4f} auc={m['roc_auc']:.4f}")
    return fold_records, oof, film_dump


# ─────────────────────────── HMM 家族 ───────────────────────────

def _strip(seq, mask):
    out = []
    for i in range(len(seq)):
        m = mask[i] > 0
        w = seq[i][m]
        if len(w) > 0:
            out.append(w)
    return out


def run_hmm_gmm(seq, mask, y, folds, n_states=3, n_mix=2, threshold=0.5, seed=0):
    from analysis.rev12.gmmhmm import ClassPairHMM, GMMHMM
    fold_records, oof = [], np.zeros(len(y))
    for f in folds:
        tr, va = f['tr'], f['va']
        seqs_tr = _strip(seq[tr], mask[tr])
        lens_tr = [len(s) for s in seqs_tr]
        Xtr = np.concatenate(seqs_tr)
        ytr = y[tr]
        # 需按类别切分：ClassPairHMM.fit 传入 per-sample 列表
        model = ClassPairHMM(factory=lambda: GMMHMM(n_states=n_states, n_mix=n_mix,
                                                     n_iter=20, random_state=f['seed'] * 31 + f['fold']))
        model.fit(seqs_tr, ytr, lens_tr)
        seqs_va = _strip(seq[va], mask[va])
        p = model.predict_proba(seqs_va)
        oof[va] += p
        m = compute_metrics(y[va], p, threshold)
        fold_records.append({'seed': f['seed'], 'fold': f['fold'], **m})
    return fold_records, oof


def run_hmm_cat(seq, mask, y, folds, n_states=3, threshold=0.5):
    from analysis.rev12.gmmhmm import ClassPairCategoricalHMM
    fold_records, oof = [], np.zeros(len(y))
    for f in folds:
        tr, va = f['tr'], f['va']
        syms_tr = []
        for i in range(len(tr)):
            m = mask[tr[i]] > 0
            w = seq[tr[i]][m]
            syms_tr.append(w[:, :7].argmax(axis=1).astype(int) if len(w) else np.array([], dtype=int))
        model = ClassPairCategoricalHMM(n_states=n_states, n_iter=20, random_state=f['fold'])
        model.fit(syms_tr, y[tr])
        syms_va = []
        for i in range(len(va)):
            m = mask[va[i]] > 0
            w = seq[va[i]][m]
            syms_va.append(w[:, :7].argmax(axis=1).astype(int) if len(w) else np.array([], dtype=int))
        p = model.predict_proba(syms_va)
        oof[va] += p
        m = compute_metrics(y[va], p, threshold)
        fold_records.append({'seed': f['seed'], 'fold': f['fold'], **m})
    return fold_records, oof


def summarize(fold_records, oof, y, seeds, threshold=0.5):
    """折级汇总 + 每 seed 的 OOF 指标 + 全体 OOF 指标。"""
    import pandas as pd
    df = pd.DataFrame(fold_records)
    per_seed = {}
    for s in sorted(df['seed'].unique()):
        idx = np.where(df['seed'] == s)[0]
        # 该 seed 下每个样本被预测一次
        per_seed[int(s)] = {'macro_f1': float(df.iloc[idx]['macro_f1'].mean()),
                            'accuracy': float(df.iloc[idx]['accuracy'].mean()),
                            'balanced_accuracy': float(df.iloc[idx]['balanced_accuracy'].mean()),
                            'roc_auc': float(df.iloc[idx]['roc_auc'].mean()),
                            'f1_class_1': float(df.iloc[idx]['f1_class_1'].mean())}
    overall = compute_metrics(y, oof, threshold)
    return {
        'per_fold_mean': {
            'macro_f1': float(df['macro_f1'].mean()), 'macro_f1_std': float(df['macro_f1'].std()),
            'accuracy': float(df['accuracy'].mean()), 'accuracy_std': float(df['accuracy'].std()),
            'balanced_accuracy': float(df['balanced_accuracy'].mean()),
            'balanced_accuracy_std': float(df['balanced_accuracy'].std()),
            'roc_auc': float(df['roc_auc'].mean()), 'roc_auc_std': float(df['roc_auc'].std()),
            'f1_class_1': float(df['f1_class_1'].mean()), 'f1_class_0': float(df['f1_class_0'].mean()),
        },
        'per_seed': per_seed,
        'overall_oof': overall,
        'n_folds': int(len(df)),
    }
