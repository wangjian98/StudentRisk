"""MetaMamba-7d training script.

Same architecture as Meta-Mamba (S6 + FiLM + Task-Contrastive + FOMAML),
but input is 7-dim event-type one-hot only (no 4 continuous features).
"""
import os
import sys
import json
import argparse
import time
import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import load_dataset
from models.base import set_seed, evaluate_predictions, save_results
from models.meta_mamba.model import MetaMambaClassifier
from models.meta_mamba.train import train_one_fold, run_maml_fewshot, task_contrastive_loss
from models.meta_mamba_7d.data import build_event_sequences_7d


def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(_ROOT, 'configs', 'default.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def run(seeds=(42, 123, 777), n_splits: int = 5, threshold: float = 0.5,
        out_dir: str = None, save_oof: bool = True, config: dict = None,
        device: str = None, max_len: int = 128, run_fewshot: bool = True,
        use_film: bool = True, use_tc: bool = True):
    if config is None:
        config = load_config()
    mcfg = config.get('meta_mamba', config.get('attention', {}))
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if out_dir is None:
        out_dir = os.path.join(_ROOT, 'results', 'meta_mamba_7d')
    os.makedirs(out_dir, exist_ok=True)

    print(f"[MetaMamba-7d] device={device}, max_len={max_len}", flush=True)
    print(f"[MetaMamba-7d] Loading data + building 7-dim event sequences ...", flush=True)
    ide_logs, labels_df, y, student_ids = load_dataset()
    sequences, masks, task_ids = build_event_sequences_7d(ide_logs, student_ids, max_len=max_len)
    n_tasks = int(task_ids.max()) + 1
    print(f"[MetaMamba-7d] sequences.shape={sequences.shape}, n_tasks={n_tasks}", flush=True)

    n = len(y)
    oof = np.zeros(n)
    fold_idx = np.zeros(n, dtype=int)
    fold_records = []

    t0 = time.time()
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx_, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n), y)):
            set_seed(seed * 1000 + fold_idx_)
            model = MetaMambaClassifier(
                d_event=sequences.shape[-1],  # 7
                d_model=mcfg.get('d_model', 64),
                d_state=mcfg.get('d_state', 16),
                n_layers=mcfg.get('n_layers', 2),
                n_tasks=n_tasks,
                dropout=mcfg.get('dropout', 0.2),
            ).to(device)
            model, _ = train_one_fold(
                model,
                sequences[tr_idx], masks[tr_idx], task_ids[tr_idx], y[tr_idx],
                sequences[va_idx], masks[va_idx], task_ids[va_idx], y[va_idx],
                epochs=mcfg.get('epochs', 40),
                lr=mcfg.get('lr', 1e-3),
                weight_decay=mcfg.get('weight_decay', 1e-3),
                batch_size=mcfg.get('batch_size', 16),
                patience=mcfg.get('patience', 10),
                contrastive_weight=(mcfg.get('contrastive_weight', 0.3) if use_tc else 0.0),
                use_film=use_film,
                device=device,
            )
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(
                    torch.from_numpy(sequences[va_idx]).float().to(device),
                    torch.from_numpy(masks[va_idx]).float().to(device),
                    torch.from_numpy(task_ids[va_idx]).long().to(device),
                )).cpu().numpy()
            oof[va_idx] += p
            fold_idx[va_idx] = fold_idx_
            fold_m = evaluate_predictions(y[va_idx], p, threshold=threshold)
            fold_records.append({'seed': seed, 'fold': fold_idx_, **fold_m})
        print(f"[MetaMamba-7d]   seed={seed} done at {time.time()-t0:.1f}s", flush=True)

    oof /= len(seeds)
    elapsed = time.time() - t0
    overall = evaluate_predictions(y, oof, threshold=threshold)
    fold_df = pd.DataFrame(fold_records)
    fold_summary = {
        'macro_f1_mean': float(fold_df['macro_f1'].mean()),
        'macro_f1_std':  float(fold_df['macro_f1'].std()),
        'f1_class_1_mean': float(fold_df['f1_class_1'].mean()),
        'f1_class_1_std':  float(fold_df['f1_class_1'].std()),
        'roc_auc_mean':   float(fold_df['roc_auc'].mean()),
        'roc_auc_std':    float(fold_df['roc_auc'].std()),
    }
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # FOMAML 5-shot
    fewshot = None
    if run_fewshot:
        try:
            fewshot = run_maml_fewshot(
                model, sequences, masks, task_ids, y,
                n_way=min(n_tasks, 4), k_shot=5, n_query=10,
                inner_lr=0.01, inner_steps=3, seeds=(42,), device=device,
            )
            print(f"[MetaMamba-7d] FOMAML few-shot F1 = "
                  f"{fewshot['mean_f1']:.4f} ± {fewshot['std_f1']:.4f} "
                  f"on {fewshot['n_tasks_evaluated']} tasks", flush=True)
        except Exception as e:
            print(f"[MetaMamba-7d] Few-shot eval failed: {e}", flush=True)
            fewshot = {'error': str(e)}

    payload = {
        'model': 'MetaMamba-7d',
        'ablation': {'no_film': not use_film, 'no_tc': not use_tc},
        'feature_dimension': 7,
        'config': mcfg,
        'n_params': n_params,
        'threshold': threshold,
        'n_seeds': len(seeds),
        'n_splits': n_splits,
        'n_students': n,
        'fail_rate': float(y.mean()),
        'n_tasks': n_tasks,
        'label_convention': 'Failed=1, Passed=0',
        'input_type': 'event_sequence_7d',
        'overall': overall,
        'per_fold_summary': fold_summary,
        'per_fold': fold_records,
        'fewshot_fomaml': fewshot,
        'elapsed_seconds': elapsed,
    }
    save_results(out_dir, payload)
    fold_df.to_csv(os.path.join(out_dir, 'fold_metrics.csv'), index=False)
    if save_oof:
        np.save(os.path.join(out_dir, 'oof_probs.npy'), oof)
        np.save(os.path.join(out_dir, 'labels.npy'), y)
        np.save(os.path.join(out_dir, 'fold_idx.npy'), fold_idx)
        np.save(os.path.join(out_dir, 'student_ids.npy'), student_ids)
        np.save(os.path.join(out_dir, 'task_ids.npy'), task_ids)
    with open(os.path.join(out_dir, 'config_used.json'), 'w') as f:
        json.dump({'seeds': list(seeds), 'n_splits': n_splits, 'threshold': threshold,
                   'meta_mamba_cfg': mcfg, 'n_params': n_params,
                   'max_len': max_len, 'n_tasks': n_tasks,
                   'feature_dimension': 7}, f, indent=2)

    print(f"\n[MetaMamba-7d] DONE in {elapsed:.1f}s. Overall: "
          f"acc={overall['accuracy']:.4f}, macro_f1={overall['macro_f1']:.4f}, "
          f"f1_failed={overall['f1_class_1']:.4f}, roc_auc={overall['roc_auc']:.4f}", flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--max-len', type=int, default=128)
    parser.add_argument('--out-dir', type=str, default=None)
    parser.add_argument('--no-oof', action='store_true')
    parser.add_argument('--no-fewshot', action='store_true')
    parser.add_argument('--no-film', action='store_true', help='Ablation: remove FiLM modulation')
    parser.add_argument('--no-tc', action='store_true', help='Ablation: remove Task-Contrastive loss')
    args = parser.parse_args()
    config = load_config()
    seeds = args.seeds if args.seeds else config.get('cv', {}).get('seeds', [42, 123, 777])
    n_splits = args.n_splits or config.get('cv', {}).get('n_splits', 5)
    run(seeds=seeds, n_splits=n_splits, threshold=args.threshold,
        out_dir=args.out_dir, save_oof=not args.no_oof, config=config,
        max_len=args.max_len, run_fewshot=not args.no_fewshot,
        use_film=not args.no_film, use_tc=not args.no_tc)


if __name__ == '__main__':
    main()