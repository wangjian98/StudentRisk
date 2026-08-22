#!/usr/bin/env python3
"""nested_cv.py — Mini Nested Cross-Validation on MetaMamba-7d.

Strict nested CV: outer test fold is NEVER seen during HP selection.
Each outer fold independently:
  1. Inner: train 4 HP configs on (75% of outer-train), evaluate on (25% of outer-train)
  2. Pick best HP by inner Macro-F1
  3. Refit on full outer-train with best HP, evaluate on outer-test

Total: 5 outer × (4 inner + 1 refit) = 25 single-fold trainings.
Per-fold training on T4 ~ 70-90s -> estimated total ~30-40 min.

Outputs (in outputs/nested_cv/):
- per_outer.jsonl  (incremental, one JSON per line per completed outer fold)
- summary.json     (final aggregated stats)
- summary.md       (human-readable report)
- progress.log     (timestamped training log)
"""
import os
import sys
import json
import time
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from scipy import stats

ROOT = '/home/ubuntu/StudentRisk'
sys.path.insert(0, ROOT)

from data import load_dataset
from models.base import set_seed, evaluate_predictions
from models.meta_mamba.model import MetaMambaClassifier
from models.meta_mamba.train import train_one_fold
from models.meta_mamba_7d.data import build_event_sequences_7d

OUTER_K = 5
INNER_K = 2
HP_GRID = [
    {'lr': 1e-3, 'contrastive_weight': 0.0, 'tag': 'lr1e-3_noTC'},
    {'lr': 1e-3, 'contrastive_weight': 0.3, 'tag': 'lr1e-3_TC'},
    {'lr': 5e-4, 'contrastive_weight': 0.0, 'tag': 'lr5e-4_noTC'},
    {'lr': 5e-4, 'contrastive_weight': 0.3, 'tag': 'lr5e-4_TC'},
]
OUTER_SEED = 42
INNER_SEED_BASE = 42
EPOCHS = 40
PATIENCE = 10

OUT_DIR = os.path.join(ROOT, 'outputs', 'nested_cv')
os.makedirs(OUT_DIR, exist_ok=True)
PROGRESS_LOG = os.path.join(OUT_DIR, 'progress.log')
PER_OUTER = os.path.join(OUT_DIR, 'per_outer.jsonl')


def log(msg, also_print=True):
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
    with open(PROGRESS_LOG, 'a') as f:
        f.write(line + '\n')
    if also_print:
        print(line, flush=True)


def train_one(tr_idx, va_idx, hp, n_tasks, sequences, masks, task_ids, y, device):
    set_seed(OUTER_SEED)
    model = MetaMambaClassifier(
        d_event=sequences.shape[-1],
        d_model=64,
        d_state=16,
        n_layers=2,
        n_tasks=n_tasks,
        dropout=0.2,
    ).to(device)
    model, _ = train_one_fold(
        model,
        sequences[tr_idx], masks[tr_idx], task_ids[tr_idx], y[tr_idx],
        sequences[va_idx], masks[va_idx], task_ids[va_idx], y[va_idx],
        epochs=EPOCHS,
        lr=hp['lr'],
        weight_decay=1e-3,
        batch_size=16,
        patience=PATIENCE,
        contrastive_weight=hp['contrastive_weight'],
        use_film=True,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(
            torch.from_numpy(sequences[va_idx]).float().to(device),
            torch.from_numpy(masks[va_idx]).float().to(device),
            torch.from_numpy(task_ids[va_idx]).long().to(device),
        )).cpu().numpy()
    m = evaluate_predictions(y[va_idx], p, threshold=0.5)
    return m


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'Nested CV start. device={device}, OUTER_K={OUTER_K}, INNER_K={INNER_K}, |HP_GRID|={len(HP_GRID)}')

    log('Loading dataset ...')
    ide_logs, labels_df, y, student_ids = load_dataset()
    sequences, masks, task_ids = build_event_sequences_7d(ide_logs, student_ids, max_len=128)
    n = len(y)
    n_tasks = int(task_ids.max()) + 1
    log(f'Data loaded: sequences={sequences.shape}, n_tasks={n_tasks}, n={n}')

    skf_outer = StratifiedKFold(n_splits=OUTER_K, shuffle=True, random_state=OUTER_SEED)
    outer_splits = list(skf_outer.split(np.zeros(n), y))
    log(f'Outer splits ready: train sizes = {[len(tr) for tr, _ in outer_splits]}')

    outer_test_records = []
    total_t0 = time.time()
    for outer_i, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        outer_t0 = time.time()
        log(f'--- Outer {outer_i+1}/{OUTER_K}: train={len(outer_train_idx)}, test={len(outer_test_idx)} ---')

        skf_inner = StratifiedKFold(n_splits=INNER_K, shuffle=True, random_state=INNER_SEED_BASE + outer_i)
        inner_splits = list(skf_inner.split(np.zeros(len(outer_train_idx)), y[outer_train_idx]))
        inner_train_local, inner_val_local = inner_splits[0]
        inner_train_idx = outer_train_idx[inner_train_local]
        inner_val_idx = outer_train_idx[inner_val_local]
        log(f'  Inner: train={len(inner_train_idx)}, val={len(inner_val_idx)}')

        hp_val_scores = []
        for hp_i, hp in enumerate(HP_GRID):
            t0 = time.time()
            m = train_one(inner_train_idx, inner_val_idx, hp, n_tasks, sequences, masks, task_ids, y, device)
            elapsed = time.time() - t0
            log(f'  HP[{hp_i}] {hp["tag"]}: val macro_f1={m["macro_f1"]:.4f} ({elapsed:.1f}s)')
            hp_val_scores.append({'hp': hp, 'val_macro_f1': m['macro_f1']})

        best_i = int(np.argmax([s['val_macro_f1'] for s in hp_val_scores]))
        best_hp = hp_val_scores[best_i]['hp']
        log(f'  Selected HP: {best_hp["tag"]} (val macro_f1={hp_val_scores[best_i]["val_macro_f1"]:.4f})')

        t0 = time.time()
        outer_test_metrics = train_one(outer_train_idx, outer_test_idx, best_hp, n_tasks, sequences, masks, task_ids, y, device)
        refit_elapsed = time.time() - t0
        log(f'  REFIT on outer-train, eval on outer-test: macro_f1={outer_test_metrics["macro_f1"]:.4f} ({refit_elapsed:.1f}s)')

        outer_test_records.append({
            'outer_i': outer_i,
            'best_hp': best_hp,
            'hp_val_scores': hp_val_scores,
            'outer_test_macro_f1': outer_test_metrics['macro_f1'],
            'outer_test_full_metrics': outer_test_metrics,
            'refit_seconds': refit_elapsed,
        })
        with open(PER_OUTER, 'a') as f:
            f.write(json.dumps(outer_test_records[-1]) + '\n')

        outer_total = time.time() - outer_t0
        log(f'Outer {outer_i+1} done in {outer_total:.1f}s')

    test_f1 = np.array([r['outer_test_macro_f1'] for r in outer_test_records])
    mean_f1 = float(test_f1.mean())
    std_f1 = float(test_f1.std(ddof=1))
    se = std_f1 / np.sqrt(OUTER_K)
    t_crit = stats.t.ppf(0.975, df=OUTER_K - 1)
    ci_low = mean_f1 - t_crit * se
    ci_high = mean_f1 + t_crit * se

    summary = {
        'method': 'Mini Nested CV (5 outer x 2 inner x 4 HP)',
        'reference_model': 'MetaMamba-7d',
        'outer_seed': OUTER_SEED,
        'outer_k': OUTER_K,
        'inner_k': INNER_K,
        'hp_grid': HP_GRID,
        'outer_test_macro_f1': [float(x) for x in test_f1],
        'mean_macro_f1': mean_f1,
        'std_macro_f1': std_f1,
        'se_macro_f1': float(se),
        'ci_95_low': float(ci_low),
        'ci_95_high': float(ci_high),
        'total_seconds': time.time() - total_t0,
    }
    summary_path = os.path.join(OUT_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Compare with naive 8-fold estimate from outputs/comparison.csv
    naive_macro_f1 = None
    cmp_csv = os.path.join(ROOT, 'outputs', 'comparison.csv')
    if os.path.exists(cmp_csv):
        import csv as _csv
        with open(cmp_csv) as f:
            for row in _csv.DictReader(f):
                if row.get('model_key') == 'meta_mamba_7d':
                    naive_macro_f1 = float(row['macro_f1'])
                    break

    md = []
    md.append('# MetaMamba-7d — Mini Nested CV Report')
    md.append('')
    md.append(f'> Method: **5-fold outer × 2-fold inner × 4 HP grid** = 25 single-fold trainings')
    md.append(f'> Outer seed = {OUTER_SEED}, Inner seed base = {INNER_SEED_BASE}')
    md.append(f'> Wall time: {summary["total_seconds"]/60:.1f} min on T4')
    md.append('')
    md.append('## Headline')
    md.append('')
    md.append(f'- **Nested-CV Macro-F1**: **{mean_f1:.4f}** (std {std_f1:.4f}, 95% CI [{ci_low:.4f}, {ci_high:.4f}])')
    if naive_macro_f1 is not None:
        diff = naive_macro_f1 - mean_f1
        sign = 'optimistic' if diff > 0 else 'pessimistic'
        md.append(f'- Naive 8-fold × 3-seed estimate (from outputs/comparison.csv): **{naive_macro_f1:.4f}**')
        md.append(f'- Difference: {diff:+.4f} (naive {sign} by {abs(diff)*100:.2f} pp)')
    md.append('')
    md.append('## Per-outer-fold detail')
    md.append('')
    md.append('| Outer # | Best HP | Best Val Macro-F1 | Test Macro-F1 | Refit time (s) |')
    md.append('|---|---|---|---|---|')
    for r in outer_test_records:
        best_val = max(s['val_macro_f1'] for s in r['hp_val_scores'])
        md.append(f"| {r['outer_i']+1} | {r['best_hp']['tag']} | {best_val:.4f} | {r['outer_test_macro_f1']:.4f} | {r['refit_seconds']:.1f} |")
    md.append('')
    md.append('## HP selection (inner val Macro-F1 per outer fold)')
    md.append('')
    md.append('| Outer # | ' + ' | '.join(h['tag'] for h in HP_GRID) + ' |')
    md.append('|---|' + '---|' * len(HP_GRID))
    for r in outer_test_records:
        scores = [f"{s['val_macro_f1']:.4f}" for s in r['hp_val_scores']]
        md.append(f"| {r['outer_i']+1} | " + ' | '.join(scores) + ' |')
    md.append('')
    md.append('## Interpretation')
    md.append('')
    md.append('Nested CV controls for HP-selection optimism that naive single-CV estimates suffer from.')
    md.append('A drop from naive to nested is expected; the magnitude is the "optimism gap" — how much the naive')
    md.append('estimate overfits by peeking at the validation set during HP choice.')
    md.append('Negative bias (naive < nested) would be unusual.')

    md_path = os.path.join(OUT_DIR, 'summary.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md))

    log(f'Nested CV DONE. mean macro_f1={mean_f1:.4f} ± {std_f1:.4f}, '
        f'95% CI=[{ci_low:.4f}, {ci_high:.4f}], total={summary["total_seconds"]:.0f}s')
    log(f'Wrote {summary_path}, {md_path}')


if __name__ == '__main__':
    main()