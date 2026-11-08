"""
regimemamba.run_lopo — LOPO（留一 problem-part 任务）跨任务迁移，对齐 rev12/exp4_lopo.py 的口径

对每名学生取"主 part"作为 task_id；留出该 part 的全部学生做测试，其余学生训练。
每个 task × seed 独立训练一次；超参在第一个 task 上按内层验证选优后冻结（与 rev12 相同做法）。
输出：results/regimemamba/lopo/<variant>/{results.json, fold_metrics.csv}
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
from analysis.rev12.common import (load_cache, build_space, make_folds, save_json,
                                   save_fold_csv, compute_metrics, inner_split,
                                   deconfound_seq)
from analysis.rev12.models import set_seed
from analysis.regimemamba.run_exp import (train_regime, predict_regime, fold_data,
                                           HP_SPACE, TRAIN)
from analysis.regimemamba.model import RegimeMambaClassifier

OUT_ROOT = os.path.join(_ROOT, 'results', 'regimemamba', 'lopo')


def select_hp(seq, mask, task, y, dev, n_regime, device, log):
    best, best_score = None, -1
    for hp in HP_SPACE:
        d, _, _, _ = fold_data(seq, mask, task, y, dev)
        m = train_regime(hp, d, n_regime, device=device, seed=dev['seed'] * 1000 + dev['fold'])
        p, _ = predict_regime(m, d['seq_va'], d['mask_va'], d['task_va'], device)
        s = compute_metrics(d['y_va'], p)['macro_f1']
        log(f"    [hp] {hp} -> {s:.4f}")
        if s > best_score:
            best, best_score = hp, s
    return best, best_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='lopo')
    ap.add_argument('--dim', type=int, default=28)
    ap.add_argument('--n_regime', type=int, default=4)
    ap.add_argument('--seeds', default='42,123,777')
    ap.add_argument('--max_len', type=int, default=128)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(','))
    log = lambda s: print(s, flush=True)

    cache = load_cache()
    y = cache['y'].astype(int)
    part, offsets, student_ids = cache['part'], cache['offsets'], cache['student_ids']
    task_ids = np.zeros(len(student_ids), dtype=np.int64)
    for i in range(len(student_ids)):
        s, e = offsets[i], offsets[i + 1]
        if e > s:
            vals, cnt = np.unique(part[s:e], return_counts=True)
            task_ids[i] = int(vals[np.argmax(cnt)]) - 1

    all_rows = []
    for vname, dec in [('default', False), ('deconf', True)]:
        seq, mask, task = build_space(cache, args.dim, 1.0, args.max_len)
        if dec:
            seq = deconfound_seq(seq)
            task = np.zeros_like(task)
        out_dir = os.path.join(OUT_ROOT, args.tag, vname)
        os.makedirs(out_dir, exist_ok=True)
        hp_cache = {'hp': None}
        for p_task in sorted(set(task_ids.tolist())):
            te = np.where(task_ids == p_task)[0]
            tr = np.where(task_ids != p_task)[0]
            if len(te) < 10 or len(tr) < 20:
                continue
            for seed in seeds:
                set_seed(seed * 1000 + p_task)
                if hp_cache['hp'] is None:
                    hp_cache['hp'], _ = select_hp(seq, mask, task, y,
                                                 {'seed': seed, 'fold': p_task, 'tr': tr, 'va': te},
                                                 args.n_regime, args.device, log)
                hp = hp_cache['hp']
                d, _, _, _ = fold_data(seq, mask, task, y,
                                        {'seed': seed, 'fold': p_task, 'tr': tr, 'va': te})
                model = train_regime(hp, d, args.n_regime, device=args.device,
                                     seed=seed * 1000 + p_task)
                p, _ = predict_regime(model, seq[te], mask[te], task[te], args.device)
                m = compute_metrics(y[te], p)
                all_rows.append({'variant': vname, 'task': int(p_task), 'seed': int(seed),
                                 'n_test': int(len(te)), **m})
                log(f"    [lopo/{vname}] task={p_task} seed={seed} auc={m['roc_auc']:.4f} "
                    f"acc={m['accuracy']:.4f} balAcc={m['balanced_accuracy']:.4f}")

    import pandas as pd
    df = pd.DataFrame(all_rows)
    summary = {}
    for vname in df['variant'].unique():
        sub = df[df['variant'] == vname]
        per_task = sub.groupby('task').agg(
            auc=('roc_auc', 'mean'), acc=('accuracy', 'mean'),
            bal=('balanced_accuracy', 'mean'), n=('n_test', 'first')).reset_index()
        w = per_task['n'].values / per_task['n'].sum()
        summary[vname] = {
            'auc_weighted': float((per_task['auc'].values * w).sum()),
            'acc_weighted': float((per_task['acc'].values * w).sum()),
            'bal_weighted': float((per_task['bal'].values * w).sum()),
            'auc_mean_over_tasks': float(per_task['auc'].mean()),
            'per_task': per_task.to_dict('records'),
        }
    payload = {'tag': args.tag, 'arch': 'regimemamba', 'dim': args.dim,
               'n_regime': args.n_regime, 'seeds': list(seeds), 'summary': summary}
    save_json(os.path.join(OUT_ROOT, args.tag, 'results.json'), payload)
    save_fold_csv(os.path.join(OUT_ROOT, args.tag, 'fold_metrics.csv'), all_rows)
    log(f"[lopo] {json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)} for k, v in summary.items()})}")


if __name__ == '__main__':
    main()
