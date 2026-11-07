"""
rev12.exp3_film — 实验 3：FiLM 消融的稳定复现与机制解释（审稿意见 调整5 · 硬伤4）

硬伤 4：7d 去 FiLM 崩 41pp，11d 去 FiLM 只掉 1.3pp —— 同一模块两个极端，摘要却只引用
前者。审稿人必然追问。

本实验做三件事：
 1) **多 seed 复现**：{7d, 11d, 28d} × {FiLM on, FiLM off} × 5 seeds（25 折/格），
    给出均值 ± 标准差与配对检验，判断"7d 崩溃"是否稳定复现。
 2) **机制假设检验**：假设 "FiLM 在**稀疏/低维输入**下充当稳定器（task-aware 调制
    补偿弱输入），在**高维富输入**下可被其它通路替代"。
    可检验的预测：去 FiLM 的性能损失应随输入维度/信息量增加而**单调变小**。
    → 计算 Δ(去FiLM) vs 输入维度 的关系（7d / 11d / 28d）。
 3) **γ/β 可视化**：导出训练好的 FiLM 调制参数（按 task 展开），检验
    "FiLM 真的在按 part 调整表征"（γ 的 task 间离散度、β 的 task 间离散度）。

输出：results/rev12/exp3_film/<dim>_film{on,off}/  + film_params.npz
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

from analysis.rev12.common import (load_cache, build_space, make_folds, save_json,
                                   save_fold_csv, compute_metrics)
from analysis.rev12.models import HP_SPACE
from analysis.rev12 import runner as R

OUT_ROOT = os.path.join(_ROOT, 'results', 'rev12')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dims', default='7,11,28')
    ap.add_argument('--seeds', default='42,123,777,2024,7')
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tag', default='exp3_film')
    ap.add_argument('--only', default='',
                    help='只跑指定格，逗号分隔（如 7_on,11_off）')
    args = ap.parse_args()

    dims = [int(d) for d in args.dims.split(',') if d]
    seeds = tuple(int(s) for s in args.seeds.split(','))

    cache = load_cache()
    y = cache['y']
    folds = make_folds(y, seeds, args.n_splits)
    print(f"[exp3] students={len(y)} folds={len(folds)} seeds={seeds}", flush=True)

    base = os.path.join(OUT_ROOT, args.tag)
    summary = {}
    only = {c.strip() for c in args.only.split(',') if c.strip()}
    for dim in dims:
        seq, mask, task = build_space(cache, dim, 1.0, args.max_len)
        for use_film in (True, False):
            tag = f"d{dim}_film{'on' if use_film else 'off'}"
            if only and f"{dim}_{'on' if use_film else 'off'}" not in only:
                continue
            out_dir = os.path.join(base, tag)
            if os.path.exists(os.path.join(out_dir, 'results.json')):
                print(f"[exp3] skip {tag} (done)", flush=True)
                continue
            os.makedirs(out_dir, exist_ok=True)
            print(f"[exp3] === {tag} ===", flush=True)
            t0 = time.time()
            dev = folds[0]
            hp, hp_score = R.select_deep_hp('metamamba', seq, mask, task, y, dev,
                                            HP_SPACE['metamamba'], use_film=use_film,
                                            device=args.device)
            print(f"  [hp] {hp} (dev inner macroF1={hp_score:.4f})", flush=True)
            fr, oof, film_dump = R.run_deep('metamamba', seq, mask, task, y, folds, hp,
                                            use_film=use_film, device=args.device,
                                            collect_film=True,
                                            log=lambda s: print(s, flush=True))
            s = R.summarize(fr, oof, y, sorted({f['seed'] for f in folds}))
            payload = {'arch': 'metamamba', 'dim': dim, 'use_film': use_film, 'hp': hp,
                       'hp_dev_score': hp_score, 'seeds': list(seeds),
                       'n_folds': len(folds), 'summary': s, 'elapsed_s': time.time() - t0}
            save_json(os.path.join(out_dir, 'results.json'), payload)
            save_fold_csv(os.path.join(out_dir, 'fold_metrics.csv'), fr)
            np.save(os.path.join(out_dir, 'oof_probs.npy'), oof)
            if film_dump:
                np.savez(os.path.join(out_dir, 'film_params.npz'),
                         gamma=np.stack([d['gamma'] for d in film_dump]),
                         beta=np.stack([d['beta'] for d in film_dump]),
                         seeds=np.array([d['seed'] for d in film_dump]),
                         folds=np.array([d['fold'] for d in film_dump]))
            summary[tag] = s['per_fold_mean']
            print(f"  -> {tag}: macroF1={s['per_fold_mean']['macro_f1']:.4f} "
                  f"auc={s['per_fold_mean']['roc_auc']:.4f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    print("[exp3] done", flush=True)


if __name__ == '__main__':
    main()
