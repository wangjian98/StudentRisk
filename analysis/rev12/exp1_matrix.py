"""
rev12.exp1_matrix — 实验 1：模型 × 输入空间 对照矩阵（审稿意见 调整1 + 调整3）

矩阵：{RF, LSTM, BiLSTM, Attention, MetaMamba} × {7d, 11d, 28d}
      + HMM 家族（HMM-Multinomial@7d、HMM-Gaussian@11d、HMM-GMM@28d）
协议：5 折 × 3 seeds；训练折内 15% 内层验证；每架构 4 个候选超参（对称预算），
      max_epochs=200、patience=20（对所有深度架构一致，含 MetaMamba）
输出：results/rev12/exp1_matrix/<model>_d<dim>/{fold_metrics.csv, results.json, oof_probs.npy}
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

from analysis.rev12.common import (build_cache, load_cache, build_space, build_tabular,
                                   make_folds, save_json, save_fold_csv, cache_dir,
                                   deconfound_seq, deconfound_tabular)
from analysis.rev12.models import HP_SPACE, DEEP_ARCHS
from analysis.rev12 import runner as R

OUT_ROOT = os.path.join(_ROOT, 'results', 'rev12')


def run_cell(arch, dim, cache, y, folds, out_dir, device='cuda', prefix_frac=1.0,
             max_len=128, use_film=True, log=print, prebuilt=None, deconfound=False):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    if prebuilt is not None:
        seq, mask, task = prebuilt['seq'], prebuilt['mask'], prebuilt['task']
        X, names = prebuilt['X'], prebuilt['names']
    elif arch == 'rf7_all':
        from analysis.rev12.common import build_all_event_counts
        X, names = build_all_event_counts(cache)
        seq, mask, task = (None, None, None)
    else:
        seq, mask, task = build_space(cache, dim, prefix_frac, max_len)
        X, names = build_tabular(cache, dim, prefix_frac, max_len)
    if deconfound and seq is not None:
        seq = deconfound_seq(seq)
        X, names = deconfound_tabular(X, names)
        # 去混淆必须同时切断 task 身份：否则 FiLM 仍按 part 调制 → 仍泄漏作业身份
        task = np.zeros_like(task)

    hp, hp_score, dev_fold = None, None, folds[0]
    if arch == 'rf7_all':
        # 遗留基线：固定配置（与既有 RF-7d 结果同口径，用于连续性对照）
        hp = {'n_estimators': 200, 'max_depth': 10, 'min_samples_leaf': 1}
        log(f"  [hp] RF-7d(all-events, legacy): {hp}")
        fr, oof = R.run_rf(X, y, folds, hp)
    elif arch == 'rf':
        hp, hp_score = R.select_rf_hp(X, y, dev_fold, HP_SPACE['rf'])
        log(f"  [hp] RF d{dim}: {hp} (dev inner macroF1={hp_score:.4f})")
        fr, oof = R.run_rf(X, y, folds, hp)
    elif arch in DEEP_ARCHS:
        hp, hp_score = R.select_deep_hp(arch, seq, mask, task, y, dev_fold,
                                        HP_SPACE[arch], use_film=use_film, device=device)
        log(f"  [hp] {arch} d{dim}: {hp} (dev inner macroF1={hp_score:.4f})")
        fr, oof, _ = R.run_deep(arch, seq, mask, task, y, folds, hp, use_film=use_film,
                                device=device, log=log)
    elif arch == 'hmm_cat':
        hp = {'n_states': 3}
        fr, oof = R.run_hmm_cat(seq, mask, y, folds, n_states=3)
    elif arch == 'hmm_gmm':
        hp = {'n_states': 3, 'n_mix': 2}
        fr, oof = R.run_hmm_gmm(seq, mask, y, folds, n_states=3, n_mix=2)
    else:
        raise ValueError(arch)

    n_seeds = len({f['seed'] for f in folds})
    oof = oof / max(n_seeds, 1)          # 多 seed 累加 → 取均值
    summary = R.summarize(fr, oof, y, sorted({f['seed'] for f in folds}))
    payload = {
        'arch': arch, 'dim': dim, 'hp': hp, 'hp_dev_score': hp_score,
        'prefix_frac': prefix_frac, 'max_len': max_len, 'use_film': use_film,
        'deconfound': deconfound,
        'n_folds': len(folds), 'seeds': sorted({f['seed'] for f in folds}),
        'summary': summary, 'elapsed_s': time.time() - t0,
        'protocol': '5fold x 3seed stratified-by-student; inner 15% val for early stop + HP',
    }
    save_json(os.path.join(out_dir, 'results.json'), payload)
    save_fold_csv(os.path.join(out_dir, 'fold_metrics.csv'), fr)
    np.save(os.path.join(out_dir, 'oof_probs.npy'), oof)
    log(f"  -> {arch} d{dim}: macroF1={summary['per_fold_mean']['macro_f1']:.4f} "
        f"acc={summary['per_fold_mean']['accuracy']:.4f} "
        f"balAcc={summary['per_fold_mean']['balanced_accuracy']:.4f} "
        f"auc={summary['per_fold_mean']['roc_auc']:.4f} "
        f"({time.time()-t0:.1f}s)")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spaces', default='7,11,28')
    ap.add_argument('--archs', default='rf,lstm,bilstm,attention,metamamba,hmm_cat,hmm_gmm')
    ap.add_argument('--seeds', default='42,123,777')
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tag', default='exp1_matrix')
    ap.add_argument('--no-film', action='store_true')
    ap.add_argument('--deconfound', action='store_true',
                    help='去掉 part/exercise 身份通道（消除 part-prior 捷径）')
    ap.add_argument('--only', default='',
                    help='只跑指定 cell，逗号分隔（如 rf_d7,metamamba_d11）')
    args = ap.parse_args()

    spaces = [int(s) for s in args.spaces.split(',') if s]
    archs = [a for a in args.archs.split(',') if a]
    seeds = tuple(int(s) for s in args.seeds.split(','))

    cache = load_cache()
    y = cache['y']
    folds = make_folds(y, seeds, args.n_splits)
    print(f"[exp1] n_students={len(y)} fail_rate={y.mean():.4f} folds={len(folds)}", flush=True)

    base = os.path.join(OUT_ROOT, args.tag)
    only = {c.strip() for c in args.only.split(',') if c.strip()}
    for dim in spaces:
        for arch in archs:
            if arch == 'rf7_all':
                dim = 7
            if only and f'{arch}_d{dim}' not in only:
                continue
            # HMM 家族只在自然空间上跑（Multinomial@7d, GMM@28d），避免无意义组合
            if arch == 'hmm_cat' and dim != 7:
                continue
            if arch == 'hmm_gmm' and dim != 28:
                continue
            out_dir = os.path.join(base, f"{arch}_d{dim}")
            if os.path.exists(os.path.join(out_dir, 'results.json')):
                print(f"[exp1] skip {arch} d{dim} (done)", flush=True)
                continue
            print(f"[exp1] === {arch} d{dim} ===", flush=True)
            try:
                run_cell(arch, dim, cache, y, folds, out_dir, device=args.device,
                         max_len=args.max_len, use_film=not args.no_film,
                         deconfound=args.deconfound)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[exp1] FAILED {arch} d{dim}: {e}", flush=True)
    print("[exp1] done", flush=True)


if __name__ == '__main__':
    main()
