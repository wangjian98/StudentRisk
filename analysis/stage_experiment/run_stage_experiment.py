#!/usr/bin/env python3
"""
阶段级掉队预测 · 主实验：RF 基线 vs MetaMamba（11d / 9d / 7d）
==============================================================
协议（与本地 experiment_I 口径一致 + 246 上同一折）：
  · 5 折，按**学生**分组（同一学生不跨折），按学生课程级标签分层
  · 每个模型跑 3 seeds（42 / 123 / 777），OOF 概率取均值
  · 主指标：part 内 pooled AUC（个体判别力口径）+ 总体 AUC + judgeable
  · 基线：阶段级 part-prior（仅用当前 part 的掉队率先验）

用法：
  python3 run_stage_experiment.py --data stage_dataset.npz --models rf_full rf_no_active mm11
  python3 run_stage_experiment.py --data stage_dataset.npz --models mm11 --smoke   # 速度探测
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc
from models.base import set_seed
from models.meta_mamba.model import MetaMambaClassifier
from models.meta_mamba.train import train_one_fold


def build_rf_features(ds, kind, tr, te):
    DER, SELF, RAW = ds['DER'], ds['SELF'], ds['RAW']
    part = ds['part']
    ztr = sc.peer_z(ds, tr, tr, part)
    zte = sc.peer_z(ds, tr, te, part)
    A, VOL, B = DER[:, [0, 1, 2, 3]], DER[:, [0, 1]], DER[:, [4, 5, 6]]
    C = np.column_stack([DER[:, 7], DER[:, 8]])
    ACT = [7, 8, 9]
    if kind == 'rf_full':
        return (np.hstack([A[tr], B[tr], C[tr], ztr, SELF[tr]]),
                np.hstack([A[te], B[te], C[te], zte, SELF[te]]))
    if kind == 'rf_no_active':
        return (np.hstack([B[tr], C[tr], np.delete(ztr, ACT, 1), np.delete(SELF[tr], ACT, 1)]),
                np.hstack([B[te], C[te], np.delete(zte, ACT, 1), np.delete(SELF[te], ACT, 1)]))
    if kind == 'rf_active_only':
        return A[tr], A[te]
    if kind == 'rf_volume_only':
        return VOL[tr], VOL[te]
    if kind == 'rf_completion_only':
        return C[tr], C[te]
    if kind == 'rf_peer_all':
        return ztr, zte
    raise ValueError(kind)


def run_rf(ds, kind, folds, seeds):
    from sklearn.ensemble import RandomForestClassifier
    n = len(ds['drop'])
    oof = np.zeros(n)
    for tr, te in folds:
        Xtr, Xte = build_rf_features(ds, kind, tr, te)
        acc = np.zeros(len(te))
        for sd in seeds:
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                                       class_weight='balanced', random_state=sd, n_jobs=-1)
            m.fit(Xtr, ds['drop'][tr])
            acc += m.predict_proba(Xte)[:, 1]
        oof[te] = acc / len(seeds)
    return oof


def run_metamamba(ds, variant, folds, seeds, args, device):
    seq = {'mm11': ds['seq11'], 'mm9': ds['seq9'], 'mm7': ds['seq7']}[variant]
    d_event = seq.shape[2]
    n = len(ds['drop'])
    oof = np.zeros(n)
    fold_times = []
    for fi, (tr, te) in enumerate(folds):
        t0 = time.time()
        acc = np.zeros(len(te))
        for sd in seeds:
            set_seed(sd)
            model = MetaMambaClassifier(
                d_event=d_event, d_model=args.d_model, d_state=16, n_layers=2,
                n_tasks=7, dropout=args.dropout)
            model, _ = train_one_fold(
                model, seq[tr], ds['mask'][tr], ds['task_ids'][tr], ds['drop'][tr],
                seq[te], ds['mask'][te], ds['task_ids'][te], ds['drop'][te],
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                batch_size=args.batch_size, patience=args.patience,
                contrastive_weight=args.tc, use_film=(not args.no_film), device=device)
            model.eval()
            with torch.no_grad():
                p = model(torch.from_numpy(seq[te]).float().to(device),
                          torch.from_numpy(ds['mask'][te]).float().to(device),
                          torch.from_numpy(ds['task_ids'][te]).long().to(device))
            acc += torch.sigmoid(p).cpu().numpy()
        oof[te] = acc / len(seeds)
        dt = time.time() - t0
        fold_times.append(dt)
        print(f'    [{variant}] fold{fi}: n_tr={len(tr)} n_te={len(te)} {dt:.0f}s', flush=True)
    return oof, fold_times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', default='results/stage_experiment')
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 777])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-3)
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--tc', type=float, default=0.0, help='task-contrastive 权重')
    ap.add_argument('--device', default=None)
    ap.add_argument('--smoke', action='store_true', help='只跑 1 折 1 seed 3 epoch 测速')
    ap.add_argument('--tag', default='')
    ap.add_argument('--no-film', action='store_true', help='消融：关闭任务感知 FiLM 调制')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}  torch={torch.__version__}  cuda={torch.cuda.is_available()}',
          flush=True)

    ds = sc.load_dataset(args.data)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part'])
    ds['DER'] = sc.derive(ds)
    ds['SELF'] = sc.self_ref(ds)
    y = ds['drop']
    part = ds['part']

    folds = sc.group_folds(ds['sid'], ds['final'], n_splits=args.folds, seed=42)
    print(f'阶段样本 {len(y)} / 学生 {len(np.unique(ds["sid"]))} / 掉队正例 {int(y.sum())} '
          f'/ 折大小 {[len(te) for _, te in folds]}', flush=True)

    if args.smoke:
        folds = folds[:1]
        args.seeds = args.seeds[:1]
        args.epochs = 3

    prior = sc.stage_prior_oof(y, part, folds)
    res = {'meta': dict(device=device, torch=torch.__version__, seeds=args.seeds,
                        folds=args.folds, epochs=args.epochs, d_model=args.d_model,
                        tc=args.tc, tag=args.tag,
                        n_stage=int(len(y)), n_students=int(len(np.unique(ds['sid'])))),
           'models': {}}
    m = sc.metrics(y, prior, part, 'stage_part_prior')
    print(f'  {"stage_part_prior":22s} 总体={m["auc"]:.4f} | part内pooled={m["pooled"]:.4f}',
          flush=True)
    res['models']['stage_part_prior'] = m

    oofs = {'stage_part_prior': prior}
    for name in args.models:
        t0 = time.time()
        print(f'\n  ==== {name} ====', flush=True)
        if name.startswith('rf'):
            oof = run_rf(ds, name, folds, args.seeds)
            ft = []
        else:
            oof, ft = run_metamamba(ds, name, folds, args.seeds, args, device)
        oofs[name] = oof
        m = sc.metrics(y, oof, part, name)
        m['seconds'] = round(time.time() - t0, 1)
        m['fold_seconds'] = [round(x, 1) for x in ft]
        print(f'  {name:22s} 总体={m["auc"]:.4f} | part内pooled={m["pooled"]:.4f} | '
              f'judgeable={m["judged"]:.4f} | {m["seconds"]}s', flush=True)
        for k, v in sorted(m['per_part'].items(), key=lambda x: int(x[0])):
            print(f'      part{k}: n={v["n"]:4d} pos={v["pos"]:4d} AUC={v["auc"]:.4f}', flush=True)
        res['models'][name] = m

    # 与基线的配对 bootstrap
    boots = {}
    base = 'stage_part_prior'
    for name in args.models:
        if name in oofs:
            d = sc.paired_boot(y, oofs[name], oofs[base], part)
            boots[f'{name}_minus_{base}'] = d
            print(f'  [bootstrap] {name} − {base}: Δpooled={d["obs"]:+.4f} '
                  f'[{d["lo"]:+.4f},{d["hi"]:+.4f}] P(≤0)={d["p_le0"]*100:.2f}%', flush=True)
    res['bootstrap'] = boots

    tag = f'_{args.tag}' if args.tag else ''
    with open(os.path.join(args.out, f'stage_results{tag}.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(args.out, f'stage_oof{tag}.npz'),
                        **{k: v for k, v in oofs.items()})
    print(f'\n写出 {args.out}/stage_results{tag}.json', flush=True)


if __name__ == '__main__':
    main()
