#!/usr/bin/env python3
"""
Layer 0 证据来源分解（段级 CS1-SHR 的 31 维上下文到底靠什么）
=================================================================================
为什么要做：

  段级主结果 pooled 0.9258 是 **31 维上下文** 打出来的：
      A(4 强度对数) + B(3 比率) + C(2 进度) + peer_z(12 同伴相对) + SELF(10 自参照)
  但论文 §4.2 声称"part 不变表征层"的价值在于
  **同伴相对位置 + 进度 + 自参照偏离**，并且 peer_z_submit 是全部 7 段的第 1 重要特征。
  → 这三条主张目前是**分别**成立的（重要性表），**没有做过联合消融**：
    拆掉 SELF 还剩多少？拆掉 peer_z 还剩多少？两个都拆呢？

  额外一个部署问题：**冷启动**。SELF 是"相对自己前几段"，新学生/新学期第一批学生
  没有前几段 → 段级模型在没有 SELF 时还剩多少判别力？（part1 行的 SELF 恒为 0，
  所以 no-self 变体在 part1 上与原模型逐位一致，可直接读出"第 1 段冷启动"的水位。）

协议：与主协议完全一致（按学生分组 5 折 + 课程级标签分层，seeds 42/123/777，
      peer 统计量只用训练折，同伴归一化同 stage_common.peer_z）。
指标：part 内 pooled AUC（主）+ 配对 bootstrap 2000 次（Δ vs full）。

跑法（246）：
  python3 analysis/stage_experiment/run_layer0_decomposition.py \
         --data data/stage_dataset.npz --out results/layer0_decomp
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402

A_COLS = [0, 1, 2, 3]
B_COLS = [4, 5, 6]
C_COLS = [7, 8]
N_PEER = 12
N_SELF = 10


def build(ds, tr, te):
    """与 run_stage_fair.build_context 完全同源的特征（只是分段返回，便于切片）。"""
    DER, SELF, part = ds['DER'], ds['SELF'], ds['part']
    ztr = sc.peer_z(ds, tr, tr, part)
    zte = sc.peer_z(ds, tr, te, part)
    A = DER[:, A_COLS]
    B = DER[:, B_COLS]
    C = np.column_stack([DER[:, 7], DER[:, 8]])
    return dict(A=A, B=B, C=C, peer=ztr, SELF=SELF, ztr=ztr, zte=zte)


def assemble(bl, tr, te, use_self=True, use_peer=True):
    parts = [bl['A'][tr], bl['B'][tr], bl['C'][tr]]
    if use_peer:
        parts.append(bl['ztr'])
    if use_self:
        parts.append(bl['SELF'][tr])
    Xtr = np.hstack(parts)
    parts = [bl['A'][te], bl['B'][te], bl['C'][te]]
    if use_peer:
        parts.append(bl['zte'])
    if use_self:
        parts.append(bl['SELF'][te])
    Xte = np.hstack(parts)
    return Xtr.astype(np.float32), Xte.astype(np.float32)


def run_rf(ds, folds, seeds, use_self=True, use_peer=True):
    oof = np.zeros(len(ds['drop']))
    for tr, te in folds:
        bl = build(ds, tr, te)
        Xtr, Xte = assemble(bl, tr, te, use_self, use_peer)
        acc = np.zeros(len(te))
        for sd in seeds:
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                       max_features='sqrt', class_weight='balanced',
                                       random_state=sd, n_jobs=-1)
            m.fit(Xtr, ds['drop'][tr])
            acc += m.predict_proba(Xte)[:, 1]
        oof[te] = acc / len(seeds)
    return oof


def main(args):
    ds = sc.load_dataset(args.data)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part_ids'])
    ds['DER'] = sc.derive(ds)
    ds['SELF'] = sc.self_ref(ds)
    folds = sc.group_folds(ds['sid'], ds['final'], n_splits=5, seed=42)
    seeds = [42, 123, 777]
    y, part = ds['drop'], ds['part_ids']

    variants = {
        'full_31d (主模型)': dict(use_self=True, use_peer=True),
        'no_self_21d (去自参照=冷启动)': dict(use_self=False, use_peer=True),
        'no_peer_19d (去同伴相对)': dict(use_self=True, use_peer=False),
        'neither_9d (只留本段强度+比率+进度)': dict(use_self=False, use_peer=False),
    }
    out, oofs = {}, {}
    for name, kw in variants.items():
        oof = run_rf(ds, folds, seeds, **kw)
        oofs[name] = oof
        pp = sc.per_part_auc(y, oof, part)
        rec = dict(pooled=round(sc.pooled(pp), 4), judged=round(sc.judged(pp), 4),
                   auc=round(float(__import__('sklearn.metrics', fromlist=['roc_auc_score'])
                                     .roc_auc_score(y, oof)), 4),
                   per_part={str(k): round(v['auc'], 4) for k, v in pp.items()})
        out[name] = rec
        print(f'  ✓ {name}: pooled={rec["pooled"]} judged={rec["judged"]}', flush=True)

    # 配对 bootstrap（Δ = 变体 − full，正数 = 变体更好）
    base = 'full_31d (主模型)'
    for name in variants:
        if name == base:
            continue
        d = sc.paired_boot(y, oofs[name], oofs[base], part, n=2000)
        out[name]['delta_vs_full'] = dict(
            pooled=round(float(d['obs']), 4),
            ci=[round(float(d['lo']), 4), round(float(d['hi']), 4)],
            p_le0=round(float(d['p_le0']), 4))
        print(f'    Δ pooled vs full = {d["obs"]:+.4f} '
              f'CI[{d["lo"]:+.4f},{d["hi"]:+.4f}] P(Δ≤0)={d["p_le0"]*100:.1f}%', flush=True)

    # 冷启动读数：part1 行（该生第一段，SELF 恒为 0）
    p1 = part == 1
    out['cold_start_part1'] = {}
    for name in variants:
        pp = sc.per_part_auc(y[p1], oofs[name][p1], part[p1])
        out['cold_start_part1'][name] = dict(
            n=int(p1.sum()), pos=int(y[p1].sum()),
            auc=round(pp.get(1, {}).get('auc', float('nan')), 4) if pp else None)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'layer0_decomp.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(args.out, 'oof_layer0.npz'),
                        **{f'oof_{i}': o for i, o in enumerate(oofs.values())})

    L = ['# Layer 0 证据来源分解（段级 RF，同主协议）', '',
         '| 变体 | pooled | judged | Δ vs full | 95% CI | P(Δ≤0) |',
         '|---|---:|---:|---:|---|---:|']
    for name in variants:
        r = out[name]
        if 'delta_vs_full' in r:
            d = r['delta_vs_full']
            L.append(f'| {name} | {r["pooled"]} | {r["judged"]} | {d["pooled"]:+.4f} | '
                     f'[{d["ci"][0]:+.4f}, {d["ci"][1]:+.4f}] | {d["p_le0"]*100:.1f}% |')
        else:
            L.append(f'| {name} | {r["pooled"]} | {r["judged"]} | — | — | — |')
    L += ['', '逐段 AUC：', '', '| 变体 | ' + ' | '.join(
        f'part{g}' for g in range(1, 8)) + ' |',
        '|---|' + '---:|' * 7]
    for name in variants:
        pp = out[name]['per_part']
        L.append(f'| {name} | ' + ' | '.join(
            str(pp.get(str(g), '—')) for g in range(1, 8)) + ' |')
    L += ['', '冷启动（part1 行，SELF 恒为 0）：', '']
    for name in variants:
        c = out['cold_start_part1'][name]
        L.append(f'- {name}: n={c["n"]}，正例={c["pos"]}，AUC={c["auc"]}')
    with open(os.path.join(args.out, 'layer0_decomp.md'), 'w') as f:
        f.write('\n'.join(L))
    print(f'\n完成 → {args.out}/layer0_decomp.md', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--out', default='results/layer0_decomp')
    args = ap.parse_args()
    main(args)
