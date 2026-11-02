#!/usr/bin/env python3
"""CS2-SPC 结果后处理：配对 bootstrap + 校准 + 干预名单精度 + 汇总表。

用法（246 /home/ubuntu/studentrisk_stage）：
  python3 analysis/hazard_model/analyze_cs2_results.py --dir results/cs2_spc
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    'cs2', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_cs2_spc.py'))
cs2 = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(cs2)


def row_order(data_path, ex_path, K):
    """复现 trunc_agg 的行顺序（插入序）；用于把 RF 的 oof 对齐到排序序。"""
    ds = sc.load_dataset(data_path)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part_ids'])
    ex = np.load(ex_path, allow_pickle=True)
    raw_ex = ex['raw'].astype(np.float64)
    dsp, dpart = ds['sid'], ds['part_ids']
    part_key = {(int(dsp[i]), int(dpart[i])): i for i in range(len(dsp))}
    sittings = cs2.build_sittings(raw_ex, ex['sid'], ex['part'], ex['exercise'])
    rows, _ = cs2.trunc_agg(sittings, part_key, raw_ex, None if K <= 0 else K)
    return rows


def load(d, name):
    p = os.path.join(d, f'oof_{name}.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return z['oof'], z['y'], z['part']


def brier(y, oof):
    return float(np.mean((oof - y) ** 2))


def calibration_bins(y, oof, nb=5):
    q = np.quantile(oof, np.linspace(0, 1, nb + 1))
    q[0] -= 1e-9
    out = []
    for i in range(nb):
        m = (oof > q[i]) & (oof <= q[i + 1])
        if m.sum() == 0:
            continue
        out.append((float(q[i + 1]), int(m.sum()), float(oof[m].mean()), float(y[m].mean())))
    return out


def topk_prec(y, oof, part, frac):
    tot_n, tot_hit, per = 0, 0, []
    for g in np.unique(part):
        m = np.where(part == g)[0]
        if len(m) < 10:
            continue
        k = max(int(round(len(m) * frac)), 1)
        sel = m[np.argsort(-oof[m])[:k]]
        tot_n += len(sel)
        tot_hit += int(y[sel].sum())
        per.append(float(y[sel].mean()))
    return (tot_hit / tot_n if tot_n else float('nan'),
            float(np.mean(per)) if per else float('nan'))


def main(args):
    d = args.dir
    lines = []
    rows = []

    def add(tag, oof, y, part):
        pp = sc.per_part_auc(y, oof, part)
        pooled = sc.pooled(pp)
        p5, p5m = topk_prec(y, oof, part, 0.05)
        p1, p1m = topk_prec(y, oof, part, 0.01)
        lines.append(dict(tag=tag, pooled=float(pooled), auc=float(sc.roc_auc_score(y, oof)),
                          judged=sc.judged(pp), brier=brier(y, oof),
                          top5=p5, top5_perpart=p5m, top1=p1, top1_perpart=p1m,
                          n=int(len(y)), pos=int(y.sum())))
        return lines[-1]

    # ---- 基线 ----
    base = {}
    for k in [1, 2, 3]:
        r = load(d, f'rf_sit{k}')
        if r is not None:
            base[f'rf_sit{k}'] = add(f'RF·坐下第{k}次', *r)
    r = load(d, 'rf_full')
    if r is not None:
        base['rf_full'] = add('RF·整段', *r)

    # ---- CS2 ----
    cs2 = {}
    for K in [1, 2, 0]:
        for c in ['base', 'film', 'peer', 'film_peer']:
            r = load(d, f'cs2_{c}_K{K}')
            if r is not None:
                cs2[(c, K)] = add(f'CS2·{c}·K{K}', *r)

    # ---- 配对 bootstrap ----
    boots = {}
    pairs = []
    for K in [1, 2, 0]:
        bn = f'rf_sit{K}' if K in (1, 2, 3) else 'rf_full'
        if bn not in base:
            continue
        br = load(d, bn)
        for (c, kk), v in cs2.items():
            if kk != K:
                continue
            rc = load(d, f'cs2_{c}_K{K}')
            if rc is None:
                continue
            # 行序对齐：RF 的 oof 是插入序，CS2 的是排序序 → 用 perm 对齐
            rows = row_order(args.data, args.ex, K)
            perm = np.argsort(rows)
            oof_rf = br[0][perm]
            y_rf, part_rf = br[1][perm], br[2][perm]
            if not (np.array_equal(y_rf, rc[1]) and np.array_equal(part_rf, rc[2])):
                continue
            b = sc.paired_boot(y_rf, oof_rf, rc[0], part_rf, n=2000)
            boots[f'{c}_K{K}'] = b
            pairs.append((f'CS2·{c}·K{K}', bn, b))

    # ---- 输出 ----
    print('=' * 96)
    print(f"{'模型':<22}{'pooled':>9}{'Δ vs RF':>10}{'95%CI':>22}{'P(Δ≤0)':>9}"
          f"{'Brier':>8}{'top5%':>8}{'top1%':>8}")
    print('-' * 96)
    for v in lines:
        if v['tag'].startswith('RF'):
            print(f"{v['tag']:<22}{v['pooled']:>9.4f}{'-':>10}{'-':>22}{'-':>9}"
                  f"{v['brier']:>8.3f}{v['top5']:>8.3f}{v['top1']:>8.3f}")
    for name, bn, b in pairs:
        v = [x for x in lines if x['tag'] == name][0]
        basev = base[bn]['pooled']
        print(f"{name:<22}{v['pooled']:>9.4f}{b['obs']:>+10.4f}"
              f"{'[%+.4f,%+.4f]' % (b['lo'], b['hi']):>22}{b['p_le0']:>9.3f}"
              f"{v['brier']:>8.3f}{v['top5']:>8.3f}{v['top1']:>8.3f}")
    print('=' * 96)

    with open(os.path.join(d, 'summary.json'), 'w') as f:
        json.dump(dict(rows=lines, bootstrap=boots), f, indent=2, ensure_ascii=False)

    # markdown 表
    md = ['| 模型 | part 内 pooled | Δ vs RF | 95% CI | P(Δ≤0) | Brier | top5% 精确率 | top1% 精确率 |',
          '|---|---:|---:|---|---:|---:|---:|---:|']
    for v in lines:
        if v['tag'].startswith('RF'):
            md.append(f"| {v['tag']} | {v['pooled']:.4f} | — | — | — | {v['brier']:.3f} | "
                      f"{v['top5']:.3f} | {v['top1']:.3f} |")
    for name, bn, b in pairs:
        v = [x for x in lines if x['tag'] == name][0]
        md.append(f"| {name} | {v['pooled']:.4f} | {b['obs']:+.4f} | "
                  f"[{b['lo']:+.4f}, {b['hi']:+.4f}] | {b['p_le0']:.3f} | {v['brier']:.3f} | "
                  f"{v['top5']:.3f} | {v['top1']:.3f} |")
    out = os.path.join(d, 'summary.md')
    with open(out, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('→', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results/cs2_spc')
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--ex', default='data/exercise_dataset.npz')
    main(ap.parse_args())
