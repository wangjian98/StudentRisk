#!/usr/bin/env python3
"""CS2-FWD 结果后处理：配对 bootstrap + 汇总表。"""
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
    print('=' * 92)
    print(f"{'模型':<26}{'pooled':>9}{'Δ vs RF':>10}{'95%CI':>22}{'P(Δ≤0)':>9}"
          f"{'top5%':>8}{'top1%':>8}{'n':>7}")
    print('-' * 92)
    md = ['| 模型 | part 内 pooled | Δ vs RF(flat) | 95% CI | P(Δ≤0) | top5% 精确率 | top1% 精确率 | n |',
          '|---|---:|---:|---|---:|---:|---:|---:|']
    out = {}
    for K in args.k_list:
        base = load(d, f'rf_K{K}')
        if base is None:
            continue
        bp = sc.pooled(sc.per_part_auc(base[1], base[0], base[2]))
        p5, _ = topk_prec(base[1], base[0], base[2], 0.05)
        p1, _ = topk_prec(base[1], base[0], base[2], 0.01)
        print(f"{'RF(flat 31维)·K' + str(K):<26}{bp:>9.4f}{'-':>10}{'-':>22}{'-':>9}"
              f"{p5:>8.3f}{p1:>8.3f}{len(base[1]):>7}")
        md.append(f"| RF(flat 31维)·观察前 {K if K>0 else '全部'} 题 | {bp:.4f} | — | — | — | "
                  f"{p5:.3f} | {p1:.3f} | {len(base[1])} |")
        for c in ['base', 'peer']:
            r = load(d, f'cs2_{c}_K{K}')
            if r is None:
                continue
            pp = sc.pooled(sc.per_part_auc(r[1], r[0], r[2]))
            rows = row_order(args.data, args.ex, K)
            perm = np.argsort(rows)
            oof_rf = base[0][perm]
            y_rf, part_rf = base[1][perm], base[2][perm]
            if not (np.array_equal(y_rf, r[1]) and np.array_equal(part_rf, r[2])):
                print(f'{name}: 行序未对齐，跳过配对 bootstrap')
                continue
            b = sc.paired_boot(y_rf, oof_rf, r[0], part_rf, n=2000)
            p5c, _ = topk_prec(r[1], r[0], r[2], 0.05)
            p1c, _ = topk_prec(r[1], r[0], r[2], 0.01)
            print(f"{'CS2-FWD·' + c + '·K' + str(K):<26}{pp:>9.4f}{b['obs']:>+10.4f}"
                  f"{'[%+.4f,%+.4f]' % (b['lo'], b['hi']):>22}{b['p_le0']:>9.3f}"
                  f"{p5c:>8.3f}{p1c:>8.3f}{len(r[1]):>7}")
            md.append(f"| CS2-FWD·{c}·观察前 {K if K>0 else '全部'} 题 | {pp:.4f} | "
                      f"{b['obs']:+.4f} | [{b['lo']:+.4f}, {b['hi']:+.4f}] | {b['p_le0']:.3f} | "
                      f"{p5c:.3f} | {p1c:.3f} | {len(r[1])} |")
            out[f'cs2_{c}_K{K}'] = dict(pooled=float(pp), delta=float(b['obs']),
                                        lo=float(b['lo']), hi=float(b['hi']),
                                        p_le0=float(b['p_le0']), top5=float(p5c))
    print('=' * 92)
    with open(os.path.join(d, 'summary.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')
    with open(os.path.join(d, 'summary.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('→', os.path.join(d, 'summary.md'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='results/cs2_fwd')
    ap.add_argument('--k_list', type=int, nargs='+', default=[1, 3, 10, 0])
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--ex', default='data/exercise_dataset.npz')
    main(ap.parse_args())
