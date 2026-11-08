"""
regimemamba.routing — 把 RegimeMamba 纳入「按可观测量路由」的融合实测

复现并扩展 analysis/fusion/nested_fusion.py：
  对每个前缀（10%/20%/40%/100%），用**嵌套交叉拟合**的方式在 RF / MetaMamba / RegimeMamba
  之间选权重（每折在其余 4 折的 OOF 上选），再应用到本折，得到可对外引用的融合 AUC。

输出：docs/REGIMEMAMBA_ROUTING_20261108.csv
"""
import os
import sys
import json
import numpy as np
import csv

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

REG = os.path.join(_ROOT, 'results', 'regimemamba')
REV = os.path.join(_ROOT, 'results', 'rev12')
OUT_CSV = os.path.join(_ROOT, 'docs', 'REGIMEMAMBA_ROUTING_20261108.csv')

Z = np.load('/home/ubuntu/SR12_cache/events.npz')
y = Z['y'].astype(int)


def rank(p):
    return (np.argsort(np.argsort(p)) + 1.0) / len(p)


def auc_fast(p, yy=None):
    yy = y if yy is None else yy
    r = np.argsort(np.argsort(p)) + 1.0
    npos = float(yy.sum()); nneg = float(len(yy) - npos)
    return float((r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def boot_delta(pa, pb, n=2000, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); ds = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        ds.append(auc_fast(pa[s], y[s]) - auc_fast(pb[s], y[s]))
    ds = np.array(ds)
    return float(ds.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float((ds <= 0).mean())


def folds_of(seed, k=5):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    out = {}
    for f, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        for i in va:
            out[i] = f
    return out


def load(path):
    return np.load(path).astype(np.float64) if os.path.exists(path) else None


def _weight_grid(n, ws):
    """生成 n 个非负权重（和为 1）的候选网格"""
    if n == 2:
        for w in ws:
            yield (w, 1.0 - w)
    elif n == 3:
        for w in ws:
            for w2 in ws:
                w3 = 1.0 - w - w2
                if w3 < -1e-9:
                    continue
                yield (w, w2, w3)
    else:
        raise ValueError(f'crossfit 仅支持 2 或 3 个模型，收到 {n}')


def crossfit(preds, names, ws=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    """preds: dict name→向量；names: 参与融合的模型名（秩空间线性组合，权重和为1）

    支持 2 个模型（RF+RegimeMamba）与 3 个模型（RF+MetaMamba+RegimeMamba）两种情形。
    """
    R = {k: rank(v) for k, v in preds.items()}
    n = len(names)
    grid = list(_weight_grid(n, ws))
    out = np.zeros(len(y)); chosen = []
    for seed in (42, 123, 777):
        fmap = folds_of(seed); fids = np.array([fmap[i] for i in range(len(y))])
        for f in range(5):
            tr = np.where(fids != f)[0]; va = np.where(fids == f)[0]
            best = (-1, None)
            for wts in grid:
                p = sum(wt * R[nm] for wt, nm in zip(wts, names))
                a = auc_fast(p[tr], y[tr])
                if a > best[0]:
                    best = (a, wts)
            w = best[1]
            if w is None:
                raise RuntimeError(f'未能选出融合权重 (n={n})')
            chosen.append(w)
            out[va] = sum(wt * R[nm] for wt, nm in zip(w, names))[va]
    return out, chosen


def main():
    rows = []
    for frac, regm_tag in [('p0.1', 'main_d28_p0.1'), ('p0.2', 'main_d28_p0.2'),
                           ('p0.4', None), ('p1.0', 'main_d28_p1.0')]:
        pr = load(os.path.join(REV, 'exp2_prefix', f'rf_d28_{frac}', 'oof_probs.npy'))
        pm = load(os.path.join(REV, 'exp2_prefix', f'metamamba_d28_{frac}', 'oof_probs.npy'))
        prg = load(os.path.join(REG, regm_tag, 'oof_probs.npy')) if regm_tag else None
        if pr is None or pm is None:
            continue
        print(f'\n=== 前缀 {frac} ===')
        print(f'  RF-28d        AUC={auc_fast(pr):.4f}')
        print(f'  MetaMamba-28d AUC={auc_fast(pm):.4f}')
        if prg is not None:
            print(f'  RegimeMamba   AUC={auc_fast(prg):.4f}')
            # 两模型融合（RF + RegimeMamba）
            two, ch2 = crossfit({'rf': pr, 'rm': prg}, ['rf', 'rm'])
            d, lo, hi, pn = boot_delta(two, pr)
            print(f'  交叉拟合融合(RF+RegimeMamba) AUC={auc_fast(two):.4f} '
                  f'Δ vs RF={d:+.4f} [{lo:+.4f},{hi:+.4f}] P={pn:.3f}')
            w_rf = float(np.mean([c[0] for c in ch2]))
            print(f'    选中 w_RF 均值={w_rf:.2f} 中位={np.median([c[0] for c in ch2]):.2f}')
            rows.append({'prefix': frac, 'model': 'RF-28d', 'auc': auc_fast(pr)})
            rows.append({'prefix': frac, 'model': 'MetaMamba-28d', 'auc': auc_fast(pm)})
            rows.append({'prefix': frac, 'model': 'RegimeMamba', 'auc': auc_fast(prg)})
            rows.append({'prefix': frac, 'model': 'FUSE(RF,RegimeMamba)',
                         'auc': auc_fast(two), 'delta_vs_rf': d, 'ci': f'[{lo:+.4f},{hi:+.4f}]',
                         'p': pn, 'w_rf': w_rf})
            # 三模型融合
            three, ch3 = crossfit({'rf': pr, 'mm': pm, 'rm': prg}, ['rf', 'mm', 'rm'])
            d3, lo3, hi3, pn3 = boot_delta(three, pr)
            print(f'  交叉拟合融合(RF+MetaMamba+RegimeMamba) AUC={auc_fast(three):.4f} '
                  f'Δ vs RF={d3:+.4f} [{lo3:+.4f},{hi3:+.4f}] P={pn3:.3f}')
            rows.append({'prefix': frac, 'model': 'FUSE(RF,MetaMamba,RegimeMamba)',
                         'auc': auc_fast(three), 'delta_vs_rf': d3, 'ci': f'[{lo3:+.4f},{hi3:+.4f}]',
                         'p': pn3,
                         'w_rf': float(np.mean([c[0] for c in ch3])),
                         'w_mm': float(np.mean([c[1] for c in ch3])),
                         'w_rm': float(np.mean([c[2] for c in ch3]))})
        else:
            rows.append({'prefix': frac, 'model': 'RF-28d', 'auc': auc_fast(pr)})
            rows.append({'prefix': frac, 'model': 'MetaMamba-28d', 'auc': auc_fast(pm)})

    if rows:
        keys = ['prefix', 'model', 'auc', 'delta_vs_rf', 'ci', 'p', 'w_rf', 'w_mm', 'w_rm']
        with open(OUT_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'\n[routing] 写出 {OUT_CSV}')
    print('[routing] DONE')


if __name__ == '__main__':
    main()
