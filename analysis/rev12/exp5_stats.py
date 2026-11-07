"""
rev12.exp5_stats — 实验 5：显著性检验（审稿意见 调整4）

硬伤：MetaMamba vs RF-7d 只差 2.7pp，全文没有 p 值 —— 这是最脆弱的一处对比。

本模块提供三种检验（全部基于**同一折划分**的配对样本）：
  1. **配对 Wilcoxon 符号秩检验**：15 个外层折（5 折 × 3 seeds）逐折配对，
     比较 macro-F1 / ROC-AUC / balanced accuracy；族内用 Holm-Bonferroni 校正。
  2. **McNemar 精确检验**：在合并 OOF 预测（473 名学生，3 seed 平均概率，阈值 0.5）上，
     比较两个模型的对错配对结构。
  3. **bootstrap 置信区间**（重复级配对 bootstrap 2000 次）作为 p 值的补充。

输出：results/rev12/exp5_stats/significance.json
"""
import os
import sys
import json
import glob
import argparse
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.rev12.common import load_cache, save_json
from analysis.rev12.models import set_seed

OUT_ROOT = os.path.join(_ROOT, 'results', 'rev12')


def wilcoxon_paired(a: np.ndarray, b: np.ndarray):
    from scipy import stats
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    if np.allclose(diff, 0):
        return 1.0, 0.0, float('nan')
    try:
        res = stats.wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        stat, p = float(res.statistic), float(res.pvalue)
    except ValueError:
        stat, p = float('nan'), 1.0
    # 秩相关
    try:
        rho = float(stats.spearmanr(a, b).statistic)
    except Exception:
        rho = float('nan')
    return stat, p, rho


def mcnemar_exact(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray):
    """McNemar 精确检验：b = a 错 b 对；c = a 对 b 错"""
    from scipy import stats
    b = int(((pred_a != y) & (pred_b == y)).sum())
    c = int(((pred_a == y) & (pred_b != y)).sum())
    n = b + c
    if n == 0:
        return {'b': b, 'c': c, 'p': 1.0}
    p = float(stats.binomtest(c, n, 0.5).pvalue)
    return {'b': b, 'c': c, 'n_discordant': n, 'p': p}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot=2000, seed=0):
    """重复级配对 bootstrap：返回 Δ均值与 95% CI、P(Δ≤0)"""
    rng = np.random.RandomState(seed)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    n = len(a)
    idx = rng.randint(0, n, size=(n_boot, n))
    d = a[idx] - b[idx]
    means = d.mean(axis=1)
    return {
        'delta_mean': float((a - b).mean()),
        'ci95': [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        'p_le_zero': float((means <= 0).mean() * 100),
    }


def holm_bonferroni(pvals: dict):
    """Holm-Bonferroni：返回 {key: (p_raw, p_holm)}"""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        adj = max(adj, prev)      # 保证单调
        prev = adj
        out[k] = (float(p), float(adj))
    return out


def load_cells(base_dir):
    cells = {}
    for path in sorted(glob.glob(os.path.join(base_dir, '*'))):
        if not os.path.isdir(path):
            continue
        fm = os.path.join(path, 'fold_metrics.csv')
        rs = os.path.join(path, 'results.json')
        oof = os.path.join(path, 'oof_probs.npy')
        if not (os.path.exists(fm) and os.path.exists(rs)):
            continue
        df = pd.read_csv(fm)
        with open(rs) as f:
            res = json.load(f)
        cell = {'name': os.path.basename(path), 'fold_metrics': df, 'results': res}
        if os.path.exists(oof):
            cell['oof'] = np.load(oof)
        cells[os.path.basename(path)] = cell
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default=os.path.join(OUT_ROOT, 'exp1_matrix'))
    ap.add_argument('--out', default=os.path.join(OUT_ROOT, 'exp5_stats'))
    args = ap.parse_args()

    cache = load_cache()
    y = cache['y']
    cells = load_cells(args.base)
    print(f"[exp5] loaded cells: {list(cells.keys())}", flush=True)

    metrics = ['macro_f1', 'roc_auc', 'balanced_accuracy', 'accuracy']
    report = {'reference': 'metamamba', 'pairs': [], 'generated': pd.Timestamp.now().isoformat()}

    # 按输入空间分组比较
    by_dim = {}
    for name, c in cells.items():
        dim = int(name.split('_d')[-1])
        by_dim.setdefault(dim, []).append(name)

    for dim, names in sorted(by_dim.items()):
        ref_name = f'metamamba_d{dim}'
        if ref_name not in cells:
            continue
        ref = cells[ref_name]
        ref_fm = ref['fold_metrics']
        pvals = {m: {} for m in metrics}
        for name in names:
            if name == ref_name:
                continue
            fm = cells[name]['fold_metrics']
            merged = ref_fm.merge(fm, on=['seed', 'fold'], suffixes=('_ref', '_cmp'))
            entry = {'dim': dim, 'reference': ref_name, 'comparison': name,
                     'n_pairs': int(len(merged))}
            for m in metrics:
                a = merged[f'{m}_ref'].values
                b = merged[f'{m}_cmp'].values
                stat, p, rho = wilcoxon_paired(a, b)
                bs = paired_bootstrap(a, b, n_boot=2000, seed=7)
                entry[m] = {
                    'ref_mean': float(np.mean(a)), 'cmp_mean': float(np.mean(b)),
                    'delta': float(np.mean(a) - np.mean(b)),
                    'wilcoxon_stat': stat, 'p_raw': p, 'spearman_rho': rho, **bs,
                }
                pvals[m][name] = p
            # McNemar（合并 OOF）
            if 'oof' in ref and 'oof' in cells[name]:
                pr = (ref['oof'] >= 0.5).astype(int)
                pc = (cells[name]['oof'] >= 0.5).astype(int)
                entry['mcnemar'] = mcnemar_exact(y, pr, pc)
                entry['n_students'] = int(len(y))
            report['pairs'].append(entry)

        # Holm-Bonferroni 校正（族 = 同 dim 同指标）
        for m in metrics:
            if not pvals[m]:
                continue
            adj = holm_bonferroni(pvals[m])
            for e in report['pairs']:
                if e['dim'] == dim and e['comparison'] in adj:
                    e[m]['p_holm'] = adj[e['comparison']][1]
                    e[m]['significant_holm_005'] = bool(adj[e['comparison']][1] < 0.05)

    os.makedirs(args.out, exist_ok=True)
    save_json(os.path.join(args.out, 'significance.json'), report)

    # markdown 摘要
    lines = ['# 显著性检验结果（Wilcoxon 配对 + McNemar + bootstrap）', '',
             '参考模型：MetaMamba（同输入空间）；配对样本 = 15 个外层折（5 折 × 3 seeds）。', '',
             '| 输入空间 | 对比 | Δmacro-F1 | Wilcoxon p | Holm p | McNemar p | ΔROC-AUC | Δbal-acc |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for e in report['pairs']:
        lines.append(
            f"| {e['dim']}d | {e['comparison']} | {e['macro_f1']['delta']:+.4f} | "
            f"{e['macro_f1']['p_raw']:.4g} | {e['macro_f1'].get('p_holm', float('nan')):.4g} | "
            f"{e.get('mcnemar', {}).get('p', float('nan')):.4g} | "
            f"{e['roc_auc']['delta']:+.4f} | {e['balanced_accuracy']['delta']:+.4f} |")
    with open(os.path.join(args.out, 'significance.md'), 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines), flush=True)
    print("[exp5] done", flush=True)


if __name__ == '__main__':
    main()
