"""
rev12.aggregate — 汇总所有修订实验 → 表格 + 图 + Markdown 报告

用法（在 246 上跑）：
  python3 -m analysis.rev12.aggregate --out docs/REVISION_EXPERIMENTS_20261107.md
输出：
  docs/REVISION_EXPERIMENTS_20261107.md
  docs/figures/rev12_prefix_curve.png
  docs/figures/rev12_film_gamma_beta.png
  docs/figures/rev12_matrix_heatmap.png
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体：优先系统内 CJK 字体
_CJK_CANDIDATES = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans CJK JP',
                   'Source Han Sans CN', 'Microsoft YaHei', 'SimHei']
try:
    for _f in font_manager.findSystemFonts():
        if any(k in _f.lower() for k in ['wqy', 'cjk', 'noto', 'wqy-zenhei', 'han']):
            try:
                font_manager.fontManager.addfont(_f)
            except Exception:
                pass
    _avail = {f.name for f in font_manager.fontManager.ttflist}
    _pick = [c for c in _CJK_CANDIDATES if c in _avail]
    plt.rcParams['font.sans-serif'] = (_pick + ['DejaVu Sans']) if _pick else ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print(f"[aggregate] CJK fonts: {_pick if _pick else 'NONE (labels will be tofu)'}", flush=True)
except Exception as e:
    print(f"[aggregate] font setup failed: {e}", flush=True)

RES = os.path.join(_ROOT, 'results', 'rev12')
FIG_DIR = os.path.join(_ROOT, 'docs', 'figures')

ARCH_LABEL = {
    'rf': 'RF (表格孪生)', 'rf7_all': 'RF-7d (全量事件, 遗留口径)',
    'lstm': 'LSTM', 'bilstm': 'BiLSTM', 'attention': 'Attention',
    'metamamba': 'MetaMamba', 'hmm_cat': 'HMM-Multinomial',
    'hmm_gmm': 'HMM-GMM(自实现)',
}


def load_cells(base):
    cells = {}
    for path in sorted(glob.glob(os.path.join(base, '*'))):
        rs = os.path.join(path, 'results.json')
        if not os.path.exists(rs):
            continue
        with open(rs) as f:
            res = json.load(f)
        fm = pd.read_csv(os.path.join(path, 'fold_metrics.csv'))
        cells[os.path.basename(path)] = {'res': res, 'fm': fm}
    return cells


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return '—'
    if p < 0.0001:
        return '<1e-4'
    return f'{p:.4f}'


def matrix_table(cells, title):
    rows = []
    for name, c in cells.items():
        r = c['res']
        s = r['summary']['per_fold_mean']
        rows.append({
            'cell': name, 'arch': r['arch'], 'dim': r['dim'],
            'acc': s['accuracy'], 'bal': s['balanced_accuracy'],
            'macro_f1': s['macro_f1'], 'macro_f1_std': s['macro_f1_std'],
            'f1_1': s['f1_class_1'], 'auc': s['roc_auc'], 'auc_std': s['roc_auc_std'],
            'folds': r['n_folds'],
        })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df, ''
    df = df.sort_values(['dim', 'arch'])
    lines = [f'**{title}**', '',
             '| 输入空间 | 模型 | acc | **bal-acc** | macro-F1 | macro-F1 std | F1(FAIL) | ROC-AUC | AUC std |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for _, r in df.iterrows():
        lines.append(f"| {r['dim']}d | {ARCH_LABEL.get(r['arch'], r['arch'])} | "
                     f"{r['acc']:.4f} | {r['bal']:.4f} | {r['macro_f1']:.4f} | "
                     f"{r['macro_f1_std']:.4f} | {r['f1_1']:.4f} | {r['auc']:.4f} | {r['auc_std']:.4f} |")
    return df, '\n'.join(lines)


def prefix_table(cells):
    rows = []
    for name, c in cells.items():
        r = c['res']
        s = r['summary']['per_fold_mean']
        rows.append({'cell': name, 'arch': r['arch'], 'dim': r['dim'],
                     'prefix': r['prefix_frac'], 'acc': s['accuracy'],
                     'bal': s['balanced_accuracy'], 'macro_f1': s['macro_f1'],
                     'auc': s['roc_auc']})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df, ''
    df = df.sort_values(['cell', 'prefix'])
    lines = ['| 模型 | 输入 | 前缀比例 | acc | bal-acc | macro-F1 | ROC-AUC |',
             '|---|---|---:|---:|---:|---:|---:|']
    for _, r in df.iterrows():
        lines.append(f"| {ARCH_LABEL.get(r['arch'], r['arch'])} | {r['dim']}d | "
                     f"{int(r['prefix']*100)}% | {r['acc']:.4f} | {r['bal']:.4f} | "
                     f"{r['macro_f1']:.4f} | {r['auc']:.4f} |")
    return df, '\n'.join(lines)


def film_table(base):
    rows = []
    for path in sorted(glob.glob(os.path.join(base, '*'))):
        rs = os.path.join(path, 'results.json')
        if not os.path.exists(rs):
            continue
        with open(rs) as f:
            r = json.load(f)
        s = r['summary']['per_fold_mean']
        rows.append({'dim': r['dim'], 'film': 'on' if r['use_film'] else 'off',
                     'acc': s['accuracy'], 'bal': s['balanced_accuracy'],
                     'macro_f1': s['macro_f1'], 'macro_f1_std': s['macro_f1_std'],
                     'auc': s['roc_auc'], 'folds': r['n_folds']})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df, ''
    df = df.sort_values(['dim', 'film'])
    lines = ['| 输入空间 | FiLM | acc | bal-acc | macro-F1 | macro-F1 std | ROC-AUC | 折数 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    deltas = {}
    for dim in sorted(df['dim'].unique()):
        sub = df[df['dim'] == dim]
        on = sub[sub['film'] == 'on']
        off = sub[sub['film'] == 'off']
        d_auc = d_f1 = None
        if len(on) and len(off):
            d_auc = float(on['auc'].values[0] - off['auc'].values[0])
            d_f1 = float(on['macro_f1'].values[0] - off['macro_f1'].values[0])
            deltas[dim] = (d_auc, d_f1)
        for _, r in sub.iterrows():
            lines.append(f"| {dim}d | {'on' if r['film']=='on' else 'off'} | {r['acc']:.4f} | "
                         f"{r['bal']:.4f} | {r['macro_f1']:.4f} | {r['macro_f1_std']:.4f} | "
                         f"{r['auc']:.4f} | {int(r['folds'])} |")
    if deltas:
        lines += ['', '**去 FiLM 的代价（Δ = on − off）**', '',
                  '| 输入空间 | ΔROC-AUC | Δmacro-F1 |', '|---|---:|---:|']
        for dim in sorted(deltas):
            lines.append(f"| {dim}d | {deltas[dim][0]:+.4f} | {deltas[dim][1]:+.4f} |")
    return df, '\n'.join(lines)


def lopo_table(base):
    rows = []
    for path in sorted(glob.glob(os.path.join(base, '*'))):
        rs = os.path.join(path, 'results.json')
        if not os.path.exists(rs):
            continue
        with open(rs) as f:
            r = json.load(f)
        for vname, v in r['summary'].items():
            rows.append({'cell': r['cell'], 'arch': r['arch'], 'dim': r['dim'],
                         'variant': vname, 'auc_w': v['auc_weighted'],
                         'acc_w': v['acc_weighted'], 'bal_w': v['bal_weighted'],
                         'auc_task_mean': v['auc_mean_over_tasks']})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df, ''
    lines = ['| 模型 | 变体 | LOPO 加权 AUC | LOPO 加权 acc | LOPO 加权 bal-acc | 各 task 平均 AUC |',
             '|---|---|---:|---:|---:|---:|']
    for _, r in df.iterrows():
        lines.append(f"| {ARCH_LABEL.get(r['arch'], r['arch'])}-{r['dim']}d | {r['variant']} | "
                     f"{r['auc_w']:.4f} | {r['acc_w']:.4f} | {r['bal_w']:.4f} | "
                     f"{r['auc_task_mean']:.4f} |")
    return df, '\n'.join(lines)


def plot_prefix(df, out_png):
    if len(df) == 0:
        return None
    plt.figure(figsize=(7.2, 4.6))
    for cell, sub in df.groupby('cell'):
        sub = sub.sort_values('prefix')
        label = f"{ARCH_LABEL.get(sub['arch'].values[0], sub['arch'].values[0])}-{sub['dim'].values[0]}d"
        plt.plot(sub['prefix'] * 100, sub['auc'], marker='o', label=label)
    plt.xlabel('可用事件前缀比例（%）')
    plt.ylabel('ROC-AUC（15 折均值）')
    plt.title('早期预警曲线：前缀比例 vs 学生级判别力')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return out_png


def plot_matrix(df, out_png, title):
    if len(df) == 0:
        return None
    piv = df.pivot_table(index='arch', columns='dim', values='auc')
    piv = piv.reindex([a for a in ['rf7_all', 'rf', 'lstm', 'bilstm', 'attention',
                                   'metamamba', 'hmm_cat', 'hmm_gmm'] if a in piv.index])
    piv = piv.rename(index=ARCH_LABEL)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    im = ax.imshow(piv.values.astype(float), cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f'{c}d' for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        color='white' if v < np.nanmax(piv.values) * 0.6 else 'black', fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='ROC-AUC')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return out_png


def plot_film_gamma(base, out_png):
    """FiLM γ/β 可视化：各输入空间下按 task 展开的 γ 均值/β 均值 + γ 离散度"""
    files = sorted(glob.glob(os.path.join(base, '*_filmon', 'film_params.npz')))
    if not files:
        return None
    fig, axes = plt.subplots(1, len(files), figsize=(4.2 * len(files), 3.6))
    if len(files) == 1:
        axes = [axes]
    for ax, fp in zip(axes, files):
        z = np.load(fp)
        gamma = z['gamma']          # (n_folds, n_tasks, d_model)
        beta = z['beta']
        g_mean = gamma.mean(axis=2)      # (n_folds, n_tasks)
        b_mean = beta.mean(axis=2)
        tasks = np.arange(g_mean.shape[1])
        ax.bar(tasks - 0.2, g_mean.mean(axis=0), width=0.4, label='γ (mean over d_model)')
        ax.bar(tasks + 0.2, b_mean.mean(axis=0), width=0.4, label='β (mean over d_model)')
        ax.set_title(f"{os.path.basename(os.path.dirname(fp))}  "
                     f"γ 跨 task 离散度={g_mean.std(axis=1).mean():.4f}")
        ax.set_xlabel('task (part id)')
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(_ROOT, 'docs', 'REVISION_EXPERIMENTS_20261107.md'))
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)

    ident = load_cells(os.path.join(RES, 'exp1_matrix'))
    deconf = load_cells(os.path.join(RES, 'exp1_matrix_deconf'))
    pref = load_cells(os.path.join(RES, 'exp2_prefix'))
    prefd = load_cells(os.path.join(RES, 'exp2_prefix_deconf'))

    df_ident, t_ident = matrix_table(ident, '表 A：模型 × 输入空间（保留 part/exercise 身份通道，与论文 v2.5 学生级协议一致）')
    df_deconf, t_deconf = matrix_table(deconf, '表 B：模型 × 输入空间（去掉 part/exercise 身份通道 = 去混淆臂）')
    df_pref, t_pref = prefix_table(pref)
    df_prefd, t_prefd = prefix_table(prefd)
    df_film, t_film = film_table(os.path.join(RES, 'exp3_film'))
    df_lopo, t_lopo = lopo_table(os.path.join(RES, 'exp4_lopo'))

    sig_path = os.path.join(RES, 'exp5_stats', 'significance.json')
    t_sig = ''
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            sig = json.load(f)
        lines = ['| 输入空间 | 对比 | Δmacro-F1 | Wilcoxon p | Holm p | McNemar p | ΔROC-AUC | Δbal-acc |',
                 '|---|---|---:|---:|---:|---:|---:|---:|']
        for e in sig['pairs']:
            lines.append(
                f"| {e['dim']}d | {e['comparison']} | {e['macro_f1']['delta']:+.4f} | "
                f"{fmt_p(e['macro_f1']['p_raw'])} | {fmt_p(e['macro_f1'].get('p_holm'))} | "
                f"{fmt_p(e.get('mcnemar', {}).get('p'))} | {e['roc_auc']['delta']:+.4f} | "
                f"{e['balanced_accuracy']['delta']:+.4f} |")
        t_sig = '\n'.join(lines)

    figs = []
    p = plot_prefix(df_pref, os.path.join(FIG_DIR, 'rev12_prefix_curve.png'))
    if p:
        figs.append(p)
    p = plot_prefix(df_prefd, os.path.join(FIG_DIR, 'rev12_prefix_curve_deconf.png'))
    if p:
        figs.append(p)
    p = plot_matrix(df_ident, os.path.join(FIG_DIR, 'rev12_matrix_heatmap.png'),
                    '模型 × 输入空间（保留身份通道）')
    if p:
        figs.append(p)
    p = plot_matrix(df_deconf, os.path.join(FIG_DIR, 'rev12_matrix_heatmap_deconf.png'),
                    '模型 × 输入空间（去混淆）')
    if p:
        figs.append(p)
    p = plot_film_gamma(os.path.join(RES, 'exp3_film'),
                        os.path.join(FIG_DIR, 'rev12_film_gamma_beta.png'))
    if p:
        figs.append(p)

    parts = ['# 论文 v2.5 对比设计修订实验结果（rev12）', '',
             f'生成时间：{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', '',
             '本文件由 `analysis/rev12/aggregate.py` 自动汇总；原始结果见 `results/rev12/`。', '']
    for title, table in [('## 1. 模型 × 输入空间矩阵（同协议重跑）', t_ident),
                         ('## 2. 去混淆矩阵（去掉 part / exercise 身份通道）', t_deconf),
                         ('## 3. 早期预警前缀曲线（保留身份通道）', t_pref),
                         ('## 4. 早期预警前缀曲线（去混淆）', t_prefd),
                         ('## 5. FiLM 消融（多 seed）', t_film),
                         ('## 6. LOPO 跨任务迁移', t_lopo),
                         ('## 7. 显著性检验', t_sig)]:
        if table:
            parts += [title, '', table, '']
    if figs:
        parts += ['## 8. 图', '']
        for f in figs:
            parts += [f'![{os.path.basename(f)}](figures/{os.path.basename(f)})', '']

    with open(args.out, 'w') as f:
        f.write('\n'.join(parts))
    print(f"[aggregate] wrote {args.out} ({len(parts)} lines), figures={len(figs)}")


if __name__ == '__main__':
    main()
