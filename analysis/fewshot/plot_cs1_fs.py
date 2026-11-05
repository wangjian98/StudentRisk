#!/usr/bin/env python3
"""
CS1-FSP 结果出图：少样本学习曲线 + 方法对比 + 稀有子类
用法（246）：python3 analysis/fewshot/plot_cs1_fs.py --json results/cs1_fs/cs1_fs.json \
                --out results/cs1_fs
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for f in ['/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
          '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
          '/usr/share/fonts/truetype/arphic/uming.ttc']:
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=f).get_name()]
        break
plt.rcParams['axes.unicode_minus'] = False

METHOD_LABEL = {
    'proto': '双原型(线性)', 'nn': '1-NN', 'logit': '逻辑回归(线性)',
    'tree2': '浅树d2', 'tree3': '浅树d3', 'rf': 'RF(直接训练)',
    'ls': '图传播', 'ls_rf': '图传播+RF(提案主模型)', 'proto_rf': '原型+RF',
    'metric_proto': '度量学习原型网',
}


def main(args):
    with open(args.json, encoding='utf-8') as f:
        D = json.load(f)
    os.makedirs(args.out, exist_ok=True)
    shots = [1, 2, 4, 8, 16]
    fsets = ['all29', 'peer10', 'peer_submit']

    # --- 图 1：学习曲线（all29）---
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for m in ['proto', 'logit', 'rf', 'ls', 'ls_rf', 'proto_rf']:
        ys = []
        for K in shots:
            r = D['fs_a_inpart_coldstart'].get(f'all29|{m}|K={K}')
            ys.append(r['pooled'] if r else np.nan)
        ax.plot(shots, ys, marker='o', label=METHOD_LABEL.get(m, m))
    for name, key in [('REF-inpart-full (全标签 RF)', 'REF-inpart-full (段内全标签 RF, 少样本同空间)'),
                      ('REF-crosspart (跨段全标签 RF)', 'REF-crosspart (跨段全标签 RF)'),
                      ('REF-global-main (主协议 RF)', 'REF-global-main (主协议原始空间 RF)')]:
        v = D['reference'].get(key, {}).get('pooled')
        if v is not None:
            ax.axhline(v, ls='--', lw=1, color='gray')
            ax.text(shots[-1], v, f' {name} {v}', va='center', fontsize=8, color='gray')
    ax.set_xlabel('每段标注数 K（正例 K + 负例 K）')
    ax.set_ylabel('part 内 pooled AUC')
    ax.set_title('FS-A 段内冷启动学习曲线（all29 特征集）')
    ax.set_xticks(shots)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig_fs_learning_curve.png'), dpi=160)
    plt.close(fig)

    # --- 图 2：K=4 / K=8 方法横向对比（三个特征集）---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, fs in zip(axes, fsets):
        ms = ['proto', 'nn', 'logit', 'tree2', 'tree3', 'rf', 'ls', 'ls_rf', 'proto_rf']
        for K, off in [(4, -0.2), (8, 0.2)]:
            ys = []
            for m in ms:
                r = D['fs_a_inpart_coldstart'].get(f'{fs}|{m}|K={K}')
                ys.append(r['pooled'] if r else np.nan)
            xs = np.arange(len(ms)) + off
            ax.bar(xs, ys, width=0.38, label=f'K={K}')
        ax.set_xticks(np.arange(len(ms)))
        ax.set_xticklabels([METHOD_LABEL[m] for m in ms], rotation=40, ha='right', fontsize=7)
        ax.set_title(fs, fontsize=10)
        ax.grid(alpha=.3, axis='y')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('part 内 pooled AUC')
    fig.suptitle('FS-A 小标签预算下的方法对比（少样本 + 段锚定转导标准化）', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig_fs_method_compare.png'), dpi=160)
    plt.close(fig)

    # --- 图 3：稀有子类（完课仍挂科）---
    if 'fs_e_rare_class' in D:
        rec = D['fs_e_rare_class']
        ms = [m for m in METHOD_LABEL if m != 'metric_proto']
        fig, ax = plt.subplots(figsize=(9, 4.6))
        for K, off in [(2, -0.2), (4, 0.0), (8, 0.2)]:
            ys = [rec.get(f'{m}|K={K}', {}).get('auc_mean', np.nan) for m in ms]
            ax.bar(np.arange(len(ms)) + off, ys, width=0.38, label=f'K={K}')
        ax.axhline(rec.get('ref_full_RF_auc', np.nan), ls='--', color='gray',
                   label=f"全标签 RF {rec.get('ref_full_RF_auc')}")
        ax.set_xticks(np.arange(len(ms)))
        ax.set_xticklabels([METHOD_LABEL[m] for m in ms], rotation=40, ha='right', fontsize=8)
        ax.set_ylabel('学生级 AUC')
        ax.set_title(f"FS-E 稀有子类：走完全程仍挂科（n={rec.get('n')}，正例={rec.get('pos')}）")
        ax.grid(alpha=.3, axis='y')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, 'fig_fs_rare_class.png'), dpi=160)
        plt.close(fig)

    print(f'图已写出 → {args.out}/fig_fs_*.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default='results/cs1_fs/cs1_fs.json')
    ap.add_argument('--out', default='results/cs1_fs')
    main(ap.parse_args())
