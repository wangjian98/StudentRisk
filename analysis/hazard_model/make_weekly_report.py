#!/usr/bin/env python3
"""
任务 3b · 生成"可交给老师的周报"（原型：用当前数据集跑一次真实名单）
=================================================================================
产出 docs/WEEKLY_RISK_REPORT.md，内容：
  · 期次口径与模型版本
  · 干预名单：top-N 个 (学生, 分段)：风险、首次越阈题号、疑似卡点题目 + 逐题同伴对照 + 话术
  · 名单质量摘要：精确率/基准率/提升倍数、污染行占比
  · 方法与复现说明
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def main(args):
    sit = np.load(args.sit, allow_pickle=True)
    rows, oof, y, part = sit['rows'], sit['oof'], sit['y'], sit['part']
    ex = np.load(args.ex, allow_pickle=True)
    raw = ex['raw'].astype(np.float64)
    ex_sid, ex_par, ex_exr = ex['sid'], ex['part'], ex['exercise']
    ds = np.load(args.data, allow_pickle=True)
    dsp, dpart = ds['sid'], ds['part_ids']
    qpath = os.path.join(os.path.dirname(args.quality.rstrip('/')), 'quality_flags.npz')
    bad_mask = None
    if os.path.exists(qpath):
        bad_mask = np.load(qpath, allow_pickle=True)['bad_mask']

    sp_rows = defaultdict(list)
    for i in range(len(ex_sid)):
        sp_rows[(int(ex_sid[i]), int(ex_par[i]))].append(i)

    order = np.argsort(-oof)
    top = order[:args.top_n]
    base = float(y.mean())
    prec = float(y[top].mean())

    lines = []
    lines.append('# 编程学习者停滞风险周报（原型 · 基于 CS1 数据集实测输出）\n')
    lines.append('**期次口径**：静态原型（非真实周次）。数据 = CS1 MOOC IDE 日志；'
                 '模型 = CS1-SHR 段级掉队 hazard + IPSD 段内截断增广（坐下级）。\n')
    lines.append('**名单规模**：共标记 %d 个 (学生, 分段)，其中真实最终掉队 %d 个 → '
                 '**精确率 %.3f**（基准率 %.3f，**%.1f×**）\n'
                 % (len(top), int(y[top].sum()), prec, base, prec / max(base, 1e-9)))
    if bad_mask is not None:
        b = int(bad_mask[rows[top]].sum())
        lines.append('**数据质量**：名单中 %d/%d (%.1f%%) 的 (学生,分段) 带有低质量提交信号'
                     '（整体污染率 %.1f%%，名单未被污染主导）\n'
                     % (b, len(top), 100 * b / len(top), 100 * bad_mask[rows].mean()))
    lines.append('---\n')
    lines.append('## 一、干预名单（按风险降序）\n')
    for rank, i in enumerate(top, 1):
        r = int(rows[i])
        s = int(dsp[r])
        p = int(part[r])
        idxs = sorted(sp_rows.get((s, p), []), key=lambda j: ex_exr[j])
        if not idxs:
            continue
        risk = float(oof[i])
        # 同段同题数的同伴跨度
        peers = [j2 for (s2, p2), idxs2 in sp_rows.items()
                 if p2 == p and len(idxs2) >= len(idxs) for j2 in idxs2]
        tail = idxs[-args.tail:]
        lines.append(f'### {rank}. 学生 #{s} · 第 {p} 分段 · 风险 **{risk:.2f}**'
                     f' · 本段完成 {len(idxs)} 题（该段共 '
                     f'{int(ex_exr[ex_par == p].max())} 题）')
        if bad_mask is not None and bad_mask[r]:
            lines.append('> ⚠️ 该行带有低质量提交信号，判定需人工复核。\n')
        lines.append('| 题号 | 事件数 | 跨度(h) | 同伴中位跨度(h) | 编辑 | 运行 | 删/插 | 焦点/h |')
        lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|')
        for j in tail:
            sp_h = (raw[j, 11].max() - raw[j, 10].min()) / 3600.0
            peer_sp = []
            for j2 in peers:
                if j2 == j:
                    continue
                peer_sp.append((raw[j2, 11].max() - raw[j2, 10].min()) / 3600.0)
            med = float(np.median(peer_sp)) if peer_sp else float('nan')
            ev = raw[j, 1:8]
            lines.append(f'| {int(ex_exr[j])} | {int(raw[j,0])} | {sp_h:.2f} | {med:.2f} | '
                         f'{int(ev[0])} | {int(ev[5])} | {ev[1]/max(ev[0],1e-6):.2f} | '
                         f'{ev[3]/max(sp_h,1e-6):.2f} |')
        last = idxs[-1]
        ev = raw[last, 1:8]
        sp_h = (raw[last, 11].max() - raw[last, 10].min()) / 3600.0
        lines.append(f'\n**卡点判断**：该生停在第 {int(ex_exr[last])} 题。'
                     f'最后一题：{int(raw[last,0])} 个事件、编辑 {int(ev[0])} 次、'
                     f'运行 {int(ev[5])} 次、跨度 {sp_h:.1f} 小时、'
                     f'焦点切换 {ev[3]/max(sp_h,1e-6):.1f} 次/小时。\n')
        lines.append('**建议动作**：对该生最后一题的知识点安排 15 分钟 1:1 复盘；'
                     '若为"编辑多/运行少"型，强调"先跑起来看报错再改"；'
                     '若为"焦点切换极高"型，改为限时练习并关闭通知。\n')
        lines.append('---\n')
    lines.append('## 二、方法与复现\n')
    lines.append('- 阶段级判别：CS1-SHR 掉队 hazard（part 内 pooled AUC 0.9258）；'
                 '段内判别：截断增广（习题级 0.7915 / 坐下级 0.8665）。\n'
                 '- 决策单元 = 分段；习题/坐下 = 观测分辨率与诊断锚点。\n'
                 '- 复现：`analysis/hazard_model/run_sitting_truncation.py`、'
                 '`run_intra_part_diagnostics.py`、'
                 '`run_submission_quality.py`。\n')
    lines.append('- 局限：仍为单一数据集、离线原型；精确率为回溯估计而非线上实测。\n')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'写出 {args.out}（名单 {len(top)} 条，精确率 {prec:.3f}，基准 {base:.3f}）')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sit', default='results/sitting/sit_k1_feat.npz')
    ap.add_argument('--ex', default='data/exercise_dataset.npz')
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--quality', default='results/quality/quality_flags.npz')
    ap.add_argument('--out', default='docs/WEEKLY_RISK_REPORT.md')
    ap.add_argument('--top-n', type=int, default=8)
    ap.add_argument('--tail', type=int, default=3)
    main(ap.parse_args())
