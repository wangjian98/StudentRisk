#!/usr/bin/env python3
"""
截止压力特征族（deadline-pressure features）——全新特征族，且天然可解释
=================================================================================
动机：
  · IDE_logs.csv 里有一列 **timeToDeadline**（距作业截止的秒数，随时间单调递减），
    目前我们的阶段级特征（rf29）**完全没用它**（只作 MetaMamba 11 维里的一个通道）。
  · 理论上"截止压力"是 CS1 里最经典、最可解释的风险信号：赶在截止前最后一刻
    提交 = 高压冲刺；离截止还很久就停手 = 提前放弃（这正是掉队的定义性行为）。
  · 而且 timeToDeadline 单调递减 → "最早/最晚事件"可直接用 ttd 的 max/min 定位，
    不需要时间戳排序，聚合极快。

特征（每个 (学生, part)）：
  ttd_start   : 该段第一次事件的 ttd（可用窗口大小）
  ttd_end     : 该段最后一次事件的 ttd（结束时还剩多少时间）
  utilization : 1 − ttd_end/ttd_start（消耗掉的可用时间比例）
  submit_slack: 最后一次 submit 事件时距截止的秒数（负值=已超时）
  submit_ratio: submit_slack / ttd_start（0=压哨提交，1=刚开工就交）
  run_slack   : 最后一次 run 时的 ttd
  rate_per_h  : 每可用小时的事件数（进度压力）
  rush_ratio  : 落在最后 10% 时间窗内的事件占比（冲刺强度）
  no_submit   : 该段是否没有 submit（负向信号）

评估：与 rf29 同协议（按学生分组 5 折 × 3 seeds），配对 bootstrap 2000 次
"""
import argparse
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

EVENT_TYPES = ['text_insert', 'text_remove', 'text_paste',
               'focus_gained', 'focus_lost', 'run', 'submit']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ide', default='/home/ubuntu/IDE_logs/IDE_logs.csv')
    ap.add_argument('--out', default='data/deadline_feats.npz')
    ap.add_argument('--chunksize', type=int, default=1_500_000)
    args = ap.parse_args()

    t0 = time.time()
    acc = defaultdict(lambda: dict(ttd_max=-1e18, ttd_min=1e18, n=0,
                                   sub_min=1e18, run_min=1e18, n_sub=0,
                                   rush=0))
    rows = 0
    for chunk in pd.read_csv(args.ide,
                             usecols=['student', 'part', 'eventType', 'timeToDeadline'],
                             chunksize=args.chunksize):
        ttd = chunk['timeToDeadline'].astype(float).values
        stu = chunk['student'].values
        part = chunk['part'].values
        evt = chunk['eventType'].values
        df = pd.DataFrame({'s': stu, 'p': part, 'ttd': ttd, 'evt': evt})
        # 每组：ttd max/min、submit 的 ttd min、run 的 ttd min、计数
        g = df.groupby(['s', 'p'], sort=False)
        for key, sub in g:
            a = acc[(int(key[0]), int(key[1]))]
            t = sub['ttd'].values
            a['ttd_max'] = max(a['ttd_max'], float(t.max()))
            a['ttd_min'] = min(a['ttd_min'], float(t.min()))
            a['n'] += len(t)
            m = sub['evt'].values == 'submit'
            if m.any():
                a['sub_min'] = min(a['sub_min'], float(t[m].min()))
                a['n_sub'] += int(m.sum())
            m = sub['evt'].values == 'run'
            if m.any():
                a['run_min'] = min(a['run_min'], float(t[m].min()))
            a['rush'] += int((t < 0.10 * t.max()).sum())
        rows += len(chunk)
        print(f'  read {rows/1e6:.1f}M | groups={len(acc)} | {time.time()-t0:.0f}s',
              flush=True)

    keys = sorted(acc.keys())
    N = len(keys)
    F = np.zeros((N, 9), dtype=np.float32)
    names = ['ttd_start_h', 'ttd_end_h', 'utilization', 'submit_slack_h',
             'submit_ratio', 'run_slack_h', 'rate_per_h', 'rush_ratio', 'no_submit']
    sid = np.zeros(N, dtype=np.int64)
    par = np.zeros(N, dtype=np.int64)
    for i, (s, p) in enumerate(keys):
        a = acc[(s, p)]
        start = max(a['ttd_max'], 1.0)
        end = max(a['ttd_min'], 0.0)
        F[i, 0] = start / 3600.0
        F[i, 1] = end / 3600.0
        F[i, 2] = 1.0 - end / start
        F[i, 3] = (a['sub_min'] if a['n_sub'] else start) / 3600.0
        F[i, 4] = (a['sub_min'] / start) if a['n_sub'] else 1.0
        F[i, 5] = (a['run_min'] if a['run_min'] < 1e17 else start) / 3600.0
        F[i, 6] = a['n'] / max(start / 3600.0, 1e-6)
        F[i, 7] = a['rush'] / max(a['n'], 1)
        F[i, 8] = 0.0 if a['n_sub'] else 1.0
        sid[i] = s
        par[i] = p

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, feats=F, sid=sid, part=par, names=np.array(names))
    print(f'写出 {args.out}  N={N}  dim={F.shape[1]}  用时 {time.time()-t0:.0f}s',
          flush=True)
    print('  特征名:', names, flush=True)


if __name__ == '__main__':
    main()
