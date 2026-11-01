#!/usr/bin/env python3
"""
学生级事件符号序列（用于复现 EDM 2025 的 HMM 设置）
=================================================================================
Murali et al. (EDM 2025) 在 CS1 MOOC 上用「60 秒时间窗」做增量窗口建模。
本脚本把 IDE_logs.csv 还原成**每生一条按时间排序的事件符号序列**，再按 60 秒
时间窗取窗内主导事件类型作为符号 → 得到与论文设置可对照的离散序列。

输出：data/student_event_seqs.npz
  ids     : object array, 每生一条 int8 符号序列
  lengths : 每生序列长度
  win_sec : 时间窗秒数
"""
import argparse
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

EVENT_TYPES = ['text_insert', 'text_remove', 'text_paste',
               'focus_gained', 'focus_lost', 'run', 'submit']
ET_IDX = {e: i for i, e in enumerate(EVENT_TYPES)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ide', default='/home/ubuntu/IDE_logs/IDE_logs.csv')
    ap.add_argument('--out', default='data/student_event_seqs.npz')
    ap.add_argument('--win-sec', type=int, default=3600)
    ap.add_argument('--chunksize', type=int, default=1_500_000)
    args = ap.parse_args()

    t0 = time.time()
    store = defaultdict(lambda: ([], []))   # student -> (ts list, ev list)
    rows = 0
    for chunk in pd.read_csv(args.ide, usecols=['student', 'eventType', 'timestamp'],
                             chunksize=args.chunksize):
        # 显式转 ns：pandas 2.x 对字符串列可能返回 datetime64[us]，
        # 直接 .astype('int64')//10**9 会把时间戳缩小 1000 倍（已踩坑修正）
        chunk['ts'] = pd.to_datetime(chunk['timestamp']).values.astype('datetime64[ns]').astype('int64') // 10**9
        for s, g in chunk.groupby('student', sort=False):
            ev = g['eventType'].map(ET_IDX).fillna(0).astype(np.int8).values
            ts = g['ts'].values
            lst_t, lst_e = store[int(s)]
            lst_t.append(ts.astype(np.int64))
            lst_e.append(ev)
        rows += len(chunk)
        print(f'  read {rows/1e6:.1f}M | students={len(store)} | {time.time()-t0:.0f}s',
              flush=True)

    ids, lengths = [], []
    win = args.win_sec
    for s in sorted(store.keys()):
        lst_t, lst_e = store[s]
        ts = np.concatenate(lst_t)
        ev = np.concatenate(lst_e)
        order = np.argsort(ts, kind='stable')
        ts, ev = ts[order], ev[order]
        # 60 秒窗 → 窗内主导事件类型
        bins = (ts - ts[0]) // win
        ub = np.unique(bins)
        syms = np.empty(len(ub), dtype=np.int8)
        for j, b in enumerate(ub):
            v = ev[bins == b]
            syms[j] = np.bincount(v, minlength=7).argmax()
        ids.append(syms)
        lengths.append(len(syms))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, ids=np.array(ids, dtype=object),
                        lengths=np.array(lengths), win_sec=win)
    L = np.array(lengths)
    print(f'写出 {args.out} | 学生 {len(ids)} | 窗数 中位 {int(np.median(L))} '
          f'最大 {L.max()} | 用时 {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
