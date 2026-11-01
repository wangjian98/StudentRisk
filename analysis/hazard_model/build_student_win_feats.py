#!/usr/bin/env python3
"""
学生级窗口特征（用于忠实复现 EDM 2025 的 GMM-HMM 设置）
=================================================================================
Murali et al. (EDM 2025) 在 CS1 MOOC 上的特征协议（论文 §3.2 / Table 3）：
  · 7 类事件的 **频率与计数**（text_insert / remove / paste / focus_gained /
    focus_lost / run / submit）
  · 每个特征同时算两遍：**非增量**（仅当前时间窗）与**增量**（截至当前窗累计）
  · 时间窗：CS1 用 60 秒窗（因交互时长差异大，改用从 110 分钟起的对数时间轴）

本脚本把 IDE_logs.csv 还原成每生的窗级特征序列：
  非增量部分 = 该窗内 7 类事件计数
  增量部分   = 截至该窗的 7 类事件累计计数
  → 每生一个 (T, 14) 的矩阵

同时保留按 1 小时分箱的**主导事件符号序列**（Arm 2a 的离散 HMM 输入）。

输出：data/student_win_feats.npz
  win_ids : object array, 每生一条 int8 符号序列（1 小时窗主导事件）
  win_cnt : object array, 每生一个 (T, 14) float32 矩阵
  lengths : 窗数
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
    ap.add_argument('--out', default='data/student_win_feats.npz')
    ap.add_argument('--win-sec', type=int, default=3600)
    ap.add_argument('--chunksize', type=int, default=1_500_000)
    args = ap.parse_args()

    t0 = time.time()
    store = defaultdict(lambda: ([], []))   # student -> (ts list, ev list)
    rows = 0
    for chunk in pd.read_csv(args.ide, usecols=['student', 'eventType', 'timestamp'],
                             chunksize=args.chunksize):
        # 显式转 ns（pandas 2.x 对本数据返回 datetime64[us]，直接 astype('int64')
        # 会把时间戳缩小 1000 倍 —— 已踩坑修正）
        ts_all = (pd.to_datetime(chunk['timestamp']).values.astype('datetime64[ns]')
                  .astype('int64') // 10**9)
        ev_all = chunk['eventType'].map(ET_IDX).fillna(0).astype(np.int8).values
        stu_all = chunk['student'].values
        for s in np.unique(stu_all):
            m = stu_all == s
            lst_t, lst_e = store[int(s)]
            lst_t.append(ts_all[m].astype(np.int64))
            lst_e.append(ev_all[m])
        rows += len(chunk)
        print(f'  read {rows/1e6:.1f}M | students={len(store)} | {time.time()-t0:.0f}s',
              flush=True)

    win_ids, win_cnt, lengths = [], [], []
    win = args.win_sec
    for s in sorted(store.keys()):
        lst_t, lst_e = store[s]
        ts = np.concatenate(lst_t)
        ev = np.concatenate(lst_e)
        order = np.argsort(ts, kind='stable')
        ts, ev = ts[order], ev[order]
        bins = (ts - ts[0]) // win
        ub, inv = np.unique(bins, return_inverse=True)
        T = len(ub)
        cnt = np.zeros((T, 7), dtype=np.float32)
        np.add.at(cnt, inv, np.eye(7, dtype=np.float32)[ev])
        cum = np.cumsum(cnt, axis=0)
        win_cnt.append(np.hstack([cnt, cum]))
        win_ids.append(cnt.argmax(1).astype(np.int8))
        lengths.append(T)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, win_ids=np.array(win_ids, dtype=object),
                        win_cnt=np.array(win_cnt, dtype=object),
                        lengths=np.array(lengths), win_sec=win)
    L = np.array(lengths)
    print(f'写出 {args.out} | 学生 {len(win_ids)} | 窗数 中位 {int(np.median(L))} '
          f'最大 {L.max()} | 用时 {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
