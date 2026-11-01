#!/usr/bin/env python3
"""
习题级数据集构建：(学生, part, 习题) 级行为聚合
=================================================================================
结构勘查结论（inspect_exercise_structure.py）：
  · 每个 part 有 10–30 道习题（part1/2 各 30，part3 19，part4 16，part5/6 19，part7 11）
  · 97.1% 的 (学生,part) 组触碰**连续前缀**，跳题仅 0.6% → "最后触碰的习题"可作停点
  · 每个 (part,习题) 的学生覆盖度高（part1 第 1 题 475 人 → 第 30 题 407 人）

聚合内容（每个 (s,p,e)）：
  7 类事件计数、事件总数、跨度小时、ttd 起止与利用率、submit/run 数、
  首末时间戳（用 ttd 的 max/min 定位，ttd 单调递减）

标签（在评测脚本里由本文件的 touched 序列推导）：
  drop_final : 该生**最终**会在某个 part 掉队 → 该生所有习题行都标 1（早期预警用）
  drop_at    : 该生掉队的**停点习题**（其最终 part 内最后触碰的习题）→ 经典 hazard 事件

输出：data/exercise_dataset.npz
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
    ap.add_argument('--out', default='data/exercise_dataset.npz')
    ap.add_argument('--chunksize', type=int, default=2_000_000)
    args = ap.parse_args()

    t0 = time.time()
    # (s,p,e) -> [n, ins, rem, pas, fg, fl, run, sub, ttd_max, ttd_min, ts_min, ts_max]
    acc = {}
    rows = 0
    for chunk in pd.read_csv(args.ide,
                             usecols=['student', 'part', 'exercise', 'eventType',
                                      'timestamp', 'timeToDeadline'],
                             chunksize=args.chunksize):
        chunk['ts'] = (pd.to_datetime(chunk['timestamp']).values.astype('datetime64[ns]')
                       .astype('int64') // 10**9)
        chunk['ev'] = chunk['eventType'].map(ET_IDX).fillna(0).astype(np.int8)
        g = chunk.groupby(['student', 'part', 'exercise'], sort=False)
        for key, sub in g:
            k = (int(key[0]), int(key[1]), int(key[2]))
            a = acc.get(k)
            n = len(sub)
            ev = sub['ev'].values
            ttd = sub['timeToDeadline'].astype(float).values
            ts = sub['ts'].values
            cnt = np.bincount(ev, minlength=7)
            tmin, tmax = int(ts.min()), int(ts.max())
            if a is None:
                acc[k] = np.array([n, cnt[0], cnt[1], cnt[2], cnt[3], cnt[4], cnt[5],
                                   cnt[6], ttd.max(), ttd.min(), tmin, tmax],
                                  dtype=np.float64)
            else:
                a[0] += n
                a[1:8] += cnt
                a[8] = max(a[8], ttd.max())
                a[9] = min(a[9], ttd.min())
                a[10] = min(a[10], tmin)
                a[11] = max(a[11], tmax)
        rows += len(chunk)
        print(f'  read {rows/1e6:.1f}M | (s,p,e) groups={len(acc)} | '
              f'{time.time()-t0:.0f}s', flush=True)

    print(f'聚合完成：{rows} 事件 / {len(acc)} 组 / {time.time()-t0:.0f}s', flush=True)

    keys = sorted(acc.keys())
    N = len(keys)
    F = np.zeros((N, 12), dtype=np.float32)
    sid = np.zeros(N, dtype=np.int64)
    par = np.zeros(N, dtype=np.int64)
    exr = np.zeros(N, dtype=np.int64)
    for i, (s, p, e) in enumerate(keys):
        a = acc[(s, p, e)]
        F[i, :] = a
        sid[i], par[i], exr[i] = s, p, e

    # 每个 (s,p) 的习题序列（有序），用于推导停点
    order = defaultdict(list)
    for (s, p, e) in keys:
        order[(s, p)].append(e)
    sp_keys = sorted(order.keys())
    sp_ex = np.array([np.array(sorted(order[k]), dtype=np.int32) for k in sp_keys],
                     dtype=object)
    sp_s = np.array([k[0] for k in sp_keys], dtype=np.int64)
    sp_p = np.array([k[1] for k in sp_keys], dtype=np.int64)
    # 每个 part 的可用习题总数
    part_tot = {}
    for p in np.unique(par):
        part_tot[int(p)] = int(par[par == p].max())

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, raw=F, sid=sid, part=par, exercise=exr,
                        sp_s=sp_s, sp_p=sp_p, sp_ex=sp_ex,
                        part_tot=np.array([part_tot[int(p)] for p in np.unique(par)]),
                        part_ids=np.array(sorted(part_tot.keys())),
                        event_types=np.array(EVENT_TYPES))
    print(f'写出 {args.out}  N={N}（习题级样本）  dim=12  '
          f'每生平均习题行 {N/len(set(sid.tolist())):.1f}  用时 {time.time()-t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    main()
