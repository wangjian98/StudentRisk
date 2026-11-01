#!/usr/bin/env python3
"""
习题级结构勘查（决定掉队判点如何下沉到习题）
=================================================================================
要回答的五个问题：
  Q1 每个 part 的习题集合与题量（全局有多少题可用）
  Q2 学生在 part 内触碰的习题是否是**前缀**（即能否用"最后触碰的习题"当停点）
  Q3 是否存在跳题（触碰了后面的题但没碰前面的题）
  Q4 每个 (part, exercise) 的学生覆盖度（决定稀疏度与收缩策略）
  Q5 习题级样本量与事件量分布
"""
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

EVENT_TYPES = ['text_insert', 'text_remove', 'text_paste',
               'focus_gained', 'focus_lost', 'run', 'submit']


def main():
    ide = '/home/ubuntu/IDE_logs/IDE_logs.csv'
    t0 = time.time()
    # 只读 student/part/exercise（先不读事件列，省时间）
    touched = defaultdict(set)      # (s,p) -> set(exercise)
    part_ex = defaultdict(set)      # p -> set(exercise)
    cnt_spe = defaultdict(int)      # (p,e) -> #students
    rows = 0
    for chunk in pd.read_csv(ide, usecols=['student', 'part', 'exercise'],
                             chunksize=3_000_000):
        for (s, p, e), g in chunk.groupby(['student', 'part', 'exercise'], sort=False):
            touched[(int(s), int(p))].add(int(e))
            part_ex[int(p)].add(int(e))
            cnt_spe[(int(p), int(e))] += 1
        rows += len(chunk)
        print(f'  read {rows/1e6:.1f}M | (s,p) groups={len(touched)} | '
              f'{time.time()-t0:.0f}s', flush=True)

    print('\n=== Q1 每个 part 的可用习题集合 ===')
    for p in sorted(part_ex):
        ex = sorted(part_ex[p])
        print(f'  part {p}: 习题 {min(ex)}..{max(ex)}  共 {len(ex)} 题')

    print('\n=== Q2/Q3 前缀性检查（学生是否跳题）===')
    prefix_ok = 0
    skip = 0
    gaps = defaultdict(int)
    for (s, p), ex in touched.items():
        ex = sorted(ex)
        avail = sorted(part_ex[p])
        if ex == avail[:len(ex)]:
            prefix_ok += 1
        else:
            # 找第一个缺失的可用题
            first_missing = None
            for a in avail:
                if a not in ex:
                    first_missing = a
                    break
            if first_missing is not None and min(ex) > first_missing:
                skip += 1
            gaps[len(ex)] += 1
    n_groups = len(touched)
    print(f'  (学生,part) 组 {n_groups} | 触碰前缀完全一致 {prefix_ok} ({prefix_ok/n_groups:.1%})'
          f' | 存在跳题 {skip} ({skip/n_groups:.1%})')

    print('\n=== Q4 每个 (part, exercise) 的学生覆盖度 ===')
    rowsout = []
    for p in sorted(part_ex):
        line = []
        for e in sorted(part_ex[p]):
            line.append(f'{cnt_spe[(p,e)]}')
        print(f'  part {p}: ' + ' '.join(line))
        rowsout.append(len(part_ex[p]))

    print('\n=== Q5 (学生, part) 组内的习题数分布 ===')
    ex_counts = np.array([len(v) for v in touched.values()])
    print(f'  习题数：min {ex_counts.min()} p25 {int(np.percentile(ex_counts,25))} '
          f'中位 {int(np.median(ex_counts))} p75 {int(np.percentile(ex_counts,75))} '
          f'max {ex_counts.max()}')
    np.savez_compressed('/home/ubuntu/studentrisk_stage/data/exercise_structure.npz',
                        touched_keys=np.array([f'{s}|{p}' for (s, p) in touched]),
                        touched_vals=np.array([np.array(sorted(v), dtype=np.int32)
                                              for v in touched.values()], dtype=object),
                        part_ex_n=np.array(rowsout))
    print(f'  用时 {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
