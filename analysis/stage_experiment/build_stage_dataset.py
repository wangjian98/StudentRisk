#!/usr/bin/env python3
"""
阶段级数据集构建：IDE_logs.csv → (学生, part) 级事件序列张量
==============================================================
为"阶段级掉队预测"任务准备数据。

标签（从日志事实推导，无需新标注）：
  drop = 1  ⇔ 该 (学生, part) 是他到达过的最远 part，且从未到达 part7
            （即：做完这一段之后他就不再推进了）
  final = passed.csv 的课程级 failed 标签

每条阶段样本的输入 = 该学生在**该 part 内**的事件序列（取最���近 MAX_LEN 条）
  seq11 : [et_oh(7) | log_interval(1) | deadline_norm(1) | part_norm(1) | exercise_norm(1)]
  seq9  : 同上去掉 part_norm / exercise_norm（任务身份只走 FiLM，避免双重注入）
  seq7  : 只保留 7 维事件类型 one-hot（对照论文 7d 变体）

同时输出 agg10（阶段内聚合：7 事件计数 + total + distinct_exercises + span_hours），
用于 RF / 特征工程基线（与本地 experiment_I 的 RAW 口径一致）。

用法：
  python3 build_stage_dataset.py --ide <IDE_logs.csv> --passed <passed.csv> \
      --out stage_dataset.npz --max-len 128
"""
import argparse
import os
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

EVENT_TYPES = ['text_insert', 'text_remove', 'text_paste',
               'focus_gained', 'focus_lost', 'run', 'submit']
ET_IDX = {e: i for i, e in enumerate(EVENT_TYPES)}
PARTS = list(range(1, 8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ide', default='/home/ubuntu/IDE_logs/IDE_logs.csv')
    ap.add_argument('--passed', default='/home/ubuntu/IDE_logs/passed.csv')
    ap.add_argument('--out', default='stage_dataset.npz')
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--chunksize', type=int, default=1_500_000)
    ap.add_argument('--limit-chunks', type=int, default=0, help='>0 只读前 N 个 chunk（调试）')
    args = ap.parse_args()

    t0 = time.time()
    MAXLEN = args.max_len
    groups = {}          # (s,p) -> {'rows': list, 'cnt': Counter}
    n_rows = 0
    n_chunks = 0
    for chunk in pd.read_csv(args.ide,
                             usecols=['student', 'part', 'exercise', 'eventType',
                                      'timestamp', 'timeToDeadline'],
                             chunksize=args.chunksize):
        n_chunks += 1
        chunk['ts'] = pd.to_datetime(chunk['timestamp'])
        for (s, p), g in chunk.groupby(['student', 'part'], sort=False):
            key = (int(s), int(p))
            d = groups.get(key)
            if d is None:
                d = {'rows': [], 'cnt': Counter()}
                groups[key] = d
            d['cnt'].update({k: int(v) for k, v in g['eventType'].value_counts().items()})
            d['rows'].append(g)
            # 合并并裁剪：只保留时间上最晚的 MAX_LEN 条（与学生级 tail(max_len) 口径一致）
            if len(d['rows']) > 1:
                d['rows'] = [pd.concat(d['rows'], ignore_index=True)]
            d['rows'][0] = d['rows'][0].sort_values('ts', kind='stable').tail(MAXLEN)
        n_rows += len(chunk)
        print(f'  read {n_rows/1e6:.1f}M events | groups={len(groups)} | {time.time()-t0:.0f}s',
              flush=True)
        if args.limit_chunks and n_chunks >= args.limit_chunks:
            print('  (limit-chunks 生效，停止读取)', flush=True)
            break

    print(f'读取完成：{n_rows} 事件 / {len(groups)} 个 (学生,part) 组 / {time.time()-t0:.0f}s',
          flush=True)

    # ---- 标签
    pdf = pd.read_csv(args.passed)
    pdf['student'] = pdf['student'].astype(int)
    pdf['failed'] = (~pdf['passed'].astype(bool)).astype(int)
    y_map = dict(zip(pdf['student'], pdf['failed']))
    students = sorted(pdf['student'].unique().tolist())

    max_part = {}
    for (s, p) in groups:
        max_part[s] = max(max_part.get(s, 0), p)

    keys = sorted(groups.keys())
    N = len(keys)
    L = MAXLEN
    seq11 = np.zeros((N, L, 11), dtype=np.float32)
    seq9 = np.zeros((N, L, 9), dtype=np.float32)
    seq7 = np.zeros((N, L, 7), dtype=np.float32)
    mask = np.zeros((N, L), dtype=np.float32)
    task_ids = np.zeros(N, dtype=np.int64)
    part_ids = np.zeros(N, dtype=np.int64)
    sid_arr = np.zeros(N, dtype=np.int64)
    drop = np.zeros(N, dtype=np.int64)
    final = np.zeros(N, dtype=np.int64)
    agg10 = np.zeros((N, 10), dtype=np.float32)
    first_ts = np.zeros(N, dtype=np.float64)
    last_ts = np.zeros(N, dtype=np.float64)

    n_parts_max = 7
    n_ex_max = max(int(g['rows'][0]['exercise'].max()) for g in groups.values())

    for i, (s, p) in enumerate(keys):
        d = groups[(s, p)]
        ev = d['rows'][0].sort_values('ts', kind='stable').reset_index(drop=True)
        n_ev = len(ev)
        task_ids[i] = p - 1
        part_ids[i] = p
        sid_arr[i] = s
        final[i] = y_map.get(s, 0)
        drop[i] = 1 if (p == max_part[s] and p < 7) else 0

        # --- 聚合特征（与本地 stage_matrix 同口径）
        cnt = d['cnt']
        agg10[i, :7] = [cnt.get(e, 0) for e in EVENT_TYPES]
        agg10[i, 7] = sum(cnt.values())
        agg10[i, 8] = int(ev['exercise'].nunique())
        ts = ev['ts']
        span_h = (ts.max() - ts.min()).total_seconds() / 3600.0
        agg10[i, 9] = span_h
        first_ts[i] = ts.min().value / 1e9
        last_ts[i] = ts.max().value / 1e9

        # --- 序列特征
        et_ids = ev['eventType'].map(ET_IDX).fillna(0).astype(int).values
        et_oh = np.zeros((n_ev, 7), dtype=np.float32)
        et_oh[np.arange(n_ev), et_ids] = 1.0

        if n_ev > 1:
            iv = (np.diff(ts.values).astype('timedelta64[s]').astype(float))
            iv = np.maximum(iv, 1.0)
            li = (np.log1p(iv) / 10.0).clip(0, 1).astype(np.float32)
            li = np.concatenate([[0.0], li])
        else:
            li = np.zeros(1, dtype=np.float32)

        dl = ev['timeToDeadline'].astype(float).values
        dn = (np.log1p(np.maximum(dl, 0)) / 20.0).clip(0, 1).astype(np.float32)

        pn = np.clip((ev['part'].astype(float).values - 1.0) / max(n_parts_max - 1, 1),
                     0, 1).astype(np.float32)
        xn = np.clip((ev['exercise'].astype(float).values - 1.0) / max(n_ex_max - 1, 1),
                     0, 1).astype(np.float32)

        f11 = np.concatenate([et_oh, li.reshape(-1, 1), dn.reshape(-1, 1),
                              pn.reshape(-1, 1), xn.reshape(-1, 1)], axis=1).astype(np.float32)
        seq11[i, L - n_ev:] = f11
        seq9[i, L - n_ev:] = f11[:, :9]
        seq7[i, L - n_ev:] = f11[:, :7]
        mask[i, L - n_ev:] = 1.0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, seq11=seq11, seq9=seq9, seq7=seq7, mask=mask,
                        task_ids=task_ids, part_ids=part_ids, sid=sid_arr,
                        drop=drop, final=final, agg10=agg10,
                        first_ts=first_ts, last_ts=last_ts,
                        event_types=np.array(EVENT_TYPES))
    print(f'写出 {args.out}  N={N}  max_len={L}  掉队正例={int(drop.sum())}  '
          f'挂科={int(final.sum())}  用时 {time.time()-t0:.0f}s', flush=True)
    print(f'  每阶段平均事件数 {agg10[:,7].mean():.0f}，序列实际使用长度 '
          f'{mask.sum(1).mean():.0f}/{L}', flush=True)


if __name__ == '__main__':
    main()
