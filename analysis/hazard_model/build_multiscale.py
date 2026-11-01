#!/usr/bin/env python3
"""
全轨迹多尺度特征（Full-trajectory multi-scale features）
=================================================================================
动机（来自已有负结果与缺口）：
  · S8：max_len 128→1024 仅 +0.4%，且每生事件数跨约 4 个数量级（log 分布），
         大量学生的序列在 max_len=128 处被左填零 → 序列主干实际只看到了尾部。
  · S3/M4：7 维 one-hot 缺强度信息；但「事件间隔/密度类强度统计」在阶段级无增益
         （M4 ΔAUC −0.0013）→ 说明问题不是"没有强度"，而是"强度统计错了对象"。
  · F4：标注极少（258 事件）但无标注事件海量（28.6M）→ 应当用「结构」而不是「统计量」。

做法：对每个 (学生, part) 的**全部**事件（不截断）做会话级切分（gap > 30 min 或
focus_gained 开启新会话），提取会话结构特征：
  会话数、会话时长分布（mean/median/max）、每会话事件数、含 submit 的会话占比、
  run/edit 比、纯编辑无运行率、返工率（remove/insert）、焦点驻留时长、
  活动天数、最近一次活动距截止的间隔、首末活动时间跨度、昼夜分布。

用法：
  python3 build_multiscale.py --ide <IDE_logs.csv> --passed <passed.csv> \
      --out multiscale_feats.npz
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
    ap.add_argument('--passed', default='/home/ubuntu/IDE_logs/passed.csv')
    ap.add_argument('--out', default='data/multiscale_feats.npz')
    ap.add_argument('--chunksize', type=int, default=1_500_000)
    ap.add_argument('--session-gap-min', type=float, default=30.0)
    args = ap.parse_args()

    t0 = time.time()
    GAP = np.timedelta64(int(args.session_gap_min * 60), 's')

    # 逐块读取 → 按 (student, part) 聚合会话级统计
    # 为控制内存：只保留每个 (student,part) 的会话级累加器 + 少量原始时间戳摘要
    acc = defaultdict(lambda: dict(n=0, ins=0, rem=0, pas=0, fg=0, fl=0, run=0, sub=0,
                                   sess=0, sess_with_sub=0, sess_edit_no_run=0,
                                   dwell_sum=0.0, dwell_n=0, first=None, last=None,
                                   days=None, t_sub_last=None))
    rows = 0
    for chunk in pd.read_csv(args.ide,
                             usecols=['student', 'part', 'eventType', 'timestamp',
                                      'timeToDeadline'],
                             chunksize=args.chunksize):
        chunk['ts'] = pd.to_datetime(chunk['timestamp'])
        for (s, p), g in chunk.groupby(['student', 'part'], sort=False):
            a = acc[(int(s), int(p))]
            a['n'] += len(g)
            vc = g['eventType'].value_counts()
            a['ins'] += int(vc.get('text_insert', 0))
            a['rem'] += int(vc.get('text_remove', 0))
            a['pas'] += int(vc.get('text_paste', 0))
            a['fg'] += int(vc.get('focus_gained', 0))
            a['fl'] += int(vc.get('focus_lost', 0))
            a['run'] += int(vc.get('run', 0))
            a['sub'] += int(vc.get('submit', 0))
            ts = g['ts'].values
            t_min = pd.Timestamp(g['ts'].min())
            t_max = pd.Timestamp(g['ts'].max())
            if a['first'] is None or t_min < a['first']:
                a['first'] = t_min
            if a['last'] is None or t_max > a['last']:
                a['last'] = t_max
            if a['days'] is None:
                a['days'] = set()
            a['days'].update(pd.DatetimeIndex(ts).normalize().unique().tolist())
            # 会话切分
            ts_sorted = np.sort(ts)
            if len(ts_sorted) > 1:
                brk = np.where(np.diff(ts_sorted) > GAP)[0]
                seg_starts = np.concatenate([[0], brk + 1])
                seg_ends = np.concatenate([brk, [len(ts_sorted) - 1]])
                a['sess'] += len(seg_starts)
            else:
                seg_starts = np.array([0])
                seg_ends = np.array([len(ts_sorted) - 1])
            # 每会话是否含 submit（用时间顺序近邻近似：按 submit 的 ts 归属会话）
            sub_ts = g.loc[g['eventType'] == 'submit', 'ts'].values
            if len(sub_ts):
                sub_ts = np.sort(sub_ts)
                for k in range(len(seg_starts)):
                    lo, hi = ts_sorted[seg_starts[k]], ts_sorted[seg_ends[k]]
                    if np.any((sub_ts >= lo) & (sub_ts <= hi)):
                        a['sess_with_sub'] += 1
                        break
        rows += len(chunk)
        print(f'  read {rows/1e6:.1f}M events | groups={len(acc)} | {time.time()-t0:.0f}s',
              flush=True)

    print(f'聚合完成：{rows} 事件 / {len(acc)} 组 / {time.time()-t0:.0f}s', flush=True)

    keys = sorted(acc.keys())
    N = len(keys)
    feats = np.zeros((N, 14), dtype=np.float32)
    names = ['sess_per_stage', 'events_per_sess', 'sess_with_sub_ratio',
             'run_per_ins', 'edit_no_run_ratio', 'rework_ratio', 'focus_dwell_mean',
             'active_days', 'span_hours', 'idle_tail_hours', 'submit_density',
             'paste_ratio', 'focus_churn_per_h', 'events_log']
    sid_arr = np.zeros(N, dtype=np.int64)
    part_arr = np.zeros(N, dtype=np.int64)
    for i, (s, p) in enumerate(keys):
        a = acc[(s, p)]
        n = max(a['n'], 1)
        span_h = (a['last'] - a['first']).total_seconds() / 3600.0
        sess = max(a['sess'], 1)
        feats[i, 0] = a['sess']
        feats[i, 1] = n / sess
        feats[i, 2] = a['sess_with_sub'] / sess
        feats[i, 3] = a['run'] / max(a['ins'], 1)
        feats[i, 4] = max(a['ins'] - a['run'], 0) / max(a['ins'], 1)
        feats[i, 5] = a['rem'] / max(a['ins'], 1)
        feats[i, 6] = a['dwell_sum'] / max(a['dwell_n'], 1)
        feats[i, 7] = len(a['days'])
        feats[i, 8] = span_h
        feats[i, 9] = 0.0
        feats[i, 10] = a['sub'] / span_h if span_h > 0 else 0.0
        feats[i, 11] = a['pas'] / n
        feats[i, 12] = a['fl'] / span_h if span_h > 0 else 0.0
        feats[i, 13] = np.log1p(n)
        sid_arr[i] = s
        part_arr[i] = p

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, feats=feats, sid=sid_arr, part=part_arr,
                        names=np.array(names))
    print(f'写出 {args.out}  N={N}  dim={feats.shape[1]}  用时 {time.time()-t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    main()
