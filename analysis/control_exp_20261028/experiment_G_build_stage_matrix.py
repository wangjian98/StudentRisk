"""
实验 G：固化 (student, part) 阶段级特征矩阵
=============================================
产出 stage_matrix.npz：
  - feat  : (N_stage, 10)  = 7 维事件计数 + total + distinct_exercises + span_hours
  - sid   : (N_stage,)    student id
  - pid   : (N_stage,)    part (原始 1..7)
  - first : (N_stage,)    该 (student,part) 首次事件时间戳（epoch 秒）
  - last  : (N_stage,)    该 (student,part) 末次事件时间戳
  - y     : (N_students,) 课程级标签（failed=1），按 student id 排序
  - students : (N_students,) student id 列表

后续所有 stage-level / hazard / early-warning 实验直接复用本文件，无需重读 1.3GB 原始日志。
"""
import os
from collections import defaultdict

import numpy as np
import pandas as pd

IDE = os.environ.get('STUDENTRISK_IDE_LOGS', '/root/.openclaw/data-backups/IDE_logs.csv')
PASS = os.environ.get('STUDENTRISK_PASSED', '/root/.openclaw/data-backups/passed.csv')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'F_stage_probe')
os.makedirs(OUT, exist_ok=True)

EV = ['text_insert', 'text_remove', 'text_paste',
      'focus_gained', 'focus_lost', 'run', 'submit']

cnt = defaultdict(lambda: defaultdict(int))
ts_min, ts_max, ex_n = {}, {}, {}
n = 0
for chunk in pd.read_csv(IDE, usecols=['student', 'part', 'exercise', 'eventType', 'timestamp'],
                         chunksize=2_000_000):
    chunk['timestamp'] = pd.to_datetime(chunk['timestamp'])
    g = chunk.groupby(['student', 'part', 'eventType']).size()
    for (s, p, e), v in g.items():
        cnt[(int(s), int(p))][e] += int(v)
    g2 = chunk.groupby(['student', 'part'])['timestamp'].agg(['min', 'max'])
    for (s, p), r in g2.iterrows():
        k = (int(s), int(p))
        if k not in ts_min or r['min'] < ts_min[k]:
            ts_min[k] = r['min']
        if k not in ts_max or r['max'] > ts_max[k]:
            ts_max[k] = r['max']
    g3 = chunk.groupby(['student', 'part'])['exercise'].nunique()
    for (s, p), v in g3.items():
        k = (int(s), int(p))
        ex_n[k] = ex_n.get(k, 0) + int(v)
    n += len(chunk)
    print(f'  read {n/1e6:.1f}M events', flush=True)

pdf = pd.read_csv(PASS)
pdf['student'] = pdf['student'].astype(int)
pdf['failed'] = (~pdf['passed'].astype(bool)).astype(int)
students = sorted(pdf['student'].unique().tolist())
y_map = dict(zip(pdf['student'], pdf['failed']))

keys = sorted(cnt.keys())
feat = np.zeros((len(keys), 10), dtype=np.float32)
sid = np.zeros(len(keys), dtype=np.int64)
pid = np.zeros(len(keys), dtype=np.int64)
first = np.zeros(len(keys), dtype=np.float64)
last = np.zeros(len(keys), dtype=np.float64)
for i, k in enumerate(keys):
    s, p = k
    d = cnt[k]
    v = [d.get(e, 0) for e in EV] + [sum(d.get(e, 0) for e in EV),
                                    float(ex_n.get(k, 0)),
                                    (ts_max[k] - ts_min[k]).total_seconds() / 3600.0]
    feat[i] = v
    sid[i], pid[i] = s, p
    first[i] = ts_min[k].timestamp()
    last[i] = ts_max[k].timestamp()

np.savez_compressed(f'{OUT}/stage_matrix.npz',
                    feat=feat, sid=sid, pid=pid, first=first, last=last,
                    y=np.array([y_map[s] for s in students], dtype=np.int64),
                    students=np.array(students, dtype=np.int64),
                    event_types=np.array(EV))
print(f'  stage samples = {len(keys)}, students = {len(students)}, '
      f'students in logs = {len(set(sid.tolist()))}')
print(f'  → {OUT}/stage_matrix.npz')
