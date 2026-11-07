"""
rev12.common — 统一数据/特征/协议层（论文 v12 修订实验包）

本文件实现"论文 v2.5 对比设计修订"所需的四件事：

1) **同协议数据底座**：473 名学生、28.5M 条 IDE 事件，一次性编码为**缓存**，
   之后所有实验（末 128 事件 / 前缀曲线 / LOPO）都从同一缓存派生，
   保证"同一数据构造、同一折划分、同一指标"。

2) **三个输入空间（feature spaces）**
   - 7d  : 7 类事件 one-hot（与既有 RF-7d / LSTM-7d 对齐）
   - 11d : 7 one-hot + 间隔 + 截止距离 + part + exercise（与既有 MetaMamba-11d 对齐）
   - 28d : 7 one-hot + 21 个行为/上下文通道（本文新定义，替代"引用值"的 28 维）
     所有通道**严格 past-only**（不使用任何未来信息），这一点在 prefix 实验里可被检验。

3) **学生级聚合孪生（tabular twin）**：把同一空间的通道聚合成表格特征，
   使 RF 与序列模型在**同一信息量**下比较（架构维度解耦）。

4) **评估协议**：5 折 × 3 seeds 分层交叉验证（按学生分组）；训练折内再切 15%
   内层验证集用于早停与超参选择；指标含 balanced accuracy（审稿意见要求）。

约定：y=1 表示挂科（FAILED），y=0 表示通过（PASSED）。
"""
import os
import json
import time
import numpy as np
import pandas as pd

EVENT_TYPES = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit',
]
EVENT_TYPE_TO_IDX = {et: i for i, et in enumerate(EVENT_TYPES)}

N_CHANNELS = 28          # 28d 空间的通道总数
CHANNEL_NAMES_28 = (
    [f'et_{et}' for et in EVENT_TYPES]                     # 0-6
    + ['log_interval', 'log_deadline', 'part_norm', 'exercise_norm']   # 7-10
    + ['pos_in_student', 'time_since_start', 'hour_sin', 'hour_cos',    # 11-14
       'dow_norm', 'is_weekend', 'cum_type_frac', 'cum_submit',        # 15-18
       'cum_run', 'cum_focus_lost', 'cum_focus_gained', 'cum_distinct_ex',  # 19-22
       'gap_since_submit', 'gap_since_run', 'gap_since_focus_lost',     # 23-25
       'same_type_run_len', 'part_event_share']                        # 26-27
)
CHANNEL_NAMES_11 = (
    [f'et_{et}' for et in EVENT_TYPES]
    + ['log_interval', 'log_deadline', 'part_norm', 'exercise_norm']
)
CHANNEL_NAMES_7 = [f'et_{et}' for et in EVENT_TYPES]

CACHE_ENV = 'SR12_CACHE'
DEFAULT_CACHE = '/home/ubuntu/SR12_cache'


def cache_dir() -> str:
    return os.environ.get(CACHE_ENV, DEFAULT_CACHE)


# ────────────────────────────── 缓存构建 ──────────────────────────────

def build_cache(ide_logs_path: str, passed_path: str, out_dir: str = None) -> str:
    """把 28.5M 行 IDE 日志压成按学生切片的紧凑缓存（一次性，后续秒级加载）。"""
    out_dir = out_dir or cache_dir()
    os.makedirs(out_dir, exist_ok=True)
    cache_file = os.path.join(out_dir, 'events.npz')
    if os.path.exists(cache_file):
        print(f"[cache] already exists: {cache_file}", flush=True)
        return cache_file

    t0 = time.time()
    print(f"[cache] reading {ide_logs_path} ...", flush=True)
    df = pd.read_csv(
        ide_logs_path,
        usecols=['student', 'part', 'exercise', 'eventType', 'timestamp', 'timeToDeadline'],
    )
    print(f"[cache] raw shape={df.shape} ({time.time()-t0:.1f}s)", flush=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['student', 'timestamp'], kind='mergesort')
    print(f"[cache] sorted ({time.time()-t0:.1f}s)", flush=True)

    labels = pd.read_csv(passed_path)
    labels['passed'] = labels['passed'].astype(bool)
    labels = labels.sort_values('student')
    lab_map = {int(s): int((not bool(p))) for s, p in zip(labels['student'].values,
                                                           labels['passed'].values)}
    y_all = np.array(list(lab_map.values()), dtype=np.int8)   # Failed=1

    et = df['eventType'].map(EVENT_TYPE_TO_IDX).fillna(0).astype(np.int8).values
    ts = df['timestamp'].astype('int64').values // 10**9      # epoch seconds
    part = df['part'].astype(np.int16).values
    ex = df['exercise'].astype(np.int16).values
    ttd = df['timeToDeadline'].astype(np.float32).values
    sid = df['student'].values
    print(f"[cache] extracted columns ({time.time()-t0:.1f}s)", flush=True)

    # 按学生切分（student 列已排序）
    order = np.argsort(sid, kind='mergesort')
    sid_sorted = sid[order]
    et, ts, part, ex, ttd = et[order], ts[order], part[order], ex[order], ttd[order]
    uniq, counts = np.unique(sid_sorted, return_counts=True)
    starts = np.cumsum(counts) - counts
    # 只保留在 passed.csv 中出现的学生
    lab_ids = np.array(list(lab_map.keys()))
    keep = np.isin(uniq, lab_ids)
    keep_students = uniq[keep]
    keep_counts = counts[keep]
    keep_starts = starts[keep]
    student_ids = keep_students
    y = np.array([lab_map[int(s)] for s in keep_students], dtype=np.int8)
    n_missing = len(lab_map) - len(keep_students)
    if n_missing:
        print(f"[cache] WARNING: {n_missing} students in passed.csv have no IDE logs "
              f"-> dropped", flush=True)
    offsets = np.zeros(len(keep_students) + 1, dtype=np.int64)
    np.cumsum(keep_counts, out=offsets[1:])
    idx = np.concatenate([np.arange(s, s + c, dtype=np.int64)
                          for s, c in zip(keep_starts, keep_counts)])

    np.savez(
        cache_file,
        et=et[idx], ts=ts[idx], part=part[idx], exercise=ex[idx], ttd=ttd[idx],
        offsets=offsets, student_ids=keep_students, y=y,
    )
    print(f"[cache] written {cache_file} "
          f"(events={len(idx)}, students={len(keep_students)}, {time.time()-t0:.1f}s)", flush=True)
    meta = {
        'n_events': int(len(idx)), 'n_students': int(len(keep_students)),
        'fail_rate': float(y.mean()), 'ex_max': int(ex[idx].max()),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'ide_logs_path': ide_logs_path, 'passed_path': passed_path,
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    return cache_file


def load_cache(cache_file: str = None):
    """加载缓存 → dict(et, ts, part, exercise, ttd, offsets, student_ids, y, ex_max)"""
    if cache_file is None:
        cache_file = os.path.join(cache_dir(), 'events.npz')
    z = np.load(cache_file, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    meta_path = os.path.join(os.path.dirname(cache_file), 'meta.json')
    d['ex_max'] = int(d['exercise'].max())
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            d['meta'] = json.load(f)
    return d


# ────────────────────────── 学生级事件编码（28 通道） ──────────────────────────

def encode_student(et, ts, part, exercise, ttd, ex_max) -> np.ndarray:
    """把一名学生的完整事件序列编码成 (n_events, 28) float32（严格 past-only）。"""
    n = len(et)
    out = np.zeros((n, N_CHANNELS), dtype=np.float32)
    # 0-6 事件类型 one-hot
    out[np.arange(n), et] = 1.0
    # 7 事件间隔
    if n > 1:
        dt = np.diff(ts, prepend=ts[0]).astype(np.float64)
        out[:, 7] = np.clip(np.log1p(np.maximum(dt, 1.0)) / np.log1p(3600.0), 0, 1)
    # 8 截止距离
    out[:, 8] = np.clip(np.log1p(np.maximum(ttd.astype(np.float64), 0.0))
                        / np.log1p(7 * 86400.0), 0, 1)
    # 9 part
    out[:, 9] = np.clip((part.astype(np.float32) - 1.0) / 6.0, 0, 1)
    # 10 exercise
    out[:, 10] = np.clip((exercise.astype(np.float32) - 1.0) / max(ex_max - 1, 1), 0, 1)
    # 11 在该生全部事件中的位置
    out[:, 11] = np.arange(n) / max(n - 1, 1)
    # 12 距该生首个事件的时长
    out[:, 12] = np.clip(np.log1p(np.maximum(ts - ts[0], 0)) / np.log1p(30 * 86400.0), 0, 1)
    # 13/14 时刻 sin/cos
    hour = (ts % 86400) / 3600.0
    out[:, 13] = np.sin(2 * np.pi * hour / 24.0)
    out[:, 14] = np.cos(2 * np.pi * hour / 24.0)
    # 15/16 星期
    dow = ((ts // 86400) + 4) % 7          # 1970-01-01 = Thursday
    out[:, 15] = dow / 6.0
    out[:, 16] = (dow >= 5).astype(np.float32)
    # 17 累计同类型占比
    cum_type = np.zeros(n, dtype=np.float32)
    for c in range(7):
        idx_c = np.where(et == c)[0]
        if len(idx_c):
            cum_type[idx_c] = np.arange(1, len(idx_c) + 1)
    out[:, 17] = cum_type / np.maximum(np.arange(1, n + 1), 1)
    # 18-21 累计计数（log 归一）
    for c, ch in ((6, 18), (5, 19), (4, 20), (3, 21)):   # submit, run, focus_lost, focus_gained
        out[:, ch] = np.clip(np.log1p(np.cumsum(et == c)) / 8.0, 0, 1)
    # 22 累计 distinct exercise
    uniq_ex = np.unique(exercise)
    distinct = np.searchsorted(uniq_ex, exercise, side='right')
    out[:, 22] = np.clip(np.log1p(distinct) / 10.0, 0, 1)
    # 23-25 距上次 submit / run / focus_lost 的事件数（向量化：累计取最大）
    arange = np.arange(n)
    for c, ch in ((6, 23), (5, 24), (4, 25)):
        last_idx = np.maximum.accumulate(np.where(et == c, arange, -1))
        gap = np.where(last_idx >= 0, arange - last_idx, arange + 1).astype(np.float64)
        out[:, ch] = np.clip(np.log1p(gap) / 5.0, 0, 1)
    # 26 连续同类型长度（向量化 run-length）
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = et[1:] != et[:-1]
    grp = np.cumsum(change) - 1
    same = (arange - np.maximum.accumulate(np.where(change, arange, 0)) + 1).astype(np.float64)
    out[:, 26] = np.clip(np.log1p(same) / 5.0, 0, 1)
    # 27 当前 part 占该生已发生事件的比例
    out[:, 27] = _part_share(part)
    return out


def _part_share(part: np.ndarray) -> np.ndarray:
    """cumcount(part) / (arange+1)：该生已发生事件中属于当前 part 的比例（past-only）。"""
    n = len(part)
    # 累计每个 part 的计数 → 当前 part 的累计计数
    vals, inv = np.unique(part, return_inverse=True)
    counts = np.zeros(len(vals), dtype=np.int64)
    np.add.at(counts, inv, 1)
    cum = np.cumsum(counts)[inv] - counts[inv] + 1     # 到当前事件为止（含自身）该 part 的事件数
    return (cum / np.maximum(np.arange(1, n + 1), 1)).astype(np.float32)


def slice_window(enc: np.ndarray, prefix_frac: float, max_len: int):
    """取该生"前 prefix_frac 事件"里的最后 max_len 条事件 → (L, 28)"""
    n = len(enc)
    n_prefix = n if prefix_frac >= 1.0 else max(1, int(round(n * prefix_frac)))
    lo = max(0, n_prefix - max_len)
    return enc[lo:n_prefix]


def build_spaces_multi(cache: dict, dim: int, prefixes, max_len: int = 128):
    """一次遍历学生，构造多个前缀下的序列（避免重复编码 28.5M 事件）。

    返回 {prefix: (sequences, mask, task_ids)}
      · sequences: (n_students, max_len, dim) float32，左 padding
      · mask:      (n_students, max_len) float32
      · task_ids:  (n_students,) int64 —— 该前缀内出现最多的 part（0-indexed），
                   按前缀分别计算，避免用未来事件确定身份
    """
    et, ts = cache['et'], cache['ts']
    part, exercise, ttd = cache['part'], cache['exercise'], cache['ttd']
    offsets, student_ids = cache['offsets'], cache['student_ids']
    n_students = len(student_ids)
    ex_max = cache['ex_max']

    results, todo = {}, []
    for p in prefixes:
        f = os.path.join(cache_dir(), f'seq_d{dim}_p{p}_L{max_len}.npz')
        if os.path.exists(f):
            z = np.load(f)
            results[p] = (z['sequences'], z['mask'], z['task_ids'])
        else:
            todo.append(p)
    if not todo:
        return results

    t0 = time.time()
    seqs = {p: np.zeros((n_students, max_len, dim), dtype=np.float32) for p in todo}
    masks = {p: np.zeros((n_students, max_len), dtype=np.float32) for p in todo}
    tasks = {p: np.zeros(n_students, dtype=np.int64) for p in todo}
    for i in range(n_students):
        s, e = offsets[i], offsets[i + 1]
        if e <= s:
            continue
        enc = encode_student(et[s:e], ts[s:e], part[s:e], exercise[s:e], ttd[s:e], ex_max)
        for p in todo:
            n_prefix = e - s if p >= 1.0 else max(1, int(round((e - s) * p)))
            lo = max(s, s + n_prefix - max_len)
            w = enc[lo - s: n_prefix]
            L = len(w)
            seqs[p][i, max_len - L:] = w[:, :dim]
            masks[p][i, max_len - L:] = 1.0
            pv = part[lo:e]
            vals, cnt = np.unique(pv, return_counts=True)
            tasks[p][i] = int(vals[np.argmax(cnt)]) - 1
    print(f"[rev12.common] built seq d={dim} prefixes={todo} "
          f"({time.time()-t0:.1f}s)", flush=True)
    for p in todo:
        f = os.path.join(cache_dir(), f'seq_d{dim}_p{p}_L{max_len}.npz')
        np.savez(f, sequences=seqs[p], mask=masks[p], task_ids=tasks[p])
        results[p] = (seqs[p], masks[p], tasks[p])
    return results


def build_space(cache: dict, dim: int, prefix_frac: float, max_len: int = 128,
                cache_file: str = None):
    """构造某输入空间下的 (sequences, mask, task_ids) —— 序列模型输入。"""
    out = build_spaces_multi(cache, dim, [prefix_frac], max_len)
    return out[prefix_frac]


DECONF_CHANNELS = [9, 10, 27]     # part_norm / exercise_norm / part_event_share
DECONF_NAME_PATTERNS = ['part_norm', 'exercise_norm', 'part_event_share']


def deconfound_seq(seq: np.ndarray) -> np.ndarray:
    """去掉"作业身份"通道（part / exercise 归一化）—— 消除 part-prior 捷径。"""
    s = seq.copy()
    for c in DECONF_CHANNELS:
        if c < s.shape[-1]:
            s[:, :, c] = 0.0
    return s


def deconfound_tabular(X: np.ndarray, names):
    """表格孪生的去混淆：剔除与 part / exercise 身份相关的聚合列。"""
    keep = [i for i, n in enumerate(names)
            if not any(p in n for p in DECONF_NAME_PATTERNS)]
    return X[:, keep], [names[i] for i in keep]


def build_all_event_counts(cache: dict):
    """遗留基线：每生全部事件的 7 类事件计数（与既有 RF-7d 0.8584 同口径）。"""
    et, offsets = cache['et'], cache['offsets']
    n = len(offsets) - 1
    X = np.zeros((n, 7), dtype=np.float32)
    for i in range(n):
        s, e = offsets[i], offsets[i + 1]
        if e > s:
            X[i] = np.bincount(et[s:e], minlength=7)
    names = list(CHANNEL_NAMES_7)
    return X, names


def build_tabulars_multi(cache: dict, dim: int, prefixes, max_len: int = 128):
    """构造多个前缀下的学生级表格特征（序列空间的聚合孪生）→ {prefix: (X, names)}"""
    et, ts = cache['et'], cache['ts']
    part, exercise, ttd = cache['part'], cache['exercise'], cache['ttd']
    offsets, student_ids = cache['offsets'], cache['student_ids']
    n_students = len(student_ids)
    ex_max = cache['ex_max']

    names = []
    for c in range(7):
        names += [f'{CHANNEL_NAMES_28[c]}_count', f'{CHANNEL_NAMES_28[c]}_frac']
    for c in range(7, dim):
        names += [f'{CHANNEL_NAMES_28[c]}_mean', f'{CHANNEL_NAMES_28[c]}_std',
                  f'{CHANNEL_NAMES_28[c]}_last']

    results, todo = {}, []
    for p in prefixes:
        f = os.path.join(cache_dir(), f'tab_d{dim}_p{p}_L{max_len}.npz')
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True)
            results[p] = (z['X'], list(z['names']))
        else:
            todo.append(p)
    if not todo:
        return results

    Xs = {p: np.zeros((n_students, len(names)), dtype=np.float32) for p in todo}
    t0 = time.time()
    for i in range(n_students):
        s, e = offsets[i], offsets[i + 1]
        if e <= s:
            continue
        enc = encode_student(et[s:e], ts[s:e], part[s:e], exercise[s:e], ttd[s:e], ex_max)
        for p in todo:
            n_prefix = e - s if p >= 1.0 else max(1, int(round((e - s) * p)))
            lo = max(s, s + n_prefix - max_len)
            w = enc[lo - s: n_prefix]
            L = len(w)
            feats = []
            for c in range(7):
                cnt = float(w[:, c].sum())
                feats += [cnt, cnt / max(L, 1)]
            for c in range(7, dim):
                col = w[:, c]
                feats += [float(col.mean()), float(col.std()), float(col[-1])]
            Xs[p][i] = np.asarray(feats, dtype=np.float32)
    print(f"[rev12.common] built tabular d={dim} prefixes={todo} "
          f"({time.time()-t0:.1f}s)", flush=True)
    for p in todo:
        f = os.path.join(cache_dir(), f'tab_d{dim}_p{p}_L{max_len}.npz')
        np.savez(f, X=Xs[p], names=np.array(names, dtype=object))
        results[p] = (Xs[p], names)
    return results


def build_tabular(cache: dict, dim: int, prefix_frac: float, max_len: int = 128,
                  cache_file: str = None):
    """构造某输入空间下的学生级表格特征（序列空间的聚合孪生）→ (X, names)"""
    out = build_tabulars_multi(cache, dim, [prefix_frac], max_len)
    return out[prefix_frac]


# ─────────────────────────────── 协议与指标 ───────────────────────────────

def make_folds(y: np.ndarray, seeds=(42, 123, 777), n_splits: int = 5):
    """5 折 × 多 seed 的分层划分（学生为单位，天然按学生分组）。"""
    from sklearn.model_selection import StratifiedKFold
    out = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for f, (tr, va) in enumerate(skf.split(np.zeros(len(y)), y)):
            out.append({'seed': int(seed), 'fold': int(f), 'tr': tr, 'va': va})
    return out


def inner_split(y_tr: np.ndarray, val_frac: float = 0.15, seed: int = 0):
    """训练折内部再切一层验证集（分层），用于早停与超参选择（评估折不参与）。"""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y_tr))
    inner_tr, inner_va = train_test_split(
        idx, test_size=val_frac, random_state=seed, stratify=y_tr)
    return inner_tr, inner_va


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """完整指标集（含审稿意见要求的 balanced accuracy）。"""
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 average_precision_score, confusion_matrix,
                                 balanced_accuracy_score, matthews_corrcoef)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_class_1': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'f1_class_0': float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        'mcc': float(matthews_corrcoef(y_true, y_pred)),
        'n': int(len(y_true)),
        'pos_rate_pred': float(y_pred.mean()),
    }
    try:
        out['roc_auc'] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out['roc_auc'] = float('nan')
    try:
        out['pr_auc'] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        out['pr_auc'] = float('nan')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    out['cm'] = {'TN': int(cm[0, 0]), 'FP': int(cm[0, 1]),
                 'FN': int(cm[1, 0]), 'TP': int(cm[1, 1])}
    return out


def save_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def save_fold_csv(path: str, rows: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path
