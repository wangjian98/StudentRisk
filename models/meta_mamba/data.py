"""Event-sequence encoding for Meta-Mamba.

Each student → fixed-length event sequence (L, D_event).
Per-event features (D_event = 11):
  - event_type (7 one-hot)
  - time_interval_log (1, log-scaled seconds since previous event)
  - deadline_dist_norm (1, normalized 0-1)
  - part_id (1, scaled to [0,1])
  - exercise_id_norm (1, normalized exercise number)

Sequence truncation: take the most recent MAX_LEN events per student.
Padding: left-pad with zeros (so model sees the most recent at the end).
"""
import os
import numpy as np
import pandas as pd


EVENT_TYPES = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit',
]
EVENT_TYPE_TO_IDX = {et: i for i, et in enumerate(EVENT_TYPES)}


def build_event_sequences(
    ide_logs: pd.DataFrame,
    student_ids: np.ndarray,
    max_len: int = 256,
):
    """Build (n_students, max_len, 11) event-sequence tensor + per-student task ids.

    Args:
      ide_logs: DataFrame of events (must have columns: student, part, exercise, eventType, timestamp, timeToDeadline)
      student_ids: ordered list of student IDs to include
      max_len: sequence length (most recent events)

    Returns:
      sequences: np.ndarray (n_students, max_len, 11) float32
      mask:      np.ndarray (n_students, max_len)         float32 (1 = real, 0 = pad)
      task_ids:  np.ndarray (n_students,)                int64   (the dominant problem part for that student)
      student_event_count: (n_students,) int64
    """
    print(f"[meta_mamba.data] Building event sequences "
          f"(n_students={len(student_ids)}, max_len={max_len}) ...", flush=True)

    n_students = len(student_ids)
    D_event = 11
    sequences = np.zeros((n_students, max_len, D_event), dtype=np.float32)
    mask = np.zeros((n_students, max_len), dtype=np.float32)
    task_ids = np.zeros(n_students, dtype=np.int64)
    event_counts = np.zeros(n_students, dtype=np.int64)

    grouped = ide_logs.groupby('student', sort=False)
    n_parts_max = int(ide_logs['part'].max()) if 'part' in ide_logs.columns else 7
    n_exercise_max = int(ide_logs['exercise'].max()) if 'exercise' in ide_logs.columns else 1

    for i, sid in enumerate(student_ids):
        if sid not in grouped.groups:
            continue
        ev = grouped.get_group(sid).sort_values('timestamp').reset_index(drop=True)
        n_ev = len(ev)
        event_counts[i] = n_ev

        # Determine the dominant part for this student (most frequent part_id)
        if 'part' in ev.columns:
            task_ids[i] = int(ev['part'].mode().iloc) - 1  # 0-indexed
        else:
            task_ids[i] = 0

        # Truncate to the most recent max_len events
        ev = ev.tail(max_len).reset_index(drop=True)
        L = len(ev)

        # Parse events
        timestamps = ev['timestamp'].values  # datetime64
        # time_interval: seconds since previous event (log-normalized)
        if L > 1:
            intervals_sec = (np.diff(timestamps).astype('timedelta64[s]').astype(float))
            intervals_sec = np.maximum(intervals_sec, 1.0)  # avoid log(0)
            log_intervals = np.log1p(intervals_sec) / 10.0  # normalize roughly to [0, 1]
            log_intervals = log_intervals.clip(0, 1).astype(np.float32)
        else:
            log_intervals = np.array([], dtype=np.float32)
        # prepend (log_interval=0 for first event)
        if L > 1:
            log_intervals = np.concatenate([[0.0], log_intervals])
        else:
            log_intervals = np.array([0.0], dtype=np.float32)

        # Event type one-hot (7-dim)
        et_ids = ev['eventType'].map(EVENT_TYPE_TO_IDX).fillna(0).astype(int).values
        et_oh = np.zeros((L, 7), dtype=np.float32)
        et_oh[np.arange(L), et_ids] = 1.0

        # Deadline distance (normalized): we have timeToDeadline in seconds (positive = time remaining)
        deadline_sec = ev['timeToDeadline'].astype(float).values
        # log-normalize
        deadline_norm = np.log1p(np.maximum(deadline_sec, 0)) / 20.0  # rough normalize
        deadline_norm = deadline_norm.clip(0, 1).astype(np.float32)

        # Part / exercise (normalized)
        part_norm = (ev['part'].astype(float).values - 1.0) / max(n_parts_max - 1, 1)
        part_norm = part_norm.clip(0, 1).astype(np.float32)
        ex_norm = (ev['exercise'].astype(float).values - 1.0) / max(n_exercise_max - 1, 1)
        ex_norm = ex_norm.clip(0, 1).astype(np.float32)

        # Combine: (L, 11) = [et_oh(7) | log_interval(1) | deadline(1) | part(1) | exercise(1)]
        feat = np.concatenate([
            et_oh,
            log_intervals.reshape(-1, 1),
            deadline_norm.reshape(-1, 1),
            part_norm.reshape(-1, 1),
            ex_norm.reshape(-1, 1),
        ], axis=1).astype(np.float32)

        # Left-pad: put events at the END of the sequence
        sequences[i, max_len - L:] = feat
        mask[i, max_len - L:] = 1.0

    n_tasks = int(task_ids.max()) + 1 if len(task_ids) > 0 else 1
    print(f"[meta_mamba.data] Built: sequences={sequences.shape}, mask_sum={mask.sum():.0f}, "
          f"n_tasks={n_tasks}, mean events/student={event_counts.mean():.0f}, "
          f"max events/student={event_counts.max()}", flush=True)
    return sequences, mask, task_ids, event_counts