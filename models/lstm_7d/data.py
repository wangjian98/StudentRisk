"""7-dim event sequence builder: per-student one-hot event-type sequence."""
import numpy as np
import pandas as pd


# Same 7 event types as RF-7d (for fair 7-dim comparison)
EVENT_TYPES_7 = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit',
]
EVENT_TYPE_TO_IDX = {et: i for i, et in enumerate(EVENT_TYPES_7)}


def build_event_sequences_7d(
    ide_logs: pd.DataFrame,
    student_ids: np.ndarray,
    max_len: int = 128,
):
    """Build (n_students, max_len, 7) event sequences using ONLY event-type one-hot.

    NO 46-dim aggregate features, NO continuous features (time interval, deadline, etc.).
    Only the 7-dim event type information per event.

    Returns:
      sequences: (n_students, max_len, 7) float32 — event-type one-hot
      mask:      (n_students, max_len) float32 — 1=real, 0=pad
      task_ids:  (n_students,) int64 — problem part (for fair comparison with MetaMamba)
    """
    print(f"[lstm_7d.data] Building 7-dim event sequences "
          f"(n_students={len(student_ids)}, max_len={max_len}) ...", flush=True)

    n_students = len(student_ids)
    sequences = np.zeros((n_students, max_len, 7), dtype=np.float32)
    mask = np.zeros((n_students, max_len), dtype=np.float32)
    task_ids = np.zeros(n_students, dtype=np.int64)

    grouped = ide_logs.groupby('student', sort=False)

    for i, sid in enumerate(student_ids):
        if sid not in grouped.groups:
            continue
        ev = grouped.get_group(sid).sort_values('timestamp').reset_index(drop=True)

        # Determine task_id from dominant problem part
        if 'part' in ev.columns:
            task_ids[i] = int(ev['part'].mode().iloc[0]) - 1
        else:
            task_ids[i] = 0

        # Take most recent max_len events
        ev = ev.tail(max_len).reset_index(drop=True)
        L = len(ev)

        # Encode each event as 7-dim one-hot
        et_ids = ev['eventType'].map(EVENT_TYPE_TO_IDX).fillna(0).astype(int).values
        et_oh = np.zeros((L, 7), dtype=np.float32)
        et_oh[np.arange(L), et_ids] = 1.0

        # Left-pad
        sequences[i, max_len - L:] = et_oh
        mask[i, max_len - L:] = 1.0

    n_tasks = int(task_ids.max()) + 1 if len(task_ids) > 0 else 1
    print(f"[lstm_7d.data] Built: sequences={sequences.shape}, n_tasks={n_tasks}, "
          f"mask_sum={mask.sum():.0f}", flush=True)
    return sequences, mask, task_ids