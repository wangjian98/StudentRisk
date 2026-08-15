"""7-dim raw feature builder: per-student count per event type."""
import numpy as np
import pandas as pd


# 7 event types from CS1 dataset (matches CodeEMO's EVENT_TYPES)
EVENT_TYPES_7 = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit',
]


def build_7dim_features(ide_logs: pd.DataFrame, student_ids: np.ndarray) -> np.ndarray:
    """Compute 7-dim feature: each student's count per event type.

    Args:
      ide_logs: DataFrame of all events (must have columns: student, eventType)
      student_ids: ordered list of student IDs to include

    Returns:
      X: np.ndarray of shape (n_students, 7) float32
         Column order matches EVENT_TYPES_7
    """
    n_students = len(student_ids)
    print("[rf7.data] Building 7-dim features for {} students ...".format(n_students), flush=True)

    # Count events per (student, event_type)
    counts = (ide_logs
              .groupby(['student', 'eventType'])
              .size()
              .unstack(fill_value=0))
    # Ensure all event types present
    counts = counts.reindex(columns=EVENT_TYPES_7, fill_value=0)
    # Ensure all students present (in correct order)
    counts = counts.reindex(index=student_ids, fill_value=0)

    X = counts.values.astype(np.float32)
    feature_names = list(counts.columns)
    print(f"[rf7.data] Built: X.shape={X.shape}, feature_names={feature_names}", flush=True)
    print(f"[rf7.data] Stats: min={X.min():.0f}, max={X.max():.0f}, mean={X.mean():.1f}", flush=True)
    return X, feature_names