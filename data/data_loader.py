"""Data loader for CS1 student risk dataset.

Conventions:
  - y=1 represents FAILED (挂科)
  - y=0 represents PASSED (通过)
  - loaded from passed.csv where passed=True means student passed (→ y=0)

Source dataset:
  - IDE_logs.csv: 28.5M rows, columns: student, part, exercise, eventType, timestamp, timeToDeadline
  - passed.csv: 473 students with passed=True/False
"""
import os
import numpy as np
import pandas as pd


# 7 event types as defined in the CS1 dataset
EVENT_TYPES = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit',
]


def load_ide_logs(path: str) -> pd.DataFrame:
    """Load IDE event logs from CSV.

    Expected columns: student, part, exercise, eventType, timestamp, timeToDeadline
    """
    print(f"[data_loader] Loading IDE logs from {path} ...", flush=True)
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"[data_loader]   shape={df.shape}, n_students={df['student'].nunique()}, "
          f"n_events={len(df)}, event_types={df['eventType'].unique().tolist()}", flush=True)
    return df


def load_passed_labels(path: str) -> pd.DataFrame:
    """Load passed/failed labels from CSV.

    Returns DataFrame with columns: student, passed (bool).
    """
    print(f"[data_loader] Loading passed labels from {path} ...", flush=True)
    df = pd.read_csv(path)
    print(f"[data_loader]   shape={df.shape}, "
          f"passed=True={int((df['passed']==True).sum())}, "
          f"passed=False={int((df['passed']==False).sum())}", flush=True)
    return df


def to_failed_label(passed_series: pd.Series) -> np.ndarray:
    """Convert passed (True/False) → failed (1/0) label.

    Convention: Failed=1, Passed=0
    """
    return (~passed_series.astype(bool)).astype(int).values


def load_dataset(ide_logs_path: str = None, passed_path: str = None):
    """High-level: load both IDE logs and labels, return standardized structures.

    Returns:
      ide_logs: DataFrame of all events
      labels_df: DataFrame with columns [student, failed] (failed=1 convention)
      y: np.ndarray of failed labels, ordered by student ID
      student_ids: np.ndarray of student IDs (sorted)
    """
    if ide_logs_path is None:
        ide_logs_path = os.environ.get(
            'STUDENTRISK_IDE_LOGS', '/home/ubuntu/IDE_logs/IDE_logs.csv')
    if passed_path is None:
        passed_path = os.environ.get(
            'STUDENTRISK_PASSED', '/home/ubuntu/IDE_logs/passed.csv')

    ide_logs = load_ide_logs(ide_logs_path)
    passed_df = load_passed_labels(passed_path)

    # Build labels DataFrame with failed=1 convention
    labels_df = pd.DataFrame({
        'student': passed_df['student'].astype(int).values,
        'passed':  passed_df['passed'].astype(bool).values,
    })
    labels_df['failed'] = (labels_df['passed'] == False).astype(int)

    # Sort by student for consistent ordering
    labels_df = labels_df.sort_values('student').reset_index(drop=True)
    student_ids = labels_df['student'].values
    y = labels_df['failed'].values

    n_total = len(y)
    n_failed = int(y.sum())
    n_passed = int((1 - y).sum())
    fail_rate = y.mean()

    print(f"[data_loader] Final label distribution (Failed=1 convention):", flush=True)
    print(f"             n_total={n_total}, n_failed={n_failed} ({fail_rate:.4f}), "
          f"n_passed={n_passed} ({1-fail_rate:.4f})", flush=True)

    return ide_logs, labels_df, y, student_ids