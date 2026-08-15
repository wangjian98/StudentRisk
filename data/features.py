"""46-dim feature engineering for CS1 student behavior.

Feature composition (46 dims):
  1. Event-base statistics (28d): 7 event types × 4 statistics (mean/std/cv/shannon_entropy)
  2. Behavioral trajectory (10d): improvement, consistency, trend, mean/std/min/max/median/iqr interval, duration_per_event
  3. Emotion compound (6d): edit_ratio_mean/std, delete_ratio_mean/std, focus_ratio_mean/std
  4. Meta info (2d): num_problems, total_events
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy


EVENT_TYPES = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit',
]


def _safe_float(val, default=0.0):
    """Convert to finite float, fallback to default if NaN/inf."""
    try:
        v = float(val)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _per_event_stats(events_for_student: pd.DataFrame, total_duration_sec: float):
    """Compute 4 statistics per event type for a single student.

    Stats: mean interval (sec), std interval, coefficient of variation, Shannon entropy
    """
    out = {}
    for et in EVENT_TYPES:
        ts = events_for_student.loc[
            events_for_student['eventType'] == et, 'timestamp'
        ].sort_values().values
        n = len(ts)
        if n == 0:
            mean_v = std_v = cv_v = ent_v = 0.0
        else:
            if n >= 2:
                intervals = np.diff(ts).astype('timedelta64[s]').astype(float)
                mean_v = _safe_float(intervals.mean())
                std_v  = _safe_float(intervals.std())
                cv_v   = std_v / (mean_v + 1e-9) if mean_v > 0 else 0.0
            else:
                mean_v = total_duration_sec
                std_v  = 0.0
                cv_v   = 0.0
            ent_v = _safe_float(shannon_entropy(np.ones(n) / n)) if n > 0 else 0.0
        out[f'{et}_mean'] = mean_v
        out[f'{et}_std']  = std_v
        out[f'{et}_cv']   = cv_v
        out[f'{et}_ent']  = ent_v
    return out


def _trajectory_stats(events_for_student: pd.DataFrame, total_duration_sec: float):
    """Compute 10 trajectory / consistency / interval features."""
    out = {}
    # Aggregate over all events
    all_ts = events_for_student['timestamp'].sort_values().values
    n_total = len(all_ts)
    if n_total < 2:
        intervals = np.array([total_duration_sec])
    else:
        intervals = np.diff(all_ts).astype('timedelta64[s]').astype(float)
        intervals = intervals[np.isfinite(intervals)]

    n_intervals = len(intervals)

    # improvement: improvement/(improvement+regression)
    n_insert = int((events_for_student['eventType'] == 'text_insert').sum())
    n_remove = int((events_for_student['eventType'] == 'text_remove').sum())
    improvement = n_insert / (n_insert + n_remove + 1e-9)

    # consistency: 1 - std_interval/mean_interval (CV-based)
    if n_intervals > 0 and intervals.mean() > 0:
        consistency = 1.0 - (intervals.std() / (intervals.mean() + 1e-9))
    else:
        consistency = 0.0

    # trend: linear slope of timestamps (in days)
    if n_total >= 2:
        try:
            t_d = (all_ts - all_ts[0]).astype('timedelta64[s]').astype(float)
            x = np.arange(len(t_d))
            slope = np.polyfit(x, t_d, 1)[0] if len(x) >= 2 else 0.0
        except Exception:
            slope = 0.0
    else:
        slope = 0.0

    out['improvement']    = float(improvement)
    out['consistency']    = float(_safe_float(consistency))
    out['trend']          = float(_safe_float(slope))
    out['mean_interval']  = float(_safe_float(intervals.mean() if n_intervals > 0 else total_duration_sec))
    out['std_interval']   = float(_safe_float(intervals.std()  if n_intervals > 0 else 0.0))
    out['min_interval']   = float(_safe_float(intervals.min()  if n_intervals > 0 else total_duration_sec))
    out['max_interval']   = float(_safe_float(intervals.max()  if n_intervals > 0 else total_duration_sec))
    out['duration_per_event'] = float(_safe_float(total_duration_sec / max(n_total, 1)))
    out['median_interval'] = float(_safe_float(np.median(intervals) if n_intervals > 0 else total_duration_sec))
    out['iqr_interval']    = float(_safe_float(np.percentile(intervals, 75) - np.percentile(intervals, 25)
                                                if n_intervals > 0 else 0.0))
    return out


def _emotion_ratios(events_for_student: pd.DataFrame):
    """Compute emotion compound features (6 dims).

    Ratios:
      edit_ratio = (text_insert + text_remove + text_paste) / total_events
      delete_ratio = text_remove / total_events
      focus_ratio = (focus_gained - focus_lost) / total_events

    Each ratio: mean and std computed across exercise parts.
    """
    out = {}
    if len(events_for_student) == 0:
        for k in ['edit_ratio_mean', 'edit_ratio_std',
                  'delete_ratio_mean', 'delete_ratio_std',
                  'focus_ratio_mean',  'focus_ratio_std']:
            out[k] = 0.0
        return out

    # Compute per-part ratios, then take mean/std
    parts = events_for_student['part'].unique()
    edit_ratios, del_ratios, focus_ratios = [], [], []
    for p in parts:
        sub = events_for_student[events_for_student['part'] == p]
        n_total = len(sub)
        if n_total == 0:
            continue
        n_ins  = (sub['eventType'] == 'text_insert').sum()
        n_rm   = (sub['eventType'] == 'text_remove').sum()
        n_pas  = (sub['eventType'] == 'text_paste').sum()
        n_gain = (sub['eventType'] == 'focus_gained').sum()
        n_loss = (sub['eventType'] == 'focus_lost').sum()
        edit_ratios.append((n_ins + n_rm + n_pas) / n_total)
        del_ratios.append(n_rm / n_total)
        focus_ratios.append((n_gain - n_loss) / n_total)

    out['edit_ratio_mean']   = float(_safe_float(np.mean(edit_ratios)   if edit_ratios   else 0.0))
    out['edit_ratio_std']    = float(_safe_float(np.std(edit_ratios)    if edit_ratios   else 0.0))
    out['delete_ratio_mean'] = float(_safe_float(np.mean(del_ratios)    if del_ratios    else 0.0))
    out['delete_ratio_std']  = float(_safe_float(np.std(del_ratios)     if del_ratios    else 0.0))
    out['focus_ratio_mean']  = float(_safe_float(np.mean(focus_ratios)  if focus_ratios  else 0.0))
    out['focus_ratio_std']   = float(_safe_float(np.std(focus_ratios)   if focus_ratios  else 0.0))
    return out


def _meta_info(events_for_student: pd.DataFrame):
    out = {}
    out['num_problems']  = int(events_for_student['exercise'].nunique())
    out['total_events']  = int(len(events_for_student))
    return out


def build_features(ide_logs: pd.DataFrame, student_ids: np.ndarray) -> np.ndarray:
    """Build 46-dim feature matrix for each student.

    Args:
      ide_logs: DataFrame of all events (must have columns: student, part, exercise, eventType, timestamp)
      student_ids: array of student IDs to include (ordered)

    Returns:
      X: np.ndarray of shape (n_students, 46)
    """
    print(f"[features] Building 46-dim features for {len(student_ids)} students ...", flush=True)

    rows = []
    grouped = ide_logs.groupby('student', sort=False)

    for sid in student_ids:
        if sid not in grouped.groups:
            # student with no events → zero vector
            row = {f'f{i}': 0.0 for i in range(46)}
            rows.append(row)
            continue

        ev = grouped.get_group(sid)
        if len(ev) == 0:
            row = {f'f{i}': 0.0 for i in range(46)}
            rows.append(row)
            continue

        # total duration in seconds
        ts = ev['timestamp'].values
        if len(ts) >= 2:
            duration = (ts[-1] - ts[0]).astype('timedelta64[s]').astype(float)
            if not np.isfinite(duration) or duration <= 0:
                duration = 1.0
        else:
            duration = 1.0

        feat = {}
        feat.update(_per_event_stats(ev, duration))   # 28 dims
        feat.update(_trajectory_stats(ev, duration))  # 10 dims
        feat.update(_emotion_ratios(ev))             # 6 dims
        feat.update(_meta_info(ev))                  # 2 dims
        rows.append(feat)

    df = pd.DataFrame(rows)

    # Sanitize: NaN/inf → 0
    df = df.fillna(0).replace([np.inf, -np.inf], 0)

    assert df.shape[1] == 46, f"Expected 46 features, got {df.shape[1]}"
    print(f"[features] Built feature matrix: {df.shape}", flush=True)
    return df.values.astype(np.float32), list(df.columns)