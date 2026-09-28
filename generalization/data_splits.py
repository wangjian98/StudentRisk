"""Data split utilities for StudentRisk generalization experiments.

Five split types implemented for the v3.1 paper:

  EXP-1: learning_curve_split()  — stratified subsampling at varying training-set ratios
  EXP-2: temporal_split()        — chronological train/test split by last-event timestamp
  EXP-3: lopo_split()            — Leave-One-Problem-Part-Out
  EXP-4: perturbation_split()    — apply a robustness perturbation to IDE event log
  helper: stratified_subsample() — stratified subsample utility

All splits return a `(train_idx, test_idx, meta)` triple, aligned with the
``sklearn.model_selection.KFold.split()`` interface so the existing OOF pipeline
can consume them transparently.

Conventions:
  - y is the full label vector (failed=1, passed=0), aligned with ``y[i]``
    referring to student at the same row in ``labels_df``.
  - ``train_idx`` / ``test_idx`` are ``np.ndarray`` of int indices into ``y``.
  - ``meta`` is a plain dict carrying split-specific metadata.

The module also runs a self-test when executed as a script
(``python -m generalization.data_splits``) using either the real CS1 dataset or
synthesized fallback data (because the IDE_logs CSV is too large to live in the
project repo).
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

# Project path bootstrap so this module is runnable both as a package and as
# a standalone script (``python generalization/data_splits.py``).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ────────────────────────────── helpers ──────────────────────────────


def stratified_subsample(y: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified subsample: take ``n`` indices preserving class proportions.

    Args:
      y: label vector (n_total,)
      n: desired total subsample size (clipped to len(y))
      rng: numpy random Generator

    Returns:
      np.ndarray of selected indices into ``y`` (shape: (n_actual,))
    """
    y = np.asarray(y)
    n_total = len(y)
    n = max(1, min(int(n), n_total))
    classes, counts = np.unique(y, return_counts=True)
    # proportional allocation per class
    props = counts / counts.sum()
    alloc = np.maximum(1, np.round(props * n).astype(int))  # ensure ≥1 per present class
    # if rounding pushes us over n, trim the largest classes first
    while alloc.sum() > n:
        biggest = int(np.argmax(alloc))
        if alloc[biggest] > 1:
            alloc[biggest] -= 1
        else:
            break
    selected: List[int] = []
    for cls, n_cls in zip(classes, alloc):
        cls_idx = np.where(y == cls)[0]
        n_cls = min(int(n_cls), len(cls_idx))
        if n_cls <= 0:
            continue
        chosen = rng.choice(cls_idx, size=n_cls, replace=False)
        selected.extend(chosen.tolist())
    selected = np.array(sorted(selected), dtype=np.int64)
    return selected


# ─────────────────────── EXP-1: learning curve ──────────────────────


def learning_curve_split(
    y: np.ndarray,
    train_idx: np.ndarray,
    ratios: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]:
    """Stratified subsampling at multiple training-set sizes.

    For each ratio ``r`` in ``ratios``, take ``round(r * len(train_idx))``
    stratified samples from ``train_idx`` to form ``sub_train_idx``.
    The returned second element (``test_idx``) is the **original** ``train_idx``
    — this keeps the contract that ``len(sub_train_idx) ⊆ train_idx`` and
    consumers can use ``(sub_train_idx, train_idx, meta)`` as a drop-in for
    sklearn's ``split()`` interface.

    Args:
      y: full label vector
      train_idx: indices of the training pool (typically a single OOF fold)
      ratios: tuple of fractions in (0, 1]
      seed: RNG seed

    Returns:
      List of (sub_train_idx, original_train_idx, {'ratio': r, 'n_train': k})
    """
    rng = np.random.default_rng(seed)
    train_idx = np.asarray(train_idx, dtype=np.int64)
    y_sub = y[train_idx]
    n_pool = len(train_idx)
    out: List[Tuple[np.ndarray, np.ndarray, Dict[str, float]]] = []
    for r in ratios:
        n_take = max(2, int(round(r * n_pool)))
        sub = stratified_subsample(y_sub, n_take, rng)
        sub_idx = train_idx[sub]
        out.append((sub_idx, train_idx, {'ratio': float(r), 'n_train': int(len(sub_idx))}))
    return out


# ───────────────────── EXP-2: temporal split ────────────────────────


def temporal_split(
    ide_logs_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    split_pct: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Chronological split by each student's last event timestamp.

    Students are ordered by ``max(timestamp)`` of their events; the first
    ``split_pct`` of students (chronologically earliest) become the training
    set, the remaining ``1 - split_pct`` become the test set.

    Args:
      ide_logs_df: full event log (must have columns: student, timestamp)
      labels_df: labels DataFrame (must have column: student, sorted asc)
      split_pct: fraction of students for the training set (default 0.8)

    Returns:
      (train_idx, test_idx, meta) where meta contains:
        - 'method': 'temporal_80_20' (or similar)
        - 'cutoff': the median last-timestamp value
        - 'n_train', 'n_test'
        - 'split_pct'
    """
    assert 'student' in ide_logs_df.columns, 'ide_logs_df must have column: student'
    assert 'timestamp' in ide_logs_df.columns, 'ide_logs_df must have column: timestamp'
    assert 'student' in labels_df.columns, 'labels_df must have column: student'

    student_ids = labels_df['student'].values
    sid_to_pos = {int(s): i for i, s in enumerate(student_ids)}

    # Last timestamp per student
    last_ts = ide_logs_df.groupby('student')['timestamp'].max()
    # Order students by last_ts ascending (chronological)
    ordered_sids = last_ts.sort_values(ascending=True).index.values
    n_total = len(ordered_sids)
    n_train = int(round(split_pct * n_total))
    n_train = max(1, min(n_train, n_total - 1))  # ensure both halves non-empty

    train_sids = ordered_sids[:n_train]
    test_sids = ordered_sids[n_train:]

    train_idx = np.array([sid_to_pos[int(s)] for s in train_sids if int(s) in sid_to_pos],
                         dtype=np.int64)
    test_idx = np.array([sid_to_pos[int(s)] for s in test_sids if int(s) in sid_to_pos],
                        dtype=np.int64)

    cutoff_ts = last_ts.sort_values(ascending=True).iloc[n_train - 1] if n_train > 0 else None
    meta = {
        'method': f'temporal_{int(split_pct * 100)}_{int((1 - split_pct) * 100)}',
        'cutoff': str(cutoff_ts) if cutoff_ts is not None else None,
        'split_pct': float(split_pct),
        'n_train': int(len(train_idx)),
        'n_test':  int(len(test_idx)),
    }
    return train_idx, test_idx, meta


# ───────────────────── EXP-3: leave-one-part-out ────────────────────


def lopo_split(
    labels_df: pd.DataFrame,
    ide_logs_df: pd.DataFrame,
    n_parts: int = 7,
) -> List[Tuple[np.ndarray, np.ndarray, Dict[str, object]]]:
    """Leave-One-Problem-Part-Out cross-validation.

    For each problem part ``p ∈ [0, n_parts)``:
      - compute each student's *dominant* part = argmax of (count of events in part).
      - students with dominant part == p → test set (one per part).
      - students with dominant part != p → train set.
    A warning is printed if a part has < 20 students.

    Args:
      labels_df: DataFrame (must have column: student)
      ide_logs_df: DataFrame (must have columns: student, part)
      n_parts: number of problem parts (default 7)

    Returns:
      List of (train_idx, test_idx, meta) where meta contains:
        - 'left_out_part': int (0-indexed)
        - 'n_test': int
        - 'n_train': int
        - 'method': 'lopo'
    """
    assert 'student' in ide_logs_df.columns, 'ide_logs_df must have column: student'
    assert 'part' in ide_logs_df.columns, 'ide_logs_df must have column: part'

    student_ids = labels_df['student'].values
    sid_to_pos = {int(s): i for i, s in enumerate(student_ids)}

    # Count events per (student, part) and compute dominant part
    grouped = ide_logs_df.groupby(['student', 'part']).size().unstack(fill_value=0)
    # Ensure all parts present
    for p in range(1, n_parts + 1):  # part is 1-indexed in the raw data
        if p not in grouped.columns:
            grouped[p] = 0
    # dominant part is the column with max count, then -1 to make 0-indexed
    dominant_part_1idx = grouped.idxmax(axis=1)            # 1..n_parts
    dominant_part_0idx = (dominant_part_1idx - 1).astype(int)
    sid_to_dominant = {int(s): int(p) for s, p in dominant_part_0idx.items()}

    # Build per-part test index lists
    out: List[Tuple[np.ndarray, np.ndarray, Dict[str, object]]] = []
    for p in range(n_parts):
        test_sids = [s for s, dp in sid_to_dominant.items() if dp == p and int(s) in sid_to_pos]
        train_sids = [s for s, dp in sid_to_dominant.items() if dp != p and int(s) in sid_to_pos]
        test_idx = np.array([sid_to_pos[int(s)] for s in test_sids], dtype=np.int64)
        train_idx = np.array([sid_to_pos[int(s)] for s in train_sids], dtype=np.int64)
        if len(test_idx) < 20:
            warnings.warn(
                f"[lopo_split] Part {p} has only {len(test_idx)} students (<20). "
                f"Keeping in split but result may be unreliable.",
                RuntimeWarning,
            )
        meta = {
            'method': 'lopo',
            'left_out_part': int(p),
            'n_test': int(len(test_idx)),
            'n_train': int(len(train_idx)),
        }
        out.append((train_idx, test_idx, meta))
    return out


# ───────────────────── EXP-4: perturbation split ────────────────────


def perturbation_split(
    ide_logs_df: pd.DataFrame,
    labels_df: pd.DataFrame,            # noqa: ARG001 — kept for interface parity
    perturbation: str = 'event_mask',
    severity: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Apply a perturbation to the IDE event log.

    Four perturbation types:

      event_mask — randomly drop ``severity`` fraction of events per student.
      time_noise — add log-normal multiplicative noise to event timestamps.
      type_swap  — randomly re-label ``severity`` fraction of events with a
                     different (random) event type.
      truncate   — keep only the most recent ``1 - severity`` fraction of
                   events per student.

    Args:
      ide_logs_df: source event log
      labels_df:   labels (unused here, kept for symmetry with other splits)
      perturbation: one of {'event_mask', 'time_noise', 'type_swap', 'truncate'}
      severity: fraction in (0, 1) controlling the perturbation strength.
      seed: RNG seed.

    Returns:
      (perturbed_ide_logs_df, meta) where meta contains the parameters used.
    """
    assert perturbation in {'event_mask', 'time_noise', 'type_swap', 'truncate'}, \
        f"Unknown perturbation={perturbation!r}"
    rng = np.random.default_rng(seed)
    df = ide_logs_df.copy()
    n_orig = len(df)

    if perturbation == 'event_mask':
        # drop events per-student with probability = severity
        keep_mask = np.ones(len(df), dtype=bool)
        # do it per student for fairness
        for sid, grp_idx in df.groupby('student').groups.items():
            n = len(grp_idx)
            n_drop = int(round(severity * n))
            if n_drop > 0:
                drop_pos = rng.choice(n, size=n_drop, replace=False)
                grp_idx_arr = grp_idx.values
                keep_mask[grp_idx_arr[drop_pos]] = False
        df = df[keep_mask].reset_index(drop=True)

    elif perturbation == 'time_noise':
        # add log-normal multiplicative noise to timestamp
        # noise_factor ~ LogNormal(0, severity) (severity acts as scale)
        noise = rng.lognormal(mean=0.0, sigma=max(severity, 1e-3), size=len(df))
        new_ts = pd.to_datetime(df['timestamp'].values).astype('int64')  # ns
        new_ts = (new_ts * noise).astype('int64')
        df['timestamp'] = pd.to_datetime(new_ts)

    elif perturbation == 'type_swap':
        # randomly re-label some events with a different (random) event type
        event_types = [
            'text_insert', 'text_remove', 'text_paste',
            'focus_gained', 'focus_lost', 'run', 'submit',
        ]
        n_swap = int(round(severity * len(df)))
        if n_swap > 0:
            swap_pos = rng.choice(len(df), size=n_swap, replace=False)
            cur = df['eventType'].values
            new_types = rng.choice(event_types, size=n_swap)
            # ensure new type != old type
            for k in range(n_swap):
                i = int(swap_pos[k])
                if new_types[k] == cur[i]:
                    new_types[k] = event_types[(event_types.index(cur[i]) + 1) % len(event_types)]
            cur = cur.copy()
            cur[swap_pos] = new_types
            df['eventType'] = cur

    elif perturbation == 'truncate':
        # keep only most recent (1-severity) fraction of each student's events
        keep_mask = np.zeros(len(df), dtype=bool)
        for sid, grp_idx in df.groupby('student').groups.items():
            n = len(grp_idx)
            n_keep = max(1, int(round((1.0 - severity) * n)))
            # take the last n_keep events (most recent)
            grp_idx_sorted = df.loc[grp_idx, 'timestamp'].sort_values().index.values
            keep_idx = grp_idx_sorted[-n_keep:]
            keep_mask[keep_idx] = True
        df = df[keep_mask].reset_index(drop=True)

    meta = {
        'method': 'perturbation',
        'perturbation': perturbation,
        'severity': float(severity),
        'n_events_orig': int(n_orig),
        'n_events_after': int(len(df)),
        'seed': int(seed),
    }
    return df, meta


# ─────────────────── synthetic data fallback (self-test) ───────────


EVENT_TYPES_SYNTH = [
    'text_insert', 'text_remove', 'text_paste',
    'focus_gained', 'focus_lost', 'run', 'submit',
]


def _make_synthetic_dataset(n_students: int = 473, fail_rate: float = 0.66,
                             n_parts: int = 7, avg_events_per_student: int = 2000,
                             seed: int = 42):
    """Build a small synthetic IDE_logs + labels dataframe that mirrors the real
    CS1 schema. Used only when the real IDE_logs.csv is unavailable.

    Args:
      n_students: number of students
      fail_rate:  fraction of failed students
      n_parts:    number of problem parts
      avg_events_per_student: rough average event count per student
      seed: RNG seed

    Returns:
      (ide_logs_df, labels_df, y, student_ids)
    """
    rng = np.random.default_rng(seed)
    student_ids = np.arange(n_students, dtype=np.int64)
    failed = (rng.random(n_students) < fail_rate).astype(int)
    labels_df = pd.DataFrame({
        'student': student_ids,
        'passed':  (1 - failed).astype(bool),
        'failed':  failed.astype(int),
    }).sort_values('student').reset_index(drop=True)

    rows = []
    for sid, fl in zip(student_ids, failed):
        n_ev = max(50, int(rng.poisson(avg_events_per_student)))
        # failed students tend to have more churn (more text_remove, less submit)
        if fl == 1:
            p_remove = 0.30
            p_insert = 0.45
            p_submit = 0.05
        else:
            p_remove = 0.20
            p_insert = 0.50
            p_submit = 0.10
        p_focus = (1.0 - p_remove - p_insert - p_submit) / 4
        probs = np.array([p_insert, p_remove, 0.05, p_focus, p_focus, 0.05, p_submit])
        probs /= probs.sum()
        et = rng.choice(EVENT_TYPES_SYNTH, size=n_ev, p=probs)
        # dominant part for LOPO test
        dom = int(rng.integers(0, n_parts))
        # part is 1-indexed in raw data
        parts = rng.choice(np.arange(1, n_parts + 1), size=n_ev, p=np.full(n_parts, 1.0 / n_parts))
        # make dom part slightly more frequent for this student
        if rng.random() < 0.6:
            n_dom = max(1, int(0.4 * n_ev))
            parts[:n_dom] = dom + 1
            rng.shuffle(parts)
        exercises = rng.integers(1, 20, size=n_ev)
        # timestamps monotonically increasing per student
        base = pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(sid) % 90)
        deltas = np.cumsum(rng.exponential(60.0, size=n_ev))  # seconds
        ts = pd.to_datetime([base.value] * n_ev) + pd.to_timedelta(deltas, unit='s')
        time_to_deadline = rng.integers(0, 30 * 24 * 3600, size=n_ev)
        for k in range(n_ev):
            rows.append({
                'student': int(sid),
                'part': int(parts[k]),
                'exercise': int(exercises[k]),
                'eventType': str(et[k]),
                'timestamp': ts[k],
                'timeToDeadline': int(time_to_deadline[k]),
            })
    ide_logs = pd.DataFrame(rows)
    y = labels_df['failed'].values
    student_ids = labels_df['student'].values
    return ide_logs, labels_df, y, student_ids


def _load_or_synthesize():
    """Try the real CS1 dataset; fall back to small synthetic data.

    Returns:
      (ide_logs_df, labels_df, y, student_ids, is_synthetic: bool)
    """
    try:
        from data import load_dataset
        ide_logs, labels_df, y, student_ids = load_dataset()
        return ide_logs, labels_df, y, student_ids, False
    except Exception as e:
        warnings.warn(f"[data_splits] Real dataset unavailable ({type(e).__name__}: {e}). "
                      f"Falling back to SYNTHETIC data for self-test.", RuntimeWarning)
        ide_logs, labels_df, y, student_ids = _make_synthetic_dataset()
        return ide_logs, labels_df, y, student_ids, True


# ─────────────────────────── self test ─────────────────────────────


def _self_test():
    """End-to-end self-test: load (or synthesize) data, exercise all 5 split fns."""
    print("\n[data_splits] === SELF-TEST START ===")
    ide_logs, labels_df, y, student_ids, is_synth = _load_or_synthesize()
    print(f"[data_splits] data: n_students={len(student_ids)}, "
          f"n_events={len(ide_logs)}, fail_rate={y.mean():.3f}, "
          f"synthetic={is_synth}")

    # 1. learning_curve_split
    print("\n[data_splits] 1) learning_curve_split")
    fake_train_idx = np.arange(len(y), dtype=np.int64)  # use all as pool
    splits = learning_curve_split(y, fake_train_idx, ratios=(0.25, 0.50, 0.75, 1.00), seed=42)
    for sub_idx, full_idx, meta in splits:
        y_sub = y[sub_idx]
        n_total = len(sub_idx)
        n_fail = int(y_sub.sum())
        print(f"   ratio={meta['ratio']:.2f}: n_train={n_total}, fail_rate={y_sub.mean():.3f} "
              f"(n_fail={n_fail}/{n_total})")

    # 2. temporal_split
    print("\n[data_splits] 2) temporal_split")
    tr, te, meta = temporal_split(ide_logs, labels_df, split_pct=0.8)
    print(f"   meta={meta}")
    print(f"   y_train fail_rate={y[tr].mean():.3f}, y_test fail_rate={y[te].mean():.3f}")
    assert len(tr) + len(te) == len(y), 'temporal split should cover all students'

    # 3. lopo_split
    print("\n[data_splits] 3) lopo_split")
    lopo = lopo_split(labels_df, ide_logs, n_parts=7)
    for tr_idx, te_idx, meta in lopo:
        print(f"   left_out_part={meta['left_out_part']}: "
              f"n_train={meta['n_train']}, n_test={meta['n_test']}, "
              f"y_test fail_rate={y[te_idx].mean():.3f}")
    total_test = sum(m['n_test'] for _, _, m in lopo)
    assert abs(total_test - len(y)) <= 1, f'lopo should cover all students (got {total_test})'

    # 4. perturbation_split (4 types)
    print("\n[data_splits] 4) perturbation_split (4 types)")
    for p in ['event_mask', 'time_noise', 'type_swap', 'truncate']:
        df_p, m = perturbation_split(ide_logs, labels_df, perturbation=p, severity=0.20, seed=42)
        print(f"   {p}: {m['n_events_orig']} → {m['n_events_after']} events")

    # 5. stratified_subsample helper
    print("\n[data_splits] 5) stratified_subsample")
    rng = np.random.default_rng(0)
    sub = stratified_subsample(y, n=100, rng=rng)
    print(f"   n={len(sub)}, fail_rate={y[sub].mean():.3f} (vs overall {y.mean():.3f})")

    print("\n[data_splits] === SELF-TEST END (all 5 splits passed) ===\n")


if __name__ == '__main__':
    _self_test()