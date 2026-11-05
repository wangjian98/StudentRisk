"""Per-Student Leave-One-Part-Out (王建 2026-10-11 真实意图, v2 重构版)

实验设计 (王建):
  - 重构数据: 每个 (student, part) 对应 1 个 sequence (该学生在该 part 上的事件序列)
  - 验证: 用 FiLM 训练后, 对**同一学生**训练时**未见过**的 partid (该学生做过但本次 fold hold-out) 的识别能力

对比:
  - FiLM ON  (use_film=True)  - 训练时用 FiLM 调制
  - FiLM OFF (use_film=False) - 训练时不用 FiLM (作 ablation baseline)

数据:
  - 选 N=30 个全 7-partid 学生 (213 候选人中抽样)
  - 每个学生做 7-fold: hold 一个 part, 用其他 6 parts + 其他所有学生训练
  - FiLM ON/OFF × 3 seeds = 6 fold/学生

输出:
  - results/per_student_lopo/per_student_lopo_records.csv
  - results/per_student_lopo/per_student_lopo_summary.json
  - results/per_student_lopo/per_student_lopo_by_part_film.csv
"""
import os
import sys
import json
import time
import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data import load_dataset
from models.base import set_seed
from models.meta_mamba.model import MetaMambaClassifier
from models.meta_mamba.train import train_one_fold
from models.lstm_7d.data import EVENT_TYPE_TO_IDX, build_event_sequences_7d


def load_config(path=None):
    if path is None:
        path = os.path.join(_ROOT, 'configs', 'default.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def build_per_student_part_sequences(ide_logs, student_ids, max_len=128, target_students=None):
    """重构数据: 每个 (student, part) 一个 sequence.

    返回:
      sequences: (n_pairs, max_len, 7) float32
      masks:     (n_pairs, max_len) float32
      task_ids:  (n_pairs,) int64   (0-indexed part)
      pair_sids: (n_pairs,) int64   (对应 student_id)
      pair_idx:  (n_pairs,) int64   (在 pair_sids 中的索引, 用于映射回原 student)

    只保留 target_students 列表中学生做过的 (student, part) 对。
    """
    if target_students is None:
        target_students = sorted(student_ids.tolist())
    target_set = set(int(s) for s in target_students)

    sequences_list, masks_list, task_ids_list, pair_sids = [], [], [], []

    for sid in target_students:
        s_ev = ide_logs[ide_logs.student == sid]
        if len(s_ev) == 0 or 'part' not in s_ev.columns:
            continue
        parts = sorted(s_ev['part'].unique().tolist())
        for part in parts:
            part_ev = s_ev[s_ev.part == part].sort_values('timestamp').reset_index(drop=True)
            L = min(len(part_ev), max_len)
            recent = part_ev.tail(L).reset_index(drop=True)
            seq = np.zeros((max_len, 7), dtype=np.float32)
            mask = np.zeros(max_len, dtype=np.float32)
            et_ids = recent['eventType'].map(EVENT_TYPE_TO_IDX).fillna(0).astype(int).values
            seq[max_len - L:] = np.eye(7, dtype=np.float32)[et_ids]
            mask[max_len - L:] = 1.0
            sequences_list.append(seq)
            masks_list.append(mask)
            task_ids_list.append(int(part) - 1)  # 1-indexed → 0-indexed
            pair_sids.append(int(sid))

    sequences = np.array(sequences_list, dtype=np.float32)
    masks = np.array(masks_list, dtype=np.float32)
    task_ids = np.array(task_ids_list, dtype=np.int64)
    pair_sids = np.array(pair_sids, dtype=np.int64)
    print(f'[per-pair] Built {len(sequences)} (student, part) pairs '
          f'from {len(target_set)} target students', flush=True)
    print(f'  shape: seq={sequences.shape}, task_ids={task_ids.shape}, '
          f'unique students={len(np.unique(pair_sids))}, unique parts={sorted(np.unique(task_ids).tolist())}', flush=True)
    return sequences, masks, task_ids, pair_sids


def select_full_part_students(ide_logs, n_target=30, seed=42):
    """选 n_target 个做了全部 7 parts 的学生"""
    parts_per_student = ide_logs.groupby('student')['part'].nunique()
    full_students = parts_per_student[parts_per_student == 7].index.tolist()
    print(f'[select] 全 7-partid 学生候选: {len(full_students)} 人 (共 {ide_logs["student"].nunique()} 学生)', flush=True)
    if len(full_students) <= n_target:
        return sorted(full_students)
    rng = np.random.RandomState(seed)
    selected = rng.choice(full_students, size=n_target, replace=False).tolist()
    return sorted(selected)


def main():
    # ---------- 配置 ----------
    seeds = (42,)                # 1 seed 先跑 (省时间), 后续可加 seeds
    epochs = 20                  # 减少 epochs (原 30, 加 patience=5 早停)
    batch_size = 16
    max_len = 64                 # 减少 max_len (原 128) 加速
    n_target_students = 15        # 抽样 15 个全 7-partid 学生 (减少 50%)
    film_settings = (True, False)  # FiLM ON / OFF

    out_dir = os.path.join(_ROOT, 'results', 'per_student_lopo')
    os.makedirs(out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[per-student-LOPO-v2] device={device}, n_students={n_target_students}, '
          f'seeds={seeds}, epochs={epochs}, film_settings={film_settings}', flush=True)

    # ---------- 数据 (重构为 per-(student, part)) ----------
    ide_logs, labels_df, y_all, student_ids_all = load_dataset()
    target_students = select_full_part_students(ide_logs, n_target=n_target_students)

    sequences, masks, task_ids, pair_sids = build_per_student_part_sequences(
        ide_logs, student_ids_all, max_len=max_len, target_students=target_students
    )
    n_tasks = int(task_ids.max()) + 1

    # 每个 (student, part) pair 的 label: 取该学生的 failed label
    sid_to_label = dict(zip(labels_df['student'].astype(int), labels_df['failed'].astype(int).values))
    y = np.array([sid_to_label[int(s)] for s in pair_sids])
    print(f'[per-student-LOPO-v2] sequences={sequences.shape}, n_tasks={n_tasks}, '
          f'y dist: failed={int(y.sum())}/{len(y)} ({y.mean():.3f})', flush=True)

    config = load_config()
    mcfg = config.get('meta_mamba', config.get('attention', {}))

    # ---------- pair_sid → indices 映射 ----------
    sid_to_indices = {}
    for i, sid in enumerate(pair_sids):
        sid_to_indices.setdefault(int(sid), []).append(i)

    all_records = []
    t_global = time.time()

    # ---------- 主循环 ----------
    for student_id in target_students:
        s_indices = np.array(sid_to_indices.get(int(student_id), []))
        if len(s_indices) == 0:
            continue
        s_parts = sorted(np.unique(task_ids[s_indices]).tolist())
        if len(s_parts) != 7:
            continue

        for held_out_part in s_parts:
            # 测试集: 该学生的 hold_out_part pair
            test_idx = np.array([i for i in s_indices if task_ids[i] == held_out_part])
            # 训练池: 该学生的非 held_out_part pairs + 其他所有 pairs
            train_pool = np.array([
                i for i in range(len(pair_sids))
                if pair_sids[i] != student_id or task_ids[i] != held_out_part
            ])
            y_test = y[test_idx]
            n_test = len(test_idx)

            for seed in seeds:
                tr_abs, va_abs = train_test_split(
                    train_pool, test_size=0.1, stratify=y[train_pool], random_state=seed
                )

                for use_film in film_settings:
                    set_seed(seed)
                    model = MetaMambaClassifier(
                        d_event=sequences.shape[-1],
                        d_model=mcfg.get('d_model', 64),
                        d_state=mcfg.get('d_state', 16),
                        n_layers=mcfg.get('n_layers', 2),
                        n_tasks=n_tasks,
                        dropout=mcfg.get('dropout', 0.2),
                    ).to(device)
                    t0 = time.time()
                    model, _ = train_one_fold(
                        model,
                        sequences[tr_abs], masks[tr_abs], task_ids[tr_abs], y[tr_abs],
                        sequences[va_abs], masks[va_abs], task_ids[va_abs], y[va_abs],
                        epochs=epochs, lr=1e-3, weight_decay=1e-3,
                        batch_size=batch_size, patience=8,
                        contrastive_weight=mcfg.get('contrastive_weight', 0.3),
                        use_film=use_film,
                        device=device,
                    )
                    train_time = time.time() - t0

                    model.eval()
                    with torch.no_grad():
                        test_prob = torch.sigmoid(model(
                            torch.from_numpy(sequences[test_idx]).float().to(device),
                            torch.from_numpy(masks[test_idx]).float().to(device),
                            torch.from_numpy(task_ids[test_idx]).long().to(device),
                        )).cpu().numpy()

                    y_pred = (test_prob >= 0.5).astype(int)
                    try:
                        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
                    except Exception:
                        tn = fp = fn = tp = 0
                    acc = accuracy_score(y_test, y_pred)
                    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                    f1_failed = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
                    f1_passed = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
                    if len(np.unique(y_test)) == 2 and len(np.unique(y_pred)) > 1:
                        roc_auc = float(roc_auc_score(y_test, test_prob))
                        pr_auc = float(average_precision_score(y_test, test_prob))
                    else:
                        roc_auc = float('nan')
                        pr_auc = float('nan')

                    rec = {
                        'student_id': int(student_id),
                        'held_out_part_1idx': int(held_out_part + 1),
                        'seed': int(seed),
                        'use_film': bool(use_film),
                        'n_train': int(len(tr_abs)),
                        'n_val': int(len(va_abs)),
                        'n_test': int(n_test),
                        'n_test_passed': int((y_test == 0).sum()),
                        'n_test_failed': int((y_test == 1).sum()),
                        'train_time_s': float(train_time),
                        'accuracy': float(acc),
                        'macro_f1': float(macro_f1),
                        'f1_passed': float(f1_passed),
                        'f1_failed': float(f1_failed),
                        'roc_auc': float(roc_auc),
                        'pr_auc': float(pr_auc),
                    }
                    all_records.append(rec)

            elapsed = time.time() - t_global
            n_done = len(all_records)
            n_total = len(target_students) * 7 * len(seeds) * len(film_settings)
            eta = elapsed / max(n_done, 1) * (n_total - n_done)
            print(f'  [student {student_id:4d} | part {held_out_part+1}] '
                  f'n_test={n_test} | done={n_done}/{n_total} ({n_done/n_total:.0%}) | '
                  f'elapsed={elapsed:.0f}s | ETA={eta:.0f}s', flush=True)

            pd.DataFrame(all_records).to_csv(
                os.path.join(out_dir, 'per_student_lopo_records.csv'), index=False)

    elapsed = time.time() - t_global
    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(out_dir, 'per_student_lopo_records.csv'), index=False)
    print(f'\n[per-student-LOPO-v2] 完成: {len(df)} 个 fold, 总耗时 {elapsed:.0f}s', flush=True)

    # ---------- 聚合分析 ----------
    summary = {
        'experiment': 'per_student_lopo_v2',
        'design': 'Per-Student Leave-One-Part-Out (王建 2026-10-11)',
        'description': '验证 FiLM 训练后, 对同一学生未见过 partid 的识别能力',
        'data_construction': 'per-(student, part) sequence 重构 (每个学生每个做过的 part 一个序列)',
        'n_target_students': len(target_students),
        'seeds': list(seeds),
        'epochs': epochs,
        'film_settings': ['ON', 'OFF'],
        'elapsed_seconds': float(elapsed),
        'n_records': int(len(df)),
    }

    # 按 (held_out_part, use_film) 聚合
    by_part_film = df.groupby(['held_out_part_1idx', 'use_film']).agg({
        'accuracy': ['mean', 'std', 'count'],
        'macro_f1': ['mean', 'std'],
        'f1_passed': ['mean', 'std'],
        'f1_failed': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'pr_auc': ['mean', 'std'],
    }).reset_index()
    by_part_film.columns = ['_'.join(c).strip('_') for c in by_part_film.columns]
    by_part_film.to_csv(os.path.join(out_dir, 'per_student_lopo_by_part_film.csv'), index=False)

    # 按 use_film 聚合 (核心对比)
    by_film = df.groupby('use_film').agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'f1_passed': ['mean', 'std'],
        'f1_failed': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'pr_auc': ['mean', 'std'],
    }).reset_index()
    by_film.columns = ['_'.join(c).strip('_') for c in by_film.columns]
    summary['overall_by_film'] = by_film.to_dict(orient='records')

    # 按 held_out_part 聚合 (不分 film)
    by_part = df.groupby('held_out_part_1idx').agg({
        'accuracy': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
    }).reset_index()
    by_part.columns = ['_'.join(c).strip('_') for c in by_part.columns]
    by_part.to_csv(os.path.join(out_dir, 'per_student_lopo_by_part.csv'), index=False)

    with open(os.path.join(out_dir, 'per_student_lopo_summary.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # 控制台打印关键对比
    print('\n' + '=' * 80)
    print('【核心发现】FiLM ON vs OFF 在 per-student LOPO 上的表现')
    print('=' * 80)
    for _, row in by_film.iterrows():
        setting = 'FiLM ON ' if row['use_film'] else 'FiLM OFF'
        print(f'  {setting}: acc={row["accuracy_mean"]:.3f}±{row["accuracy_std"]:.3f}, '
              f'macro_f1={row["macro_f1_mean"]:.3f}, '
              f'f1_failed={row["f1_failed_mean"]:.3f}, '
              f'roc_auc={row["roc_auc_mean"]:.3f}')
    print('=' * 80, flush=True)


if __name__ == '__main__':
    main()
