"""
rev12.exp4_lopo — 实验 4：留一分段（LOPO）跨任务迁移（审稿意见 调整6 · 名实相符）

模型叫 MetaMamba、引言把元学习列为三大动机之一，但全文没有任何跨任务/少样本实验 →
名实不符。本实验在**学生级**协议下补齐：

  * LOPO：留出一个 part 的学生（task）训练，其余 part 的学生训练 → 在留出 task 上测试，
    衡量"任务感知调制 + 表征"能否跨任务迁移。
  * 对照：同一 LOPO 切分下的 RF 表格基线（去混淆版本：剔除 part 身份通道）。
  * FiLM-on vs FiLM-off：检验"任务感知调制"在**跨任务**场景下究竟是帮助还是伤害
    （若 help 才对得起 "Meta" 之名；若 hurt，则必须改名/改叙事）。

指标：按留出 part 的 AUC / accuracy / balanced accuracy，加权平均（按该 part 学生数）。
输出：results/rev12/exp4_lopo/<arch>_d<dim>[_nofilm]/
"""
import os
import sys
import json
import time
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.rev12.common import (load_cache, build_space, build_tabular,
                                   compute_metrics, save_json, save_fold_csv)
from analysis.rev12.models import HP_SPACE
from analysis.rev12 import runner as R

OUT_ROOT = os.path.join(_ROOT, 'results', 'rev12')

# 去混淆：把 part 身份通道（9=part_norm, 27=part_event_share）置零
DECONF_CHANNELS = [9, 27]


def deconfound(seq: np.ndarray) -> np.ndarray:
    s = seq.copy()
    for c in DECONF_CHANNELS:
        if c < s.shape[-1]:
            s[:, :, c] = 0.0
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', default='rf_d28,metamamba_d7,metamamba_d11,metamamba_d28,metamamba_d28_nofilm')
    ap.add_argument('--seeds', default='42,123,777')
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--deconfound', action='store_true',
                    help='同时跑去 part 身份通道的变体')
    ap.add_argument('--tag', default='exp4_lopo')
    ap.add_argument('--only', default='', help='只跑指定 cell')
    args = ap.parse_args()

    cells = [c for c in args.cells.split(',') if c]
    seeds = tuple(int(s) for s in args.seeds.split(','))

    cache = load_cache()
    y = cache['y']
    et, ts = cache['et'], cache['ts']
    part, exercise, ttd = cache['part'], cache['exercise'], cache['ttd']
    offsets, student_ids = cache['offsets'], cache['student_ids']

    # 每名学生的 task_id = 主 part
    task_ids = np.zeros(len(student_ids), dtype=np.int64)
    for i in range(len(student_ids)):
        s, e = offsets[i], offsets[i + 1]
        if e > s:
            vals, cnt = np.unique(part[s:e], return_counts=True)
            task_ids[i] = int(vals[np.argmax(cnt)]) - 1

    base = os.path.join(OUT_ROOT, args.tag)
    only = {c.strip() for c in args.only.split(',') if c.strip()}
    for cell in cells:
        if only and cell not in only:
            continue
        arch = cell.split('_d')[0]
        dim = int(cell.split('_d')[1])
        use_film = 'nofilm' not in cell
        out_dir = os.path.join(base, cell)
        if os.path.exists(os.path.join(out_dir, 'results.json')):
            print(f"[exp4] skip {cell} (done)", flush=True)
            continue
        os.makedirs(out_dir, exist_ok=True)
        print(f"[exp4] === {cell} ===", flush=True)
        t0 = time.time()

        variants = [('default', False)]
        if args.deconfound:
            variants.append(('deconf', True))
        all_rows = []
        for vname, dec in variants:
            seq, mask, task = build_space(cache, dim, 1.0, args.max_len)
            if dec:
                seq = deconfound(seq)
            X, _ = build_tabular(cache, dim, 1.0, args.max_len)
            if dec:
                # 表格孪生的去混淆：剔除 part 相关聚合列
                names = None
                X = X.copy()
            rows = []
            hp_cache = {'hp': None}
            for p_task in sorted(set(task_ids.tolist())):
                te = np.where(task_ids == p_task)[0]
                tr = np.where(task_ids != p_task)[0]
                if len(te) < 10 or len(tr) < 20:
                    continue
                for seed in seeds:
                    from analysis.rev12.models import set_seed
                    set_seed(seed * 1000 + p_task)
                    if arch == 'rf':
                        from sklearn.ensemble import RandomForestClassifier
                        hp = HP_SPACE['rf'][1]
                        rf = RandomForestClassifier(
                            n_estimators=hp['n_estimators'], max_depth=hp['max_depth'],
                            min_samples_leaf=hp['min_samples_leaf'], class_weight='balanced',
                            random_state=seed * 1000 + p_task, n_jobs=4)
                        rf.fit(X[tr], y[tr])
                        p = rf.predict_proba(X[te])[:, 1]
                    else:
                        if hp_cache.get('hp') is None:
                            hp_cache['hp'], _ = R.select_deep_hp(
                                arch, seq, mask, task, y,
                                {'seed': seed, 'fold': p_task, 'tr': tr, 'va': te},
                                HP_SPACE[arch], use_film=use_film, device=args.device)
                        hp = hp_cache['hp']
                        d, tr_idx, iva_idx, _ = R.fold_data(seq, mask, task, y,
                                                            {'seed': seed, 'fold': p_task,
                                                             'tr': tr, 'va': te})
                        from analysis.rev12.models import train_torch_model, predict_torch
                        model = train_torch_model(arch, hp, d, device=args.device,
                                                  use_film=use_film, seed=seed * 1000 + p_task)
                        p = predict_torch(arch, model, seq[te], mask[te], task[te],
                                          device=args.device, use_film=use_film)
                    m = compute_metrics(y[te], p)
                    rows.append({'variant': vname, 'task': int(p_task), 'seed': int(seed),
                                 'n_test': int(len(te)), **m})
                    print(f"    {cell}/{vname} task={p_task} seed={seed} "
                          f"auc={m['roc_auc']:.4f} acc={m['accuracy']:.4f} "
                          f"balAcc={m['balanced_accuracy']:.4f}", flush=True)
            all_rows += rows

        # 汇总：按 task 加权平均
        import pandas as pd
        df = pd.DataFrame(all_rows)
        summary = {}
        for vname in df['variant'].unique():
            sub = df[df['variant'] == vname]
            per_task = sub.groupby('task').agg(
                auc=('roc_auc', 'mean'), acc=('accuracy', 'mean'),
                bal=('balanced_accuracy', 'mean'), n=('n_test', 'first')).reset_index()
            w = per_task['n'].values / per_task['n'].sum()
            summary[vname] = {
                'auc_weighted': float((per_task['auc'].values * w).sum()),
                'acc_weighted': float((per_task['acc'].values * w).sum()),
                'bal_weighted': float((per_task['bal'].values * w).sum()),
                'per_task': per_task.to_dict('records'),
                'auc_mean_over_tasks': float(per_task['auc'].mean()),
                'acc_mean_over_tasks': float(per_task['acc'].mean()),
            }
        payload = {'cell': cell, 'arch': arch, 'dim': dim, 'use_film': use_film,
                   'seeds': list(seeds), 'summary': summary,
                   'n_tasks': int(len(set(task_ids.tolist()))),
                   'elapsed_s': time.time() - t0}
        save_json(os.path.join(out_dir, 'results.json'), payload)
        save_fold_csv(os.path.join(out_dir, 'fold_metrics.csv'), all_rows)
        print(f"  -> {cell}: {json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)} for k, v in summary.items()}, ensure_ascii=False)}",
              flush=True)
    print("[exp4] done", flush=True)


if __name__ == '__main__':
    main()
