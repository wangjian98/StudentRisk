#!/usr/bin/env python3
"""
习题级 hazard 模型 + 与 part 级的公平对照
=================================================================================
任务设定（同一标签 drop_final = 该生最终会掉队）：
  · 习题级 E：样本 = (学生, part, 习题)；特征 = 该习题及其之前的**前缀行为**
    （前缀累计 + 同习题同伴偏离 + 自身习题偏离 + 进度上下文）
  · part 级 P：样本 = (学生, part)；特征 = 该 part 的聚合行为（rf29 同源口径）
  两者都用**按学生分组** 5 折 × 3 seeds；学生级分数 = 行分数取 max

核心对照（按"观察到的行为量"匹配）：
  · P 在第 K 个 part 截止 ≈ E 在第 30K 个习题截止（一个 part 约 30 题）
  · 关键问题：**习题分辨率能否用更少的行为量做出同等或更早的预警？**

另外计算：
  · 经典 hazard 标签 drop_at（停点习题）→ part 内 pooled AUC，与 part 级 0.9258 对照
  · 预警点分析：全量模型下，每个学生在第几个习题上被首次标记（精确率/覆盖率）
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402
from analysis.hazard_model.run_dchm import build_features  # noqa: E402


def fit_rf(Xtr, ytr, Xte, seeds, n_jobs=-1):
    if len(set(np.asarray(ytr).tolist())) < 2:
        return np.full(len(Xte), float(np.mean(ytr)) if len(ytr) else 0.0)
    acc = np.zeros(len(Xte))
    for sd in seeds:
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                   max_features='sqrt', class_weight='balanced',
                                   random_state=sd, n_jobs=n_jobs)
        m.fit(Xtr, ytr)
        acc += m.predict_proba(Xte)[:, 1]
    return acc / len(seeds)


def build_exercise_prefix_features(ex):
    """输入 exercise_dataset 的 raw 矩阵（12 维），输出前缀特征矩阵"""
    raw = ex['raw'].astype(np.float64)
    sid, par, exr = ex['sid'], ex['part'], ex['exercise']
    N = len(sid)
    # 组内按习题序排列
    groups = defaultdict(list)
    for i in range(N):
        groups[(int(sid[i]), int(par[i]))].append(i)
    D = np.zeros((N, 30), dtype=np.float64)
    names = []
    for j in range(30):
        names.append(f'f{j}')
    for (s, p), idx in groups.items():
        idx = np.array(idx)
        order = np.argsort(exr[idx])
        idx = idx[order]
        ex_sorted = exr[idx]
        cnt = raw[idx, 1:8]                 # 7 类事件计数
        n = raw[idx, 0]
        ttd_max = raw[idx, 8]
        ttd_min = raw[idx, 9]
        ts_min = raw[idx, 10]
        ts_max = raw[idx, 11]
        cum_cnt = np.cumsum(cnt, axis=0)
        cum_n = np.cumsum(n)
        cum_sub = np.cumsum(raw[idx, 7])
        cum_run = np.cumsum(raw[idx, 6])
        first_ts = ts_min[0]
        ttd_start = ttd_max[0]
        for k, i in enumerate(idx):
            e = ex_sorted[k]
            span_h = (ts_max[k] - first_ts) / 3600.0
            util = 1.0 - ttd_min[k] / max(ttd_start, 1.0)
            avail_h = max(ttd_start / 3600.0, 1e-6)
            f = []
            # 前缀累计族（7 类）
            f += list(cum_cnt[k])
            f.append(np.log1p(cum_n[k]))            # 累计事件量
            f.append(np.log1p(cum_sub[k]))           # 累计提交
            f.append(np.log1p(cum_run[k]))           # 累计运行
            f.append(span_h)
            f.append(util)
            f.append(cum_n[k] / avail_h)             # 进度速率
            f.append(np.log1p(e))                    # part 内习题序
            f.append(float(p))                       # part
            f.append(e / 30.0)                       # 相对进度
            # 本习题自身相对前缀的偏离（自我参照）
            if k > 0:
                prev_n = cum_n[k - 1] / k
                f.append((n[k] - prev_n) / (prev_n + 1e-6))
                f.append(span_h / (np.log1p(cum_n[k]) + 1e-6))
            else:
                f.append(0.0)
                f.append(0.0)
            # 占位：同伴偏离在下面统一补（需要全局 peer 统计）
            D[i, :len(f)] = f
            D[i, len(f):len(f) + 7] = 0.0
    return D, names


def peer_anchor_exercise(D, raw, sid, par, exr, rows, peer_cols):
    """同伴偏离：在同一 (part, 习题) 内对当前习题自身行为做 z-score
       peer_cols 里的原始维度：n, ins, rem, run, sub, span_h, utilization"""
    cell = defaultdict(list)
    for i in rows:
        cell[(int(par[i]), int(exr[i]))].append(i)
    for key, idxs in cell.items():
        idxs = np.array(idxs)
        for c, col in enumerate(peer_cols):
            v = raw[idxs, col].astype(float)
            mu, sd = v.mean(), v.std()
            sd = sd if sd > 1e-9 else 1.0
            D[idxs, 22 + c] = np.clip((v - mu) / sd, -5, 5)


def main(args):
    ds = sc.load_dataset(args.data)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part_ids'])
    ds['DER'] = sc.derive(ds)
    ds['SELF'] = sc.self_ref(ds)
    ex = np.load(args.ex, allow_pickle=True)
    raw = ex['raw'].astype(np.float64)
    sid, par, exr = ex['sid'], ex['part'], ex['exercise']
    N = len(sid)
    seeds = [42, 123, 777]
    folds = sc.group_folds(ds['sid'], ds['final'], n_splits=5, seed=42)

    # ---- 标签 ----
    max_part = {}
    for s, p in zip(ds['sid'], ds['part']):
        max_part[int(s)] = max(max_part.get(int(s), 0), int(p))
    drop_final_stu = {s: (1 if max_part[s] < 7 else 0) for s in max_part}
    lab_final = np.array([drop_final_stu[int(s)] for s in sid])
    # 停点习题：学生在最终 part 内最后触碰的习题（且最终 part < 7）
    stop_ex = {}
    sp_s, sp_p, sp_ex = ex['sp_s'], ex['sp_p'], ex['sp_ex']
    for i in range(len(sp_s)):
        s, p = int(sp_s[i]), int(sp_p[i])
        if max_part[s] < 7 and p == max_part[s]:
            stop_ex[(s, p)] = int(sp_ex[i].max())
    lab_at = np.zeros(N, dtype=int)
    for i in range(N):
        s, p = int(sid[i]), int(par[i])
        if (s, p) in stop_ex and int(exr[i]) == stop_ex[(s, p)]:
            lab_at[i] = 1
    print(f'习题级样本 {N} | drop_final 正例行 {int(lab_final.sum())} '
          f'（学生 {int(sum(drop_final_stu.values()))}）| drop_at 事件 {int(lab_at.sum())}',
          flush=True)

    # ---- 特征 ----
    D, _ = build_exercise_prefix_features(ex)
    peer_cols = [0, 1, 2, 5, 6, 11, 12]   # n, ins, rem, run, sub, span_h(12=util?) 见下
    # 计算 span_h/util 列供 peer 使用
    raw2 = np.zeros((N, 13))
    raw2[:, :12] = raw
    span_h = np.zeros(N)
    util = np.zeros(N)
    groups = defaultdict(list)
    for i in range(N):
        groups[(int(sid[i]), int(par[i]))].append(i)
    for (s, p), idx in groups.items():
        idx = np.array(idx)
        o = np.argsort(exr[idx])
        idx = idx[o]
        first_ts = raw[idx, 10].min()
        ttd_start = max(raw[idx, 8].max(), 1.0)
        for k, i in enumerate(idx):
            span_h[i] = (raw[i, 11] - first_ts) / 3600.0
            util[i] = 1.0 - raw[i, 9] / ttd_start
    raw2[:, 11] = span_h
    raw2[:, 12] = util
    peer_anchor_exercise(D, raw2, sid, par, exr, np.arange(N), peer_cols)
    print(f'特征矩阵 {D.shape}', flush=True)

    # 习题序（part 内）
    ex_idx = np.zeros(N, dtype=int)
    for (s, p), idx in groups.items():
        idx = np.array(idx)
        o = np.argsort(exr[idx])
        for k, i in enumerate(idx[o]):
            ex_idx[i] = k + 1

    # 学生级索引
    stu_list = sorted(max_part.keys())
    stu_pos = {s: i for i, s in enumerate(stu_list)}

    # 行序：用于"末行"聚合（最新证据）
    row_order = np.lexsort((exr, par, sid))

    def eval_exercise_cutoff(k):
        """前 k 个习题的前缀特征 → drop_final
        返回：part 内 pooled AUC（行级，个体判别力口径）+ 学生级 AUC（末行聚合）"""
        sel = np.where(ex_idx <= k)[0]
        oof = np.zeros(N)
        for tr, te in folds:
            tr_s = set(int(x) for x in ds['sid'][tr])
            trm = np.array([i for i in sel if int(sid[i]) in tr_s])
            tem = np.array([i for i in sel if int(sid[i]) not in tr_s])
            if len(trm) < 50 or len(tem) == 0:
                continue
            oof[tem] = fit_rf(D[trm], lab_final[trm], D[tem], seeds)
        # 行级：part 内 pooled AUC
        m_row = sc.metrics(lab_final[sel], oof[sel], par[sel], f'ex_k{k}')
        # 学生级：末行聚合（最新证据，避免"行多者易被 max 命中"的偏差）
        ps = np.zeros(len(stu_list))
        last_row = {}
        for i in sel:
            s_i = int(sid[i])
            key = (int(par[i]), int(exr[i]))
            if s_i not in last_row or key > last_row[s_i][0]:
                last_row[s_i] = (key, i)
        for s_i, (_, i) in last_row.items():
            ps[stu_pos[s_i]] = oof[i]
        yv = np.array([drop_final_stu[s] for s in stu_list])
        mask = np.array([s in last_row for s in stu_list])
        auc = float(roc_auc_score(yv[mask], ps[mask])) if mask.sum() > 20 else float('nan')
        return auc, oof, ps, mask, yv, m_row

    print('\n=== E) 习题级模型（标签 drop_final），按"观察到的习题数"截止 ===')
    print(f'{"截止习题数":<14}{"学生级 AUC":<14}{"覆盖学生":<12}')
    ex_results = {}
    for k in args.ex_cutoffs:
        t0 = time.time()
        auc, oof, ps, mask, yv, m_row = eval_exercise_cutoff(k)
        ex_results[k] = dict(auc=auc, ps=ps, mask=mask, yv=yv)
        print(f'  前 {k:<8} 行级 part 内 pooled={m_row["pooled"]:.4f} '
              f'学生级(末行) AUC={auc:.4f} 覆盖学生={int(mask.sum())} '
              f'[{time.time()-t0:.0f}s]', flush=True)

    # ---- P) part 级同标签对照 ----
    print('\n=== P) part 级同标签对照（drop_final），按"观察到的 part 数"截止 ===')
    print(f'{"截止 part 数":<14}{"学生级 AUC":<14}{"该 part 的习题量(中位)":<20}')
    part_results = {}
    for K in args.part_cutoffs:
        t0 = time.time()
        sel_rows = np.where(ds['part'] <= K)[0]
        oof = np.zeros(len(ds['drop']))
        for tr, te in folds:
            tr_s = set(int(x) for x in ds['sid'][tr])
            trm = np.array([i for i in sel_rows if int(ds['sid'][i]) in tr_s])
            tem = np.array([i for i in sel_rows if int(ds['sid'][i]) not in tr_s])
            if len(trm) < 50 or len(tem) == 0:
                continue
            Xtr, Xte = build_features(ds, trm, tem, mode='rf29')
            ylab = np.array([drop_final_stu[int(ds['sid'][i])] for i in trm])
            oof[tem] = fit_rf(Xtr, ylab, Xte, seeds)
        ylab_all = np.array([drop_final_stu[int(ds['sid'][i])] for i in sel_rows])
        m_row = sc.metrics(ylab_all, oof[sel_rows], ds['part'][sel_rows], f'part_K{K}')
        # 学生级：末行聚合（同一口径）
        ps = np.zeros(len(stu_list))
        last_row = {}
        for i in sel_rows:
            s_i = int(ds['sid'][i])
            key = int(ds['part'][i])
            if s_i not in last_row or key > last_row[s_i][0]:
                last_row[s_i] = (key, i)
        for s_i, (_, i) in last_row.items():
            ps[stu_pos[s_i]] = oof[i]
        yv = np.array([drop_final_stu[s] for s in stu_list])
        mask = np.array([s in last_row for s in stu_list])
        auc = float(roc_auc_score(yv[mask], ps[mask])) if mask.sum() > 20 else float('nan')
        part_results[K] = dict(auc=auc, ps=ps, mask=mask, yv=yv, pooled=m_row['pooled'])
        print(f'  前 {K:<3} 个 part   行级 part 内 pooled={m_row["pooled"]:.4f} '
              f'学生级(末行) AUC={auc:.4f} 覆盖学生={int(mask.sum())} '
              f'[{time.time()-t0:.0f}s]', flush=True)

    # ---- 经典 hazard 标签 drop_at 的 part 内 pooled AUC ----
    print('\n=== 习题级经典 hazard 标签（drop_at，停点习题）part 内 pooled AUC ===')
    oof_at = np.zeros(N)
    for tr, te in folds:
        Xtr, Xte = D[tr], D[te]
        oof_at[te] = fit_rf(Xtr, lab_at[tr], Xte, seeds)
    m_at = sc.metrics(lab_at, oof_at, par, 'ex_hazard_drop_at')
    print(f'  习题级 hazard（drop_at）  overall={m_at["auc"]:.4f} '
          f'pooled={m_at["pooled"]:.4f} judged={m_at["judged"]:.4f}')
    # (part, exercise) 内 pooled AUC（更细的同层比较）
    cell = par.astype(np.int64) * 100 + exr.astype(np.int64)
    m_cell = sc.metrics(lab_at, oof_at, cell, 'ex_hazard_drop_at_cell')
    print(f'    (part,习题) 内 pooled AUC = {m_cell["pooled"]:.4f} '
          f'（细胞数 {len(np.unique(cell))}）')
    # 事件前提前量：掉队学生在停点之前几个习题被模型标记
    per_sp = defaultdict(list)
    for i in range(N):
        per_sp[(int(sid[i]), int(par[i]))].append(i)
    lead = []
    for (s_i, p_i), idx in per_sp.items():
        idx = np.array(idx)
        o = np.argsort(exr[idx])
        idx = idx[o]
        if lab_at[idx].sum() == 0:
            continue
        stall_i = idx[np.where(lab_at[idx] > 0)[0][0]]
        pos = np.where(idx == stall_i)[0][0]
        pre = idx[:pos]
        if len(pre) == 0:
            continue
        first_flag = None
        for j, i in enumerate(pre):
            if oof_at[i] >= 0.5:
                first_flag = j
                break
        if first_flag is not None:
            lead.append(pos - first_flag)
    if lead:
        lead = np.array(lead)
        print(f'    停点前被首次标记的习题数：n={len(lead)} 中位 {int(np.median(lead))} '
              f'p75 {int(np.percentile(lead,75))} 最大 {lead.max()}')

    # ---- 预警点分析 ----
    print('\n=== 预警点分析：全量习题级模型下，学生被首次标记的位置 ===')
    auc, oof_full, ps_full, mask_full, yv_full, m_row_full = eval_exercise_cutoff(999)
    # 逐 (学生,part) 的首次越阈习题
    per_sp = defaultdict(list)
    for i in range(N):
        per_sp[(int(sid[i]), int(par[i]))].append(i)
    flagged = defaultdict(lambda: [0, 0])   # 习题序 -> [TP, n_flagged]
    for (s_i, p_i), idx in per_sp.items():
        idx = np.array(idx)
        o = np.argsort(exr[idx])
        idx = idx[o]
        hit = None
        for i in idx:
            if oof_full[i] >= 0.5:
                hit = int(exr[i])
                break
        if hit is not None:
            flagged[hit][0] += int(drop_final_stu[s_i])
            flagged[hit][1] += 1
    tot_tp = sum(v[0] for v in flagged.values())
    tot_n = sum(v[1] for v in flagged.values())
    print(f'  全量模型下被标记的 (学生,part) 共 {tot_n} 个，真实最终掉队 {tot_tp} 个'
          f'（精确率 {tot_tp/max(tot_n,1):.2f}）')
    for k in sorted(flagged):
        tp, n = flagged[k]
        print(f'  首次越阈在第 {k:>2} 个习题：n={n:<4} 真实掉队 {tp:<4} '
              f'精确率 {tp/max(n,1):.2f}')

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'exercise_results.npz'),
                        lab_final=lab_final, lab_at=lab_at, ex_idx=ex_idx,
                        sid=sid, part=par, exercise=exr,
                        **{f'ex_k{k}_ps': v['ps'] for k, v in ex_results.items()},
                        **{f'part_K{K}_ps': v['ps'] for K, v in part_results.items()})
    print(f'\n写出 {args.out}/exercise_results.npz')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--ex', default='data/exercise_dataset.npz')
    ap.add_argument('--out', default='results/exercise')
    ap.add_argument('--ex-cutoffs', type=int, nargs='+', default=[1, 2, 3, 5, 10, 999])
    ap.add_argument('--part-cutoffs', type=int, nargs='+', default=[1, 2, 3])
    main(ap.parse_args())
