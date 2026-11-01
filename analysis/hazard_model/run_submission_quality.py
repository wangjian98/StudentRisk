#!/usr/bin/env python3
"""
任务 3a · 提交质量校验（"乱点题号刷提交"的检出与影响评估）
=================================================================================
背景：完成度/提交族特征是阶段级判别力的主力（完成度单独 0.8646），但若学习者
      通过"点提交"刷进度，这些特征就会被污染 → 需要量化污染程度与影响。

检出规则（逐 (学生, part, 习题)）：
  Q1 空提交      : 有 submit 事件，但该习题的 insert 事件 < 5 且 run == 0
  Q2 刷提交      : n_submit / n_events > 0.5（提交占比异常高）
  Q3 秒提交      : 有 submit 且该习题跨度 < 10 分钟（点完就走）
  Q4 无运行提交  : 有 submit 但整段 run == 0（提交了但从未运行程序）

影响评估：
  1) 污染规模：多少 (学生,part) 行、多少学生被命中
  2) 剔除污染行后，阶段级 pooled AUC 是否变化（RF 全特征）
  3) 关键行为学结论是否变化（"提前收手" utilization 的风险比）
  4) IPSD 坐下级 top5% 名单是否变化（是否被污染行主导）
"""
import argparse
import os
import sys
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


def main(args):
    ds = sc.load_dataset(args.data)
    ds['base_ex'] = sc.base_exercises(ds['agg10'], ds['part_ids'])
    ds['DER'] = sc.derive(ds)
    ds['SELF'] = sc.self_ref(ds)
    y = ds['drop'].astype(int)
    part, sid = ds['part'], ds['sid']
    n = len(y)
    seeds = [42, 123, 777]
    folds = sc.group_folds(sid, ds['final'], n_splits=5, seed=42)

    ex = np.load(args.ex, allow_pickle=True)
    raw = ex['raw'].astype(np.float64)     # n,7cnt,ttd_max,ttd_min,ts_min,ts_max
    ex_sid, ex_par = ex['sid'], ex['part']
    # (生,段) → 习题级质量标记
    sp_ex = defaultdict(list)
    for i in range(len(ex_sid)):
        sp_ex[(int(ex_sid[i]), int(ex_par[i]))].append(i)
    q_empty, q_farm, q_quick, q_norun = set(), set(), set(), set()
    for k, idxs in sp_ex.items():
        for i in idxs:
            n_ev = raw[i, 0]
            n_ins, n_rem, n_pas, n_fg, n_fl, n_run, n_sub = raw[i, 1:8]
            span = (raw[i, 11] - raw[i, 10]) / 3600.0
            if n_sub >= 1 and n_ins < 5 and n_run == 0:
                q_empty.add(k)
            if n_ev > 0 and n_sub / n_ev > 0.5:
                q_farm.add(k)
            if n_sub >= 1 and span < 10 / 3600.0:
                q_quick.add(k)
            if n_sub >= 1 and n_run == 0:
                q_norun.add(k)
    # 主口径：只用强信号（Q1 空提交 / Q2 刷提交 / Q3 秒提交）；Q4「无运行提交」过宽（54.5%），
    # 多为正常提交（部分练习不需运行），仅作参考不计入污染集
    allq = q_empty | q_farm | q_quick
    print('=== 提交质量检出规模 ===')
    for nm, s in [('Q1 空提交', q_empty), ('Q2 刷提交(提交占比>0.5)', q_farm),
                   ('Q3 秒提交(<10 分钟)', q_quick), ('Q4 无运行提交', q_norun)]:
        stu = len({k[0] for k in s})
        print(f'  {nm:<26} 命中 (生,段) {len(s):>4} / {len(sp_ex)}  涉及学生 {stu:>4}')
    print(f'  强信号合计(Q1/Q2/Q3)     (生,段) {len(allq):>4}  涉及学生 '
          f'{len({k[0] for k in allq})}   ← 主污染集')
    print(f'  参考：Q4 无运行提交        (生,段) {len(q_norun):>4}  涉及学生 '
          f'{len({k[0] for k in q_norun})}（过宽，不计入污染集）')

    # 映射到阶段行
    part_key = {}
    for i in range(n):
        part_key[(int(sid[i]), int(part[i]))] = i
    bad_rows = np.array([part_key[k] for k in allq if k in part_key], dtype=int)
    bad_mask = np.zeros(n, dtype=bool)
    bad_mask[bad_rows] = True
    print(f'  → 阶段级行层面：{int(bad_mask.sum())} / {n} 行 '
          f'({bad_mask.mean():.2%}) 带有低质量提交')

    # ---- 影响 2：剔除污染行后的阶段级 pooled AUC ----
    clean_rows = np.where(~bad_mask)[0]

    def run(train_filter, test_rows, tag=''):
        """train_filter: 'all' 用全部行训练 / 'clean' 只用干净行训练；
           测试集固定为 test_rows（同一测试集，避免人群变化造成的假差异）"""
        oof = np.zeros(len(test_rows))
        pos = {int(r): i for i, r in enumerate(test_rows)}
        for tr, te in folds:
            tr2 = tr if train_filter == 'all' else tr[~bad_mask[tr]]
            te2 = np.array([r for r in te if int(r) in pos])
            if len(tr2) < 50 or len(te2) == 0:
                continue
            Xtr, Xte = build_features(ds, tr2, te2, mode='rf29')
            acc = np.zeros(len(te2))
            for sd in seeds:
                m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                           max_features='sqrt', class_weight='balanced',
                                           random_state=sd, n_jobs=-1)
                m.fit(Xtr, y[tr2])
                acc += m.predict_proba(Xte)[:, 1]
            for r, v in zip(te2, acc / len(seeds)):
                oof[pos[int(r)]] = v
        mm = sc.metrics(y[test_rows], oof, part[test_rows], tag)
        return mm, oof

    m_full_test, _ = run('all', clean_rows, 'train_all_test_clean')
    m_clean_train, _ = run('clean', clean_rows, 'train_clean_test_clean')
    m_allrows, _ = run('all', np.arange(n), 'all_rows')
    print('\n=== 影响 2：污染对判别力的影响（同一测试集 = 未污染行）===')
    print(f'  参照：全部 {n} 行上的 pooled = {m_allrows["pooled"]:.4f}')
    print(f'  测试集固定为干净行 (n={len(clean_rows)})：')
    print(f'    训练用全部行（含污染）  pooled = {m_full_test["pooled"]:.4f}')
    print(f'    训练用干净行            pooled = {m_clean_train["pooled"]:.4f}')
    print(f'    Δ = {m_clean_train["pooled"]-m_full_test["pooled"]:+.4f}  '
          f'（同一测试集，可直接比较）')

    # ---- 影响 3：关键行为学结论是否变化 ----
    dl = np.load(args.dl, allow_pickle=True)
    dkey = {(int(dl['sid'][i]), int(dl['part'][i])): i for i in range(len(dl['sid']))}
    rows_dl = np.array([dkey[(int(s), int(p))] for s, p in zip(sid, part)])
    util = dl['feats'][rows_dl][:, list(map(str, dl['names'])).index('utilization')]
    print('\n=== 影响 3："提前收手"（低利用率）风险比是否被污染行驱动 ===')
    for nm, msk in [('全部行', np.ones(n, dtype=bool)), ('剔除污染行', ~bad_mask)]:
        v = util[msk]
        yy = y[msk]
        q1, q3 = np.percentile(v, [25, 75])
        lo, hi = v <= q1, v >= q3
        if lo.sum() < 20 or hi.sum() < 20:
            print(f'  {nm}: 样本不足')
            continue
        r_lo, r_hi = yy[lo].mean(), yy[hi].mean()
        print(f'  {nm:<12} 低利用率掉队率={r_lo:.3f}  高利用率={r_hi:.3f}  '
              f'风险比={r_hi/max(r_lo,1e-9):.2f}  n={int(msk.sum())}')

    # ---- 影响 4：IPSD 坐下级 top5% 名单是否被污染主导 ----
    sit = os.path.join(os.path.dirname(args.out.rstrip('/')), 'sitting', 'sit_k1_feat.npz')
    if os.path.exists(sit):
        Z = np.load(sit, allow_pickle=True)
        rows_s, oof_s = Z['rows'], Z['oof']
        lab = Z['y']
        o = np.argsort(-oof_s)
        k = int(len(o) * 0.05)
        top = o[:k]
        bad_in_top = int(bad_mask[rows_s[top]].sum())
        print('\n=== 影响 4：坐下级 top5% 名单中的污染行占比 ===')
        print(f'  top5% 共 {k} 个 (生,段)，其中带低质量提交 {bad_in_top} 个 '
              f'({bad_in_top/k:.1%})；整体污染率 {bad_mask[rows_s].mean():.1%}')

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, 'quality_flags.npz'),
                        bad_mask=bad_mask, q_empty=np.array(sorted(q_empty), dtype=object),
                        q_farm=np.array(sorted(q_farm), dtype=object),
                        q_quick=np.array(sorted(q_quick), dtype=object),
                        q_norun=np.array(sorted(q_norun), dtype=object))
    print(f'\n写出 {args.out}/quality_flags.npz')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--ex', default='data/exercise_dataset.npz')
    ap.add_argument('--dl', default='data/deadline_feats.npz')
    ap.add_argument('--out', default='results/quality')
    main(ap.parse_args())
