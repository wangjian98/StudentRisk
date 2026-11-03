#!/usr/bin/env python3
"""
CS1 概率校准 + 选择性预测（拒答带）审计
=================================================================================
为什么做这个（论文附录 B 明列的两项待补项）：

  ① **校准尚未系统评估**：v10 只报了 Brier 0.0493 / LogLoss 0.1883，
     没有 ECE、可靠性图、分段校准曲线 → 无法回答"论文的概率能不能直接
     拿来定阈值做干预"。
  ② **拒答带与干预名单**：产品要的不是点估计，而是
     "名单里 10 个人有 8 个真有问题"的**覆盖保证**。
     需要的是 coverage–precision 曲线（选择性预测 / risk–coverage），
     而不是单一 AUC。

本脚本是**纯后处理**：直接吃各实验已经产出的 OOF 概率，不重训任何模型，
所以几分钟就能跑完，且不会动到已定稿的主结果。

输出：
  results/calibration_selective/{calibration.json, calibration.md}

指标定义：
  ECE        = Σ (n_b/N) · |mean_pred_b − mean_y_b|   （等宽 10 桶 + 等频 10 桶）
  校准斜率   = logistic(y ~ logit(p)) 的斜率（<1 = 过度自信，>1 = 过度保守）
  校准截距   = 同上截距（<0 = 系统性高估）
  AURC       = area under risk–coverage curve（越低越好；选择性预测的质量）
  P@cov      = 覆盖 cov% 时的精确率（产品口径：名单里有几个真问题）
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402


# ---------------------------------------------------------------- 基础指标
def ece_equal_width(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    tot, e = 0.0, []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        w = m.sum() / len(y)
        gap = abs(p[m].mean() - y[m].mean())
        tot += w * gap
        e.append(dict(bin=i, n=int(m.sum()), pred=float(p[m].mean()),
                      obs=float(y[m].mean()), gap=float(gap)))
    return float(tot), e


def ece_equal_freq(y, p, n_bins=10):
    idx = np.argsort(p)
    chunks = np.array_split(idx, n_bins)
    tot, e = 0.0, []
    for i, c in enumerate(chunks):
        if len(c) == 0:
            continue
        w = len(c) / len(y)
        gap = abs(p[c].mean() - y[c].mean())
        tot += w * gap
        e.append(dict(bin=i, n=int(len(c)), pred=float(p[c].mean()),
                      obs=float(y[c].mean()), gap=float(gap)))
    return float(tot), e


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def calib_slope_intercept(y, p):
    z = logit(p).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float('nan'), float('nan')
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(z, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def brier_logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    brier = float(np.mean((p - y) ** 2))
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return brier, ll


# ---------------------------------------------------------------- 选择性预测
def selective(y, p, part):
    """按风险降序取前 cov%，报告精确率/召回率/提升倍数；并给 risk–coverage 曲线。"""
    n = len(y)
    order = np.argsort(-p)
    ys = y[order]
    cum_pos = np.cumsum(ys)
    coverages = np.array([1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100]) / 100.0
    base = float(y.mean())
    rows = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        tp = int(cum_pos[k - 1])
        prec = tp / k
        rec = tp / max(int(y.sum()), 1)
        rows.append(dict(coverage=round(float(c), 4), n=k, tp=tp,
                         precision=round(prec, 4), recall=round(rec, 4),
                         lift=round(prec / base, 2) if base > 0 else None))
    # risk–coverage 曲线（每一点 = 覆盖 k 个样本时，集合内的错误率）
    risk = 1.0 - cum_pos / np.arange(1, n + 1)
    aurc = float(np.mean(risk))
    # 阈值口径（产品里"risk ≥ 阈值"）
    thr_rows = []
    for thr in [0.3, 0.5, 0.7]:
        m = p >= thr
        if m.sum() == 0:
            thr_rows.append(dict(thr=thr, n=0))
            continue
        lift_val = round(float(y[m].mean() / base), 2) if base > 0 else None
        thr_rows.append(dict(thr=thr, n=int(m.sum()),
                             precision=round(float(y[m].mean()), 4),
                             recall=round(float(y[m].sum() / max(int(y.sum()), 1)), 4),
                             lift=lift_val))
    return dict(base_rate=round(base, 4), coverage_table=rows,
                aurc=round(aurc, 5), threshold_table=thr_rows)


# ---------------------------------------------------------------- 交叉拟合重校准
def crossfit_recalibrate(y, p, groups, method='platt', n_splits=5, seed=20261103):
    """按学生分组交叉拟合 Platt / Isotonic，报告**样本外**的校准改善（避免自评乐观）。"""
    from sklearn.isotonic import IsotonicRegression
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = ug.copy()
    rng.shuffle(order)
    fold_of = {g: i % n_splits for i, g in enumerate(order)}
    fo = np.array([fold_of[g] for g in groups])
    p_new = np.zeros_like(p)
    for k in range(n_splits):
        tr, te = fo != k, fo == k
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            p_new[te] = p[te]
            continue
        if method == 'platt':
            lr = LogisticRegression(C=1e6, max_iter=2000).fit(
                logit(p[tr]).reshape(-1, 1), y[tr])
            p_new[te] = lr.predict_proba(logit(p[te]).reshape(-1, 1))[:, 1]
        else:
            iso = IsotonicRegression(out_of_bounds='clip').fit(p[tr], y[tr])
            p_new[te] = iso.predict(p[te])
    return p_new


# ---------------------------------------------------------------- 数据装载
def reconstruct_fwd_rows(ex):
    """复现 run_cs2_fwd.py 中 trunc_raw 的 rows 顺序（sittings 字典插入序）。"""
    ex_sid, ex_par = ex['sid'], ex['part']
    seen, order = set(), []
    for s, p in zip(ex_sid, ex_par):
        k = (int(s), int(p))
        if k not in seen:
            seen.add(k)
            order.append(k)
    return order


def load_specs(ds, root, specs):
    """每个 spec: name, path, oof_key, y来源, part来源, rows来源"""
    out = []
    sid_all = ds['sid']
    for sp in specs:
        f = os.path.join(root, sp['path'])
        if not os.path.exists(f):
            print(f'  [skip] {sp["name"]}: {f} 不存在')
            continue
        d = np.load(f, allow_pickle=True)
        if 'rows' in d:
            rows = d['rows'].astype(int)
            oof = d['oof']
            y = d['y'] if sp.get('y_key') is None else d[sp['y_key']]
            part = d['part'] if sp.get('part_key') is None else d[sp['part_key']]
            sid = sid_all[rows]
        else:
            rows = None
            oof = d[sp['oof_key']]
            y = d[sp['y_key']]
            part = d[sp['part_key']]
            sid = None
        out.append(dict(name=sp['name'], oof=oof.astype(float), y=y.astype(int),
                        part=part.astype(int), sid=sid, rows=rows))
    return out


def main(args):
    ds = sc.load_dataset(args.data)
    specs = [
        dict(name='段级 RF（主模型，整段证据）', path='results/stage_experiment/stage_oof_rf.npz',
             oof_key='rf_full', y_key=None, part_key=None),
        dict(name='坐下级 K=1（第 1 次坐下）', path='results/sitting/sit_k1_feat.npz',
             oof_key=None, y_key=None, part_key=None),
        dict(name='IPSD 前 1 题', path='results/ipsd/ipsd_results.npz',
             oof_key=None, y_key=None, part_key=None),
        dict(name='CS2-FWD RF K1（段内早警·含SELF）', path='results/cs2_fwd/oof_rf_K1.npz',
             oof_key=None, y_key=None, part_key=None),
        dict(name='CS2-FWD 深度 base K1（含SELF）', path='results/cs2_fwd/oof_cs2_base_K1.npz',
             oof_key=None, y_key=None, part_key=None),
        dict(name='CS2-FWD RF K1（剔除SELF·纯净）', path='results/cs2_fwd_clean/oof_rf_K1.npz',
             oof_key=None, y_key=None, part_key=None),
        dict(name='CS2-FWD 深度 base K1（剔除SELF·纯净）', path='results/cs2_fwd_clean/oof_cs2_base_K1.npz',
             oof_key=None, y_key=None, part_key=None),
    ]
    # IPSD 的字段名特殊，单独处理
    dsets = []
    for sp in specs:
        f = os.path.join(args.root, sp['path'])
        if not os.path.exists(f):
            print(f'  [skip] {sp["name"]}: {sp["path"]} 不存在')
            continue
        d = np.load(f, allow_pickle=True)
        if sp['name'].startswith('IPSD'):
            rows = d['rows_e1'].astype(int)
            dsets.append(dict(name=sp['name'], oof=d['oof_e1'].astype(float),
                              y=d['ddrop'].astype(int), part=d['dpart'].astype(int),
                              sid=ds['sid'][rows], rows=rows))
        elif 'rows' in d:
            rows = d['rows'].astype(int)
            dsets.append(dict(name=sp['name'], oof=d['oof'].astype(float),
                              y=d['y'].astype(int), part=d['part'].astype(int),
                              sid=ds['sid'][rows], rows=rows))
        else:
            okey = sp['oof_key'] or 'oof'
            ykey = sp['y_key'] or 'y'
            pkey = sp['part_key'] or 'part'
            if sp['name'].startswith('段级 RF'):
                # stage_oof_rf.npz 与 stage_dataset 同序 → 标签直接取 ds
                dsets.append(dict(name=sp['name'], oof=d[okey].astype(float),
                                  y=ds['drop'].astype(int), part=ds['part_ids'].astype(int),
                                  sid=ds['sid'], rows=np.arange(len(ds['sid']))))
            else:
                dsets.append(dict(name=sp['name'], oof=d[okey].astype(float),
                                  y=d[ykey].astype(int),
                                  part=d[pkey].astype(int), sid=None, rows=None))
    # CS2-FWD 的 OOF 没有 rows 索引 → 用 exercise 顺序重建 sid
    ex = np.load(os.path.join(args.root, 'data/exercise_dataset.npz'), allow_pickle=True)
    fwd_order = reconstruct_fwd_rows(ex)
    part_key = {(int(ds['sid'][i]), int(ds['part_ids'][i])): i for i in range(len(ds['sid']))}
    fwd_rows = []
    for s, p in fwd_order:
        r = part_key.get((s, p))
        if r is not None:
            fwd_rows.append(r)
    for d in dsets:
        if d['sid'] is None and 'FWD' in d['name']:
            rr = np.array(fwd_rows[:len(d['oof'])])
            d['sid'] = ds['sid'][rr]
            d['rows'] = rr

    out = {}
    for d in dsets:
        y, p, part, sid = d['y'], np.clip(d['oof'], 1e-6, 1 - 1e-6), d['part'], d['sid']
        rec = {'n': int(len(y)), 'pos': int(y.sum()), 'base_rate': round(float(y.mean()), 4)}
        pp = sc.per_part_auc(y, p, part)
        rec['auc'] = round(float(roc_auc_score(y, p)), 4)
        rec['pooled'] = round(sc.pooled(pp), 4)
        b, ll = brier_logloss(y, p)
        rec['brier'] = round(b, 4)
        rec['logloss'] = round(ll, 4)
        e10, bins = ece_equal_width(y, p, 10)
        ef10, _ = ece_equal_freq(y, p, 10)
        rec['ece_width10'] = round(e10, 4)
        rec['ece_freq10'] = round(ef10, 4)
        sl, ic = calib_slope_intercept(y, p)
        rec['calib_slope'] = round(sl, 4)
        rec['calib_intercept'] = round(ic, 4)
        # 交叉拟合重校准（样本外）
        if sid is not None:
            for m in ['platt', 'isotonic']:
                pn = crossfit_recalibrate(y, p, sid, m)
                b2, ll2 = brier_logloss(y, pn)
                e2, _ = ece_equal_width(y, pn, 10)
                rec[f'recal_{m}'] = dict(brier=round(b2, 4), logloss=round(ll2, 4),
                                        ece=round(e2, 4))
        # 分段校准
        per_part = {}
        for g in np.unique(part):
            m = part == g
            if m.sum() < 20:
                continue
            e_g, _ = ece_equal_width(y[m], p[m], 5)
            per_part[int(g)] = dict(n=int(m.sum()), base=round(float(y[m].mean()), 4),
                                    pred=round(float(p[m].mean()), 4),
                                    ece5=round(e_g, 4))
        rec['per_part_calibration'] = per_part
        # 选择性预测
        rec['selective'] = selective(y, p, part)
        rec['bins_width10'] = bins
        out[d['name']] = rec
        print(f"  ✓ {d['name']}: pooled={rec['pooled']} ECE10={rec['ece_width10']} "
              f"slope={rec['calib_slope']} AURC={rec['selective']['aurc']} "
              f"P@top5%={rec['selective']['coverage_table'][4]['precision']}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'calibration.json'), 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # markdown
    L = ['# CS1 概率校准 + 选择性预测审计', '',
         f'生成时间：{__import__("datetime").datetime.now():%Y-%m-%d %H:%M}（纯后处理，不重训模型）', '']
    for name, r in out.items():
        L += [f'## {name}', '',
              f'- n={r["n"]}，正例={r["pos"]}，基准率={r["base_rate"]}',
              f'- AUC={r["auc"]}，part 内 pooled={r["pooled"]}',
              f'- Brier={r["brier"]}，LogLoss={r["logloss"]}，'
              f'ECE(等宽10)={r["ece_width10"]}，ECE(等频10)={r["ece_freq10"]}',
              f'- 校准斜率={r["calib_slope"]}，截距={r["calib_intercept"]}']
        for m in ['platt', 'isotonic']:
            if f'recal_{m}' in r:
                q = r[f'recal_{m}']
                L.append(f'- 交叉拟合 {m} 重校准（样本外）：Brier={q["brier"]}，'
                         f'ECE={q["ece"]}，LogLoss={q["logloss"]}')
        s = r['selective']
        L += ['', '| 覆盖 | 人数 | 精确率 | 召回 | 提升 |', '|---:|---:|---:|---:|---:|']
        for row in s['coverage_table']:
            L.append(f'| {row["coverage"]*100:.0f}% | {row["n"]} | {row["precision"]} | '
                     f'{row["recall"]} | {row["lift"]} |')
        L += ['', '阈值口径：' + '；'.join(
            f'risk≥{t["thr"]}: n={t["n"]}, P={t.get("precision")}, R={t.get("recall")}'
            for t in s['threshold_table']), '',
              f'AURC={s["aurc"]}', '']
        if r['per_part_calibration']:
            L += ['| 分段 | n | 基准率 | 平均预测 | 段内ECE(5桶) |', '|---:|---:|---:|---:|---:|']
            for g, v in sorted(r['per_part_calibration'].items()):
                L.append(f'| {g} | {v["n"]} | {v["base"]} | {v["pred"]} | {v["ece5"]} |')
            L.append('')
    with open(os.path.join(args.out, 'calibration.md'), 'w') as f:
        f.write('\n'.join(L))
    print(f'\n完成 → {args.out}/calibration.md', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/home/ubuntu/studentrisk_stage')
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--out', default='results/calibration_selective')
    args = ap.parse_args()
    main(args)
