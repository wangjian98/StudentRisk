#!/usr/bin/env python3
"""
HMM 基线（补齐 EDM 2025 暴露的基线缺口）+ 复现其设置
=================================================================================
背景：Murali, Conati & Poole (EDM 2025) 在**同一个 CS1 MOOC 数据集**上比较
      FUMA / GMM-HMM / LSTM / 分层 dummy，最后窗口准确率：
          HMM 76.97±1.42 > LSTM 71.51±2.05 > Baseline 53.95±2.19 > FUMA 34.65±2.44
      我们的 v10 基线表里没有 HMM → EDM 审稿人必问。

两个 arm：
  Arm 1（本文主协议）：**阶段级** HMM 基线
      · MultinomialHMM（7 类事件符号）· GaussianHMM（seq9 连续通道，无 part 身份）
      · 两个模型分别按 drop=1 / drop=0 训练（照搬其"每类一个 HMM + 先验"的做法）
      · 5 折按学生分组 × 3 seeds，内层选 n_components（不用评估折调参）
      · 主指标：part 内 pooled AUC，与 RF 0.9258 配对 bootstrap
  Arm 2（复现 EDM'25 设置）：**学生级** HMM
      · 每生按时间排序的完整事件序列 → 按 500 事件分箱成符号序列
      · 8 折 CV（与其一致），两个 HMM（fail / pass）+ 先验
      · 输出 accuracy（与其同口径）+ AUC（与我们学生级诚实口径对照）

用法：
  python3 run_hmm_baseline.py --data data/stage_dataset.npz --out results/hmm
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.stage_experiment import stage_common as sc  # noqa: E402


def seq_ids(seq, mask):
    """one-hot → 事件 id 序列（只保留有效位）"""
    ids = seq.argmax(-1)
    out = []
    for i in range(len(ids)):
        m = mask[i] > 0
        v = ids[i][m]
        out.append(v if len(v) >= 2 else np.array([], dtype=int))
    return out


def fit_score_multinomial(seqs_tr, y_tr, seqs_te, K, seed):
    from hmmlearn.hmm import CategoricalHMM
    models = {}
    for cls in (1, 0):
        idx = [i for i in range(len(seqs_tr)) if y_tr[i] == cls and len(seqs_tr[i]) >= K]
        if len(idx) < 10:
            models[cls] = None
            continue
        X = np.concatenate([seqs_tr[i] for i in idx]).reshape(-1, 1)
        lens = [len(seqs_tr[i]) for i in idx]
        m = CategoricalHMM(n_components=K, n_iter=20, tol=1e-3, random_state=seed)
        m.fit(X, lengths=lens)
        models[cls] = m
    prior1 = float(np.mean(y_tr)) if len(y_tr) else 0.5
    out = np.zeros(len(seqs_te))
    for i, s in enumerate(seqs_te):
        if len(s) < 2 or models.get(1) is None or models.get(0) is None:
            out[i] = prior1
            continue
        X = s.reshape(-1, 1)
        try:
            s1 = models[1].score(X) + np.log(max(prior1, 1e-9))
            s0 = models[0].score(X) + np.log(max(1 - prior1, 1e-9))
        except Exception:
            out[i] = prior1
            continue
        out[i] = 1.0 / (1.0 + np.exp(-(s1 - s0)))
    return out


def fit_score_gaussian(Xtr, ytr, Xte, mask_te, K, seed):
    """GaussianHMM：观测 = seq9 连续通道（无 part 身份），按 mask 截断"""
    from hmmlearn.hmm import GaussianHMM
    models = {}
    for cls in (1, 0):
        rows = [i for i in range(len(Xtr)) if ytr[i] == cls]
        if len(rows) < 10:
            models[cls] = None
            continue
        seqs = []
        for i in rows:
            m = mask_tr_[i] > 0
            v = Xtr[i][m]
            if len(v) >= K:
                seqs.append(v)
        if len(seqs) < 10:
            models[cls] = None
            continue
        L = min(len(s) for s in seqs)
        L = max(L, K)
        X = np.concatenate([s[-L:] for s in seqs])
        lens = [L] * len(seqs)
        m = GaussianHMM(n_components=K, covariance_type='diag', n_iter=20, tol=1e-3,
                        random_state=seed, init_params='stmc')
        m.fit(X, lengths=lens)
        models[cls] = m
    prior1 = float(np.mean(ytr))
    out = np.zeros(len(Xte))
    for i in range(len(Xte)):
        m = mask_te[i] > 0
        v = Xte[i][m]
        if len(v) < K or models.get(1) is None or models.get(0) is None:
            out[i] = prior1
            continue
        L = models[1].n_features  # dummy
        try:
            s1 = models[1].score(v) + np.log(max(prior1, 1e-9))
            s0 = models[0].score(v) + np.log(max(1 - prior1, 1e-9))
        except Exception:
            out[i] = prior1
            continue
        out[i] = 1.0 / (1.0 + np.exp(-(s1 - s0)))
    return out


def main(args):
    from hmmlearn.hmm import CategoricalHMM, GaussianHMM
    ds = sc.load_dataset(args.data)
    y = ds['drop'].astype(int)
    part, sid = ds['part'], ds['sid']
    n = len(y)
    seq7, seq9, mask = ds['seq7'], ds['seq9'], ds['mask']
    seeds = [42, 123, 777]
    folds = sc.group_folds(sid, ds['final'], n_splits=5, seed=42)

    seqs = seq_ids(seq7, mask)
    global mask_tr_
    print(f'[stage] 有效序列长度：中位 {int(np.median([len(s) for s in seqs]))} '
          f'最大 {max(len(s) for s in seqs)}', flush=True)

    # ---------- Arm 1：阶段级 HMM（含内层选 K） ----------
    oof = {}
    t0 = time.time()
    for name, kind in (('hmm_multinomial', 'multi'), ('hmm_gaussian', 'gauss')):
        oofv = np.zeros(n)
        for fi, (tr, te) in enumerate(folds):
            # 内层验证：训练折内按学生切 15%
            rng = np.random.default_rng(42 + fi)
            stu = np.unique(sid[tr])
            val_stu = set(rng.choice(stu, size=max(int(0.15 * len(stu)), 5), replace=False))
            m_val = np.array([int(sid[i]) in val_stu for i in tr])
            fit_idx, val_idx = tr[~m_val], tr[m_val]
            best_K, best_score = None, -1
            for K in (2, 3, 4, 5):
                try:
                    if kind == 'multi':
                        pv = fit_score_multinomial([seqs[i] for i in fit_idx],
                                                   y[fit_idx], [seqs[i] for i in val_idx],
                                                   K, seeds[0])
                    else:
                        mask_tr_ = mask[fit_idx]
                        pv = fit_score_gaussian(seq9[fit_idx], y[fit_idx], seq9[val_idx],
                                               mask[val_idx], K, seeds[0])
                except Exception as e:
                    print(f'    K={K} 失败: {e}', flush=True)
                    continue
                if len(set(y[val_idx].tolist())) > 1:
                    sc_ = roc_auc_score(y[val_idx], pv)
                    if sc_ > best_score:
                        best_score, best_K = sc_, K
            if best_K is None:
                best_K = 2
            acc = np.zeros(len(te))
            for sd in seeds:
                if kind == 'multi':
                    acc += fit_score_multinomial([seqs[i] for i in tr], y[tr],
                                                 [seqs[i] for i in te], best_K, sd)
                else:
                    mask_tr_ = mask[tr]
                    acc += fit_score_gaussian(seq9[tr], y[tr], seq9[te], mask[te],
                                             best_K, sd)
            oofv[te] = acc / len(seeds)
            print(f'  [{name}] fold{fi} K={best_K} ({time.time()-t0:.0f}s)', flush=True)
        oof[name] = oofv

    print('\n=== Arm 1 · 阶段级 HMM（主协议）===')
    for k, v in oof.items():
        m = sc.metrics(y, v, part, k)
        print(f'  {k:<16} overall={m["auc"]:.4f} pooled={m["pooled"]:.4f} '
              f'judged={m["judged"]:.4f}')

    # 配对 bootstrap vs RF 基线（用 run_candidates 存的 rf OOF）
    rf_path = os.path.join(os.path.dirname(args.out.rstrip('/')), 'candidates',
                           'oof_candidates_run1.npz')
    if os.path.exists(rf_path):
        Z = np.load(rf_path, allow_pickle=True)
        rf = Z['rf']
        idx_by_g = [np.where(part == g)[0] for g in np.unique(part)]
        rng = np.random.default_rng(20261101)
        boot = [np.concatenate([rng.choice(ii, len(ii), replace=True) for ii in idx_by_g])
                for _ in range(2000)]

        def pooled_of(yy, vv, gg):
            return sc.pooled(sc.per_part_auc(yy, vv, gg))
        base = np.array([pooled_of(y[p], rf[p], part[p]) for p in boot])
        print('\n=== vs RF 基线（配对 bootstrap 2000 次）===')
        for k, v in oof.items():
            obs = pooled_of(y, v, part) - pooled_of(y, rf, part)
            bs = np.array([pooled_of(y[p], v[p], part[p]) - b for p, b in zip(boot, base)])
            print(f'  {k:<16} Δpooled={obs:+.4f} '
                  f'95%CI[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}] '
                  f'P(Δ≤0)={float((bs<=0).mean())*100:.2f}%')
        np.savez_compressed(os.path.join(args.out, 'oof_hmm_stage.npz'),
                            **{k: v for k, v in oof.items()}, y=y, sid=sid, part=part)

    # ---------- Arm 2：学生级复现 EDM'25 设置 ----------
    if os.path.exists(args.stu_seqs):
        S = np.load(args.stu_seqs, allow_pickle=True)
        ids = S['ids']          # 学生级符号序列（对象数组）
        lengths = S['lengths']
        yfail = ds['final'].astype(int)
        stu = np.unique(sid)
        pos = {int(s): i for i, s in enumerate(stu)}
        order = np.argsort(stu)
        stu_sorted = [int(stu[i]) for i in order]
        seqs_stu = [ids[i] for i in order]
        lab = np.array([yfail[sid == s][0] for s in stu_sorted])
        print(f'\n=== Arm 2 · 学生级 HMM（复现 EDM\'25 设置）===')
        print(f'  学生 {len(stu_sorted)} | 符号序列长度中位 '
              f'{int(np.median([len(s) for s in seqs_stu]))}')
        rng = np.random.default_rng(42)
        fold8 = rng.permutation(len(stu_sorted)) % 8
        pred = np.zeros(len(stu_sorted))
        for f in range(8):
            tr = np.where(fold8 != f)[0]
            te = np.where(fold8 == f)[0]
            acc = np.zeros(len(te))
            for K in (3,):   # 复现其 HMM 设置（N∈{2..5}，此处取 3 作代表）
                p = fit_score_multinomial([seqs_stu[i] for i in tr], lab[tr],
                                          [seqs_stu[i] for i in te], K, 42)
                acc += p
            pred[te] = acc
        acc_score = float(((pred > 0.5).astype(int) == lab).mean())
        auc = float(roc_auc_score(lab, pred))
        base_rate = float(lab.mean())
        print(f'  8 折（与其一致）accuracy = {acc_score:.4f}  '
              f'AUC = {auc:.4f}  | 多数类基线 accuracy = {max(base_rate,1-base_rate):.4f}')
        print(f'  对照 EDM\'25 报告值：HMM accuracy = 0.7697 (n=469)')
        np.savez_compressed(os.path.join(args.out, 'hmm_student_level.npz'),
                            pred=pred, lab=lab, stu=np.array(stu_sorted))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/stage_dataset.npz')
    ap.add_argument('--stu-seqs', default='data/student_event_seqs.npz')
    ap.add_argument('--out', default='results/hmm')
    main(ap.parse_args())
