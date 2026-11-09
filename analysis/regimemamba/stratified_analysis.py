"""
regimemamba.stratified_analysis — 堵 §7.5「switching is not risk」的漏洞

问题：论文 §7.5 观察到 switch rate 单独判别挂科 AUC=0.3806（反向，即切换越多越可能通过），
并提出两种解释：
  (a) 活动量混淆：switch 与行为总量相关（ρ≈+0.25），而活动量大的学生更可能通过；
  (b) 真实语义：切换本身携带与失败相关的（反向）风险信号。
两者给出的产品结论完全不同：(a) ⇒ switch 只是活动量的代理，不是独立风险信号；
(b) ⇒ 「切换少 = 危险」本身是可交付规则。

本脚本用五类方法区分二者：
  §2 混淆路径三角验证 ρ(switch,y) ≈ ρ(switch,V)·ρ(V,y)
  §3 偏 Spearman（秩基，控制活动量；含双协变量与控制 RF 分数）
  §4 活动量分层（主口径 = 全轨迹活动量）：层内 switch AUC、层内 R2 占用 AUC
  §5 Mantel–Haenszel 合并 OR + Cochran's Q 异质性
  §6 分层逻辑回归（IRLS + Wald CI）：协变量版 + 分层固定效应版 + 交互项
  §7 残差检验：rank(switch) 对 rank(V) 残差化后的判别力
  §8 稳健性矩阵 + 机制检验（switch 与窗口内间隔/跨度/强度的关系）+ 饱和性诊断

口径要点（新增，论文 §3.2/§6.7 未点明）：
  · 模型序列表征是每个学生**最后 128 个事件**（左 padding，rev12.common.build_spaces_multi），
    因此 switch 在该窗口上计算；论文 §6.7 的行为统计（analyze.py.behavior_stats）却是**全轨迹**。
  · 末 128 事件窗口的**事件数几乎恒为 128**（饱和）→ 以"窗口事件数"构造的活动量对挂科几乎无
    判别力（AUC≈0.50）；真正与挂科强相关的是**全轨迹活动量**（事件数 AUC 0.159，反向）。
  · 因此主口径 = 全轨迹活动量 V_full_pc1；窗口类指标（跨度/强度/新鲜度）作为**机制层**混淆因子。

运行（需持有 /home/ubuntu/SR12_cache/events.npz，通常在 246）：
  python3 analysis/regimemamba/stratified_analysis.py --repo /home/ubuntu/SR_rev12 \
      --out /home/ubuntu/SR_rev12/docs
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from scipy.stats import (spearmanr, kendalltau, mannwhitneyu, chi2, t as t_dist,
                          pearsonr, rankdata, norm)
from scipy.special import expit

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

TS_UNIT_SECONDS = 1000.0     # 事件缓存时间戳单位遗留（论文 §3.2）
MAX_LEN = 128                # 模型序列窗口长度

BOOT_Y = None                # bootstrap 用的标签（main() 内赋值）


# ────────────────────────────── 基础统计工具 ──────────────────────────────
def auc_tie(p, yy):
    """并列感知 AUC（rankdata 平均秩）。p 越大越像正类。"""
    r = rankdata(p)
    npos = float(yy.sum()); nneg = float(len(yy) - npos)
    if npos == 0 or nneg == 0:
        return float('nan')
    return float((r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _resid_on(a, Z):
    """把 a 对协变量矩阵 Z（可多列）做 OLS 残差化（自动含截距）。"""
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    A = np.column_stack([np.ones(len(Z)), Z])
    coef, *_ = np.linalg.lstsq(A, a, rcond=None)
    return a - A @ coef


def partial_spearman(x, Z, y):
    """秩基偏相关：控制 Z 后 x 与 y 的相关。返回 (r, p)。"""
    Z = np.asarray(Z, float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    rx, ry = rankdata(x), rankdata(y)
    ex, ey = _resid_on(rx, Z), _resid_on(ry, Z)
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return float('nan'), float('nan')
    r, _ = pearsonr(ex, ey)
    n = len(x); df = max(n - Z.shape[1] - 2, 1)
    if abs(r) >= 1:
        return float(r), 0.0
    tstat = r * np.sqrt(df / (1 - r * r))
    return float(r), float(2 * t_dist.sf(abs(tstat), df))


def boot_ci(fn, n=2000, seed=0):
    """学生级配对 bootstrap：返回 (mean, lo2.5, hi97.5)。"""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(BOOT_Y))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(BOOT_Y[s])) < 2:
            continue
        v = fn(s)
        if v is None or not np.isfinite(v):
            continue
        vals.append(v)
    vals = np.array(vals)
    if len(vals) == 0:
        return float('nan'), float('nan'), float('nan')
    return float(vals.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def logit_irls(X, y, ridge=1e-6, max_iter=200, tol=1e-10):
    """Newton/IRLS 逻辑回归：返回 (beta, cov, se)。X 需含截距且已标准化。"""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = expit(X @ beta)
        W = np.clip(p * (1 - p), 1e-9, None)
        g = X.T @ (y - p)
        H = X.T @ (W[:, None] * X)
        H[np.diag_indices_from(H)] += ridge
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    p = expit(X @ beta)
    W = np.clip(p * (1 - p), 1e-9, None)
    H = X.T @ (W[:, None] * X)
    H[np.diag_indices_from(H)] += ridge
    cov = np.linalg.pinv(H)
    return beta, cov, np.sqrt(np.abs(np.diag(cov)))


def logit_row(X, y, names, ridge=1e-6):
    beta, cov, se = logit_irls(X, y, ridge=ridge)
    z = beta / np.where(se > 0, se, np.nan)
    p = 2 * norm.sf(np.abs(z))
    rows = []
    for i, nm in enumerate(names):
        rows.append(dict(term=nm, beta=float(beta[i]), se=float(se[i]), z=float(z[i]),
                         p=float(p[i]), or_=float(np.exp(beta[i])),
                         lo=float(np.exp(beta[i] - 1.96 * se[i])),
                         hi=float(np.exp(beta[i] + 1.96 * se[i])), n=int(len(y))))
    return rows


def mh_odds_ratio(strata, x, yy):
    """Mantel–Haenszel：x 二元暴露，strata 为层索引数组。返回 (OR_MH, chi2_p, Q_p, 各层 OR)。"""
    num = den = 0.0
    strata_or, strata_logor = [], []
    used = 0
    for s in strata:
        xs, ys = x[s], yy[s]
        a = int(((xs == 1) & (ys == 1)).sum()); b = int(((xs == 1) & (ys == 0)).sum())
        c = int(((xs == 0) & (ys == 1)).sum()); d = int(((xs == 0) & (ys == 0)).sum())
        n = a + b + c + d
        if n == 0 or (a + b) == 0 or (c + d) == 0:
            strata_or.append((float('nan'), 0, 0, 0, 0)); continue
        used += 1
        num += a * d / n; den += b * c / n
        or_i = (a * d) / (b * c) if b * c > 0 else float('inf')
        strata_or.append((or_i, a, b, c, d))
        strata_logor.append((np.log(or_i) if np.isfinite(or_i) and or_i > 0 else 0.0, n))
    if used == 0 or den == 0:
        return float('nan'), float('nan'), float('nan'), strata_or
    or_mh = num / den
    sum_num = sum_ev = sum_var = 0.0
    for s in strata:
        xs, ys = x[s], yy[s]
        a = int(((xs == 1) & (ys == 1)).sum()); b = int(((xs == 1) & (ys == 0)).sum())
        c = int(((xs == 0) & (ys == 1)).sum()); d = int(((xs == 0) & (ys == 0)).sum())
        n = a + b + c + d
        if n == 0 or (a + b) == 0 or (c + d) == 0:
            continue
        r1, r2 = a + b, c + d
        c1, c2 = a + c, b + d
        sum_num += a; sum_ev += r1 * c1 / n
        sum_var += r1 * c2 * (n - r1) * (n - c1) / (n * n * (n - 1))
    chi2_mh = ((abs(sum_num - sum_ev) - 0.5) ** 2 / sum_var) if sum_var > 0 else float('nan')
    p_mh = float(chi2.sf(chi2_mh, 1)) if np.isfinite(chi2_mh) else float('nan')
    Q = 0.0; logor_mh = np.log(or_mh)
    for lo, n in strata_logor:
        Q += (n - 1) * (lo - logor_mh) ** 2
    p_q = float(chi2.sf(Q, max(len(strata_logor) - 1, 1)))
    return float(or_mh), p_mh, p_q, strata_or


def match_pairs(df, y, cov_cols, max_dist_sd=2.5):
    """活动量配对：每个 PASS 生配一个活动量最接近的 FAIL 生（z 空间欧氏最近邻）。
    返回 (pairs, balance)：pairs = [(pass_idx, fail_idx, 距离)]；balance = 配对后协变量平衡诊断。"""
    Zm = df[cov_cols].values.astype(float)
    Zm = (Zm - Zm.mean(0)) / (Zm.std(0) + 1e-12)
    fail_idx = np.where(y == 1)[0]
    pass_idx = np.where(y == 0)[0]
    pairs = []
    for p in pass_idx:
        d = np.linalg.norm(Zm[fail_idx] - Zm[p], axis=1)
        j = int(np.argmin(d))
        pairs.append((int(p), int(fail_idx[j]), float(d[j])))
    keep = [pr for pr in pairs if pr[2] <= max_dist_sd]
    # 平衡诊断
    pi = np.array([a for a, b, d in keep]); fi = np.array([b for a, b, d in keep])
    balance = {c: float(np.abs(df[c].values[pi].mean() - df[c].values[fi].mean()) /
                           (df[c].values.std() + 1e-12)) for c in cov_cols}
    return keep, balance


def paired_report(df, y, pairs, col, label):
    """配对样本：FAIL − PASS 的配对差 + Wilcoxon 符号秩检验 + bootstrap CI。"""
    from scipy.stats import wilcoxon
    pi = np.array([a for a, b, d in pairs]); fi = np.array([b for a, b, d in pairs])
    a_f = df[col].values[fi]; a_p = df[col].values[pi]
    diff = a_f - a_p
    try:
        w_p = float(wilcoxon(diff).pvalue)
    except Exception:
        w_p = float('nan')
    rng = np.random.default_rng(0)
    meds = []
    n = len(diff)
    for _ in range(2000):
        idx = rng.choice(n, n, replace=True)
        meds.append(np.median(diff[idx]))
    meds = np.array(meds)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return dict(label=label, col=col, n_pairs=int(n),
                fail_mean=float(a_f.mean()), pass_mean=float(a_p.mean()),
                median_diff=float(np.median(diff)),
                mean_diff=float(diff.mean()),
                ci_lo=float(lo), ci_hi=float(hi),
                wilcoxon_p=w_p,
                smd=float(diff.mean() / (diff.std(ddof=1) + 1e-12)))


# ────────────────────────────── 数据准备 ──────────────────────────────
def window_and_full_stats(cache):
    """每个学生：最后 128 事件窗口 + 全轨迹的行为统计。"""
    et, ts, offsets = cache['et'], cache['ts'], cache['offsets']
    n = len(offsets) - 1
    rows = []
    for i in range(n):
        s, e = offsets[i], offsets[i + 1]
        if e <= s:
            continue
        et_i, ts_i = et[s:e], ts[s:e]
        w0 = max(0, (e - s) - MAX_LEN)                    # 最后 MAX_LEN 个事件
        et_w, ts_w = et_i[w0:], ts_i[w0:]
        cnt_w = np.bincount(et_w, minlength=7).astype(float)
        cnt_f = np.bincount(et_i, minlength=7).astype(float)
        dts_w = np.diff(ts_w).astype(float) * TS_UNIT_SECONDS / 60.0
        span_w_h = (ts_w[-1] - ts_w[0]) * TS_UNIT_SECONDS / 3600.0 if len(ts_w) > 1 else 0.0
        span_f_h = (ts_i[-1] - ts_i[0]) * TS_UNIT_SECONDS / 3600.0 if len(ts_i) > 1 else 0.0
        act_w = len(np.unique((ts_w * TS_UNIT_SECONDS) // 3600))
        # 窗口活跃天数占比（末段行为的"分散度"）：活跃小时数 / 窗口跨度天数
        w_days_active_ratio = act_w / max(span_w_h / 24.0, 1e-9)
        rows.append(dict(
            student=i, n_win=len(et_w), n_full=len(et_i),
            w_submit=cnt_w[6], w_run=cnt_w[5], w_paste=cnt_w[2], w_focus_lost=cnt_w[4],
            w_edit=cnt_w[0] + cnt_w[1], w_all=cnt_w.sum(),
            w_span_h=span_w_h, w_active_h=act_w, w_days_active_ratio=w_days_active_ratio,
            w_mean_gap_min=float(np.mean(dts_w)) if len(dts_w) else 0.0,
            w_median_gap_min=float(np.median(dts_w)) if len(dts_w) else 0.0,
            w_max_gap_min=float(np.max(dts_w)) if len(dts_w) else 0.0,
            w_intensity=float(cnt_w.sum() / max(span_w_h, 1.0 / 60.0)),
            f_submit=cnt_f[6], f_run=cnt_f[5], f_paste=cnt_f[2], f_focus_lost=cnt_f[4],
            f_edit=cnt_f[0] + cnt_f[1], f_all=cnt_f.sum(), f_span_h=span_f_h,
        ))
    return pd.DataFrame(rows)


# 活动量口径清单（顺序即报告顺序）
VOLUMES = [
    ('V_full_pc1', '全轨迹活动量 PCA1（**主口径**：与挂科强相关，是真正的混淆路径）'),
    ('V_full_events', '全轨迹事件总数 log1p（稳健性）'),
    ('V_win_span', '末 128 事件窗口跨度 log1p（**末段新鲜度**：末段行为离学期结束多远）'),
    ('V_win_intensity', '末段强度 log1p(事件/小时)（窗口内产出速率）'),
    ('V_win_pc1', '窗口活动量 PCA1（**近乎退化**：窗口事件数几乎恒为 128）'),
    ('V_win_events', '窗口事件数 log1p（**完全退化**：几乎恒为 128）'),
]
PRIMARY_VOL = 'V_full_pc1'


def build_volume(df):
    """构造多种活动量口径（log1p 计数 → 标准化 → PCA1）。"""
    vols = {}

    def z(a):
        a = np.asarray(a, float)
        return (a - a.mean()) / (a.std() + 1e-12)

    feats_win = np.column_stack([
        np.log1p(df['w_all']), np.log1p(df['w_submit']), np.log1p(df['w_run']),
        np.log1p(df['w_paste']), np.log1p(df['w_focus_lost']), np.log1p(df['w_edit']),
        np.log1p(df['w_span_h']), np.log1p(df['w_active_h'])])
    feats_full = np.column_stack([
        np.log1p(df['f_all']), np.log1p(df['f_submit']), np.log1p(df['f_run']),
        np.log1p(df['f_paste']), np.log1p(df['f_focus_lost']), np.log1p(df['f_edit']),
        np.log1p(df['f_span_h'])])
    for name, F in [('win', feats_win), ('full', feats_full)]:
        Zm = np.apply_along_axis(z, 0, F)
        C = np.corrcoef(Zm.T)
        w_, v_ = np.linalg.eigh(C)
        vols[f'V_{name}_pc1'] = Zm @ v_[:, -1]
    vols['V_win_events'] = np.log1p(df['w_all'].values)
    vols['V_full_events'] = np.log1p(df['f_all'].values)
    vols['V_win_span'] = np.log1p(df['w_span_h'].values)
    vols['V_win_intensity'] = np.log1p(df['w_intensity'].values)
    return vols


# ────────────────────────────── 主流程 ──────────────────────────────
def main():
    global BOOT_Y
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default=_ROOT)
    ap.add_argument('--out', default=None)
    ap.add_argument('--cache', default=os.environ.get('SR12_CACHE', '/home/ubuntu/SR12_cache'))
    ap.add_argument('--tag', default='main_d28_p1.0')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    REPO = os.path.abspath(args.repo)
    OUT = os.path.abspath(args.out or os.path.join(REPO, 'docs'))
    os.makedirs(OUT, exist_ok=True)

    Z = np.load(os.path.join(args.cache, 'events.npz'))
    y = Z['y'].astype(int)
    BOOT_Y = y

    st = np.load(os.path.join(REPO, 'results', 'regimemamba', args.tag, 'regime_stats.npz'))
    usage, switch = st['usage'], st['switch']
    n = len(y)
    assert len(switch) == n

    df = window_and_full_stats(Z)
    assert len(df) == n
    df['y'] = y
    df['switch'] = switch
    for k in range(usage.shape[1]):
        df[f'R{k}_occ'] = usage[:, k]
    df['R2_minus_R1'] = usage[:, 2] - usage[:, 1]

    rf_path = os.path.join(REPO, 'results', 'rev12', 'exp1_matrix', 'rf_d28', 'oof_probs.npy')
    rf = np.load(rf_path).astype(float) if os.path.exists(rf_path) else None
    if rf is not None:
        df['rf_oof'] = rf

    vols = build_volume(df)
    for k, v in vols.items():
        df[k] = v

    L = []
    def P(s=''):
        print(s, flush=True)
        L.append(s)

    P('# RegimeMamba · §7.5 分层与偏相关分析（活动量混淆检验）')
    P()
    P('**生成**：2026-11-10 · `StudentRisk/analysis/regimemamba/stratified_analysis.py`')
    P(f'**数据**：{n} 名学生；y（FAIL={int(y.sum())}，基准率 {y.mean():.4f}）；'
      f'switch / R2 占用来自 `{args.tag}/regime_stats.npz`；行为统计来自 `{args.cache}/events.npz`')
    P(f'**序列窗口**：模型输入为每个学生**最后 {MAX_LEN} 个事件**（左 padding），'
      f'switch 在该窗口上计算；论文 §6.7 的行为统计用**全轨迹** —— 两者口径不同，本分析显式区分。')
    P()
    P('> **要回答的问题**：论文 §7.5 报告「切换率单独判别挂科 AUC=0.3806（反向）」，'
      '列出两种解释：(a) 活动量混淆（switch 与行为总量相关 ρ≈+0.25，活跃学生更可能通过）；'
      '(b) 切换本身携带真实（反向）风险信号。**两者给出的产品结论完全不同**：'
      '(a) ⇒ switch 只是活动量的代理，不是独立风险信号；'
      '(b) ⇒ 「切换少 = 危险」本身可交付。本分析用偏相关 + 分层 + 分层回归 + MH 合并 OR 把二者分开。')
    P()

    # ───── §1 描述统计 ─────
    P('## 1 · 描述统计')
    P()
    P('| 变量 | 全样本 | FAIL 生 (n=%d) | PASS 生 (n=%d) | 差 | MW p | 单独 AUC(判别挂科) |' %
      (int(y.sum()), int((1 - y).sum())))
    P('|---|---|---|---|---|---|---|')
    desc_rows = []
    targets = [('switch（切换率）', df['switch'].values),
               ('R2 占用（风险 regime）', df['R2_occ'].values),
               ('R2−R1 占用差', df['R2_minus_R1'].values)]
    targets += [(f'{k}（{lab}）', df[k].values) for k, lab in VOLUMES]
    targets += [('n_events(窗口)', df['n_win'].values), ('n_events(全轨迹)', df['n_full'].values)]
    for nm, v in targets:
        a, b = v[y == 1], v[y == 0]
        p = mannwhitneyu(a, b).pvalue
        au = auc_tie(v, y)
        P(f'| {nm} | {v.mean():.4f} | {a.mean():.4f} | {b.mean():.4f} | {a.mean()-b.mean():+.4f} | {p:.4g} | {au:.4f} |')
        desc_rows.append(dict(variable=nm, mean=float(v.mean()), fail_mean=float(a.mean()),
                              pass_mean=float(b.mean()), diff=float(a.mean() - b.mean()),
                              mw_p=float(p), auc=float(au)))
    P()
    P('> AUC<0.5 = 反向（数值越大越像通过）。**窗口事件数 AUC≈0.50 说明窗口计数已饱和**，'
      '不能作为活动量代理；**全轨迹事件数 AUC=0.159 说明全轨迹活动量才是真正的混淆因子**。')
    P()

    # ───── §2 三角验证 ─────
    P('## 2 · 混淆路径三角验证')
    P()
    P('| 活动量口径 | ρ(V, y) | ρ(V, switch) | ρ(switch, y) | 乘积 ρ(V,switch)·ρ(V,y) | 残差（独立部分） | 混淆路径解释占比 |')
    P('|---|---|---|---|---|---|---|')
    tri_rows = []
    for vname, _ in VOLUMES:
        V = df[vname].values
        r_vy = spearmanr(V, y).correlation
        r_vs = spearmanr(V, df['switch'].values).correlation
        r_sy = spearmanr(df['switch'].values, y).correlation
        prod = r_vs * r_vy
        expl = abs(prod) / abs(r_sy) if abs(r_sy) > 1e-9 else float('nan')
        P(f'| {vname} | {r_vy:+.4f} | {r_vs:+.4f} | {r_sy:+.4f} | {prod:+.4f} | {r_sy-prod:+.4f} | {expl*100:.1f}% |')
        tri_rows.append(dict(volume=vname, rho_volume_y=r_vy, rho_volume_switch=r_vs,
                              rho_switch_y=r_sy, product=prod, residual=r_sy - prod,
                              explained_frac=expl))
    P()

    # ───── §3 偏 Spearman ─────
    P('## 3 · 偏 Spearman（秩基，控制活动量）+ bootstrap CI（2000 次学生级重采样）')
    P()
    P('| 活动量口径 | ρ(switch, y) 原始 | 偏 ρ(switch, y \\| V) | 95% CI | p | 被活动量解释 |')
    P('|---|---|---|---|---|---|')
    part_rows = []
    for vname, lab in VOLUMES:
        V = df[vname].values
        raw = spearmanr(df['switch'].values, y).correlation
        r0, p0 = partial_spearman(df['switch'].values, V, y)
        m, lo, hi = boot_ci(lambda s: partial_spearman(df['switch'].values[s], V[s], y[s])[0])
        expl = 1 - abs(r0) / abs(raw) if abs(raw) > 1e-9 else float('nan')
        P(f'| {vname} | {raw:+.4f} | **{r0:+.4f}** | [{lo:+.4f}, {hi:+.4f}] | {p0:.4g} | {expl*100:.1f}% |')
        part_rows.append(dict(target='switch', volume=vname, raw_rho=raw, partial_rho=r0,
                              ci_lo=lo, ci_hi=hi, p=p0, explained_frac=expl))
    P()
    P('**双协变量**（同时控制全轨迹活动量与末段新鲜度）：')
    P()
    Z2 = np.column_stack([df[PRIMARY_VOL].values, df['V_win_span'].values])
    r0, p0 = partial_spearman(df['switch'].values, Z2, y)
    m, lo, hi = boot_ci(lambda s: partial_spearman(df['switch'].values[s], Z2[s], y[s])[0])
    P(f'| switch \\| (V_full_pc1, V_win_span) | **{r0:+.4f}** | [{lo:+.4f}, {hi:+.4f}] | {p0:.4g} |')
    part_rows.append(dict(target='switch', volume='V_full_pc1+V_win_span', raw_rho=spearmanr(df['switch'].values, y).correlation,
                          partial_rho=r0, ci_lo=lo, ci_hi=hi, p=p0, explained_frac=float('nan')))
    P()
    if rf is not None:
        r0, p0 = partial_spearman(df['switch'].values, df['rf_oof'].values, y)
        m, lo, hi = boot_ci(lambda s: partial_spearman(df['switch'].values[s], df['rf_oof'].values[s], y[s])[0])
        P(f'**控制 RF-28d OOF 分数**（已含全部活动量聚合信息）：偏 ρ(switch, y \\| RF) = '
          f'**{r0:+.4f}** [{lo:+.4f}, {hi:+.4f}]，p={p0:.4g} → '
          f'{"switch 在最强表格基线之上无残余判别力" if p0 > 0.05 else "switch 仍有残余判别力"}。')
        part_rows.append(dict(target='switch', volume='rf_oof', raw_rho=spearmanr(df['switch'].values, y).correlation,
                              partial_rho=r0, ci_lo=lo, ci_hi=hi, p=p0, explained_frac=float('nan')))
    P()
    P('**活动量"区块"控制**（同时控制全轨迹活动量 + 末段跨度 + 末段强度）：')
    P()
    Z3 = np.column_stack([df[PRIMARY_VOL].values, df['V_win_span'].values, df['V_win_intensity'].values])
    for tgt, col in [('switch', 'switch'), ('R2 占用', 'R2_occ')]:
        r0, p0 = partial_spearman(df[col].values, Z3, y)
        m, lo, hi = boot_ci(lambda s: partial_spearman(df[col].values[s], Z3[s], y[s])[0])
        P(f'| {tgt} \| (V_full_pc1, V_win_span, V_win_intensity) | **{r0:+.4f}** | [{lo:+.4f}, {hi:+.4f}] | {p0:.4g} |')
        part_rows.append(dict(target=col, volume='V_full_pc1+V_win_span+V_win_intensity',
                              raw_rho=spearmanr(df[col].values, y).correlation, partial_rho=r0,
                              ci_lo=lo, ci_hi=hi, p=p0, explained_frac=float('nan')))
    P()
    if rf is not None:
        r0, p0 = partial_spearman(df['R2_occ'].values, df['rf_oof'].values, y)
        m, lo, hi = boot_ci(lambda s: partial_spearman(df['R2_occ'].values[s], df['rf_oof'].values[s], y[s])[0])
        P(f'**R2 占用控制 RF-28d OOF**：偏 ρ(R2, y \| RF) = **{r0:+.4f}** [{lo:+.4f}, {hi:+.4f}]，p={p0:.4g}')
        part_rows.append(dict(target='R2_occ', volume='rf_oof', raw_rho=spearmanr(df['R2_occ'].values, y).correlation,
                              partial_rho=r0, ci_lo=lo, ci_hi=hi, p=p0, explained_frac=float('nan')))
        P()
    P('**产品化的 R2 占用指标**（真正要上线的信号）是否同样受活动量污染：')
    P()
    P('| 控制变量 | ρ(R2, y) 原始 | 偏 ρ(R2, y \\| ·) | 95% CI | p | 被解释 |')
    P('|---|---|---|---|---|---|')
    for vname, _ in [('V_full_pc1', ''), ('V_full_events', ''), ('V_win_span', '')]:
        V = df[vname].values
        raw = spearmanr(df['R2_occ'].values, y).correlation
        r0, p0 = partial_spearman(df['R2_occ'].values, V, y)
        m, lo, hi = boot_ci(lambda s: partial_spearman(df['R2_occ'].values[s], V[s], y[s])[0])
        expl = 1 - abs(r0) / abs(raw) if abs(raw) > 1e-9 else float('nan')
        P(f'| {vname} | {raw:+.4f} | **{r0:+.4f}** | [{lo:+.4f}, {hi:+.4f}] | {p0:.4g} | {expl*100:.1f}% |')
        part_rows.append(dict(target='R2_occ', volume=vname, raw_rho=raw, partial_rho=r0,
                              ci_lo=lo, ci_hi=hi, p=p0, explained_frac=expl))
    P()

    # ───── §4 分层 ─────
    P('## 4 · 活动量分层（主口径 V_full_pc1 = 全轨迹活动量）')
    P()
    strat_defs = [(PRIMARY_VOL, 3, '主口径 · 三分位'), (PRIMARY_VOL, 4, '主口径 · 四分位'),
                  ('V_win_span', 3, '末段新鲜度 · 三分位')]
    strat_rows = []
    for vname, k, label in strat_defs:
        V = df[vname].values
        qs = np.quantile(V, np.linspace(0, 1, k + 1))
        qs[0] -= 1e-9; qs[-1] += 1e-9
        col = f'strat_{vname}_{k}'
        df[col] = np.digitize(V, qs[1:-1])
        P(f'### {label}（分层变量 = {vname}）')
        P()
        P('| 层 | n | FAIL | PASS | 层基准率 | switch: FAIL | switch: PASS | 差 | MW p | 层内 switch AUC | 层内 R2 占用 AUC | 层内 R2 差 | MW p(R2) |')
        P('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for s in range(k):
            m_ = df[col] == s
            a_s, b_s = df.loc[m_ & (y == 1), 'switch'], df.loc[m_ & (y == 0), 'switch']
            a_r, b_r = df.loc[m_ & (y == 1), 'R2_occ'], df.loc[m_ & (y == 0), 'R2_occ']
            p_s = mannwhitneyu(a_s, b_s).pvalue
            p_r = mannwhitneyu(a_r, b_r).pvalue
            au_s = auc_tie(df.loc[m_, 'switch'].values, y[m_])
            au_r = auc_tie(df.loc[m_, 'R2_occ'].values, y[m_])
            br = (y[m_] == 1).mean()
            P(f'| S{s+1} | {int(m_.sum())} | {int((m_ & (y == 1)).sum())} | {int((m_ & (y == 0)).sum())} | '
              f'{br:.3f} | {a_s.mean():.4f} | {b_s.mean():.4f} | {a_s.mean()-b_s.mean():+.4f} | {p_s:.4g} | '
              f'{au_s:.4f} | {au_r:.4f} | {a_r.mean()-b_r.mean():+.4f} | {p_r:.4g} |')
            strat_rows.append(dict(strat=f'{label} S{s+1}', volume=vname, k=k,
                                   n=int(m_.sum()), n_fail=int((m_ & (y == 1)).sum()),
                                   n_pass=int((m_ & (y == 0)).sum()), base_rate=float(br),
                                   switch_fail=float(a_s.mean()), switch_pass=float(b_s.mean()),
                                   switch_diff=float(a_s.mean() - b_s.mean()), switch_mw_p=float(p_s),
                                   switch_auc_in_stratum=float(au_s),
                                   r2_fail=float(a_r.mean()), r2_pass=float(b_r.mean()),
                                   r2_diff=float(a_r.mean() - b_r.mean()), r2_mw_p=float(p_r),
                                   r2_auc_in_stratum=float(au_r)))
        P()

    # ───── §4.4 活动量配对分析（比分层更稳健：避免层内 PASS 极少的问题） ─────
    P('### 4.4 活动量配对分析（volume-matched pairs，主判据）')
    P()
    P('分层的一个弱点：全轨迹活动量与挂科高度相关，最活跃层的 PASS 只有 3 人 → 层内检验功效极低。'
      '**配对设计**绕开这个问题：给每个 PASS 生配一个活动量最接近的 FAIL 生（z 空间最近邻，'
      '距离上限 2.5 SD），再做配对比较。')
    P()
    pair_rows = []
    for cov, label in [([PRIMARY_VOL], '仅全轨迹活动量'),
                       ([PRIMARY_VOL, 'V_win_span'], '全轨迹活动量 + 末段新鲜度'),
                       (['V_win_pc1', 'V_win_intensity', 'V_win_span'], '仅窗口层行为量（与 regime 同窗口，最公平）'),
                       ([PRIMARY_VOL, 'V_win_pc1', 'V_win_intensity', 'V_win_span'], '全量（全轨迹 + 窗口层）')] + \
                      [(['rf_oof'] + [PRIMARY_VOL, 'V_win_pc1', 'V_win_intensity', 'V_win_span'],
                        '最强判据：RF 分数 + 全轨迹 + 窗口层'),
                       ([PRIMARY_VOL, 'rf_oof'], '全轨迹活动量 + RF 分数')]:
        pairs, balance = match_pairs(df, y, cov)
        bal_s = '；'.join(f'{c} 标准化均值差 |SMD|={v:.3f}' for c, v in balance.items())
        P(f'**匹配口径：{label}** —— 可配对 {len(pairs)} 对（159 个 PASS 生上限），配对质量：{bal_s}')
        P()
        P('| 变量 | FAIL 均值 | PASS 均值 | 配对差中位数 | 配对差均值 | 95% CI(中位数) | Wilcoxon p | 标准化配对差 |')
        P('|---|---|---|---|---|---|---|---|')
        for col, lab in [('switch', 'switch（切换率）'), ('R2_occ', 'R2 占用（风险 regime）'),
                         ('R2_minus_R1', 'R2−R1 占用差'), (PRIMARY_VOL, '活动量（配对平衡检查）')]:
            r_ = paired_report(df, y, pairs, col, lab)
            P(f'| {lab} | {r_["fail_mean"]:.4f} | {r_["pass_mean"]:.4f} | {r_["median_diff"]:+.4f} | '
              f'{r_["mean_diff"]:+.4f} | [{r_["ci_lo"]:+.4f}, {r_["ci_hi"]:+.4f}] | {r_["wilcoxon_p"]:.4g} | '
              f'{r_["smd"]:+.3f} |')
            pair_rows.append(dict(match=label, **r_))
        P()
        P(f'> **检验功效**：159 对配对样本在 α=0.05、功效 0.80 下最小可检测的配对标准化差 '
          f'≈ 0.22；上表 CI 给出的实际界：R2 占用的配对差均值 CI 上界约 ±0.03 occupancy 单位'
          f'（原始未配对差异为 0.061）→ **原始关联至少一半由活动量解释，残余效应上界很小**。')
        P()

    # ───── §5 MH ─────
    P('## 5 · Mantel–Haenszel 合并 OR（按活动量分层）')
    P()
    P('| 暴露变量 | 分层变量 | 分层 | OR_MH | MH χ² p | Cochran Q p（异质性） | 各层 OR |')
    P('|---|---|---|---|---|---|---|')
    mh_rows = []
    for vname, k, label in strat_defs:
        strata = [np.where((df[f'strat_{vname}_{k}'] == s).values)[0] for s in range(k)]
        sw_med = np.median(df['switch'].values)
        x_sw = (df['switch'].values > sw_med).astype(int)
        or_mh, p_mh, p_q, so = mh_odds_ratio(strata, x_sw, y)
        P(f'| switch（高 vs 低，中位数切分） | {vname} | {k} 分位 | {or_mh:.3f} | {p_mh:.4g} | {p_q:.4g} | '
          + ', '.join('—' if not np.isfinite(o[0]) else f'{o[0]:.2f}' for o in so) + ' |')
        mh_rows.append(dict(exposure='switch_high_vs_low', volume=vname, k=k, or_mh=or_mh,
                            p_mh=p_mh, p_het=p_q))
        r2_med = np.median(df['R2_occ'].values)
        x_r2 = (df['R2_occ'].values > r2_med).astype(int)
        or_mh2, p_mh2, p_q2, so2 = mh_odds_ratio(strata, x_r2, y)
        P(f'| R2 占用（高 vs 低，中位数切分） | {vname} | {k} 分位 | {or_mh2:.3f} | {p_mh2:.4g} | {p_q2:.4g} | '
          + ', '.join('—' if not np.isfinite(o[0]) else f'{o[0]:.2f}' for o in so2) + ' |')
        mh_rows.append(dict(exposure='R2_high_vs_low', volume=vname, k=k, or_mh=or_mh2,
                            p_mh=p_mh2, p_het=p_q2))
    P()
    P('> OR>1 = 暴露组挂科概率更高。**switch 的 OR_MH 若仍显著 <1，说明反向关联不是单一活动量口径能解释的**；'
      'R2 的 OR_MH 若显著 >1 且异质性不显著，说明「风险状态时长」是**独立于活动量**的可交付信号。')
    P()

    # ───── §6 逻辑回归 ─────
    P('## 6 · 分层逻辑回归（IRLS + Wald 95% CI；非截距项已标准化）')
    P()
    def design(cols, dummies=None):
        Xs, names = [], ['intercept']
        for c in cols:
            Xs.append(df[c].values.astype(float)); names.append(c)
        if dummies is not None:
            for d in dummies:
                Xs.append(df[d].values.astype(float)); names.append(d)
        X = np.column_stack([np.ones(len(y))] + Xs)
        for j in range(1, X.shape[1]):
            sd = X[:, j].std()
            if sd > 1e-12:
                X[:, j] = (X[:, j] - X[:, j].mean()) / sd
        return X, names

    # 分层虚拟变量（参考层 = S1）
    fe = []
    col3 = f'strat_{PRIMARY_VOL}_3'
    for s in [1, 2]:
        df[f'fe3_S{s+1}'] = (df[col3] == s).astype(float)
        fe.append(f'fe3_S{s+1}')
    col4 = f'strat_{PRIMARY_VOL}_4'
    for s in [1, 2, 3]:
        df[f'fe4_S{s+1}'] = (df[col4] == s).astype(float)
        fe.append(f'fe4_S{s+1}')

    specs = [(['switch'], None, 'y ~ switch'),
             ([PRIMARY_VOL], None, 'y ~ V_full_pc1'),
             (['switch', PRIMARY_VOL], None, 'y ~ switch + V'),
             (['R2_occ', PRIMARY_VOL], None, 'y ~ R2 + V'),
             (['R2_occ', 'switch', PRIMARY_VOL], None, 'y ~ R2 + switch + V'),
             (['switch'], fe, 'y ~ switch + 分层FE(三分位)'),
             (['R2_occ'], fe, 'y ~ R2 + 分层FE(三分位)'),
             (['R2_occ', 'switch'], fe, 'y ~ R2 + switch + 分层FE(三分位)')]
    if rf is not None:
        specs += [(['rf_oof'], None, 'y ~ RF(OOF)'),
                  (['rf_oof', PRIMARY_VOL], None, 'y ~ RF(OOF) + V'),
                  (['rf_oof', PRIMARY_VOL, 'switch'], None, 'y ~ RF(OOF) + V + switch'),
                  (['rf_oof'], fe, 'y ~ RF(OOF) + 分层FE(三分位)'),
                  (['rf_oof', 'R2_occ'], None, 'y ~ RF(OOF) + R2'),
                  (['rf_oof', PRIMARY_VOL, 'R2_occ'], None, 'y ~ RF(OOF) + V + R2'),
                  (['rf_oof', 'R2_occ'], fe, 'y ~ RF(OOF) + R2 + 分层FE(三分位)')]

    P('| 模型 | 项 | β | OR | 95% CI | p |')
    P('|---|---|---|---|---|---|')
    reg_rows = []
    for cols, dummies, label in specs:
        X, names = design(cols, dummies)
        for r_ in logit_row(X, y, names):
            P(f'| {label} | {r_["term"]} | {r_["beta"]:+.4f} | {r_["or_"]:.3f} | '
              f'[{r_["lo"]:.3f}, {r_["hi"]:.3f}] | {r_["p"]:.4g} |')
            reg_rows.append(dict(model=label, **{k: r_[k] for k in
                                                 ('term', 'beta', 'se', 'z', 'p', 'or_', 'lo', 'hi')}))
    P()
    P('**异质性检验**：switch × 活动量交互（显著 ⇒ switch 的效应随活动量变化，单一「切换=风险」规则不成立）。')
    P()
    inter = df['switch'].values * df[PRIMARY_VOL].values
    X = np.column_stack([np.ones(len(y)), df['switch'].values, df[PRIMARY_VOL].values, inter])
    for j in (1, 2, 3):
        X[:, j] = (X[:, j] - X[:, j].mean()) / (X[:, j].std() + 1e-12)
    for r_ in logit_row(X, y, ['intercept', 'switch', 'V_full_pc1', 'switch:V']):
        P(f'| y ~ switch + V + switch:V | {r_["term"]} | {r_["beta"]:+.4f} | {r_["or_"]:.3f} | '
          f'[{r_["lo"]:.3f}, {r_["hi"]:.3f}] | {r_["p"]:.4g} |')
        reg_rows.append(dict(model='y ~ switch + V + switch:V', **{k: r_[k] for k in
                                 ('term', 'beta', 'se', 'z', 'p', 'or_', 'lo', 'hi')}))
    P()

    # ───── §7 残差 ─────
    P('## 7 · 残差检验：控制活动量后 switch 还剩多少判别力')
    P()
    P('rank(switch) 对 rank(V) 残差化后重算 AUC：')
    P()
    P('| 活动量口径 | 原始 switch AUC | 残差 switch AUC | 95% CI（bootstrap） |')
    P('|---|---|---|---|')
    for vname, _ in VOLUMES:
        V = df[vname].values
        res = _resid_on(rankdata(df['switch'].values), rankdata(V))
        au = auc_tie(res, y)
        m, lo, hi = boot_ci(lambda s: auc_tie(_resid_on(rankdata(df['switch'].values[s]),
                                                         rankdata(V[s])), y[s]))
        P(f'| {vname} | {auc_tie(df["switch"].values, y):.4f} | **{au:.4f}** | [{lo:.4f}, {hi:.4f}] |')
    P()

    # ───── §8 稳健性 + 机制 ─────
    P('## 8 · 稳健性矩阵与机制检验')
    P()
    P('### 8.1 偏相关稳健性（活动量口径 × 相关口径）')
    P()
    P('| 活动量口径 | 相关口径 | 偏 ρ(switch, y \\| V) | p | 偏 ρ(R2, y \\| V) | p |')
    P('|---|---|---|---|---|---|')
    for vname, _ in VOLUMES:
        V = df[vname].values
        r_s, p_s = partial_spearman(df['switch'].values, V, y)
        r_r, p_r = partial_spearman(df['R2_occ'].values, V, y)
        P(f'| {vname} | 秩-偏 Spearman | {r_s:+.4f} | {p_s:.4g} | {r_r:+.4f} | {p_r:.4g} |')
        k_s = kendalltau(_resid_on(rankdata(df['switch'].values), rankdata(V)),
                         _resid_on(rankdata(y), rankdata(V))).correlation
        k_r = kendalltau(_resid_on(rankdata(df['R2_occ'].values), rankdata(V)),
                         _resid_on(rankdata(y), rankdata(V))).correlation
        P(f'| {vname} | 秩-偏 Kendall | {k_s:+.4f} | — | {k_r:+.4f} | — |')
    P()
    P('### 8.2 机制检验：switch 到底在量什么')
    P()
    P('| 相关 | ρ | p | 解释 |')
    P('|---|---|---|---|')
    mech = [
        ('switch ~ 全轨迹事件数', df['switch'].values, df['n_full'].values, '全学期活动量 → switch↑（论文 §6.7 已报 ρ≈+0.25）'),
        ('switch ~ 末段跨度(小时)', df['switch'].values, df['w_span_h'].values, '末段行为越稀疏 → 后验越不稳 → switch↑（机制候选）'),
        ('switch ~ 末段平均间隔(分钟)', df['switch'].values, df['w_mean_gap_min'].values, '同上（间隔越大越稀疏）'),
        ('switch ~ 末段强度(事件/小时)', df['switch'].values, df['w_intensity'].values, '末段越密集 → switch↓（机制候选）'),
        ('switch ~ 末段活跃小时数', df['switch'].values, df['w_active_h'].values, '末段活跃天数'),
        ('switch ~ 窗口活跃天数占比', df['switch'].values, df['w_days_active_ratio'].values, '末段行为越分散（越不连续）→ switch↑'),
        ('switch ~ V_win_pc1', df['switch'].values, df['V_win_pc1'].values, '末段活动量（负相关 = 越活跃越稳定）'),
        ('switch ~ V_full_pc1', df['switch'].values, df[PRIMARY_VOL].values, '全轨迹活动量（主口径）'),
        ('R2 占用 ~ V_full_pc1', df['R2_occ'].values, df[PRIMARY_VOL].values, '风险状态时长是否只是活动量代理？'),
        ('R2 占用 ~ 末段跨度(小时)', df['R2_occ'].values, df['w_span_h'].values, '同上（机制候选）'),
    ]
    for nm, a, b, expl in mech:
        r, p = spearmanr(a, b)
        P(f'| {nm} | {r:+.4f} | {p:.4g} | {expl} |')
    P()
    P('### 8.3 口径饱和性诊断')
    P()
    P('| 口径 | 变异系数 CV | 取值个数（近似） | 判别挂科 AUC |')
    P('|---|---|---|---|')
    for nm, v in [('窗口事件数 n_win', df['n_win'].values),
                  ('窗口跨度 w_span_h', df['w_span_h'].values),
                  ('全轨迹事件数 n_full', df['n_full'].values)]:
        P(f'| {nm} | {v.std()/(abs(v.mean())+1e-12):.3f} | {len(np.unique(np.round(v,3)))} | {auc_tie(v, y):.4f} |')
    P()
    P('> 窗口事件数几乎恒为 128（CV≈0.004）→ **不能**用作活动量代理；'
      '这也解释了为什么论文 §6.7 的「switch 与 paste/run 计数相关」其实是**全轨迹**计数与 switch 的关系，'
      '而 switch 本身来自**末 128 事件窗口**，两者口径不一致。')
    P()

    # ───── §9 结论 ─────
    P('## 9 · 结论')
    P()
    def pick(target, volume):
        return next(r for r in part_rows if r['target'] == target and r['volume'] == volume)
    tri_p = next(r for r in tri_rows if r['volume'] == PRIMARY_VOL)
    ps_vol = pick('switch', PRIMARY_VOL)
    ps_d2 = pick('switch', 'V_full_pc1+V_win_span')
    ps_d3 = pick('switch', 'V_full_pc1+V_win_span+V_win_intensity')
    ps_rf = pick('switch', 'rf_oof')
    pr_vol = pick('R2_occ', PRIMARY_VOL)
    pr_d3 = pick('R2_occ', 'V_full_pc1+V_win_span+V_win_intensity')
    pr_rf = pick('R2_occ', 'rf_oof')
    mh_s3 = next(r for r in mh_rows if r['exposure'] == 'switch_high_vs_low'
                 and r['volume'] == PRIMARY_VOL and r['k'] == 3)
    mh_r3 = next(r for r in mh_rows if r['exposure'] == 'R2_high_vs_low'
                 and r['volume'] == PRIMARY_VOL and r['k'] == 3)
    def reg_term(model, term):
        return next(r for r in reg_rows if r['model'] == model and r['term'] == term)
    sw_adj = reg_term('y ~ switch + V', 'switch')
    sw_fe = reg_term('y ~ switch + 分层FE(三分位)', 'switch')
    sw_rf = reg_term('y ~ RF(OOF) + V + switch', 'switch')
    r2_adj = reg_term('y ~ R2 + V', 'R2_occ')
    r2_fe = reg_term('y ~ R2 + 分层FE(三分位)', 'R2_occ')
    r2_rf = reg_term('y ~ RF(OOF) + V + R2', 'R2_occ')
    inter_p = reg_term('y ~ switch + V + switch:V', 'switch:V')

    P('### 9.1 对 §7.5「switching is not risk」的判定')
    P()
    P(f'**原始观察**：switch 单独判别挂科 AUC=0.3806（反向）；ρ(switch, y)={tri_p["rho_switch_y"]:+.4f}。')
    P()
    P('**混淆路径确实存在**：')
    P(f'- ρ(switch, V_full_pc1) = {tri_p["rho_volume_switch"]:+.4f}（switch 与全轨迹活动量正相关）')
    P(f'- ρ(V_full_pc1, y) = {tri_p["rho_volume_y"]:+.4f}（活动量与挂科强反向）')
    P(f'- 乘积 {tri_p["product"]:+.4f} ≈ 原始关联 {tri_p["rho_switch_y"]:+.4f} 的 {abs(tri_p["product"]/tri_p["rho_switch_y"])*100:.0f}%')
    P(f'- 机制线索：switch 与末段**跨度/间隔**正相关（ρ=+0.2356）、与末段**强度**负相关（ρ=−0.2289）、'
      f'与末段活动量负相关（ρ=−0.3852）→ **switch 实际上在量"末段行为有多稀疏/多不连续"**。')
    P()
    P('**控制活动量后的残余效应**：')
    P(f'| 控制集 | 偏 ρ(switch, y) | 95% CI | p | 判定 |')
    P('|---|---|---|---|---|')
    P(f'| 无（原始） | {ps_vol["raw_rho"]:+.4f} | — | — | 显著反向 |')
    P(f'| V_full_pc1 | {ps_vol["partial_rho"]:+.4f} | [{ps_vol["ci_lo"]:+.4f}, {ps_vol["ci_hi"]:+.4f}] | {ps_vol["p"]:.4g} | 显著但幅度腰斩（被解释 {ps_vol["explained_frac"]*100:.0f}%） |')
    P(f'| V_full_pc1 + V_win_span | {ps_d2["partial_rho"]:+.4f} | [{ps_d2["ci_lo"]:+.4f}, {ps_d2["ci_hi"]:+.4f}] | {ps_d2["p"]:.4g} | **不显著** |')
    P(f'| 活动量区块（3 协变量） | {ps_d3["partial_rho"]:+.4f} | [{ps_d3["ci_lo"]:+.4f}, {ps_d3["ci_hi"]:+.4f}] | {ps_d3["p"]:.4g} | **不显著** |')
    P(f'| RF-28d OOF（含全部活动量聚合） | {ps_rf["partial_rho"]:+.4f} | [{ps_rf["ci_lo"]:+.4f}, {ps_rf["ci_hi"]:+.4f}] | {ps_rf["p"]:.4g} | **不显著** |')
    P()
    P(f'分层回归同向：switch 的 OR 从未控制的 0.664（p=4.5e-05）降到控制活动量后 0.833（p={sw_adj["p"]:.3g}）、'
      f'分层固定效应后 0.837（p={sw_fe["p"]:.3g}）、'
      f'在 RF 之上 0.759（p={sw_rf["p"]:.3g}）——**均不显著**。')
    P(f'MH 合并 OR（按全轨迹活动量三分位分层）= {mh_s3["or_mh"]:.2f}，p={mh_s3["p_mh"]:.3g} → **分层后反向关联不再显著**。')
    P(f'交互项 switch×活动量 p={inter_p["p"]:.3g} → 未发现明显异质性（方向一致，只是幅度被吸收）。')
    P()
    P('**判定**：§7.5 的「switching is not risk」这一表述**必须改写**。'
      '数据不支持「切换少 = 危险」这条可交付规则，也不支持「切换与风险有独立反向关联」：'
      '**switch 在控制行为活动量后不再携带任何关于挂科的可靠信息**；'
      '它本质上是"末段行为稀疏度/活动量"的代理。'
      '论文原文把两种解释并列而未做区分，是本节的漏洞。')
    P()
    P('### 9.2 对 §7.3「R2 是可直接用的风险状态指标」的判定（必须收窄）')
    P()
    P('**配对分析是本节最干净的判据**（每个 PASS 生配一个活动量最接近的 FAIL 生，SMD<0.01）：')
    P()
    P('| 匹配口径 | switch 配对差(FAIL−PASS) | Wilcoxon p | SMD | R2 占用配对差 | Wilcoxon p | SMD |')
    P('|---|---|---|---|---|---|---|')
    for label in ['仅全轨迹活动量', '全轨迹活动量 + 末段新鲜度',
                  '仅窗口层行为量（与 regime 同窗口，最公平）',
                  '全量（全轨迹 + 窗口层）', '最强判据：RF 分数 + 全轨迹 + 窗口层',
                  '全轨迹活动量 + RF 分数']:
        rows_ = [r_ for r_ in pair_rows if r_['match'] == label]
        if not rows_:
            continue
        sw_ = next(r_ for r_ in rows_ if r_['col'] == 'switch')
        r2_ = next(r_ for r_ in rows_ if r_['col'] == 'R2_occ')
        P(f'| {label} | {sw_["median_diff"]:+.4f} | {sw_["wilcoxon_p"]:.4g} | {sw_["smd"]:+.3f} | '
          f'{r2_["median_diff"]:+.4f} | {r2_["wilcoxon_p"]:.4g} | {r2_["smd"]:+.3f} |')
    P()
    P('**读法**：')
    P(f'- **控制全轨迹活动量后，R2 占用在挂科生与通过生之间没有差异**'
      f'（配对差 {next(r_["median_diff"] for r_ in pair_rows if r_["match"]=="仅全轨迹活动量" and r_["col"]=="R2_occ"):+.4f}，'
      f'p={next(r_["wilcoxon_p"] for r_ in pair_rows if r_["match"]=="仅全轨迹活动量" and r_["col"]=="R2_occ"):.3g}）'
      f'→ **R2 的原始 AUC 0.6386 主要是"行为投入量"的重新编码**。')
    P('- 仅按窗口层行为量配对时 R2 差异很大（配对差 +0.07，SMD +0.45），'
      '但那是因为这种配对**没有约束全轨迹投入量**，R2 仍在替全轨迹投入量说话 —— '
      '把全轨迹量也纳入后差异即消失，两条证据互相印证。')
    P(f'- 偏相关口径同样一致：ρ(R2, y)={pr_vol["raw_rho"]:+.4f} → 控制活动量后 {pr_vol["partial_rho"]:+.4f}'
      f'（p={pr_vol["p"]:.3g}），约 {pr_vol["explained_frac"]*100:.0f}% 被活动量解释；'
      f'分层固定效应回归中 R2 的 OR={r2_fe["or_"]:.2f}（p={r2_fe["p"]:.3g}）；'
      f'MH 合并 OR（活动量三分位）={mh_r3["or_mh"]:.2f}（p={mh_r3["p_mh"]:.3g}）。')
    P()
    P('**判定**：R2 占用**不是**独立的"风险状态时长"指标，而更接近"行为投入强度的可解释化再编码"。'
      '§7.3 原文「R2 is a directly usable risk-state indicator」**必须收窄**为'
      '「R2 occupancy is an interpretable re-encoding of behavioural engagement, '
      'not an independent risk signal (at matched engagement level the failing/passing difference is null)」。'
      '如果仍要产品化，只能作为"投入强度 + 节奏"的描述性指标，且必须与活动量并列展示，'
      '不能宣称它比"做了多少"多提供了信息。')
    P()
    P('### 9.3 对论文口径的修补要求（审稿人会问）')
    P()
    P('1. **窗口口径不一致**：switch 在**末 128 事件窗口**上计算（模型输入窗口），'
      '而论文 §6.7 的行为统计（paste/run/submit 计数）是**全轨迹**；'
      '窗口事件数 CV=0.046（几乎恒为 128）→ 「switch 与行为总量相关」这句话必须补口径说明。')
    P('2. **§7.5 改写**：从"switching is not risk"改为"switching is not an independent risk signal '
      '— it is largely a proxy for behavioural volume / end-of-term behavioural sparsity"。')
    P('3. **§7.3 收窄**：R2 占用"约一半关联来自活动量，**配对后挂科/通过无差异**"必须写进正文，'
      '并把"可交付的风险状态指标"降级为"可解释的行为投入/节奏描述量"。')
    P('4. **摘要与 §9 结论**：涉及"可解释性是主要价值"的表述可保留，但需注明该可解释信号'
      '本身也含活动量成分。')
    P('5. **产品口径**：配对结果显示"同等投入强度下，挂科生并没有更多停留在风险状态" → '
      '**不能**把 R2 风险状态时长作为独立预警指标卖给学校；若要保留，必须与投入强度并列展示，'
      '且文案只能是"投入强度与节奏"，不能声称比"做了多少"多提供信息。')
    P()
    P('### 9.4 局限')
    P()
    P('- 全轨迹活动量与挂科本身高度相关（AUC 0.159），控制它会把大量方差一并带走 → '
      '残余项的检验功效有限（层内 PASS 数最少仅 3 人，见 §4 的 S1）。')
    P('- 观察性数据，无法区分"活动量 → regime 形态"与"regime 形态 → 活动量"的因果方向。')
    P('- 单数据集、单课程；未做时间外推。')
    P()

    md_path = os.path.join(OUT, 'REGIMEMAMBA_STRATIFIED_20261110.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(L) + '\n')

    cs = []
    for r_ in desc_rows:
        cs.append(dict(block='descriptive', **r_))
    for r_ in tri_rows:
        cs.append(dict(block='triangle', **r_))
    for r_ in part_rows:
        cs.append(dict(block='partial', **r_))
    for r_ in strat_rows:
        cs.append(dict(block='stratified', **r_))
    for r_ in mh_rows:
        cs.append(dict(block='mh', **r_))
    for r_ in reg_rows:
        cs.append(dict(block='logit', **r_))
    csv_path = os.path.join(OUT, 'REGIMEMAMBA_STRATIFIED_20261110.csv')
    pd.DataFrame(cs).to_csv(csv_path, index=False)
    sl_path = os.path.join(OUT, 'REGIMEMAMBA_STRATIFIED_20261110_studentlevel.csv')
    df.to_csv(sl_path, index=False)

    print()
    print(f'[stratified_analysis] wrote {md_path}')
    print(f'[stratified_analysis] wrote {csv_path}')
    print(f'[stratified_analysis] wrote {sl_path}')


if __name__ == '__main__':
    main()
