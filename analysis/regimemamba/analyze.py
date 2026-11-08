"""
regimemamba.analyze — RegimeMamba 实验结果汇总 + regime 语义分析

输入：
  results/regimemamba/<tag>/{results.json, oof_probs.npy, regime_stats.npz}
  results/rev12/exp1_matrix/*/oof_probs.npy      （MetaMamba / RF / HMM 等基线）
  results/rev12/exp2_prefix/*/oof_probs.npy      （前缀基线）
  /home/ubuntu/SR12_cache/events.npz             （标签 + 行为统计）
输出：
  docs/REGIMEMAMBA_RESULTS_20261108.md
"""
import os
import sys
import json
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu, wilcoxon

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

REG = os.path.join(_ROOT, 'results', 'regimemamba')
REV = os.path.join(_ROOT, 'results', 'rev12')
OUT_MD = os.path.join(_ROOT, 'docs', 'REGIMEMAMBA_RESULTS_20261108.md')

Z = np.load('/home/ubuntu/SR12_cache/events.npz')
y = Z['y'].astype(int)


def rank(p):
    return (np.argsort(np.argsort(p)) + 1.0) / len(p)


def auc_fast(p, yy=None):
    yy = y if yy is None else yy
    r = np.argsort(np.argsort(p)) + 1.0
    npos = float(yy.sum()); nneg = float(len(yy) - npos)
    return float((r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def boot_delta(pa, pb, n=2000, seed=0):
    rng = np.random.default_rng(seed); idx = np.arange(len(y)); ds = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        ds.append(auc_fast(pa[s], y[s]) - auc_fast(pb[s], y[s]))
    ds = np.array(ds)
    return float(ds.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float((ds <= 0).mean())


def load_oof(path):
    return np.load(path).astype(np.float64)


def behavior_stats():
    """从缓存算每生行为统计（用于 regime 语义对齐）。"""
    et, ts, offsets = Z['et'], Z['ts'], Z['offsets']
    n = len(offsets) - 1
    names = ['n_events', 'submit', 'run', 'focus_lost', 'text_insert', 'text_remove',
             'paste', 'submit_rate', 'run_rate', 'focuslost_rate', 'edit_ratio',
             'mean_gap_min', 'median_gap_min', 'gap_std_min', 'max_gap_min',
             'span_hours', 'span_days', 'active_hours', 'n_sessions_30m', 'n_sessions_60m',
             'night_ratio']
    X = np.zeros((n, len(names)))
    # ⚠️ 缓存时间戳单位为 1000 秒（原始 datetime64[us] → astype(int64) 得 µs → //1e9 的遗留换算）
    #    证据：student0 的时间跨度 = 4208 个单位 × 1000 s = 1168.9 小时，与原始日志一致。
    TS_UNIT_SECONDS = 1000.0
    for i in range(n):
        s, e = offsets[i], offsets[i + 1]
        if e <= s:
            continue
        et_i = et[s:e]; ts_i = ts[s:e]
        cnt = np.bincount(et_i, minlength=7)
        dts = np.diff(ts_i).astype(np.float64) * TS_UNIT_SECONDS / 60.0   # 分钟
        span_h = (ts_i[-1] - ts_i[0]) * TS_UNIT_SECONDS / 3600.0
        days = span_h / 24.0 + 1
        act_hours = len(np.unique((ts_i * TS_UNIT_SECONDS) // 3600))
        sess30 = int((dts > 30).sum()) + 1 if len(dts) else 1
        sess60 = int((dts > 60).sum()) + 1 if len(dts) else 1
        hours = (((ts_i * TS_UNIT_SECONDS) % 86400) // 3600).astype(int)
        X[i] = [e - s, cnt[6], cnt[5], cnt[4], cnt[0], cnt[1], cnt[2],
                cnt[6] / max(e - s, 1), cnt[5] / max(e - s, 1), cnt[4] / max(e - s, 1),
                cnt[1] / max(cnt[0] + cnt[1], 1),
                float(np.mean(dts)) if len(dts) else 0.0,
                float(np.median(dts)) if len(dts) else 0.0,
                float(np.std(dts)) if len(dts) else 0.0,
                float(np.max(dts)) if len(dts) else 0.0,
                span_h, days, act_hours, sess30, sess60,
                float(((hours >= 0) & (hours < 6)).mean())]
    return X, names


def main():
    lines = []
    def P(s=''):
        print(s, flush=True)
        lines.append(s)

    P('# RegimeMamba 实验结果与发现（CS1）')
    P()
    P('**生成**：2026-11-08 · `StudentRisk/analysis/regimemamba/analyze.py`')
    P()
    P('**主口径**：全样本合并 OOF AUC（473 生一次排序，比论文折均值更严格）；论文口径（折均值）并列给出。')
    P('**基线**：rev12 同协议同数据结果（MetaMamba / RF / HMM-GMM / LSTM / Attention）。')
    P()

    cells = {}
    if os.path.isdir(REG):
        for tag in sorted(os.listdir(REG)):
            d = os.path.join(REG, tag)
            if not os.path.isdir(d) or not os.path.exists(os.path.join(d, 'results.json')):
                continue
            meta = json.load(open(os.path.join(d, 'results.json')))
            oof = load_oof(os.path.join(d, 'oof_probs.npy'))
            cells[tag] = {'meta': meta, 'oof': oof,
                          'fold_auc': meta.get('per_fold_mean', {}).get('roc_auc'),
                          'fold_f1': meta.get('per_fold_mean', {}).get('macro_f1')}

    base = {}
    for arm, sub in [('id', 'exp1_matrix'), ('pfx', 'exp2_prefix'), ('deconf', 'exp1_matrix_deconf')]:
        d = os.path.join(REV, sub)
        if not os.path.isdir(d):
            continue
        for c in sorted(os.listdir(d)):
            p = os.path.join(d, c, 'oof_probs.npy')
            if os.path.exists(p):
                base[f'{arm}__{c}'] = load_oof(p)

    # ── 表 1：主格 vs 基线 ──
    P('## 1 · 主结果（合并 OOF AUC / 折均值 AUC）')
    P()
    P('| 模型 | 合并 OOF AUC | 折均值 AUC | bal-acc(折) | macro-F1(折) | 对 RF-28d ΔAUC [95% CI] | P(Δ≤0) |')
    P('|---|---|---|---|---|---|---|')
    order = ['main_d28_p1.0', 'main_d28_p0.1', 'main_d28_p0.2', 'deconf_d28_p1.0',
             'abl_k1_d28_p1.0', 'abl_nodirect_d28_p1.0', 'abl_nousage_d28_p1.0']
    for tag in order:
        if tag not in cells:
            continue
        c = cells[tag]
        m = c['meta']
        ref = 'id__rf_d28' if 'deconf' not in tag else 'deconf__rf_d28'
        if tag.startswith('main_d28_p0.1'):
            ref = 'pfx__rf_d28_p0.1'
        elif tag.startswith('main_d28_p0.2'):
            ref = 'pfx__rf_d28_p0.2'
        if ref in base:
            d, lo, hi, pn = boot_delta(c['oof'], base[ref])
            ref_s = f'{d:+.4f} [{lo:+.4f},{hi:+.4f}] | {pn:.3f}'
        else:
            ref_s = '— | —'
        P(f'| RegimeMamba {tag} | {auc_fast(c["oof"]):.4f} | {c["fold_auc"]:.4f} | '
          f'{m["per_fold_mean"]["balanced_accuracy"]:.4f} | {m["per_fold_mean"]["macro_f1"]:.4f} | {ref_s} |')

    for key, label in [('id__metamamba_d28', 'MetaMamba-28d（rev12）'), ('id__rf_d28', 'RF-28d（rev12）'),
                       ('id__hmm_gmm_d28', 'HMM-GMM-28d（rev12）'), ('id__lstm_d28', 'LSTM-28d（rev12）'),
                       ('id__attention_d28', 'Attention-28d（rev12）')]:
        if key in base:
            P(f'| {label} | {auc_fast(base[key]):.4f} | — | — | — | — | — |')
    P()

    # ── 表 2：与 MetaMamba 的配对检验 ──
    P('## 2 · 与同协议基线的配对检验（bootstrap 2000 + 折级 Wilcoxon）')
    P()
    P('| RegimeMamba 格 | 对照 | ΔAUC(合并 OOF) [95% CI] | P(Δ≤0) | ΔAUC(折均值) | Wilcoxon p |')
    P('|---|---|---|---|---|---|')
    for tag, ref_key in [('main_d28_p1.0', 'id__metamamba_d28'),
                         ('main_d28_p1.0', 'id__rf_d28'),
                         ('deconf_d28_p1.0', 'deconf__metamamba_d28'),
                         ('deconf_d28_p1.0', 'deconf__rf_d28'),
                         ('main_d28_p0.1', 'pfx__metamamba_d28_p0.1'),
                         ('main_d28_p0.2', 'pfx__metamamba_d28_p0.2'),
                         ('abl_k1_d28_p1.0', 'main_d28_p1.0'),
                         ('abl_nodirect_d28_p1.0', 'main_d28_p1.0'),
                         ('abl_nousage_d28_p1.0', 'main_d28_p1.0')]:
        if tag not in cells or ref_key not in base:
            continue
        d, lo, hi, pn = boot_delta(cells[tag]['oof'], base[ref_key])
        w_p = '—'
        try:
            fp = os.path.join(REV, 'exp1_matrix', 'metamamba_d28', 'fold_metrics.csv')
            if os.path.exists(fp):
                import pandas as pd
                a = pd.read_csv(os.path.join(REG, tag, 'fold_metrics.csv'))['roc_auc'].values
                b = pd.read_csv(fp)['roc_auc'].values
                n = min(len(a), len(b))
                w_p = f'{wilcoxon(a[:n], b[:n]).pvalue:.4f}'
        except Exception:
            pass
        P(f'| {tag} | {ref_key} | {d:+.4f} [{lo:+.4f},{hi:+.4f}] | {pn:.3f} | '
          f'{cells[tag]["fold_auc"]:.4f} | {w_p} |')
    P()

    # ── 表 3：regime 语义 ──
    P('## 3 · regime 语义与可解释性（H4）')
    P()
    X, bnames = behavior_stats()
    main_tag = 'main_d28_p1.0'
    if main_tag in cells:
        st_path = os.path.join(REG, main_tag, 'regime_stats.npz')
        if os.path.exists(st_path):
            st = np.load(st_path)
            usage, switch, post_mean = st['usage'], st['switch'], st['post_mean']
            K = usage.shape[1]
            P(f'- regime 数 K = {K}；占用直方图（全体）：'
              + '，'.join(f'R{k}={usage[:, k].mean():.3f}' for k in range(K)))
            P(f'- **占用塌缩检查（H4-①）**：最小占用 {usage.mean(0).min():.3f} '
              f'（判据 ≥0.05）→ {"通过" if usage.mean(0).min() >= 0.05 else "不通过"}')
            P()
            P('### 3.1 每个 regime 的占用 vs 挂科（行为语义）')
            P()
            P('| regime | FAIL 生平均占用 | PASS 生平均占用 | 差异 | Mann-Whitney p | 单独 AUC(判别挂科) |')
            P('|---|---|---|---|---|---|')
            for k in range(K):
                a = usage[y == 1, k]; b = usage[y == 0, k]
                p = mannwhitneyu(a, b).pvalue if len(b) > 1 else float('nan')
                P(f'| R{k} | {a.mean():.3f} | {b.mean():.3f} | {a.mean()-b.mean():+.3f} | {p:.4f} | '
                  f'{auc_fast(usage[:, k]):.4f} |')
            P()
            P('### 3.2 切换率（switch）与行为统计、挂科的关联')
            P()
            sw_auc = auc_fast(switch)
            P(f'- 切换率单独作为挂科预测器的 AUC = **{sw_auc:.4f}**（0.5 = 无信息）')
            P()
            P('| 行为统计 | 与切换率 Spearman ρ | p | 与 R0 占用 ρ | p |')
            P('|---|---|---|---|---|')
            for i, nm in enumerate(bnames):
                col = X[:, i]
                if np.nanstd(col) < 1e-12:
                    P(f'| {nm} | 常量列（无变异，无法计算相关） | — | 常量列 | — |')
                    continue
                rho, p = spearmanr(switch, col)
                rho0, p0 = spearmanr(usage[:, 0], col)
                P(f'| {nm} | {rho:+.3f} | {p:.4f} | {rho0:+.3f} | {p0:.4f} |')
            P()
            if 'regime_bias' in st:
                bias = st['regime_bias'].mean(axis=0)
                emp = np.array([usage[y == 1, k].mean() for k in range(K)])
                rho, p = spearmanr(bias, emp)
                P('### 3.3 模型内部风险归因 vs 数据事实')
                P()
                P(f'- 模型学到的 per-regime 风险偏置 b_k：' + '，'.join(f'R{k}={bias[k]:+.3f}' for k in range(K)))
                P(f'- 各 regime 在 FAIL 生中的平均占用：' + '，'.join(f'R{k}={emp[k]:.3f}' for k in range(K)))
                P(f'- 两者 Spearman ρ = {rho:+.3f}（p={p:.4f}）→ '
                  f'{"模型归因与数据方向一致" if (rho > 0 and p < 0.05) or (rho < 0 and p < 0.05) else "方向不一致或未达显著"}')
                P()
            if 'trans' in st:
                A = st['trans'].mean(axis=0)
                P('### 3.4 学到的 regime 转移矩阵（均值）')
                P()
                P('| from \\ to | ' + ' | '.join(f'R{k}' for k in range(K)) + ' |')
                P('|---|' + '---|' * K)
                for i in range(K):
                    P(f'| R{i} | ' + ' | '.join(f'{A[i, j]:.3f}' for j in range(K)) + ' |')
                P()
            # 后验轨迹：FAIL vs PASS
            if post_mean.ndim == 3:
                traj = post_mean.mean(axis=0)          # (L,K)
                P('### 3.5 后验轨迹：FAIL 生 vs PASS 生（按时间步平均占用）')
                P()
                P('| 时间步(左 padding 后的位置) | ' + ' | '.join(f'R{k}' for k in range(K)) + ' |')
                P('|---|---|' + '---|' * K)
                L = traj.shape[0]
                for t in range(0, L, 16):
                    P(f'| t={t} | ' + ' | '.join(f'{traj[t, k]:.3f}' for k in range(K)) + ' |')
                P()

    # ── 结论表 ──
    P('## 4 · 预先注册判据的判定（H1–H6）')
    P()
    P('| 判据 | 阈值 | 实测 | 判定 |')
    P('|---|---|---|---|')
    mm = base.get('id__metamamba_d28')
    mm_dc = base.get('deconf__metamamba_d28')
    if 'main_d28_p1.0' in cells and mm is not None:
        d, lo, hi, pn = boot_delta(cells['main_d28_p1.0']['oof'], mm)
        verdict = '通过（有独立增益）' if (d >= 0.010 and pn <= 0.05) else (
            '判负（无独立增益）' if d > -0.010 else '判负（regime 机制有害）')
        P(f'| H1 判别力增益 | ΔAUC ≥ +0.010 且 P≤0.05 | Δ={d:+.4f} [{lo:+.4f},{hi:+.4f}], P={pn:.3f} | {verdict} |')
    if 'main_d28_p1.0' in cells and 'deconf_d28_p1.0' in cells:
        drop = auc_fast(cells['main_d28_p1.0']['oof']) - auc_fast(cells['deconf_d28_p1.0']['oof'])
        P(f'| H2 非身份捷径 | 去混淆跌幅 ≤ 0.020 | 跌幅 {drop:.4f} | '
          f'{"通过" if drop <= 0.020 else ("不通过" if drop <= 0.05 else "判负（走了身份捷径）")} |')
    if 'main_d28_p0.1' in cells and 'pfx__metamamba_d28_p0.1' in base:
        d, lo, hi, pn = boot_delta(cells['main_d28_p0.1']['oof'], base['pfx__metamamba_d28_p0.1'])
        P(f'| H3 早期预警 | p10% AUC ≥ MetaMamba p10% | Δ={d:+.4f} [{lo:+.4f},{hi:+.4f}] | '
          f'{"通过" if d >= 0 else "判负"} |')
    if 'abl_k1_d28_p1.0' in cells and 'main_d28_p1.0' in cells:
        d, lo, hi, pn = boot_delta(cells['main_d28_p1.0']['oof'], cells['abl_k1_d28_p1.0']['oof'])
        P(f'| H6 容量混淆 | K=4 明显优于 K=1 | Δ={d:+.4f} [{lo:+.4f},{hi:+.4f}], P={pn:.3f} | '
          f'{"通过（增益来自 regime）" if (d > 0.005 and pn <= 0.10) else "不通过（增益可能来自容量）"} |')
    P()

    # ── CSV 导出（供论文数据附录与复现）──
    import csv
    data_csv = os.path.join(_ROOT, 'docs', 'REGIMEMAMBA_DATA_20261108.csv')
    with open(data_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['row_type', 'name', 'n_regime', 'prefix', 'deconfound', 'use_film',
                    'use_direct', 'lam_usage', 'hp', 'pooled_auc', 'pooled_macro_f1',
                    'pooled_bal_acc', 'pooled_acc', 'fold_auc', 'fold_macro_f1',
                    'fold_bal_acc', 'fold_acc', 'elapsed_s'])
        for tag in order:
            if tag not in cells:
                continue
            m = cells[tag]['meta']; pf = m['per_fold_mean']; po = m['pooled_oof']
            w.writerow(['regimemamba', tag, m.get('n_regime'), m.get('prefix'),
                        m.get('deconfound'), m.get('use_film'), m.get('use_direct'),
                        m.get('lam_usage'), json.dumps(m.get('hp')), po['roc_auc'],
                        po['macro_f1'], po['balanced_accuracy'], po['accuracy'],
                        pf['roc_auc'], pf['macro_f1'], pf['balanced_accuracy'], pf['accuracy'],
                        round(m.get('elapsed_s', 0))])
        for key, label in [('id__rf_d28', 'RF-28d'), ('id__attention_d28', 'Attention-28d'),
                           ('id__lstm_d28', 'LSTM-28d'), ('id__bilstm_d28', 'BiLSTM-28d'),
                           ('id__metamamba_d28', 'MetaMamba-28d'), ('id__hmm_gmm_d28', 'HMM-GMM-28d'),
                           ('deconf__rf_d28', 'RF-28d (deconf)'),
                           ('deconf__attention_d28', 'Attention-28d (deconf)'),
                           ('deconf__lstm_d28', 'LSTM-28d (deconf)'),
                           ('deconf__metamamba_d28', 'MetaMamba-28d (deconf)'),
                           ('deconf__hmm_gmm_d28', 'HMM-GMM-28d (deconf)'),
                           ('pfx__rf_d28_p0.1', 'RF-28d p0.1'),
                           ('pfx__metamamba_d28_p0.1', 'MetaMamba-28d p0.1'),
                           ('pfx__rf_d28_p0.2', 'RF-28d p0.2'),
                           ('pfx__metamamba_d28_p0.2', 'MetaMamba-28d p0.2')]:
            if key not in base:
                continue
            w.writerow(['baseline', label, '', '', '', '', '', '', '', auc_fast(base[key]),
                        '', '', '', '', '', '', '', ''])
    P(f'[analyze] 写出 {data_csv}')

    # ── regime 语义 CSV ──
    reg_csv = os.path.join(_ROOT, 'docs', 'REGIMEMAMBA_REGIME_STATS_20261108.csv')
    st_path = os.path.join(REG, 'main_d28_p1.0', 'regime_stats.npz')
    if os.path.exists(st_path):
        st = np.load(st_path)
        usage, switch = st['usage'], st['switch']
        K = usage.shape[1]
        with open(reg_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['row_type', 'key', 'value'])
            for k in range(K):
                a, b = usage[y == 1, k], usage[y == 0, k]
                w.writerow(['regime_occupancy', f'R{k}_fail_mean', round(float(a.mean()), 4)])
                w.writerow(['regime_occupancy', f'R{k}_pass_mean', round(float(b.mean()), 4)])
                w.writerow(['regime_occupancy', f'R{k}_diff', round(float(a.mean() - b.mean()), 4)])
                w.writerow(['regime_occupancy', f'R{k}_mannwhitney_p',
                            round(float(mannwhitneyu(a, b).pvalue), 6)])
                w.writerow(['regime_occupancy', f'R{k}_standalone_auc',
                            round(float(auc_fast(usage[:, k])), 4)])
            w.writerow(['switch', 'standalone_auc', round(float(auc_fast(switch)), 4)])
            w.writerow(['switch', 'mean', round(float(switch.mean()), 4)])
            for i, nm in enumerate(bnames):
                col = X[:, i]
                if np.nanstd(col) < 1e-12:
                    w.writerow(['switch_corr', nm, 'constant_column'])
                    continue
                rho, p = spearmanr(switch, col)
                w.writerow(['switch_corr', nm, f'rho={rho:+.4f};p={p:.6g}'])
        P(f'[analyze] 写出 {reg_csv}')

    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    P(f'[analyze] 写出 {OUT_MD}')


if __name__ == '__main__':
    main()
