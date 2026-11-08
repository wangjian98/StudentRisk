"""
regimemamba.fill_paper — 用真实结果回填论文占位符（避免人工转录错误）

用法：python3 analysis/regimemamba/fill_paper.py

设计要点：
1) 中英文两篇论文共用同一批占位符 token，因此**叙述类**占位符必须按语言分别生成，
   否则英文论文会被灌入中文。数值类占位符语言无关，两篇共用。
2) H6 判定必须与 analyze.py 用同一套预注册判据（Δ>0.005 且 P≤0.10），
   不能因为 K=1 对照“健康”就默认通过。
"""
import os
import sys
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, sys.path)

REG = os.path.join(_ROOT, 'results', 'regimemamba')
REV = os.path.join(_ROOT, 'results', 'rev12')
DOCS = os.path.join(_ROOT, 'docs')
CACHE = os.environ.get('SR12_CACHE', '/home/ubuntu/SR12_cache')
CACHE_FILE = os.path.join(CACHE, 'events.npz')

PAPER_FILES = {
    'zh': 'REGIMEMAMBA_PAPER_20261108_zh.md',
    'en': 'REGIMEMAMBA_PAPER_20261108_en.md',
}

Z = np.load(CACHE_FILE)
y = Z['y'].astype(int)


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
    return (float(ds.mean()), float(np.percentile(ds, 2.5)),
            float(np.percentile(ds, 97.5)), float((ds <= 0).mean()))


def main():
    sub = {}       # 通用数值 + 中文叙述
    sub_en = {}    # 英文叙述（覆盖同名键）

    # ── K=1 对照 ──
    k1_dir = os.path.join(REG, 'abl_k1_d28_p1.0')
    main_dir = os.path.join(REG, 'main_d28_p1.0')
    mm_oof = np.load(os.path.join(REV, 'exp1_matrix', 'metamamba_d28', 'oof_probs.npy')).astype(float)
    if os.path.exists(os.path.join(k1_dir, 'results.json')) and os.path.exists(os.path.join(main_dir, 'oof_probs.npy')):
        m = json.load(open(os.path.join(k1_dir, 'results.json')))
        oof = np.load(os.path.join(k1_dir, 'oof_probs.npy')).astype(float)
        main_oof = np.load(os.path.join(main_dir, 'oof_probs.npy')).astype(float)
        pf = m['per_fold_mean']
        a_k1 = auc_fast(oof)
        d, lo, hi, pn = boot_delta(main_oof, oof)              # K=4 - K=1
        d_mm, lo_mm, hi_mm, pn_mm = boot_delta(mm_oof, oof)   # MetaMamba - K=1 (参考)
        degenerate = bool(a_k1 < 0.6 or not np.isfinite(pf['roc_auc']))
        # 预注册判据，与 analyze.py 完全一致
        h6_pass = bool(d > 0.005 and pn <= 0.10)

        sub['K1_POOLED'] = f'{a_k1:.4f}'
        sub['K1_FOLD'] = (f"{pf['roc_auc']:.4f}" if np.isfinite(pf['roc_auc']) else 'NaN')
        sub['K1_BALACC'] = f"{pf['balanced_accuracy']:.4f}"
        sub['K1_F1'] = f"{pf['macro_f1']:.4f}"
        sub['K1_DELTA'] = f'{d:+.4f}'
        sub['K1_CI'] = f'{lo:+.4f},{hi:+.4f}'
        sub['K1_P'] = f'{pn:.3f}'
        sub['K1_REF'] = (f'(参考：MetaMamba - K=1 = {d_mm:+.4f} [{lo_mm:+.4f},{hi_mm:+.4f}], P={pn_mm:.3f})')
        sub_en['K1_REF'] = (f'(reference: MetaMamba - K=1 = {d_mm:+.4f} [{lo_mm:+.4f},{hi_mm:+.4f}], P={pn_mm:.3f})')

        if degenerate:
            # 对照格退化（首轮除零缺陷）：H6 无法判定
            sub['K1_ABSTRACT'] = ('**regime 结构的容量对照未能完成判定**：K=1 单状态对照在本轮发生实现级退化'
                                  f'（合并 OOF AUC={a_k1:.4f}），因此“增益来自 regime 结构还是来自容量”这一问'
                                  '题在本文中仍未解决（H6 判为未完成）。')
            sub_en['K1_ABSTRACT'] = (
                '**the capacity control for the regime structure could not be adjudicated**: the K=1 '
                f'single-state control degenerated in this run (pooled OOF AUC={a_k1:.4f}), so whether the '
                'gain comes from regime structure or from capacity remains unresolved (H6 incomplete).')
            sub['K1_VERDICT'] = ('K=1 对照**未能学出可用信号**（合并 OOF AUC 低于 0.60 或折内 AUC 为 NaN），'
                                 '因此该差距不能作为"regime 结构的收益"来解释；'
                                 'K=1 格在本轮运行中被发现因除零缺陷而退化（见 §4.5 修复说明），修复后重跑结果见上表。')
            sub_en['K1_VERDICT'] = (
                'The K=1 control **failed to learn a usable signal** (pooled OOF AUC below 0.60 or within-fold '
                'AUC is NaN), so this gap cannot be interpreted as a benefit of regime structure; the K=1 cell '
                'was found to have degenerated due to a divide-by-zero defect (see the §4.5 fix note), and the '
                'table above reports the re-run after the fix.')
            sub['K1_VERDICT_SHORT'] = '⚠️ 对照格退化，结论待重跑后判定'
            sub_en['K1_VERDICT_SHORT'] = '⚠️ control cell degenerated; verdict pending re-run'
            sub['K1_H6'] = f'K=1 退化（合并 OOF AUC={a_k1:.4f}）'
            sub_en['K1_H6'] = f'K=1 degenerated (pooled OOF AUC={a_k1:.4f})'
            sub['K1_VERDICT_DETAIL'] = ('K=1 对照在本轮发生实现级退化，因此 **H6 暂不能判定**；'
                                        '这本身是本文方法学上的一条教训（消融链必须前置健全性检查）。')
            sub_en['K1_VERDICT_DETAIL'] = (
                'The K=1 control degenerated in this run, so **H6 cannot be adjudicated**; this is itself a '
                'methodological lesson of the paper (ablation chains need up-front sanity checks).')
            # 主张类文本：退化时一律不得宣称“regime 结构有效”
            sub['K1_CLAIM_MAIN'] = ('显式的行为 regime 切换**尚无法在本文中被判定为有效的结构性归纳偏置**'
                                    '（K=1 对照退化，H6 未完成）')
            sub_en['K1_CLAIM_MAIN'] = ('explicit behavioural regime switching **cannot yet be claimed as an '
                                       'effective structural inductive bias in this paper** (the K=1 control '
                                       'degenerated; H6 is unresolved)')
            sub['K1_CLAIM_SHORT'] = 'regime 结构的容量对照退化，结论待重跑后判定'
            sub_en['K1_CLAIM_SHORT'] = 'the capacity control for the regime structure degenerated; verdict pending re-run'
            sub['K1_FINDING1_HEAD'] = '发现 1：regime 结构的容量对照退化，本轮无法判定其归纳偏置是否有效'
            sub_en['K1_FINDING1_HEAD'] = ('Finding 1: the capacity control for the regime structure degenerated, '
                                          'so its inductive bias cannot be adjudicated in this run')
            sub['K1_FINDING1_BODY'] = ('K=1 对照退化使得本文无法对“regime 结构是否是有效归纳偏置”给出结论；'
                                        '这本身是本文方法学上的一条教训。')
            sub_en['K1_FINDING1_BODY'] = (
                'Because the K=1 control degenerated, this paper cannot conclude whether regime structure is an '
                'effective inductive bias; this is itself a methodological lesson.')
            sub['K1_MAINCLAIM_CELL'] = 'regime 结构的容量对照退化，H6 未完成；且不超越表格统计，部分依赖身份'
            sub_en['K1_MAINCLAIM_CELL'] = ('the capacity control for the regime structure degenerated (H6 '
                                            'unresolved); does not surpass tabular statistics and partly relies on identity')
            sub['K1_BOUNDARY_ITEM'] = ('1. **结构上尚未证实**：允许选择性状态空间模型拥有若干行为状态并可切换，'
                                       '是否带来实质收益，本轮因 K=1 对照退化而无法判定（见 §6.3）。')
            sub_en['K1_BOUNDARY_ITEM'] = (
                '1. **Not yet established structurally.** Whether letting a selective SSM own several behavioural '
                'states and switch among them yields a substantial gain could not be adjudicated in this run, '
                'because the K=1 control degenerated (§6.3).')
        elif h6_pass:
            sub['K1_ABSTRACT'] = (
                f'**regime 结构本身有效**：K=4 相对 K=1 单状态对照的合并 OOF 增益为 {d:+.4f} '
                f'[{lo:+.4f},{hi:+.4f}]（P(Δ≤0)={pn:.3f}），说明增益来自 regime 结构而非容量。')
            sub_en['K1_ABSTRACT'] = (
                f'**the regime structure is effective in itself**: the pooled OOF gain of K=4 over the K=1 '
                f'single-state control is {d:+.4f} [{lo:+.4f},{hi:+.4f}] (P(Δ≤0)={pn:.3f}), indicating that '
                'the gain comes from regime structure rather than capacity.')
            sub['K1_VERDICT'] = (f'K=4 相对 K=1 单状态对照的合并 OOF 增益为 {d:+.4f} [{lo:+.4f},{hi:+.4f}]，'
                                 f'P(Δ<=0)={pn:.3f} -> 增益确实来自 regime 结构而非容量（H6 通过）。{sub["K1_REF"]}')
            sub_en['K1_VERDICT'] = (
                f'The pooled OOF gain of K=4 over the K=1 single-state control is {d:+.4f} '
                f'[{lo:+.4f},{hi:+.4f}], P(Δ<=0)={pn:.3f} -> the gain does come from regime structure rather '
                f'than capacity (H6 passes). {sub_en["K1_REF"]}')
            sub['K1_VERDICT_SHORT'] = '✅ 通过（增益来自 regime 结构）'
            sub_en['K1_VERDICT_SHORT'] = '✅ passes (gain comes from regime structure)'
            sub['K1_H6'] = f'Δ={d:+.4f} [{lo:+.4f},{hi:+.4f}], P={pn:.3f}'
            sub_en['K1_H6'] = f'Δ={d:+.4f} [{lo:+.4f},{hi:+.4f}], P={pn:.3f}'
            sub['K1_VERDICT_DETAIL'] = (f'K=4 相对 K=1 单状态对照的合并 OOF 增益为 {d:+.4f} [{lo:+.4f},{hi:+.4f}]'
                                        f'（P(Δ<=0)={pn:.3f}）。{sub["K1_REF"]}')
            sub_en['K1_VERDICT_DETAIL'] = (
                f'The pooled OOF gain of K=4 over the K=1 single-state control is {d:+.4f} '
                f'[{lo:+.4f},{hi:+.4f}] (P(Δ<=0)={pn:.3f}). {sub_en["K1_REF"]}')
            sub['K1_CLAIM_MAIN'] = ('显式的行为 regime 切换**确实是一种有效的结构性归纳偏置**'
                                    '（它本身能带来显著增益）')
            sub_en['K1_CLAIM_MAIN'] = ('explicit behavioural regime switching **is** an effective structural '
                                       'inductive bias (it yields a significant gain by itself)')
            sub['K1_CLAIM_SHORT'] = 'regime 结构有效（相对 K=1 的显著增益）'
            sub_en['K1_CLAIM_SHORT'] = 'regime structure is effective (significant gain over the K=1 control)'
            sub['K1_FINDING1_HEAD'] = '发现 1：regime 结构是有效的归纳偏置，但"有效"的程度依赖口径'
            sub_en['K1_FINDING1_HEAD'] = ('Finding 1: regime structure is an effective inductive bias, but '
                                          '"effective" depends on the caliber')
            sub['K1_FINDING1_BODY'] = (
                '这说明在同样的参数预算、同样的数据、同样的训练流程下，**让状态空间模型显式地拥有若干"行为状态"'
                '并允许状态间切换，是有实质收益的结构性归纳偏置**。但必须同时指出：这一收益在**折内均值口径下'
                '大幅缩水**（RegimeMamba 0.9368 vs MetaMamba 0.9334，Wilcoxon P=0.4887）。')
            sub_en['K1_FINDING1_BODY'] = (
                'Under the same parameter budget, data and training loop, **letting a state-space model own '
                'several explicit behavioural states and switch among them is a structural inductive bias with '
                'real payoff**. At the same time this payoff shrinks markedly under the **within-fold mean** '
                'caliber (RegimeMamba 0.9368 vs MetaMamba 0.9334, Wilcoxon P=0.4887).')
            sub['K1_MAINCLAIM_CELL'] = 'regime 结构有效；但不超越表格统计，且部分依赖身份'
            sub_en['K1_MAINCLAIM_CELL'] = ('regime structure helps, but does not surpass tabular statistics and '
                                           'partly relies on identity')
            sub['K1_BOUNDARY_ITEM'] = ('1. **结构上有效**：允许选择性状态空间模型拥有若干行为状态并可切换，'
                                       '是一个有实质收益的归纳偏置（相对单状态对照的差距见 §6.3）。')
            sub_en['K1_BOUNDARY_ITEM'] = (
                '1. **Structurally effective.** Allowing a selective SSM to own several behavioural states and '
                'switch among them yields a substantial gain over a single-state control (§6.3).')
        else:
            # 修复后 K=1 恢复正常，但增益不显著 → H6 判负
            sub['K1_ABSTRACT'] = (
                '**regime 结构的容量对照未能支持“增益来自 regime”**：K=1 单状态对照在本轮因除零缺陷退化后已修复'
                f'并重跑（合并 OOF AUC={a_k1:.4f}，对照恢复正常），但 K=4 相对 K=1 的合并 OOF 增益仅为 '
                f'{d:+.4f} [{lo:+.4f},{hi:+.4f}]（P(Δ≤0)={pn:.3f}），未达预注册判据（Δ>0.005 且 P≤0.10）；'
                '因此 H6 判负——**本文不主张增益来自 regime 结构而非模型容量**。')
            sub_en['K1_ABSTRACT'] = (
                '**the capacity control does not support "the gain comes from the regime"**: after the '
                'divide-by-zero defect was fixed and the K=1 single-state control re-run, the control behaves '
                f'normally again (pooled OOF AUC={a_k1:.4f}), yet the pooled OOF gain of K=4 over K=1 is only '
                f'{d:+.4f} [{lo:+.4f},{hi:+.4f}] (P(Δ≤0)={pn:.3f}), which misses the preregistered criterion '
                '(Δ>0.005 and P≤0.10); H6 therefore fails and **this paper does not claim that the gain comes '
                'from regime structure rather than model capacity**.')
            sub['K1_VERDICT'] = (f'K=4 相对 K=1 单状态对照的合并 OOF 增益为 {d:+.4f} [{lo:+.4f},{hi:+.4f}]，'
                                 f'P(Δ<=0)={pn:.3f}，未达预注册判据（Δ>0.005 且 P≤0.10）-> **H6 判负**：'
                                 f'无法排除“增益来自模型容量而非 regime 结构”。{sub["K1_REF"]}')
            sub_en['K1_VERDICT'] = (
                f'The pooled OOF gain of K=4 over the K=1 single-state control is {d:+.4f} '
                f'[{lo:+.4f},{hi:+.4f}], P(Δ<=0)={pn:.3f}, which misses the preregistered criterion '
                '(Δ>0.005 and P≤0.10) -> **H6 fails**: a gain driven by model capacity rather than regime '
                f'structure cannot be ruled out. {sub_en["K1_REF"]}')
            sub['K1_VERDICT_SHORT'] = '❌ 判负（增益未显著超越 K=1）'
            sub_en['K1_VERDICT_SHORT'] = '❌ fails (gain does not significantly exceed K=1)'
            sub['K1_H6'] = f'Δ={d:+.4f} [{lo:+.4f},{hi:+.4f}], P={pn:.3f}'
            sub_en['K1_H6'] = f'Δ={d:+.4f} [{lo:+.4f},{hi:+.4f}], P={pn:.3f}'
            sub['K1_VERDICT_DETAIL'] = (f'修复除零缺陷后，K=1 单状态对照恢复正常（合并 OOF AUC={a_k1:.4f}），'
                                        f'而 K=4 相对 K=1 的合并 OOF 增益仅 {d:+.4f} [{lo:+.4f},{hi:+.4f}]'
                                        f'（P(Δ<=0)={pn:.3f}），未达预注册判据（Δ>0.005 且 P≤0.10）→ H6 判负。'
                                        f'{sub["K1_REF"]}')
            sub_en['K1_VERDICT_DETAIL'] = (
                f'After the divide-by-zero fix, the K=1 single-state control behaves normally again (pooled OOF '
                f'AUC={a_k1:.4f}), while the pooled OOF gain of K=4 over K=1 is only {d:+.4f} '
                f'[{lo:+.4f},{hi:+.4f}] (P(Δ<=0)={pn:.3f}), missing the preregistered criterion (Δ>0.005 and '
                f'P≤0.10) -> H6 fails. {sub_en["K1_REF"]}')
            sub['K1_CLAIM_MAIN'] = ('显式的行为 regime 切换**未能**在本文中被证明是有效的结构性归纳偏置'
                                    '（相对 K=1 单状态对照的增益不显著，H6 判负）')
            sub_en['K1_CLAIM_MAIN'] = ('explicit behavioural regime switching is **not** shown to be an '
                                       'effective structural inductive bias in this paper (the gain over the '
                                       'K=1 single-state control is not significant; H6 fails)')
            sub['K1_CLAIM_SHORT'] = 'regime 结构相对 K=1 的增益不显著（H6 判负），且部分依赖身份'
            sub_en['K1_CLAIM_SHORT'] = ('the gain of regime structure over K=1 is not significant (H6 fails), and '
                                         'it partly relies on identity')
            sub['K1_FINDING1_HEAD'] = '发现 1：regime 结构相对单状态对照的增益不显著，未能证明其为有效归纳偏置'
            sub_en['K1_FINDING1_HEAD'] = ('Finding 1: the gain of regime structure over the single-state control '
                                          'is not significant, so it is not shown to be an effective inductive bias')
            sub['K1_FINDING1_BODY'] = (
                '因此本文**不主张**“让状态空间模型显式地拥有若干行为状态并允许状态间切换”带来实质收益：'
                f'在同样的参数预算、同样的数据、同样的训练流程下，K=4 相对 K=1 的合并 OOF 增益仅 '
                f'{d:+.4f}（P(Δ≤0)={pn:.3f}），无法排除增益来自模型容量。'
                '同时必须指出，K=4 与同协议 MetaMamba 在**折内均值口径下**亦无显著差异'
                '（0.9368 vs 0.9334，Wilcoxon P=0.4887）——**regime 结构的价值主要体现在可解释性而非精度**。')
            sub_en['K1_FINDING1_BODY'] = (
                'This paper therefore **does not claim** that "letting a state-space model own several explicit '
                'behavioural states and switch among them" brings a substantial gain: under the same parameter '
                'budget, data and training loop, the pooled OOF gain of K=4 over K=1 is only '
                f'{d:+.4f} (P(Δ≤0)={pn:.3f}), so a gain driven by model capacity cannot be ruled out. At the same '
                'time, K=4 and the same-protocol MetaMamba do not differ significantly under the **within-fold '
                'mean** caliber (0.9368 vs 0.9334, Wilcoxon P=0.4887) — **the value of the regime structure lies '
                'in interpretability rather than in accuracy**.')
            sub['K1_MAINCLAIM_CELL'] = 'regime 结构相对 K=1 的增益不显著（H6 判负）；且不超越表格统计，部分依赖身份'
            sub_en['K1_MAINCLAIM_CELL'] = ('the gain of regime structure over K=1 is not significant (H6 fails); '
                                           'it does not surpass tabular statistics and partly relies on identity')
            sub['K1_BOUNDARY_ITEM'] = ('1. **不主张结构上有效**：允许选择性状态空间模型拥有若干行为状态并可切换，'
                                       '是否带来实质收益，本轮未获支持（K=4 vs K=1 增益不显著，H6 判负，见 §6.3）。')
            sub_en['K1_BOUNDARY_ITEM'] = (
                '1. **No claim of structural effectiveness.** Whether letting a selective SSM own several '
                'behavioural states and switch among them yields a substantial gain is not supported in this '
                'run (the K=4 vs K=1 gain is not significant; H6 fails, §6.3).')
    else:
        for k in ['K1_POOLED', 'K1_FOLD', 'K1_BALACC', 'K1_F1']:
            sub[k] = '待补'
        for k in ['K1_DELTA', 'K1_CI', 'K1_P']:
            sub[k] = '待补'
        sub['K1_VERDICT'] = '待补（K=1 对照重跑中）'
        sub['K1_VERDICT_SHORT'] = '待补'
        sub['K1_H6'] = '待补'
        sub['K1_VERDICT_DETAIL'] = '待补'
        sub['K1_CLAIM_MAIN'] = '待补（K=1 对照重跑中）'
        sub['K1_CLAIM_SHORT'] = '待补'
        sub['K1_FINDING1_HEAD'] = '发现 1：待补（K=1 对照重跑中）'
        sub['K1_FINDING1_BODY'] = '待补'
        sub['K1_MAINCLAIM_CELL'] = '待补'
        sub['K1_BOUNDARY_ITEM'] = '1. **待补**'
        sub_en['K1_VERDICT'] = 'pending (K=1 control re-running)'
        sub_en['K1_VERDICT_SHORT'] = 'pending'
        sub_en['K1_H6'] = 'pending'
        sub_en['K1_VERDICT_DETAIL'] = 'pending'
        sub_en['K1_CLAIM_MAIN'] = 'pending (K=1 control re-running)'
        sub_en['K1_CLAIM_SHORT'] = 'pending'
        sub_en['K1_FINDING1_HEAD'] = 'Finding 1: pending (K=1 control re-running)'
        sub_en['K1_FINDING1_BODY'] = 'pending'
        sub_en['K1_MAINCLAIM_CELL'] = 'pending'
        sub_en['K1_BOUNDARY_ITEM'] = '1. **pending**'

    # ── LOPO ──
    # run_lopo.py 写盘路径为 results/regimemamba/lopo/<tag>/results.json（tag 默认 'lopo'），
    # 即 results/regimemamba/lopo/lopo/results.json；这里按多个候选位置依次查找，
    # 避免因目录层级不一致而静默回退成「待补」。
    lopo_path = None
    _lopo_candidates = [
        os.path.join(REG, 'lopo', 'lopo', 'results.json'),   # tag='lopo'（当前实际布局）
        os.path.join(REG, 'lopo', 'results.json'),            # tag 缺省/直接落盘布局
    ]
    try:
        for _d, _subdir, _files in os.walk(os.path.join(REG, 'lopo')):
            if 'results.json' in _files:
                _lopo_candidates.append(os.path.join(_d, 'results.json'))
    except OSError:
        pass
    for _c in _lopo_candidates:
        if os.path.exists(_c):
            lopo_path = _c
            break
    print(f'[fill] LOPO results 路径: {lopo_path}')
    if lopo_path:
        lp = json.load(open(lopo_path))
        s = lp['summary']
        dflt = s.get('default', {}); dec = s.get('deconf', {})
        a = dflt.get('auc_weighted', float('nan'))
        a_dec = dec.get('auc_weighted', float('nan'))
        sub['LOPO_DEFAULT'] = f'{a:.4f}'
        sub['LOPO_BAL_DEFAULT'] = f"{dflt.get('bal_weighted', float('nan')):.4f}"
        sub['LOPO_DECONF'] = f'{a_dec:.4f}'
        sub['LOPO_BAL_DECONF'] = f"{dec.get('bal_weighted', float('nan')):.4f}"
        if np.isfinite(a):
            if a >= 0.6169:
                sub['LOPO_VERDICT'] = (f'RegimeMamba 的 LOPO 加权 AUC 为 {a:.4f}，'
                                       f'高于 MetaMamba 默认臂（0.6169）与 RF（0.6123）→ H5 通过。')
                sub_en['LOPO_VERDICT'] = (
                    f'RegimeMamba attains a weighted LOPO AUC of {a:.4f}, above the same-protocol MetaMamba '
                    'arm (0.6169) and RF (0.6123) -> H5 passes.')
                sub['LOPO_VERDICT_SHORT'] = '✅ 通过'
                sub_en['LOPO_VERDICT_SHORT'] = '✅ passes'
                sub['LOPO_H5'] = f'{a:.4f} ≥ 0.6169'
                sub_en['LOPO_H5'] = f'{a:.4f} ≥ 0.6169'
                sub['LOPO_BOUNDARY'] = ('LOPO 加权 AUC ' + f'{a:.4f}' +
                                         '，高于同协议 MetaMamba（0.6169）与 RF（0.6123），但仍远低于分布内水平（0.95），'
                                         '且样本量小（每 part 仅数十人），不构成"跨课程泛化已解决"的证据。')
                sub_en['LOPO_BOUNDARY'] = (
                    'The weighted LOPO AUC is ' + f'{a:.4f}' +
                    ', above the same-protocol MetaMamba (0.6169) and RF (0.6123), but still far below the '
                    'in-distribution level (0.95), and the sample is small (a few dozen students per part), so '
                    'this is not evidence that cross-curriculum generalisation has been solved.')
            else:
                sub['LOPO_VERDICT'] = (f'RegimeMamba 的 LOPO 加权 AUC 为 {a:.4f}，'
                                       f'未达到预注册阈值 0.6169 → H5 判负；'
                                       f'去混淆臂为 {a_dec:.4f}。')
                sub_en['LOPO_VERDICT'] = (
                    f'RegimeMamba attains a weighted LOPO AUC of {a:.4f}, below the preregistered threshold '
                    f'0.6169 -> H5 fails; the deconfounded arm scores {a_dec:.4f}.')
                sub['LOPO_VERDICT_SHORT'] = '❌ 判负'
                sub_en['LOPO_VERDICT_SHORT'] = '❌ fails'
                sub['LOPO_H5'] = f'{a:.4f} < 0.6169'
                sub_en['LOPO_H5'] = f'{a:.4f} < 0.6169'
                sub['LOPO_BOUNDARY'] = ('LOPO 加权 AUC ' + f'{a:.4f}' +
                                         '，未超过同协议 MetaMamba（0.6169）→ 不主张跨任务泛化优势。')
                sub_en['LOPO_BOUNDARY'] = (
                    'The weighted LOPO AUC is ' + f'{a:.4f}' +
                    ', below the same-protocol MetaMamba (0.6169) -> no cross-task generalisation claim is made.')
        else:
            sub['LOPO_VERDICT'] = 'LOPO 加权 AUC 为 NaN/非有限值，无法判定 H5（判为未完成）。'
            sub_en['LOPO_VERDICT'] = ('The weighted LOPO AUC is NaN/non-finite, so H5 cannot be adjudicated '
                                       '(reported as incomplete).')
            sub['LOPO_VERDICT_SHORT'] = '⚠️ 未完成（AUC 非有限）'
            sub_en['LOPO_VERDICT_SHORT'] = '⚠️ incomplete (non-finite AUC)'
            sub['LOPO_H5'] = 'NaN'
            sub_en['LOPO_H5'] = 'NaN'
            sub['LOPO_BOUNDARY'] = 'LOPO 结果为非有限值，本轮不作跨任务泛化主张。'
            sub_en['LOPO_BOUNDARY'] = 'The LOPO result is non-finite; no cross-task generalisation claim is made.'
    else:
        for k in ['LOPO_DEFAULT', 'LOPO_BAL_DEFAULT', 'LOPO_DECONF', 'LOPO_BAL_DECONF']:
            sub[k] = '待补'
        sub['LOPO_VERDICT'] = '待补（LOPO 运行中）'
        sub['LOPO_VERDICT_SHORT'] = '待补'
        sub['LOPO_H5'] = '待补'
        sub['LOPO_BOUNDARY'] = '待补（LOPO 运行中）'
        sub_en['LOPO_VERDICT'] = 'pending (LOPO still running)'
        sub_en['LOPO_VERDICT_SHORT'] = 'pending'
        sub_en['LOPO_H5'] = 'pending'
        sub_en['LOPO_BOUNDARY'] = 'pending (LOPO still running)'

    # ── 写回论文 ──
    for lang, fn in PAPER_FILES.items():
        p = os.path.join(DOCS, fn)
        if not os.path.exists(p):
            print(f'[fill] missing {p}')
            continue
        vals = dict(sub)
        if lang == 'en':
            vals.update(sub_en)      # 英文叙述覆盖中文叙述
        txt = open(p).read()
        n = 0
        for k, v in vals.items():
            token = '{{' + k + '}}'
            if token in txt:
                txt = txt.replace(token, str(v)); n += 1
        left = txt.count('{{')
        open(p, 'w').write(txt)
        print(f'[fill] {fn}: {n} placeholders filled, {left} unresolved marker(s) left')

    print(json.dumps({k: v for k, v in sub.items()}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
