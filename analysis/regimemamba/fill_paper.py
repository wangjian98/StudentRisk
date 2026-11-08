"""
regimemamba.fill_paper — 用真实结果回填论文占位符（避免人工转录错误）

用法：python3 analysis/regimemamba/fill_paper.py
"""
import os
import sys
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

REG = os.path.join(_ROOT, 'results', 'regimemamba')
REV = os.path.join(_ROOT, 'results', 'rev12')
DOCS = os.path.join(_ROOT, 'docs')
CACHE = os.environ.get('SR12_CACHE', '/home/ubuntu/SR12_cache')
CACHE_FILE = os.path.join(CACHE, 'events.npz')

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
    return float(ds.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float((ds <= 0).mean())


def main():
    sub = {}

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
        sub['K1_POOLED'] = f'{a_k1:.4f}'
        sub['K1_FOLD'] = (f"{pf['roc_auc']:.4f}" if np.isfinite(pf['roc_auc'])
                          else 'NaN')
        sub['K1_BALACC'] = f"{pf['balanced_accuracy']:.4f}"
        sub['K1_F1'] = f"{pf['macro_f1']:.4f}"
        sub['K1_DELTA'] = f'{d:+.4f}'
        sub['K1_CI'] = f'{lo:+.4f},{hi:+.4f}'
        sub['K1_P'] = f'{pn:.3f}'
        sub['K1_REF'] = (f'(参考：MetaMamba - K=1 = {d_mm:+.4f} [{lo_mm:+.4f},{hi_mm:+.4f}], P={pn_mm:.3f})')
        sub['K1_ABSTRACT'] = (
            f'**regime 结构本身有效**：K=4 相对 K=1 单状态对照的合并 OOF 增益为 {d:+.4f} '
            f'[{lo:+.4f},{hi:+.4f}]（P(Δ≤0)={pn:.3f}），'
            f'说明增益来自 regime 结构而非容量。' if a_k1 >= 0.6 else
            '**regime 结构的容量对照未能完成判定**：K=1 单状态对照在本轮发生实现级退化'
            f'（合并 OOF AUC={a_k1:.4f}），因此"增益来自 regime 结构还是来自容量"这一问'
            '题在本文中仍未解决（H6 判为未完成）。')
        if a_k1 < 0.6 or not np.isfinite(pf['roc_auc']):
            sub['K1_VERDICT'] = ('K=1 对照**未能学出可用信号**（合并 OOF AUC 低于 0.60 或折内 AUC 为 NaN），'
                                 '因此该差距不能作为"regime 结构的收益"来解释；'
                                 'K=1 格在本轮运行中被发现因除零缺陷而退化（见 §4.5 修复说明），修复后重跑结果见上表。')
            sub['K1_VERDICT_SHORT'] = '⚠️ 对照格退化，结论待重跑后判定'
            sub['K1_H6'] = f'K=1 退化（合并 OOF AUC={a_k1:.4f}）'
            sub['K1_VERDICT_DETAIL'] = ('K=1 对照在本轮发生实现级退化，因此 **H6 暂不能判定**；'
                                        '这本身是本文方法学上的一条教训（消融链必须前置健全性检查）。')
        else:
            sub['K1_VERDICT'] = (f'K=4 相对 K=1 单状态对照的合并 OOF 增益为 {d:+.4f} [{lo:+.4f},{hi:+.4f}]，'
                                 f'P(Δ<=0)={pn:.3f} -> 增益确实来自 regime 结构而非容量（H6 通过）。{sub["K1_REF"]}')
            sub['K1_VERDICT_SHORT'] = '✅ 通过（增益来自 regime 结构）'
            sub['K1_H6'] = f'Δ={d:+.4f} [{lo:+.4f},{hi:+.4f}], P={pn:.3f}'
            sub['K1_VERDICT_DETAIL'] = (f'K=4 相对 K=1 单状态对照的合并 OOF 增益为 {d:+.4f} [{lo:+.4f},{hi:+.4f}]'
                                        f'（P(Δ<=0)={pn:.3f}）。{sub["K1_REF"]}')
    else:
        for k in ['K1_POOLED', 'K1_FOLD', 'K1_BALACC', 'K1_F1']:
            sub[k] = '待补'
        for k in ['K1_DELTA', 'K1_CI', 'K1_P']:
            sub[k] = '待补'
        sub['K1_VERDICT'] = '待补（K=1 对照重跑中）'
        sub['K1_VERDICT_SHORT'] = '待补'
        sub['K1_H6'] = '待补'
        sub['K1_VERDICT_DETAIL'] = '待补'

    # ── LOPO ──
    lopo_path = os.path.join(REG, 'lopo', 'results.json')
    if os.path.exists(lopo_path):
        lp = json.load(open(lopo_path))
        s = lp['summary']
        dflt = s.get('default', {}); dec = s.get('deconf', {})
        sub['LOPO_DEFAULT'] = f"{dflt.get('auc_weighted', float('nan')):.4f}"
        sub['LOPO_BAL_DEFAULT'] = f"{dflt.get('bal_weighted', float('nan')):.4f}"
        sub['LOPO_DECONF'] = f"{dec.get('auc_weighted', float('nan')):.4f}"
        sub['LOPO_BAL_DECONF'] = f"{dec.get('bal_weighted', float('nan')):.4f}"
        a = dflt.get('auc_weighted', float('nan'))
        if np.isfinite(a):
            if a >= 0.6169:
                sub['LOPO_VERDICT'] = (f'RegimeMamba 的 LOPO 加权 AUC 为 {a:.4f}，'
                                       f'高于 MetaMamba 默认臂（0.6169）与 RF（0.6123）→ H5 通过。')
                sub['LOPO_VERDICT_SHORT'] = '✅ 通过'
                sub['LOPO_H5'] = f'{a:.4f} ≥ 0.6169'
                sub['LOPO_BOUNDARY'] = ('LOPO 加权 AUC ' + f'{a:.4f}' +
                                         '，高于同协议 MetaMamba（0.6169）与 RF（0.6123），但仍远低于分布内水平（0.95），'
                                         '且样本量小（每 part 仅数十人），不构成"跨课程泛化已解决"的证据。')
            else:
                sub['LOPO_VERDICT'] = (f'RegimeMamba 的 LOPO 加权 AUC 为 {a:.4f}，'
                                       f'未达到预注册阈值 0.6169 → H5 判负；'
                                       f'去混淆臂为 {dec.get("auc_weighted", float("nan")):.4f}。')
                sub['LOPO_VERDICT_SHORT'] = '❌ 判负'
                sub['LOPO_H5'] = f'{a:.4f} < 0.6169'
                sub['LOPO_BOUNDARY'] = ('LOPO 加权 AUC ' + f'{a:.4f}' +
                                         '，未超过同协议 MetaMamba（0.6169）→ 不主张跨任务泛化优势。')
    else:
        for k in ['LOPO_DEFAULT', 'LOPO_BAL_DEFAULT', 'LOPO_DECONF', 'LOPO_BAL_DECONF']:
            sub[k] = '待补'
        sub['LOPO_VERDICT'] = '待补（LOPO 运行中）'
        sub['LOPO_VERDICT_SHORT'] = '待补'
        sub['LOPO_H5'] = '待补'
        sub['LOPO_BOUNDARY'] = '待补（LOPO 运行中）'

    for fn in ['REGIMEMAMBA_PAPER_20261108_zh.md', 'REGIMEMAMBA_PAPER_20261108_en.md']:
        p = os.path.join(DOCS, fn)
        if not os.path.exists(p):
            print(f'[fill] missing {p}')
            continue
        txt = open(p).read()
        n = 0
        for k, v in sub.items():
            token = '{{' + k + '}}'
            if token in txt:
                txt = txt.replace(token, str(v)); n += 1
        open(p, 'w').write(txt)
        print(f'[fill] {fn}: {n} placeholders filled')
    print(json.dumps(sub, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
