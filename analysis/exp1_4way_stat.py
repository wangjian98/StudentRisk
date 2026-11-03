"""实验1：4-way 消融 × 7d/11d × 3 seeds 统计显著性分析

不做新训练，复用已有 results/ 下的 4-way × 2 windows × 3 seeds × 5 folds = 120 fold records
"""
import json, os
from pathlib import Path
import numpy as np
from scipy import stats

RES = Path('/root/.openclaw/workspace-staging/pc-ceo_assistant/StudentRisk/results')

VARIANTS = {
    'full':          {'use_film': True,  'tc': 0.3},
    'no_tc':         {'use_film': True,  'tc': 0.0},
    'no_film':       {'use_film': False, 'tc': 0.3},
    'no_film_no_tc': {'use_film': False, 'tc': 0.0},
}

WINDOWS = {
    '7d': {
        'full':          RES / 'meta_mamba_7d',
        'no_tc':         RES / 'meta_mamba_7d_ablation_no_tc',
        'no_film':       RES / 'meta_mamba_7d_ablation_no_film',
        'no_film_no_tc': RES / 'meta_mamba_7d_ablation_no_film_no_tc',
    },
    '11d': {
        'full':          RES / 'meta_mamba',
        'no_tc':         RES / 'meta_mamba_ablation_no_tc',
        'no_film':       RES / 'meta_mamba_ablation_no_film',
        'no_film_no_tc': RES / 'meta_mamba_ablation_no_film_no_tc',
    },
}

def load_fold_records(p: Path):
    if not (p / 'results.json').exists():
        return None
    d = json.load(open(p / 'results.json'))
    records = d.get('per_fold', [])
    # 每条 record 是 {seed, fold, macro_f1, f1_class_1, roc_auc, ...}
    return records

def paired_t(a, b):
    a, b = np.array(a), np.array(b)
    if len(a) < 3:
        return None, None
    t, p = stats.ttest_rel(a, b)
    return float(t), float(p)

def cohens_d_paired(a, b):
    a, b = np.array(a), np.array(b)
    d = (a - b).mean() / (a - b).std(ddof=1)
    return float(d)

def main():
    print('=' * 100)
    print(' 实验1 报告：4-way 消融 × 7d/11d × 3 seeds × 5 folds 统计显著性分析')
    print('=' * 100)
    print()

    # 1) 加载所有数据
    data = {}  # data[window][variant] = list of fold records
    for win, m in WINDOWS.items():
        data[win] = {}
        for var, p in m.items():
            recs = load_fold_records(p)
            if recs is None:
                print(f'[WARN] {win}/{var} MISSING')
                data[win][var] = None
            else:
                data[win][var] = recs
                print(f'[OK]   {win}/{var}: {len(recs)} fold records (seeds: {sorted(set(r["seed"] for r in recs))})')

    print()
    print('=' * 100)
    print(' 1. 描述性统计 (Macro-F1 / f1_class_1 / ROC-AUC: mean ± std over 15 fold records)')
    print('=' * 100)
    print()
    header = f"{'window':<7} {'variant':<15} {'FiLM':<5} {'TC':<5} {'macro_f1':<14} {'f1_class_1':<14} {'roc_auc':<14}"
    print(header)
    print('-' * len(header))
    summary = {}
    for win in ['7d', '11d']:
        for var in ['full', 'no_tc', 'no_film', 'no_film_no_tc']:
            recs = data[win].get(var)
            if recs is None:
                continue
            mac = [r['macro_f1'] for r in recs]
            f1c = [r['f1_class_1'] for r in recs]
            auc = [r['roc_auc'] for r in recs]
            print(f"{win:<7} {var:<15} {str(VARIANTS[var]['use_film'])[0]:<5} {VARIANTS[var]['tc']:<5} "
                  f"{np.mean(mac):.4f}±{np.std(mac):.4f}  {np.mean(f1c):.4f}±{np.std(f1c):.4f}  "
                  f"{np.mean(auc):.4f}±{np.std(auc):.4f}")
            summary[(win, var)] = {'macro_f1': mac, 'f1_class_1': f1c, 'roc_auc': auc}
    print()

    # 2) 配对 t 检验 (paired by seed,fold)
    print('=' * 100)
    print(' 2. 配对 t 检验 (paired by seed×fold): Full vs 其他三个变体')
    print('=' * 100)
    print()
    for win in ['7d', '11d']:
        full = summary.get((win, 'full'))
        if not full:
            continue
        print(f'--- {win} (Full baseline: macro_f1={np.mean(full["macro_f1"]):.4f}±{np.std(full["macro_f1"]):.4f}) ---')
        for var in ['no_tc', 'no_film', 'no_film_no_tc']:
            other = summary.get((win, var))
            if not other:
                continue
            print(f'\n  vs {var}:')
            for metric in ['macro_f1', 'f1_class_1', 'roc_auc']:
                a, b = full[metric], other[metric]
                if len(a) != len(b):
                    print(f'    {metric:<12}  n mismatch ({len(a)} vs {len(b)})')
                    continue
                t, p = paired_t(a, b)
                d = cohens_d_paired(a, b)
                delta = np.mean(b) - np.mean(a)
                sig = '***' if p is not None and p < 0.001 else '**' if p is not None and p < 0.01 else '*' if p is not None and p < 0.05 else 'ns'
                print(f'    {metric:<12}  Δ={delta:+.4f}  t={t if t is not None else 0:6.2f}  p={p if p is not None else 0:.4f} {sig:<3}  d={d:+.3f}')
        print()

    # 3) 关键对比：FiLM 加 / 不加
    print('=' * 100)
    print(' 3. FiLM 主效应：固定 TC，比较 (full + no_tc) vs (no_film + no_film_no_tc)')
    print('=' * 100)
    print()
    for win in ['7d', '11d']:
        # 把 macro_f1 在 seed/fold 维度上对齐
        with_film = []
        without_film = []
        for s in [42, 123, 777]:
            for f in range(5):
                w_film_03 = next((r for r in data[win]['full'] if r['seed']==s and r['fold']==f), None)
                w_film_00 = next((r for r in data[win]['no_tc'] if r['seed']==s and r['fold']==f), None)
                wo_film_03 = next((r for r in data[win]['no_film'] if r['seed']==s and r['fold']==f), None)
                wo_film_00 = next((r for r in data[win]['no_film_no_tc'] if r['seed']==s and r['fold']==f), None)
                if all([w_film_03, w_film_00, wo_film_03, wo_film_00]):
                    with_film.append((w_film_03['macro_f1'] + w_film_00['macro_f1']) / 2)
                    without_film.append((wo_film_03['macro_f1'] + wo_film_00['macro_f1']) / 2)
        if len(with_film) < 3:
            continue
        t, p = paired_t(with_film, without_film)
        d = cohens_d_paired(with_film, without_film)
        delta = np.mean(without_film) - np.mean(with_film)
        sig = '***' if p is not None and p < 0.001 else '**' if p is not None and p < 0.01 else '*' if p is not None and p < 0.05 else 'ns'
        print(f'  {win}: with_film={np.mean(with_film):.4f}±{np.std(with_film):.4f}   '
              f'without_film={np.mean(without_film):.4f}±{np.std(without_film):.4f}   '
              f'Δ={delta:+.4f}  t={t:.2f}  p={p:.4f} {sig}  d={d:+.3f}')
    print()

    # 4) 关键对比：TC 加 / 不加
    print('=' * 100)
    print(' 4. TC 主效应：固定 FiLM，比较 (full + no_film) vs (no_tc + no_film_no_tc)')
    print('=' * 100)
    print()
    for win in ['7d', '11d']:
        with_tc = []
        without_tc = []
        for s in [42, 123, 777]:
            for f in range(5):
                w_tc_full = next((r for r in data[win]['full'] if r['seed']==s and r['fold']==f), None)
                w_tc_nofilm = next((r for r in data[win]['no_film'] if r['seed']==s and r['fold']==f), None)
                wo_tc_notc = next((r for r in data[win]['no_tc'] if r['seed']==s and r['fold']==f), None)
                wo_tc_nofilm_notc = next((r for r in data[win]['no_film_no_tc'] if r['seed']==s and r['fold']==f), None)
                if all([w_tc_full, w_tc_nofilm, wo_tc_notc, wo_tc_nofilm_notc]):
                    with_tc.append((w_tc_full['macro_f1'] + w_tc_nofilm['macro_f1']) / 2)
                    without_tc.append((wo_tc_notc['macro_f1'] + wo_tc_nofilm_notc['macro_f1']) / 2)
        if len(with_tc) < 3:
            continue
        t, p = paired_t(with_tc, without_tc)
        d = cohens_d_paired(with_tc, without_tc)
        delta = np.mean(without_tc) - np.mean(with_tc)
        sig = '***' if p is not None and p < 0.001 else '**' if p is not None and p < 0.01 else '*' if p is not None and p < 0.05 else 'ns'
        print(f'  {win}: with_tc={np.mean(with_tc):.4f}±{np.std(with_tc):.4f}   '
              f'without_tc={np.mean(without_tc):.4f}±{np.std(without_tc):.4f}   '
              f'Δ={delta:+.4f}  t={t:.2f}  p={p:.4f} {sig}  d={d:+.3f}')
    print()

    # 5) 7d vs 11d 一致性
    print('=' * 100)
    print(' 5. 7d vs 11d 一致性 (Same variant, different feature dim)')
    print('=' * 100)
    print()
    print(f"{'variant':<15} {'7d macro_f1':<16} {'11d macro_f1':<16} {'Δ_11d_7d':<10} {'ratio':<8}")
    print('-' * 70)
    for var in ['full', 'no_tc', 'no_film', 'no_film_no_tc']:
        s7 = summary.get(('7d', var))
        s11 = summary.get(('11d', var))
        if not s7 or not s11:
            continue
        m7, m11 = np.mean(s7['macro_f1']), np.mean(s11['macro_f1'])
        delta = m11 - m7
        ratio = (m11 / m7) if m7 > 0 else 0
        print(f"{var:<15} {m7:.4f}±{np.std(s7['macro_f1']):.4f}    {m11:.4f}±{np.std(s11['macro_f1']):.4f}    {delta:+.4f}     {ratio:.3f}")

if __name__ == '__main__':
    main()
