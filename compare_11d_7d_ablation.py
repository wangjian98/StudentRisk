"""Quick comparison: 11d vs 7d ablation impact."""
import json, os
RES = '/home/ubuntu/StudentRisk/results'

def load(rel):
    pj = os.path.join(RES, rel, 'results.json')
    if not os.path.exists(pj):
        return None
    d = json.load(open(pj))
    pf = d['per_fold_summary']
    uf = d.get('use_film')
    tc = d.get('tc_weight')
    if uf is None:
        ab = d.get('ablation', {})
        uf = not ab.get('no_film', False)
        tc = 0.0 if ab.get('no_tc', False) else 0.3
    return {
        'rel': rel, 'use_film': uf, 'tc': tc,
        'macro_f1': pf['macro_f1_mean'], 'std': pf['macro_f1_std'],
        'roc_auc': pf['roc_auc_mean'], 'f1_fail': pf['f1_class_1_mean'],
        'n_seeds': d.get('n_seeds', 0),
    }


VERSIONS = {
    '11d': {
        'full':          'meta_mamba',
        'no_tc':         'meta_mamba_ablation_no_tc',
        'no_film':       'meta_mamba_ablation_no_film',
        'no_film_no_tc': 'meta_mamba_ablation_no_film_no_tc',
    },
    '7d': {
        'full':          'meta_mamba_7d',
        'no_tc':         'meta_mamba_7d_ablation_no_tc',
        'no_film':       'meta_mamba_7d_ablation_no_film',
        'no_film_no_tc': 'meta_mamba_7d_ablation_no_film_no_tc',
    },
}


def main():
    print('=' * 90)
    print(' MetaMamba 11d vs 7d -- FiLM & TC ablation comparison')
    print('=' * 90)
    print()
    print('11d input = (B, L, 11)  -- 7 event types one-hot + 4 continuous features')
    print('7d  input = (B, L,  7)  -- only 7 event types one-hot')
    print('BOTH still receive task_ids as model input (not in sequence feature)')
    print('Protocol: 5-fold StratifiedKFold x 3 seeds (42/123/777)')
    print()

    rows = []
    for ver, mapping in VERSIONS.items():
        for label, rel in mapping.items():
            r = load(rel)
            if r is None:
                continue
            rows.append({'version': ver, 'label': label, **r})

    print('%-6s  %-15s  %-5s  %-5s  %-22s  %-8s  %-8s' % (
        'ver', 'variant', 'FiLM', 'TC', 'macro_f1 (mean+/-std)', 'roc_auc', 'f1_fail'))
    print('-' * 90)
    for r in rows:
        uf_s = 'T' if r['use_film'] else 'F'
        tc_s = '%.1f' % r['tc']
        print('%-6s  %-15s  %-5s  %-5s  %.4f+/-%.4f     %.4f    %.4f' % (
            r['version'], r['label'], uf_s, tc_s,
            r['macro_f1'], r['std'], r['roc_auc'], r['f1_fail']))

    print()
    print('=' * 90)
    print(' Delta contribution (Macro-F1):')
    print('=' * 90)
    for ver in ['11d', '7d']:
        sub = [r for r in rows if r['version'] == ver]
        if not sub:
            continue
        full = next(r for r in sub if r['label'] == 'full')
        print()
        print('[%s] baseline full macro_f1 = %.4f +/- %.4f' % (
            ver, full['macro_f1'], full['std']))
        for label in ['no_tc', 'no_film', 'no_film_no_tc']:
            row = next((r for r in sub if r['label'] == label), None)
            if row is None:
                continue
            delta = row['macro_f1'] - full['macro_f1']
            std_ratio = row['std'] / max(full['std'], 1e-6)
            print('  %-15s  macro_f1=%.4f+/-%.4f   delta=%+.4f   std x%.2f' % (
                label, row['macro_f1'], row['std'], delta, std_ratio))


if __name__ == '__main__':
    main()
