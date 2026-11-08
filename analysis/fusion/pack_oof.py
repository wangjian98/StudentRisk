import numpy as np, os, sys

ROOT = '/root/.openclaw/workspace-staging/pc-ceo_assistant/StudentRisk/results/rev12'
GROUPS = {
    'id': os.path.join(ROOT, 'exp1_matrix'),
    'deconf': os.path.join(ROOT, 'exp1_matrix_deconf'),
    'pfx': os.path.join(ROOT, 'exp2_prefix'),
    'pfxd': os.path.join(ROOT, 'exp2_prefix_deconf'),
}
out = {}
for g, d in GROUPS.items():
    if not os.path.isdir(d):
        continue
    for c in sorted(os.listdir(d)):
        p = os.path.join(d, c, 'oof_probs.npy')
        if os.path.exists(p):
            out[f'{g}__{c}'] = np.load(p).astype(np.float64)
np.savez('/tmp/oof_pack.npz', **out)
print('cells:', len(out), 'bytes:', os.path.getsize('/tmp/oof_pack.npz'))
print('\n'.join(sorted(out)))
