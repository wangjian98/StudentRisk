"""阶段级实验公共件：特征、分组折、指标。

特征口径与本地 experiment_I_validations.py 完全一致（已验证），保证
"本地 RF 基线" 与 "246 上 MetaMamba" 可直接对比。
"""
import numpy as np
from sklearn.metrics import roc_auc_score

PARTS = list(range(1, 8))
EVENT_TYPES = ['text_insert', 'text_remove', 'text_paste',
               'focus_gained', 'focus_lost', 'run', 'submit']


def load_dataset(path):
    D = np.load(path, allow_pickle=True)
    out = dict(
        agg10=D['agg10'].astype(np.float64), mask=D['mask'], task_ids=D['task_ids'],
        part_ids=D['part_ids'], sid=D['sid'], drop=D['drop'], final=D['final'],
        first_ts=D['first_ts'], last_ts=D['last_ts'],
        seq11=D['seq11'], seq9=D['seq9'], seq7=D['seq7'],
    )
    out['RAW'] = out['agg10']
    out['part'] = out['part_ids']
    out['stage_idx'] = _stage_index(out['sid'])
    return out


def _stage_index(sid):
    """每个学生在自己的 part 序列中的序号（0,1,2...）"""
    out = np.zeros(len(sid), dtype=np.int64)
    seen = {}
    for i, s in enumerate(sid):
        s = int(s)
        out[i] = seen.get(s, 0)
        seen[s] = out[i] + 1
    return out


def base_exercises(agg10, part_ids):
    """每 part 题量基准 = 全体 distinct_exercises 众数"""
    from collections import Counter
    base = {}
    for p in np.unique(part_ids):
        c = Counter(agg10[part_ids == p][:, 8].astype(int))
        base[int(p)] = c.most_common(1)[0][0]
    return base


def derive(ds):
    """DER: 9 维派生特征（与本地一致）"""
    RAW = ds['RAW']
    part = ds['part']
    SPAN = np.maximum(RAW[:, 9], 1e-3)
    INS = np.maximum(RAW[:, 0], 1e-3)
    base = ds['base_ex']
    der = np.column_stack([
        np.log1p(RAW[:, 7]), np.log1p(RAW[:, 8]), np.log1p(RAW[:, 9]),
        np.log1p(RAW[:, 7] / SPAN), RAW[:, 1] / INS, RAW[:, 4] / SPAN,
        RAW[:, 6] / np.maximum(RAW[:, 7], 1),
        np.array([RAW[j, 8] / base[part[j]] for j in range(len(RAW))]),
        ds['stage_idx'].astype(float),
    ])
    return der


def self_ref(ds):
    """自参照：当前阶段相对自己前几阶段的偏离"""
    RAW = ds['RAW']
    sid = ds['sid']
    n = len(RAW)
    out = np.zeros((n, RAW.shape[1]))
    by = {}
    for j in range(n):
        by.setdefault(int(sid[j]), []).append(j)
    for s, js in by.items():
        js = sorted(js, key=lambda j: ds['stage_idx'][j])
        for pos, j in enumerate(js):
            if pos:
                h = RAW[js[:pos]]
                out[j] = (RAW[j] - h.mean(0)) / (h.std(0) + 1e-9)
    return out


def peer_z(ds, tr_rows, tgt_rows, group_full):
    RAW = ds['RAW']
    out = np.zeros((len(tgt_rows), RAW.shape[1]))
    for g in np.unique(group_full[tr_rows]):
        tr_g = np.intersect1d(np.where(group_full == g)[0], tr_rows)
        if len(tr_g) == 0:
            continue
        Xg = RAW[tr_g]
        med = np.median(Xg, 0)
        mad = np.maximum(np.median(np.abs(Xg - med), 0), 1e-3)
        for i, r in enumerate(tgt_rows):
            if group_full[r] == g:
                out[i] = (RAW[r] - med) / mad
    return np.clip(out, -5, 5)


def group_folds(sid, student_label, n_splits=5, seed=42):
    """按学生分组 + 按学生课程级标签分层的 K 折（同一学生不跨折）。"""
    sid = np.asarray(sid)
    uniq = np.unique(sid)
    lab = {int(s): int(student_label[sid == s][0]) for s in uniq}
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(n_splits)]
    for cls in (0, 1):
        ss = [int(s) for s in uniq if lab[int(s)] == cls]
        rng.shuffle(ss)
        for i, s in enumerate(ss):
            buckets[i % n_splits].append(s)
    out = []
    for b in range(n_splits):
        te_s = set(buckets[b])
        te = np.array([j for j in range(len(sid)) if int(sid[j]) in te_s])
        tr = np.array([j for j in range(len(sid)) if int(sid[j]) not in te_s])
        out.append((tr, te))
    return out


def per_part_auc(y, oof, group):
    out = {}
    for g in np.unique(group):
        m = group == g
        if len(set(y[m])) > 1:
            out[int(g)] = dict(n=int(m.sum()), pos=int(y[m].sum()),
                               auc=float(roc_auc_score(y[m], oof[m])))
    return out


def pooled(pp):
    if not pp:
        return float('nan')
    a = np.array([v['auc'] for v in pp.values()])
    n = np.array([v['n'] for v in pp.values()], float)
    return float(np.average(a, weights=n))


def judged(pp):
    j = [v['auc'] for v in pp.values() if v['pos'] >= 10 and v['n'] - v['pos'] >= 10]
    return float(np.mean(j)) if j else float('nan')


def metrics(y, oof, group, tag=''):
    pp = per_part_auc(y, oof, group)
    return dict(tag=tag, auc=float(roc_auc_score(y, oof)), pooled=pooled(pp),
                judged=judged(pp), per_part={str(k): v for k, v in pp.items()})


def stage_prior_oof(y, group, folds):
    pr = np.zeros(len(y))
    for tr, te in folds:
        for g in np.unique(group[tr]):
            mtr = tr[group[tr] == g]
            mte = te[group[te] == g]
            if len(mtr):
                pr[mte] = y[mtr].mean()
            elif len(mte):
                pr[mte] = y[tr].mean()
    return pr


def paired_boot(y, oa, ob, group, n=2000, seed=20261030):
    rng = np.random.default_rng(seed)
    idx_by_g = [np.where(group == g)[0] for g in np.unique(group)]
    obs = pooled(per_part_auc(y, oa, group)) - pooled(per_part_auc(y, ob, group))
    ds = np.empty(n)
    for it in range(n):
        pick = np.concatenate([rng.choice(ii, len(ii), replace=True) for ii in idx_by_g])
        ds[it] = (pooled(per_part_auc(y[pick], oa[pick], group[pick]))
                  - pooled(per_part_auc(y[pick], ob[pick], group[pick])))
    return dict(obs=float(obs), lo=float(np.percentile(ds, 2.5)),
                hi=float(np.percentile(ds, 97.5)), p_le0=float((ds <= 0).mean()))
