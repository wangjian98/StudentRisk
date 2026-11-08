"""rev12 融合权重 —— 嵌套（交叉拟合）选择，避免"同数据选权重"的乐观上界

做法：
  用仓库同一套折划分（StratifiedKFold 5 折 × seed 42/123/777）重建每生所属 fold；
  对每个 (seed, fold)：在其余 4 折的 OOF 概率上扫权重 w，选最优 w；
  再把该 w 应用于本折 OOF 概率 → 得到"交叉拟合的融合预测"；
  池化后计算 AUC（这才是可对外引用的融合数字）。
  同时对比：固定权重 0.5 / 全 MetaMamba / 全 RF。
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold

Z = np.load('/home/ubuntu/SR12_cache/events.npz')
y = Z['y'].astype(int)
P = np.load('/tmp/oof_pack.npz')


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


def folds_of(seed, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    out = {}
    for f, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        for i in va:
            out[i] = f
    return out


def crossfit(prefix, ws=np.arange(0, 1.01, 0.05)):
    """prefix: 'p0.1' 等；返回 (交叉拟合融合预测, 每折选中 w 列表)"""
    kr, km = f'pfx__rf_d28_{prefix}', f'pfx__metamamba_d28_{prefix}'
    if kr not in P or km not in P:
        return None, None
    pr, pm = P[kr], P[km]
    rr, rm = rank(pr), rank(pm)
    pred = np.zeros(len(y)); chosen = []
    for seed in (42, 123, 777):
        fmap = folds_of(seed)
        fids = np.array([fmap[i] for i in range(len(y))])
        for f in range(5):
            tr = np.where(fids != f)[0]; va = np.where(fids == f)[0]
            best = (-1, None)
            for w in ws:
                a = auc_fast(w * rr[tr] + (1 - w) * rm[tr], y[tr])
                if a > best[0]:
                    best = (a, w)
            w = best[1]; chosen.append((seed, f, float(w)))
            pred[va] = w * rr[va] + (1 - w) * rm[va]
    return pred, chosen


print('=' * 80)
print('【嵌套交叉拟合的融合权重 + 融合 AUC（可对外引用口径）】')
print('=' * 80)
for prefix in ['p0.1', 'p0.2', 'p0.4', 'p1.0']:
    kr, km = f'pfx__rf_d28_{prefix}', f'pfx__metamamba_d28_{prefix}'
    if kr not in P or km not in P:
        continue
    pr, pm = P[kr], P[km]
    rr, rm = rank(pr), rank(pm)
    pred, chosen = crossfit(prefix)
    ws = [c[2] for c in chosen]
    print(f'\n--- 前缀 {prefix} ---')
    print(f'  RF-28d            AUC={auc_fast(pr):.4f}')
    print(f'  MetaMamba-28d     AUC={auc_fast(pm):.4f}')
    print(f'  固定 50/50 rank   AUC={auc_fast((rr + rm) / 2):.4f}')
    print(f'  交叉拟合融合      AUC={auc_fast(pred):.4f}   选中 w_RF: 均值={np.mean(ws):.2f} '
          f'中位={np.median(ws):.2f} 范围[{min(ws):.2f},{max(ws):.2f}]')
    d, lo, hi, pn = boot_delta(pred, pr)
    print(f'      Δ vs RF = {d:+.4f} [{lo:+.4f},{hi:+.4f}] P(Δ<=0)={pn:.3f}')
    d2, lo2, hi2, pn2 = boot_delta(pred, pm)
    print(f'      Δ vs MM = {d2:+.4f} [{lo2:+.4f},{hi2:+.4f}] P(Δ<=0)={pn2:.3f}')

print()
print('=' * 80)
print('【全程（p1.0）跨族融合：RF + MetaMamba + LSTM + Attention + HMM-GMM】')
print('=' * 80)
cells = ['rf_d28', 'metamamba_d28', 'lstm_d28', 'attention_d28', 'hmm_gmm_d28']
R = {c: rank(P[f'id__{c}']) for c in cells}
pred = np.zeros(len(y)); chosen = []
for seed in (42, 123, 777):
    fmap = folds_of(seed); fids = np.array([fmap[i] for i in range(len(y))])
    for f in range(5):
        tr = np.where(fids != f)[0]; va = np.where(fids == f)[0]
        best = (-1, None)
        for w_rf in np.arange(0, 1.01, 0.1):
            for w_mm in np.arange(0, 1.01 - w_rf + 1e-9, 0.1):
                rest = max(1 - w_rf - w_mm, 0)
                p = w_rf * R['rf_d28'] + w_mm * R['metamamba_d28'] + rest / 3 * (
                    R['lstm_d28'] + R['attention_d28'] + R['hmm_gmm_d28'])
                a = auc_fast(p[tr], y[tr])
                if a > best[0]:
                    best = (a, (w_rf, w_mm, rest))
        w = best[1]; chosen.append(w)
        va_pred = w[0] * R['rf_d28'] + w[1] * R['metamamba_d28'] + w[2] / 3 * (
            R['lstm_d28'] + R['attention_d28'] + R['hmm_gmm_d28'])
        pred[va] = va_pred[va]
print(f'  RF-28d 单模型      AUC={auc_fast(P["id__rf_d28"]):.4f}')
print(f'  交叉拟合多族融合   AUC={auc_fast(pred):.4f}')
d, lo, hi, pn = boot_delta(pred, P['id__rf_d28'])
print(f'  Δ vs RF = {d:+.4f} [{lo:+.4f},{hi:+.4f}] P(Δ<=0)={pn:.3f}')
import collections
print('  权重选择分布 (w_RF, w_MM, w_rest) top5:', collections.Counter([tuple(np.round(c, 1)) for c in chosen]).most_common(5))
print('\n[nested_fusion] DONE')
