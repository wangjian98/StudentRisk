"""rev12 融合实测 v2 —— 修正标度 + 校准 + 选择性预测 + 分前缀权重扫描

标度说明：exp1_matrix.py:77 已做 `oof = oof / n_seeds`，故 oof_probs.npy 已是 seed 均值概率，
本次不再做任何缩放。
"""
import numpy as np, json, os
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

Z = np.load('/home/ubuntu/SR12_cache/events.npz')
y = Z['y'].astype(int)
P = np.load('/tmp/oof_pack.npz')


def g(k):
    return P[k].astype(np.float64)


def rank(p):
    return (np.argsort(np.argsort(p)) + 1.0) / len(p)


def auc_fast(p, yy=None):
    yy = y if yy is None else yy
    r = np.argsort(np.argsort(p)) + 1.0          # 整数名次，越大风险越高
    npos = float(yy.sum()); nneg = float(len(yy) - npos)
    return float((r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metrics(p):
    return dict(auc=float(auc_fast(p)),
                bal_acc=float(balanced_accuracy_score(y, p >= 0.5)),
                macro_f1=float(f1_score(y, p >= 0.5, average='macro')),
                ece=ece(p))


def ece(p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            tot += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return tot


def boot_delta(pa, pb, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); ds = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        ds.append(auc_fast(pa[s], y[s]) - auc_fast(pb[s], y[s]))
    ds = np.array(ds)
    return float(ds.mean()), float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float((ds <= 0).mean())


def cv_platt(p, folds=5, seed=0):
    """5 折交叉拟合 Platt（sigmoid 校准），返回 (校准后 p, OOF ECE, OOF Brier)"""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed); idx = np.arange(len(y))
    rng.shuffle(idx)
    parts = np.array_split(idx, folds)
    out = np.zeros_like(p, dtype=float)
    for k in range(folds):
        va = parts[k]; tr = np.concatenate([parts[j] for j in range(folds) if j != k])
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(p[tr].reshape(-1, 1), y[tr])
        out[va] = lr.predict_proba(p[va].reshape(-1, 1))[:, 1]
    return out, ece(out), float(np.mean((out - y) ** 2))


def selective(p, ks=(0.03, 0.05, 0.10, 0.20)):
    order = np.argsort(-p)
    base = y.mean()
    rows = []
    for k in ks:
        n = int(len(y) * k)
        sel = order[:n]
        rows.append((k, n, float(y[sel].mean()), float(y[sel].mean() / base), float(y[sel].sum() / y.sum())))
    return rows


print('=' * 90)
print('【1】身份臂 28d 单模型（正确标度：threshold / 校准 / 选择性预测）')
print('=' * 90)
for cell in ['rf_d28', 'metamamba_d28', 'lstm_d28', 'attention_d28', 'hmm_gmm_d28']:
    p = g(f'id__{cell}')
    m = metrics(p)
    print(f'{cell:16s} AUC={m["auc"]:.4f} bal-acc={m["bal_acc"]:.4f} macroF1={m["macro_f1"]:.4f} ECE={m["ece"]:.4f}')

print()
print('=' * 90)
print('【2】融合（概率均值 / 秩均值）vs RF-28d，配对 bootstrap 2000')
print('=' * 90)
rf = g('id__rf_d28'); mm = g('id__metamamba_d28')
lg = g('id__lstm_d28'); at = g('id__attention_d28'); hg = g('id__hmm_gmm_d28')
combos = {
    'RF-28d (基线)': rf,
    'MetaMamba-28d': mm,
    'HMM-GMM-28d': hg,
    '融合A prob(RF,MM)': (rf + mm) / 2,
    '融合A rank(RF,MM)': (rank(rf) + rank(mm)) / 2,
    '融合C rank(RF,MM,HMM)': (rank(rf) + rank(mm) + rank(hg)) / 3,
    '融合D prob(RF,MM,LSTM)': (rf + mm + lg) / 3,
    '融合F rank(全5族)': (rank(rf) + rank(mm) + rank(lg) + rank(at) + rank(hg)) / 5,
}
for name, p in combos.items():
    a = auc_fast(p); m = metrics(p)
    d, lo, hi, pn = boot_delta(p, rf)
    print(f'{name:22s} AUC={a:.4f} bal-acc={m["bal_acc"]:.4f} mF1={m["macro_f1"]:.4f} ECE={m["ece"]:.4f} | '
          f'ΔAUC vs RF={d:+.4f} [{lo:+.4f},{hi:+.4f}] P(Δ<=0)={pn:.3f}')

print()
print('=' * 90)
print('【3】分前缀（早期预警）：单模型 + 融合 + 权重扫描')
print('=' * 90)
for frac in ['p0.1', 'p0.2', 'p0.4', 'p1.0']:
    kr, km = f'pfx__rf_d28_{frac}', f'pfx__metamamba_d28_{frac}'
    if kr not in P or km not in P:
        continue
    pr, pm = g(kr), g(km)
    print(f'\n--- 前缀 {frac} ---')
    others = [k for k in P if k.startswith('pfx__') and k.endswith('_' + frac) and k not in (kr, km)]
    for k in sorted(others):
        print(f'   {k[5:]:22s} AUC={auc_fast(g(k)):.4f}')
    print(f'   {"rf_d28":22s} AUC={auc_fast(pr):.4f}')
    print(f'   {"metamamba_d28":22s} AUC={auc_fast(pm):.4f}')
    fr = (rank(pr) + rank(pm)) / 2
    d, lo, hi, pn = boot_delta(fr, pr)
    print(f'   {"FUSE rank 50/50":22s} AUC={auc_fast(fr):.4f}  Δ vs RF={d:+.4f} [{lo:+.4f},{hi:+.4f}] P={pn:.3f}'
          f'  Δ vs MM={boot_delta(fr, pm)[0]:+.4f}')
    best = None
    for w in np.arange(0, 1.01, 0.05):
        p = w * rank(pr) + (1 - w) * rank(pm)
        a = auc_fast(p)
        if best is None or a > best[1]:
            best = (w, a)
    print(f'   最优权重扫描: w_RF={best[0]:.2f} → AUC={best[1]:.4f}  (注意: 同数据选权重=乐观上界)')

print()
print('=' * 90)
print('【4】选择性预测（学生级 top-k 精确率 / 提升倍数）')
print('=' * 90)
base = y.mean()
print(f'基准率（挂科率）= {base:.4f}')
for name, p in [('RF-28d', rf), ('MetaMamba-28d', mm),
                ('FUSE rank(RF,MM,LSTM,Att,HMM)', (rank(rf) + rank(mm) + rank(lg) + rank(at) + rank(hg)) / 5)]:
    print(f'\n{name}:')
    for k, n, prec, lift, rec in selective(p):
        print(f'   top{k*100:>4.0f}%  n={n:3d}  精确率={prec:.3f}  提升={lift:.1f}x  召回={rec:.3f}')

print()
print('=' * 90)
print('【5】交叉拟合 Platt 校准（5 折 OOF，无泄漏）')
print('=' * 90)
for name, p in [('RF-28d', rf), ('MetaMamba-28d', mm),
                ('FUSE rank 全5族', (rank(rf) + rank(mm) + rank(lg) + rank(at) + rank(hg)) / 5)]:
    pc, e, b = cv_platt(p)
    print(f'{name:24s} 校准前 ECE={ece(p):.4f} → 校准后 ECE={e:.4f}  Brier={b:.4f}  '
          f'AUC(不变)={auc_fast(p):.4f}')

print()
print('=' * 90)
print('【6】MetaMamba vs HMM（同协议，修正标度后复核）')
print('=' * 90)
for name, a_p, b_p in [('身份臂 28d', g('id__metamamba_d28'), g('id__hmm_gmm_d28')),
                       ('身份臂 7d ', g('id__metamamba_d7'), g('id__hmm_cat_d7')),
                       ('去混淆 28d', g('deconf__metamamba_d28'), g('deconf__hmm_gmm_d28'))]:
    d, lo, hi, pn = boot_delta(a_p, b_p)
    print(f'{name}: MM={auc_fast(a_p):.4f} vs HMM={auc_fast(b_p):.4f} Δ={d:+.4f} [{lo:+.4f},{hi:+.4f}] P(Δ<=0)={pn:.4f}')

print('\n[fusion_v2] DONE')
