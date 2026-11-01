"""实验 C：头对头对比 —— 同一批 473 学生上，论文模型 vs 去混淆 RF，配对显著性检验"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

B = 'oof'
y = np.load(f'{B}/meta_mamba_7d/labels.npy').ravel()
task = np.load(f'{B}/meta_mamba_7d/task_ids.npy').ravel()
n = len(y)
rng = np.random.default_rng(99)

# part 6 (0-indexed) = 唯一正负样本各>=5 的可判别 part
valid = [t for t in np.unique(task) if y[task == t].sum() >= 5 and (1 - y[task == t]).sum() >= 5]


def load_oof(p):
    return np.load(p).ravel()


CAND = {
    'meta_mamba (11d, 论文主模型)': load_oof(f'{B}/meta_mamba/oof_probs.npy'),
    'meta_mamba_7d (论文7d)': load_oof(f'{B}/meta_mamba_7d/oof_probs.npy'),
    'rf7 (论文基线)': load_oof(f'{B}/rf7/oof_probs.npy'),
    'V0_raw RF (49d, 本次对照)': load_oof('expB_oof_V0_raw.npy'),
    'V1_deconf RF (49d)': load_oof('expB_oof_V1_deconf.npy'),
    'V3_deconf+part1h RF (56d)': load_oof('expB_oof_V3_deconf+part1h.npy'),
}

print("=" * 108)
print("C1) 同一批 473 学生上的头对头对比")
print("=" * 108)
print(f"  可判别 part（正负各>=5）: {valid}")
print(f"  {'模型':32s} {'AUC':>8s} {'F1macro':>8s} {'grp_bal':>9s} " +
      " ".join(f"{'part'+str(t):>9s}" for t in valid))
res = {}
for name, p in CAND.items():
    yp = (p >= 0.5).astype(int)
    aucs, ws = [], []
    for t in np.unique(task):
        m = task == t
        if len(set(y[m])) > 1:
            aucs.append(roc_auc_score(y[m], p[m]))
            ws.append(m.sum())
    gb = float(np.average(aucs, weights=ws))
    pa = {t: roc_auc_score(y[task == t], p[task == t]) for t in valid}
    res[name] = dict(auc=roc_auc_score(y, p), grp_bal=gb,
                     f1m=f1_score(y, yp, average='macro', zero_division=0),
                     acc=accuracy_score(y, yp), part_auc=pa, p=p)
    print(f"  {name:32s} {res[name]['auc']:8.4f} {res[name]['f1m']:8.4f} {gb:9.4f} " +
          " ".join(f"{pa[t]:9.4f}" for t in valid))

print()
print("=" * 108)
print("C2) 配对 bootstrap：V1_deconf RF 是否显著优于论文主模型 meta_mamba？")
print("=" * 108)


def paired(pA, pB, mask_fn, label, nboot=3000):
    diffs = []
    for _ in range(nboot):
        if mask_fn is None:
            b = rng.integers(0, n, n)
            yy = y[b]
            if len(set(yy)) < 2:
                continue
            diffs.append(roc_auc_score(yy, pA[b]) - roc_auc_score(yy, pB[b]))
        else:
            idx = mask_fn()
            vals = []
            ok = True
            for _ in range(300):
                b = rng.choice(idx, size=len(idx), replace=True)
                yy = y[b]
                if len(set(yy)) < 2:
                    ok = False
                    break
                vals.append(roc_auc_score(yy, pA[b]) - roc_auc_score(yy, pB[b]))
            if not ok:
                continue
            diffs.append(np.mean(vals))
    d = np.array(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"  {label:52s} Δ={d.mean():+.4f} 95%CI [{lo:+.4f},{hi:+.4f}] P(Δ<=0)={(d<=0).mean()*100:5.2f}%")


v1 = CAND['V1_deconf RF (49d)']
mm = CAND['meta_mamba (11d, 论文主模型)']
mm7 = CAND['meta_mamba_7d (论文7d)']
rf7 = CAND['rf7 (论文基线)']
paired(v1, mm, None, "总体AUC: V1_deconf RF − meta_mamba(11d)")
paired(v1, mm7, None, "总体AUC: V1_deconf RF − meta_mamba_7d")
paired(v1, rf7, None, "总体AUC: V1_deconf RF − rf7(论文基线)")
for t in valid:
    paired(v1, mm, lambda t=t: np.where(task == t)[0], f"part{t}内AUC: V1_deconf RF − meta_mamba(11d)")
    paired(v1, rf7, lambda t=t: np.where(task == t)[0], f"part{t}内AUC: V1_deconf RF − rf7")

print()
print("=" * 108)
print("C3) 结论表")
print("=" * 108)
out = {k: {kk: vv for kk, vv in v.items() if kk != 'p'} for k, v in res.items()}
out['valid_parts'] = [int(t) for t in valid]
json.dump(out, open('experiment_C_results.json', 'w'), indent=2, ensure_ascii=False)
print("  → experiment_C_results.json")
