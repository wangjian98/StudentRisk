"""实验 H3：lead-time（仅用前 k 个 stage 的行为预测最终挂科）——修正版"""
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

D = np.load('results/F_stage_probe/stage_matrix.npz', allow_pickle=True)
FEAT, SID, PID = D['feat'], D['sid'], D['pid']
Y_STU, STUDENTS = D['y'], D['students']
EVN, PARTS = 7, list(range(1, 8))
sidx = {int(s): i for i, s in enumerate(STUDENTS)}
per = defaultdict(dict)
for j in range(len(SID)):
    per[int(SID[j])][int(PID[j])] = FEAT[j].astype(float)
STU = sorted(per)
info = {}
for s in STU:
    x49 = np.zeros(49)
    for p, x in per[s].items():
        x49[(p - 1) * EVN:(p - 1) * EVN + EVN] = x[:EVN]
    info[s] = dict(main=max(per[s], key=lambda p: per[s][p][7]), x49=x49,
                   label=int(Y_STU[sidx[s]]))
n = len(STU)
y = np.array([info[s]['label'] for s in STU])
main = np.array([info[s]['main'] for s in STU])

st_s, st_p, st_x, st_i = [], [], [], []
for s in STU:
    for j, k in enumerate(sorted(per[s])):
        st_s.append(s); st_p.append(k); st_x.append(per[s][k]); st_i.append(j)
st_s = np.array(st_s); st_p = np.array(st_p); st_x = np.array(st_x); st_i = np.array(st_i)
s2row = {s: i for i, s in enumerate(STU)}
st_row = np.array([s2row[s] for s in st_s])


def deconf(main_p, X, tr_rows, target_rows):
    out = np.zeros((len(target_rows), X.shape[1]))
    for p in np.unique(main_p):
        grp = np.where(main_p == p)[0]
        tg = np.intersect1d(grp, tr_rows)
        if len(tg) == 0:
            continue
        Xg = X[tg]
        med = np.median(Xg, 0); mad = np.maximum(np.median(np.abs(Xg - med), 0), 1e-3)
        pos = {r: i for i, r in enumerate(tg)}
        for i, r in enumerate(target_rows):
            if main_p[r] != p:
                continue
            if r in pos and len(tg) > 1:
                X2 = np.delete(Xg, pos[r], axis=0)
                m2 = np.median(X2, 0); d2 = np.maximum(np.median(np.abs(X2 - m2), 0), 1e-3)
                out[i] = (X[r] - m2) / d2
            else:
                out[i] = (X[r] - med) / mad
    return np.clip(out, -5, 5)


folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros((n, 1)), y))
print(f'{"k":>2s} {"覆盖人数":>8s} {"AUC":>8s} {"judgeable":>10s} {"prior":>8s}')
for kmax in [1, 2, 3]:
    Xk = np.zeros((n, 49))
    cnt = defaultdict(int)
    for j in range(len(st_y := st_x)):
        if st_i[j] < kmax:
            Xk[st_row[j], (st_p[j] - 1) * EVN:(st_p[j] - 1) * EVN + EVN] += st_x[j][:EVN]
            cnt[st_s[j]] += 1
    used = np.array([cnt.get(s, 0) > 0 for s in STU])
    oof = np.zeros(n); pri = np.zeros(n)
    for tr, te in folds:
        zr = deconf(main, Xk, tr, np.arange(n))
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features='sqrt',
                                   class_weight='balanced', random_state=42, n_jobs=-1)
        m.fit(zr[tr], y[tr]); oof[te] = m.predict_proba(zr[te])[:, 1]
        for p in np.unique(main):
            mtr = tr[main[tr] == p]; mte = te[main[te] == p]
            if len(mtr):
                pri[mte] = y[mtr].mean()
    yy = y[used]; oo = oof[used]
    jud = []
    for p in np.unique(main[used]):
        mm = (main[used] == p)
        if len(set(yy[mm])) > 1 and (yy[mm] == 0).sum() >= 10:
            jud.append(roc_auc_score(yy[mm], oo[mm]))
    print(f'{kmax:>2d} {used.sum():>8d} {roc_auc_score(yy,oo):>8.4f} '
          f'{(np.mean(jud) if jud else float("nan")):>10.4f} {roc_auc_score(yy,pri[used]):>8.4f}')
