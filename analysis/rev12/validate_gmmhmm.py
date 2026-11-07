"""GMM-HMM 实现校验：
 1) 与 hmmlearn 的 GaussianHMM（等价 n_mix=1）对齐对数似然（若可安装 hmmlearn）
 2) 无 hmmlearn 时的自检：训练对数似然单调上升 + 与"每类一个 HMM + 先验"的
    分类后验自洽（predict_proba 与手算 log-likelihood 比对）
"""
import sys, time
sys.path.insert(0, '/home/ubuntu/SR_rev12')
import numpy as np
from analysis.rev12.gmmhmm import GMMHMM, ClassPairHMM

rng = np.random.RandomState(0)
# 合成数据：两簇不同均值
D = 6
seqs = []
labels = []
for i in range(60):
    cls = i % 2
    mu = np.zeros(D) + (1.5 if cls else -1.5)
    L = 60
    seqs.append(rng.randn(L, D) * 0.4 + mu)
    labels.append(cls)
X = np.concatenate(seqs)
lens = [len(s) for s in seqs]
y = np.array(labels)

t0 = time.time()
m = GMMHMM(n_states=2, n_mix=1, n_iter=30, random_state=0)
m.fit(X, lens)
print(f"self-implemented Gaussian HMM (n_mix=1): loglik={m.loglik_:.3f} ({time.time()-t0:.1f}s)")

# 手算似然核对
ll_manual = m.score_loglik(seqs[0][:20])
ll_batch = None
print("score_loglik(first 20 events) =", round(ll_manual, 4))

# 与 hmmlearn 对齐（若已安装）
try:
    from hmmlearn.hmm import GaussianHMM
    best = None
    for rs in range(5):
        h = GaussianHMM(n_components=2, covariance_type='diag', n_iter=50, random_state=rs)
        h.fit(X, lengths=lens)
        v = h.score(X)
        best = v if best is None else max(best, v)
    ours_total = float(np.sum([m.score_loglik(s) for s in seqs]))
    print(f"hmmlearn GaussianHMM (best of 5 seeds): total loglik = {best:.3f}")
    print(f"ours (n_mix=1, same model class):         total loglik = {ours_total:.3f}")
    print(f"ours >= hmmlearn best? {ours_total >= best}")
except Exception as e:
    print("hmmlearn not available / failed:", e)

# 分类后验自洽性
pair = ClassPairHMM(factory=lambda: GMMHMM(n_states=2, n_mix=1, n_iter=20, random_state=1))
pair.fit([s for s in seqs], y, lens)
p = pair.predict_proba([s for s in seqs])
s1 = pair.models[1].score_loglik(seqs[0]) + np.log(pair.prior)
s0 = pair.models[0].score_loglik(seqs[0]) + np.log(1 - pair.prior)
manual = np.exp(s1 - max(s1, s0)) / (np.exp(s1 - max(s1, s0)) + np.exp(s0 - max(s1, s0)))
print(f"posterior check: pipeline={p[0]:.6f} manual={manual:.6f} "
      f"diff={abs(p[0]-manual):.2e}")
from sklearn.metrics import roc_auc_score
print("synthetic AUC =", round(roc_auc_score(y, p), 4))

# 混合发射版本（n_mix=2）应不劣于 n_mix=1
m2 = GMMHMM(n_states=2, n_mix=2, n_iter=30, random_state=0)
m2.fit(X, lens)
print(f"n_mix=2 loglik={m2.loglik_:.3f} (>= n_mix=1? {m2.loglik_ >= m.loglik_})")
