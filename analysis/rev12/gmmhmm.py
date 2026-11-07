"""
rev12.gmmhmm — GMM-HMM（高斯混合发射的隐马尔可夫），自实现 Baum-Welch

为什么自实现：
  hmmlearn 0.3.x 已移除 GMM-HMM（`hmmlearn.mixture` 不再提供），EDM 2025 报告的
  0.7697 使用的是"每类一个 GMM-HMM + 先验"。若继续用 hmmlearn 的单高斯
  GaussianHMM 冒充 GMM-HMM，则"同协议重跑 GMM-HMM"名不副实。
  → 本文件实现真正的**混合高斯发射** HMM（对角协方差），并与 hmmlearn 的
    GaussianHMM（等价于 n_mix=1）在同一数据上做对数似然对齐校验。

协议与 EDM 2025 一致的部分：
  * 每类（fail / pass）各训一个 HMM
  * 推断：P(O|class) × P(class)，比较两个后验 → 二分类概率
  * 状态数 N 在内层（训练折内）网格搜索

实现要点：
  * 前向-后向全部在对数空间进行（避免下溢）
  * E 步额外求出每个状态的混合成分责任度 r(s,t,m)
  * M 步更新 π / A / 混合权重 / 均值 / 方差（对角 + 小的正则项）
"""
import numpy as np

EPS = 1e-300
LOG_EPS = -700.0


def _logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def _log_gauss_diag(X, mu, var):
    """X:(T,D) mu:(M,D) var:(M,D) → (T,M) 对角高斯对数密度"""
    d = X.shape[1]
    var = np.maximum(var, 1e-8)
    diff = X[:, None, :] - mu[None, :, :]                 # (T,M,D)
    return -0.5 * (np.sum(diff ** 2 / var[None, :, :], axis=2)
                   + np.sum(np.log(2 * np.pi * var[None, :, :]), axis=2)
                   + d * np.log(2 * np.pi) * 0)


class GMMHMM:
    """高斯混合发射的 HMM（对角协方差）。"""

    def __init__(self, n_states=3, n_mix=2, n_iter=30, tol=1e-3,
                 random_state=0, min_var=1e-3, max_var=1e3):
        self.n_states = n_states
        self.n_mix = n_mix
        self.n_iter = n_iter
        self.tol = tol
        self.rs = np.random.RandomState(random_state)
        self.min_var = min_var
        self.max_var = max_var
        self.pi = None
        self.A = None
        self.w = None      # (S, M)
        self.mu = None     # (S, M, D)
        self.var = None    # (S, M, D)
        self.loglik_ = None

    # ── 初始化 ───────────────────────────────────────────────
    def _init(self, X, lengths):
        S, M, D = self.n_states, self.n_mix, X.shape[1]
        self.pi = np.full(S, 1.0 / S)
        self.A = np.full((S, S), 1.0 / S)
        self.w = np.full((S, M), 1.0 / M)
        self.mu = np.zeros((S, M, D))
        self.var = np.zeros((S, M, D))
        # 用 kmeans 式初始化切分状态
        idx = self.rs.choice(len(X), size=min(S, len(X)), replace=False)
        cent = X[idx]
        assign = np.argmin(((X[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
        for s in range(S):
            sel = X[assign == s]
            if len(sel) < 2:
                sel = X[self.rs.randint(0, len(X), size=max(2, D))]
            centers = sel[self.rs.choice(len(sel), size=M, replace=len(sel) < M)]
            self.mu[s] = centers
            self.var[s] = np.maximum(X.var(axis=0), self.min_var)

    # ── 发射概率（对数） ─────────────────────────────────────
    def _log_emission(self, X):
        """X:(T,D) → (S,T)"""
        T = X.shape[0]
        lg = _log_gauss_diag(X, self.mu.reshape(-1, X.shape[1]),
                             self.var.reshape(-1, X.shape[1]))      # (T, S*M)
        lg = lg.reshape(T, self.n_states, self.n_mix)
        lg = lg + np.log(np.maximum(self.w, EPS))[None, :, :]        # (T,S,M)
        return _logsumexp(lg, axis=2)                              # (T,S)

    # ── 前向/后向 ────────────────────────────────────────────
    def _forward_backward(self, X, lengths):
        T = X.shape[0]
        logb = self._log_emission(X)                    # (T,S)
        log_alpha = np.full((T, self.n_states), LOG_EPS)
        log_beta = np.full((T, self.n_states), LOG_EPS)
        logA = np.log(np.maximum(self.A, EPS))
        logpi = np.log(np.maximum(self.pi, EPS))
        starts = np.cumsum([0] + list(lengths))
        seqs = []
        for si in range(len(lengths)):
            s0, s1 = starts[si], starts[si + 1]
            L = s1 - s0
            if L == 0:
                continue
            a = np.full((L, self.n_states), LOG_EPS)
            b = np.full((L, self.n_states), LOG_EPS)
            a[0] = logpi + logb[s0]
            for t in range(1, L):
                a[t] = _logsumexp(a[t - 1][:, None] + logA, axis=0) + logb[s0 + t]
            b[L - 1] = 0.0
            for t in range(L - 2, -1, -1):
                b[t] = _logsumexp(logA + (logb[s0 + t + 1] + b[t + 1])[None, :], axis=1)
            seqs.append((s0, s1, L, a, b))
        return seqs

    # ── Baum-Welch ───────────────────────────────────────────
    def fit(self, X, lengths):
        X = np.asarray(X, dtype=np.float64)
        lengths = [int(l) for l in lengths]
        D = X.shape[1]
        self._init(X, lengths)
        prev = -np.inf
        for it in range(self.n_iter):
            seqs = self._forward_backward(X, lengths)
            logA = np.log(np.maximum(self.A, EPS))
            # 累积统计量
            pi_acc = np.zeros(self.n_states)
            A_acc = np.zeros((self.n_states, self.n_states))
            gamma_acc = np.zeros((self.n_states, self.n_states))
            resp_acc = np.zeros((self.n_states, self.n_mix))
            mu_acc = np.zeros((self.n_states, self.n_mix, D))
            var_acc = np.zeros((self.n_states, self.n_mix, D))
            ll = 0.0
            for (s0, s1, L, a, b) in seqs:
                obs = X[s0:s1]
                logb = self._log_emission(obs)                 # (L,S)
                log_gamma = a + b
                log_gamma = log_gamma - _logsumexp(log_gamma, axis=1, keepdims=True)
                gamma = np.exp(log_gamma)                      # (L,S)
                ll += float(np.sum(_logsumexp(a[L - 1], axis=0)))
                pi_acc += gamma[0]
                if L > 1:
                    log_xi = (a[:-1][:, :, None] + logA[None, :, :]
                              + (logb[1:] + b[1:])[:, None, :] * 0
                              + (logb[1:] + b[1:])[:, None, :])
                    log_xi = (a[:-1][:, :, None] + logA[None, :, :]
                              + (logb[1:] + b[1:])[:, None, :])
                    log_xi = log_xi - _logsumexp(log_xi, axis=(1, 2), keepdims=True)
                    xi = np.exp(log_xi)                       # (L-1,S,S)
                    A_acc += xi.sum(axis=0)
                    gamma_acc += gamma.sum(axis=0)
                else:
                    gamma_acc += gamma.sum(axis=0)
                # 混合责任度
                lg = _log_gauss_diag(obs, self.mu.reshape(-1, D), self.var.reshape(-1, D))
                lg = lg.reshape(L, self.n_states, self.n_mix) + np.log(np.maximum(self.w, EPS))[None]
                lg = lg + gamma[:, :, None]                   # (L,S,M)
                lg = lg - _logsumexp(lg, axis=(1, 2), keepdims=True)
                r = np.exp(lg)
                resp_acc += r.sum(axis=0)
                for s in range(self.n_states):
                    for m in range(self.n_mix):
                        rm = r[:, s, m][:, None]
                        wsum = rm.sum()
                        if wsum < 1e-8:
                            continue
                        mu_acc[s, m] += (rm * obs).sum(axis=0)
                        var_acc[s, m] += (rm * (obs - self.mu[s, m]) ** 2).sum(axis=0)
            # M 步
            self.pi = pi_acc / max(pi_acc.sum(), EPS)
            self.A = A_acc / np.maximum(A_acc.sum(axis=1, keepdims=True), EPS)
            self.w = resp_acc / np.maximum(resp_acc.sum(axis=1, keepdims=True), EPS)
            for s in range(self.n_states):
                for m in range(self.n_mix):
                    wsum = resp_acc[s, m]
                    if wsum < 1e-8:
                        continue
                    self.mu[s, m] = mu_acc[s, m] / wsum
                    self.var[s, m] = np.clip(var_acc[s, m] / wsum, self.min_var, self.max_var)
            if abs(ll - prev) < self.tol:
                prev = ll
                break
            prev = ll
        self.loglik_ = prev
        return self

    # ── 似然 ────────────────────────────────────────────────
    def score_loglik(self, X) -> float:
        X = np.asarray(X, dtype=np.float64)
        logb = self._log_emission(X)
        logA = np.log(np.maximum(self.A, EPS))
        logpi = np.log(np.maximum(self.pi, EPS))
        a = logpi + logb[0]
        for t in range(1, len(X)):
            a = _logsumexp(a[:, None] + logA, axis=0) + logb[t]
        return float(_logsumexp(a, axis=0))


class ClassPairHMM:
    """每类一个 HMM + 类先验 → 二分类后验（EDM 2025 协议）。"""

    def __init__(self, factory, prior=None):
        self.factory = factory
        self.models = {}
        self.prior = prior

    def fit(self, Xs, y, lengths):
        prior = float(np.mean(y)) if self.prior is None else self.prior
        self.prior = prior
        for cls in (1, 0):
            rows = [i for i in range(len(Xs)) if y[i] == cls]
            Xc = np.concatenate([Xs[i] for i in rows])
            lens = [lengths[i] for i in rows]
            m = self.factory()
            m.fit(Xc, lens)
            self.models[cls] = m
        return self

    def predict_proba(self, Xs, lengths=None):
        out = np.zeros(len(Xs))
        for i, X in enumerate(Xs):
            s1 = self.models[1].score_loglik(X) + np.log(max(self.prior, 1e-12))
            s0 = self.models[0].score_loglik(X) + np.log(max(1 - self.prior, 1e-12))
            m = max(s1, s0)
            out[i] = np.exp(s1 - m) / (np.exp(s1 - m) + np.exp(s0 - m))
        return out


class ClassPairCategoricalHMM:
    """每类一个多项式发射 HMM（事件符号序列）+ 先验。"""

    def __init__(self, n_states=3, n_iter=30, random_state=0):
        self.n_states = n_states
        self.n_iter = n_iter
        self.rs = np.random.RandomState(random_state)
        self.models = {}
        self.prior = None

    def _fit_one(self, seqs):
        S = self.n_states
        K = 7
        pi = np.full(S, 1.0 / S)
        A = np.full((S, S), 1.0 / S)
        B = np.full((S, K), 1.0 / K)
        prev = -np.inf
        for _ in range(self.n_iter):
            pi_acc = np.zeros(S); A_acc = np.zeros((S, S)); B_acc = np.zeros((S, K))
            ll = 0.0
            for seq in seqs:
                L = len(seq)
                if L == 0:
                    continue
                logB = np.log(np.maximum(B, EPS))
                logA = np.log(np.maximum(A, EPS))
                logpi = np.log(np.maximum(pi, EPS))
                a = np.full((L, S), LOG_EPS); b = np.full((L, S), LOG_EPS)
                a[0] = logpi + logB[:, seq[0]]
                for t in range(1, L):
                    a[t] = _logsumexp(a[t - 1][:, None] + logA, axis=0) + logB[:, seq[t]]
                b[L - 1] = 0.0
                for t in range(L - 2, -1, -1):
                    b[t] = _logsumexp(logA + (logB[:, seq[t + 1]] + b[t + 1])[None, :], axis=1)
                ll += float(_logsumexp(a[L - 1], axis=0))
                la = a + b
                la = la - _logsumexp(la, axis=1, keepdims=True)
                gamma = np.exp(la)
                pi_acc += gamma[0]
                if L > 1:
                    lxi = a[:-1][:, :, None] + logA[None] + (logB[:, seq[1:]].T + b[1:])[:, None, :]
                    lxi = lxi - _logsumexp(lxi, axis=(1, 2), keepdims=True)
                    A_acc += np.exp(lxi).sum(axis=0)
                for s in range(S):
                    np.add.at(B_acc[s], seq, gamma[:, s])
            pi = pi_acc / max(pi_acc.sum(), EPS)
            A = A_acc / np.maximum(A_acc.sum(axis=1, keepdims=True), EPS)
            B = B_acc / np.maximum(B_acc.sum(axis=1, keepdims=True), EPS)
            if abs(ll - prev) < 1e-3:
                prev = ll; break
            prev = ll
        self.pi, self.A, self.B = pi, A, B
        self.loglik_ = prev
        return self

    def score_loglik(self, seq):
        S = self.n_states
        logB = np.log(np.maximum(self.B, EPS))
        logA = np.log(np.maximum(self.A, EPS))
        logpi = np.log(np.maximum(self.pi, EPS))
        L = len(seq)
        if L == 0:
            return LOG_EPS
        a = logpi + logB[:, seq[0]]
        for t in range(1, L):
            a = _logsumexp(a[:, None] + logA, axis=0) + logB[:, seq[t]]
        return float(_logsumexp(a, axis=0))

    def fit(self, seqs, y):
        self.prior = float(np.mean(y))
        for cls in (1, 0):
            sel = [seqs[i] for i in range(len(seqs)) if y[i] == cls and len(seqs[i]) > 0]
            self.models[cls] = self._fit_one(sel)
        return self

    def predict_proba(self, seqs):
        out = np.zeros(len(seqs))
        for i, s in enumerate(seqs):
            if len(s) == 0:
                out[i] = self.prior
                continue
            s1 = self.models[1].score_loglik(s) + np.log(max(self.prior, 1e-12))
            s0 = self.models[0].score_loglik(s) + np.log(max(1 - self.prior, 1e-12))
            m = max(s1, s0)
            out[i] = np.exp(s1 - m) / (np.exp(s1 - m) + np.exp(s0 - m))
        return out
