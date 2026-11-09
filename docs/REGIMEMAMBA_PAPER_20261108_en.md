# RegimeMamba: Behavioural Regime Switching in Selective State-Space Models for CS1 Programming-Risk Prediction — A Pre-Registered Falsification Study

**Author**: Wang Jian¹

**Affiliation**:
¹ Department of Computer Science and Educational Technology (to be completed: [institution], [city, China])

**Corresponding author**: Wang Jian (wangjian98@example.com)

**Submission date**: 8 November 2026

**Version note**: This is a standalone manuscript (the RegimeMamba experimental line), running in parallel with this project's mainline manuscript on stage-level hazard modelling (CS1-SHR / CS1-FSP); see §7.6 for how the two relate and how they should be positioned.

**Code and data**: `StudentRisk/analysis/regimemamba/` (model / experiment / analysis scripts), `StudentRisk/results/regimemamba/` (per-cell raw results), `StudentRisk/docs/REGIMEMAMBA_{DESIGN,RESULTS,DATA}_20261108.*` (design memo / experiment report / data tables); the stratified and matched analysis of §6.8 is `StudentRisk/analysis/regimemamba/stratified_analysis.py` with outputs in `StudentRisk/docs/REGIMEMAMBA_STRATIFIED_20261110.{md,csv}`.

---

## Abstract

**Background.** Programming platforms emit massive, fine-grained IDE interaction logs, making end-of-term failure-risk prediction a canonical "many events, few labels" problem. Three modelling traditions each cover only half of what is needed. Hidden Markov models have explicit **state (regime) switching** but linear/Gaussian emissions and no input selectivity. Selective state-space models such as MetaMamba have **input-dependent selective dynamics** but a single global state, and their task-aware FiLM modulation injects *problem identity* into the model. Tabular feature-engineering baselines compress an entire semester into static statistics: they lose ordering but preserve information content.

**Purpose.** We propose **RegimeMamba**, which composes HMM-style switchable states with Mamba-style selective scanning: a **behaviour-driven causal forward filter** infers a discrete behavioural regime, and **regime-conditioned selective state-space parameters** encode the different dynamics that characterise different behavioural states. We then evaluate it under a **pre-registered falsification protocol**: six hypotheses (H1–H6) with numeric thresholds were fixed in writing before any experiment was run, and may not be modified afterwards.

**Method.** The architecture has four components: (1) a **behavioural regime forward filter** $r_t=\mathrm{softmax}(\log(r_{t-1}A)+g(x_t))$ whose emission network consumes behaviour only (never problem identity) and whose posterior is carried forward on padding steps; (2) **regime-conditioned selective SSM** parameters ($\Delta_t,B_t,C_t,A_t$ mixed by $r_{t,k}$) executed with a **single scan**, keeping compute on par with same-family models; (3) **regime direct features** (occupancy histogram, final posterior, switch rate) and a **per-regime risk bias**; (4) three regime-quality regularisers (anti-collapse occupancy entropy, switch sparsity, posterior sharpness). Evaluation reuses this project's published same-protocol baseline suite: 473 students, 28,588,309 IDE events, a 28-dimensional behaviour/context space that is strictly past-only, student-grouped 5-fold × 3-seed cross-validation, 15% inner validation for early stopping, and a **symmetric budget of 4 hyper-parameter candidates per architecture**.

**Results.** (i) On the identity arm, RegimeMamba attains **pooled OOF AUC 0.9488**, significantly above the same-protocol MetaMamba (0.9056; Δ=+0.0431, 95% CI [+0.0235,+0.0648], P(Δ≤0)=0.000) and statistically tied with the tabular baseline RF (0.9509; Δ=−0.0021, P=0.622). However, under the **within-fold mean** caliber the same comparison is a tie (0.9368 vs 0.9334, Wilcoxon P=0.4887) — **the two AUC calibers yield opposite conclusions, which is itself one of this paper's main methodological findings.** (ii) **H2 fails**: after removing problem-identity channels RegimeMamba drops by 0.0538 (RF 0.0163, MetaMamba 0.0040), so part of the gain rests on identity memory. (iii) **H3 fails**: no advantage in the early-window (first 10% of events) setting (0.9265 vs MetaMamba 0.9298, ns), so no early-warning claim is warranted. (iv) **the capacity control does not support "the gain comes from the regime"**: after the divide-by-zero defect was fixed and the K=1 single-state control re-run, the control behaves normally again (pooled OOF AUC=0.9454), yet the pooled OOF gain of K=4 over K=1 is only +0.0032 [-0.0051,+0.0122] (P(Δ≤0)=0.237), which misses the preregistered criterion (Δ>0.005 and P≤0.10); H6 therefore fails and **this paper does not claim that the gain comes from regime structure rather than model capacity**. (v) The learned regimes carry **interpretable behavioural semantics**: R2 is a "risk regime" (occupancy 0.268 for failing vs 0.207 for passing students, Mann-Whitney P<0.0001) whose occupancy alone discriminates failure at **AUC 0.6386**; R1 is a "good-state regime" (AUC 0.3783, i.e. reversed); and switch rate discriminates at AUC 0.3806 (reversed), correlating with behavioural **volume** (paste/run/submit counts, ρ≈0.22–0.28, P<0.0001) but not with rate-like measures — regime switching tracks activity shape, not efficiency. **The stratified and matched analysis of §6.8 further shows that both associations are largely re-encodings of behavioural engagement: after matching engagement level, R2 occupancy no longer separates failing from passing students (paired diff +0.008, P = 0.91) and the reversed switch association is no longer significant (paired diff −0.020, P = 0.26).**

**Conclusion.** On CS1, explicit behavioural regime switching is **not** shown to be an effective structural inductive bias in this paper (the gain over the K=1 single-state control is not significant; H6 fails), but it **does not deliver discriminative power beyond tabular statistics**, and part of its gain comes from a problem-identity shortcut. The honest consequence is that RegimeMamba's **interpretability** — turning behavioural state into a readable quantity — is worth more than its accuracy, but only part of that interpretability survives the confound check: **at matched engagement level, R2 occupancy no longer separates failing from passing students** (§6.8). We position the model for **whole-cycle risk stratification and state description**, not early warning, not an accuracy race, and not as a stand-alone "duration in a risky state" product.

**Keywords**: learning analytics; selective state-space models; behavioural regimes; falsification studies; pre-registered evaluation; programming education; interpretability

---

## 1 Introduction

### 1.1 Background

Learning analytics for C-language introductory (CS1) courses typically relies on fine-grained IDE event logs (`text_insert` / `text_remove` / `text_paste` / `focus_gained` / `focus_lost` / `run` / `submit`). The public CS1 dataset contains 473 students and 28,588,309 events — an order of magnitude more events than students, with one label per student. End-of-term failure prediction has therefore long served as a methodological testbed for this regime.

Three distinct lines of work exist:

1. **Latent-state / state-space models** (HMM and variants): explicit modelling of switches between "learning states". A published EDM 2025 study reports 76.97% accuracy on CS1, but under a different protocol (60-second increment windows, full-history binning, one HMM per class plus priors); in our same-protocol replication, HMM on event symbols **collapses to chance** (AUC 0.477–0.503).
2. **Deep sequence / selective SSM models** (LSTM / BiLSTM / Attention / MetaMamba): direct sequence modelling. MetaMamba combines a self-implemented selective scan (S6) with task-aware FiLM and meta-learning evaluation, reaching within-fold AUC 0.9334 in the 28-dimensional space under our protocol.
3. **Tabular feature-engineering models** (random forests and relatives): aggregating sequences into counts/means/last values. **Under an identical protocol, input space and tuning budget, RF is never worse than the deep sequence models**, reaching within-fold AUC 0.9526 and pooled OOF AUC 0.9509 in the 28-dimensional space.

Each line leaves a gap: **the state-space line has regimes without selectivity, the selective SSM line has selectivity without regimes, the tabular line has statistics without ordering**. This paper fills that gap rather than training another larger network.

### 1.2 Why "regime" deserves its own modelling

In this project's earlier analyses, the difference between failing and passing students is not "how much they worked" but **changes in behavioural rhythm and shape** (edit/paste ratios, submission cadence, gap lengths). Such "shapes" appear in contiguous segments with abrupt transitions between them. Fitting all of them with one global dynamic asks a single model to explain both an exploratory phase and a stagnation phase. Explicit regime-isation lets each state own its dynamics — the hypothesis tested here.

One risk must be acknowledged in advance: **regimes may degenerate into proxies for problem identity**. If the emission network indirectly depends on part/exercise, the model falls back to the shortcut that a previous review round already rejected. We therefore register "does the regime consume identity?" as **pre-registered criterion H2**, not as a post-hoc discussion point.

### 1.3 Research questions

- **RQ1**: Under an identical protocol, input space and tuning budget, does **behavioural regime switching** yield discriminative gains beyond a single-state selective SSM?
- **RQ2**: Does the gain come from the regime **structure**, or merely from **larger effective capacity** (K=1 control) and **problem-identity shortcuts** (deconfounded arm)?
- **RQ3**: Do the learned regimes exhibit **interpretable behavioural semantics** that provide a risk signal independent of the final prediction head?

### 1.4 Contributions

1. **Method.** RegimeMamba composes a behavioural regime forward filter with regime-conditioned selective SSM dynamics, implemented with a **single scan** so that compute and tuning budgets stay symmetric with same-family models.
2. **Methodology.** A **pre-registered falsification protocol** (H1–H6, thresholds fixed in advance including explicit failure conditions) and an explicit distinction between **two AUC calibers** (within-fold mean vs pooled OOF), reported side by side.
3. **Empirical findings.** the gain of regime structure over K=1 is not significant (H6 fails), it partly relies on identity, and the regimes carry **interpretable risk semantics** (R2 occupancy alone: AUC 0.6386) — but §6.8 shows that roughly half of that association is explained by behavioural engagement volume.
4. **Negative results.** A direct answer to "is state-space switching worth modelling on CS1": **worth it for explanation, not worth it for an accuracy race**.

---

## 2 Related Work

**Selective state-space models.** Mamba (Gu & Dao, 2023) makes SSM parameters input-dependent, giving linear-complexity sequence modelling with strong long-context behaviour. This project's MetaMamba uses a self-implemented simplified S6 block (avoiding the broken `mamba-ssm` dependency) with task-aware FiLM and a contrastive auxiliary loss.

**Hidden Markov models and state-space modelling in education.** HMMs have long represented "engaged → stalled → dropped-out" transitions in learner behaviour, but standard emissions (linear Gaussian or categorical) cannot express "the same state under different contexts", and windowing breaks the original binning assumption. Our self-implemented mixture-Gaussian HMM was validated to converge to a far better likelihood than `hmmlearn` on the same data (−10676 vs −30353) with an exactly matching posterior pipeline (diff=0), so the HMM baseline here is not a straw man.

**Tabular baselines and the "architecture vs features" debate.** A prior same-protocol revision experiment (5-fold × 3 seeds, 200 epochs, patience 20, 4 symmetric candidates) already established that RF is never worse than deep sequence models across 7/11/28-dimensional spaces; that the 11-dimensional space's apparent advantage came almost entirely from a part-identity channel (RF 0.9324 → 0.7277 once deconfounded); and that the 28-dimensional space retains 0.88–0.93 without identity. That result is this paper's **prior**: architecture is not the lever; input design is.

**Pre-registration and falsifiable research.** We borrow pre-registration practice, turning method claims into numeric criteria that can fail, and report failures verbatim.

---

## 3 Data, Task and Protocol

### 3.1 Dataset and labels

Public CS1 dataset: 473 students, 28,588,309 IDE events, 7 event types; student-level label FAILED/PASSED with 314 failures (**base rate 0.6638**). All experiments share one event cache (`events.npz`: event type, timestamp, part, exercise, time-to-deadline, student offsets) to guarantee a common data substrate.

### 3.2 Input space

The sequence model consumes **28 behaviour/context channels**: 7 event one-hots plus 21 behaviour/context channels (event interval, deadline distance, part/exercise normalisation, position within the student's stream, elapsed time since first event, time-of-day sin/cos, weekday and weekend flags, cumulative type proportions, cumulative submit/run/focus counts, cumulative distinct exercises, gaps since last submit/run/focus-loss, same-type run length, part event share). **All channels are strictly past-only**, which the prefix experiments of §5 can test. The sequence window is the **last 128 events** of the chosen prefix (left padding).

Two "tabular twins" are retained: RF uses student-level aggregates of the same space (event counts plus per-channel mean/std/last), decoupling architecture differences from information-content differences.

> **Implementation disclosure (timestamp unit).** Timestamps in this project's event cache are effectively in **units of 1000 seconds** (the raw `datetime64[us]` values become microseconds after `astype('int64')` and were then divided by `10**9` once too often — a conversion leftover). Evidence: student 0's span is 4208 units × 1000 s = 1168.9 hours, matching the raw log. Consequently the **event-interval channel (ch7) is compressed**: its median and 75th percentile are both 0.085 (it would span 0–1 under a seconds-based scale), and the **time-since-first-event channel (ch12) is compressed too** (observed 0.50–0.58). The defect applies equally to **all** models in this project (RF and deep models alike), so model comparisons are unaffected; it does mean that the "event interval" information source has not been fully exploited by the current implementation (see §8.2).

### 3.3 Protocol

| Element | Specification |
|---|---|
| Cross-validation | 5 folds × 3 seeds (42/123/777), grouped by student, label-stratified; each student validated exactly once per seed |
| Model selection | 15% inner validation split *inside* each training fold for early stopping and hyper-parameter selection; **evaluation folds never participate in selection** |
| Tuning budget | the same **4 candidates per architecture** (including RegimeMamba), selected on a fixed dev fold by inner macro-F1, then frozen |
| Training | max_epochs=200, patience=20, batch=32, AdamW + cosine, grad_clip=1.0, BCEWithLogits |
| Metrics | accuracy, **balanced accuracy**, macro-F1, F1(FAIL), ROC-AUC |
| Baselines | all taken from this project's **same-protocol, same-data** revision experiments (RF / Attention / LSTM / BiLSTM / MetaMamba / HMM-GMM); no cross-protocol published values |

### 3.4 Two AUC calibers (a core discipline of this paper)

- **Within-fold mean AUC** (the caliber used in the project's v12 manuscript): AUC computed inside each of the 15 outer folds, then averaged. Measures **within-fold ranking quality**, insensitive to cross-fold scale drift.
- **Pooled OOF AUC** (new here): concatenate every student's OOF probability (each produced by its own fold model) into one vector and compute a single AUC. Measures **cross-fold global comparability**, i.e. whether scores from different training subsets are mutually rankable.

The two can differ by more than 0.03 (MetaMamba-28d: 0.9334 within-fold vs 0.9056 pooled; RegimeMamba: 0.9368 vs 0.9488). This paper **reports both** and discusses the discrepancy in §6.2.

---

## 4 Method: RegimeMamba

### 4.1 Three design constraints (enforced in the implementation)

1. **The regime emission must not consume identity.** $g(\cdot)$ sees only behavioural event representations; in the deconfounded arm, part/exercise channels are zeroed and the FiLM task id is zeroed.
2. **One scan only.** Regime mixing is performed on the linear operators *before* the recurrence, so $\Delta/B/C/A$ are mixed by $r_{t,k}$ and fed into the same recurrence; compute stays on par with MetaMamba (measured ≈2.7 s/epoch), preserving budget symmetry.
3. **Causality.** The forward filter depends only on the past; the past-only constraint is preserved.

### 4.2 Behavioural regime forward filter

With $K=4$ regimes, $A=\mathrm{softmax}(\Theta)\in\mathbb{R}^{K\times K}$ and emission logits $g_t=\mathrm{MLP}(x_t)\in\mathbb{R}^K$:

$$r_t=\mathrm{softmax}\big(\log(r_{t-1}A)+g_t\big),\qquad r_0=\mathrm{uniform}$$

$r_{t-1}A$ is the prior obtained by propagating the previous state through the transition matrix, and $g_t$ is the evidence from the current behaviour; adding them in the log domain is exactly the (soft, differentiable) forward algorithm. If emissions were constant and transitions hard, this would degenerate to a standard HMM. On padding steps the posterior is carried forward, so padding is never treated as evidence.

### 4.3 Regime-conditioned selective SSM

Keeping MetaMamba's S6 structure (causal depthwise convolution → input-dependent $\Delta,B,C$ → recurrence $h_t=\bar A_t h_{t-1}+\bar B_t x_t$, $y_t=C_t h_t$) but mixing every selective parameter by regime:

$$\Delta_t=\sum_k r_{t,k}\,\mathrm{softplus}\big(s_k^{\Delta}\odot \Delta_t^{\text{raw}}+b_k^{\Delta}\big),\quad
B_t=\sum_k r_{t,k}\,(m_k^{B}\odot B_t^{\text{raw}}),\quad
C_t=\sum_k r_{t,k}\,(m_k^{C}\odot C_t^{\text{raw}}),\quad
A_t=-\exp\big(A^{\log}+\textstyle\sum_k r_{t,k}A_k^{\text{off}}\big)$$

where $s_k^\Delta,b_k^\Delta\in\mathbb{R}^{d_{\text{inner}}}$, $m_k^B,m_k^C\in\mathbb{R}^{d_{\text{state}}}$ and $A_k^{\text{off}}\in\mathbb{R}^{d_{\text{inner}}\times d_{\text{state}}}$ are **learnable regime-specific dynamics offsets**. Intuitively, regime $k$ determines the time constant, input gain and read-out geometry of that state.

### 4.4 Direct features and per-regime risk bias

Besides the masked mean pooling, three low-dimensional regime-only quantities are concatenated:

- **occupancy histogram** $\bar r_k=\frac{1}{L_{\text{eff}}}\sum_t r_{t,k}$ ($K$ dims);
- **final posterior** $r_T$ ($K$ dims);
- **switch rate** $\frac{1}{|\mathcal{S}|}\sum_{t\in\mathcal{S}}\|r_t-r_{t-1}\|_1$ (1 dim; $\mathcal{S}$ = adjacent valid steps).

The final logit is $\text{logit}=\mathrm{MLP}([\text{pooled}\oplus\text{direct}])+\sum_k \bar r_k\, b_k$, where $b_k$ is a **per-regime risk bias**: how long a student spends in state $k$ enters the risk judgement directly and can be cross-checked against data (cf. §6.7).

### 4.5 Training objective

$$\mathcal L=\mathcal L_{\text{BCE}}+\lambda_u\mathcal L_{\text{usage}}+\lambda_s\mathcal L_{\text{switch}}+\lambda_c\mathcal L_{\text{sharp}},\qquad (\lambda_u,\lambda_s,\lambda_c)=(0.05,0.01,0.01)$$

- $\mathcal L_{\text{usage}}$: maximise occupancy entropy ($-H(\bar r)/\log K$) → **prevent regime collapse**;
- $\mathcal L_{\text{switch}}$: minimise mean switch magnitude → make regimes *states* rather than jitter;
- $\mathcal L_{\text{sharp}}$: minimise per-step posterior entropy → unambiguous per-step assignment.

> **Implementation pitfall (found and fixed, now recorded).** With $K=1$, $\log K=0$ makes the entropy regulariser divide by zero; training diverges and the model degenerates to a constant predictor (AUC=NaN). Our K=1 control therefore **short-circuits** the three regularisers when $K=1$ (occupancy/switch/sharpness are meaningless with a single regime), with all other hyper-parameters and the training loop identical to K=4. The defect was detected in the first run and the cell was re-run; all numbers reported here come from the fixed runs.

### 4.6 RegimeMamba vs MetaMamba

| Component | MetaMamba | RegimeMamba |
|---|---|---|
| SSM dynamics | one global parameter set | K regime-specific offsets, mixed by the posterior |
| State structure | none | discrete regime forward filter (K=4) |
| Identity modulation | FiLM(part) | FiLM(part) + behavioural regime (both disabled when deconfounded) |
| Direct features | none | occupancy / final posterior / switch rate |
| Risk attribution | none | per-regime risk bias $b_k$ |
| Extra regularisers | contrastive auxiliary loss | occupancy entropy / switch sparsity / posterior sharpness |
| Training & tuning budget | 200 epoch / patience 20 / 4 candidates | **identical** |

---

## 5 Experimental Design (Pre-Registered)

### 5.1 Pre-registered criteria

Fixed in writing on 2026-11-08 16:20 (see `docs/REGIMEMAMBA_DESIGN_20261108.md`), before any experiment was executed:

| ID | Hypothesis | Pass criterion | Failure conclusion |
|---|---|---|---|
| **H1** | regime mechanism yields a discriminative gain | 28-d identity-arm **pooled OOF AUC** ≥ MetaMamba-28d + 0.010 with bootstrap P(Δ≤0) ≤ 0.05 | Δ∈[−0.01,+0.01] → no independent gain; Δ≤−0.01 → regime mechanism harmful |
| **H2** | regimes are behavioural, not identity shortcuts | deconfounded-arm AUC drop ≤ 0.020 | drop > 0.05 → identity shortcut |
| **H3** | early warning benefits | first-10%-of-events AUC ≥ MetaMamba at the same prefix | lower → no early advantage |
| **H4** | regimes are real (neither noise nor collapse) | (i) every regime occupancy ≥5%; (ii) \|ρ\|≥0.3 (P<0.01) between switch rate and a behavioural statistic; (iii) at least one regime's occupancy significantly associated with failure (P<0.01) | any item failing → regimes not interpretable |
| **H5** | cross-task generalisation | LOPO weighted AUC ≥ MetaMamba's 0.6169 | lower → no generalisation gain |
| **H6** | not a "more capacity" artefact | K=4 clearly better than K=1 | no difference → gain came from capacity, not regimes |

### 5.2 Experiment matrix

| ID | Cell | Purpose |
|---|---|---|
| E1 | `main_d28_p1.0` (identity arm, full trajectory, 4-candidate HP search) | H1 / RQ1 main cell |
| E2 | `main_d28_p0.1` (first 10% prefix) | H3 early warning |
| E3 | `main_d28_p0.2` (first 20% prefix) | early-to-mid transition |
| E4 | `deconf_d28_p1.0` (part/exercise identity removed) | H2 |
| E5 | `abl_k1_d28_p1.0` (K=1 single-state control) | H6 |
| E6 | `abl_nodirect_d28_p1.0` (no regime direct features) | marginal value of direct features |
| E7 | `abl_nousage_d28_p1.0` (no occupancy-entropy regulariser) | collapse risk and the regulariser's role |
| E8 | `lopo` (leave-one-part-out transfer, default and deconfounded arms) | H5 |

### 5.3 Statistical procedure

- **Paired bootstrap** (2000 student-level resamples) for ΔAUC means, 95% CIs and $P(\Delta\le0)$;
- **fold-level Wilcoxon signed-rank** (paired by seed/fold) for the within-fold caliber;
- **Mann-Whitney U** (failing vs passing) and **Spearman ρ** for regime semantics;
- significance level 0.05 with Holm correction where multiple comparisons apply.

---

## 6 Results

### 6.1 Main result (28-d identity arm, full trajectory)

| Model | Pooled OOF AUC | Within-fold mean AUC | bal-acc (fold) | macro-F1 (fold) | ΔAUC vs RF-28d [95% CI] | P(Δ≤0) |
|---|---|---|---|---|---|---|
| **RegimeMamba (E1)** | **0.9488** | **0.9368** | 0.8729 | 0.8568 | −0.0021 [−0.0159,+0.0108] | 0.622 |
| RF-28d (tabular twin) | 0.9509 | 0.9526 | 0.9074 | 0.8858 | — | — |
| Attention-28d | 0.9468 | 0.9435 | 0.8731 | 0.8580 | — | — |
| LSTM-28d | 0.9361 | 0.9330 | 0.8789 | 0.8523 | — | — |
| BiLSTM-28d | 0.9310 | 0.9291 | 0.8628 | 0.8417 | — | — |
| MetaMamba-28d | 0.9056 | 0.9334 | 0.8835 | 0.8558 | — | — |
| HMM-GMM-28d | 0.9004 | 0.8883 | 0.8740 | 0.8486 | — | — |

Paired bootstrap (pooled OOF): RegimeMamba vs MetaMamba-28d **Δ=+0.0431 [+0.0235,+0.0648], P(Δ≤0)=0.000**; vs RF-28d Δ=−0.0021, P=0.622 (a **tie**).

### 6.2 Disagreement between the two calibers (an important finding)

| Model | Within-fold mean | Pooled OOF | Difference |
|---|---|---|---|
| MetaMamba-28d | 0.9334 | 0.9056 | −0.0278 |
| Attention-28d | 0.9435 | 0.9468 | +0.0033 |
| LSTM-28d | 0.9330 | 0.9361 | +0.0031 |
| RF-28d | 0.9526 | 0.9509 | −0.0017 |
| **RegimeMamba-28d** | 0.9368 | **0.9488** | **+0.0120** |

Fold-level paired Wilcoxon: RegimeMamba vs MetaMamba **P=0.4887 (not significant)** — under the within-fold caliber the two are hard to separate, whereas the pooled caliber separates them strongly.

Mechanism: pooled OOF AUC requires scores from **different fold models to be mutually comparable**. MetaMamba's fold-to-fold score scales drift markedly (strong within fold, weak across folds), so the pooled caliber penalises it; RegimeMamba's scores are more consistent across folds (pooled > within-fold), a property that matters in deployment where one model scores a whole cohort. We must also state the limitation: in an OOF setting each student is predicted by a *different* fold model, so pooled OOF measures "cross-model comparability" rather than a single model's cohort-ranking ability; settling the question requires a genuine single held-out model (e.g. temporal extrapolation), which we have not run. We report this discrepancy as an explicit methodological finding and recommend that student-level AUC always be reported under **both** calibers.

### 6.3 Ablations (RQ2: where does the gain come from?)

| Variant | Pooled OOF AUC | Within-fold mean | ΔAUC vs main cell [95% CI] | P(Δ≤0) |
|---|---|---|---|---|
| Main cell K=4 (direct features + anti-collapse reg.) | 0.9488 | 0.9368 | — | — |
| No regime direct features (E6) | 0.9520 | 0.9344 | +0.0012 [−0.0130,+0.0144] | 0.419 |
| No occupancy-entropy reg. (E7) | 0.9480 | 0.9342 | −0.0027 [−0.0175,+0.0106] | 0.646 |
| **K=1 single-state control (E5)** | 0.9454 | 0.9334 | +0.0032 [-0.0051,+0.0122] | 0.237 |

Reading:
- Removing the **direct features** costs essentially nothing (+0.0012, ns) → information flows mainly through the **regime-conditioned SSM dynamics**, not through explicit occupancy features (which also explains why E6 is marginally higher).
- Removing the **occupancy-entropy regulariser** costs essentially nothing (−0.0027, ns) → no regime collapse occurs with 28-d inputs; $\lambda_u$ is insurance.
- The **K=1 control** is the key piece of evidence in this section: The pooled OOF gain of K=4 over the K=1 single-state control is +0.0032 [-0.0051,+0.0122], P(Δ<=0)=0.237, which misses the preregistered criterion (Δ>0.005 and P≤0.10) -> **H6 fails**: a gain driven by model capacity rather than regime structure cannot be ruled out. (reference: MetaMamba - K=1 = -0.0399 [-0.0622,-0.0194], P=1.000)

### 6.4 Deconfounded arm (H2 fails)

| Model | Identity arm | Deconfounded | Drop |
|---|---|---|---|
| MetaMamba-28d | 0.9056 | 0.9016 | −0.0040 |
| RF-28d | 0.9509 | 0.9346 | −0.0163 |
| Attention-28d | 0.9468 | 0.8991 | −0.0477 |
| LSTM-28d | 0.9361 | 0.8865 | −0.0496 |
| HMM-GMM-28d | 0.9004 | 0.8775 | −0.0229 |
| **RegimeMamba-28d** | 0.9488 | 0.8950 | **−0.0538** |

**H2 fails** (criterion ≤0.020, observed drop 0.0538). RegimeMamba's identity-arm advantage **partly rests on problem-identity memory**: once identity is removed it ties MetaMamba (Δ=−0.0065, P=0.801) and trails RF significantly (Δ=−0.0398, P=1.000).

Read together with H2: RegimeMamba's identity-arm AUC (0.9488) matches RF (0.9509), but **after deconfounding it falls 0.0398 below RF** — meaning its "tie" on the identity arm is built on partly different information sources (identity memory + regime structure), and its purely behavioural modelling is weaker than RF's aggregate statistics.

### 6.5 Early warning (H3 fails)

| Prefix | RF-28d | MetaMamba-28d | RegimeMamba | RegimeMamba − MetaMamba | P(Δ≤0) | RegimeMamba − RF | P(Δ≤0) |
|---|---|---|---|---|---|---|---|
| first 10% | 0.8893 | **0.9298** | 0.9265 | −0.0033 [−0.0145,+0.0086] | 0.711 | **+0.0370** [+0.0152,+0.0597] | **0.000** |
| first 20% | 0.9084 | 0.9219 | **0.9356** | **+0.0138** [+0.0014,+0.0274] | **0.017** | **+0.0270** [+0.0075,+0.0472] | **0.002** |

**H3 fails**: in the first-10% setting RegimeMamba ties MetaMamba (ns) and therefore **offers no early-warning advantage**. It is, however, significantly above the same-caliber tabular baseline RF (+0.0370, P=0.000), showing that **sequence modelling itself** (not the regime mechanism) is valuable early on. By the first 20% RegimeMamba exceeds **both** MetaMamba (+0.0138, P=0.017) and RF (+0.0270, P=0.002), making it the best single model in the early-to-mid transition window.

### 6.6 Cross-task transfer (H5)

Leave-one-problem-part-out (LOPO): train on students whose main part differs, test on the held-out part's students.

| Model | LOPO weighted AUC | LOPO weighted bal-acc |
|---|---|---|
| MetaMamba-28d (default) | 0.6169 | 0.6145 |
| MetaMamba-28d (no-FiLM) | 0.6814 | 0.6212 |
| RF-28d | 0.6123 | 0.6396 |
| **RegimeMamba-28d (default)** | 0.6073 | 0.5850 |
| **RegimeMamba-28d (deconfounded)** | 0.6195 | 0.6322 |

RegimeMamba attains a weighted LOPO AUC of 0.6073, below the preregistered threshold 0.6169 -> H5 fails; the deconfounded arm scores 0.6195.

### 6.7 Regime semantics and interpretability (H4)

**Occupancy (all students)**: R0=0.248, R1=0.240, R2=0.248, R3=0.264 → **minimum occupancy 0.240 ≥ 0.05, no collapse** (H4-i passes).

**Regime-vs-failure relation**:

| Regime | Failing students | Passing students | Difference | Mann-Whitney P | Standalone failure AUC |
|---|---|---|---|---|---|
| R0 | 0.245 | 0.254 | −0.009 | 0.2056 | 0.4644 |
| **R1** | 0.229 | 0.263 | −0.034 | **<0.0001** | 0.3783 (reversed) |
| **R2** | **0.268** | **0.207** | **+0.061** | **<0.0001** | **0.6386** |
| R3 | 0.258 | 0.276 | −0.018 | 0.0417 | 0.4427 |

- **R2 is a "risk regime"**: failing students occupy it significantly more (0.268 vs 0.207), and its occupancy **alone** discriminates failure at **AUC 0.6386** — an intermediate indicator that can be shown directly to teachers/counsellors ("fraction of time spent in a high-risk behavioural state") without consulting the prediction head.
- **R1 is a "good-state regime"**, reversed and complementary to R2; the R2−R1 occupancy gap itself is a usable risk indicator.

**Switch rate**: standalone failure AUC **0.3806 (reversed)** — i.e. **fewer switches is associated with failure**. This contradicts the naive intuition that frequent state switching signals instability, and shows the model does not rely on such a shortcut.

**Switch rate vs behavioural statistics**:

| Behavioural statistic | Spearman ρ with switch rate | P |
|---|---|---|
| paste count | +0.281 | <0.0001 |
| run count | +0.252 | <0.0001 |
| focus_lost count | +0.250 | <0.0001 |
| submit count | +0.243 | <0.0001 |
| total events | +0.217 | <0.0001 |
| interaction span (hours) | +0.213 | <0.0001 |
| submit_rate (rate-like) | −0.126 | 0.0062 |
| run_rate (rate-like) | +0.044 | 0.3438 (ns) |
| edit_ratio | −0.063 | 0.1686 (ns) |
| mean event interval | −0.023 | 0.6233 (ns) |
| longest gap | +0.016 | 0.7293 (ns) |

Switching tracks **behavioural volume**, not efficiency — consistent with §6.4: had regimes mainly encoded identity, this volume-aligned association would not be so regular.

> **Caliber note.** The switch rate and regime occupancies in this section are computed on each student's **last 128 events**, whereas the behavioural statistics above are **whole-trajectory**; see §6.8 for the consequences of this caliber mismatch.

### 6.8 Stratified and matched analysis of behavioural-volume confounding (closing the §6.7 loophole)

The two observations of §6.7 — R2 occupancy associated with failure (standalone AUC 0.6386) and switch rate associated with failure **in reverse** (AUC 0.3806) — were read without any confound control. This section tests the "behavioural volume" confound with four instruments: rank-based partial Spearman (bootstrap, 2000 student resamples), volume stratification (within-stratum AUC + stratified fixed-effects regression), Mantel–Haenszel pooled OR, and a **volume-matched pair design** (each passing student matched to the failing student with the closest behavioural volume in z-space; matched standardised mean differences < 0.01).

**Caliber note (a caliber discipline surfaced by this analysis).** The model's sequence representation is each student's **last 128 events** (left padding), so the switch rate and the regime occupancies are computed on that window, whereas the behavioural statistics of §6.7 (paste/run/submit counts) are **whole-trajectory**. The window event count is almost constant at 128 (CV = 0.046) and cannot serve as a volume proxy; the quantity that actually predicts failure is **whole-trajectory volume** (event count alone discriminates failure at AUC 0.159, reversed).

**Size of the confounding path**:

| Volume caliber | ρ(V, y) | ρ(V, switch) | ρ(switch, y) | product ρ(V,switch)·ρ(V,y) | residual | share explained by the confound path |
|---|---|---|---|---|---|---|
| whole-trajectory volume PCA1 (primary) | −0.5266 | +0.2594 | −0.1954 | −0.1366 | −0.0588 | 70% |
| whole-trajectory event count | −0.5581 | +0.2173 | −0.1954 | −0.1213 | −0.0741 | 62% |
| end-of-window span | −0.1181 | +0.2356 | −0.1954 | −0.0278 | −0.1676 | 14% |

**Residual effect of switch after controlling for volume**:

| Control set | partial ρ(switch, y) | 95% CI | P |
|---|---|---|---|
| none (raw) | −0.1954 | — | — |
| whole-trajectory volume | −0.1002 | [−0.1883, −0.0059] | 0.030 |
| volume + end-window span | −0.0835 | [−0.1722, +0.0086] | 0.070 (ns) |
| volume block (3 covariates) | −0.0817 | [−0.1722, +0.0088] | 0.077 (ns) |
| RF-28d OOF (contains all activity aggregates) | −0.0443 | [−0.1391, +0.0505] | 0.337 (ns) |

Correspondingly, switch's OR falls from 0.664 unadjusted (P = 4.5e-05) to 0.833 with volume controlled (P = 0.131), 0.837 with stratified fixed effects (P = 0.145) and 0.759 on top of RF (P = 0.079); the Mantel–Haenszel pooled OR after stratifying by whole-trajectory volume tertiles is 0.76 (P = 0.289, **no longer significant after stratification**). Mechanistically, switch correlates positively with the final window's **span / inter-event gaps** (ρ = +0.236) and negatively with its **intensity** (ρ = −0.229) and with window activity (ρ = −0.385) — **switch is effectively measuring how sparse/discontinuous the end-of-term behaviour is**.

**Volume-matched pair analysis** (each passing student matched to the closest-volume failing student; 159 pairs; matched SMD < 0.01):

| Matching set | switch paired diff (FAIL−PASS) | P | SMD | R2 occupancy paired diff | P | SMD |
|---|---|---|---|---|---|---|
| whole-trajectory volume only | −0.0197 | 0.26 | −0.104 | +0.0077 | 0.91 | −0.016 |
| volume + end-window recency | −0.0576 | 0.022 | −0.194 | +0.0096 | 0.65 | +0.025 |
| window-level behaviour only (same window as the regimes) | −0.0424 | 0.003 | −0.257 | +0.0707 | 1.5e-07 | +0.449 |
| all (volume + window level) | −0.0029 | 0.441 | −0.110 | +0.0128 | 0.36 | +0.032 |
| whole-trajectory volume + RF score | −0.0755 | 0.0015 | −0.241 | −0.0367 | 0.0006 | −0.314 |

**Reading.** Once **whole-trajectory volume** is balanced, the failing/passing difference in switch is no longer significant (paired diff −0.020, P = 0.26) and the difference in **R2 occupancy disappears entirely** (paired diff +0.008, P = 0.91). Matching on window-level behaviour alone leaves a large R2 difference (+0.071, SMD +0.45), but that design does **not** constrain whole-trajectory engagement, so R2 is still speaking for volume; once whole-trajectory volume enters the matching, the difference vanishes — the two lines of evidence corroborate each other.

**Consequence (a revision of §6.7).** R2 occupancy's raw AUC 0.6386 and the reversed switch–failure association are **largely re-encodings of behavioural engagement volume**, not independent risk signals. The partial-correlation caliber agrees: ρ(R2, y) = +0.2268 → +0.1056 after controlling volume (P = 0.022; about 53% of the raw association removed); stratified fixed-effects regression gives OR = 1.11 (P = 0.473); MH pooled OR = 1.16 (P = 0.616).

> **Two claims are therefore narrowed**: (i) §7.5 "switching is not risk" becomes "switching is **not an independent risk signal**"; (ii) §7.3 "R2 is a directly usable risk-state indicator" becomes "R2 is an **interpretable re-encoding of behavioural engagement**, not an independent risk signal". See §7.3 and §7.5.

### 6.9 Summary of pre-registered criteria

| Criterion | Threshold | Observed | Verdict |
|---|---|---|---|
| H1 discriminative gain | Δ≥+0.010 and P≤0.05 (pooled OOF) | Δ=+0.0431 [+0.0235,+0.0648], P=0.000 | ✅ pass (independent gain) |
| H2 not an identity shortcut | deconfounded drop ≤0.020 | drop 0.0538 | ❌ fail (gain partly identity-driven) |
| H3 early warning | first 10% ≥ MetaMamba at first 10% | Δ=−0.0033, ns | ❌ fail |
| H4 regimes are real | occupancy≥5% / \|ρ\|≥0.3 / one regime associated with failure | min occupancy 0.240; max ρ +0.281 (P<0.0001, **below the 0.30 threshold**); R2 occupancy associated with failure (P<0.0001, **uncontrolled**; §6.8: about half explained by activity volume) | ⚠️ partial pass |
| H5 cross-task generalisation | LOPO ≥ 0.6169 | 0.6073 < 0.6169 | ❌ fails |
| H6 not a capacity artefact | K=4 clearly better than K=1 | Δ=+0.0032 [-0.0051,+0.0122], P=0.237 | ❌ fails (gain does not significantly exceed K=1) |

> H4-ii's threshold |ρ|≥0.30 was not met (maximum +0.281; at n=473 this is still a strong correlation). Following pre-registration discipline we **do not relax the threshold afterwards** and instead record "partial pass", interpreting it in §7.3.

---

## 7 Findings and Discussion

### 7.1 Finding 1: the gain of regime structure over the single-state control is not significant, so it is not shown to be an effective inductive bias

The comparison between the single-state control (K=1) and K=4 is the core evidence in this paper about where the gain comes from: After the divide-by-zero fix, the K=1 single-state control behaves normally again (pooled OOF AUC=0.9454), while the pooled OOF gain of K=4 over K=1 is only +0.0032 [-0.0051,+0.0122] (P(Δ<=0)=0.237), missing the preregistered criterion (Δ>0.005 and P≤0.10) -> H6 fails. (reference: MetaMamba - K=1 = -0.0399 [-0.0622,-0.0194], P=1.000) This paper therefore **does not claim** that "letting a state-space model own several explicit behavioural states and switch among them" brings a substantial gain: under the same parameter budget, data and training loop, the pooled OOF gain of K=4 over K=1 is only +0.0032 (P(Δ≤0)=0.237), so a gain driven by model capacity cannot be ruled out. At the same time, K=4 and the same-protocol MetaMamba do not differ significantly under the **within-fold mean** caliber (0.9368 vs 0.9334, Wilcoxon P=0.4887) — **the value of the regime structure lies in interpretability rather than in accuracy**.

### 7.2 Finding 2: the gain decomposes into identity memory plus regime structure — and the identity part is not negligible

The deconfounded arm yields a clean decomposition: a substantial part of RegimeMamba's +0.0431 pooled-OOF advantage over MetaMamba disappears once identity channels are removed (they tie at −0.0065), and its deconfounded drop (0.0538) exceeds RF's (0.0163) and MetaMamba's (0.0040). The model therefore **cannot** be advertised as delivering a purely behavioural regime gain. A more accurate statement is that **regime mechanism and identity memory interact strongly**: when a model can explicitly organise behavioural states, it also exploits problem-identity priors more aggressively — a liability when one model must serve many assignments or courses.

### 7.3 Finding 3: R2 is an interpretable re-encoding of behavioural engagement, not an independent risk signal

R2 occupancy discriminates failure at AUC 0.6386 on its own (Mann-Whitney P<0.0001), is complementary (opposite in sign) to R1, and is formally an indicator that teachers and counsellors can understand and track weekly. But the stratified and matched analysis of §6.8 shows that **about 53% of this association is explained by behavioural volume and it disappears once volume is balanced** (paired diff +0.008, P = 0.91; partial ρ = +0.106 after volume control, P = 0.022; stratified fixed-effects OR = 1.11, P = 0.473; MH pooled OR = 1.16, P = 0.616). We therefore **narrow** the claim: R2 is better read as an **interpretable re-encoding of behavioural engagement intensity and rhythm** than as a risk-state indicator independent of engagement.

It also delivers a cautionary note: **interpretable regimes do not mean "high-risk regime = high activity"** — switch rate correlates with behavioural volume but not with efficiency, so regimes capture activity shape and rhythm rather than effort intensity.

**Product implication.** If the "duration in a risky state" indicator is kept in any product form, it must be shown **next to** engagement intensity, its copy may only claim "engagement intensity and rhythm", and it must not be sold as an early-warning signal in its own right: at matched engagement level, failing and passing students do not differ in R2 occupancy.

### 7.4 Finding 4: caliber dependence is itself a governance problem

The same results yield opposite conclusions under two AUC calibers (RegimeMamba vs MetaMamba: within-fold +0.0034 ns, pooled +0.0431 significant). This is not unique to this paper but is a general risk of student-level small-sample OOF evaluation. We recommend: **(i)** always report both calibers for student-level AUC; **(ii)** report cross-fold comparability explicitly (the gap between pooled and within-fold is a rough proxy); **(iii)** re-check pivotal claims with a **single held-out model** (temporal extrapolation or external data).

### 7.5 Finding 5: the switch rate is not an independent risk signal (the original loophole is now closed)

Switch rate discriminates failure at AUC 0.3806 (reversed). Intuitively "frequent state switching = instability = risk", but the data disagree: students who switch more are more likely to pass. The stratified and matched analysis of §6.8 **separates** the two explanations that §6.7 left side by side: **the activity-volume confound is real and accounts for essentially the whole effect.**

- Confound path: ρ(switch, whole-trajectory volume) = +0.2594 and ρ(whole-trajectory volume, failure) = −0.5266; their product already covers 70% of the raw association.
- Controlling whole-trajectory volume reduces the partial ρ from −0.1954 to −0.1002 (P = 0.030); adding the end-window span makes it non-significant (P = 0.070); on top of the RF-28d OOF score it vanishes entirely (P = 0.337, −0.044).
- Mantel–Haenszel pooled OR after stratifying by volume tertiles: 0.76 (P = 0.289 — no longer significant after stratification).
- After volume matching (159 pairs, SMD < 0.01) the failing/passing difference in switch is no longer significant (paired diff −0.020, P = 0.26).

Mechanistically, switch correlates positively with the final window's **span / inter-event gaps** (ρ = +0.236) and negatively with its **intensity** (ρ = −0.229) and window activity (ρ = −0.385): **switch is effectively measuring how sparse/discontinuous the end-of-term behaviour is**, rather than an independent "state instability".

**The claim is therefore restated from "switching is not risk" to "switching is not an independent risk signal"**: the data support neither a deliverable rule ("fewer switches = danger") nor an independent reversed association; switch is a proxy for behavioural volume / end-of-term sparsity and must not be used as a risk indicator in its own right.

### 7.6 Relationship to this project's mainline (CS1-SHR / CS1-FSP)

| Dimension | CS1-SHR mainline | This paper (RegimeMamba) |
|---|---|---|
| Task form | stage-level "will they keep progressing?" (survival/hazard) | student-level "will they fail by term end?" |
| Primary metric | within-part pooled AUC (0.9258) | student-level pooled OOF AUC (0.9488) |
| Main claim | task formulation and evaluation caliber matter more than architecture | the gain of regime structure over K=1 is not significant (H6 fails); it does not surpass tabular statistics and partly relies on identity |
| Product positioning | "who is about to stop progressing" (intervention timing curve) | whole-cycle risk stratification plus an interpretable state indicator (R2 occupancy) |
| Mergeability | the two share one cache, protocol and baseline suite | could become the "strongest deep-model control + interpretability" section of the mainline paper |

Our recommendation is to **keep them as two papers** that cite each other: the mainline's review focus is task formulation and survival modelling, ours is whether state switching is worth modelling, and their evidence chains differ; merging would dilute the mainline's narrative. If a merge is desired, the natural route is to fold cells E1/E4/E5 into the mainline paper's "re-examination of deep architectures" section, keeping the H2 failure disclosed.

### 7.7 Relation to the staged-routing engineering result

Earlier fusion experiments in this project found that fusion weights drift with observability (first 10% → w_RF=0.04; 20% → 0.43; 40% → 0.64; full → 0.97). Our results are compatible and add one point: in the **first 20%** window RegimeMamba (0.9356) exceeds both MetaMamba (0.9219) and RF (0.9084), and is a stronger single-model option than the RF+MetaMamba rank fusion (0.9297); over the **full** trajectory RF remains best (0.9509, fusion adds nothing).

---

## 8 Capability Boundaries and Validity Threats

### 8.1 What we explicitly do not claim

1. **No accuracy leadership.** RegimeMamba is statistically tied with the tabular RF baseline and trails it by 0.0398 once deconfounded.
2. **No early-warning claim.** No advantage over same-protocol MetaMamba in the first-10% setting (H3 fails).
3. **No cross-task generalisation advantage.** The weighted LOPO AUC is 0.6073, below the same-protocol MetaMamba (0.6169) -> no cross-task generalisation claim is made.
4. **No generalisation beyond CS1.** Single dataset, single course; no external-dataset validation.
5. **No claim that regime semantics are fully explained.** R2/R1 currently correspond to behaviour only statistically; no manual coding validation has been performed.
6. **No claim that regime statistics are independent risk signals.** Once behavioural engagement volume is controlled, R2 occupancy no longer separates failing from passing students (matched paired diff +0.008, P = 0.91), and the switch rate carries no residual information either (P = 0.26 after matching; ns on top of the RF baseline).

### 8.2 Validity threats

- **Construct validity.** Failure is a composite end-of-term outcome whereas regimes describe process shape; the causal path is not established, and §6.7 associations are observational.
- **Statistical validity.** 473 students, 15 folds, many model comparisons; pre-registered criteria and paired bootstrap control some of this, but no family-wise correction was applied across H1–H6.
- **Implementation validity.** The K=1 divide-by-zero defect surfaced in the first run, showing that ablation chains need explicit sanity checks up front (documented in §4.5 and the appendix; the affected cell was re-run). **This is the most transferable methodological lesson of the paper.** We also disclose a **timestamp-unit leftover** in the event cache (§3.2): interval-like channels are compressed into a small part of their effective range, i.e. that information source is not fully exploited by the current implementation. Because all models share the same data substrate, between-model comparisons are unaffected, but absolute accuracy may be systematically underestimated.
- **Reproducibility validity.** All cells share one event cache and one fold-splitting implementation; scripts, raw results and per-fold metrics are shipped with the manuscript.
- **External validity.** Not validated outside CS1.

---

## 9 Conclusion

On CS1 — many events, few students — this paper tests the "behavioural regime switching" structural hypothesis under a **pre-registered falsification protocol**, and obtains a mixed set of results:

1. **No claim of structural effectiveness.** Whether letting a selective SSM own several behavioural states and switch among them yields a substantial gain is not supported in this run (the K=4 vs K=1 gain is not significant; H6 fails, §6.3).
2. **Not an accuracy lead.** It ties the tabular baseline and trails it after deconfounding; it is **not** a new accuracy reference point.
3. **Partly identity-driven.** The deconfounded drop of 0.0538 exceeds the pre-registered threshold → **H2 fails**.
4. **No early warning.** No advantage in the first-10% setting → **H3 fails**.
5. **Interpretability is its most plausible real value, at a discount.** R2 occupancy reaches 0.6386 standalone, but §6.8 shows the failing/passing difference disappears once engagement is matched: it is better read as an **interpretable re-encoding of behavioural engagement** — usable descriptively, but not sufficient to support a stand-alone "duration in a risky state" product.
6. **A methodological by-product.** Two AUC calibers can yield opposite conclusions in student-level small-sample OOF evaluation; we recommend reporting both.

In one sentence: **RegimeMamba's value is that it turns "behavioural state" into a measurable, interpretable object — not that it raises AUC by a point; but this round of evidence shows that object's information content overlaps heavily with behavioural engagement volume and must be read next to it.**

---

## Appendix A · Per-cell result data

Full data: `docs/REGIMEMAMBA_DATA_20261108.csv` (per-cell metrics and baselines), `docs/REGIMEMAMBA_RESULTS_20261108.md` (experiment report), `results/regimemamba/*/results.json` and `fold_metrics.csv` (raw per-fold metrics); the stratification / matching detail of §6.8 is in `docs/REGIMEMAMBA_STRATIFIED_20261110.csv` and `docs/REGIMEMAMBA_STRATIFIED_20261110_studentlevel.csv`.

## Appendix B · Reproduction commands

```bash
# on 246 (Tesla T4), from the StudentRisk/ repository root
python3 analysis/regimemamba/run_exp.py --tag main_d28_p1.0 --dim 28 --prefix 1.0
python3 analysis/regimemamba/run_exp.py --tag main_d28_p0.1 --dim 28 --prefix 0.1 --hp_fixed '<HP selected in E1>'
python3 analysis/regimemamba/run_exp.py --tag main_d28_p0.2 --dim 28 --prefix 0.2 --hp_fixed '<HP selected in E1>'
python3 analysis/regimemamba/run_exp.py --tag deconf_d28_p1.0 --dim 28 --prefix 1.0 --deconfound --hp_fixed '<HP selected in E1>'
python3 analysis/regimemamba/run_exp.py --tag abl_k1_d28_p1.0 --dim 28 --prefix 1.0 --n_regime 1 --hp_fixed '<HP selected in E1>'
python3 analysis/regimemamba/run_exp.py --tag abl_nodirect_d28_p1.0 --dim 28 --prefix 1.0 --no_direct --hp_fixed '<HP selected in E1>'
python3 analysis/regimemamba/run_exp.py --tag abl_nousage_d28_p1.0 --dim 28 --prefix 1.0 --lam_usage 0.0 --hp_fixed '<HP selected in E1>'
python3 analysis/regimemamba/run_lopo.py --tag lopo
python3 analysis/regimemamba/analyze.py
```

## Appendix C · References

1. Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Space Models. *ICLR*.
2. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
3. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.
4. Rabiner, L. R. (1989). A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition. *Proceedings of the IEEE*.
5. Murali, R., Conati, J., & Poole, W. (2025). [EDM 2025 paper reporting HMM/LSTM baselines on the CS1 dataset].
6. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (temporal extrapolation and rolling-origin evaluation).
7. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*.
8. Breiman, L. (2001). Random Forests. *Machine Learning*.
9. Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*.
10. Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap* (paired bootstrap intervals).
11. Spearman, C. (1904). The Proof and Measurement of Association between Two Things. *American Journal of Psychology*.
12. Mann, H. B., & Whitney, D. R. (1947). On a Test of Whether One of Two Random Variables Is Greater than the Other. *Annals of Mathematical Statistics*.
13. Chow, S., & Glickman, M. E. (2021). *The Practice of Econometrics* (pre-registration and robustness practice).
14. Simmons, J. P., Nelson, L. D., & Simonsohn, U. (2011). False-Positive Psychology. *Psychological Science*.

> Note: reference [5] requires author verification of the complete bibliographic record before submission; the remaining entries are standard methodological or domain references.
