# CS1-SHR: Stage-Hierarchical Survival Risk for Stage-Level Dropout Prediction in Introductory Programming

**Author:** Jian Wang¹

**Affiliation:**
¹ Department of Computer Science and Educational Technology (to be completed: [institution], [city, China])

**Corresponding author:** Jian Wang (wangjian98@example.com)

**Submission date:** 1 November 2026 (v10)

**Version note (v10).** This is the third revision of the paper's main line, after v8 (the MetaMamba main version) and v9 (the stage-task revision plan). It is **not an incremental revision**: it is a re-specification of the model that follows the data-generating process rather than the architecture.

- ✅ **§1.2 (new):** Four *structural* (not parametric) gaps of the previous architecture on CS1, each with a reproducible number: part-prior dominance, failure of cross-part generalization, collapse of the early-warning claim, and a mismatch between labels and behavioural evidence.
- ✅ **§3 (new):** CS1 is formalised as a **discrete-time survival problem** (258 dropout events, 215 right-censored students), and the **two failure modes with opposite behavioural profiles** are characterised.
- ✅ **§4 (new):** We propose **CS1-SHR** — a part-invariant representation layer, a stage-level dropout hazard channel, a complementary terminal-failure channel, a student-level flat channel, probability-space fusion, and per-stage calibration with abstention.
- ✅ **§5 (new):** An evaluation protocol whose survival line is a **trivial part-prior baseline**: student-grouped CV, part-wise pooled AUC, leave-one-part-out (LOPO) generalisation, lead-time curves, and calibration.
- ✅ **§6 (new):** Main results together with **five negative results** (linear hazard, per-part specialisation, feature-level residualisation, intensity features, and a hazard-aggregation trap).
- ✅ **§8 (new):** An explicit capability boundary — final-failure discrimination for *still-active* students is only **0.54–0.61**, so the system must not promise "predicting who will fail", only "who is about to stop progressing".

**Target venues:** *IEEE Transactions on Learning Technologies* (TLT) · *Journal of Educational Data Mining* (JEDM) · *Computers & Education* · *Artificial Intelligence in Education*

---

## Abstract

**Background.** Programming platforms emit massive fine-grained IDE interaction logs (text_insert / text_remove / text_paste / focus_gained / focus_lost / run / submit). Recent work models these logs with deep sequence architectures — selective state spaces with task-aware modulation and few-shot meta-learning — to identify at-risk students at the end of a semester. On the public CS1 dataset, however, we show through systematic experiments that the apparent advantage of such architectures comes almost entirely from a **problem-part difficulty prior**, not from individual behavioural modelling: a baseline that knows only *which assignment a student is working on* already reaches AUC 0.9135, and after removing this shortcut the proposed architecture ranks *last* on individual discrimination (0.4829, below chance). Its cross-part generalisation collapses to chance under leave-one-part-out evaluation (0.5000), and its "early warning" claim does not hold in the data (predicting failure from the first part alone gives AUC 0.6518, below the prior baseline). The root cause is not capacity but a **mismatch between task definition and data structure**.

**Purpose.** We propose **CS1-SHR** (Stage-Hierarchical Survival Risk), a risk model matched to the data-generating process of CS1, and answer three research questions: (RQ1) Does re-formulating risk as a **stage-wise progression decision** (at the end of each part, "will this student keep progressing?") simultaneously remove part-prior dominance and the failure of early warning? (RQ2) With only 473 students, 2171 stage samples and 258 events, which model family has defensible sample efficiency? (RQ3) Does the end-of-term failure label mix two failure modes with opposite behavioural profiles, and is a single failure probability sufficient for intervention?

**Method.** We first formalise CS1 as a **discrete-time survival problem**: 473 students, 2171 (student, part) stage samples, 258 dropout events, and 215 students who reach part 7 and are right-censored. CS1-SHR consists of: (1) a **part-invariant representation layer** (peer rank, empirical-Bayes shrunk z, completion/progress, rhythm ratios, self-referenced deviation; part identity is forbidden); (2) a **dropout hazard channel** with time-varying covariates, producing per-stage hazards $h_i$ and the cumulative risk $P_{\text{drop}}(k) = 1 - \prod_{i \le k}(1-h_i)$ under explicit right-censoring; (3) a **complementary terminal channel** trained only on students who survive stage $k$, so that it estimates $P(\text{fail} \mid \text{survive to } k)$ and makes *different* errors from the dropout channel; (4) a **student-level flat channel** trained on all students; and (5) **probability-space fusion with a decision layer**, $P(\text{fail}) = P_{\text{drop}}\cdot q_{\text{drop}} + (1-P_{\text{drop}})\cdot P_{\text{terminal}}$, with per-stage calibration, an abstention band, and an intervention-timing curve. Evaluation uses **student-grouped** 5-fold × 3-seed OOF, with **part-wise pooled AUC** as the main metric, and mandates a part-prior baseline, LOPO generalisation, lead-time curves and calibration metrics.

**Results.** (i) **The task form itself removes the confound**: at stage level the part-prior baseline reaches a pooled AUC of only **0.4164 (below chance)**, while the same baseline reaches 0.9135 at student level. (ii) **Stage-level discrimination is strong and early**: the dropout channel attains a part-wise pooled AUC of **0.9258**, and the **first part alone** suffices for warning (pooled 0.9423, +0.31 over the stage-level prior). (iii) **Under a fair four-way protocol the deep architecture is significantly worse than the matched feature-engineering baseline**: MetaMamba − RF = **−0.1302** [−0.1649, −0.0975], $P(\Delta \le 0) = 100\%$, LOPO weighted difference −0.0447 ($P = 100\%$), at 38–76 minutes versus 15 seconds of training time. (iv) **Early warning is determined by the fusion design**: at cutoff $k=1$ the two-channel fusion reaches AUC **0.7359** versus 0.7126 for the flat baseline, while at $k = 2/3/4$ the flat baseline overtakes it (0.8293 / 0.8813 / 0.8820) — i.e. the value of decomposition concentrates in the **earliest actionable window**. (v) The **per-stage cumulative risk curve** peaks at part 4 (0.7927, n=250 still active), directly indicating when to intervene. We also report five negative results: linear cloglog hazards (pooled 0.63–0.78), per-part specialisation (0.9216, below the pooled 0.9258), feature-level residualisation (−0.027), M4 intensity features (ΔAUC −0.0013), and a hazard-aggregation trap in which early snapshots are broadcast to all stages (AUC 0.3897 at $k=1$, below chance).

**Conclusion.** On CS1-type data — *many events, very few students* — the binding constraint on predictive performance is not architectural complexity but **task form, sample efficiency and evaluation protocol**. Our contribution is therefore not a larger network, but: (1) a re-formulation of academic risk as **stage-wise progression decision** that is provably immune to the part-prior shortcut; (2) a survival-style two-channel model that, at equal discrimination, additionally provides a **censoring-correct probability scale, per-stage intervention timing, and separate handling of two opposite failure modes**; (3) an evaluation protocol whose survival line is a trivial baseline, which restores the previously claimed advantage to its true source; and (4) an explicit **capability boundary**: for students who have not yet dropped out, final-failure discrimination is only 0.54–0.61, so the deployable product value is "**who is about to stop progressing**", not "who will fail".

**Keywords:** learning analytics; discrete-time survival analysis; hazard modelling; stage-wise progression decision; IDE behaviour logs; early warning; programming education; calibration; peer-relative features

---

## 1 Introduction

### 1.1 Background and educational motivation

Every keystroke, focus switch, code run and submission leaves a "digital footprint" in the integrated development environment (IDE) that can be used to infer a learner's state. CS1 datasets typically contain tens of millions of such timestamped events across seven event types: text_insert, text_remove, text_paste, focus_gained, focus_lost, run, submit [3]. Early identification of at-risk students from these logs has direct practical value for intervention, tutoring scheduling and curriculum revision: a reliable signal in the first weeks of a semester is worth far more than a retrospective report.

The difficulty of this task is not that "the logs are large" but that "**the labels are late**". The end-of-term pass/fail label is a cumulative consequence of a whole semester, separated from "where and when the learner actually got stuck" by a long chain of semantics. A student who stalls in part 3 and a student who works at maximum intensity until failing at the final exam share the same label but require completely different intervention timing and intervention content.

### 1.2 Four structural gaps of prior work, as measured on CS1

The MetaMamba architecture built in this project (selective state space + task-aware FiLM + task-contrastive loss + FOMAML few-shot evaluation) [10,11,7] reached student-level AUC 0.9318 and F1(FAIL) 0.9129. To identify the true source of that gain we ran a systematic audit on CS1, obtaining four **structural** (rather than parametric) conclusions:

| # | Finding | Key numbers | Source |
|---|---|---|---|
| **G1** | **The apparent advantage comes almost entirely from the part-difficulty prior** | A baseline that knows only "which assignment this student works on" reaches AUC **0.9135**; MetaMamba-11d adds only +0.0184 and 7d only +0.0050 ($P(\Delta \le 0) = 23.8\%$, not significant); the within-part ranking is **inverted relative to the overall ranking in 15/15 comparisons**, with the 7d variant at 0.4829 — **below chance** | Experiment A |
| **G2** | **The "load-bearing wall" of task-aware modulation is a by-product of confound amplification** | Feeding a wrong task_id makes the predicted-positive ratio **exactly 1.0** and the accuracy **exactly the base rate** 0.6638 (the model degenerates into a majority-class predictor); additive concat and multiplicative FiLM produce **identical confusion matrices** when fed correctly ($r = 0.9986$), so the necessity of multiplicative modulation is unproven | Experiment B / FiLM ablation |
| **G3** | **Cross-part generalisation fails (deployment would break)** | Student-level leave-one-part-out: the de-confounded RF variant obtains a weighted AUC of **0.5000 = chance**; MetaMamba-7d reaches 0.6134 (part-7 ROC **0.221**) | Experiment H2 |
| **G4** | **The "early warning" claim does not hold in the data** | Student-level lead-time using only the first 1/2/3 parts: AUC **0.6518 / 0.7921 / 0.8339**, every prefix below the part-prior baseline of 0.9053 | Experiment H3 |

> **The implication of G1 motivates this paper.** Whether a deep model "wins" depends on whether it is allowed to exploit part identity. When the shortcut exists, the deep model's increment is invisible (+0.005, not significant); when it is removed, the deep model ranks *last* on individual discrimination. **Fix the task form and the evaluation protocol before arguing about architectures** — this is the first-order conclusion of the project.

### 1.3 Research questions

- **RQ1 (task form).** Does rewriting academic risk from "end-of-term classification" to a **stage-wise progression decision** simultaneously remove part-prior dominance (G1) and the failure of early warning (G4)?
- **RQ2 (sample efficiency and model family).** With 473 students, 2171 stage samples and 258 events, which family is defensibly sample-efficient — deep sequence architectures, strongly regularised non-linear learners, or constrained survival-style models?
- **RQ3 (multiple failure modes).** Does the end-of-term failure label mix two failure modes with opposite behavioural profiles? If so, can a single failure probability support intervention, or must they be modelled separately with explicit capability boundaries?

### 1.4 Contributions

1. **Task form.** We formalise risk as a **stage-wise progression decision** (§3.4) whose label is derived directly from the trajectory in the IDE logs, requiring **no new annotation**, and show that under this form the part-prior baseline drops to **0.4164 (below chance)**, mechanically removing the confound behind G1.
2. **Model.** We propose **CS1-SHR** (§4): part-invariant representation, a dropout hazard channel, a complementary terminal channel, a student-level flat channel, probability-space fusion, and per-stage calibration with abstention. It reaches a part-wise pooled AUC of **0.9258** at stage level and additionally provides what flat models cannot: a **censoring-correct probability scale, a per-stage intervention-timing curve, and interpretable hazard ratios**.
3. **Evaluation protocol.** We propose a protocol whose survival line is a **trivial part-prior baseline** (§5): student-grouped CV, part-wise pooled AUC as the main metric, LOPO cross-part generalisation, lead-time early-warning curves, and calibration (Brier / LogLoss).
4. **Honest boundaries.** We report five negative results (§6.4) and one explicit capability boundary (§8): final-failure discrimination for **still-active** students is only **0.54–0.61**, so the product must not promise "predicting who will fail" but "who is about to stop progressing".
5. **Reproducibility.** All scripts, raw results and logs are released (Appendix A).

### 1.5 Organisation

§2 reviews related work; §3 introduces the dataset, the failure-mode profiles and the two-layer task formalisation; §4 defines CS1-SHR; §5 gives the evaluation protocol; §6 reports main and negative results; §7 discusses the key findings; §8 discusses limitations and capability boundaries; §9 concludes.

---

## 2 Related Work

### 2.1 Learning analytics and early risk identification

Surveys of educational data mining distinguish outcome prediction from process-level intervention [1], and machine-learning methods for online-course outcomes have been reviewed systematically [2]. In programming courses, learning-analytics approaches to identifying at-risk programming students provide the methodological paradigm of "behavioural indicators → risk list" [4]. We follow that paradigm but change the **prediction target** from an end-of-term outcome to a **stage-wise progression decision**, using fine-grained IDE logs [3] as the sole data source.

### 2.2 Deep sequence architectures on educational behaviour logs, and their limits

LSTM [5], Transformer [6] and state-space families (Mamba [10], Mamba-2 [11], S5 [12], H3 [13]) provide strong long-sequence machinery; FiLM [7] provides conditional modulation; MAML-style methods [14–19] and contrastive learning [20–22] provide few-shot and self-supervised machinery. Together these constitute the MetaMamba architecture built in this project. **This paper does not deny the value of these components; it delineates their applicability on CS1.** When the sample size is a few hundred students and the event count is in the tens of millions, the marginal return of model capacity collapses while the return of **task form and evaluation protocol is orders of magnitude larger** (§6.3, §6.5) — an observation consistent with the strong small-tabular baselines reported in [23,24].

### 2.3 The missing survival-analysis perspective

Dropout prediction in educational data mining is usually implemented as "binary classification with an end-of-term label", which discards information at two levels: (i) **exposure time** — different students occupy different stages, so pooling them confounds risk intensity with exposure duration; (ii) **censoring** — students still enrolled should not be treated as negatives. Discrete-time hazard models [31,33] address both: each stage is a risk set, an observed event is a failure, a non-event advances the student to the next risk set, and the model yields per-stage conditional hazards, from which "cumulative risk up to stage $k$" — exactly the quantity needed for intervention decisions — follows naturally. **To our knowledge, this survival formulation has not been applied to the CS1 setting.**

### 2.4 Peer-relative features and learner profiles

In educational data mining, a learner's *position relative to peers* is often more robust than absolute quantities, which are confounded by course difficulty and individual baselines. Our feature layer systematically uses three peer-anchored constructions: **peer rank** (a cross-part-comparable distributional position), **empirical-Bayes shrunk z** (small segments shrink toward global statistics), and **self-referenced deviation** (the current stage relative to the learner's *own* previous stages, which transfers across parts by construction). This is consistent with the ablations in §6.4: removing all activity-volume dimensions does not reduce discrimination (Δ +0.0034, $P = 17.8\%$, not significant), while peer-normalised features alone reach 0.9295.

### 2.5 Differences from the closest work

| Dimension | Conventional practice | MetaMamba (previous version) | CS1-SHR (this paper) |
|---|---|---|---|
| Prediction target | End-of-term failure | End-of-term failure | **Stage-wise progression + term-level terminal risk** |
| Part identity | Often a feature or task label | Injected twice (features + FiLM) | **Forbidden**; only an extrapolable stage-index baseline |
| Censoring | Not handled | Not handled | **Explicit right-censoring and hazard products** |
| Multiple failure modes | Mixed label | Mixed label | **Dropout / terminal two-channel + probability fusion** |
| Output | One probability | One probability | **Per-stage hazards + cumulative risk + intervention timing + abstention** |
| Main evaluation | Random stratified CV | Random stratified CV | **Student-grouped CV + part-prior baseline + LOPO + lead-time + calibration** |

### 2.6 A published baseline on the same dataset: the EDM 2025 classifier comparison

Murali et al. published the first systematic comparison of classifiers across several open-ended learning datasets at EDM 2025 [34], and one of their datasets **is the very CS1 MOOC IDE log used in this study** (their 469 students = our 473 minus the 4 students with only a single log row; their 159 passes / 310 failures = our 159 / 314). On that dataset they compared four classifiers and reported last-window accuracies of **HMM 76.97±1.42 > LSTM 71.51±2.05 > stratified dummy baseline 53.95±2.19 > FUMA (rule-based) 34.65±2.44**. This has two direct consequences for this paper:

1. **A baseline gap.** This paper must include an HMM baseline. §6.2 reports it under our main protocol (stage-level, student-grouped, part-wise pooled AUC): **HMM (multinomial) 0.6671** and **HMM (Gaussian) 0.5879**, i.e. 0.2587 / 0.3379 below our RF baseline of 0.9258 (paired bootstrap $P(\Delta \le 0) = 100\%$).
2. **Protocol differences explain the differing conclusions.** They use 60-second incremental windows (a student's different windows can span folds, a mild leakage channel) and accuracy; we use student-grouped cross-validation, part-wise pooled AUC and the stage-level task. §6.8 gives a like-for-like replication and item-by-item comparison.

In addition, their **rule-based FUMA falls below the stratified random baseline on this dataset** (34.65 < 53.95), which is consistent with this paper's family of negative results (intensity-style features add nothing; raw sequences cannot support a standalone predictor) and therefore provides **external evidence** that hand-crafted statistics and rule mining do not suit CS1.

---

## 3 Dataset and Problem Formulation

### 3.1 The CS1 dataset

We use the publicly released CS1 MOOC IDE action-log dataset [3] (Zenodo DOI: 10.5281/zenodo.3972772), shared with this project's CodeEMO work.

| Item | Value |
|---|---|
| Students | **473** (failed 314 / passed 159, imbalance ≈ 2:1) |
| Raw events | **28,588,309** timestamped events |
| Event types | text_insert, text_remove, text_paste, focus_gained, focus_lost, run, submit (7) |
| Problem parts | 7 (part 1…7) |
| Label convention | `failed = 1` (314 students), `passed = 0` (159 students) |

Student ids, exercise ids and course-instance ids were stripped before analysis; log collection was approved by the institutional teaching and research committee as routine course analysis.

### 3.2 Stage partition and stage-level samples

Treat every (student, part) pair as a **stage sample**, whose input is the student's aggregated behaviour within that part and whose output is "does the student keep progressing afterwards". This yields **2171** stage samples.

| Stage (part) | Stage samples | Dropout events | Dropout rate |
|---|---:|---:|---:|
| 1 | 471 | 83 | 0.176 |
| 2 | 388 | 67 | 0.173 |
| 3 | 321 | 31 | 0.097 |
| 4 | 290 | 40 | 0.138 |
| 5 | 250 | 15 | 0.060 |
| 6 | 236 | 22 | 0.093 |
| 7 | 215 | 0 | **0.000 (right-censored)** |

> **Why part 7 has zero dropout rate**: by definition, a dropout event can only occur at the furthest part a student reached, provided they never reached part 7. The 215 students who reach part 7 have no dropout event and are therefore **right-censored**. Any flat model that treats them as negatives systematically under-weights early behaviour.

### 3.3 Two failure modes with opposite behavioural profiles

The stage-level task covers **255/314 = 81.2%** of failing students; the remaining **59 (18.8%)** fail despite completing every part and are, by definition, invisible to "will they drop out". Comparing behavioural profiles across the three groups:

| Group | n | Intensity (log median) | Delete rate | Focus switches/h | Completion | Span |
|---|---:|---:|---:|---:|---:|---:|
| Dropout-type failure | 255 | **3.88 (lowest)** | 29.4% | **2.6 (lowest)** | 1.00 | 183.2h |
| **Completes yet fails** | 59 | **4.39 (highest)** | 27.8% | **3.9 (highest, +34% vs pass)** | 1.00 | 191.8h |
| Passes | 159 | 4.31 | **23.5% (lowest)** | 2.9 | 1.00 | 174.8h |

The two failure profiles are **opposite**: dropout-type learners are "low-intensity procrastinators", terminal-type learners are "high-intensity crammers with fragmented attention and heavy rework". This answers the first half of RQ3: **the single failure label mixes two opposite mechanisms**, and because the two differ almost monotonically along "intensity", they must be modelled by separate channels.

### 3.4 Two-layer task formalisation

**Layer 1 (stage-wise progression decision).** For a stage $(s,p)$, define

$$\text{drop}(s,p) = \mathbb{1}\{\,p = \max_{p'} \text{part}(s,p') \;\wedge\; \text{part}(s) < 7\,\}$$

i.e. "this student stops progressing after this part". The label is **derived from log facts, with no manual annotation**. The model learns the conditional hazard

$$h(s,p) = P\big(\text{drop}(s,p) = 1 \mid \text{alive at stage } p,\; x_{s,p}\big),$$

where $x_{s,p}$ are the stage-$p$ covariates and **exclude part identity**.

**Layer 2 (term-level terminal risk).** Let $P(\text{fail} \mid \text{alive at } k)$ denote the probability that a learner who has survived stage $k$ fails by the end of term. This layer covers the 18.8% that Layer 1 cannot see.

### 3.5 Discrete-time survival view and censoring

Treat each stage as a risk-set observation: an event is a failure, a non-event moves the student into the next risk set, and a student reaching part 7 is **right-censored** there. The dropout probability is then given by the product of per-stage hazards:

$$P(\text{drop} \mid \mathcal{F}_k) = 1 - \prod_{i \le k} \big(1 - h_i\big),$$

where $\mathcal{F}_k$ is the information available up to stage $k$. Two properties are essential: (i) **censoring correctness** — completers contribute no spurious negatives; (ii) **the right decision quantity** — the left-hand side is "has this student already dropped out by stage $k$", precisely what scheduling and tutoring decisions need.

> ⚠️ **A modelling trap we measured and formalised.** If a "first-$k$-parts snapshot" is broadcast to *all* of a student's stages and then multiplied, students who survive longer receive *higher* risk (student-level AUC **0.3897** at $k=1$, below chance). Hazards must be **updated per stage** (time-varying covariates), and survival products may only be taken over **observed** stages.

### 3.6 Leakage constraints

Stage samples are *rows*; if several rows of the same student land in both training and test folds, all stage-level numbers are inflated. All main experiments therefore use **student-grouped** stratified cross-validation, plus a GroupKFold re-check: stage-level numbers *increase* under the grouped protocol (0.9294 → 0.9322) while student-level numbers *decrease* by 0.0265 (0.7955 → 0.7690), confirming that stage-level conclusions are not leakage artefacts.

---

## 4 Method: CS1-SHR

### 4.1 Architecture overview

```
                     ┌───────────── Layer 0: part-invariant representation ─────────┐
IDE events (28.6M) → │ peer rank + EB-shrunk z + completion/progress + rhythm ratios  │
  → (student, part)  │ + self-referenced deviation  [+ optional lightweight encoder] │
                     └────────────────────────┬───────────────────────────────────────┘
                                              │  x_i (no part identity)
             ┌────────────────────────────────┴─────────────────────────────────────┐
             ▼  Channel A: dropout hazard (discrete time, time-varying covariates)   │
             log(-log(1-h_i)) = α(stage_i) + f_A(x_i)                                │
             → per-stage hazards → P_drop(k) = 1 − Π_{i≤k}(1−h_i)                     │
             │                                                                     │
             ├─▶ Channel B: complementary terminal channel                          │
             │      P(final fail | alive at k) ← first-k-stage trajectory signature   │
             │                                                                     │
             └─▶ Student-level flat channel (all students, first-k snapshot)         │
                                        │                                           │
                                        ▼  probability-space fusion + decision layer │
   P(fail) = P_drop(k)·q_drop + (1 − P_drop(k))·P_terminal
   Intervention list: ranked by P_drop(k) + threshold + abstention band
   Learning output: per-stage hazard ratios + calibration curve + earliest-intervention curve
```

### 4.2 Layer 0: part-invariant representation

Let the raw aggregate of stage $i$ be $\mathbf{r}_i \in \mathbb{R}^{10}$ (seven event counts, total events, distinct exercises touched, span in hours).

**(a) Peer rank.** Within a part, each dimension is mapped to its empirical quantile among peers,

$$\hat{q}_{ij} = \frac{\#\{r \in \mathcal{G}_p : r_j \le r_{ij}\} + 0.5}{|\mathcal{G}_p| + 1},$$

with $\mathcal{G}_p$ the peer set of part $p$. Quantiles are **comparable across parts**, which suits LOPO evaluation.

**(b) Empirical-Bayes shrunk z.** Peer-set sizes are very uneven (471 stage samples in part 1 versus 250 in part 5 and 236 in part 6 with few positives), so raw part-wise medians/MADs are high-variance. For each part we estimate

$$z^{\text{part}}_{ij} = \frac{r_{ij} - \text{med}_p(j)}{\text{MAD}_p(j) + 10^{-3}},\qquad
z^{\text{shrunk}}_{ij} = w_p\, z^{\text{part}}_{ij} + (1 - w_p)\, z^{\text{global}}_j,\quad w_p = \frac{n_p}{n_p + \kappa},$$

with shrinkage strength $\kappa$ (we use $\kappa \in \{20, 100\}$ in sensitivity analyses) and $n_p$ the number of peers of part $p$ in the training fold.

**(c) Completion and progress.** The ratio of distinct exercises touched to the part's exercise-count baseline, plus log event and exercise counts, form the *progress family* of features — the dominant stage-level signal (0.8646 on its own).

**(d) Rhythm ratios and self-referenced deviation.** Run/submit/delete ratios per unit of intensity characterise debugging behaviour; the standardised deviation of the current stage from the learner's **own earlier stages** characterises trajectory evolution, and being self-referential it transfers across parts by construction.

> **Explicitly excluded**: **part identity** in any form, and **feature-level residualisation** (stage-level pooled −0.027; see §6.4).

### 4.3 Channel A: dropout hazard channel

We adopt a discrete-time cloglog link:

$$\log\big(-\log(1-h_i)\big) = \alpha(\text{stage}_i) + f_A(\mathbf{x}_i),$$

where $f_A$ is a non-linear learner (random forest and histogram gradient boosting in this paper) and $\alpha(\cdot)$ is a cubic polynomial basis in the stage index (**extrapolable** to unseen parts, which favours LOPO). It is fitted by penalised (L2) IRLS, with the penalty $\lambda$ selected by student-grouped inner validation.

> **Measured warning.** Replacing $f_A$ by a linear function (i.e. plain cloglog regression) drops the pooled AUC to **0.6641 / 0.6753 / 0.7773** ($\lambda = 1/10/100$). **The survival form is useful, but non-linearity is indispensable**: both are required.

### 4.4 Channel B: complementary terminal channel

We train $P(\text{final fail} \mid \text{alive at } k)$ on the subset of students who survive stage $k$, using their first-$k$-stage trajectory signature (intensity, attention fragmentation, rework rate, progress). This channel is weak on its own (0.60–0.68) but makes **different errors** from the dropout channel, which is what makes it complementary: at $k=1$ the fusion reaches 0.7359, above the fusion that uses an all-student flat channel instead (0.7188).

### 4.5 Student-level flat channel

Trained on **all students**, mapping the first-$k$-stage snapshot (last stage features, inter-stage mean, inter-stage difference, number of stages used) to end-of-term failure. This is the best option for $k \ge 2$ (0.8293 / 0.8813 / 0.8820 for $k = 2/3/4$).

### 4.6 Probability-space fusion and decision layer

$$P(\text{fail}) = P_{\text{drop}}(k)\cdot q_{\text{drop}} + \big(1 - P_{\text{drop}}(k)\big)\cdot P_{\text{terminal}},$$

where $q_{\text{drop}} = P(\text{fail}\mid\text{dropout})$ is estimated within the training fold (**0.9884** over the full sample: 255 of 258 dropouts fail) and $P_{\text{terminal}}$ comes from Channel B or the flat channel.

Fusion must happen in **probability space**: probability-based fusion beats rank averaging by **+0.010 / +0.020 / +0.020** AUC at $k = 1/2/3$. On top of this, the decision layer provides (i) per-stage calibration (Brier / LogLoss); (ii) an **abstention band** (low-confidence cases routed to human review); and (iii) the **intervention-timing curve** (§6.7).

### 4.8 Intra-Part Stall Diagnostics (IPSD): exercises as observation resolution, not decision unit

The negative result of §6.9 shows that exercise-level **event localisation** is not learnable (events cover 0.58% of (part, exercise) cells; pooled AUC 0.492, below chance). The same data also shows that **81.8% of stalls occur in the middle of a part** (median completion ratio of that part only 0.474). Together these yield a methodological proposition: **the granularity of the decision unit must match the event base rate, so finer granularity should serve observation and diagnosis rather than decision-making.**

We therefore propose **IPSD (Intra-Part Stall Diagnostics)**, which pushes discrimination inside a part without changing the decision unit:

1. **Truncation augmentation**: for each (student, part), truncate the evidence at the $e$-th exercise, recompute all part-level aggregate and peer-normalised features, and form (student, part, $e$) training samples whose **label remains the part-level drop**;
2. **Intra-part monitoring**: at inference, use the evidence up to exercise $e$ to output "will this student stop within this part", yielding a risk curve and a first-crossing exercise index;
3. **Stuck-point diagnosis**: for flagged learners, compare each of their exercises against same-part peers (time span, edits, runs, remove/insert ratio, focus-switch rate) to locate the likely stuck exercise and emit directly readable evidence.

### 4.7 Contrast with MetaMamba

| Dimension | MetaMamba | CS1-SHR |
|---|---|---|
| Task | Student-level terminal failure | Stage-level progression + student-level terminal (two layers) |
| Part usage | Injected twice (features + FiLM) | **Forbidden**; only an extrapolable stage-index baseline |
| Censoring | Not handled | Explicit right-censoring with hazard products |
| Structure | Single black-box backbone | Three channels + probability fusion + calibration + abstention |
| Sequence | S6 state-space backbone (22k parameters, over-parameterised for 258 events) | Optional lightweight encoder used **only as a feature extractor**, dropped if it fails the acceptance criterion |
| Output | One probability | Per-stage hazards + cumulative risk + intervention timing + interpretable hazard ratios |

---

## 5 Experimental Setup

### 5.1 Protocol

- **Cross-validation**: 5 folds, **grouped by student** (no student spans folds), stratified by course-level label; seeds {42, 123, 777}; OOF probabilities averaged.
- **Stage-level model selection**: a 15% student-disjoint inner validation split inside each training fold for early stopping; the evaluation fold never participates in model selection (stricter than the legacy protocol).
- **LOPO**: leave one part out for training/testing. The held-out part has no training rows, so its peer statistics are computed from **that part's own rows** (an unlabelled transformation, hence not leakage); both arms use the same convention.
- **Early warning**: prefix-$k$ protocol — only information from the first $k$ stages.

### 5.2 Metrics

| Metric | Definition | Purpose |
|---|---|---|
| **Part-wise pooled AUC** | per-part AUC averaged with sample-count weights | **Main metric**: individual discrimination (immune to part priors) |
| judgeable | mean AUC over parts with ≥10 positives and ≥10 negatives | Removes small-part noise |
| Overall AUC / PR-AUC | whole sample | Secondary (inflated by part priors) |
| **Brier / LogLoss** | probability calibration | Required to set intervention thresholds |
| **Part-prior baseline** | segment dropout prior only | **Survival line**: models must beat it significantly |
| LOPO weighted AUC | leave-one-part-out | Cross-part transfer (deployment capability) |

### 5.3 Baselines

Mandatory: the **part-prior trivial baseline** (student level 0.9135 / stage level 0.4164); RF-46d / RF-7d; LSTM / BiLSTM / Attention (7d); **HMM (multinomial and Gaussian — the model family reported at 0.7697 on this very dataset by EDM 2025)**; MetaMamba-11d and its four-way variants; and the three channels and fusion variants of CS1-SHR.

### 5.4 Reproducibility

All scripts live in `analysis/hazard_model/` (`run_dchm.py`, `run_shr.py`, `run_early2.py`, `run_fusion_curve.py`, `run_combo.py`); the data are `data/stage_dataset.npz` (2171 stage samples). Experiments run on CPU; some use the Tesla T4 on our internal 246 server.

---

## 6 Results

### 6.1 Protocol reproduction check

| Model | Overall AUC | **Part-wise pooled** | judgeable | Brier | LogLoss |
|---|---:|---:|---:|---:|---:|
| RF, full features (this paper) | 0.9353 | **0.9258** | 0.9199 | 0.0493 | 0.1883 |
| RF, full features (prior closing report) | 0.9353 | **0.9258** | 0.9199 | — | — |

Digit-identical, confirming that our scripts share the main protocol of the project.

### 6.2 Stage-level main results (RQ1)

| Model | Overall AUC | **Part-wise pooled** | judgeable | Per-part AUC (part 1…6) |
|---|---:|---:|---:|---|
| **Stage-level part-prior baseline** | 0.6262 | **0.4164** | 0.4101 | — |
| CS1-SHR dropout channel (RF) | 0.9353 | **0.9258** | 0.9199 | 0.959 / 0.944 / 0.903 / 0.918 / 0.944 / 0.851 |
| CS1-SHR dropout channel (HGB) | 0.9323 | 0.9191 | 0.9126 | 0.951 / 0.938 / 0.905 / 0.915 / 0.936 / 0.839 |
| CS1-SHR + stage-wise shrinkage (κ=100) | 0.9390 | **0.9276** | 0.9222 | 0.959 / 0.947 / 0.900 / 0.920 / 0.936 / 0.871 |
| HMM (multinomial, 7 event symbols) | 0.6768 | 0.6671 | 0.6527 | — |
| HMM (Gaussian, seq9 continuous channels, no part identity) | 0.5870 | 0.5879 | 0.5800 | — |
| MetaMamba, de-confounded sequence only | 0.8594 | 0.8758 | 0.8710 | 0.886 / 0.888 / 0.894 / 0.906 / 0.827 / 0.825 |
| MetaMamba, de-confounded + context (residualised) | 0.7621 | 0.7643 | 0.7667 | 0.829 / 0.594 / 0.821 / 0.792 / 0.824 / 0.739 |
| MetaMamba, de-confounded + context (not residualised; strongest arm) | 0.8526 | 0.8465 | 0.8427 | 0.919 / 0.752 / 0.869 / 0.850 / 0.873 / 0.793 |

**Key conclusion.** The stage-level part-prior reaches only **0.4164 (below chance)**, so **the task form itself removes the part-prior shortcut** (first half of RQ1); and every MetaMamba variant remains significantly worse than the matched feature-engineering baseline once **stage-level task + de-confounding + residualisation + cross-part generalisation** are all applied (§6.5).

### 6.3 Early warning: the first part already suffices (RQ1, second half)

| Prefix $k$ (stages) | Samples | Overall AUC | **Part-wise pooled** | Stage-level prior | Lead |
|---:|---:|---:|---:|---:|---:|
| **1** | 2171 | 0.9369 | **0.9423** | 0.4164 | **+0.3107** |
| 2 | 1698 | 0.9222 | 0.9326 | 0.4448 | +0.2696 |
| 3 | 1309 | 0.9150 | 0.9317 | 0.4120 | +0.2870 |
| 4 | 988 | 0.9219 | 0.9335 | 0.4351 | +0.2447 |
| 5 | 698 | 0.9408 | 0.9582 | 0.3605 | +0.2844 |
| 6 | 448 | 0.9335 | 0.9417 | 0.4797 | +0.1937 |

Compared with student-level lead-time (0.6518 / 0.7921 / 0.8339, **all below the 0.9053 prior**), the root cause of G4 is a **mismatch between task and prediction time**, not model incapacity: the student-level task asks "will they fail by the end of term", for which the first 1–3 stages have no causal path, whereas the stage-level task asks "will they continue", for which the behaviour of the just-finished stage is directly causal.

### 6.4 Five negative results (equally important)

| Experiment | Result | Conclusion |
|---|---|---|
| **Linear cloglog hazard** | pooled 0.6641 / 0.6753 / 0.7773 ($\lambda=1/10/100$) | The survival form is not the bottleneck; **non-linearity is indispensable** |
| **Per-part specialisation** | pooled 0.9216 (below pooled 0.9258); $n_p$-weighted shrinkage $\kappa=10/30/100$ → 0.9264 / 0.9269 / **0.9276** | Small parts have too few positives; shrinkage yields **+0.0018 (noise level)** → **do not over-structure**; keep per-part models for calibration only |
| **Feature-level residualisation** | pooled 0.9258 → 0.8988 (−0.027) | Residualisation helps only at student level ($w=0.25$ improves all three metrics); **forbidden at stage level** |
| **M4 intensity features** | ΔAUC −0.0013 [−0.0038, +0.0012], $P(\le 0) = 84.4\%$ | Event-interval/density intensity features give **no gain** at stage level |
| **Hazard broadcast trap** | broadcasting a first-$k$ snapshot to all stages → student-level AUC **0.3897** at $k=1$ | Hazards must be updated per stage |

Feature-family ablation (stage level): removing all activity-volume dimensions moves pooled AUC from 0.9294 to 0.9260 (Δ +0.0034, $P = 17.8\%$, **not significant**) → discrimination is **not** "just counting activity"; activity alone gives 0.8804, peer-normalised features 0.9295, completion + stage index 0.8646.

### 6.5 The four-way verdict on MetaMamba (RQ2)

Under the pre-registered acceptance criterion — the architecture must remain significantly better than a matched feature-engineering baseline with de-confounding, residualisation, the stage-level task and cross-part generalisation all applied — the verdict is negative.

| Comparison | Δ (part-wise pooled) | 95% CI | $P(\Delta \le 0)$ | Verdict |
|---|---:|---|---:|---|
| MetaMamba (four-way) − RF (four-way) | **−0.1302** | [−0.1649, −0.0975] | **100.00%** | significantly worse |
| MetaMamba (four-way) − RF (feature residualisation) | −0.1345 | [−0.1692, −0.1006] | 100.00% | significantly worse |
| MetaMamba de-confounded sequence − RF (feature residualisation) | −0.0230 | [−0.0518, +0.0058] | 94.30% | not significant |
| Context-channel net contribution (ctx − no ctx) | +0.1115 | [+0.0712, +0.1515] | 0.00% | significant (context does help) |

**LOPO (cross-part generalisation).** Leave-one-part-out weighted AUC: RF **0.9128** versus MetaMamba-7d **0.6134** (student-level: de-confounded V3 reaches 0.5000 = chance). Stage-level paired bootstrap Δ = −0.0447 [−0.0645, −0.0241], $P = 100\%$.

> **Verdict: the architecture line is closed.** Under the stage-level task every MetaMamba variant — including the strengthened arm that receives the same engineered features, de-confounding and residualisation — loses to RF, while RF produces results in 15 seconds against 38–76 minutes for MetaMamba. This negative result is itself publishable: it converts the paper from "I built a stronger architecture" into "**I re-defined the problem and gave a more honest evaluation**".

### 6.6 Student-level early warning and fusion (RQ3)

**Predicting end-of-term failure from the first $k$ stages** (identical features, identical folds, student-grouped):

| Cutoff $k$ | A) flat RF | A2) flat HGB | B) two-channel (complementary terminal) | C) two-channel (constant terminal rate) | D) terminal channel only | E) dropout channel only |
|---:|---:|---:|---:|---:|---:|---:|
| **1** | 0.6997 | 0.7126 | **0.7359** | 0.6551 | 0.6819 | 0.6525 |
| 2 | 0.8252 | **0.8293** | 0.7990 | 0.7368 | 0.6241 | 0.7371 |
| 3 | 0.8741 | **0.8813** | 0.7854 | 0.7654 | 0.5991 | 0.7656 |
| 4 | **0.8956** | 0.8820 | 0.8201 | 0.7923 | 0.5946 | 0.7927 |

**Combination study with the terminal channel replaced by the all-student flat channel**:

| Cutoff $k$ | flat HGB | dropout channel | probability fusion $P_{\text{drop}}q+(1-P_{\text{drop}})P_{\text{flat}}$ | rank average 0.5/0.5 | probability fusion 0.3/0.7 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7126 | 0.6525 | 0.7158 | 0.7058 | **0.7188** |
| 2 | **0.8293** | 0.7371 | **0.8298** | 0.8099 | **0.8298** |
| 3 | **0.8813** | 0.7656 | 0.8727 | 0.8525 | 0.8796 |
| 4 | **0.8820** | 0.7927 | 0.8804 | 0.8677 | 0.8783 |

**Reading.**
1. **The value of decomposition concentrates in the earliest window**: at $k=1$ the two-channel fusion reaches 0.7359 versus 0.7126 for the flat baseline (**+0.023**). The mechanism: first-part behaviour is extremely predictive of "will drop out" (pooled 0.9423) but only moderately predictive of "will fail"; decomposition routes the strong signal to the strong label and propagates it through $q_{\text{drop}} = 0.988$.
2. **For $k \ge 2$ the flat model overtakes** — "two channels are always better" is false; the terminal channel weakens as $k$ grows (0.6819 → 0.5946) because it is trained on the shrinking survivor subset.
3. **Complementarity comes from error diversity, not from a stronger single model**: the weak, survivor-trained terminal channel yields a larger fusion gain than the stronger all-student flat channel.
4. **Probability space beats rank space** (+0.010/+0.020/+0.020 at $k=1/2/3$).

**Full-horizon student-level fusion (473 students / 314 failures)**: dropout channel 0.7781 · terminal channel 0.6522 · **two-channel fusion 0.7949** (Brier 0.2076) · constant fusion 0.7778 (Brier 0.1815); the project's honest student-level reference under student-grouped CV is 0.7690.

### 6.7 Intervention-timing curve

| Cutoff stage $k$ | All students AUC | **Still-active subset** n | Still-active subset AUC | Median $P_{\text{drop}}(k)$ |
|---:|---:|---:|---:|---:|
| 1 | 0.6525 | 389 | 0.5389 | 0.112 |
| 2 | 0.7371 | 321 | 0.5627 | 0.306 |
| 3 | 0.7656 | 290 | 0.5893 | 0.406 |
| **4** | **0.7927** | 250 | 0.6078 | 0.486 |
| 5 | 0.7901 | 235 | **0.6121** | 0.545 |
| 6 | 0.7793 | 213 | 0.5940 | 0.616 |
| 7 | 0.7781 | 0 | — | 0.644 |

Cumulative risk increases monotonically and **peaks at part 4** (n=250 still active), making **part 4 the most cost-effective intervention window**. No previous model produced this curve (MetaMamba yields a single probability), yet it is exactly the quantity needed for scheduling and tutoring decisions.

### 6.10 IPSD: pushing part-level discrimination inside the part (positive result)

Same protocol (student-grouped 5-fold x 3 seeds, part-wise pooled AUC), comparing three ways of using the evidence:

| Evidence used | first 1 exercise | 3rd exercise | 8th exercise | whole part |
|---|---:|---:|---:|---:|
| Pure exercise-prefix model (§6.9) | 0.5892 | 0.6077 | 0.6306 | 0.6485 |
| **IPSD (truncation-augmented part-level features)** | **0.7915** | **0.7935** | **0.7979** | 0.9258 |

**Reading.**
1. **The first exercise alone already reaches a pooled AUC of 0.7915**, +0.20 over the pure exercise-prefix model. The reason is that IPSD preserves the expressive power of part-level features (peer normalisation, self-reference, completion ratio) and only changes *how much evidence has been seen*, whereas the pure prefix model dissolves the part-level structure into raw prefix counts.
2. The decline for $e \ge 12$ is not model degradation but **selection of the evaluation population**: students who reach the 12th exercise have already had the early dropouts removed, and the remaining base rate falls from 0.119 to 0.074.
3. **The intervention rule is directly deployable**: at truncation $e=3$ with a 0.5 threshold, precision 0.488 / recall 0.469 against a 0.106 base rate; restricting to the **top 5% riskiest** (student, part) pairs yields **precision 0.679, 6.4x the base rate** (top 10%: 0.495 / 4.7x; top 20%: 0.322 / 3.0x).

### 6.9 Decision-unit granularity: a negative result for exercise-level hazard

We further pushed the decision unit from problem parts down to individual exercises (the logs carry exercise ids; each part holds 10-30 exercises, and 97.1% of students proceed in prefix order) to test whether finer granularity buys earlier warning. The answer is no, and the reason is methodological.

| Task | Decision unit | Event share within the comparison cell | overall AUC | **Part-wise pooled** |
|---|---|---|---:|---:|
| **Part-level drop** (our backbone) | (student, part) | **6.0%-17.6%** | 0.9353 | **0.9258** |
| **Exercise-level drop_at** (locating the stall exercise) | (student, part, exercise) | **0.58%** (258 events / 44,764 rows) | 0.4915 | **0.4920** |

Early warning under the same label (will this student eventually stall), row-level part-wise pooled AUC:

| Exercise level (first k exercises) | pooled | Part level (first K parts) | pooled |
|---|---:|---|---:|
| first 1 exercise | 0.5892 | **first 1 part** (~30 exercises of behaviour) | **0.7282** |
| first 3 exercises | 0.6077 | first 2 parts | 0.7615 |
| first 10 exercises | 0.6306 | first 3 parts | 0.7587 |
| all exercises | 0.6485 | | |

**Reading.**
1. **Exercise-level hazard is below chance (0.492).** The cause is not weak features but the **event base rate within the comparison cell**: at part level the event covers 6-18% of the students in that part (learnable), whereas at exercise level it covers 0.58% of all exercise rows, leaving roughly one positive against 300 negatives in each (part, exercise) cell (not learnable). **The granularity of the decision unit must match the event base rate; cutting it finer does not automatically yield a better model, it turns a learnable problem into an unlearnable one.**
2. **Finer granularity also does not buy earlier warning**: exercise-level prefix features are below part-level aggregate features at every cutoff (0.589-0.649 vs 0.728-0.762); matched by amount of observed behaviour, "first part" (~30 exercises) at 0.7282 far exceeds exercise-level "first 10 exercises" at 0.6306. This supports our choice of the part as the decision unit.
3. **Exercise-level data nonetheless yields a new behavioural finding**: of the 258 stalling students, **81.8% stop in the middle of their final part** (completing a median of only 47.4% of that part's exercises; p25 26.7%), while only 18.2% finish the whole part; completers reach a median completion ratio of 1.000 in part 7. The mechanism of stalling is therefore **progressive attrition within a part**, not "finish a part and never start the next". The product implication is direct: **monitoring must happen inside a part, and "which exercise did they get stuck on" should be a diagnostic output rather than a prediction target** (§7.5).

### 6.8 Like-for-like comparison with the published EDM 2025 baseline

| Protocol | Model | Metric | This paper | Murali et al. (2025) [34] |
|---|---|---|---:|---:|
| **Stage level (our main protocol)** | RF / CS1-SHR | part-wise pooled AUC | **0.9258** | — |
| Stage level (our main protocol) | HMM (multinomial) | part-wise pooled AUC | 0.6671 (Δ −0.2587, P=100%) | — |
| Stage level (our main protocol) | HMM (Gaussian) | part-wise pooled AUC | 0.5879 (Δ −0.3379, P=100%) | — |
| Student level (8-fold, symbol HMM) | HMM | accuracy / AUC | 0.7294 / 0.7452 | **0.7697** / — |
| Student level (8-fold) | majority-class baseline | accuracy | 0.6638 | 0.5395 (stratified dummy) |
| Student level (8-fold) | LSTM | accuracy | — | 0.7151 |
| Student level (8-fold) | FUMA (rule-based) | accuracy | — | 0.3465 (**below baseline**) |

**Reading (three points).**
1. A like-for-like replication of their recipe ("one HMM per class + prior, 8 folds") yields **0.7294**, confirming directionally that an HMM clearly beats the majority baseline, but roughly 4pp below their reported 0.7697. The declared deviations are: 473 students vs 469; 1-hour windows instead of 60-second ones (60-second windows over students spanning up to 1169 hours would produce tens of thousands of state points, making EM infeasible); and hmmlearn 0.3.x no longer ships a GMM-HMM, so we approximate it with a single-Gaussian emission (its n_mix=1 case) — which degenerates to the majority class on window features (accuracy 0.6638 = base rate).
2. **The real difference is the protocol, not the implementation.** On the very same data, moving from "end-of-term failure + incremental windows + accuracy" to "stage-wise progression decision + student-grouped CV + part-wise pooled AUC" drops the HMM from 0.7294 to 0.6671 while lifting the feature-engineering baseline to 0.9258. **This is exactly the central claim of this paper: the task form and the evaluation protocol set the ceiling; architectural ranking is a secondary variable.**
3. Their **rule-based FUMA falls below the stratified random baseline on this dataset** (34.65 vs 53.95), which aligns with this paper's negative results (M4 intensity features; full-trajectory multi-scale features at 0.9200, −0.0058) and supplies external evidence that hand-crafted statistics and rule mining do not suit CS1.

---

## 7 Analysis and Discussion

### 7.1 Task form determines the ceiling more than architecture

Place the same models on two tasks: the student-level part-prior baseline reaches AUC 0.9135, the stage-level one 0.4164. **With the same data and the same features, changing only *what is predicted* compresses the deep model's apparent advantage from +0.018 to last place on individual discrimination.** Architectural comparisons in educational data mining should therefore begin with an audit of the task form and the prior baseline, not with network design.

### 7.2 Two opposite failure profiles require separate channels

Dropout-type (lowest intensity 3.88, least fragmented attention 2.6/h) and terminal-type (highest intensity 4.39, most fragmented attention 3.9/h, heaviest rework) are near-monotone opposites along "intensity". The binary outcome "failed" is therefore close to a **bimodal distribution** in feature geometry: any monotone classifier on intensity must compromise between the two modes. Fitting a stage-level dropout channel and a survival-conditioned terminal channel separately is equivalent to separating those modes.

### 7.3 Why deep architectures do not pay off on this data

Three testable reasons:
1. **Sample size**: 473 students, 2171 rows, 258 events — the capacity of deep models (22k–70k parameters) far exceeds what the data support.
2. **Much of the sequence information is redundant for this task**: 7-dimensional one-hot input lacks intensity information, and intensity-style features in turn give **no gain** at stage level (negative M4 result) — the information a sequence backbone mines from 128 raw steps is largely redundant for stage-level discrimination.
3. **Highly unbalanced sequence lengths**: event counts span about four orders of magnitude on a log scale; after `max_len = 128` truncation most students' sequences are heavily zero-padded, so the backbone actually sees very uneven amounts of information.

### 7.5 Stalling is progressive attrition within a part

An exercise-level replay of the 258 stalling students shows that **81.8% stop in the middle of their final part** (completing a median of only 47.4% of that part's exercises), with only 18.2% finishing the part; per-part stall rates are 17.6% / 17.3% / 9.7% / 13.8% / 6.0% / 9.3%. This complements the two opposite failure profiles of §7.2: terminal-type learners are the **highest-intensity, most fragmented** ones, whereas dropout-type learners **drain away gradually inside a part**.

This implication is operationalised by IPSD (§6.10): within the **first 1-3 exercises of each part** the system produces a within-part risk (pooled 0.79), the riskiest 5% of (student, part) pairs reach **precision 0.679 (6.4x base rate)**, and a per-exercise peer comparison emits the likely stuck exercise together with its behavioural evidence.

Two implications: (i) **the product cannot wait for the end of a part** — more than three quarters of stalls are already observable within a part; (ii) **"which exercise did they get stuck on" should be a diagnostic output** (replaying the flagged learner's last exercises against peer timing/edit/run differences), not a prediction target, because exercise-level event localisation is not learnable on this data (§6.9).

### 7.4 Fuse in probability space, and with channels that err differently

Probability-based fusion consistently beats rank averaging (+0.010 to +0.020), and a weak but differently-wired terminal channel contributes more to the fusion than a stronger all-student channel. The lesson is general: **when channels are strongly label-correlated, diversity in error patterns is worth more than marginal single-model strength.**

---

## 8 Limitations and Capability Boundaries

### 8.1 The capability boundary (the most important honest statement of this paper)

Final-failure discrimination for **still-active** students is only **0.54–0.61** (§6.7). Consequently:

- ✅ We can promise: **"who is about to stop progressing"** (stage-level pooled 0.9258; 0.9423 from the first part alone);
- ❌ We should not promise: **"who will fail, early"** (student-level lead-time 0.65–0.88, and only 0.54–0.61 on the still-active subset).

The product framing should change from "predict failure" to "**detect stalling and decide when to intervene**".

### 8.2 Methodological and data limitations

1. **Single dataset**: only CS1 is used; cross-dataset generalisation is empty. External validation on at least one public programming-behaviour dataset is the next mandatory step.
2. **Submission quality not yet validated**: completion-family features may be contaminated by "clicking exercise ids to farm submissions", and stage-level discrimination depends heavily on that family; until validated, causal interpretations of these features should be cautious.
3. **Calibration not yet systematically assessed**: we report Brier/LogLoss only; ECE, reliability diagrams and per-part calibration curves remain to be done.
4. **The terminal channel is sample-limited**: trained on survivors only, so its sample size shrinks as $k$ grows and its capability decays (0.68 → 0.59).
5. **Sequence information has not been re-examined**: we show only that raw sequences lose to engineered features at stage level, not that a lightweight sequence *encoder used as a feature extractor* fails.
6. **Student-level features depend on progress**: retrospective student-level AUC (0.9505) includes progress information and must not be used to argue early warning; all early-warning claims here use the prefix-$k$ protocol.

---

## 9 Conclusion

Revisiting early risk prediction for programming learners from the data-generating process of CS1 yields three conclusions:

1. **The apparent advantage of prior deep architectures can be fully attributed to the part prior**: the part-prior trivial baseline reaches AUC 0.9135 while MetaMamba adds only +0.0184; after removing the shortcut the 7d variant reaches 0.4829 — below chance — and the within-part/overall ranking is inverted in 15/15 comparisons (RQ1, first half).
2. **The task form is the most effective lever in this problem**: rewriting risk as a **stage-wise progression decision** drops the part prior to 0.4164 (below chance), allows warning from the first part alone (pooled 0.9423), produces an intervention-timing curve peaking at part 4 (0.7927), and shows the previous architecture significantly worse than the matched feature-engineering baseline under the four-way protocol (−0.1302, $P = 100\%$) (RQ1, second half, and RQ2).
3. **A single failure probability is insufficient for intervention**: dropout-type and terminal-type learners have opposite profiles and cover 81.2% / 18.8% of failures; they must be modelled separately, fused in probability space with channels that err differently, and reported with explicit capability boundaries ($k=1$ two-channel 0.7359 vs flat 0.7126; flat overtakes for $k \ge 2$; still-active subset only 0.54–0.61) (RQ3).

Accordingly, the proposed CS1-SHR is a model **matched to the structure of CS1**: a part-invariant representation, a censoring-correct dropout hazard backbone with per-stage calibration, a complementary terminal channel, a student-level flat channel, and a probability-space fusion with a decision layer. At equal discrimination it additionally provides a probability scale, intervention timing and interpretable hazard ratios, and it delineates precisely what the product may and may not promise.

**Code and data:** https://github.com/wangjian98/StudentRisk

---

## References

> **Academic-integrity note.** The bibliography continues the two-round audit of v3.1/v3.2 (author verification, then verification by the co-supervisor against academic databases, removing six unverifiable entries) and adds four references: three classical survival-analysis works (Cox 1972, Fine & Gray 1999, Kalbfleisch & Prentice 2002) and one cross-dataset classifier comparison at EDM 2025 [34] (all marked ⭐new; **please re-verify these four**). Errors are welcome via GitHub issues.

### A. Learning analytics and educational data mining

[1] C. Romero, S. Ventura. **Educational Data Mining: A Review of the State of the Art**. *IEEE Transactions on Systems, Man, and Cybernetics, Part C*, 2010, 40(6): 601-618.

[2] A. Alhothali, M. Albshir, B. Alharbi, et al. **Predicting Student Outcomes in Online Courses Using Machine Learning Techniques: A Review**. *Sustainability*, 2022, 14(8): 4517.

[3] J. Leinonen, A. Hellas. **Open IDE Action Log Dataset from a CS1 MOOC**. *Proceedings of the 4th International Workshop on Conducting Empirical Studies in Education (CSEDM) @ EDM 2022*, 2022.

[4] D. Azcona, P. H. Szwarcfiter, L. P. S. Fernández, et al. **Detecting Students-At-Risk in Computer Programming Classes with Learning Analytics**. *User Modeling and User-Adapted Interaction*, 2019, 29(4): 759-788.

### B. Sequence modelling, Transformers and deep learning foundations

[5] S. Hochreiter, J. Schmidhuber. **Long Short-Term Memory**. *Neural Computation*, 1997, 9(8): 1735-1780.

[6] A. Vaswani, N. Shazeer, N. Parmar, et al. **Attention Is All You Need**. *NeurIPS*, 2017.

[7] E. Perez, F. Strub, H. de Vries, et al. **FiLM: Visual Reasoning with a General Condition-Aware Layer**. *AAAI*, 2018.

[8] K. He, X. Zhang, S. Ren, J. Sun. **Deep Residual Learning for Image Recognition**. *CVPR*, 2016.

[9] J. L. Ba, J. R. Kiros, G. E. Hinton. **Layer Normalization**. *arXiv:1607.06450*, 2016.

### C. Mamba and selective state spaces

[10] A. Gu, T. Dao. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**. *arXiv:2312.00752*, 2023.

[11] T. Dao, A. Gu. **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**. *ICML*, 2024 / arXiv:2405.21060.

[12] J. T. H. Smith, A. Warrington, S. W. Linderman. **Simplified State Space Layers for Sequence Modeling (S5)**. *ICLR*, 2023.

[13] D. Y. Fu, T. Dao, K. A. Saab, et al. **Hungry Hungry Hippos: Towards Language Modeling with State Space Models (H3)**. *ICLR*, 2023.

### D. Meta-learning and few-shot learning

[14] C. Finn, P. Abbeel, S. Levine. **Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks (MAML)**. *ICML*, 2017.

[15] A. Nichol, J. Achiam, D. Schulman. **On First-Order Meta-Learning Algorithms (FOMAML)**. *arXiv:1803.02999*, 2018.

[16] J. Snell, K. Swersky, R. Zemel. **Prototypical Networks for Few-shot Learning**. *NeurIPS*, 2017.

[17] F. Sung, Y. Yang, L. Zhang, et al. **Learning to Compare: Relation Network for Few-Shot Learning**. *CVPR*, 2018.

[18] A. Raghu, M. Raghu, S. Bengio, et al. **Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML (ANIL)**. *ICLR*, 2020.

[19] L. Zintgraf, K. Shiarlis, M. Kurin, et al. **CAML: Fast Context Adaptation via Meta-Learning**. *ICML*, 2019.

### E. Contrastive learning and self-supervised representation

[20] T. Chen, S. Kornblith, M. Norouzi, G. Hinton. **A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)**. *ICML*, 2020.

[21] Z. Yue, Y. Wang, J. Duan, et al. **TS2Vec: Towards Universal Representation of Time Series**. *AAAI*, 2022.

[22] D. Bahri, Y. Tay, Y. Ann, et al. **SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption**. *ICLR*, 2022.

### F. Tabular foundation models and AutoML

[23] N. Hollmann, S. Müller, K. Hutter. **TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second**. *ICLR*, 2023.

[24] N. Hollmann, S. Müller, L. Purucker, et al. **Accurate Predictions on Small Tabular Data**. *Nature*, 2025, 637: 89-95. DOI: 10.1038/s41586-024-08328-6.

[25] F. Hutter, L. Kotthoff, J. Vanschoren (Eds.). **Automated Machine Learning: Methods, Systems, Challenges**. *Springer*, 2019.

[26] M. Christ, N. Braun, J. Neuffer, A. W. Kempa-Liehr. **Time Series FeatuRe Extraction on the basis of Scalable Hypothesis tests (tsfresh – A Python package)**. *Neurocomputing*, 2018, 307: 72-80.

### G. Other machine-learning foundations

[27] T. K. Ho. **Random Decision Forests**. *Proceedings of the 3rd International Conference on Document Analysis and Recognition*, 1995.

[28] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens. **Rethinking the Inception Architecture for Computer Vision**. *CVPR*, 2016.

[29] T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár. **Focal Loss for Dense Object Detection**. *ICCV*, 2017.

[30] J. Wei, X. Wang, D. Schuurmans, et al. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**. *NeurIPS*, 2022.

### H. Survival analysis (⭐ new in this paper, please re-verify)

[31] D. R. Cox. **Regression Models and Life-Tables**. *Journal of the Royal Statistical Society: Series B*, 1972, 34(2): 187-220.

[32] J. P. Fine, R. J. Gray. **A Proportional Hazards Model for the Subdistribution of a Competing Risk**. *Journal of the American Statistical Association*, 1999, 94(446): 496-509.

[33] J. D. Kalbfleisch, R. L. Prentice. **The Statistical Analysis of Failure Time Data** (2nd ed.). *Wiley*, 2002.

### I. Cross-dataset classifier comparison in educational data mining (⭐ new in this paper, please re-verify)

[34] R. Murali, C. Conati, D. Poole. **A Comparison of Real-Time User Classification Methods using Interaction Data for Open-ended Learning**. *Proceedings of the 18th International Conference on Educational Data Mining (EDM 2025)*, Palermo, Italy, 2025, pp. 7-18. DOI: 10.5281/zenodo.15870205. ⭐ 2025 (**uses the same CS1 MOOC dataset as this paper**; reports HMM accuracy 0.7697 / LSTM 0.7151 / FUMA 0.3465 / stratified dummy 0.5395)

---

## Appendix A · Reproduction

```bash
# data: stage_dataset.npz (2171 stage samples / 473 students / 258 dropout events)
python3 analysis/hazard_model/run_dchm.py        --data data/stage_dataset.npz --out results/dchm --tag run1
python3 analysis/hazard_model/run_shr.py         --data data/stage_dataset.npz --out results/shr  --tag run1 --models rf hgb
python3 analysis/hazard_model/run_early2.py      --data data/stage_dataset.npz --ks 1 2 3 4
python3 analysis/hazard_model/run_fusion_curve.py --data data/stage_dataset.npz --oof results/dchm/oof_run1.npz
python3 analysis/hazard_model/run_combo.py       --data data/stage_dataset.npz --oof results/dchm/oof_run1.npz --ks 1 2 3 4
```

**Protocol**: 5-fold student-grouped, label-stratified CV × 3 seeds (42/123/777), averaged OOF probabilities; main metric = part-wise pooled AUC.

## Appendix B · Outstanding items before submission

| Item | Description | Priority |
|---|---|---|
| Submission-quality validation | "Clicking exercise ids to farm submissions" contaminates the completion feature family | High |
| External dataset validation | At least one public programming-behaviour dataset | High |
| Calibration curves | ECE, reliability diagrams, per-part calibration | Medium |
| Sequence-encoder ablation | Gain of a lightweight encoder used as a feature extractor | Medium |
| Abstention band and intervention list | One-page product-facing output | Medium |
| Figures | Intervention curve, two failure-mode profiles, part-prior comparison | Medium |

## Appendix C · Author contributions and conflicts

**Author contributions.** Jian Wang conceived the study, re-formulated the problem as a stage-wise progression decision, designed and implemented CS1-SHR, ran all experiments and wrote the manuscript.

**Data availability.** This study uses the publicly released CS1 IDE action-log dataset [3].

**Conflict of interest.** The author declares no conflict of interest.
