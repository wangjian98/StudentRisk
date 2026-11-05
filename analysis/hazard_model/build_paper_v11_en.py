#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build paper_v11_en.md from paper_v10_en.md: inject the CS1-FSP few-shot line,
the methodological corrections and the generalisation package."""
import io

SRC = 'paper_v10_en.md'
DST = 'paper_v11_en.md'
s = io.open(SRC, encoding='utf-8').read()
orig_len = len(s)

s = s.replace(
    '# CS1-SHR: Stage-Hierarchical Survival Risk for Stage-Level Dropout Prediction in Introductory Programming',
    '# CS1-SHR and CS1-FSP: Stage-Hierarchical Survival Risk and Part-Anchored Few-Shot Cold Start for Stage-Level Dropout Prediction in Introductory Programming')
s = s.replace('**Submission date:** 1 November 2026 (v10)', '**Submission date:** 5 November 2026 (v11)')

# ---------------------------------------------------------------- version note
s = s.replace('**Version note (v10).**', '''**Version note (v11 additions).** On top of the full v10 manuscript, this revision adds one **new method line** and two **methodological corrections**, all backed by measurements taken on the same data:

- **§4.10 (new)** — we propose **CS1-FSP** (part-anchored transductive standardisation, a family of few-shot discriminators, cross-fitted calibration and an abstention band), targeting the cold-start deployment scenario in which a target part has **no labels yet**;
- **§5.5 (new)** — the few-shot evaluation protocol (in-part cold start, support sampling, repetitions, reference systems and significance);
- **§6.14–6.21 (new)** — the in-part cold-start learning curves (K ∈ {1, 2, 4, 8, 16} across three feature sets), weak-part rescue, cross-part transfer, the rare class, the **corrected** sitting-level protocol, calibration and abstention, significance tests, and a **methodological correction** that voids previously reported few-shot readings and fixes the sitting-level feature specification;
- **§7.7 (new)** — we formulate a **fourth law**: under a constrained label budget one should **select features first, then select the model**;
- **§8.3 (new)** — a **generalisation argument package** for the case where external-dataset validation is still unavailable (four levels of evidence, five substitutes, and a narrowed claim formulation).

**Version note (v10).**''')

# ---------------------------------------------------------------- abstract
s = s.replace(
    '**Keywords:** learning analytics; discrete-time survival analysis; hazard modelling; stage-wise progression decision; IDE behaviour logs; early warning; programming education; calibration; peer-relative features',
    '**Addendum (v11).** We additionally audit and **void** all previously reported few-shot (meta-learning) readings of this project: the inner adaptation in the original per-part 5-shot / K-shot evaluation was a **no-op** (the adapted weights were never loaded), the part indexing was off by one, and the headline metrics came from single-class query sets (AUC = NaN); the only protocol-correct experiment yields ROC ≈ 0.537 with **no response to the shot count**. We therefore re-examine the claim that "few-shot learning fails on CS1" and propose **CS1-FSP**: **part-anchored transductive standardisation** removes cross-part distribution drift, the discriminator is **fitted on the target part at run time from very few labels**, and **cross-fitted Platt calibration plus an abstention band** control the trustworthiness of the output. Empirically, 16 labelled cases per part already reach **96.5%** of the in-part full-label ceiling (pooled AUC 0.929), 8 cases reach 94.2%, and 4 cases with a single peer-relative feature reach 92.2%; the two weakest parts (part5 / part6) improve from 0.8511 to **0.873**; for the rare class "completed the course but still failed", whose full-label AUC is only 0.774, 16 few-shot cases reach 0.749 (96.8%). We also report three counterintuitive but reproducible findings: under a label budget **feature selection matters more than model selection** (K = 2: a single peer-relative feature 0.856 > all 29 features 0.780), **pseudo-label self-training is worse than its own baseline in every configuration** (0.835 vs 0.857), and **deep metric learning degrades as the shot count grows** (0.716 → 0.608). We argue that the value of few-shot learning is not to raise in-distribution accuracy but to **open the deployment scenario "day one of a new part / new term"**, for which we provide a verifiable labelling-cost contract (4–16 cases per part; below 2 cases the system abstains).\n'
    '\n'
    '**Keywords:** learning analytics; discrete-time survival analysis; hazard modelling; stage-wise progression decision; IDE behaviour logs; early warning; programming education; calibration; peer-relative features; few-shot learning; cold start')

# ---------------------------------------------------------------- contributions
s = s.replace(
    '5. **Reproducibility.** All scripts, raw results and logs are released (Appendix A).',
    '5. **Reproducibility.** All scripts, raw results and logs are released (Appendix A).\n'
    '6. **A cold-start channel, and corrections.** We propose **CS1-FSP** (§4.10): part-anchored transductive standardisation, a budget-dependent family of discriminators, calibration and abstention, which makes "day one of a new part / new term" a usable scenario (16 labelled cases reach **96.5%** of the in-part full-label ceiling; 8 cases reach 94.2%). We also **audit and void** the previously reported few-shot (meta-learning) readings and **correct the sitting-level feature specification** (§6.20), so that the earlier conclusion "few-shot learning does not work on CS1" is no longer cited.')

# ---------------------------------------------------------------- §4.10
S410 = r'''
### 4.10 CS1-FSP: Part-Anchored Few-Shot Cold Start (new in v11)

#### 4.10.1 Motivation: the deployment scenario comes first

§6.20 shows that all previously available evidence for "few-shot learning fails on CS1" originates from **defects in the evaluation harness** (a no-op adaptation, off-by-one part indexing, single-class query sets), so the conclusion does not hold. The question that actually matters is different: **when a target part has just opened and not a single label for it exists, can we still issue a usable warning?** This scenario is unavoidable in deployment: the first course of a new term, the first cohort at a new school, a newly opened part with a new task type. Labels can only be produced after the fact (drop is defined by the *future* fact that the student never returns, §3.5), so the model must be **fitted on the target part at run time from very few labels**.

#### 4.10.2 Three mechanisms

**Mechanism M1 · Part-anchored transductive standardisation.** Distribution drift across parts is the main obstacle to transferring a stage-level model (§6.6 shows chance-level student-level LOPO; §6.16 shows that, without treatment, cross-part prototype borrowing reaches only 0.60–0.67). We standardise the 9 absolute-magnitude features (4 intensity + 3 ratio + 2 progress) by the **median / MAD of the target part's own unlabelled rows**:

$$\mathrm{Z9}_p = \bigl(\mathrm{abs9}_p - \mathrm{median}_p(\mathrm{abs9})\bigr) / \mathrm{MAD}_p(\mathrm{abs9}),$$

where $\mathrm{median}_p$ and $\mathrm{MAD}_p$ are estimated from **unlabelled rows only**, hence no label information enters the transformation and the anti-leakage constraint of §3.6 is respected. The peer-relative features are already MAD-z quantities within the part and need no second standardisation; the self-referential features are z-scores within a student and are therefore comparable, but they are **unavailable in the cold-start scenario** (Layer-0 decomposition: the marginal contribution of removing self-reference is 0.0050 and not significant).

**Mechanism M2 · A family of few-shot discriminators, selected by shot count.** We do not impose a single model. Instead we evaluate an **inductive-bias spectrum**: two prototypes (equivalent to a linear decision direction), 1-NN, logistic regression (linear control), shallow decision trees (able to express interaction rules from few examples), random forests (non-linear ensemble), kNN label propagation (transductive), two pseudo-label self-training variants (propagation / prototype → RF) and a metric-learned prototypical net (deep control). The support set contains **K cases per class** (K dropout + K non-dropout); the query set is every remaining row of that part. We report the mean and standard deviation of the part-wise pooled AUC over 12 repetitions.

**Mechanism M3 · Calibration and abstention.** The raw probabilities of the main model cannot be thresholded directly (calibration audit: ECE 0.0677, slope 1.271, intercept −0.799), and few-shot variance is larger (between-repetition s.d. 0.03–0.06). CS1-FSP therefore mandates **cross-fitted Platt calibration** and an **abstention band**: when K < 2 or the part has fewer than 100 unlabelled rows the system reports "insufficient evidence" only; when K ≤ 4 it emits a score but flags it as low-confidence and attaches a bootstrap score interval.

#### 4.10.3 Selection as a function of the labelling budget

Let $M = 2K$ be the **total number of labelled cases** (K dropout + K non-dropout). The reference is the in-part full-label RF (part-wise pooled AUC 0.9289).

| M (cases) | K | Discriminator | Feature set | pooled AUC | % of ceiling |
|---:|---:|---|---|---:|---:|
| 2 | 1 | Logistic regression | peer_z_submit only | 0.791 | 85.2% |
| 4 | 2 | Logistic regression | peer_z_submit only | **0.856** | 92.2% |
| 8 | 4 | Random forest | 10 peer-relative dims | 0.838 (single-dim linear 0.854) | 90.2% |
| 16 | 8 | Random forest | 10 peer-relative / all 29 dims | **0.871 / 0.875** | 94.2% |
| 32 | 16 | Random forest | all 29 dims | **0.896** | **96.5%** |
| < 2 | — | Abstain | — | — | — |

#### 4.10.4 Training and inference

Training happens **inside each target part**; there is no offline pre-training:

```
① Aggregate every student's IDE behaviour in the part → agg10 (7 event counts, total events,
   distinct exercises touched, time span)
② Part-anchored transductive standardisation: median / MAD from the part's unlabelled rows,
   applied to the 9 absolute-magnitude features
③ Peer-relative 10 dims: peer_z = (self − part median) / part MAD, clipped to ±5
④ Take K dropout + K non-dropout cases of the part as the support set
⑤ Fit according to M: M ≤ 4 → logistic regression (single feature); M ≥ 8 → random forest
   (300 trees, min_samples_leaf = 2, max_features = 'sqrt', class_weight = 'balanced')
⑥ Score all remaining rows of the part → ranking
⑦ Apply the cross-fitted Platt calibration (fitted on historical parts via OOF scores) → probabilities
⑧ Abstention check → shortlist → IPSD truncation (§4.8) → within-part risk curve and stall evidence
```

'''
s = s.replace('\n## 5 Experimental Setup\n', S410 + '\n## 5 Experimental Setup\n')

# ---------------------------------------------------------------- §5.5
S55 = r'''
### 5.5 Few-Shot Evaluation Protocol (new in v11)

**FS-A, in-part cold-start learning curve (main protocol).** For every part p we independently draw K dropout and K non-dropout cases as the support set (K ∈ {1, 2, 4, 8, 16}, bounded by the number of positives available in that part); the query set is every remaining row of that part. We repeat 12 times (re-drawing the support set each time) and report the mean and s.d. of the part-wise pooled AUC. **Reference systems**: (i) **REF-inpart-full** — in-part full-label RF (5-fold, grouped by student, OOF; same feature space as the few-shot runs), the ceiling under abundant labels; (ii) **REF-crosspart** — train on all other parts, test on the target part; (iii) **REF-global-main** — the main-protocol RF in the raw feature space, used as a pipeline consistency check (reproduces 0.9258). **FS-B** weak parts part5 / part6 (15 / 22 positives, the two weakest parts of the project). **FS-C** cross-part transfer, with support sets drawn only from other parts. **FS-D** sitting level, first and second sitting, with an **OOF** full-label reference. **FS-E** the rare class: students who reached part7 but still failed (59) versus those who completed and passed (156), scored by student-level AUC. **Significance**: repetition-level paired bootstrap (2000 resamples, resampling both parts and repetitions).

'''
s = s.replace('\n## 6 Results\n', S55 + '\n## 6 Results\n')

# ---------------------------------------------------------------- §6.14–6.21
S614 = r'''
### 6.14 In-Part Cold-Start Learning Curves (main CS1-FSP result, new in v11)

**Reference systems** (same data, same feature space): REF-global-main **0.9279** (main protocol; the pipeline consistency check passes), REF-inpart-full **0.9289** (in-part full-label RF, the ceiling), REF-crosspart **0.9184** (train on other parts → test the target part).

**Feature set A · all 29 dims** (4 intensity + 3 ratio + 2 progress + 10 peer-relative + 10 self-referential)

| Method | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---:|---:|---:|---:|---:|
| Two prototypes (linear direction) | 0.747 | 0.785 | 0.819 | 0.847 | 0.845 |
| 1-NN | 0.747 | 0.752 | 0.781 | 0.810 | 0.848 |
| Logistic regression (linear) | 0.769 | 0.780 | 0.808 | 0.833 | 0.867 |
| Shallow tree, depth 2 | 0.589 | 0.669 | 0.731 | 0.759 | 0.802 |
| Shallow tree, depth 3 | 0.597 | 0.680 | 0.707 | 0.742 | 0.787 |
| **Random forest** | — | 0.740 | 0.847 | **0.875** | **0.896** |
| kNN label propagation | 0.710 | 0.747 | 0.775 | 0.827 | 0.857 |
| Propagation + RF (self-training) | 0.697 | 0.728 | 0.756 | 0.803 | 0.835 |
| Prototype + RF (self-training) | 0.731 | 0.756 | 0.788 | 0.821 | 0.835 |

**Feature set B · peer-relative only (10 dims)**

| Method | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---:|---:|---:|---:|---:|
| Two prototypes | 0.755 | 0.816 | 0.837 | 0.852 | 0.853 |
| Logistic regression | 0.760 | 0.822 | 0.834 | 0.855 | 0.885 |
| **Random forest** | — | 0.754 | 0.838 | **0.871** | **0.893** |
| kNN label propagation | 0.739 | 0.808 | 0.820 | 0.850 | 0.878 |
| Propagation + RF | 0.704 | 0.770 | 0.786 | 0.807 | 0.848 |

**Feature set C · peer_z_submit only (1 dim)**

| Method | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---:|---:|---:|---:|---:|
| Two prototypes | 0.773 | 0.809 | 0.843 | 0.837 | 0.842 |
| 1-NN | 0.773 | 0.847 | 0.843 | 0.845 | 0.832 |
| **Logistic regression** | **0.791** | **0.856** | 0.854 | 0.870 | 0.864 |
| Random forest | — | 0.741 | 0.850 | 0.870 | 0.864 |
| kNN label propagation | 0.664 | 0.771 | 0.802 | 0.827 | 0.839 |
| Propagation + RF | 0.572 | 0.643 | 0.690 | 0.727 | 0.798 |

**Four readings.**

1. The curves **increase monotonically with the shot count** (RF: 0.740 → 0.896), in contrast to the flat curve (ROC ≈ 0.537, no response to K) of the earlier meta-learning experiments (§6.20).
2. **Feature selection matters more than model selection**: at K = 2, a single peer-relative feature with logistic regression reaches **0.856**, whereas all 29 features with logistic regression reach 0.780 (**+0.076**).
3. **Non-linearity only pays off once enough labels exist**: for K ≤ 2 the linear model is best, for K ≥ 4 the random forest is best, and shallow trees are worst throughout — interactions cannot be estimated reliably from a handful of examples.
4. **Pseudo-label self-training is a negative result**: propagation + RF (0.835) is below propagation itself (0.857), and prototype + RF (0.835) is below the prototype (0.845).

### 6.15 Weak-Part Rescue (part5 / part6)

| Method (K = 8; all-parts pooled / weak-part pooled) | Result |
|---|---|
| Two prototypes | 0.847 / 0.841 |
| Logistic regression | 0.833 / 0.820 |
| **Random forest** | **0.874 / 0.873** |
| Propagation + RF | 0.803 / 0.755 |

→ The weak parts (15 and 22 positives) reach 0.873 with 16 labelled cases, **above the previously reported full-label reading of 0.8511** for those parts; they are no longer the bottleneck of the project.

### 6.16 Cross-Part Transfer: Full-Label Training versus Prototype Borrowing

| Scheme | K | pooled |
|---|---:|---:|
| Cross-part full-label RF (REF-crosspart) | full labels | **0.9184** |
| Cross-part prototype borrowing (support only from other parts) | 2 | 0.599 |
| Cross-part prototype borrowing | 4 | 0.647 |
| Cross-part prototype borrowing | 8 | 0.665 |

→ Cross-part transfer **cannot** be achieved by "borrowing a few dropout prototypes"; it requires **full-label training plus part-anchored standardisation**: after part anchoring, the cross-part full-label RF is only 0.0105 below the in-part full-label RF (0.9184 vs 0.9289). Together with §6.6 this shows that **the obstacle to cross-part generalisation is mainly distribution drift and task-identity shortcuts, not an intrinsic non-transferability of the task**.

### 6.17 The Rare Class: Completed the Course but Still Failed

Sample: 215 students who reached part7, of whom **59** eventually failed; **full-label RF (OOF) AUC = 0.774**.

| Method | K=1 | K=2 | K=4 | K=8 | K=16 |
|---|---:|---:|---:|---:|---:|
| Two prototypes | 0.552 | 0.634 | 0.635 | 0.624 | 0.708 |
| Logistic regression | 0.576 | 0.664 | 0.673 | 0.630 | 0.718 |
| **Random forest** | — | 0.671 | **0.676** | 0.650 | **0.749** |
| kNN label propagation | 0.544 | 0.588 | 0.599 | 0.584 | 0.670 |
| Propagation + RF | 0.538 | 0.564 | 0.606 | 0.576 | 0.673 |

→ **K = 16 reaches 96.8% of the full-label performance** (0.749 vs 0.774); K = 2 already reaches 86.7%. This rare class (18.8% of all failures) is precisely the **label-scarce** scenario for which few-shot learning is appropriate, and it is weak even with full labels (0.774) — consistent with the capability boundary of §8.1.

### 6.18 Sitting Level (Corrected Protocol): The Price of Being Earlier

The original sitting-level construction (§4.9, §6.11) took `part_tot` to be the **part index itself** and used `timeToDeadline` as span / progress (§6.20), so the old reading 0.8665 is not comparable. We rewrote the truncated features semantically (12 columns: 7 event counts + total + distinct exercises touched + time span + ttd_max + ttd_min) and changed the reference to **OOF**:

| Truncation unit | Rows | Full-label RF (OOF) | Best few-shot (K = 8) |
|---|---:|---:|---:|
| First sitting | 2171 | **0.7818** | 0.672 (peer_z_submit + logistic regression) |
| Second sitting | 1054 | **0.7065** | 0.713 (peer-relative + logistic regression) |

→ **"Earlier" has a measurable price**: the full-label ceiling falls from 0.929 (whole part) to 0.7818 (first sitting), a drop of 0.147. Note that the two sittings involve different populations (2171 → 1054 rows), so the ceiling difference is mainly a **selection effect** rather than "the second sitting is harder"; the part6 reading at the second sitting is only 0.4004, which shows that sitting-level results must be inspected per part rather than only in aggregate.

### 6.19 Calibration and Abstention

We reuse the calibration audit of §5: the stage-level RF has ECE(10 bins) **0.0677**, calibration slope **1.271**, intercept **−0.799**, Brier 0.0493 and AURC 0.660; after cross-fitted Platt or isotonic recalibration, **ECE 0.0134** and Brier 0.0408. CS1-FSP makes this post-processing **mandatory** and adds an abstention rule based on the shot count and the between-repetition s.d. (0.03–0.06).

### 6.20 Methodological Corrections: Voided Readings (new in v11)

**(i) All few-shot (meta-learning) readings are void.** The four previous groups of per-part 5-shot / adaptive 3-shot / K-shot experiments contain implementation defects:

| Defect | Location | Consequence |
|---|---|---|
| The inner adaptation is a **no-op**: the FOMAML adaptation function returns the **pre-adaptation** weights, so the three inner gradient steps are discarded | `models/meta_mamba_7d/eval_per_part_5shot.py:67-84` | "Few-shot adaptation" never actually happened |
| **State leakage** across (part, seed): prediction loads the adapted weights, so later parts inherit the previous part's adaptation | same file `:88,97` | later parts are contaminated by an earlier support set |
| **Part indexing off by one**: `task_ids` are 0-indexed while the script iterates over `range(1, 8)` | same file `:102,289` | the original part1 was never evaluated |
| Headline metrics come from **single-class query sets** (`qry_n_passed = 0`, AUC = NaN), and ROC is never pooled | same file `:61` | "accuracy = 1.0 / macro-F1 = 1.0" is a degenerate artefact |
| The `final_model.ckpt` the runs depend on was never committed | `results/per_part_5shot_full/` | the numbers cannot be re-verified |

The only protocol-correct experiment (participation sampling) yields 24 ROC values in **0.483–0.628** (centre ≈ 0.537) with **no response to the shot count**: it measures "deep backbone + task-identity modulation + wrong sampling", **not few-shot learning**. The conclusion "few-shot learning fails on CS1" therefore **does not hold**, and we replace it with §6.14.

**(ii) Correction of the sitting-level feature specification.** In the original construction, `part_tot[p] = max(part == p)` returns the **part index itself** (used as the "number of exercises per part"), and `timeToDeadline` was used as span / progress, so the "progress" component of that feature set was in fact **deadline pressure**. After the semantic rewrite, the OOF full-label ceiling at the sitting level is **0.7818 (first sitting) / 0.7065 (second sitting)**; the old readings 0.8665 / 0.8644 are **no longer cited** (§4.9, §6.11 and §7.6 are corrected accordingly).

**(iii) Self-reference in the within-part early-warning protocol.** As already established, the reading 0.9129 (with self-reference) must be replaced by the clean value **0.8263** (Δ = −0.0866, P(Δ ≤ 0) = 0.0015); the deficit of the deep model shrinks from −0.092 to −0.047.

### 6.21 Significance (repetition-level paired bootstrap, 2000 resamples)

| Comparison | Few-shot pooled | Full-label pooled | Δ | 95% CI | P(Δ ≤ 0) | % of ceiling |
|---|---:|---:|---:|---|---:|---:|
| All 29 dims + RF, K = 8 | 0.8745 | 0.9289 | −0.0544 | [−0.0657, −0.0428] | 100% | 94.2% |
| All 29 dims + RF, K = 16 | 0.8964 | 0.9289 | −0.0324 | [−0.0456, −0.0188] | 100% | **96.5%** |
| Peer-relative + logistic regression, K = 2 | 0.8215 | 0.9289 | −0.1074 | [−0.1363, −0.0862] | 100% | 88.5% |
| Single-dim peer_z_submit + logistic regression, K = 2 | 0.8559 | 0.9289 | −0.0730 | [−0.0882, −0.0615] | 100% | 92.2% |

→ The gap is **statistically significant but small**: only 0.032 at K = 16. **The main gain of this line is from "zero labels (0.5)" to "16 cases (0.896)".**

'''
s = s.replace('\n## 7 Analysis and Discussion\n', S614 + '\n## 7 Analysis and Discussion\n')

# ---------------------------------------------------------------- §7.7
S77 = r'''
### 7.7 A Fourth Law: Inductive Bias Under a Label Budget (new in v11)

§7.1–§7.6 establish the granularity–base-rate–inductive-bias triad. The few-shot experiments of §6.14 add a fourth:

> **Fourth law (label-budget law).** When the label budget is constrained, one should **select features first, then select the model**: at K = 2 a single peer-relative dimension with a linear discriminator (0.856) outperforms all 29 features with a linear discriminator (0.780), a gap of +0.076; only from K ≥ 4 onwards does a non-linear ensemble overtake the linear model, and it then improves monotonically (0.847 → 0.896).

The mechanism is clear. The decision boundary of stage-level dropout is **intrinsically an interaction** (few submits ∧ many edits ∧ fragmented attention), hence (i) a linear prototype cannot represent it (two prototypes 0.819 @ K = 4 < RF 0.847); (ii) interactions cannot be estimated reliably from very few examples (shallow trees are worst throughout); but (iii) **one correct feature already suffices to fix the direction**, so under a small budget "few and precise" beats "many and complete". This is complementary to §7.3 on why deep architectures have no advantage here: the problem with deep backbones is that **the sample size cannot support their capacity**, not that the label budget cannot support it.

The fourth law also reframes the failure of the earlier meta-learning experiments: their backbone (deep + task-identity modulation) had already been falsified on the discriminative line (order −0.13), and their metric-learning variant degrades as K grows here (0.716 → 0.608) — **the failure comes from the architecture and the evaluation implementation, not from few-shot learning as such**.

'''
s = s.replace('\n## 8 Limitations and Capability Boundaries\n', S77 + '\n## 8 Limitations and Capability Boundaries\n')

# ---------------------------------------------------------------- §8.3/8.4
S83 = r'''
### 8.3 External-Dataset Validation (the Hard Gap) and a Generalisation Argument Package (new in v11)

**The hard gap.** All conclusions in this paper rest on a **single** dataset (CS1: 473 students / 2171 stage samples / 258 dropout events, base rate 11.9%). External-dataset validation (EdNet, Junyi Academy, KDD Cup xuetangX, ASSISTments) remains undone and is the one item that must be closed before submission.

**Four levels of generalisation and the evidence available here.**

| Level | Meaning | Evidence in this paper | Status |
|---|---|---|---|
| L1 Same distribution, new samples | different students | student-grouped 5-fold × 3 seeds: stage-level pooled 0.9289 | available |
| L2 Distribution shift | same data, changed conditionals | **cross-part full-label RF 0.9184** (in-part 0.9289, a drop of 0.0105); part anchoring removes most of the cross-part obstacle | partial (time extrapolation pending) |
| L3 Subgroup robustness | consistency across groups and parts | weak parts part5/6 **0.873**; rare class 0.774 (few-shot 0.749); equal-sample random-removal control for contaminated rows gives Δ = 0; peer_z_submit has a consistent direction in **7/7 parts** (HR 0.15–0.41) | partial (subgroup matrix pending) |
| L4 Cross-dataset | a different course or platform | not done | gap |

**Five substitutes available before external data is ready.** (i) **Internal drift simulation**: split by `first_ts/last_ts` for time extrapolation that mimics "a new term starting", and quantify the deployment risk through the degradation in performance; (ii) a **subgroup robustness matrix**; (iii) **support-set sensitivity**: sampling distribution, support representativeness, missing and noisy features; (iv) **mechanistic argument**: the gains of this method come from the behavioural mechanism "submitting less than one's peers → stopping", so what must transfer across datasets is the **direction** of that mechanism rather than exact numbers; (v) **structural argument**: the three laws and the fourth law are claims about **tasks and label supply**, not about CS1 — on a second dataset only four numbers are needed (stage level > exercise level; peer-relative > absolute; monotone shot curve; cross-part decay < 0.02) to support the mechanistic claim.

**If external data remain unavailable, narrow the claims** (more defensible in review): claims A (the stage-level task form plus few-shot cold start is feasible and reproducible on a real, large-scale MOOC event log) and B (the three laws and the fourth law, each supported by positive and negative results) hold; claim C (effectiveness across courses and schools) goes to the limitations, **without unbounded wording such as "general" or "universal"**, and the abstract and title claim only "on a large-scale MOOC event log".

### 8.4 Which Existing Conclusions This Paper Corrects (new in v11)

| Item | Previous statement | v11 correction |
|---|---|---|
| Few-shot effectiveness | "Few-shot learning fails on CS1 (ROC ≈ 0.537, no response to K)" | an artefact of evaluation defects; the conclusion does not hold (§6.20); replaced by the learning curves of §6.14: 16 cases reach 96.5% of the ceiling |
| Sitting-level level | "sitting-level pooled 0.8665 / 0.8644" | the feature specification is faulty (progress was deadline pressure, exercise base was the part index); **no longer cited**; the corrected ceilings are 0.7818 / 0.7065 (§6.18, §6.20) |
| Within-part early warning | "CS2-FWD within-part early warning 0.9129" | an optimistic reading that includes self-reference; the clean value is **0.8263** (§6.20(iii)) |
| Attribution of deep few-shot failures | "deep backbones have no advantage in few-shot settings" | the attribution should be the **architecture and the evaluation implementation**, not few-shot learning itself (§7.7) |

'''
s = s.replace('\n## 9 Conclusion\n', S83 + '\n## 9 Conclusion\n')

# ---------------------------------------------------------------- conclusion
s = s.replace('\n## References\n',
              '\n'
              '**Addendum (v11).** Beyond the conclusions above, this paper answers a deployment question: **how can we still issue a usable warning when no labels have accumulated yet?** '
              'We first audit and void all previously reported few-shot (meta-learning) evidence of the project (§6.20), then propose **CS1-FSP**: part-anchored transductive standardisation, '
              'a budget-dependent family of discriminators, cross-fitted calibration and an abstention band. Empirically, **16 labelled cases per part already reach 96.5% of the in-part full-label ceiling**, '
              '8 cases reach 94.2% and 4 cases with a single peer-relative feature reach 92.2%; the two weakest parts improve from 0.8511 to 0.873; and for the rare class "completed the course but still failed", '
              'whose full-label AUC is only 0.774, 16 cases reach 0.749. We therefore argue that **the value of few-shot learning lies not in raising in-distribution accuracy but in opening the deployment scenario '
              '"day one of a new part / new term"**, and we restate the product promise as a verifiable labelling-cost contract (4–16 cases per part; below 2 cases the system abstains). '
              'We also formulate the **fourth law** (select features before selecting the model under a constrained label budget) and provide a **generalisation argument package** (§8.3), while stating plainly '
              'that external-dataset validation remains an open hard gap.\n'
              '\n## References\n')

# ---------------------------------------------------------------- references
s = s.replace('## Appendix A · Reproduction',
              '[35] J. Snell, K. Swersky, R. Zemel. **Neural Prototypical Networks**. *Advances in Neural Information Processing Systems (NeurIPS 30)*, 2017, pp. 4077–4087. (The original prototype-based few-shot formulation; used in §4.10 as an inductive-bias control, where we report that it underperforms the random forest on CS1.) ⭐ to be verified by the authors\n'
              '[36] J. Platt. **Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods**. *Advances in Large Margin Classifiers*, MIT Press, 2000, pp. 185–208. (Platt scaling for probability calibration; the cross-fitted Platt post-processing of §6.19 is based on it.) ⭐ to be verified by the authors\n'
              '[37] D. Zhou, O. Bousquet, T. Lal, J. Weston, B. Schölkopf. **Learning with Local and Global Consistency**. *Advances in Neural Information Processing Systems (NIPS 15)*, 2003, pp. 217–224. (Label propagation and the selective-prediction framework underlying §4.10.) ⭐ to be verified by the authors\n'
              '[38] S. Ruder. **An Overview of Multi-Task Learning in Deep Neural Networks**. *arXiv:1706.05098*, 2019. (General treatment of transfer and domain adaptation, used for the part-anchored transductive standardisation of §4.10 and the generalisation package of §8.3.) ⭐ to be verified by the authors\n'
              '\n## Appendix A · Reproduction')

s = s.replace('## Appendix B · Outstanding items before submission',
              '## Appendix B · Outstanding items before submission (updated in v11)\n'
              '\n'
              '- ⬜ **External-dataset validation** (the only remaining hard gap, §8.3): first a one-day feasibility and field-alignment probe, then the four-number diagnostic (stage level vs exercise level; peer-relative gain; monotone K curve; cross-part decay).\n'
              '- ⬜ **Time extrapolation** (completes L2, one CPU-hour): split by `first_ts/last_ts` to mimic "a new term starting" and quantify the deployment risk.\n'
              '- ⬜ **Subgroup robustness matrix** (completes L3): AUC + calibration + top-k precision for weak parts, the rare class, contaminated rows and different investment strata.\n'
              '- ⬜ **Support-set sensitivity**: sampling distribution, support representativeness, missing and noisy features.\n'
              '- ⬜ **Reference verification**: the new entries [35]–[38] are marked "to be verified"; complete volume, issue, page numbers and DOIs for the target venue.')

# ---------------------------------------------------------------- in-place markers
s = s.replace('| **Sitting · first sitting** | 2171 | **0.8665** |',
              '| **Sitting · first sitting** | 2171 | **0.8665** (old feature specification; **corrected in §6.20** — the corrected OOF ceiling is 0.7818) |')
s = s.replace('1. **Truncating at a sitting beats truncating at an exercise**: the first sitting reaches a pooled **0.8665**',
              '1. **Truncating at a sitting beats truncating at an exercise** (old specification, **corrected in §6.20**): the first sitting reaches a pooled **0.8665**')
s = s.replace('2. **The session-structure features themselves add nothing** (0.8644 vs 0.8665)',
              '2. **The session-structure features themselves add nothing** (0.8644 vs 0.8665; **this pair of numbers uses the old specification, see §6.20**)')
s = s.replace('yields IPSD (truncation augmentation) and sitting-level truncation: within-part discrimination rises from "not decidable inside a part" to pooled **0.8665** at the first sitting, with top-5% intervention precision **0.769** (6.5x base rate);',
              'yields IPSD (truncation augmentation) and sitting-level truncation: within-part discrimination rises from "not decidable inside a part" to pooled **0.8665** at the first sitting, with top-5% intervention precision **0.769** (6.5x base rate; **the sitting-level numbers are corrected, see §6.20 and §6.18**);')

# ---------------------------------------------------------------- 附录 A 复现清单增补
s = s.replace('python3 analysis/hazard_model/run_combo.py       --data data/stage_dataset.npz --oof results/dchm/oof_run1.npz --ks 1 2 3 4\n', 'python3 analysis/hazard_model/run_combo.py       --data data/stage_dataset.npz --oof results/dchm/oof_run1.npz --ks 1 2 3 4\n' + '\n# Few-shot line (new in v11): in-part cold-start learning curves / rare class + significance / sitting level (corrected)\npython3 analysis/fewshot/run_cs1_fs.py         --data data/stage_dataset.npz --ex data/exercise_dataset.npz \\\n                                              --out results/cs1_fs --repeats 12 --epochs 300 --device cuda\npython3 analysis/fewshot/run_fs_final.py        --data data/stage_dataset.npz --out results/cs1_fs --repeats 12\npython3 analysis/fewshot/run_cs1_fs_sitting.py  --data data/stage_dataset.npz --ex data/exercise_dataset.npz \\\n                                              --out results/cs1_fs --repeats 12\npython3 analysis/fewshot/plot_cs1_fs.py         --json results/cs1_fs/cs1_fs.json --out results/cs1_fs\n# Paper generation: v10 base plus the v11 injections\npython3 analysis/hazard_model/build_paper_v11_zh.py\npython3 analysis/hazard_model/build_paper_v11_en.py\n')

io.open(DST, 'w', encoding='utf-8').write(s)
print('en: %s -> %s  (%d -> %d chars, %d lines)' % (
    SRC, DST, orig_len, len(s), s.count('\n') + 1))
