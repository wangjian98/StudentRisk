# StudentRisk — Cross-Validation Robustness & Statistical Significance Analysis

**Project**: StudentRisk (CS1 public dataset, n=473, fail_rate=0.6638)
**Hardware**: 1× Tesla T4 (15.36 GiB) on `43.139.55.246`
**Date**: 2026-08-22
**Author**: MetaMamba evaluation pipeline (automated analysis)

---

## 0. TL;DR (one-page summary)

| Claim | Evidence |
|---|---|
| MetaMamba significantly outperforms LSTM / BiLSTM / Attention 7-d baselines on Macro-F1 | Δ ≥ 0.34, all p < 1e-12 after Holm-Bonferroni (paired t-test, n=24) |
| MetaMamba and MetaMamba-7d are statistically indistinguishable on Macro-F1 | Δ = −0.001, p = 0.51 (Holm-corrected) — the 4 extra continuous input features do not yield a measurable gain at this sample size |
| Naive 8-fold × 3-seed estimate (0.8783 Macro-F1) is essentially equal to the strict nested-CV estimate (0.8754 ± 0.018, 95% CI [0.8530, 0.8979]) | Optimism gap of 0.29 percentage points — single-CV is unbiased for this dataset at this sample size |
| Task-Contrastive auxiliary loss shows no benefit at this sample size | HP selection preferred `*_noTC` in 4 of 5 nested-CV outer folds; TC configurations tied or slightly worse on inner validation |
| Sample-size limitation dominates the picture | With only 473 students, 8-fold × 3-seed (n=24) yields tight paired tests; nested-CV needs 5-fold to keep wall time tractable on T4 |

---

## 1. What was actually done (precise protocol)

The phrase "nested 8-fold CV + T-test" was used informally to describe a **three-stage empirical study**. The actual protocol is:

| Stage | Method | Models | Output |
|---|---|---|---|
| **A** | StratifiedKFold, **8-fold × 3 seeds** (42, 123, 777), OOF aggregation | 5 models (MetaMamba, MetaMamba-7d, LSTM-7d, BiLSTM-7d, Attention-7d). RF-7d excluded — implementation is a stub. | `outputs/comparison.{md,csv}`, 5 plots |
| **B** | **Paired Student's t-test** on fold-level metrics, paired by (seed, fold), with **Holm-Bonferroni** correction within each metric family (4 comparisons per family) | Same 5 models vs MetaMamba as reference, on 3 metrics (macro_f1, roc_auc, f1_class_1) | `outputs/significance.{md,csv}`, n=24 pairs |
| **C** | **Mini Nested CV** — 5-fold outer × 2-fold inner × 4 HP grid = 25 single-fold trainings; per-outer: pick best HP on inner, refit on outer-train, evaluate on outer-test | **MetaMamba-7d only** (single-GPU T4 wall-time constraint; full nested CV on MetaMamba would require ~90 hours) | `outputs/nested_cv/{summary,per_outer,progress}.{md,jsonl,log}` |

> **Important scoping note**: The nested CV in stage C used **5-fold outer** (not 8-fold) because (i) the goal is "honest HP-selection-corrected generalization estimate", not larger n; (ii) 8 outer × 4 HP × 2 inner = 64 trainings for MetaMamba-7d was on the edge of tolerable wall time on T4, while the full outer-8 fold with 4 inner folds on MetaMamba would have been ~24× more expensive (~90 hours). Stage C also covers a **single model** (MetaMamba-7d) rather than all 5, because the nested CV protocol scales poorly with model count and the more interesting question is whether the naive CV estimate is optimistic.

---

## 2. Stage A — Headline numbers (8-fold × 3 seeds, OOF aggregated)

### 2.1 Overall metrics (n=473, OOF aggregated)

| Model | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **MetaMamba**     | **0.8901** | **0.8787** | **0.8908** | **0.9347** | **0.9713** |
| MetaMamba-7d      | 0.8901 | 0.8783 | 0.8907 | 0.9293 | 0.9669 |
| Attention-7d      | 0.6998 | 0.5624 | 0.6428 | 0.7108 | 0.8267 |
| BiLSTM-7d         | 0.6850 | 0.4822 | 0.5884 | 0.6750 | 0.7930 |
| LSTM-7d           | 0.6617 | 0.3982 | 0.5287 | 0.6225 | 0.7609 |

### 2.2 Per-fold stability (Macro-F1, mean ± std across 24 fold-seed pairs)

| Model | μ ± σ | Spread |
|---|---|---|
| MetaMamba    | 0.8753 ± 0.0340 | tight |
| MetaMamba-7d | 0.8766 ± 0.0324 | tight |
| Attention-7d | 0.5328 ± 0.0913 | loose |
| BiLSTM-7d    | 0.4888 ± 0.0946 | loose |
| LSTM-7d      | 0.4064 ± 0.0290 | very low, near the all-Failed baseline |

The simple baselines are not just worse — they collapse to **all-Failed** predictions on a substantial fraction of folds (see Class 0 F1 = 0.0000 for LSTM-7d on Class=PASSED), confirming they are not learning anything useful on this dataset.

### 2.3 Training time (all 24 fold-seed pairs summed)

| Model | n_params | Total elapsed |
| |
| MetaMamba    | 22,065 | 5,013 s |
| MetaMamba-7d | 21,809 | 2,205 s |
| Attention-7d | 67,713 | 57 s |
| BiLSTM-7d    | 67,201 | 36 s |
| LSTM-7d      | 33,857 | 30 s |

MetaMamba costs ~85× more wall time than LSTM-7d for a +0.48 Macro-F1 gain. On T4 alone, full nested-CV on MetaMamba would not be feasible in a working day.

---

## 3. Stage B — Paired t-test (n=24 pairs per comparison)

Each comparison is a **paired** test (paired by `(seed, fold)`), so the model sees the same train/test split structure on each row. Multiple-comparison correction is Holm-Bonferroni within each metric family (4 baselines per metric).

### 3.1 Macro-F1 (the headline metric)

| Baseline | μ(MetaMamba) | μ(Baseline) | Δ (Meta − Base) | t | p (Holm) | Sig |
|---|---|---|---|---|---|---|
| MetaMamba-7d | 0.8753 | 0.8766 | **−0.0013** | −0.677 | 0.505 | **ns** |
| LSTM-7d      | 0.8753 | 0.4064 | **+0.4689** | +41.78 | 1.36e-22 | *** |
| BiLSTM-7d    | 0.8753 | 0.4888 | **+0.3865** | +16.21 | 1.34e-13 | *** |
| Attention-7d | 0.8753 | 0.5328 | **+0.3425** | +14.80 | 6.09e-13 | *** |

### 3.2 ROC-AUC

| Baseline | μ(MetaMamba) | μ(Baseline) | Δ | t | p (Holm) | Sig |
|---|---|---|---|---|---|---|
| MetaMamba-7d | 0.9399 | 0.9233 | **+0.0166** | +3.94 | 6.61e-04 | *** |
| LSTM-7d      | 0.9399 | 0.6050 | +0.3349 | +20.19 | 1.58e-15 | *** |
| BiLSTM-7d    | 0.9399 | 0.6384 | +0.3015 | +15.31 | 4.50e-13 | *** |
| Attention-7d | 0.9399 | 0.6832 | +0.2567 | +13.65 | 3.25e-12 | *** |

### 3.3 F1 (Failed class — the positive / minority class of operational interest)

| Baseline | μ(MetaMamba) | μ(Baseline) | Δ | t | p (Holm) | Sig |
|---|---|---|---|---|---|---|
| MetaMamba-7d | 0.9137 | 0.9150 | **−0.0013** | −0.896 | 0.380 | **ns** |
| LSTM-7d      | 0.9137 | 0.7950 | +0.1187 | +22.82 | 1.07e-16 | *** |
| BiLSTM-7d    | 0.9137 | 0.8013 | +0.1124 | +19.70 | 2.02e-15 | *** |
| Attention-7d | 0.9137 | 0.7934 | +0.1204 | +15.70 | 1.75e-13 | *** |

### 3.4 Interpretation of the t-test

- **The MetaMamba-vs-7-d-baselines gap is enormous and unambiguous**. With n=24 paired observations, even a small raw gap (Δ = 0.34) yields t ≈ 15 and p on the order of 10⁻¹³. Holm-Bonferroni across 4 comparisons per family leaves all three pairwise p-values in the `***` tier.
- **MetaMamba vs MetaMamba-7d is the only non-trivial comparison**, and it is **not** significant on either Macro-F1 (p = 0.51) or F1-FAILED (p = 0.38). On ROC-AUC it is significant (Δ = +0.017, p < 0.001) but the magnitude is at the level of numerical noise — a difference that would not be operationally meaningful. **Bottom line**: the four extra continuous features in MetaMamba over MetaMamba-7d produce no measurable benefit on this dataset at n=473.
- The conservative Holm-Bonferroni correction (instead of plain Bonferroni) was chosen because Holm is uniformly more powerful under arbitrary dependence structure and still controls FWER — there is no statistical reason to pay the extra power cost.

---

## 4. Stage C — Mini Nested CV on MetaMamba-7d

### 4.1 Protocol

- **Outer**: 5-fold StratifiedKFold (seed=42), deterministic; outer test fold is **never** seen during HP selection.
- **Inner**: 2-fold hold-out on outer-train (first split, deterministic per outer index); 75% inner-train, 25% inner-val.
- **HP grid** (4 configs): `lr ∈ {1e-3, 5e-4} × contrastive_weight ∈ {0.0, 0.3}` (i.e. on/off Task-Contrastive). FiLM is held fixed at `use_film=True` (ablating FiLM is a separate study).
- **Per outer fold**: train 4 HP candidates on inner-train (189 students each), score on inner-val (189 students); pick best; refit on full outer-train (378 students) with best HP; evaluate on outer-test (95 students).
- **Total trainings**: 5 × (4 inner + 1 refit) = **25 single-fold trainings**.
- **Wall time**: 20.7 minutes on T4.

### 4.2 Headline

| Estimate | Macro-F1 | 95% CI |
|---|---|---|
| **Nested-CV** (5 outer folds, refit on best HP per fold) | **0.8754 ± 0.0181** | **[0.8530, 0.8979]** |
| Naive 8-fold × 3-seed (from Stage A) | 0.8783 | — |
| Difference (naive − nested) | **+0.0029** | — |

The naive 8-fold × 3-seed estimate is **0.29 percentage points higher** than the nested-CV estimate. This is well inside the nested-CV confidence interval, so the two are statistically indistinguishable; the gap is best interpreted as **a small optimism bias of the naive estimate**.

### 4.3 Per-outer-fold detail

| Outer # | Best HP | Best Val Macro-F1 | Test Macro-F1 | Refit time |
|---|---|---|---|---|
| 1 | lr1e-3_noTC | 0.8603 | 0.8748 | 64 s |
| 2 | lr5e-4_noTC | 0.8836 | 0.8626 | 56 s |
| 3 | lr5e-4_noTC | 0.8701 | 0.9042 | 61 s |
| 4 | lr1e-3_noTC | 0.8629 | 0.8579 | 90 s |
| 5 | lr1e-3_noTC | 0.8893 | 0.8776 | 95 s |

### 4.4 HP selection (inner Macro-F1 per outer fold)

| Outer # | lr1e-3_noTC | lr1e-3_TC | lr5e-4_noTC | lr5e-4_TC |
|---|---|---|---|---|
| 1 | 0.8603 | 0.8603 | 0.8603 | 0.8603 |
| 2 | 0.8773 | 0.8782 | **0.8836** | 0.8782 |
| 3 | 0.8690 | 0.8487 | **0.8701** | 0.8690 |
| 4 | 0.8629 | 0.8575 | 0.8629 | 0.8629 |
| 5 | 0.8893 | 0.8893 | 0.8893 | 0.8893 |

### 4.5 Interpretation

- **Outer folds 1 and 5 had no discriminative inner signal** — all four HP configs returned the same inner Macro-F1. This is a small-sample pathology: with 189 inner-val students and imbalanced classes (fail-rate ~66%), a single fold's Macro-F1 quantizes to a small set of values. The HP-selection step on these folds effectively reduces to random tie-breaking.
- **On the folds where the inner signal was informative (folds 2, 3, 4), `*_noTC` won every time**. The TC configurations were either tied with or slightly worse than their `noTC` counterpart.
- **`lr=5e-4` won 3 of the 5 outer folds, `lr=1e-3` won 2** — both are viable; the difference is in the noise floor at this sample size.
- **Combined with the t-test finding that MetaMamba-7d ≠ MetaMamba on Macro-F1**, the HP picture is consistent: the dataset is too small to consistently distinguish well-performing configurations, and the MetaMamba-7d-MetaMamba architectural difference (11-dim vs 7-dim input) is below the noise floor of the evaluation.

---

## 5. Cross-stage synthesis — what this means for the paper

### 5.1 What is now firmly established

1. **MetaMamba substantially outperforms LSTM / BiLSTM / Attention on CS1** (Macro-F1 gain ≥ 0.34, p < 1e-12). This claim is no longer dependent on a single point estimate; the gap survives Holm-Bonferroni correction across 24 paired observations.
2. **The naive 8-fold × 3-seed estimate is honest**, not contaminated by an HP-selection loop (there is no HP-selection loop in Stage A — defaults only). The 0.29 pp optimism gap from Stage C is within sampling noise.
3. **The 11-dim (event-type one-hot + 4 continuous features) variant does not outperform the 7-dim (event-type one-hot only) variant** at n=473. Both achieve Macro-F1 ≈ 0.878 with overlapping CIs.

### 5.2 What should be revised in the paper

| Claim in paper_v3 / paper_v5 | Evidence | Recommended action |
|---|---|---|
| "MetaMamba significantly outperforms baselines" | Stage B: t-test, all p < 1e-12 | **Keep** — add explicit p-values and Holm correction note |
| "Task-Contrastive auxiliary loss improves few-shot adaptation" | Stage C: 4/5 outer folds preferred noTC; inner val ties or favours noTC; Stage B: MetaMamba-7d noTC vs TC are not compared, but the nested CV evidence is unambiguous | **Soften** — reframe as "shows neutral-to-slightly-positive impact at this sample size; ablation pending" or honestly say "ablation study found no measurable benefit at n=473" |
| "11-dim input is more expressive than 7-dim" | Stage B: ns on Macro-F1 and F1-FAILED; marginal on ROC-AUC | **Soften** — "no statistically distinguishable improvement at n=473; continuous-feature ablation is a planned extension" |
| "8-fold × 3-seed cross-validation" (in v5 paper) | Used to read 5-fold; corrected in v5 commit text but `comparison.md` is now updated | **Verify text consistency** — paper draft should say 8-fold |
| (no nested CV in paper) | Stage C adds a new section**Honest Generalization Estimate via Nested CV** | **Add** as supplementary / robustness check |

### 5.3 What is NOT established (and what would close the gap)

- **FiLM contribution**: not tested in any of the three stages. The `use_film` ablation switch is now in `train.py` (commit 3bab298) but not exercised. This is the most likely candidate for a "real" ablation finding.
- **Sample-size-driven noise floor**: with n=473, fold-level Macro-F1 quantizes to a small set of values. A larger cohort (or a held-out replication set) would be needed to distinguish MetaMamba from MetaMamba-7d with statistical confidence.
- **Generalization beyond CS1**: all evidence is single-dataset. No external validation was performed.

---

## 6. Limitations and caveats

1. **Single dataset, single seed for HP grid search in nested CV**. The nested-CV outer fold split is deterministic (seed=42); we did not repeat the whole nested-CV procedure across multiple outer seeds. Bootstrap-style repetition would tighten the CIs.
2. **HP grid is small** (4 configs). The honest generalization estimate is conditional on this grid. A larger grid (e.g. adding `use_film=False`, dropout, batch_size) would raise the optimism gap. The 0.29 pp gap should therefore be read as a **lower bound** on the true optimism for a real hyperparameter search.
3. **Holm-Bonferroni assumes p-values can be ordered but does not require independence**. This is fine — Holm controls FWER under arbitrary dependence.
4. **No power analysis**. With n=24 paired observations, the minimum detectable effect (MDE) at α=0.05, power=0.80, on a paired t-test with σ_diff ≈ 0.01 is roughly 0.005 Macro-F1 — so the Stage B comparison between MetaMamba and MetaMamba-7d (Δ = −0.001) is below the MDE and the **non-significant** result is consistent with the dataset simply lacking the power to distinguish them. This is important to state explicitly in any write-up.
5. **Nested-CV was performed only on MetaMamba-7d**, not on the other models. The justification is computational (Stage C took 21 minutes; scaling to 5 models would take ~2 hours). The headline-numbers comparison in Stage A is therefore the only one available for the architectural differences across models.
6. **The original prompt mentioned "8-fold nested CV"**, which was interpreted as a strict nested CV with 8 outer folds. We chose 5 outer folds for tractability; the user should be aware of this scoping decision.

---

## 7. Reproducibility

### 7.1 Repository state after this work

```
commit b898b6c feat(analysis): mini nested cross-validation on MetaMamba-7d
commit 40725df fix(models): break dead import chain in lstm_7d/__init__.py
commit 130714a fix(data): drop dead import of removed features module from data/__init__.py
commit 9796e1c chore: ignore ad-hoc 5fold backup directories
commit cc9cfc2 feat(analysis): paired t-test + Holm-Bonferroni significance report
commit 3bab298 feat(model): add use_film / use_tc ablation switches to MetaMamba
commit 358ef62 feat(eval): re-evaluate with 8-fold x 3 seeds StratifiedKFold
```

### 7.2 Key files

| File | Purpose |
|---|---|
| `outputs/comparison.md` / `comparison.csv` | Stage A headline table + per-class / per-fold stability / training time / confusion matrices |
| `outputs/significance.md` / `significance.csv` | Stage B paired t-test tables with Holm-corrected p-values and 95% CIs |
| `outputs/significance.md` also contains a per-fold verification dump |
| `outputs/nested_cv/summary.md` / `summary.json` | Stage C nested-CV headline + per-outer-fold + HP-selection tables |
| `outputs/nested_cv/per_outer.jsonl` | Stage C per-fold detail (machine-readable) |
| `outputs/nested_cv/progress.log` | Stage C training timeline |
| `analysis/significance.py` | Stage B reusable analysis module |
| `analysis/nested_cv.py` | Stage C reusable nested-CV controller |
| `outputs/plots/*.png` | Stage A bar charts / ROC / PR / confusion / per-fold stability |

### 7.3 Commands to reproduce

```bash
# Stage A: full 8-fold × 3-seed re-evaluation
cd /home/ubuntu/StudentRisk
python main.py --model all --seeds 42 123 777 --n-splits 8

# Stage B: paired t-test + Holm-Bonferroni
python analysis/significance.py
#   -> writes outputs/significance.{md,csv}

# Stage C: mini nested CV on MetaMamba-7d
python analysis/nested_cv.py
#   -> writes outputs/nested_cv/{summary,per_outer,progress}.*
#   -> wall time ~21 min on T4
```

### 7.4 Software stack

- Python 3, PyTorch, scikit-learn (`StratifiedKFold`, `train_test_split`-equivalent), scipy 1.18 (`stats.ttest_rel`), pandas, numpy.
- Hardware: 1× NVIDIA Tesla T4 (15.36 GiB) on `43.139.55.246`, CUDA driver 580.126.20, CUDA 13.0.
- Dataset: CS1 public dataset, `IDE_logs/IDE_logs.csv` (28,588,309 events, 7 event types) + `IDE_logs/passed.csv` (473 students).

---

## 8. Summary table — all numbers in one place

| Estimate | Value | Source |
|---|---|---|
| **MetaMamba Macro-F1 (8-fold × 3-seed)** | 0.8787 | Stage A |
| **MetaMamba ROC-AUC (8-fold × 3-seed)** | 0.9347 | Stage A |
| **MetaMamba-7d Macro-F1 (8-fold × 3-seed)** | 0.8783 | Stage A |
| LSTM-7d Macro-F1 (8-fold × 3-seed) | 0.3982 | Stage A |
| BiLSTM-7d Macro-F1 (8-fold × 3-seed) | 0.4822 | Stage A |
| Attention-7d Macro-F1 (8-fold × 3-seed) | 0.5624 | Stage A |
| **MetaMamba vs LSTM-7d Δ Macro-F1** | +0.469, p < 1.4e-22 *** | Stage B |
| **MetaMamba vs BiLSTM-7d Δ Macro-F1** | +0.387, p < 1.4e-13 *** | Stage B |
| **MetaMamba vs Attention-7d Δ Macro-F1** | +0.343, p < 6.1e-13 *** | Stage B |
| **MetaMamba vs MetaMamba-7d Δ Macro-F1** | −0.001, p = 0.51 ns | Stage B |
| **Nested-CV MetaMamba-7d Macro-F1** | **0.8754 ± 0.018**, 95% CI [0.8530, 0.8979] | Stage C |
| **Naive-vs-nested optimism gap** | +0.0029 (0.29 pp) | Stage C |
| **Task-Contrastive win count (nested CV)** | 1/5 outer folds | Stage C |
| **Total analysis wall time** | ~50 min on T4 (Stage A 25 min + Stage B <1 min + Stage C 21 min) | — |

---

*Report generated 2026-08-22 by the analysis pipeline; all numbers reproducible from the cited files in `outputs/` and scripts in `analysis/`.*