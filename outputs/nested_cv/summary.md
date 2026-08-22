# MetaMamba-7d — Mini Nested CV Report

> Method: **5-fold outer × 2-fold inner × 4 HP grid** = 25 single-fold trainings
> Outer seed = 42, Inner seed base = 42
> Wall time: 20.7 min on T4

## Headline

- **Nested-CV Macro-F1**: **0.8754** (std 0.0181, 95% CI [0.8530, 0.8979])
- Naive 8-fold × 3-seed estimate (from outputs/comparison.csv): **0.8783**
- Difference: +0.0029 (naive optimistic by 0.29 pp)

## Per-outer-fold detail

| Outer # | Best HP | Best Val Macro-F1 | Test Macro-F1 | Refit time (s) |
|---|---|---|---|---|
| 1 | lr1e-3_noTC | 0.8603 | 0.8748 | 64.2 |
| 2 | lr5e-4_noTC | 0.8836 | 0.8626 | 56.1 |
| 3 | lr5e-4_noTC | 0.8701 | 0.9042 | 60.6 |
| 4 | lr1e-3_noTC | 0.8629 | 0.8579 | 90.1 |
| 5 | lr1e-3_noTC | 0.8893 | 0.8776 | 94.5 |

## HP selection (inner val Macro-F1 per outer fold)

| Outer # | lr1e-3_noTC | lr1e-3_TC | lr5e-4_noTC | lr5e-4_TC |
|---|---|---|---|---|
| 1 | 0.8603 | 0.8603 | 0.8603 | 0.8603 |
| 2 | 0.8773 | 0.8782 | 0.8836 | 0.8782 |
| 3 | 0.8690 | 0.8487 | 0.8701 | 0.8690 |
| 4 | 0.8629 | 0.8575 | 0.8629 | 0.8629 |
| 5 | 0.8893 | 0.8893 | 0.8893 | 0.8893 |

## Interpretation

Nested CV controls for HP-selection optimism that naive single-CV estimates suffer from.
A drop from naive to nested is expected; the magnitude is the "optimism gap" — how much the naive
estimate overfits by peeking at the validation set during HP choice.
Negative bias (naive < nested) would be unusual.