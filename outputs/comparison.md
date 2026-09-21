# StudentRisk — Multi-Model Evaluation Report
> Label convention: **Failed=1 (positive class)**, Passed=0
> Dataset: CS1 (n=473, fail_rate=0.6638)
> Cross-validation: 5-fold × 3 seeds (StratifiedKFold)
> Threshold: 0.5

---

## 1. Overall Metrics (5-fold × N seeds OOF)

| Model | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| RF-7d (raw event counts) | 0.8584 | 0.8496 | 0.8615 | 0.9179 | 0.9606 |
| MetaMamba | 0.8858 | 0.8736 | 0.8865 | 0.9319 | 0.9695 |
| LSTM-7d | 0.6681 | 0.4124 | 0.5394 | 0.6172 | 0.7496 |
| BiLSTM-7d | 0.6723 | 0.4614 | 0.5718 | 0.6416 | 0.7639 |
| Attention-7d | 0.6786 | 0.5109 | 0.6048 | 0.7114 | 0.8331 |
| MetaMamba-7d | 0.8858 | 0.8736 | 0.8865 | 0.9233 | 0.9631 |

## 2. Per-Class Precision / Recall / F1

**Class 0 = PASSED** (predicted to pass)

| Model | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| RF-7d (raw event counts) | 0.7300 | 0.9182 | 0.8134 | 159 |
| MetaMamba | 0.8144 | 0.8553 | 0.8344 | 159 |
| LSTM-7d | 1.0000 | 0.0126 | 0.0248 | 159 |
| BiLSTM-7d | 0.6111 | 0.0692 | 0.1243 | 159 |
| Attention-7d | 0.5946 | 0.1384 | 0.2245 | 159 |
| MetaMamba-7d | 0.8144 | 0.8553 | 0.8344 | 159 |

**Class 1 = FAILED** (positive class)

| Model | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| RF-7d (raw event counts) | 0.9524 | 0.8280 | 0.8859 | 314 |
| MetaMamba | 0.9248 | 0.9013 | 0.9129 | 314 |
| LSTM-7d | 0.6667 | 1.0000 | 0.8000 | 314 |
| BiLSTM-7d | 0.6747 | 0.9777 | 0.7984 | 314 |
| Attention-7d | 0.6858 | 0.9522 | 0.7973 | 314 |
| MetaMamba-7d | 0.9248 | 0.9013 | 0.9129 | 314 |

## 3. Per-Fold Stability (Macro-F1 mean ± std)

| Model | Macro-F1 Mean | Macro-F1 Std | ROC-AUC Mean | ROC-AUC Std |
|---|---|---|---|---|
| RF-7d (raw event counts) | 0.8539 | 0.0355 | 0.9175 | 0.0252 |
| MetaMamba | 0.8759 | 0.0226 | 0.9410 | 0.0205 |
| LSTM-7d | 0.4186 | 0.0336 | 0.6072 | 0.0521 |
| BiLSTM-7d | 0.4710 | 0.0841 | 0.6165 | 0.0689 |
| Attention-7d | 0.5177 | 0.1035 | 0.6669 | 0.0834 |
| MetaMamba-7d | 0.8760 | 0.0191 | 0.9259 | 0.0190 |

## 4. Confusion Matrices (OOF aggregated)

Format: rows = true class, cols = predicted class. Class 0=PASSED, Class 1=FAILED

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| RF-7d (raw event counts) | 146 | 13 | 54 | 260 |
| MetaMamba | 136 | 23 | 31 | 283 |
| LSTM-7d | 2 | 157 | 0 | 314 |
| BiLSTM-7d | 11 | 148 | 7 | 307 |
| Attention-7d | 22 | 137 | 15 | 299 |
| MetaMamba-7d | 136 | 23 | 31 | 283 |

## 5. Training Time

| Model | n_params | Elapsed (sec) |
|---|---|---|
| RF-7d (raw event counts) | N/A | 5.3 |
| MetaMamba | 22,065 | 4326.6 |
| LSTM-7d | 33,857 | 98.0 |
| BiLSTM-7d | 67,201 | 157.7 |
| Attention-7d | 67,713 | 614.4 |
| MetaMamba-7d | 21,809 | 1810.7 |

## 6. Visualizations

See `outputs/plots/`:

- `metric_comparison.png` — Bar chart of accuracy / macro-F1 / ROC-AUC per model
- `roc_curves_all.png` — ROC curves (all models overlaid)
- `pr_curves_all.png` — Precision-Recall curves (all models overlaid)
- `confusion_matrices.png` — Confusion matrices grid
- `per_fold_stability.png` — Per-fold F1 stability box plot

