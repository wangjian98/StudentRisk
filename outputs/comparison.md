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
| MetaMamba | 0.8858 | 0.8736 | 0.8865 | 0.9318 | 0.9696 |
| LSTM-7d | 0.6638 | 0.4107 | 0.5373 | 0.6190 | 0.7502 |
| BiLSTM-7d | 0.6723 | 0.4518 | 0.5657 | 0.6430 | 0.7641 |
| Attention-7d | 0.6913 | 0.5302 | 0.6204 | 0.6938 | 0.8213 |
| MetaMamba-7d | 0.8837 | 0.8715 | 0.8845 | 0.9188 | 0.9608 |

## 2. Per-Class Precision / Recall / F1

**Class 0 = PASSED** (predicted to pass)

| Model | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| RF-7d (raw event counts) | 0.7300 | 0.9182 | 0.8134 | 159 |
| MetaMamba | 0.8144 | 0.8553 | 0.8344 | 159 |
| LSTM-7d | 0.5000 | 0.0126 | 0.0245 | 159 |
| BiLSTM-7d | 0.6429 | 0.0566 | 0.1040 | 159 |
| Attention-7d | 0.6757 | 0.1572 | 0.2551 | 159 |
| MetaMamba-7d | 0.8095 | 0.8553 | 0.8318 | 159 |

**Class 1 = FAILED** (positive class)

| Model | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| RF-7d (raw event counts) | 0.9524 | 0.8280 | 0.8859 | 314 |
| MetaMamba | 0.9248 | 0.9013 | 0.9129 | 314 |
| LSTM-7d | 0.6652 | 0.9936 | 0.7969 | 314 |
| BiLSTM-7d | 0.6732 | 0.9841 | 0.7995 | 314 |
| Attention-7d | 0.6927 | 0.9618 | 0.8053 | 314 |
| MetaMamba-7d | 0.9246 | 0.8981 | 0.9111 | 314 |

## 3. Per-Fold Stability (Macro-F1 mean ± std)

| Model | Macro-F1 Mean | Macro-F1 Std | ROC-AUC Mean | ROC-AUC Std |
|---|---|---|---|---|
| RF-7d (raw event counts) | 0.8539 | 0.0355 | 0.9175 | 0.0252 |
| MetaMamba | 0.8774 | 0.0242 | 0.9410 | 0.0200 |
| LSTM-7d | 0.4204 | 0.0395 | 0.6073 | 0.0513 |
| BiLSTM-7d | 0.4612 | 0.0713 | 0.6194 | 0.0660 |
| Attention-7d | 0.5206 | 0.0976 | 0.6592 | 0.0832 |
| MetaMamba-7d | 0.8744 | 0.0205 | 0.9251 | 0.0217 |

## 4. Confusion Matrices (OOF aggregated)

Format: rows = true class, cols = predicted class. Class 0=PASSED, Class 1=FAILED

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| RF-7d (raw event counts) | 146 | 13 | 54 | 260 |
| MetaMamba | 136 | 23 | 31 | 283 |
| LSTM-7d | 2 | 157 | 2 | 312 |
| BiLSTM-7d | 9 | 150 | 5 | 309 |
| Attention-7d | 25 | 134 | 12 | 302 |
| MetaMamba-7d | 136 | 23 | 32 | 282 |

## 5. Training Time

| Model | n_params | Elapsed (sec) |
|---|---|---|
| RF-7d (raw event counts) | N/A | 5.4 |
| MetaMamba | 22,065 | 3008.3 |
| LSTM-7d | 33,857 | 16.9 |
| BiLSTM-7d | 67,201 | 20.2 |
| Attention-7d | 67,713 | 29.1 |
| MetaMamba-7d | 21,809 | 1271.0 |

## 6. Visualizations

See `outputs/plots/`:

- `metric_comparison.png` — Bar chart of accuracy / macro-F1 / ROC-AUC per model
- `roc_curves_all.png` — ROC curves (all models overlaid)
- `pr_curves_all.png` — Precision-Recall curves (all models overlaid)
- `confusion_matrices.png` — Confusion matrices grid
- `per_fold_stability.png` — Per-fold F1 stability box plot

