# StudentRisk — Multi-Model Evaluation Report
> Label convention: **Failed=1 (positive class)**, Passed=0
> Dataset: CS1 (n=473, fail_rate=0.6638)
> Cross-validation: 5-fold × 3 seeds (StratifiedKFold)
> Threshold: 0.5

---

## 1. Overall Metrics (5-fold × N seeds OOF)

| Model | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| RF-7d (raw event counts) | 0.8626 | 0.8524 | 0.8651 | 0.9178 | 0.9618 |
| MetaMamba | 0.8879 | 0.8761 | 0.8887 | 0.9290 | 0.9687 |
| LSTM-7d | 0.6681 | 0.4292 | 0.5502 | 0.6302 | 0.7574 |
| BiLSTM-7d | 0.6977 | 0.5450 | 0.6314 | 0.7080 | 0.8154 |
| Attention-7d | 0.6871 | 0.5440 | 0.6277 | 0.7011 | 0.8312 |
| MetaMamba-7d | 0.8837 | 0.8715 | 0.8845 | 0.9195 | 0.9625 |

## 2. Per-Class Precision / Recall / F1

**Class 0 = PASSED** (predicted to pass)

| Model | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| RF-7d (raw event counts) | 0.7474 | 0.8931 | 0.8138 | 159 |
| MetaMamba | 0.8155 | 0.8616 | 0.8379 | 159 |
| LSTM-7d | 0.6250 | 0.0314 | 0.0599 | 159 |
| BiLSTM-7d | 0.7000 | 0.1761 | 0.2814 | 159 |
| Attention-7d | 0.6122 | 0.1887 | 0.2885 | 159 |
| MetaMamba-7d | 0.8095 | 0.8553 | 0.8318 | 159 |

**Class 1 = FAILED** (positive class)

| Model | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| RF-7d (raw event counts) | 0.9399 | 0.8471 | 0.8911 | 314 |
| MetaMamba | 0.9279 | 0.9013 | 0.9144 | 314 |
| LSTM-7d | 0.6688 | 0.9904 | 0.7985 | 314 |
| BiLSTM-7d | 0.6975 | 0.9618 | 0.8086 | 314 |
| Attention-7d | 0.6958 | 0.9395 | 0.7995 | 314 |
| MetaMamba-7d | 0.9246 | 0.8981 | 0.9111 | 314 |

## 3. Per-Fold Stability (Macro-F1 mean ± std)

| Model | Macro-F1 Mean | Macro-F1 Std | ROC-AUC Mean | ROC-AUC Std |
|---|---|---|---|---|
| RF-7d (raw event counts) | 0.8527 | 0.0320 | 0.9171 | 0.0261 |
| MetaMamba | 0.8777 | 0.0230 | 0.9375 | 0.0200 |
| LSTM-7d | 0.4275 | 0.0401 | 0.6174 | 0.0577 |
| BiLSTM-7d | 0.5513 | 0.1239 | 0.6626 | 0.0943 |
| Attention-7d | 0.5412 | 0.0979 | 0.6662 | 0.0906 |
| MetaMamba-7d | 0.8761 | 0.0220 | 0.9268 | 0.0206 |

## 4. Confusion Matrices (OOF aggregated)

Format: rows = true class, cols = predicted class. Class 0=PASSED, Class 1=FAILED

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| RF-7d (raw event counts) | 142 | 17 | 48 | 266 |
| MetaMamba | 137 | 22 | 31 | 283 |
| LSTM-7d | 5 | 154 | 3 | 311 |
| BiLSTM-7d | 28 | 131 | 12 | 302 |
| Attention-7d | 30 | 129 | 19 | 295 |
| MetaMamba-7d | 136 | 23 | 32 | 282 |

## 5. Training Time

| Model | n_params | Elapsed (sec) |
|---|---|---|
| RF-7d (raw event counts) | N/A | 5.0 |
| MetaMamba | 22,065 | 1004.4 |
| LSTM-7d | 33,857 | 24.8 |
| BiLSTM-7d | 67,201 | 27.7 |
| Attention-7d | 67,713 | 34.2 |
| MetaMamba-7d | 21,809 | 801.8 |

## 6. Visualizations

See `outputs/plots/`:

- `metric_comparison.png` — Bar chart of accuracy / macro-F1 / ROC-AUC per model
- `roc_curves_all.png` — ROC curves (all models overlaid)
- `pr_curves_all.png` — Precision-Recall curves (all models overlaid)
- `confusion_matrices.png` — Confusion matrices grid
- `per_fold_stability.png` — Per-fold F1 stability box plot

