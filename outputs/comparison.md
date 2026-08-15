# StudentRisk — Multi-Model Evaluation Report
> Label convention: **Failed=1 (positive class)**, Passed=0
> Dataset: CS1 (n=473, fail_rate=0.6638)
> Cross-validation: 5-fold × 3 seeds (StratifiedKFold)
> Threshold: 0.5

---

## 1. Overall Metrics (5-fold × N seeds OOF)

| Model | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.8436 | 0.8283 | 0.8451 | 0.9162 | 0.9616 |
| RF-7d (raw event counts) | 0.8626 | 0.8524 | 0.8651 | 0.9178 | 0.9618 |
| LSTM | 0.8457 | 0.8313 | 0.8474 | 0.9272 | 0.9654 |
| BiLSTM | 0.8457 | 0.8322 | 0.8478 | 0.9293 | 0.9664 |
| Attention | 0.8541 | 0.8437 | 0.8569 | 0.9293 | 0.9640 |
| MetaMamba | 0.8879 | 0.8761 | 0.8887 | 0.9290 | 0.9687 |
| LSTM-7d | 0.6681 | 0.4292 | 0.5502 | 0.6302 | 0.7574 |
| BiLSTM-7d | 0.6977 | 0.5450 | 0.6314 | 0.7080 | 0.8154 |
| Attention-7d | 0.6871 | 0.5440 | 0.6277 | 0.7011 | 0.8312 |
| MetaMamba-7d | 0.8837 | 0.8715 | 0.8845 | 0.9195 | 0.9625 |

## 2. Per-Class Precision / Recall / F1

**Class 0 = PASSED** (predicted to pass)

| Model | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Random Forest | 0.7457 | 0.8113 | 0.7771 | 159 |
| RF-7d (raw event counts) | 0.7474 | 0.8931 | 0.8138 | 159 |
| LSTM | 0.7443 | 0.8239 | 0.7821 | 159 |
| BiLSTM | 0.7389 | 0.8365 | 0.7847 | 159 |
| Attention | 0.7344 | 0.8868 | 0.8034 | 159 |
| MetaMamba | 0.8155 | 0.8616 | 0.8379 | 159 |
| LSTM-7d | 0.6250 | 0.0314 | 0.0599 | 159 |
| BiLSTM-7d | 0.7000 | 0.1761 | 0.2814 | 159 |
| Attention-7d | 0.6122 | 0.1887 | 0.2885 | 159 |
| MetaMamba-7d | 0.8095 | 0.8553 | 0.8318 | 159 |

**Class 1 = FAILED** (positive class)

| Model | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Random Forest | 0.9000 | 0.8599 | 0.8795 | 314 |
| RF-7d (raw event counts) | 0.9399 | 0.8471 | 0.8911 | 314 |
| LSTM | 0.9057 | 0.8567 | 0.8805 | 314 |
| BiLSTM | 0.9113 | 0.8503 | 0.8797 | 314 |
| Attention | 0.9359 | 0.8376 | 0.8840 | 314 |
| MetaMamba | 0.9279 | 0.9013 | 0.9144 | 314 |
| LSTM-7d | 0.6688 | 0.9904 | 0.7985 | 314 |
| BiLSTM-7d | 0.6975 | 0.9618 | 0.8086 | 314 |
| Attention-7d | 0.6958 | 0.9395 | 0.7995 | 314 |
| MetaMamba-7d | 0.9246 | 0.8981 | 0.9111 | 314 |

## 3. Per-Fold Stability (Macro-F1 mean ± std)

| Model | Macro-F1 Mean | Macro-F1 Std | ROC-AUC Mean | ROC-AUC Std |
|---|---|---|---|---|
| Random Forest | 0.8280 | 0.0353 | 0.9162 | 0.0267 |
| RF-7d (raw event counts) | 0.8527 | 0.0320 | 0.9171 | 0.0261 |
| LSTM | 0.8324 | 0.0293 | 0.9245 | 0.0222 |
| BiLSTM | 0.8359 | 0.0310 | 0.9264 | 0.0204 |
| Attention | 0.8342 | 0.0310 | 0.9287 | 0.0197 |
| MetaMamba | 0.8777 | 0.0230 | 0.9375 | 0.0200 |
| LSTM-7d | 0.4275 | 0.0401 | 0.6174 | 0.0577 |
| BiLSTM-7d | 0.5513 | 0.1239 | 0.6626 | 0.0943 |
| Attention-7d | 0.5412 | 0.0979 | 0.6662 | 0.0906 |
| MetaMamba-7d | 0.8761 | 0.0220 | 0.9268 | 0.0206 |

## 4. Confusion Matrices (OOF aggregated)

Format: rows = true class, cols = predicted class. Class 0=PASSED, Class 1=FAILED

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| Random Forest | 129 | 30 | 44 | 270 |
| RF-7d (raw event counts) | 142 | 17 | 48 | 266 |
| LSTM | 131 | 28 | 45 | 269 |
| BiLSTM | 133 | 26 | 47 | 267 |
| Attention | 141 | 18 | 51 | 263 |
| MetaMamba | 137 | 22 | 31 | 283 |
| LSTM-7d | 5 | 154 | 3 | 311 |
| BiLSTM-7d | 28 | 131 | 12 | 302 |
| Attention-7d | 30 | 129 | 19 | 295 |
| MetaMamba-7d | 136 | 23 | 32 | 282 |

## 5. Training Time

| Model | n_params | Elapsed (sec) |
|---|---|---|
| Random Forest | N/A | 5.2 |
| RF-7d (raw event counts) | N/A | 5.0 |
| LSTM | 36,353 | 17.9 |
| BiLSTM | 69,697 | 18.1 |
| Attention | 70,209 | 27.1 |
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

