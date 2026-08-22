# StudentRisk — Multi-Model Evaluation Report
> Label convention: **Failed=1 (positive class)**, Passed=0
> Dataset: CS1 (n=473, fail_rate=0.6638)
> Cross-validation: 8-fold × 3 seeds (StratifiedKFold)
> Threshold: 0.5
> **RF-7d excluded from this run**: implementation is currently a stub (`_StubRFModel`); main.py reported `_StubRFModel() takes no arguments` during the 2026-08-21/22 run. To re-include, complete `models/rf7/train.py` and re-execute.




---

## 1. Overall Metrics (8-fold × 3 seeds OOF)

| Model | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| MetaMamba | 0.8901 | 0.8787 | 0.8908 | 0.9347 | 0.9713 |
| LSTM-7d | 0.6617 | 0.3982 | 0.5287 | 0.6225 | 0.7609 |
| BiLSTM-7d | 0.6850 | 0.4822 | 0.5884 | 0.6750 | 0.7930 |
| Attention-7d | 0.6998 | 0.5624 | 0.6428 | 0.7108 | 0.8267 |
| MetaMamba-7d | 0.8901 | 0.8783 | 0.8907 | 0.9293 | 0.9669 |

## 2. Per-Class Precision / Recall / F1

**Class 0 = PASSED** (predicted to pass)

| Model | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| MetaMamba | 0.8166 | 0.8679 | 0.8415 | 159 |
| LSTM-7d | 0.0000 | 0.0000 | 0.0000 | 159 |
| BiLSTM-7d | 0.7778 | 0.0881 | 0.1582 | 159 |
| Attention-7d | 0.6735 | 0.2075 | 0.3173 | 159 |
| MetaMamba-7d | 0.8204 | 0.8616 | 0.8405 | 159 |

**Class 1 = FAILED** (positive class)

| Model | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| MetaMamba | 0.9309 | 0.9013 | 0.9159 | 314 |
| LSTM-7d | 0.6631 | 0.9968 | 0.7964 | 314 |
| BiLSTM-7d | 0.6813 | 0.9873 | 0.8062 | 314 |
| Attention-7d | 0.7028 | 0.9490 | 0.8076 | 314 |
| MetaMamba-7d | 0.9281 | 0.9045 | 0.9161 | 314 |

## 3. Per-Fold Stability (Macro-F1 mean ± std)

| Model | Macro-F1 Mean | Macro-F1 Std | ROC-AUC Mean | ROC-AUC Std |
|---|---|---|---|---|
| MetaMamba | 0.8753 | 0.0340 | 0.9399 | 0.0307 |
| LSTM-7d | 0.4064 | 0.0290 | 0.6050 | 0.0691 |
| BiLSTM-7d | 0.4888 | 0.0946 | 0.6384 | 0.0867 |
| Attention-7d | 0.5328 | 0.0913 | 0.6832 | 0.0836 |
| MetaMamba-7d | 0.8766 | 0.0324 | 0.9233 | 0.0331 |

## 4. Confusion Matrices (OOF aggregated)

Format: rows = true class, cols = predicted class. Class 0=PASSED, Class 1=FAILED

| Model | TN | FP | FN | TP |
|---|---|---|---|---|
| MetaMamba | 138 | 21 | 31 | 283 |
| LSTM-7d | 0 | 159 | 1 | 313 |
| BiLSTM-7d | 14 | 145 | 4 | 310 |
| Attention-7d | 33 | 126 | 16 | 298 |
| MetaMamba-7d | 137 | 22 | 30 | 284 |

## 5. Training Time

| Model | n_params | Elapsed (sec) |
|---|---|---|
| MetaMamba | 22,065 | 5013.1 |
| LSTM-7d | 33,857 | 29.6 |
| BiLSTM-7d | 67,201 | 35.6 |
| Attention-7d | 67,713 | 56.6 |
| MetaMamba-7d | 21,809 | 2204.6 |

## 6. Visualizations

See `outputs/plots/`:

- `metric_comparison.png` — Bar chart of accuracy / macro-F1 / ROC-AUC per model
- `roc_curves_all.png` — ROC curves (all models overlaid)
- `pr_curves_all.png` — Precision-Recall curves (all models overlaid)
- `confusion_matrices.png` — Confusion matrices grid
- `per_fold_stability.png` — Per-fold F1 stability box plot

