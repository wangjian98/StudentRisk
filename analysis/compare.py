"""Comparison + Markdown report for all models in StudentRisk.

Reads each model's results.json and produces:
  - outputs/comparison.csv (per-class metrics + overall)
  - outputs/comparison.md  (Markdown report)
"""
import os
import json
import numpy as np
import pandas as pd


_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
RESULTS_DIR = os.path.join(_ROOT, 'results')
OUTPUTS_DIR = os.path.join(_ROOT, 'outputs')


MODEL_ORDER = ['rf', 'rf7', 'lstm', 'bilstm', 'attention', 'meta_mamba',
                 'lstm_7d', 'bilstm_7d', 'attention_7d', 'meta_mamba_7d']
MODEL_NAMES = {
    'rf':            'Random Forest',
    'rf7':           'RF-7d (raw event counts)',
    'lstm':       'LSTM',
    'bilstm':     'BiLSTM',
    'attention':  'Attention',
    'meta_mamba':    'MetaMamba',
    'lstm_7d':       'LSTM-7d',
    'bilstm_7d':     'BiLSTM-7d',
    'attention_7d':  'Attention-7d',
    'meta_mamba_7d': 'MetaMamba-7d',
}


def _read_results(model_name: str):
    path = os.path.join(RESULTS_DIR, model_name, 'results.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def build_comparison() -> pd.DataFrame:
    """Collect overall metrics from all model results into a single DataFrame."""
    rows = []
    for m in MODEL_ORDER:
        d = _read_results(m)
        if d is None:
            continue
        o = d['overall']
        n_params = d.get('n_params', None)
        rows.append({
            'model':           MODEL_NAMES[m],
            'model_key':       m,
            'accuracy':        o['accuracy'],
            'precision_pass':  o['precision_class_0'],
            'recall_pass':     o['recall_class_0'],
            'f1_pass':         o['f1_class_0'],
            'support_pass':    o['support_class_0'],
            'precision_fail':  o['precision_class_1'],
            'recall_fail':     o['recall_class_1'],
            'f1_fail':         o['f1_class_1'],
            'support_fail':    o['support_class_1'],
            'macro_f1':        o['macro_f1'],
            'weighted_f1':     o['weighted_f1'],
            'roc_auc':         o['roc_auc'],
            'pr_auc':          o.get('pr_auc', np.nan),
            'n_params':        n_params if n_params else 'N/A',
            'n_seeds':         d.get('n_seeds'),
            'n_splits':        d.get('n_splits'),
            'threshold':       d.get('threshold'),
            'elapsed_sec':     round(d.get('elapsed_seconds', 0), 1),
        })
    df = pd.DataFrame(rows)
    return df


def build_comparison_markdown(df: pd.DataFrame) -> str:
    """Build a Markdown report comparing all models."""
    md = []
    md.append("# StudentRisk — Multi-Model Evaluation Report\n")
    md.append("> Label convention: **Failed=1 (positive class)**, Passed=0\n")
    md.append(f"> Dataset: CS1 (n=473, fail_rate={314/473:.4f})\n")
    md.append(f"> Cross-validation: 5-fold × {df['n_seeds'].iloc[0]} seeds (StratifiedKFold)\n")
    md.append(f"> Threshold: {df['threshold'].iloc[0]}\n")
    md.append("\n---\n\n")

    # Section 1: Overall metrics table
    md.append("## 1. Overall Metrics (5-fold × N seeds OOF)\n\n")
    md.append("| Model | Accuracy | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC |\n")
    md.append("|---|---|---|---|---|---|\n")
    for _, r in df.iterrows():
        md.append("| {m} | {a:.4f} | {mf:.4f} | {wf:.4f} | {auc:.4f} | {pr:.4f} |\n".format(
            m=r['model'], a=r['accuracy'], mf=r['macro_f1'], wf=r['weighted_f1'],
            auc=r['roc_auc'], pr=r['pr_auc']))
    md.append("\n")

    # Section 2: Per-class metrics
    md.append("## 2. Per-Class Precision / Recall / F1\n\n")
    md.append("**Class 0 = PASSED** (predicted to pass)\n\n")
    md.append("| Model | Precision | Recall | F1 | Support |\n")
    md.append("|---|---|---|---|---|\n")
    for _, r in df.iterrows():
        md.append("| {m} | {p:.4f} | {rec:.4f} | {f:.4f} | {s} |\n".format(
            m=r['model'], p=r['precision_pass'], rec=r['recall_pass'],
            f=r['f1_pass'], s=int(r['support_pass'])))
    md.append("\n**Class 1 = FAILED** (positive class)\n\n")
    md.append("| Model | Precision | Recall | F1 | Support |\n")
    md.append("|---|---|---|---|---|\n")
    for _, r in df.iterrows():
        md.append("| {m} | {p:.4f} | {rec:.4f} | {f:.4f} | {s} |\n".format(
            m=r['model'], p=r['precision_fail'], rec=r['recall_fail'],
            f=r['f1_fail'], s=int(r['support_fail'])))
    md.append("\n")

    # Section 3: Stability (per-fold std)
    md.append("## 3. Per-Fold Stability (Macro-F1 mean ± std)\n\n")
    md.append("| Model | Macro-F1 Mean | Macro-F1 Std | ROC-AUC Mean | ROC-AUC Std |\n")
    md.append("|---|---|---|---|---|\n")
    for m in MODEL_ORDER:
        d = _read_results(m)
        if d is None:
            continue
        pf = d.get('per_fold_summary', {})
        md.append("| {m} | {mf:.4f} | {mfs:.4f} | {auc:.4f} | {aucs:.4f} |\n".format(
            m=MODEL_NAMES[m], mf=pf.get('macro_f1_mean', 0), mfs=pf.get('macro_f1_std', 0),
            auc=pf.get('roc_auc_mean', 0), aucs=pf.get('roc_auc_std', 0)))
    md.append("\n")

    # Section 4: Confusion matrices
    md.append("## 4. Confusion Matrices (OOF aggregated)\n\n")
    md.append("Format: rows = true class, cols = predicted class. Class 0=PASSED, Class 1=FAILED\n\n")
    md.append("| Model | TN | FP | FN | TP |\n")
    md.append("|---|---|---|---|---|\n")
    for m in MODEL_ORDER:
        d = _read_results(m)
        if d is None:
            continue
        cm = d['overall']['confusion_matrix']
        md.append("| {m} | {tn} | {fp} | {fn} | {tp} |\n".format(
            m=MODEL_NAMES[m], tn=cm['TN'], fp=cm['FP'], fn=cm['FN'], tp=cm['TP']))
    md.append("\n")

    # Section 5: Runtime
    md.append("## 5. Training Time\n\n")
    md.append("| Model | n_params | Elapsed (sec) |\n")
    md.append("|---|---|---|\n")
    for _, r in df.iterrows():
        nparams = r['n_params']
        nparams_str = f"{nparams:,}" if isinstance(nparams, int) else str(nparams)
        md.append("| {m} | {p} | {t} |\n".format(m=r['model'], p=nparams_str, t=r['elapsed_sec']))
    md.append("\n")

    # Section 6: Visualization
    md.append("## 6. Visualizations\n\n")
    md.append("See `outputs/plots/`:\n\n")
    md.append("- `metric_comparison.png` — Bar chart of accuracy / macro-F1 / ROC-AUC per model\n")
    md.append("- `roc_curves_all.png` — ROC curves (all models overlaid)\n")
    md.append("- `pr_curves_all.png` — Precision-Recall curves (all models overlaid)\n")
    md.append("- `confusion_matrices.png` — Confusion matrices grid\n")
    md.append("- `per_fold_stability.png` — Per-fold F1 stability box plot\n\n")

    return ''.join(md)


def run():
    """Build comparison.csv and comparison.md. Returns DataFrame."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df = build_comparison()
    csv_path = os.path.join(OUTPUTS_DIR, 'comparison.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[compare] Wrote {csv_path}", flush=True)
    md = build_comparison_markdown(df)
    md_path = os.path.join(OUTPUTS_DIR, 'comparison.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"[compare] Wrote {md_path}", flush=True)
    return df


if __name__ == '__main__':
    run()