#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 paper_v12_en.md：以 v11 英文底稿注入 rev12 同协议修订实验结果（与中文版对照）。"""
import io
import os
import sys
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'analysis'))

from analysis.rev12.aggregate import (load_cells, matrix_table, prefix_table, film_table,
                                       lopo_table, RES)

DOCS = os.path.join(_ROOT, 'docs')
SRC = os.path.join(DOCS, 'paper_v11_en.md')
DST = os.path.join(DOCS, 'paper_v12_en.md')


def fmt_p(p):
    if p is None or (isinstance(p, float) and p != p):
        return '—'
    return '<1e-4' if p < 1e-4 else f'{p:.4f}'


def build():
    s = io.open(SRC, encoding='utf-8').read()
    ident = load_cells(os.path.join(RES, 'exp1_matrix'))
    deconf = load_cells(os.path.join(RES, 'exp1_matrix_deconf'))
    pref = load_cells(os.path.join(RES, 'exp2_prefix'))
    prefd = load_cells(os.path.join(RES, 'exp2_prefix_deconf'))
    _, t_ident = matrix_table(ident, '')
    _, t_deconf = matrix_table(deconf, '')
    _, t_pref = prefix_table(pref)
    _, t_prefd = prefix_table(prefd)
    _, t_film = film_table(os.path.join(RES, 'exp3_film'))
    _, t_lopo = lopo_table(os.path.join(RES, 'exp4_lopo'))

    sig_path = os.path.join(RES, 'exp5_stats', 'significance.json')
    t_sig = ''
    if os.path.exists(sig_path):
        with open(sig_path) as f:
            sig = json.load(f)
        lines = ['| Space | Comparison | Δmacro-F1 | Wilcoxon p | Holm p | McNemar p | ΔROC-AUC | Δbal-acc |',
                 '|---|---|---:|---:|---:|---:|---:|---:|']
        for e in sig['pairs']:
            lines.append(
                f"| {e['dim']}d | {e['comparison']} | {e['macro_f1']['delta']:+.4f} | "
                f"{fmt_p(e['macro_f1']['p_raw'])} | {fmt_p(e['macro_f1'].get('p_holm'))} | "
                f"{fmt_p(e.get('mcnemar', {}).get('p'))} | {e['roc_auc']['delta']:+.4f} | "
                f"{e['balanced_accuracy']['delta']:+.4f} |")
        t_sig = '\n'.join(lines)

    s = s.replace('**Submission date:** 5 November 2026 (v11)',
                  '**Submission date:** 7 November 2026 (v12)')
    s = s.replace(
        '### 5.3 Baselines\n\n'
        "Mandatory: the **part-prior trivial baseline** (student level 0.9135 / stage level 0.4164); RF-46d / RF-7d; LSTM / BiLSTM / Attention (7d); **HMM (multinomial and Gaussian — the model family reported at 0.7697 on this very dataset by EDM 2025)**; MetaMamba-11d and its four-way variants; and the three channels and fusion variants of CS1-SHR.",
        '### 5.3 Baselines\n\n'
        "Mandatory: the **part-prior trivial baseline** (student level 0.9135 / stage level 0.4164); RF-46d / RF-7d; LSTM / BiLSTM / Attention; **HMM (multinomial and Gaussian — the model family reported at 0.7697 on this very dataset by EDM 2025)**; MetaMamba and its four-way variants; and the three channels and fusion variants of CS1-SHR.\n\n"
        "#### 5.3.1 Same-protocol baseline protocol (new in v12)\n\n"
        "External review pointed out that earlier versions placed a *locally measured* LSTM next to a *quoted* GMM-HMM score obtained under a different protocol, and that the deep baselines received neither the tuning budget, the number of epochs, nor the input space of the evaluated model. From v12 on, every student-level architecture comparison in this paper is re-run under one identical protocol:\n\n"
        "| Protocol element | Unified specification (v12) |\n|---|---|\n"
        "| Data backbone | One cache: 473 students / 28,588,309 IDE events, encoded once |\n"
        "| Input window | The **last 128 events** per student (left padding); prefix variants in §6.23 |\n"
        "| Input spaces | 7d (event one-hot) / 11d (+ interval·deadline·part·exercise) / 28d (+ 21 behavioural & contextual channels, all past-only) |\n"
        "| Cross-validation | 5 folds × 3 seeds = 15 runs, grouped by student, stratified by label; OOF probabilities averaged over seeds |\n"
        "| Model selection | A **15% inner validation split inside each training fold** for early stopping and hyper-parameter choice; evaluation folds never participate in any choice |\n"
        "| Tuning budget | **4 candidates for every architecture (including MetaMamba itself)**, identical max_epochs = 200, patience = 20, batch = 32 |\n"
        "| Metrics | macro-F1, **balanced accuracy**, F1(FAIL), ROC-AUC; significance via paired Wilcoxon + McNemar + repetition-level bootstrap |\n"
        "| HMM implementation | **Self-implemented Gaussian-mixture-emission HMM** (Baum-Welch, diagonal covariance), one HMM per class plus class prior; hmmlearn 0.3.x removed GMM-HMM, so a single-Gaussian surrogate would systematically understate this baseline |\n"
        "| Tabular twin | RF receives student-level aggregates of the **same** input space (event counts + per-channel mean/std/last), decoupling architecture from information content |")

    new_sections = f"""
### 6.22 Same-protocol model × input-space matrix (new in v12)

Every architecture re-run under the protocol of §5.3.1 — no quoted numbers anywhere in this table.

{t_ident}

#### 6.22.1 De-confounded arm (part / exercise identity channels removed)

{t_deconf}

### 6.23 Early-warning curve (new in v12)

Each model re-trained and re-tested on the **first p%** of every student's event sequence (p ∈ {{10, 20, 40, 60, 100}}%):

{t_pref}

De-confounded arm:

{t_prefd}

### 6.24 Significance tests (new in v12)

Paired Wilcoxon over the 15 outer folds (Holm-Bonferroni within family) + exact McNemar on the pooled OOF predictions + repetition-level paired bootstrap (2000 resamples):

{t_sig}

### 6.25 FiLM ablation: multi-seed replication and mechanism (new in v12)

{{'TABLE_FILM'}}

### 6.26 LOPO cross-task transfer and few-shot evidence (new in v12)

{{'TABLE_LOPO'}}
"""
    new_sections = (new_sections.replace("{'TABLE_FILM'}", t_film)
                    .replace("{'TABLE_LOPO'}", t_lopo)
                    .replace('{TABLE_FILM}', t_film)
                    .replace('{TABLE_LOPO}', t_lopo))
    anchor = '\n## 7 Analysis and Discussion'
    assert anchor in s, 'anchor not found'
    s = s.replace(anchor, new_sections + anchor)

    io.open(DST, 'w', encoding='utf-8').write(s)
    print(f'[build_paper_v12_en] wrote {DST} ({len(s)} chars)')


if __name__ == '__main__':
    build()
