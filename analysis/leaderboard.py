#!/usr/bin/env python3
"""
StudentRisk v6 — Unified Experiment Leaderboard (v2)
====================================================
Aggregates ALL experiments in /home/ubuntu/StudentRisk/results/ into a single
machine-readable CSV + a human-readable Markdown table.

Categories produced:
  A) Main models (6-model comparison)         → outputs/leaderboard_main.csv
  B) Ablation: 11d FiLM × TC × lambda         → outputs/leaderboard_ablation_11d.csv
  C) Ablation: 7d FiLM × TC × lambda          → outputs/leaderboard_ablation_7d.csv
  D) LR × D_model sweep                       → outputs/leaderboard_lr_dmodel.csv
  E) Batch size sweep (bs=8/16/32)            → outputs/leaderboard_bs.csv
  F) MaxLen sweep (ml=128/256/512/1024)       → outputs/leaderboard_maxlen.csv
  G) Unified leaderboard (best per category)  → outputs/leaderboard_unified.md

Usage:
  python analysis/leaderboard.py
"""
from __future__ import annotations
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path("/home/ubuntu/StudentRisk")
RESULTS = REPO / "results"
OUT = REPO / "outputs"
OUT.mkdir(exist_ok=True)


def _deep_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if cur is None or not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is default:
            return default
    return cur


def load_run(dirpath: Path) -> dict | None:
    rj = dirpath / "results.json"
    if not rj.exists():
        return None
    try:
        r = json.loads(rj.read_text())
    except Exception:
        return None
    o = r.get("overall", {})
    pfs = r.get("per_fold_summary", {})
    cfg = {}
    cj = dirpath / "config_used.json"
    if cj.exists():
        try:
            cfg_full = json.loads(cj.read_text())
            # Flatten meta_mamba_cfg into cfg for convenience
            if "meta_mamba_cfg" in cfg_full:
                cfg = dict(cfg_full.get("meta_mamba_cfg", {}))
                # Top-level scalars also matter
                for k in ("max_len", "n_tasks", "n_params", "feature_dimension",
                          "contrastive_weight", "batch_size", "lr", "d_model",
                          "use_film", "tc_weight"):
                    if k in cfg_full and k not in cfg:
                        cfg[k] = cfg_full[k]
            else:
                cfg = cfg_full
        except Exception:
            pass
    # Some sweep files also put ablation info at results.json top-level
    abl = r.get("ablation", {})
    use_film = abl.get("no_film", None)
    if use_film is False:
        use_film = True
    elif use_film is True:
        use_film = False
    no_tc = abl.get("no_tc", None)
    tc_weight_val = cfg.get("contrastive_weight", cfg.get("tc_weight", None))
    if "tcw" in cfg:
        tc_weight_val = cfg["tcw"]
    if no_tc is True:
        tc_weight_val = 0.0
    if use_film is None:
        use_film = cfg.get("use_film", None)

    return {
        "name": dirpath.name,
        "accuracy":   o.get("accuracy"),
        "macro_f1":   o.get("macro_f1"),
        "f1_class_0": o.get("f1_class_0"),  # PASSED
        "f1_class_1": o.get("f1_class_1"),  # FAILED (positive)
        "roc_auc":    o.get("roc_auc"),
        "pr_auc":     o.get("pr_auc"),
        "macro_f1_mean": pfs.get("macro_f1_mean"),
        "macro_f1_std":  pfs.get("macro_f1_std"),
        "f1_class_1_mean": pfs.get("f1_class_1_mean"),
        "f1_class_1_std":  pfs.get("f1_class_1_std"),
        "roc_auc_mean": pfs.get("roc_auc_mean"),
        "roc_auc_std":  pfs.get("roc_auc_std"),
        "n_seeds": r.get("n_seeds", o.get("n_seeds")),
        "n_splits": r.get("n_splits", o.get("n_splits")),
        "n_params": r.get("n_params", o.get("n_params")),
        "threshold": r.get("threshold", o.get("threshold")),
        "elapsed_sec": o.get("elapsed_sec"),
        "feature_dimension": r.get("feature_dimension"),
        "input_type": r.get("input_type"),
        "max_len": cfg.get("max_len"),
        "lr": cfg.get("lr"),
        "d_model": cfg.get("d_model"),
        "batch_size": cfg.get("batch_size"),
        "use_film": use_film,
        "tc_weight": tc_weight_val,
        "config": cfg,
    }


def fmt(x, w=6, p=4):
    if x is None:
        return " " * w
    if isinstance(x, float):
        if 0 < abs(x) < 1e-3:
            return f"{x:>{w}.{p+1}e}"
        return f"{x:>{w}.{p}f}"
    return f"{x!s:>{w}}"


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow([r.get(c) for c in columns])


def main() -> int:
    if not RESULTS.is_dir():
        print(f"[ERR] {RESULTS} not found", file=sys.stderr)
        return 1

    # ----- A) Main models -----
    main_keys = ["rf7", "lstm_7d", "bilstm_7d", "attention_7d", "meta_mamba_7d", "meta_mamba"]
    main_rows = []
    for k in main_keys:
        d = RESULTS / k
        if not d.is_dir():
            continue
        r = load_run(d)
        if r is None:
            continue
        r["model"] = k
        main_rows.append(r)

    main_cols = ["model", "accuracy", "macro_f1", "f1_class_0", "f1_class_1",
                 "roc_auc", "pr_auc", "macro_f1_mean", "macro_f1_std",
                 "roc_auc_mean", "roc_auc_std", "n_seeds", "n_params", "elapsed_sec"]
    write_csv(OUT / "leaderboard_main.csv", main_rows, main_cols)

    # ----- B) Ablation 11d -----
    abl_11d_keys = [
        ("meta_mamba", "full", True, 0.3),
        ("meta_mamba_ablation_no_tc", "no_tc", True, 0.0),
        ("meta_mamba_ablation_no_film", "no_film", False, 0.3),
        ("meta_mamba_ablation_no_film_no_tc", "no_film_no_tc", False, 0.0),
        ("meta_mamba_ablation_lam_0_1", "lam_0.1", True, 0.1),
        ("meta_mamba_ablation_lam_0_5", "lam_0.5", True, 0.5),
        ("meta_mamba_ablation_lam_1_0", "lam_1.0", True, 1.0),
    ]
    abl_11d_rows = []
    for k, variant, film, tcw in abl_11d_keys:
        d = RESULTS / k
        if not d.is_dir():
            continue
        r = load_run(d)
        if r is None:
            continue
        r["variant"] = variant
        r["use_film"] = film
        r["tc_weight"] = tcw
        r["max_len"] = r.get("max_len", 256)
        abl_11d_rows.append(r)

    abl_11d_cols = ["variant", "use_film", "tc_weight", "max_len", "accuracy",
                    "macro_f1", "f1_class_1", "roc_auc", "pr_auc",
                    "macro_f1_mean", "macro_f1_std",
                    "f1_class_1_mean", "f1_class_1_std",
                    "roc_auc_mean", "roc_auc_std", "n_seeds"]
    write_csv(OUT / "leaderboard_ablation_11d.csv", abl_11d_rows, abl_11d_cols)

    # ----- C) Ablation 7d -----
    abl_7d_keys = [
        ("meta_mamba_7d", "full", True, 0.3, 128),
        ("meta_mamba_7d_ablation_no_tc", "no_tc", True, 0.0, 128),
        ("meta_mamba_7d_ablation_no_film", "no_film", False, 0.3, 128),
        ("meta_mamba_7d_ablation_no_film_no_tc", "no_film_no_tc", False, 0.0, 128),
    ]
    abl_7d_rows = []
    for k, variant, film, tcw, ml in abl_7d_keys:
        d = RESULTS / k
        if not d.is_dir():
            continue
        r = load_run(d)
        if r is None:
            continue
        r["variant"] = variant
        r["use_film"] = film
        r["tc_weight"] = tcw
        r["max_len"] = ml
        abl_7d_rows.append(r)

    abl_7d_cols = ["variant", "use_film", "tc_weight", "max_len",
                   "accuracy", "macro_f1", "f1_class_1",
                   "roc_auc", "pr_auc",
                   "macro_f1_mean", "macro_f1_std",
                   "roc_auc_mean", "roc_auc_std", "n_seeds"]
    write_csv(OUT / "leaderboard_ablation_7d.csv", abl_7d_rows, abl_7d_cols)

    # ----- D) LR × D_model sweep -----
    sweep_keys = [
        ("sweep_lr0_0003_d32",  3e-4, 32),
        ("sweep_lr0_0003_d64",  3e-4, 64),
        ("sweep_lr0_0003_d128", 3e-4, 128),
        ("sweep_lr0_001_d32",   1e-3, 32),
        ("sweep_lr0_001_d64",   1e-3, 64),
        ("sweep_lr0_001_d128",  1e-3, 128),
        ("sweep_lr0_003_d128",  3e-3, 128),
    ]
    sweep_rows = []
    for k, lr_val, dm_val in sweep_keys:
        d = RESULTS / k
        if not d.is_dir():
            continue
        r = load_run(d)
        if r is None:
            continue
        r["variant"] = k
        r["lr"] = r.get("lr") or lr_val
        r["d_model"] = r.get("d_model") or dm_val
        r["max_len"] = r.get("max_len", 256)
        sweep_rows.append(r)

    sweep_cols = ["variant", "lr", "d_model", "max_len", "accuracy",
                  "macro_f1", "f1_class_1", "roc_auc", "pr_auc",
                  "macro_f1_mean", "macro_f1_std",
                  "f1_class_1_mean", "f1_class_1_std",
                  "roc_auc_mean", "roc_auc_std", "n_seeds"]
    write_csv(OUT / "leaderboard_lr_dmodel.csv", sweep_rows, sweep_cols)

    # ----- E) Batch size sweep (configured as 7d no_tc per config_used.json) -----
    bs_keys = [
        ("meta_mamba_7d_bs_8",  8),
        ("meta_mamba_7d_bs_16", 16),
        ("meta_mamba_7d_bs_32", 32),
    ]
    bs_rows = []
    for k, bs_val in bs_keys:
        d = RESULTS / k
        if not d.is_dir():
            continue
        r = load_run(d)
        if r is None:
            continue
        r["variant"] = k
        r["batch_size"] = r.get("batch_size") or bs_val
        # Detect no_tc from contrastive_weight=0.0
        r["use_film"] = r.get("use_film", True)
        r["tc_weight"] = r.get("tc_weight", 0.0)
        r["max_len"] = r.get("max_len", 128)
        bs_rows.append(r)

    bs_cols = ["variant", "batch_size", "use_film", "tc_weight", "max_len",
               "accuracy", "macro_f1", "f1_class_1", "roc_auc", "pr_auc",
               "macro_f1_mean", "macro_f1_std",
               "f1_class_1_mean", "f1_class_1_std",
               "roc_auc_mean", "roc_auc_std", "n_seeds"]
    write_csv(OUT / "leaderboard_bs.csv", bs_rows, bs_cols)

    # ----- F) MaxLen sweep -----
    maxlen_keys = [
        ("meta_mamba_7d",        128),
        ("meta_mamba_7d_ml256",  256),
        ("meta_mamba_7d_ml512",  512),
        ("meta_mamba_7d_ml1024", 1024),
    ]
    maxlen_rows = []
    for k, ml_val in maxlen_keys:
        d = RESULTS / k
        if not d.is_dir():
            continue
        r = load_run(d)
        if r is None:
            continue
        r["variant"] = k
        r["max_len"] = r.get("max_len") or ml_val
        r["use_film"] = r.get("use_film", True)
        r["tc_weight"] = r.get("tc_weight", 0.3)
        maxlen_rows.append(r)

    maxlen_cols = ["variant", "max_len", "use_film", "tc_weight",
                   "accuracy", "macro_f1", "f1_class_1", "roc_auc", "pr_auc",
                   "macro_f1_mean", "macro_f1_std",
                   "roc_auc_mean", "roc_auc_std", "n_seeds"]
    write_csv(OUT / "leaderboard_maxlen.csv", maxlen_rows, maxlen_cols)

    # ----- G) Unified leaderboard MD -----
    md = []
    md.append("# StudentRisk v6 — Unified Experiment Leaderboard\n")
    md.append("> Auto-generated by `analysis/leaderboard.py`. ")
    md.append("> Dataset: CS1 (n=473, fail_rate=0.6638). Protocol: 5-fold × 3 seeds (42/123/777). Threshold: 0.5.")
    md.append("> Label convention: **Failed=1 (positive class)**, Passed=0.\n")

    def best(rows, key):
        cand = [r for r in rows if r.get(key) is not None]
        if not cand:
            return None
        return max(cand, key=lambda r: r[key])

    def std_str(prefix, mean_key, std_key):
        m = prefix.get(mean_key)
        s = prefix.get(std_key)
        if m is None and s is None:
            return "    —     "
        return f"{fmt(m, 5, 4)} ± {fmt(s, 4, 4)}"

    md.append("## A. Main Models (6-model comparison)\n")
    md.append("| Model | Acc | MacroF1 | F1_pass | F1_fail | ROC-AUC | PR-AUC | MacroF1 ± std | ROC-AUC ± std | n_params |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in main_rows:
        m  = r["model"]
        a  = fmt(r.get("accuracy"), 6)
        f1 = fmt(r.get("macro_f1"), 6)
        fp = fmt(r.get("f1_class_0"), 6)
        ff = fmt(r.get("f1_class_1"), 6)
        roc = fmt(r.get("roc_auc"), 6)
        pr = fmt(r.get("pr_auc"), 6)
        ms = std_str(r, "macro_f1_mean", "macro_f1_std")
        rs = std_str(r, "roc_auc_mean", "roc_auc_std")
        np = str(r.get("n_params") or "—")
        md.append(f"| `{m}` | {a} | {f1} | {fp} | {ff} | {roc} | {pr} | {ms} | {rs} | {np} |")
    md.append("")

    md.append("## B. Ablation — 11d MetaMamba (FiLM × TC × lambda, max_len=256)\n")
    md.append("| Variant | FiLM | TC_w | Acc | MacroF1 | F1_fail | ROC-AUC | MacroF1 ± std |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in abl_11d_rows:
        v = r["variant"]
        film = str(r.get("use_film"))
        tcw  = f"{r.get('tc_weight'):.2f}" if r.get("tc_weight") is not None else "—"
        a  = fmt(r.get("accuracy"), 6)
        f1 = fmt(r.get("macro_f1"), 6)
        ff = fmt(r.get("f1_class_1"), 6)
        roc = fmt(r.get("roc_auc"), 6)
        ms = std_str(r, "macro_f1_mean", "macro_f1_std")
        md.append(f"| `{v}` | {film} | {tcw} | {a} | {f1} | {ff} | {roc} | {ms} |")
    md.append("")

    md.append("## C. Ablation — 7d MetaMamba (FiLM × TC, max_len=128)\n")
    md.append("| Variant | FiLM | TC_w | Acc | MacroF1 | F1_fail | ROC-AUC | MacroF1 ± std |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in abl_7d_rows:
        v = r["variant"]
        film = str(r.get("use_film"))
        tcw  = f"{r.get('tc_weight'):.2f}" if r.get("tc_weight") is not None else "—"
        a  = fmt(r.get("accuracy"), 6)
        f1 = fmt(r.get("macro_f1"), 6)
        ff = fmt(r.get("f1_class_1"), 6)
        roc = fmt(r.get("roc_auc"), 6)
        ms = std_str(r, "macro_f1_mean", "macro_f1_std")
        md.append(f"| `{v}` | {film} | {tcw} | {a} | {f1} | {ff} | {roc} | {ms} |")
    md.append("")

    md.append("## D. LR × D_model Sweep (MetaMamba-7d, max_len=128)\n")
    md.append("| Variant | lr | d_model | Acc | MacroF1 | F1_fail | ROC-AUC | MacroF1 ± std |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in sweep_rows:
        v = r["variant"]
        lr = f"{r.get('lr'):.4f}" if r.get("lr") is not None else "  -    "
        dm = str(r.get("d_model"))
        a  = fmt(r.get("accuracy"), 6)
        f1 = fmt(r.get("macro_f1"), 6)
        ff = fmt(r.get("f1_class_1"), 6)
        roc = fmt(r.get("roc_auc"), 6)
        ms = std_str(r, "macro_f1_mean", "macro_f1_std")
        md.append(f"| `{v}` | {lr} | {dm} | {a} | {f1} | {ff} | {roc} | {ms} |")
    md.append("")

    md.append("## E. Batch Size Sweep (MetaMamba-7d, max_len=128, **no_tc** config — `contrastive_weight=0.0`)\n")
    md.append("| Variant | bs | Acc | MacroF1 | F1_fail | ROC-AUC | MacroF1 ± std |")
    md.append("|---|---|---|---|---|---|---|")
    for r in bs_rows:
        v = r["variant"]
        bs = str(r.get("batch_size"))
        a  = fmt(r.get("accuracy"), 6)
        f1 = fmt(r.get("macro_f1"), 6)
        ff = fmt(r.get("f1_class_1"), 6)
        roc = fmt(r.get("roc_auc"), 6)
        ms = std_str(r, "macro_f1_mean", "macro_f1_std")
        md.append(f"| `{v}` | {bs} | {a} | {f1} | {ff} | {roc} | {ms} |")
    md.append("")

    md.append("## F. MaxLen Sweep (MetaMamba-7d, full config)\n")
    md.append("| Variant | max_len | Acc | MacroF1 | F1_fail | ROC-AUC | MacroF1 ± std |")
    md.append("|---|---|---|---|---|---|---|")
    for r in maxlen_rows:
        v = r["variant"]
        ml = str(r.get("max_len"))
        a  = fmt(r.get("accuracy"), 6)
        f1 = fmt(r.get("macro_f1"), 6)
        ff = fmt(r.get("f1_class_1"), 6)
        roc = fmt(r.get("roc_auc"), 6)
        ms = std_str(r, "macro_f1_mean", "macro_f1_std")
        md.append(f"| `{v}` | {ml} | {a} | {f1} | {ff} | {roc} | {ms} |")
    md.append("")

    md.append("## G. Highlights — Best Per Category\n")
    b_main = best(main_rows, "macro_f1")
    b_abl11 = best(abl_11d_rows, "macro_f1")
    b_abl7 = best(abl_7d_rows, "macro_f1")
    b_sweep = best(sweep_rows, "macro_f1")
    b_maxlen = best(maxlen_rows, "macro_f1")
    b_bs = best(bs_rows, "macro_f1")

    def line(label, r):
        if not r:
            return f"- **{label}**: _no data_"
        return (f"- **{label}** = `{r['name']}` → "
                f"Acc={fmt(r.get('accuracy'), 5)} "
                f"MacroF1={fmt(r.get('macro_f1'), 5)} "
                f"F1_fail={fmt(r.get('f1_class_1'), 5)} "
                f"ROC-AUC={fmt(r.get('roc_auc'), 5)}")

    md.append(line("Main model", b_main))
    md.append(line("11d ablation best (Macro-F1)", b_abl11))
    md.append(line("7d ablation best (Macro-F1)", b_abl7))
    md.append(line("LR × D_model sweep best", b_sweep))
    md.append(line("Batch size sweep best", b_bs))
    md.append(line("MaxLen sweep best", b_maxlen))
    md.append("")
    md.append("**Important caveat:** Batch size sweep was actually run with `contrastive_weight=0.0` (no_tc). ")
    md.append("So the bs-best (`bs=8`, F1=0.8801) reflects the combined `no_tc + bs=8` configuration, ")
    md.append("which is +0.86% over the full default (`bs=16, full`, F1=0.8715).\n")

    (OUT / "leaderboard_unified.md").write_text("\n".join(md))

    print("=== leaderboard generated ===")
    print(f"  main rows:     {len(main_rows)}")
    print(f"  abl_11d rows:  {len(abl_11d_rows)}")
    print(f"  abl_7d rows:   {len(abl_7d_rows)}")
    print(f"  sweep rows:    {len(sweep_rows)}")
    print(f"  bs rows:       {len(bs_rows)}")
    print(f"  maxlen rows:   {len(maxlen_rows)}")
    print(f"  → outputs/leaderboard_unified.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())