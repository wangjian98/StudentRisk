# Dummy task_id LOPO 实验报告

## 实验设计

- 3 种 FiLM 状态: `no_film` / `film_on` / `film_disabled`
- Dummy task_id 喂法: `true` / `zero(0)` / `-1` / `mean` / `random`
- 473 students × 5-fold StratifiedKFold, per-student OOF

## 整体指标汇总

| state | task_id_used | accuracy | f1_macro | roc_auc | pr_auc | n_students |
|---|---|---:|---:|---:|---:|---:|
| **film_disabled** | **N/A (γ=1,β=0)** | 0.6850 | 0.4777 | 0.8966 | 0.9562 | 473 |
| **film_on** | **mean** | 0.6617 | 0.3982 | 0.8911 | 0.9545 | 473 |
| **film_on** | **random** | 0.7484 | 0.6837 | 0.8787 | 0.9486 | 473 |
| **film_on** | **true** | 0.8837 | 0.8715 | 0.9188 | 0.9608 | 473 |
| **film_on** | **zero(0)** | 0.6638 | 0.3990 | 0.7593 | 0.8854 | 473 |
| **no_film** | **N/A** | 0.6808 | 0.4659 | 0.7114 | 0.8279 | 473 |

## Per-part breakdown (按 true_part 分组)

(见 per_part_breakdown.csv)
