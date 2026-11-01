# FiLM 推理时 part_id 依赖性分析

## 实验设计

- 数据: 473 students × 7 parts (in-distribution, 全部 7 part 训过)
- 训练: 5-fold StratifiedKFold × 3 seeds × 30 epochs × max_len=128
- 推理: 每折 val set 上用 **4 种 FiLM context** 喂入, 评估 val_acc / AUC / F1

### 4 种 FiLM context

| context | 含义 |
|---|---|
| **one_hot_true** | 学生自己的 part_id (baseline, 最理想) |
| **one_hot_wrong** | 用其他 part 的 task_emb 平均 (模拟 part_id 错配) |
| **mean_context** | 全部 7 part embedding 平均 (无 part 区分) |
| **random_uniform** | [0,1)^d 随机向量 (noise) |

## 📊 评测结果 (3 seeds × 5 folds = 15 runs)

| context | val_acc (mean±std) | val_f1_macro | val_auc | separation |
|---|---:|---:|---:|---:|
| **one_hot_true** | **0.8879±0.0169** | 0.8756±0.0169 | 0.9340±0.0185 | 0.5893±0.0211 |
| **mean_context** | **0.6738±0.0357** | 0.4298±0.0980 | 0.8948±0.0280 | 0.2053±0.0723 |
| **one_hot_wrong** | **0.6667±0.0125** | 0.4267±0.0471 | 0.3011±0.1572 | -0.0479±0.0458 |
| **random_uniform** | **0.6285±0.1327** | 0.5132±0.1717 | 0.8887±0.0316 | 0.1052±0.0573 |

### Context 敏感度 (Δ vs one_hot_true = 0.8879)

| context | Δ acc | 结论 |
|---|---:|---|
| mean_context | -0.2142 | ❌ FiLM 严重依赖正确 part_id |
| one_hot_wrong | -0.2212 | ❌ FiLM 严重依赖正确 part_id |
| random_uniform | -0.2594 | ❌ FiLM 严重依赖正确 part_id |

## 🔍 分析

### 关键发现

详见 `full_results_dump.md` 和 `summary.csv`

## 📁 输出文件

- `all_runs.csv` — 60 runs 完整指标
- `summary.csv` — 4 context 聚合
- `film_inference_ablation_report.md` — 本报告
- `seed{}_fold{}.pt` — 15 个 trained models

## ⏱ 训练时间

- 总耗时: 1334s (22.2 min)