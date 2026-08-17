# StudentRisk — 学业早期风险预测（CS1 课程）

基于学生 IDE 编程日志预测课程是否通过 / 挂科（Early Dropout/Failure Prediction）。

![Version](https://img.shields.io/badge/version-v4.0-blue) ![Status](https://img.shields.io/badge/status-active-success) ![Models](https://img.shields.io/badge/models-6-7d%2F11d-green) ![License](https://img.shields.io/badge/license-Academic-lightgrey)

> 🎉 **v4.0 已发布**（2026-08-17）：移除所有使用 46 维手工聚合特征的基线模型（RF / LSTM / BiLSTM / Attention），聚焦 **6 个 7-dim / 11-dim 序列模型** 的公平对比。
>
> **核心发现**：Meta-Mamba-7d（仅 7 维事件序列 + Mamba 架构）F1 = 0.9111，超过 RF-7d（F1=0.8911）+2.0%，证明 **原始事件序列 + 选择性状态空间** 比 **手工聚合** 更强；Meta-Mamba（11-dim）仅 +0.33% F1 边际增益。
>
> 详见 [`paper_v4_zh.md`](docs/paper_v4_zh.md) / [`paper_v4_en.md`](docs/paper_v4_en.md)（v4 主版本，含 5.6 节「实验结果可视化分析」）



## 数据集
- **来源**：CS1 公开课程日志（与 CodeEMO 项目共享）
- **样本**：473 名学生（其中 failed=314 / passed=159，不平衡比 ≈ 2:1）
- **原始事件**：28,588,310 条 IDE 事件，7 种事件类型
- **特征维度对比**：
  - **7 维原始事件序列**：7 种事件类型的 one-hot 序列（LSTM-7d / BiLSTM-7d / Attention-7d / Meta-Mamba-7d）+ 计数（RF-7d）
  - **11 维时序扩展**：7 维 one-hot + 时间间隔 + 截止距离 + 题号 + 练习号（仅 Meta-Mamba）

## Label 规范（重要）
- **Failed = 1**（挂科）
- **Passed = 0**（通过）
- 所有 OOF 概率 / 指标计算都遵循此约定
- 即 `probs[i] = P(failed=i)`，与 `passed.csv` 中 `passed` 字段互补

## 🎨 如何生成图（中文图表无乱码）

所有图表使用 `matplotlib` 生成。系统字体自动配置（`analysis/setup_fonts.py` 检测 `WenQuanYi Zen Hei` 文泉驿正黑），确保中文正确渲染。

### 一键生成全部 13 张图

```bash
# 1. v1 论文图（fig1-fig10，从 results/ 数据生成）
python -m analysis.generate_paper_figures

# 2. v3 专属图（fig11 跨维度对比、fig11b 升级路径、fig11c 效率散点）
python -m analysis.generate_fig11

# 3. 综合对比图（5 张，outputs/plots/ 下）
python -m analysis.visualize
```

### 输出位置

| 路径 | 内容 | 用途 |
|---|---|---|
| `docs/plots/paper/fig*.png` | 13 张论文图 | v3 论文 Figure 引用 |
| `outputs/plots/paper/fig*.png` | 10 张镜像图 | 与 analyze.py 兼容 |
| `outputs/plots/*.png` | 5 张对比图 | analyze.visualize 输出 |

### 中文字体自动应用

只要 `analysis/setup_fonts.py` 在项目根目录，`generate_paper_figures.py`、`generate_fig11.py`、`visualize.py` 都会在 `import pyplot` 之前自动调用 `setup_chinese_font()`，无需手动配置。

### 手动覆盖字体（如需要）

```python
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
```


## 项目结构

```
StudentRisk/
├── README.md
├── requirements.txt
├── main.py                      # 统一入口: --model {all,<10 models>}
├── configs/default.yaml         # 默认超参数
├── data/                        # 数据加载与特征工程
│   ├── __init__.py
│   └── data_loader.py           # 加载 IDE_logs + passed.csv, Failed=1 标签转换
├── models/                      # 6 个模型（3 类架构）
│   ├── base.py                  # 共享基类 + 评估工具
│   ├── rf7/{model,train}.py      # RF (7 维原始计数)
│   ├── lstm_7d/{model,train}.py   # LSTM (7 维事件序列)
│   ├── bilstm_7d/{model,train}.py  # BiLSTM (7 维事件序列)
│   ├── attention_7d/{model,train}.py   # Attention (7 维事件序列)
│   ├── meta_mamba/{model,train}.py  # Meta-Mamba (11 维事件序列 + S6 + FiLM + TC + FOMAML)
│   └── meta_mamba_7d/{model,train}.py # Meta-Mamba (7 维事件序列)
├── results/{rf7,meta_mamba,
│          lstm_7d,bilstm_7d,attention_7d,meta_mamba_7d}/   # 6 个模型结果
├── analysis/                    # 对比分析与可视化
│   ├── compare.py               # → comparison.csv / .md
│   ├── visualize.py             # → plots/*.png
│   └── generate_paper_figures.py  # → 6 张 paper figures
├── docs/                        # 论文与 figures
│   ├── paper_v4_zh.md / paper_v4_en.md  # ⭐ v4 主版本（移除 46 维，含 6 张图）
│   ├── paper_zh.md / paper_en.md # v3 早期版本（仍含 46 维对比）
│   ├── paper_v2_zh.md / paper_v2_en.md  # v2 7-dim 对比版
│   ├── paper_v1_zh.md / paper_v1_en.md  # v1 增强版
│   └── plots/paper/             # 6 张 paper figures
└── outputs/                     # 最终输出
    ├── comparison.csv           # 6 模型横向对比
    ├── comparison.md            # Markdown 报告
    └── plots/*.                    # 6 张对比图
```

## 模型分组（按输入特征）

| 特征维度 | 模型 | 描述 |
|---|---|---|
| **7 维原始计数** | `rf7` | sklearn RF + 7 维事件计数 |
| **7 维事件序列** | `lstm_7d` | LSTM + 7 维事件序列 (one-hot) |
| **7 维事件序列** | `bilstm_7d` | BiLSTM + 7 维事件序列 |
| **7 维事件序列** | `attention_7d` | Transformer + 7 维事件序列 |
| **7 维事件序列** | `meta_mamba_7d` | Meta-Mamba + 7 维事件序列 (S6+FiLM+TC+FOMAML) |
| **11 维时序扩展** | `meta_mamba` | Meta-Mamba + 11 维事件序列 (one-hot + 时间间隔 + 截止距离 + 题号 + 练习号) |

## 主结果（5-fold × 3 seeds OOF, threshold=0.5）

| 模型 | 输入维度 | n_params | F1(FAIL) | ROC-AUC |
|---|---|---|---|---|
| RF-7d | 7 维计数 | N/A | 0.8911 | 0.9178 |
| LSTM-7d | 7 维序列 | 33,857 | 0.7985 | 0.6302 |
| BiLSTM-7d | 7 维序列 | 67,201 | 0.8086 | 0.7080 |
| Attention-7d | 7 维序列 | 67,713 | 0.7995 | 0.7011 |
| **Meta-Mamba-7d** | **7 维序列** | **21,809** | **0.9111** | **0.9195** |
| **Meta-Mamba** | **11 维序列** | **22,065** | **0.9144** | **0.9290** |

**核心发现**：
- **Meta-Mamba-7d（仅7 维事件序列）F1=0.9111**，超过所有简单序列模型（LSTM-7d / BiLSTM-7d / Attention-7d 均 < 0.81）
- **Meta-Mamba-7d 也超过传统 RF-7d（0.8911）+4.0%**（F1），证明事件序列 + 选择性状态空间 优于 手工聚合
- **Meta-Mamba（11 维含时间间隔/截止距离/题号）F1=0.9144**，比 7 维版本仅高 0.33% —— 表明 **7 维事件序列 + Mamba 已捕获大部分信号**
- **简单 7 维序列模型（LSTM-7d/BiLSTM-7d/Attention-7d）F1 < 0.81**：时序信息必须有合适的架构（Selective SSM + FiLM + TC）才能发挥价值

## 快速开始

### 跑单个模型
```bash
cd ~/StudentRisk
python -m models.rf7.train                       # RF-7d
python -m models.lstm_7d.train                    # LSTM-7d
python -m models.bilstm_7d.train                  # BiLSTM-7d
python -m models.attention_7d.train               # Attention-7d
python -m models.meta_mamba_7d.train              # MetaMamba-7d

python -m models.lstm_7d.train                     # LSTM (7 维序列)
python -m models.bilstm_7d.train                   # BiLSTM (7 维序列)
python -m models.attention_7d.train                # Attention (7 维序列)
python -m models.rf7.train                        # RF (7 维计数)
python -m models.meta_mamba.train                 # MetaMamba (11 维)
```

### 跑全部模型（统一入口）
```bash
python main.py --model all               # 跑全部 10 个模型
python main.py --model rf7 lstm_7d bilstm_7d attention_7d meta_mamba meta_mamba_7d   # 仅 7-dim 与 11-dim 序列模型
python main.py --model rf7 meta_mamba meta_mamba_7d          # 选指定模型
python main.py --model meta_mamba_7d     # 单跑 MetaMamba-7d
```

### 查看对比 + 可视化
```bash
python -m analysis.compare     # 汇总各模型结果 → outputs/comparison.{csv,md}
python -m analysis.visualize   # 生成混淆矩阵 / ROC / PR / 对比柱状图
python -m analysis.generate_paper_figures   # 生成 6 张 paper figures
```

## 输出内容

每个模型 `results/<model>/` 包含：
- `results.json`：总体 + per-fold + per-class metrics + 配置
- `fold_metrics.csv`：每折明细
- `oof_probs.npy`：5-fold × 3 seeds OOF 概率 (473,)
- `labels.npy`：标签（failed=1 约定）
- `fold_idx.npy`：每样本所属 fold
- `config_used.json`：超参数快照

`outputs/` 汇总：
- `comparison.csv`：**10 模型** × 多个指标（per-class P/R/F1 + AUC + Macro-F1）
- `comparison.md`：Markdown 报告
- `plots/*.png`：6 张对比图（RF-7d / 4 个 7-dim 序列模型 / Meta-Mamba-7d）

`docs/` 论文：
- `paper_v4_zh.md` / `paper_v4_en.md`：⭐ v4 主版本（移除 46 维基线 + 实验结果可视化分析）
- `paper_v2_zh.md` / `paper_v2_en.md`：v2 增强版（7-dim 对比 + 10 figures + 公式推导）
- `paper_v1_zh.md` / `paper_v1_en.md`：v1 版本（基于 11-dim MetaMamba）
- `plots/paper/*.png`：6 张 paper figures

## 评估指标（每类 + 总体）

**每类（Failed=1 / Passed=0）**：
- Precision, Recall, F1, Support

**总体**：
- Accuracy, Macro-F1, Weighted-F1
- ROC-AUC, PR-AUC
- 混淆矩阵 (TN/FP/FN/TP)

**稳定性**：
- per-fold std（跨 15 folds = 5×3）

## 依赖
```
pandas>=1.5
numpy>=1.21
scikit-learn>=1.0
torch>=2.0
matplotlib>=3.5
pyyaml>=6.0
```

## 引用

```bibtex
@misc{wang2026metamamba,
  title={Meta-Mamba for Early Academic Risk Prediction in Programming Learners:
         Selective State Space, Task-Aware Modulation, and Few-Shot Meta-Learning},
  author={Wang, Jian},
  year={2026},
  url={https://github.com/wangjian98/StudentRisk},
}
```

## 仓库

https://github.com/wangjian98/StudentRisk
---

## 📌 版本历史

| 版本 | 日期 | 关键变更 |
|---|---|---|
| **v4.0** ⭐ 当前 | 2026-08-17 | **架构清洁版**：删除 4 个 46-dim 手工聚合特征模型（RF / LSTM / BiLSTM / Attention）+ 213 行 `data/features.py` + 28 个 results 文件。6 个保留模型（RF-7d / LSTM-7d / BiLSTM-7d / Attention-7d / Meta-Mamba-7d / Meta-Mamba）。新增 `paper_v4_zh.md` / `paper_v4_en.md` 主版本（含 5.6 节可视化分析）。 |
| v3.0 | 2026-08-16 | v3 完整论文版：970 行中文 / 983 行英文 + 13 张图 |
| v2.0 | 2026-08-15 | 10 模型对比版（5-fold × 3 seeds OOF）|
| v1.0 | 2026-08-15 | 初版：Meta-Mamba + 5 baselines |

完整引用审计记录：[docs/refs_audit_report.md](docs/refs_audit_report.md)
