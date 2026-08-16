# StudentRisk — 学业早期风险预测（CS1 课程）

基于学生 IDE 编程日志预测课程是否通过 / 挂科（Early Dropout/Failure Prediction）。

![Version](https://img.shields.io/badge/version-v3.1-blue) ![Status](https://img.shields.io/badge/status-audited-success) ![Papers](https://img.shields.io/badge/papers-32%20references-orange) ![License](https://img.shields.io/badge/license-Academic-lightgrey)

> 🎉 **v3.1 已发布**（2026-08-17）：经合作导师王健两轮审计（3 篇替换 + 6 处细节修正 + 22 处正文引用同步），参考文献从 39 篇降至 **32 篇真实可信文献**，编号连续 [1]-[32]。详见 [v3.1 Release Notes](https://github.com/wangjian98/StudentRisk/releases/tag/v3.1) 和 [审计修订记录](docs/refs_audit_report.md)。



## 数据集
- **来源**：CS1 公开课程日志（与 CodeEMO 项目共享）
- **样本**：473 名学生（其中 failed=314 / passed=159，不平衡比 ≈ 2:1）
- **原始事件**：28,588,310 条 IDE 事件，7 种事件类型
- **特征维度对比**：
  - **7 维原始**：7 种事件类型的 one-hot 序列 / 计数（无聚合）
  - **11 维时序**：7 维 one-hot + 时间间隔 + 截止距离 + 题号 + 练习号（仅 MetaMamba）
  - **46 维聚合**：28 事件统计 + 10 行为轨迹 + 6 情绪复合 + 2 元信息（paper-aligned 手工特征）

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
│   ├── data_loader.py           # 加载 IDE_logs + passed.csv, Failed=1 标签转换
│   └── features.py              # 46 维手工特征
├── models/                      # 10 个模型（5 类）
│   ├── base.py                  # 共享基类 + 评估工具
│   ├── rf/{model,train}.py        # RF (46 维聚合特征)
│   ├── rf7d/{model,train}.py      # RF (7 维原始计数)
│   ├── lstm/{model,train}.py      # LSTM (46 维聚合 → 1-step seq)
│   ├── lstm_7d/{model,train}.py   # LSTM (7 维事件序列)
│   ├── bilstm/{model,train}.py     # BiLSTM (46 维聚合)
│   ├── bilstm_7d/{model,train}.py  # BiLSTM (7 维事件序列)
│   ├── attention/{model,train}.py  # Attention (46 维聚合)
│   ├── attention_7d/{model,train}.py   # Attention (7 维事件序列)
│   ├── meta_mamba/{model,train}.py  # Meta-Mamba (11 维事件序列 + S6 + FiLM + TC + FOMAML)
│   └── meta_mamba_7d/{model,train}.py # Meta-Mamba (7 维事件序列)
├── results/{rf,rf7,lstm,bilstm,attention,meta_mamba,
│          lstm_7d,bilstm_7d,attention_7d,meta_mamba_7d}/   # 10 个模型结果
├── analysis/                    # 对比分析与可视化
│   ├── compare.py               # → comparison.csv / .md
│   ├── visualize.py             # → plots/*.png
│   └── generate_paper_figures.py  # → 10 张 paper figures
├── docs/                        # 论文与 figures
│   ├── paper.md                 # 早期版本
│   ├── paper_zh.md / paper_en.md # 早期中英版本
│   ├── paper_v1_zh.md / paper_v1_en.md  # v1 增强版（含 10 张图 + 公式推导）
│   ├── paper_v2_zh.md / paper_v2_en.md  # v2 新增 7-dim 对比 + MetaMamba-7d
│   └── plots/paper/             # 10 张 paper figures
└── outputs/                     # 最终输出
    ├── comparison.csv           # 10 模型横向对比
    ├── comparison.md            # Markdown 报告
    └── plots/*.                    # 5 张对比图
```

## 模型分组（按输入特征）

| 特征维度 | 模型 | 描述 |
|---|---|---|
| **7 维原始** | `rf7d` | sklearn RF + 7 维事件计数 |
| **7 维原始** | `lstm_7d` | LSTM + 7 维事件序列 (one-hot) |
| **7 维原始** | `bilstm_7d` | BiLSTM + 7 维事件序列 |
| **7 维原始** | `attention_7d` | Transformer + 7 维事件序列 |
| **7 维原始** | `meta_mamba_7d` | Meta-Mamba + 7 维事件序列 (S6+FiLM+TC+FOMAML) |
| **46 维聚合** | `rf` | sklearn RF + 46 维手工特征 |
| **46 维聚合** | `lstm` | LSTM + 46 维聚合 → 1-step seq |
| **46 维聚合** | `bilstm` | BiLSTM + 46 维聚合 |
| **46 维聚合** | `attention` | Transformer + 46 维聚合 |
| **11 维时序** | `meta_mamba` | Meta-Mamba + 11 维事件序列 (one-hot + 4 连续特征) |

## 主结果（5-fold × 3 seeds OOF, threshold=0.5）

| 模型 | 输入维度 | n_params | F1(FAIL) | ROC-AUC |
|---|---|---|---|---|
| Random Forest | 46 维聚合 | N/A | 0.8795 | 0.9162 |
| RF-7d | **7 维计数** | N/A | 0.8911 | 0.9178 |
| LSTM | 46 维聚合 | 36,353 | 0.8805 | 0.9272 |
| BiLSTM | 46 维聚合 | 69,697 | 0.8797 | 0.9293 |
| Attention | 46 维聚合 | 70,209 | 0.8840 | 0.9293 |
| **Meta-Mamba** | **11 维序列** | **22,065** | **0.9144** | **0.9290** |
| LSTM-7d | **7 维序列** | 33,857 | 0.7985 | 0.6302 |
| BiLSTM-7d | **7 维序列** | 67,201 | 0.8086 | 0.7080 |
| Attention-7d | **7 维序列** | 67,713 | 0.7995 | 0.7011 |
| **MetaMamba-7d** | **7 维序列** | **21,809** | **0.9111** | **0.9195** |

**核心发现**：
- **MetaMamba-7d（仅7 维序列）F1=0.9111**，已超过 RF-7d（0.8911）、LSTM/BiLSTM/Attention（46 维聚合）的所有结果
- **7 维事件序列 + MetaMamba 架构 ≈ 11 维 MetaMamba**（差 -0.33% F1）
- **7 维事件序列 + 简单 LSTM/BiLSTM/Attention** 表现很差（macro_F1 ~0.5）—— 时序信息必须有合适的架构（Selective SSM + FiLM + TC）才能发挥价值

## 快速开始

### 跑单个模型
```bash
cd ~/StudentRisk
python -m models.rf7.train                       # RF-7d
python -m models.lstm_7d.train                    # LSTM-7d
python -m models.bilstm_7d.train                  # BiLSTM-7d
python -m models.attention_7d.train               # Attention-7d
python -m models.meta_mamba_7d.train              # MetaMamba-7d

python -m models.rf.train                         # RF (46 维)
python -m models.lstm.train                       # LSTM (46 维)
python -m models.bilstm.train                     # BiLSTM (46 维)
python -m models.attention.train                  # Attention (46 维)
python -m models.meta_mamba.train                 # MetaMamba (11 维)
```

### 跑全部模型（统一入口）
```bash
python main.py --model all               # 跑全部 10 个模型
python main.py --model rf7d rf_7d lstm_7d attention_7d meta_mamba_7d   # 仅 7 维
python main.py --model rf lstm bilstm attention meta_mamba            # 原始 6 个
python main.py --model meta_mamba_7d     # 单跑 MetaMamba-7d
```

### 查看对比 + 可视化
```bash
python -m analysis.compare     # 汇总各模型结果 → outputs/comparison.{csv,md}
python -m analysis.visualize   # 生成混淆矩阵 / ROC / PR / 对比柱状图
python -m analysis.generate_paper_figures   # 生成 10 张 paper figures
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
- `plots/*.png`：5 张对比图（含 7-dim 与 46-dim 分组）

`docs/` 论文：
- `paper_v2_zh.md` / `paper_v2_en.md`：v2 增强版（含 7-dim 对比 + 10 张 figures + 详细公式推导）
- `paper_v1_zh.md` / `paper_v1_en.md`：v1 版本（基于 11-dim MetaMamba）
- `plots/paper/*.png`：10 张 paper figures

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
| **v3.1** ⭐ 当前 | 2026-08-17 | **审计版**：王健复核后 3 篇替换（Angulo→Alhothali、Hayward→Leinonen、Shum→Azcona）+ 6 处细节修正（Mamba ICLR 删除、CAML 改 2019、TabPFN 改 Nature+DOI、AutoML 删"新世纪版 2024"等）+ 22 处正文引用同步。32 篇真实文献，编号连续 [1]-[32]。 |
| v3.0 | 2026-08-16 | v3 完整论文版：970 行中文 / 983 行英文 + 13 张图 |
| v2.0 | 2026-08-15 | 10 模型对比版（5-fold × 3 seeds OOF）|
| v1.0 | 2026-08-15 | 初版：Meta-Mamba + 5 baselines |

完整引用审计记录：[docs/refs_audit_report.md](docs/refs_audit_report.md)
