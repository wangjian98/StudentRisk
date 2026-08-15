# StudentRisk — 学业早期风险预测（CS1 课程）

基于学生 IDE 编程日志预测课程是否通过 / 挂科（Early Dropout/Failure Prediction）。

## 数据集
- **来源**：CS1 公开课程日志（来自 CodeEMO 项目）
- **样本**：473 名学生（其中 failed=314 / passed=159，不平衡比 ≈ 2:1）
- **原始事件**：28,588,310 条 IDE 事件，7 种事件类型
- **特征维度**：46 维（28 事件基础统计 + 10 行为轨迹 + 6 情绪复合 + 2 元信息）

## Label 规范（重要）
- **Failed = 1**（挂科）
- **Passed = 0**（通过）
- 所有 OOF 概率 / 指标计算都遵循此约定
- 即 `probs[i] = P(failed=i)`，与 `passed.csv` 中 `passed` 字段互补

## 项目结构
```
StudentRisk/
├── main.py                      # 统一入口: --model {all,rf,lstm,bilstm,attention}
├── configs/default.yaml         # 默认超参数
├── data/                        # 数据加载与特征工程
├── models/{rf,lstm,bilstm,attention}/   # 4 个模型（各自可独立运行）
├── results/{rf,lstm,bilstm,attention}/  # 每个模型独立结果目录
├── analysis/                    # 对比分析与可视化
└── outputs/                     # 最终汇总报告
```

## 快速开始

### 跑单个模型
```bash
cd ~/StudentRisk
python -m models.rf.train                        # 训练 RF
python -m models.lstm.train --epochs 50 --seeds 42 123  # 训练 LSTM（自定义）
python -m models.bilstm.train
python -m models.attention.train
```

### 跑全部模型（统一入口）
```bash
python main.py --model all                # 跑全部 4 个模型
python main.py --model rf lstm            # 跑指定模型
python main.py --model all --seeds 42     # 全部使用 seed 42
```

### 查看对比 + 可视化
```bash
python -m analysis.compare     # 汇总各模型结果 → outputs/comparison.{csv,md}
python -m analysis.visualize   # 生成混淆矩阵 / ROC / PR / 对比柱状图
```

## 输出内容

每个模型 `results/<model>/` 包含：
- `results.json`：总体 + per-fold + per-class metrics + 配置
- `fold_metrics.csv`：每折明细
- `oof_probs.npy`：5-fold × 3 seeds OOF 概率 (473,)
- `labels.npy`：标签（failed=1 约定）
- `fold_idx.npy`：每样本所属 fold
- `plots/confusion_matrix.png`、`roc_curve.png`、`pr_curve.png`

`outputs/` 汇总：
- `comparison.csv`：4 模型 × 多个指标（per-class P/R/F1 + AUC + Macro-F1）
- `comparison.md`：Markdown 报告
- `plots/metric_comparison.png`、`roc_curves_all.png`、`pr_curves_all.png`

## 模型说明

| 模型 | 类别 | 描述 | 复杂度 |
|---|---|---|---|
| RF | 树模型 | sklearn RandomForest, 46-dim 直接输入 | 中 |
| LSTM | 序列 | 46-dim → MLP(seq_len=1) → 单向 LSTM | 中 |
| BiLSTM | 序列 | 46-dim → MLP → 双向 LSTM | 中 |
| ATTENTION | 序列 | 46-dim → 投影 → 多头自注意力 (Transformer Encoder) | 中 |

## 依赖
```
pandas>=1.5
numpy>=1.21
scikit-learn>=1.0
torch>=2.0
matplotlib>=3.5
pyyaml>=6.0
```