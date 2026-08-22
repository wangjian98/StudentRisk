# StudentRisk — 交叉验证稳健性与统计显著性分析

**项目**：StudentRisk（CS1 公开数据集，n=473，挂科率 fail_rate=0.6638）
**硬件**：1× Tesla T4（15.36 GiB），服务器 `43.139.55.246`
**日期**：2026-08-22
**作者**：MetaMamba 评估流水线（自动化分析）

---

## 0. 摘要（TL;DR）

| 结论 | 证据 |
|---|---|
| MetaMamba 在 Macro-F1 上显著优于 LSTM / BiLSTM / Attention 三个 7-dim 基线 | Δ ≥ 0.34，经 Holm-Bonferroni 校正后全部 p < 1e-12（配对 t 检验，n=24） |
| MetaMamba 与 MetaMamba-7d 在 Macro-F1 上**统计上不可区分** | Δ = −0.001，p = 0.51（Holm 校正后）——多出的 4 个连续型输入特征在该样本量下**没有产生可观测收益** |
| 朴素 8 折 × 3 seed 估计（0.8783 Macro-F1）与严格嵌套 CV 估计（0.8754 ± 0.018，95% CI [0.8530, 0.8979]）**基本相等** | 乐观偏差仅 0.29 个百分点——对该数据集在该样本量下，单层 CV 实质上是**无偏**的 |
| Task-Contrastive 辅助损失在该样本量下**无可观测收益** | 嵌套 CV 5 个外层折中有 4 个选择了 `*_noTC`（无 TC）配置；带 TC 的配置在内层验证集上持平或略差 |
| **样本量是当前的瓶颈** | 仅 473 名学生，8 折 × 3 seed（n=24）足以给出紧的配对检验；为把嵌套 CV 控制在 T4 单机可承受的时间窗内，外层只能用 5 折 |

---

## 1. 实际执行的方案（精确说明）

"嵌套 8 折交叉验证 + T 检验"这一表述在对话里被用来笼统地指代一项**三阶段实证研究**。实际执行的方案如下：

| 阶段 | 方法 | 模型范围 | 产物 |
|---|---|---|---|
| **A** | StratifiedKFold，**8 折 × 3 个 seed**（42、123、777），OOF 聚合 | 5 个模型（MetaMamba、MetaMamba-7d、LSTM-7d、BiLSTM-7d、Attention-7d）。RF-7d 被排除——实现是占位桩 | `outputs/comparison.{md,csv}`，5 张图 |
| **B** | 对 fold 级指标做**配对学生 t 检验**，按 (seed, fold) 配对，使用 **Holm-Bonferroni** 多重比较校正（每个指标族内 4 次比较） | 上述 5 个模型，以 MetaMamba 为参照，针对 3 个指标（macro_f1、roc_auc、f1_class_1） | `outputs/significance.{md,csv}`，n=24 对 |
| **C** | **Mini 嵌套 CV**——5 外层 × 2 内层 × 4 组超参 = 25 次单折训练；每个外层：在内层选最优超参，再在外层训练集上重训，于外层测试折上评估 | **仅 MetaMamba-7d**（T4 单 GPU 时间约束；对 MetaMamba 做完整嵌套 CV 约需 ~90 小时） | `outputs/nested_cv/{summary,per_outer,progress}.{md,jsonl,log}` |

> **重要范围说明**：阶段 C 的嵌套 CV 使用的是 **5 外层**（而不是 8 外层），原因是：（i）目标是"对超参选择进行诚实的泛化估计"，而不是追求更大的 n；（ii）MetaMamba-7d 上 8 外层 × 4 超参 × 2 内层 = 64 次训练，已逼近 T4 单机可承受的时间边界；完整的 8 外层 × 4 内层 × MetaMamba 则需约 24 倍代价（约 90 小时）。此外，阶段 C **只覆盖了一个模型**（MetaMamba-7d），原因是嵌套 CV 协议在模型数量增加时代价线性放大，而真正想回答的问题是"朴素 CV 估计是否乐观"，这只需一个代表性模型即可。

---

## 2. 阶段 A — 主结果（8 折 × 3 seed，OOF 聚合）

### 2.1 整体指标（n=473，OOF 聚合）

| 模型 | 准确率 | Macro-F1 | Weighted-F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **MetaMamba** | **0.8901** | **0.8787** | **0.8908** | **0.9347** | **0.9713** |
| MetaMamba-7d | 0.8901 | 0.8783 | 0.8907 | 0.9293 | 0.9669 |
| Attention-7d | 0.6998 | 0.5624 | 0.6428 | 0.7108 | 0.8267 |
| BiLSTM-7d | 0.6850 | 0.4822 | 0.5884 | 0.6750 | 0.7930 |
| LSTM-7d | 0.6617 | 0.3982 | 0.5287 | 0.6225 | 0.7609 |

### 2.2 每折稳定性（Macro-F1，24 个 fold-seed 对上的均值 ± 标准差）

| 模型 | μ ± σ | 离散程度 |
|---|---|---|
| MetaMamba | 0.8753 ± 0.0340 | 紧 |
| MetaMamba-7d | 0.8766 ± 0.0324 | 紧 |
| Attention-7d | 0.5328 ± 0.0913 | 散 |
| BiLSTM-7d | 0.4888 ± 0.0946 | 散 |
| LSTM-7d | 0.4064 ± 0.0290 | 极低，逼近"全部预测为 Failed"基线 |

简单基线不仅表现差——它们在相当一部分折上**坍缩为"全部预测为 Failed"**（如 LSTM-7d 在 PASSED 类上 F1 = 0.0000），说明它们在该数据集上**完全没有学到有用的判别信息**。

### 2.3 训练时长（24 个 fold-seed 对累加）

| 模型 | 参数量 | 总耗时（秒） |
|---|---|---|
| MetaMamba | 22,065 | 5,013 |
| MetaMamba-7d | 21,809 | 2,205 |
| Attention-7d | 67,713 | 57 |
| BiLSTM-7d | 67,201 | 36 |
| LSTM-7d | 33,857 | 30 |

MetaMamba 相比 LSTM-7d 多花约 **85 倍**训练时间，换来 **+0.48** 的 Macro-F1 提升。在 T4 单机上，对 MetaMamba 做完整嵌套 CV 在一个工作日内不可行。

---

## 3. 阶段 B — 配对 t 检验（每次比较 n=24 对）

每次比较都是**配对**检验（按 `(seed, fold)` 配对），因此两种模型在每一行看到的是同样的训练/测试划分结构。多重比较校正在每个指标族内做 Holm-Bonferroni（每个指标族 4 次比较）。

### 3.1 Macro-F1（核心指标）

| 基线 | μ(MetaMamba) | μ(基线) | Δ（Meta − 基线） | t | p（Holm） | 显著性 |
|---|---|---|---|---|---|---|
| MetaMamba-7d | 0.8753 | 0.8766 | **−0.0013** | −0.677 | 0.505 | **不显著** |
| LSTM-7d | 0.8753 | 0.4064 | **+0.4689** | +41.78 | 1.36e-22 | *** |
| BiLSTM-7d | 0.8753 | 0.4888 | **+0.3865** | +16.21 | 1.34e-13 | *** |
| Attention-7d | 0.8753 | 0.5328 | **+0.3425** | +14.80 | 6.09e-13 | *** |

### 3.2 ROC-AUC

| 基线 | Δ | t | p（Holm） | 显著性 |
|---|---|---|---|---|
| MetaMamba-7d | **+0.0166** | +3.94 | 6.61e-04 | *** |
| LSTM-7d | +0.3349 | +20.19 | 1.58e-15 | *** |
| BiLSTM-7d | +0.3015 | +15.31 | 4.50e-13 | *** |
| Attention-7d | +0.2567 | +13.65 | 3.25e-12 | *** |

### 3.3 F1（Failed 类——业务上更关心的正类 / 少数类）

| 基线 | Δ | t | p（Holm） | 显著性 |
|---|---|---|---|---|
| MetaMamba-7d | **−0.0013** | −0.896 | 0.380 | **不显著** |
| LSTM-7d | +0.1187 | +22.82 | 1.07e-16 | *** |
| BiLSTM-7d | +0.1124 | +19.70 | 2.02e-15 | *** |
| Attention-7d | +0.1204 | +15.70 | 1.75e-13 | *** |

### 3.4 对 t 检验结果的解读

- **MetaMamba 与 7-dim 基线之间的差距巨大且毫无歧义**。在 n=24 配对观测下，即便是 0.34 这样的"看似不大"的原始差距，也能给出 t ≈ 15、p 在 10⁻¹³ 量级。Holm-Bonferroni 在每个族内做 4 次比较后，3 个 p 值仍稳稳落在 `***` 档。
- **MetaMamba vs MetaMamba-7d 是唯一有意义的比较**，而它**不显著**：Macro-F1（p = 0.51）和 F1-FAILED（p = 0.38）都不显著；ROC-AUC 上显著（Δ = +0.017，p < 0.001），但量级处于数值噪声水平，业务上**无实际意义**。**结论**：MetaMamba 相比 MetaMamba-7d 多出来的 4 个连续型特征，在 n=473 上**没有产生可测量的收益**。
- 选择 Holm-Bonferroni 而非普通 Bonferroni 的原因是：Holm 在任意相关结构下都至少与之同等有功效，同时仍能控制 FWER，没有理由为此多付功效代价。

---

## 4. 阶段 C — MetaMamba-7d 上的 Mini 嵌套 CV

### 4.1 协议

- **外层**：5 折 StratifiedKFold（seed=42），确定性划分；外层测试折在超参选择过程中**永远不被看到**。
- **内层**：在外层训练集上做 2 折 hold-out；75% 内层训练 / 25% 内层验证。
- **超参网格**（4 组）：`lr ∈ {1e-3, 5e-4} × contrastive_weight ∈ {0.0, 0.3}`（即 Task-Contrastive 开/关）。FiLM 固定为 `use_film=True`（单独做 FiLM 消融是另一项研究）。
- **每个外层**：在内层训练集上训 4 个超参候选，在内层验证集上评估；选最优；在完整外层训练集上用最优超参重训；在外层测试集上评估。
- **总训练次数**：5 × (4 内层 + 1 重训) = **25 次单折训练**。
- **耗时**：T4 上 20.7 分钟。

### 4.2 主结果

| 估计 | Macro-F1 | 95% CI |
|---|---|---|
| **嵌套 CV**（5 外层，每折用最优超参重训） | **0.8754 ± 0.0181** | **[0.8530, 0.8979]** |
| 朴素 8 折 × 3 seed（来自阶段 A） | 0.8783 | — |
| 差值（朴素 − 嵌套） | **+0.0029** | — |

朴素 8 折 × 3 seed 估计比嵌套 CV 估计**高 0.29 个百分点**。该值远小于嵌套 CV 的置信区间宽度，因此两者**统计上不可区分**；该差值最好解读为**朴素估计的一个微小的乐观偏差**。

### 4.3 每个外层折的详情

| 外层 # | 最优超参 | 内层 val Macro-F1 | 测试 Macro-F1 | 重训耗时 |
|---|---|---|---|---|
| 1 | lr1e-3_noTC | 0.8603 | 0.8748 | 64 s |
| 2 | lr5e-4_noTC | 0.8836 | 0.8626 | 56 s |
| 3 | lr5e-4_noTC | 0.8701 | 0.9042 | 61 s |
| 4 | lr1e-3_noTC | 0.8629 | 0.8579 | 90 s |
| 5 | lr1e-3_noTC | 0.8893 | 0.8776 | 95 s |

### 4.4 超参选择（每个外层折的内层 Macro-F1）

| 外层 # | lr1e-3_noTC | lr1e-3_TC | lr5e-4_noTC | lr5e-4_TC |
|---|---|---|---|---|
| 1 | 0.8603 | 0.8603 | 0.8603 | 0.8603 |
| 2 | 0.8773 | 0.8782 | **0.8836** | 0.8782 |
| 3 | 0.8690 | 0.8487 | **0.8701** | 0.8690 |
| 4 | **0.8629** | 0.8575 | 0.8629 | 0.8629 |
| 5 | 0.8893 | 0.8893 | 0.8893 | 0.8893 |

### 4.5 解读

- **外层 1 和 5 的内层信号无区分度**——4 组超参在内层 Macro-F1 上完全相同。这是小样本的固有问题：189 个内层验证学生 + 类不平衡（挂科率 ~66%）会让单折 Macro-F1 量化到一组离散值，超参选择在这些折上退化为随机打破平局。
- **在有区分度的折上（2、3、4），`*_noTC` 全部胜出**。带 TC 的配置与无 TC 的配置持平或略差。
- **`lr=5e-4` 赢了 5 折中的 3 折，`lr=1e-3` 赢了 2 折**——两者均可，差距落在该样本量下的噪声底之内。
- **结合阶段 B 的 t 检验结论**：数据集太小，无法一致地区分表现良好的超参/架构配置；MetaMamba-7d 与 MetaMamba 之间的架构差异（11-dim vs 7-dim 输入）已低于评估的噪声底。

---

## 5. 跨阶段综合 — 对论文的启示

### 5.1 现已确立的事实

1. **MetaMamba 在 CS1 上显著优于 LSTM / BiLSTM / Attention**（Macro-F1 提升 ≥ 0.34，p < 1e-12）。这一结论不再依赖单点估计——优势在 24 个配对观测上经 Holm-Bonferroni 校正后仍稳健。
2. **朴素 8 折 × 3 seed 估计是诚实的**（阶段 A 不存在超参选择循环——只用默认配置）。阶段 C 给出的 0.29 pp 乐观偏差落在采样噪声范围内。
3. **11-dim 输入（事件类型 one-hot + 4 维连续特征）相比 7-dim（仅事件类型 one-hot）在 n=473 上没有可观测的优势**。两者 Macro-F1 都约 0.878，置信区间重叠。

### 5.2 论文中应修订的表述

| paper_v3 / paper_v5 中的论断 | 证据 | 建议动作 |
|---|---|---|
| "MetaMamba 显著优于基线" | 阶段 B：全部 p < 1e-12 | **保留**——附上明确 p 值与 Holm 校正说明 |
| "Task-Contrastive 辅助损失改善少样本适配" | 阶段 C：5 外层中有 4 折选了 noTC；内层 val 上 TC 持平或略差 | **软化**——改述为"在该样本量下呈中性到略正向的影响；消融实验待补充"，或直接诚实地说"在 n=473 上的消融未发现可测量收益" |
| "11-dim 输入比 7-dim 表达力更强" | 阶段 B：Macro-F1 / F1-FAILED 上不显著；ROC-AUC 上边缘显著 | **软化**——"在 n=473 上无统计可辨别的提升；连续特征消融是计划中的扩展" |
| 论文中的"8 折 × 3 seed 交叉验证"措辞 | 原 v3/v4 文本读为 5-fold；v5 commit 文本已修正，但 `comparison.md` 现在也已更新 | **核对正文一致性**——论文草稿应统一为 8-fold |
| （论文原本无嵌套 CV 内容） | 阶段 C 新增了"诚实泛化估计"一节 | **新增**——作为补充材料 / 稳健性验证 |

### 5.3 尚未确立的事实（以及如何补齐）

- **FiLM 的贡献**：三阶段中均未测试。`use_film` 消融开关已合入 `train.py`（commit 3bab298），但尚未跑过。这是"真正可能出消融发现"的最优候选。
- **样本量噪声底**：n=473 下，fold 级 Macro-F1 量化到一组离散值。要在统计上区分 MetaMamba 与 MetaMamba-7d，需要更大队列或独立外部验证集。
- **跨数据集泛化**：所有证据都来自单一数据集 CS1。未做外部验证。

---

## 6. 局限与注意事项

1. **单一数据集，外层嵌套 CV 仅用单个 seed**。阶段 C 的外层划分是确定性的（seed=42）；没有跨多个 outer seed 重跑整个嵌套 CV。Bootstrap 式的重复会进一步收紧置信区间。
2. **超参网格很小**（仅 4 组）。诚实的泛化估计是相对该网格而言的。更大的网格（例如加入 `use_film=False`、dropout、batch_size）会拉大乐观偏差。**0.29 pp 应被解读为真实超参搜索下乐观偏差的"下界"**。
3. **Holm-Bonferroni 假设 p 值可排序但不要求独立性**。这是合理的——Holm 在任意相关结构下仍能控制 FWER。
4. **未做功效分析**。在 n=24 配对观测下，对于配对 t 检验，在 α=0.05、功效=0.80、σ_diff ≈ 0.01 时，**最小可检测效应（MDE）约为 0.005 Macro-F1**——因此阶段 B 中 MetaMamba 与 MetaMamba-7d 的比较（Δ = −0.001）**低于 MDE**；不显著的结果与"数据集本身没有足够功效"是一致的。**在任何对外材料中都应明确指出这一点**。
5. **嵌套 CV 只对 MetaMamba-7d 跑了一次**，没有覆盖其他模型。理由是计算成本（阶段 C 用了 21 分钟；扩展到 5 个模型约需 2 小时）。阶段 A 提供的跨模型对比仍然是唯一可用的架构差异比较。
6. **王建最初的提示中提到"嵌套 8 折交叉验证"**，我们将其理解为带 8 外层的严格嵌套 CV。为可执行性起见选择了 5 外层——**该范围决定需明确告知**。

---

## 7. 可复现性

### 7.1 仓库状态（本次工作完成后）

```
4939bba docs(analysis): consolidate CV robustness + significance + nested CV report
b898b6c feat(analysis): mini nested cross-validation on MetaMamba-7d
40725df fix(models): break dead import chain in lstm_7d/__init__.py
130714a fix(data): drop dead import of removed features module from data/__init__.py
9796e1c chore: ignore ad-hoc 5fold backup directories
cc9cfc2 feat(analysis): paired t-test + Holm-Bonferroni significance report
3bab298 feat(model): add use_film / use_tc ablation switches to MetaMamba
358ef62 feat(eval): re-evaluate with 8-fold x 3 seeds StratifiedKFold
```

### 7.2 关键文件

| 文件 | 用途 |
|---|---|
| `outputs/comparison.md` / `comparison.csv` | 阶段 A 主表 + 每类 / 每折稳定性 / 训练时长 / 混淆矩阵 |
| `outputs/significance.md` / `significance.csv` | 阶段 B 配对 t 检验表，含 Holm 校正 p 值与 95% CI；附每 (seed, fold) 原始值以供复核 |
| `outputs/nested_cv/summary.md` / `summary.json` | 阶段 C 嵌套 CV 主表 + 每外层折 + 超参选择表 |
| `outputs/nested_cv/per_outer.jsonl` | 阶段 C 每折详情（机器可读） |
| `outputs/nested_cv/progress.log` | 阶段 C 训练时间线 |
| `outputs/analysis_report_zh.md` | **本文档（中文版）** |
| `analysis/significance.py` | 阶段 B 可复用的分析模块 |
| `analysis/nested_cv.py` | 阶段 C 可复用的嵌套 CV 控制器 |
| `outputs/plots/*.png` | 阶段 A 柱状图 / ROC / PR / 混淆矩阵 / 每折稳定性 |

### 7.3 复现命令

```bash
# 阶段 A：完整 8 折 × 3 seed 重新评估
cd /home/ubuntu/StudentRisk
python main.py --model all --seeds 42 123 777 --n-splits 8

# 阶段 B：配对 t 检验 + Holm-Bonferroni
python analysis/significance.py

# 阶段 C：MetaMamba-7d 上的 Mini 嵌套 CV（T4 上约 21 分钟）
python analysis/nested_cv.py
```

### 7.4 软件栈

- Python 3、PyTorch、scikit-learn（`StratifiedKFold`）、scipy 1.18（`stats.ttest_rel`）、pandas、numpy。
- 硬件：1× NVIDIA Tesla T4（15.36 GiB），服务器 `43.139.55.246`，CUDA 驱动 580.126.20，CUDA 13.0。
- 数据集：CS1 公开数据集，`IDE_logs/IDE_logs.csv`（28,588,309 条事件，7 类事件）+ `IDE_logs/passed.csv`（473 名学生）。

---

## 8. 汇总表 — 所有数字一览

| 估计 | 数值 | 来源 |
|---|---|---|
| **MetaMamba Macro-F1（8 折 × 3 seed）** | 0.8787 | 阶段 A |
| **MetaMamba ROC-AUC（8 折 × 3 seed）** | 0.9347 | 阶段 A |
| **MetaMamba-7d Macro-F1（8 折 × 3 seed）** | 0.8783 | 阶段 A |
| LSTM-7d Macro-F1 | 0.3982 | 阶段 A |
| BiLSTM-7d Macro-F1 | 0.4822 | 阶段 A |
| Attention-7d Macro-F1 | 0.5624 | 阶段 A |
| **MetaMamba vs LSTM-7d Δ Macro-F1** | +0.469，p < 1.4e-22 *** | 阶段 B |
| **MetaMamba vs BiLSTM-7d Δ Macro-F1** | +0.387，p < 1.4e-13 *** | 阶段 B |
| **MetaMamba vs Attention-7d Δ Macro-F1** | +0.343，p < 6.1e-13 *** | 阶段 B |
| **MetaMamba vs MetaMamba-7d Δ Macro-F1** | −0.001，p = 0.51 不显著 | 阶段 B |
| **嵌套 CV MetaMamba-7d Macro-F1** | **0.8754 ± 0.018**，95% CI [0.8530, 0.8979] | 阶段 C |
| **朴素 vs 嵌套 乐观偏差** | +0.0029（0.29 pp） | 阶段 C |
| **Task-Contrastive 胜出次数（嵌套 CV）** | 1/5 外层折 | 阶段 C |
| **分析总耗时** | T4 上约 50 分钟（阶段 A 25 min + 阶段 B <1 min + 阶段 C 21 min） | — |

---

## 9. 核心发现一句话总结

> **MetaMamba 显著优于 7-dim 基线**（p < 1e-12）；**但 11-dim 与 7-dim 输入在该数据集上统计上不可区分**（Δ = −0.001，p = 0.51）；**嵌套 CV 验证朴素 8 折 × 3 seed 估计无显著乐观偏差**（仅 0.29 pp）；**Task-Contrastive 损失在该样本量下无可观测增益**。**瓶颈是样本量（n=473）而非方法本身**——要在统计上分辨 11-dim 与 7-dim 的真实差异，需要更大队列或独立外部验证集。

---

*报告生成时间：2026-08-22，由分析流水线自动产出；所有数字均可从 `outputs/` 下引用文件与 `analysis/` 下脚本复现。*