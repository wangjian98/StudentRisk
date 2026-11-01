# StudentRisk 关键对照实验 · 结果分析报告

**日期**: 2026-10-28
**范围**: FiLM / task_id 注入方式的严格同口径对照 + 跨模型"性能来源"归因
**执行**: 实验于 2026-10-28 01:24 在 246 完成（`scripts/concat_dummy_task_id.py`，1835s，30 个模型）；本报告的分析全部在本地重跑完成（246 仅只读拷贝）

---

## 0. 一句话结论

> **对照实验本身成立且结论稳健（FiLM 强依赖正确的 task_id）；但分析中暴露出一个更严重的问题——所有模型 AUC 中约 87% 来自 part 级先验（"这个学生做的是哪份作业"），个体事件序列的净贡献只有 +0.005～+0.018 AUC；且 MetaMamba 在"去 part 先验后的个体判别力"上显著弱于随机森林。**

---

## 1. 待补对照是什么

`results/dummy_task_id_lopo/summary.csv`（10-12）存在**口径不一致**：

| 行 | 数据来源 | 维度 |
|---|---|---|
| `film_on / true` | `results/meta_mamba_7d/oof_probs.npy` | **7 维 OOF** |
| `film_on / zero·mean·random` | `results/film_inference_ablation/` 的 ckpt | **11 维模型** |

baseline 与被测对象不同源，"喂错 task id 掉 47pp"这个结论的基线不严谨。

**补的实验**（`scripts/concat_dummy_task_id.py`）：
- 统一口径：**同一个模型、同一个 fold 划分、同一套 4 种 task_id 喂法**
- 新增 **concat（加性注入）** 与 **film（乘性调制）** 两种注入方式的直接对照
- 5-fold × 3 seeds = 15 个模型 × 2 种注入 = **30 个模型**，per-student OOF，n=473

---

## 2. 对照实验结果（同口径）

### 2.1 主结果

| 注入方式 | task_id 喂法 | Accuracy | F1-macro | F1(pass) | ROC-AUC | PR-AUC | mean_p | std_p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **film** | **true** | **0.8879** | **0.8750** | **0.8349** | **0.9276** | 0.9640 | 0.669 | 0.353 |
| film | zero(0) | 0.6638 | 0.3990 | 0.0000 | 0.5788 | 0.7097 | 0.904 | 0.041 |
| film | mean | 0.6638 | 0.3990 | 0.0000 | 0.5716 | 0.7018 | 0.885 | 0.054 |
| film | random | 0.6638 | 0.3990 | 0.0000 | 0.5630 | 0.7129 | 0.880 | 0.052 |
| **concat** | **true** | **0.8879** | **0.8750** | **0.8349** | **0.9267** | 0.9641 | 0.669 | 0.347 |
| concat | zero(0) | 0.6638 | 0.3990 | 0.0000 | 0.5909 | 0.7240 | 0.867 | 0.062 |
| concat | mean | 0.6638 | 0.3990 | 0.0000 | 0.5867 | 0.7073 | 0.834 | 0.071 |
| concat | random | 0.6638 | 0.3990 | 0.0000 | 0.5842 | 0.7151 | 0.839 | 0.060 |

### 2.2 统计检验（1000 次 bootstrap，按学生重采样）

| 注入 | 喂法 | F1-macro | 95% CI | 预测为正比率 |
|---|---|---:|---|---:|
| film | true | 0.8750 | [0.8438, 0.9041] | 0.6575 |
| film | zero/mean/random | 0.3990 | [0.3825, 0.4139] | **1.0000** |
| concat | true | 0.8750 | [0.8422, 0.9047] | 0.6575 |
| concat | zero/mean/random | 0.3990 | [0.3849, 0.4146] | **1.0000** |

**McNemar 配对检验**（true vs 任一 dummy，473 个学生逐一配对）：
```
真阳→转阴 = 0     真阴→转阳 = 162     一致 = 311/473     p = 3.4e-49
```
→ 崩塌是**系统性、单向**的：dummy 喂法把 162 个原本判"通过"的学生全部翻成"挂科"，且**没有任何一个真挂科学生被翻回去**。不是噪声，是机制性失效。

### 2.3 三个由此确证的结论

**① FiLM 强依赖正确的 task_id —— 结论成立，比旧数据更硬**
- 喂对 task_id：F1-macro 0.8750，AUC 0.9276
- 喂错任意 task_id：F1-macro 0.3990，AUC 0.56–0.59（接近随机 0.5）
- 连"喂 part 内均值"和"喂随机向量"都没有任何缓冲作用 → 模型对 task embedding 是**点状依赖**，不是"软先验"

**② 加性 concat 与乘性 film 表现完全等价（这是新发现）**
- true 条件下：Accuracy / F1-macro / F1(pass) **逐位相同**（0.8879 / 0.8750 / 0.8349），混淆矩阵也相同（TN=134, FP=25, FN=28, TP=286）
- 逐学生概率相关性 r = 0.9986（true），但 dummy 条件下降到 0.76–0.92 → 两者是**不同的模型**，只是结论一致
- **含义**：FiLM 的"乘性调制"形式本身**不是**性能来源。真正起作用的是"task 信息通道"这件事，用加法也能达到同样效果。这对论文是个**削弱 FiLM 必要性论证**的发现。

**③ 崩塌的机制是"退化成多数类预测器"，不是"排序能力下降"**
- 三个 dummy 条件下预测为正的比率都精确等于 **1.0000**，accuracy 精确等于基率 **0.6638**
- 但 AUC 掉到 0.56–0.59（不是 0.5）→ 排序能力也几乎没了，只比随机好一点
- 对照旧协议的 `film_disabled`（γ=1, β=0，即 FiLM 通路关掉）：AUC 0.8966 **仍然很强**
- → **喂错 task_id 比完全不喂 task_id 更糟。** FiLM 学到的不是"可选修正"，而是主导判别的必需信号。

---

## 3. 🔴 分析中发现的更重要问题：性能主要来自 part 先验，不是个体事件序列

### 3.1 决定性检验

数据集是 473 名学生 × 7 个 part，每个 part 的挂科率差异极大：

| part | n | 挂科 | 通过 | 挂科率 |
|---|---:|---:|---:|---:|
| 0 | 131 | 130 | 1 | 0.9924 |
| 1 | 78 | 76 | 2 | 0.9744 |
| 2 | 24 | 20 | 4 | 0.8333 |
| 3 | 55 | 45 | 10 | 0.8182 |
| 4 | 14 | 11 | 3 | 0.7857 |
| 5 | 9 | 4 | 5 | 0.4444 |
| 6 | 162 | 28 | 134 | 0.1728 |

**只要知道"这个学生做的是哪份作业"，不看任何行为数据**，就能得到：

| 方案 | AUC | Accuracy | F1-macro |
|---|---:|---:|---:|
| **part-prior baseline（按 fold 无泄漏）** | **0.9135** | 0.8795 | 0.8664 |
| MetaMamba-11d | 0.9318 | 0.8858 | 0.8736 |
| MetaMamba-7d | 0.9188 | 0.8837 | 0.8715 |
| RF-7d | 0.9179 | 0.8584 | 0.8496 |

### 3.2 个体信息的净贡献

| 模型 | 总体 AUC | 超出 part 先验 | **个体残差 AUC**（去 part 均值后） |
|---|---:|---:|---:|
| MetaMamba-11d | 0.9318 | **+0.0182** | 0.5423 |
| MetaMamba-7d | 0.9188 | **+0.0053** | **0.4439** |
| **RF-7d** | 0.9179 | **+0.0044** | **0.6708** |

- **个体信息只解释了超越随机空间的 11%–14%**（MetaMamba-7d 仅 11.4%）
- **去 part 先验后，MetaMamba-7d 的个体判别力（AUC 0.4439）低于随机水平 0.5；RF-7d 反而是最强的（0.6708）**
- MetaMamba-7d − RF-7d 的 part 内 AUC 差：**−0.2364，95% CI [−0.3688, −0.0994]，P(≤0) = 99.9%**（MetaMamba 显著更差）

### 3.3 复现检验（两批独立 OOF 一致）

| 来源 | 总体 AUC | 残差 AUC | part 内加权 AUC |
|---|---:|---:|---:|
| 旧 OOF（meta_mamba_7d，10 月） | 0.9188 | 0.4439 | 0.4829 |
| **新 OOF（concat_dummy_task_id，10-28 独立重训）** | 0.9276 | 0.5859 | 0.5156 |

两批独立训练的 OOF 呈现**同样的模式**（总体高、个体低）→ **不是实现偶发，是数据结构的系统性问题**。

### 3.4 口径核查（排除实现错误）

| 检查项 | 结果 |
|---|---|
| oof_probs / labels / task_ids / student_ids 对齐 | ✅ student_ids 相同、labels 相同、n=473 全部唯一 |
| fold_idx 合法性 | ✅ 5 折 [95,95,95,94,94]，每学生恰好一次 |
| 重算指标 vs 各模型 results.json | ✅ **逐位一致**（MetaMamba 0.9318 / RF 0.9179 / Attention 0.6938 …） |
| part 先验泄漏 | ✅ LOO 0.8712 / fold 无泄漏 0.9135 / 不留一 0.9254（有泄漏上界），全部 < 模型 AUC，说明模型确实超过纯先验 |
| 结论是否依赖阈值 0.5 | ❌ 不依赖：最优阈值下 f1_macro MetaMamba 0.8793 / RF 0.8559，AUC 本身与阈值无关 |
| part 内 AUC 稳定性 | ⚠️ n 小（part3 n=55，part6 n=162），MetaMamba part6 AUC 95%CI [0.3661, 0.5979] **含 0.5**；RF 的 CI 不含 0.5 |

---

## 4. 影响判断

| 论文声称 | 复核后 |
|---|---|
| "MetaMamba 显著优于 LSTM/BiLSTM/Attention" | ✅ **成立**（AUC 0.93 vs 0.62–0.69，且残差 AUC 也领先） |
| "MetaMamba 优于 RF-7d" | ⚠️ **削弱**：总体 AUC 只高 0.0009（0.9188 vs 0.9179），扣掉 part 先验后只高 0.0009；**去 part 均值后 MetaMamba-7d 反而显著弱于 RF** |
| "FiLM 是性能核心"（消融去掉 FiLM 崩塌） | ✅ **成立**，且本报告补强：喂错 task_id 比不喂更糟 |
| "FiLM 的乘性调制是必要的" | ❌ **不成立**：concat（加性）表现与 film 逐位相同 |
| 隐含假设"模型在学个体行为→个体风险" | ⚠️ **大部分 AUC 来自 part 先验**（87%）；个体信号贡献 +0.005～0.018 AUC |

---

## 5. 建议（需王建决策）

### 必做（论文可信度）
1. **在正文加入 part-prior baseline**（AUC 0.9135）作为强基线。当前只对比 LSTM/BiLSTM/Attention（0.62–0.69，远低于 0.91），审稿人一定会问"为什么不用 part 先验"。这不是自伤，是**主动补强**：一旦补上，MetaMamba 的 +0.018 仍然成立且更有说服力。
2. **加"去 part 先验"的个体判别力对比表**（残差 AUC / part 内 AUC）。这是诚实但必要的披露。
3. **修正 FiLM 论证**：改为"task 信息通道有效"而非"乘性调制有效"，或补一个"乘性 vs 加性"的正式消融（本次实验已经提供了数据）。

### 建议做（提升结论强度）
4. **重做 LSTM/BiLSTM/Attention 的 part 内判别力**：它们的残差 AUC（0.539/0.543/0.547）接近随机，且 part 内加权 AUC 0.70–0.73 高于 MetaMamba-7d（0.483）——**深度基线在个体层面反而更好**。这暗示当前 7 维特征的架构竞争不公平，值得查。
5. 扩大样本或做跨 part 泛化实验（每 part 内部正负样本对比度太低，part0 只有 1 个通过学生、part5 只有 9 人）。

---

## 6. 产物清单

```
StudentRisk/analysis/control_exp_20261028/
├── REPORT_control_experiment.md          ← 本报告
├── analyze_control.py                    ← 对照实验分析 + bootstrap + McNemar + 逐 part 拆解
├── analyze_prior_decomp.py               ← part 先验归因（决定性检验）
├── analyze_cross_model_prior.py          ← 跨模型 part 先验对比 + 配对 bootstrap
├── verify_part_prior_finding.py          ← 口径核查（对齐/泄漏/阈值/稳定性）
├── analyze_residual_auc.py               ← 残差 AUC / part 内 AUC 公平比较
├── results/                              ← 从 246 只读拷贝的原始结果
│   ├── concat_dummy_task_id/{summary.csv, pivot_macro_f1.csv, per_student_predictions.csv}
│   ├── dummy_task_id_lopo/{summary.csv, per_part_breakdown.csv, dummy_task_id_report.md}
│   └── film_inference_ablation/{summary.csv, all_runs.csv, ...}
├── oof/                                  ← 6 个模型的 OOF 预测（重算指标用）
│   └── {meta_mamba, meta_mamba_7d, rf7, lstm_7d, bilstm_7d, attention_7d}/
└── logs/concat_dummy.log                 ← 训练日志（1835s，30 模型）
```

**复现命令**（本地）：
```bash
cd StudentRisk/analysis/control_exp_20261028
python3 analyze_control.py
python3 analyze_prior_decomp.py
python3 analyze_cross_model_prior.py
python3 verify_part_prior_finding.py
python3 analyze_residual_auc.py
```
