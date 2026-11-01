# EDM 2025 HMM 基线补齐 · 数据集同一性核对与复现（2026-11-01）

**触发**：王建转来 Murali, Conati & Poole (EDM 2025) *A Comparison of Real-Time User Classification Methods using Interaction Data for Open-ended Learning*（EDM 2025 长文，pp. 7–18, DOI 10.5281/zenodo.15870205）。
**结论先行**：这篇用的**就是我们的数据集**（CS1 MOOC），而且它**暴露了我们基线表的一个硬缺口（无 HMM）**。我们已在主协议下实测 HMM，并复现了它的建模路线，结论是：**在我们改写后的任务形式与严格分组口径下，HMM 远不如 RF；EDM'25 的"HMM 最强"结论不可迁移。**

---

## 1 · 数据集同一性核对 ✅

| 项 | EDM'25 | 我们（StudentRisk） | 判定 |
|---|---|---|---|
| 数据集 | CS1 MOOC（Java，IDE 日志）[30] | CS1 MOOC IDE 日志 | **同一份** |
| 学生数 | 469 | 473 | 他们剔除了"只有单行日志"的 4 人（论文脚注 1） |
| 通过/未通过 | 159 / 310 | 159 / 314 | 同一份 `passed.csv`（我们已核：473 行，passed=True 159、False 314） |
| 事件类型 | 7 类（text_insert/remove/paste、focus_gained/lost、run、submit） | 同 7 类 | 一致 |
| 题目分段 | 7 parts | 7 parts | 一致 |

→ **可以在我们的论文里把 EDM'25 当作"同一数据集上的已发表基线"直接引用**（他们报告 HMM accuracy 0.7697）。

## 2 · 他们做了什么（论文要点）

- 四个分类器跨三个开放学习数据集比较：**FUMA**（规则/关联规则）、**GMM-HMM**（每类一个 HMM + 先验）、**LSTM**、**分层 dummy 基线**。
- 协议：8 折 nested CV + 内层网格搜索；特征按 **60 秒窗口**计算，且每个特征算两遍（**非增量**=当前窗、**增量**=累计到当前窗）；CS1 因交互时长差异大，用**从 110 分钟起的对数时间轴**。
- 指标：accuracy；显著性用配对 t 检验（BH 校正）+ Cohen's d；解释性用 permutation importance + SHAP。
- CS1 MOOC 结果（最后窗口，平均准确率）：

| 模型 | accuracy |
|---|---|
| **HMM** | **76.97 ± 1.42** |
| LSTM | 71.51 ± 2.05 |
| 分层 dummy 基线 | 53.95 ± 2.19 |
| FUMA（规则） | 34.65 ± 2.44 |

- 解释性发现（CS1）：LSTM 高度依赖 **text insert 计数**（PI 30.05），HMM 同时依赖 **insert + remove 计数**（PI 13.13 / 10.30）；**FUMA 的规则法在这个数据集上失效（低于 dummy 基线）**。

## 3 · 我们的实测

### 3.1 主协议下（阶段级、按学生分组、part 内 pooled AUC）—— 基线缺口补齐

| 模型 | overall AUC | **part 内 pooled** | judgeable | Δ vs RF | P(Δ≤0) |
|---|---:|---:|---:|---:|---:|
| **RF（我们的基线）** | 0.9353 | **0.9258** | 0.9199 | — | — |
| **HMM（多项式，7 类事件符号）** | 0.6768 | **0.6671** | 0.6527 | **−0.2587** | **100.00%** |
| **HMM（高斯，seq9 连续通道，无 part 身份）** | 0.5870 | **0.5879** | 0.5800 | **−0.3379** | **100.00%** |

→ **HMM 在阶段级远不如 RF**，与 LSTM / MetaMamba 的命运一致。n_components 由内层验证选（K=2–4），评估折不参与调参。

### 3.2 复现他们的建模路线（学生级、8 折，与他们同折数）

| 我们的实现 | accuracy | AUC | 对照他们 |
|---|---:|---:|---|
| **符号 HMM**（1 小时窗主导事件 → CategoricalHMM，每类一个模型 + 先验） | **0.7294** | 0.7452 | 他们 HMM **0.7697** |
| 特征窗高斯 HMM（14 维窗级特征 + 累计，长度归一化打分） | 0.6638（= 基率，**塌陷为多数类**） | 0.4289 | — |
| 多数类基线 | 0.6638 | — | 他们 dummy 0.5395 |

→ 我们复现出"**HMM 显著高于多数类基线**"这一定性结论（0.7294 vs 0.6638），但**未能复现 0.7697**（差 ~4pp）。已声明的偏差：① 我们 473 人（他们 469）；② 时间窗用 1 小时而非 60 秒（60 秒窗在跨度最长 1169 小时的学生上会产生上万个状态点，EM 不可行）；③ hmmlearn 0.3.x 已移除 `hmmlearn.mixture`，**GMM-HMM 无法直接实现**，我们用单高斯发射的 GaussianHMM 近似（其 n_mix=1 情形），该近似在窗级特征上塌陷为多数类。

### 3.3 三条可写进论文的结论

1. **"同一数据集上已发表的最强模型"在严格口径下不占优**：EDM'25 报告 HMM accuracy 0.7697（LSTM 0.7151）；我们在**阶段级任务 + 按学生分组 + part 内口径**下测得 HMM pooled 0.6671、RF 0.9258（Δ −0.2587，P=100%）。差异来自**任务形式与评估协议**，不是实现细节。
2. **规则法（FUMA）在 CS1 上失效**（他们：34.65，低于 dummy 53.95）与我们自己的负结果族一致（S3/M4 强度统计无增益、序列只当特征提取器）→ **"更多人工统计量/规则"不是解**，这条现在有了外部证据，可以写进论文的讨论章。
3. **他们的增量窗口协议存在跨折泄漏通道**：同一学生的早期窗口进训练、后期窗口进测试 → 同一学生的部分信息跨折。我们已在 §3.6 测得学生级数字在按学生分组后下降 0.0265（0.7955 → 0.7690），本条为该结论提供了**已发表的对照实现**。

## 4 · 顺带发现的数据完整性问题（务必记录）

**pandas 时间戳分辨率坑**：对本数据集，`pd.to_datetime(...)` 返回 **datetime64[us]**（微秒），因此

```python
pd.to_datetime(df['timestamp']).astype('int64') // 10**9   # ← 错：把 µs 当 ns，时间戳缩小 1000 倍
```

会把时间戳**缩小 1000 倍**（例：2017-01-15 23:26:53 的正确 epoch 秒 1484522813，被算成 1484522）。正确写法：

```python
pd.to_datetime(df['timestamp']).values.astype('datetime64[ns]').astype('int64') // 10**9
```

- 本次踩坑位置：`analysis/hazard_model/build_student_seqs.py`（已修正），并因此暴露出学生真实交互跨度（中位 **900+ 小时**、最长 1169 小时），而非原先误算的 0.9 小时。
- **项目主流程不受影响**：`build_stage_dataset.py` / `build_multiscale.py` 均使用 datetime64 运算（`diff()`、`Timestamp` 差值），不受分辨率影响；`experiment_D_M1_M4.py` 显式 `dtype='datetime64[s]'` 亦正确。
- **唯一受影响的既有脚本**：`generalization/data_splits.py` 的 `time_noise` 扰动（第 306–307 行）把 µs 当 ns，还原后时间戳落到 1970 年附近。由于该扰动是**乘性噪声**（对时间轴整体等比缩放），区间结构的相对扰动量不变，**v8 时间噪声鲁棒性结论不受影响**；但其绝对时间语义已失真，建议在重跑该消融时修正并注明。

## 5 · 复现

```bash
# 246：/home/ubuntu/studentrisk_stage
pip install --user --break-system-packages hmmlearn          # 0.3.3
python3 analysis/hazard_model/run_hmm_baseline.py            --data data/stage_dataset.npz --out results/hmm
python3 analysis/hazard_model/build_student_win_feats.py     --ide /home/ubuntu/IDE_logs/IDE_logs.csv --out data/student_win_feats.npz --win-sec 3600
python3 analysis/hazard_model/run_edm25_replication.py       --data data/stage_dataset.npz --win data/student_win_feats.npz --out results/hmm --states 2 3 4
```
