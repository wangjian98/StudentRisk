# RegimeMamba 方法设计文档 + 实验协议（CS1 风险预测）

**版本**：v1（2026-11-08 16:20，王建指令「按 RegimeMamba 方法设计文档实验」）
**代码**：`StudentRisk/analysis/regimemamba/{model.py, run_exp.py}`
**结果目录**：`StudentRisk/results/regimemamba/<tag>/`
**环境**：246（43.139.55.246，Tesla T4），与 rev12 同一数据缓存 `/home/ubuntu/SR12_cache`

---

## 0 · 为什么做这个模型（动机，不是理由）

已有实测把「架构」从杠杆位置上摘掉了（证据见 §7），所以本实验的目的**不是**再赌一次更大的网络，而是：

1. **把 CS1 风险信号的行为学假设显式化**：CS1 的风险在行为上表现为**状态切换**（探索 → 低产 → 停滞 → 放弃），而现有基线各缺一半——
   - HMM 家族：**有 regime、无选择性**（线性/对角高斯发射，事件符号空间直接塌成先验 0.478）；
   - MetaMamba：**有选择性、无 regime**（一套全局 SSM 动力学，FiLM 还把 part 身份塞进调制 → 走捷径）；
   - RF 表格：**有统计、无时序**（把整学期压成静态向量）。
   → **RegimeMamba = 选择性扫描 × 可切换 regime**，把 HMM 的"状态"和 Mamba 的"选择性"合到一处，并且 **regime 只看行为、不看身份**。
2. **可解释性产出**：学到的 regime 若与"停滞/放弃"等行为语义对齐且与挂科率相关，就是论文级的机制发现（而非黑盒分数）。
3. **可证伪**：本实验预先注册了 H1–H6 的判负标准（§3），任何"看起来不错"的结果都必须先过这些关。

---

## 1 · 架构（与 MetaMamba 的唯一结构差异）

```
事件序列 (B,128,28d, past-only)
   │  event_embed (Linear→GELU) + LayerNorm
   ▼
行为 regime 前向滤波  →  R ∈ R^{B×L×K}      【新增】
   │   g(x_t) = MLP(x_t)                       发射（不含任何身份通道）
   │   r_t    = softmax( log(r_{t-1}A) + g(x_t) )   A = softmax 参数化 (K×K)
   │   padding 步不更新后验（carry forward）
   ▼
regime 条件化 S6 栈 ×2  →  h ∈ R^{B×L×d}    【新增】
   │   选择性参数按 regime 混合（只跑一次扫描）：
   │   dt_t = Σ_k r_{t,k}·softplus(s_k⊙dt + b_k)
   │   B_t  = Σ_k r_{t,k}·(m^B_k ⊙ B_raw_t)
   │   C_t  = Σ_k r_{t,k}·(m^C_k ⊙ C_raw_t)
   │   A_t  = −exp(A_log + Σ_k r_{t,k}·A_off_k)
   ▼
[可选] TaskFiLM(part)      ← 身份臂保留；去混淆臂关闭（task 置零）
   ▼
掩码均值池化  ⊕  regime 直通特征            【新增】
   │   usage (K)   真实步平均占用
   │   last  (K)   最后有效步后验
   │   switch(1)   平均切换幅度 ‖r_t−r_{t−1}‖₁
   ▼
logit = MLP([pooled ⊕ 直通]) + Σ_k usage_k·b_k     ← per-regime 风险偏置（可解释）
```

**关键设计约束**
- **regime 发射网络不得接 part / exercise 身份**（28d 通道 9/10/27 在去混淆臂置零）→ regime 只能是行为语义，不能是"作业身份"。
- **只做一次扫描**：regime 混合发生在扫描前的线性算子上，计算量与 MetaMamba 同量级（实测 ~2.7 s/epoch，MetaMamba 同量级），保证预算对称。
- **因果**：前向滤波只依赖过去，past-only 约束不被破坏。

## 2 · 训练目标

```
L = BCE(logit, y) + λ_u·L_usage + λ_s·L_switch + λ_c·L_sharp
L_usage = −H(平均占用)/log K      （最大化熵 → 防 regime 塌缩）
L_switch= 平均切换幅度             （最小化 → regime 成为"状态"而非抖动）
L_sharp = 平均每步后验熵/log K     （最小化 → 每步归属明确）
默认 λ_u=0.05, λ_s=0.01, λ_c=0.01
```

## 3 · 预先注册的可证伪标准（先写死，再看结果）

| 编号 | 假设 | 通过标准 | 判负结论 |
|---|---|---|---|
| **H1** | regime 机制带来判别力增益 | 28d 身份臂**合并 OOF AUC** ≥ MetaMamba-28d + 0.010 且 bootstrap P(Δ≤0) ≤ 0.05 | Δ∈[−0.01,+0.01] → "无独立增益"；Δ≤−0.01 → "regime 机制有害" |
| **H2** | regime 是行为信号而非身份捷径 | 去混淆臂 AUC 跌幅 ≤ 0.020（对照：11d 家族 −0.205、MetaMamba 28d −0.004） | 跌幅 >0.05 → "regime 走了身份捷径" |
| **H3** | 早期预警受益 | 前缀 10% AUC ≥ MetaMamba p0.1（0.9298） | 低于 → "早期无收益" |
| **H4** | regime 真实存在（非噪声、非塌缩） | ① 各 regime 占用 ≥5%；② 切换率与行为统计显著相关（\|ρ\|≥0.3, p<0.01）；③ 至少一个 regime 占用与挂科率显著相关（p<0.01） | 任一不满足 → "regime 不可解释" |
| **H5** | 跨任务泛化 | LOPO 加权 AUC ≥ MetaMamba default（0.6169） | 低于 → "无泛化增益" |
| **H6** | 不是"更多容量"的假象 | K=1 对照（无切换）明显弱于 K=4 | 无差异 → "增益来自容量不是 regime" |

> **纪律**：结果出来后不得修改上述阈值；判负结果必须照实写入文档与论文。

## 4 · 实验矩阵（协议与 rev12 完全对齐，可直接与既有结果比）

| 编号 | 格（tag） | 对照 |
|---|---|---|
| E1 | `main_d28_p1.0`（身份臂，全程，超参 4 候选） | MetaMamba-28d 0.9056 / RF-28d 0.9509 / HMM-GMM 0.9004（合并 OOF） |
| E2 | `main_d28_p0.1`（前缀 10%，早期预警） | MetaMamba p0.1 0.9298 / RF p0.1 0.8893 |
| E3 | `main_d28_p0.2`（前缀 20%） | MetaMamba p0.2 0.9219 / RF p0.2 0.9084 |
| E4 | `deconf_d28_p1.0`（去身份） | MetaMamba 去混淆 0.9016 / RF 去混淆 0.9346 |
| E5 | `abl_k1_d28_p1.0`（K=1，无切换对照） | E1 自身（H6） |
| E6 | `abl_nodirect_d28_p1.0`（去掉 regime 直通特征） | E1 自身（直通的边际贡献） |
| E7 | `abl_nousage_d28_p1.0`（去掉防塌缩正则 λ_u=0） | E1 自身（regime 是否塌缩） |
| E8 | LOPO 跨任务迁移 | MetaMamba 0.6169 / no-FiLM 0.6814 / RF 0.6123 |

**协议**：5 折 × 3 seeds（按学生分组、标签分层）｜训练折内 15% 内层验证做早停与超参选择｜评估折不参与任何选择｜max_epochs=200、patience=20、batch=32、AdamW+cosine、grad_clip=1.0｜超参 4 候选与其它架构对称｜指标 acc / bal-acc / macro-F1 / F1(FAIL) / ROC-AUC。
**口径纪律**：本文主口径 = **全样本合并 OOF AUC**（比论文的折均值更严格）；论文口径（折均值）同时报告。

## 5 · 分析计划（E4 之外的结果分析）

1. **性能**：H1/H3 判定 + 与 MetaMamba/RF/HMM 的配对 bootstrap（2000 次）+ Wilcoxon/Holm/McNemar。
2. **regime 语义**（H4）：把学到的 regime 与可观测行为统计对齐——
   - 行为统计：submit 率、run 率、focus_lost 率、平均事件间隔（gap）、最长空窗、活跃天数/小时、编辑/粘贴比；
   - 逐 regime 的占用分布、regime 间行为均值差异；
   - **regime 占用 vs 挂科率**（每个 regime 的组间差异检验）；
   - **切换率 vs 挂科**、**切换次数**与失败的关系；
   - 后验轨迹：失败生与通过生的 regime 时间线对比（dropout 前后的状态变化）。
3. **可解释风险归因**：per-regime 风险偏置 b_k 与该 regime 的挂科率是否同向（模型内部归因 vs 数据事实是否一致）。
4. **与既有结论的对话**：若 H1 判负，则结论是「CS1 的行为 regime 切换**存在且可解释**，但**不提供超越表格统计的额外判别力**」——这与"架构不是杠杆""输入设计才是杠杆"的主线一致，且补上了"状态空间切换是否值得建模"这一审稿人必问项。

## 6 · 风险与兜底

- **塌缩风险**：regime 可能退化为 1 个有效状态 → 由 λ_u 防塌缩 + E5（K=1）/E7（去 λ_u）双向验证。
- **容量混淆**：RegimeMamba 参数略多于 MetaMamba → E5（K=1 同容量对照）处理。
- **训练时长**：单格 ≈ 45–75 分钟（19 次训练 × 早停），6 格 ≈ 5–6 小时，246 后台串行跑；E1 完成后先回报一次。
- **GPU 共享**：246 上另有 2.8 GB 显存被兰兴宇的多模态服务占用，本实验约需 2–3 GB，不冲突，不触碰他人进程。

## 7 · 本实验不改变的主线结论（引用已有实测）

- 换学习器无收益：阶段级 RF/LGBM/HGB/MoE 0.9191–0.9268 差异 ns；
- 架构不是杠杆、输入才是：11d 空间去身份 AUC −0.2048，28d 仅 −0.0177；
- 窗口口径影响巨大：同特征「全量事件 0.9166 → 末 128 事件 0.6990」（−0.218）；
- 少样本元学习路线判负（旧读数为脚本 bug；深度度量学习随 K 变差；伪标签自训练全低于基线）。

---

**结果占位**：跑完后由 `analysis/regimemamba/analyze.py` 汇总，回填 §8（结果与发现）与 `docs/REGIMEMAMBA_RESULTS_20261108.md`。
