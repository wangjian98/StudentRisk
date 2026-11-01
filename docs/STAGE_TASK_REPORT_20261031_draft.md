# StudentRisk 阶段级掉队预测 · 结案报告（2026-10-31）

> **一句话**：阶段级任务成立且很强（part-prior 0.41 → RF 0.93），但**"新架构"在阶段级不成立**——
> 四项全开（去 part 混淆 + 残差化 + 阶段级 + 跨 part 泛化）后，MetaMamba 仍未赢过同口径 RF。
> → **架构线关闭**：论文主模型换 RF，MetaMamba 降为对照/负结果；卖点重构为
> **新任务定义 + 严谨评估协议 + 第 1 段即可预警（0.9423）**。

---

## 1 · 实验设置（可复现）

| 项 | 内容 |
|---|---|
| 服务器 | 246（43.139.55.246，Tesla T4 15.4GiB），工作副本 `/home/ubuntu/studentrisk_stage`（不动作者仓库）；兰兴宇的 campus-multimodal 服务全程未受影响（常驻 2434 MiB） |
| 数据 | 2171 阶段样本 / 473 学生 / 258 掉队正例；标签 `drop=1` ⇔ 该(生,part) 是他到达过的最远 part 且从未达 part7（从 IDE 日志事实推导，**无新标注**） |
| 主协议 | 5 折按**学生**分组 + 课程级标签分层；3 seeds（42/123/777）；OOF 概率取均值；**早停用 train 折内层验证集（15%，按学生切）→ 评估折不参与模型选择**（比旧协议更严：旧协议拿评估折早停，对深度模型偏乐观） |
| LOPO 协议 | 留一 part：train = 其余 part，test = 留出 part（part 1–6，part7 无正例不可评）；同伴归一化基线用被留出 part 自己的行算（**无标签变换，不构成泄漏**），两 arm 口径一致 |
| 主指标 | **part 内 pooled AUC**（个体判别力口径，n 加权）＋ 配对 bootstrap 2000 次（part 内分层重采样）；LOPO 另报逐 part AUC 与 n 加权 LOPO AUC |
| 可复现性 | 本次重跑的 `rf_full` 与 10-30 那次**逐位一致**（总体 0.9353 / pooled 0.9258，bootstrap Δpooled=+0.5152 同值）→ 折与种子口径可复现 |

## 2 · "四项全开"的口径（两个 arm 完全对称）

| 项 | MetaMamba | RandomForest |
|---|---|---|
| ① 阶段级 | stage_dataset.npz + drop 标签 | 同 |
| ② 去 part 混淆 | 删掉序列里的 **part 通道**（seq11→seq11_np，保留 exercise）+ **关掉 FiLM 的 part 调制** + task_id 置 0 → 模型看不到"这是第几 part" | 特征里本就没有 part 身份；上下文只给"相对同 part 同伴的位置"（peer_z） |
| ③ 残差化 | 打分层 M1 残差 `p − 0.25·p_prior_stage`（w 是 10-28 实验 D1 **预先**定下的值，非事后挑选）；特征层另做"剔除活跃度方向"变体 | 同样两层 |
| ④ 跨 part 泛化 | LOPO：留一 part 训练/测试 | 同协议 LOPO |
| 额外给 MM 一次机会 | 把 RF 同款 29 维上下文**广播进序列**当额外通道（d_event 10→39）——它能拿到 RF 的全部特征信息 | — |

## 3 · 结果

### 3.1 同协议主结果（5 折按学生分组 × 3 seeds）

<<MAIN_TABLE>>

### 3.2 决定性配对 bootstrap

<<BOOT_TABLE>>

### 3.3 跨 part 泛化（LOPO）

<<LOPO_TABLE>>

## 4 · 判定与结论

<<VERDICT>>

## 5 · 对论文的影响（改什么）

<<PAPER_IMPACT>>

## 6 · 仍欠 / 下一步

<<TODO>>

## 附 · 产出物

<<ARTEFACTS>>
