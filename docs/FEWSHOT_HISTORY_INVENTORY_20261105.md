# StudentRisk · 少样本 / 元学习 / 跨段迁移(LOPO) 实验全量盘点

> 盘点日期：2026-11-05　　范围：本地仓库 `StudentRisk/`（main @ `efd97e8`）+ 246 服务器 `/home/ubuntu/studentrisk_stage/`（及 `/home/ubuntu/StudentRisk/`）
> 性质：**只读盘点**，未修改任何已有文件（本文件为唯一新建文件）。
> 目的：① 给出"少样本/元学习/LOPO **已证伪清单 + 数字**"；② 给出新实验可直接用的数据文件清单；③ 给出可直接复用的代码骨架。

> ## ⚠️ 必读：本次盘点发现 3 个**实现级缺陷**，会改变部分实验的解释
>
> | 缺陷 | 影响到的实验 | 后果 |
> |---|---|---|
> | **D1 · 少样本适配是彻底的 no-op** | F1 `per_part_5shot_full`、F2 `per_part_5shot_adaptive`、F3 `per_part_3shot_A`、F5 `kshot_sweep` | `fomaml_inner_adapt()` 返回的是**适配前**的权重快照（`eval_per_part_5shot.py:70,83`），调用方再把这个快照喂回 `predict_with_state()`（`:179-189`）→ **3 步内层梯度被完全丢弃**。这四个实验的"5-shot FoMAML"数字其实是**零样本、不适配**的数字 |
> | **D2 · part 编号 off-by-one** | 同上四个 | `task_ids` 是 0-indexed dominant part（0..6），脚本却迭代 `parts=range(1, n_tasks+1)` → 日志里的 "[Part 3]" 其实是**原始 part 4**；**原始 part 1（131 人）从未被评估**；"Part 7" 是不存在的 part |
> | **D3 · 训练集泄漏进评估** | 同上四个 | `final_model` 是"最后一个 seed 的最后一个 fold"，它见过 4/5 的学生；评估却在全部 473 人上做 → query 含训练样本 → 数字是**偏乐观**的（也就是说，这几个实验是在有利条件下仍然失败的） |
>
> **不受影响、口径可信的少样本实验**：F4（Plan B participation，`eval_participation_kshot.py`）、F6/F7（`run_4_gen_experiments.py`）、F8/L7（`246:run_nofilm_lopo_fomaml.py`）、全部 LOPO 实验（L1/L3/L5/L6/L7/L8）、CS2-D 系列。
> 详见 **§1.5 实现缺陷审计**。

---

## 0 · 结论先行（TL;DR）

| # | 结论 | 关键数字 | 出处 |
|---|---|---|---|
| 0 | **⚠️ 【最重要】4 个少样本实验（per_part 5-shot / adaptive / 3-shot / kshot_sweep）的“少样本适配”代码是 no-op** —— 内层 3 步梯度被丢弃，且 part 编号 off-by-one，原始 part1 从未评估。**这些数字不能作为少样本有效/无效的依据** | `eval_per_part_5shot.py:70,83` 返回适配前快照；`parts=range(1,8)` vs 0-indexed task_ids | 见 §1.4 D1/D2/D3 |
| 1 | **可信口径下，学生级少样本 FoMAML 仍全部失败**：最好只到 AUC 0.74，且一半 part 直接 SKIP | per-part Plan-B（F4，✅实现正确）24 组 ROC 全在 0.483–0.628；A-plan（F6，✅）part7 **ROC 0.221 < 0.5** | `results/per_part_kshot_B/`、`results/generalization_4runs/summary.json` |
| 2 | **抽样协议本身是致命变量**：B-plan（participation 抽样）无论 K=5/10/15/20，**全场 ROC ≈ 0.54 ≈ 随机**，且**对 K 无响应** | overall_mean_roc_auc **0.5369**；per_part_kshot_B 24 组 ROC 全在 0.48–0.63 | `run_4_gen_experiments.py:262-357`、`results/per_part_kshot_B/planB_summary.csv` |
| 3 | **K-shot 曲线是平的**，不存在"少样本随 K 变好"的曲线 | 原始 part4: K=3→ROC 0.740, K=5→0.712；K=1/20 全部 SKIP | `246:results/kshot_sweep/summary.json` |
| 4 | **学生级 LOPO 是硬墙**：跨 part 后 ROC 从 0.919 掉到 0.61–0.64 | mean ROC **0.6134±0.0949**(3seed) / 0.6421(1seed) vs within-task **0.9188** | `results/lopo_3seed/`、`results/lopo/` |
| 5 | **但阶段级 RF 的 LOPO 几乎不掉**（−0.013）→ LOPO 失败是**模型+粒度**的性质，不是任务性质 | RF LOPO 加权 AUC **0.9128** vs within-task pooled **0.9258** | `results/stage_experiment/stage_lopo_summary.json` |
| 6 | **诊断通道（CS2-D）LOPO 只掉 0.035**，判别通道掉 0.29 → 两者性质不同 | CS2-D top1 **0.607 → 0.572**；最好启发式 0.333 | `results/cs2_diag_lopo/results.json` |
| 7 | **FiLM 在 far-OOD + few-shot 上是净伤害**（唯一“少样本能赢”的机制发现） | part7 ROC：Full+FiLM **0.221** vs −FiLM **0.701**（Δ=**+0.480**） | `246:results/generalization_nofilm/summary.json` |
| 8 | **唯一设计完整的"少样本适配 vs 不适配"对照实验没跑完**（adapted_emb 全为 null） | `246:results/few_shot_study/exp_c_5shot_emb.json` 5 条记录 `adapted_emb: null` | 同左 |
| 9 | **per-student LOPO 直接归零**（同一学生换 part 也认不出） | FiLM ON acc **0.400**、FiLM OFF acc **0.438**（二分类抛硬币），ROC 不可算 | `results/per_student_lopo/per_student_lopo_summary.json` |
| 10 | 数据资产齐全且已产品化：阶段级 2171 样本 / 习题级 44764 / 坐下级 / 原始事件 2858 万行全在 246 | 见 §3 | `246:/home/ubuntu/studentrisk_stage/data/` |
| 11 | 复现骨架现成：`stage_common.py` 一站式提供 **5 折按学生分组 + part 内 pooled AUC + 配对 bootstrap** | `stage_common.py` L100/L131/L163 | 本地仓库已入库 |

---

## 1 · 已证伪清单（含数字）

> 判负标准：与**同协议 full-label 基线**比。
> **基线锚点**：
> - 学生级 within-task 5-fold CV（MetaMamba-7d，473 学生）：`accuracy 0.8837 / macro_f1 0.8715 / roc_auc 0.9188 / f1_failed 0.9111`（`results/generalization_4runs/summary.json:exp1_5fold_cv`；脚本里也硬编码为 `reference_within_task_5fold_CV = {accuracy 0.886, macro_f1 0.874, roc_auc 0.932}`，见 `lopo_train_eval.py:183-185`、`lopo_v3_multiseed.py:196-198`）
> - 阶段级 RF within-task **part 内 pooled AUC 0.9258**（总体 0.9353）（`results/stage_experiment/stage_fair_summary.json` 第 3 行；`docs/STAGE_TASK_REPORT_20261031.md` §3.1）
> - 习题级 IPSD 前 1 题 0.7915；坐下级第 1 次坐下 0.8665 / +会话结构 0.8644（`docs/paper_v10_zh.md:502-509`）

### 1.1 少样本 / FoMAML 类（全部判负或不可用）

| # | 实验 | 脚本（路径） | shot 数 & 支持集抽样 | 主指标 | 最终数字 | 对照基线 | 判定 / 差值 |
|---|---|---|---|---|---|---|---|
| **F1** | per-part balanced 5-shot FoMAML（FiLM 全开） | `models/meta_mamba_7d/train_and_eval_per_part_5shot.py` → `models/meta_mamba_7d/eval_per_part_5shot.py:101-230` | K=5/class（**class-balanced**：5 passed + 5 failed），支持集 = **段内 dominant-part 学生**；query = 该 part 其余学生，截断 n_query_max=50；10 个 eval seed（0-9）；inner_lr=0.01, inner_steps=3, **一阶梯度**（`create_graph=False`） | 段内 roc_auc / acc / macro_f1 / f1_failed / pr_auc，**按 part 分别算再跨 part 取均值（不是 pooled）** | **仅 2/7 part 可评**（日志标 p1,p2,p4,p5 因某类 <5 SKIP；p7 n=0）<br>**日志 [Part 3]**（= task_ids 3 = **原始 part 4**）: acc 0.8889 / macroF1 0.4706 / F1(F) 0.9412 / **ROC 0.7000 ± 0.0592** / PR 0.9495<br>**日志 [Part 6]**（= task_ids 6 = **原始 part 7**）: acc 0.8400±0.0340 / macroF1 0.4564 / **F1(F)=0.000** / **ROC 0.5969 ± 0.0819** / PR 0.2477 | within-task ROC **0.9188** | **❌ 判负**（且**实现有缺陷 D1/D2/D3**）。原始 part4 −0.219；原始 part7 −0.322。**"acc 0.84" 是退化解**（该 part 134 pass/28 fail，常数全 pass = 134/162=0.827），F1(F)=0 直接暴露了这点。**原始 part 1（131 人）从未被评估** |
| **F2** | 同上 · adaptive k（k=min(5, min(n_pass,n_fail))） | `models/meta_mamba_7d/eval_only_adaptive.py`（复用 F1 的 ckpt `results/per_part_5shot_full/final_model.ckpt`） | k 上限 5，实际生效 k（按日志 part 编号）：p1=2, p2=4, p3=5, p4=3, p5=4, p6=5 | 同上 | p1 acc 0.956 **ROC NaN**；p2 acc 1.0 **ROC NaN**（n_query 16）；p3 ROC **0.6960±0.0656**；p4 acc 1.0 **ROC NaN**（n_query 8）；p5 acc 1.0 **ROC NaN**（n_query 1）；p6 ROC **0.5927±0.1025** | 同上 | **❌ 指标不可用**：6 个 part 里 4 个 ROC=NaN（query 集单类）。"acc=1.0" 全部来自退化单类 query，不是学习结果。**受 D1/D2/D3 影响** |
| **F3** | per-part adaptive **3-shot** | `models/meta_mamba_7d/eval_only_adaptive.py`（`--k 3 --adaptive`） | k 上限 3，实际 k：p1=2, p2..p6=3 | 同上 | p1 ROC **NaN**；p2 ROC **0.3882±0.4430**；p3 ROC 0.6963±0.0599；p4 ROC **NaN**（n_query 8）；p5 ROC **0.4500±0.4378**（n_query 仅 3）；p6 ROC **0.6280±0.0895** | 同上 | **❌ 判负 + 不可用**：p2/p5 的 AUC 低于 0.5；p1/p4 无法计算。**受 D1/D2/D3 影响** |
| **F4** | **Plan B** participation-based K-shot sweep（K=5/10/15/20） | `models/meta_mamba_7d/eval_participation_kshot.py`（仓库内**未入库**，`git status` 显示 `??`）→ `results/per_part_kshot_B/` | 支持集来自**在该 part 有过事件的全部学生**（pool 215–471 人），**强制 task_id = p-1**（FiLM 输入一致化）；K passed + K failed 平衡抽；query = pool 其余，截断 50 | 段内 roc_auc 等，跨 part 汇总 | 24 组 (7 part × 4 K) 的 ROC 全部落在 **0.4826–0.6276**，中心 ≈ **0.537**；acc 从 p1 0.664 → p6 0.296（纯退化） | ≈ 随机 (0.5) | **❌ 判负（最干净的反证）**：**对 K 完全无响应**（K=5→20 数值不动），全场 ≈ 随机。说明"少样本"部分在错误抽样下退化为"无信号"。**✅ 本实验实现无误**（`theta` 快照 → 就地适配 → 用适配后权重预测 → 还原，见 `eval_participation_kshot.py:199-216`），part 标签用的是原始 part 1..7（`p_idx = part-1`），**结论可直接引用** |
| **F5** | K-shot sweep（K=1/3/5/10/20） | `246:/home/ubuntu/studentrisk_stage/run_kshot_sweep.py` → `246:results/kshot_sweep/summary.json` + `run.log` | 同 F1（balanced、段内），n_query_max=30，评估 `EVAL_PARTS=(1,3,4,7)`（**这些数字直接当 task_id 用** → 同样有 D2） | F1(F) / ROC / acc（段内） | 日志 [Part 3]（= task_ids 3 = **原始 part 4**）: **K=3 → F1(F) 0.926 / ROC 0.740**；**K=5 → F1(F) 0.951 / ROC 0.712**；K=10 → F1(F) 0.919 / ROC NaN；K=1 与 K=20 全部 SKIP<br>日志 [Part 4]（= task_ids 4 = **原始 part 5**）: K=3 → F1(F) 1.0 / ROC NaN；K≥5 SKIP<br>日志 [Part 1]（= task_ids 1 = **原始 part 2**）: 所有 K 全 SKIP（只有 2 个 passed）<br>日志 [Part 7]: 所有 K 全 SKIP（**task_id 7 不存在**，n=0） | — | **❌ 假设无法检验 + 双重缺陷（D1+D2）**：docstring 假设是"近 OOD part(1,3,4) 在 K=5 就能 F1>0.85、远 OOD part7 需要 K≫5"——但 **"part1" 其实是原始 part2、"part7" 根本不存在**，假设的两支都落空；已可检的原始 part4 上 ROC 随 K **不升反降**（0.740→0.712） |
| **F6** | EXP-3 · 5-shot FoMAML **A-plan**（per-part dominant） | `run_4_gen_experiments.py:193-259` → `results/generalization_4runs/summary.json:exp3_5shot_A` | K=5/class，段内 dominant students，n_query=10，inner_lr 0.01 / steps 3，一阶，10 seed | F1(F) / ROC（段内） | p1 F1(F) **0.989** / ROC 0.7222±0.0786<br>p2 F1(F) 0.973 / ROC 0.6250±0.4041<br>p3 F1(F) 0.881 / ROC 0.6083±0.1548<br>p4 F1(F) 0.910 / ROC 0.5634±0.2917<br>**p7 F1(F) 0.018 / ROC 0.2207±0.1996**<br>overall mean F1(F) **0.7544** | within-task ROC 0.9188 / F1(F) 0.9111 | **❌ 判负**：近 OOD p1/p2 也只有 ROC 0.72/0.63（−0.20/−0.29）；**p7 ROC 0.221 < 0.5 = 比随机还差**，F1(F)=0.018 |
| **F7** | EXP-4 · 5-shot FoMAML **B-plan**（per-part participated） | `run_4_gen_experiments.py:262-357` → `results/generalization_4runs/summary.json:exp4_5shot_B` | K=5/class，参与式 pool，task_id 强制=p-1，n_query_max=50，10 seed | ROC / F1(F)（段内） | overall_mean_roc_auc **0.5369**（≈随机）；逐 part ROC 0.4826 / 0.5495 / 0.5101 / 0.5654 / 0.5237 / 0.5034 / 0.6236；**p1 F1(F)=0.034，p5 F1(F)=0.000** | ≈随机 | **❌ 判负**：与 F4 完全一致。`docs/paper_v8_generalization_zh.md:254` 明确写"**B-plan（random sampling）全场 ROC≈0.537 ≈ 随机——说明 dominant-part sampling 是 FOMAML 有效的必要条件**"（即：换一个抽样方式，FoMAML 这条线就归零） |
| **F8** | EXP-6 · 5-shot FoMAML A-plan **−FiLM** | `246:/home/ubuntu/studentrisk_stage/run_nofilm_lopo_fomaml.py:exp3_fomaml_A_nofilm` → `246:results/generalization_nofilm/fomaml_A_nofilm.json` | 同 F6，但 `use_film=False` | F1(F) / ROC（段内） | p1 F1(F) 0.962 / **ROC 1.0000**<br>p2 0.946 / 0.6528<br>p3 0.896 / 0.5018<br>p4 0.838 / 0.5606<br>**p7 F1(F) 0.165 / ROC 0.7011**<br>overall mean F1(F) **0.7613** | Full+FiLM：p7 ROC **0.2207** | **⚠️ 不是"少样本成功"，是"FiLM 有害"的机制证据**：p7 ROC **0.221 → 0.701（Δ=+0.480）**，但 p3/p4 反而从 0.608/0.563 掉到 0.502/0.561。`docs/paper_v8_generalization_zh.md:233-236` 定性为"FiLM 把模型锁在 seen-task 表征上，对远 OOD + few-shot 形成负迁移" |
| **F9** | 5-shot **task_emb-only** 适配（EXP-C） | `246:/home/ubuntu/studentrisk_stage/exp_C_few_shot.py` → `246:results/few_shot_study/exp_c_5shot_emb.json` | leave-one-task-out；K=5 support 来自留出 task；**只更新 `model.film.task_emb`**（20 步 Adam，lr=1e-3），init = 其他已训 task_emb 的均值 | macro_f1 / f1_class_1 / roc_auc（留出 task 的 query 集） | **5 条记录 `adapted_emb` 全部为 `null`**（只存下了 `no_adapt` zero-shot 对照）<br>no_adapt: held0 ROC 0.880 / held1 0.690 / held2 0.483 / held3 0.664 / held6 0.551；macro_f1 0.494/0.490/0.441/0.451/0.151 | — | **⚠️ 实验未完成**：磁盘上没有 adapted 数字，**既不能算成功也不能算证伪**。这是本项目里唯一一个"少样本适配 vs 不适配"严格对照的设计，值得补跑 |
| **F10** | notc+FoMAML 混合跑批 | `results/meta_mamba_7d_notc_fomaml/results.json` | K=5, n_query=10, inner_steps=3, 一阶 | F1 | `fewshot_fomaml.mean_f1 = 0.7673`，**std 0.3858**（n_tasks=5） | within-task macro_f1 0.8754 | **⚠️ 不稳定**：均值 0.767 但标准差 0.386 → 单个 part 崩塌，跨 part 不可用 |

### 1.2 跨段迁移 / LOPO 类

| # | 实验 | 脚本 | 协议 | 最终数字 | 判定 |
|---|---|---|---|---|---|
| **L1** | LOPO v1（5-fold 内循环 + 全量 train 最终模型） | `lopo_train_eval.py` → `results/lopo/` | 每 part 留出该 part 的学生（按 dominant part），其余训练；seed 42；5-fold 内 CV + 90/10 早停；40 epochs | 逐 part ROC：p1 0.7231 / p2 0.6184 / p3 0.6125 / p4 0.6244 / p5 0.7576 / p6 0.6000 / **p7 0.5584**<br>F1(F)：p1 0.9923 / p3 **0.000** / p7 **0.2947**<br>**mean ROC 0.6421 ± 0.0712**，acc 0.6118，macro_f1 0.3515 | **❌ 硬墙**：ROC 0.642 vs within-task 0.9188 → **−0.277** |
| **L2** | LOPO v2（每 part 只训一次，去掉内层 CV） | `lopo_v2.py`（写同一个 `results/lopo/`，覆盖了 v1） | 同上但单次训练 | `results/lopo/lopo_metrics.csv` 带 `n_train/n_val/best_val_loss/train_time_s` 列（v2 格式） | **⚠️ 与 L1 已被覆盖**，无法分离 |
| **L3** | LOPO v3 · **3 seeds**（42/123/777，epochs 30） | `lopo_v3_multiseed.py` → `results/lopo_3seed/` | 每 part × 3 seed | 逐 part ROC：**p1 0.7564 / p2 0.5768 / p3 0.5417 / p4 0.5481 / p5 0.6162 / p6 0.7333 / p7 0.5215**<br>**mean ROC 0.6134 ± 0.0949**，acc 0.6179，macro_f1 0.3644 | **❌ 硬墙复现**：0.6134 vs 0.9188 → **−0.305**。seed 方差巨大（p3 ROC std 0.203） |
| **L4** | LOPO v3 · **−FiLM**（3 seeds） | `lopo_v3_multiseed_nofilm.py` → `results/lopo_3seed_nofilm/` | 同 L3，`use_film=False` | **目录为空（只有 `results/lopo_3seed_nofilm_run.log`），日志在 part2 seed777 处中断** | **⚠️ 未跑完**（本地）。246 上的 −FiLM LOPO 是另一条实现（见 L6） |
| **L5** | **per-student** LOPO（王建 2026-10-11 意图：同一学生未见过的 part） | `per_student_lopo_v2.py` → `results/per_student_lopo/` | 重构 per-(student,part) 序列；选"做过全部 7 part"的学生（**候选 213 人中只找到 15 人**）；每人留出 1 part；FiLM ON/OFF × seed 42 × epochs 20 → 210 条记录 | FiLM OFF：acc **0.4381**、macro_f1 0.4381、F1(P) 0.3429、F1(F) 0.0952、**ROC NaN**<br>FiLM ON：acc **0.4000**、macro_f1 0.4000、F1(P) 0.2381、F1(F) 0.1619、**ROC NaN** | **❌ 彻底失败**：二分类 acc 0.40–0.44 ≈ 抛硬币；ROC 不可算（每人留出 part 只有 1 个样本）。**"同一学生换 part 也认不出"** |
| **L6** | LOPO 多 seed（论文口径 EXP-2） | `run_4_gen_experiments.py:126-191` → `results/generalization_4runs/summary.json:exp2_lopo` | 3 seed，per-part | 逐 part ROC：p1 0.7692 / p2 0.5768 / p3 0.5500 / p4 0.5467 / p5 0.6465 / p6 0.7333 / **p7 0.5113**<br>**grand_mean_roc_auc 0.6191**，acc 0.6112，macro_f1 0.3580 | **❌ 硬墙**（与 L3 一致，0.619 vs 0.919） |
| **L7** | LOPO · −FiLM（论文口径 EXP-5） | `246:run_nofilm_lopo_fomaml.py:exp2_lopo_multi_seed_nofilm` → `246:results/generalization_nofilm/lopo_nofilm.json` | 3 seed，per-part，use_film=False | 逐 part ROC：p1 0.7872 / p2 0.5175 / p3 0.3542 / p4 0.7222 / p5 0.7071 / p6 0.7833 / p7 0.5397<br>**grand_mean_roc_auc 0.6302**，grand F1(F) 0.7516，acc 0.6773 | **➖ 中性**：去掉 FiLM 后 LOPO grand ROC 0.619 → 0.630（+0.011，基本没变）；说明**学生级 LOPO 的墙不是 FiLM 造成的**，去掉 FiLM 救不回来（对比 F8 在 few-shot 场景下的 +0.48） |
| **L8** | **阶段级 LOPO（当前主协议，RF vs MetaMamba）** | `analysis/stage_experiment/compare_lopo.py` + `246:analysis/stage_experiment/run_stage_fair.py --protocol lopo` → `results/stage_experiment/stage_lopo_summary.json` | 留一 part（part1–6，part7 无正例 AUC 不可算）；train=其余 part，test=留出 part；seeds 42/123/777；epochs 30 | 逐 part：p1 MM 0.9107 / RF 0.9452；p2 0.7762 / 0.8909；p3 0.8850 / 0.9260；p4 0.8765 / 0.9100；p5 0.9217 / 0.9376；p6 0.8443 / **0.8435**<br>**n 加权 LOPO AUC：MM 0.8681 vs RF 0.9128，Δ = −0.0447，95% CI [−0.0645, −0.0241]，P(Δ≤0)=100%** | **⭐ 关键对照**：**RF 的阶段级 LOPO = 0.9128 vs 其 within-task pooled 0.9258，只掉 0.013** → **跨段迁移在阶段级 RF 上几乎免费**；掉 0.045 的是 MetaMamba（0.8681 vs 0.9258 口径不同但同为阶段级） |
| **L9** | CS2-D **诊断通道** LOPO | `analysis/hazard_model/run_cs2_diag.py --mode lopo` → `results/cs2_diag_lopo/results.json` | leave-one-part-out 训练 → 测试 | **top1 0.5721 / top3 0.7512 / MRR 0.6876，n=201**（cv 口径 0.6070/0.8060/0.7219，**仅 −0.035**）；最好启发式 max_focus_churn top1 0.3333 | **✅ 不判负**：诊断通道跨段几乎不掉，且远高于启发式。`docs/NEXT_EXPERIMENTS_20261104.md:115-116`："判别线 LOPO 是硬墙，诊断线不是" |
| **L10** | CS2-D 跨段正/反向 | `results/cs2_diag_e2l/`、`results/cs2_diag_l2e/` | part1–4 训练 → part5–6 测试；反向 | 早→晚：**top1 0.5484 / top3 0.6774 / MRR 0.6697，n=31**<br>晚→早：**top1 0.1647 / top3 0.5588 / MRR 0.4122，n=170** | **⚠️ 方向不对称**：早→晚可用（0.548 ≫ 启发式 0.333）；**晚→早 0.1647 低于启发式 0.3333** → 原因是晚段数据太少（part5+6 共 486 行，停点仅 31 段），不是规律相反（`docs/NEXT_EXPERIMENTS_20261104.md:118`） |
| **L11** | 8d 序列 · −FiLM · 3seed LOPO | `246:results/lopo_3seed_no_film_8d/records.json` | 3 seed，per-part，8 维序列、无 FiLM | 逐 part ROC：p1 0.985–0.992 / p2 0.974–0.980 / p3 0.888–0.912 / p4 0.691–0.749 / p5 0.727–0.939 / p6 0.450–0.750 / **p7 0.514–0.530** | **❌ 关键否证**：**去掉 FiLM、换 8 维序列，part7 仍然塌到 ~0.52** → far-OOD 崩塌**不是 FiLM 造成的**（与 L7 呼应） |
| **L12** | zero-shot · 4 种 FiLM context 对照 | `246:results/zero_shot_a/driver.log` | LOPO 留出 part 后，用 4 种 task context 推理 | 留出 part7（n=162）：one_hot_true AUC **0.4286** / one_hot_wrong 0.4451 / mean 0.4773 / random 0.5259，**acc 全为 0.1728**<br>留出 part6（n=9）：0.600 / 0.550 / 0.500 / 0.550 | ≈随机 | **❌ 判负**：**FiLM 的 task context 本身不携带任何可迁移信号**，换哪种 context 都≈随机（且 true context 最差 0.4286） |
| **L13** | dummy task_id（within-student 5 折，473 学生 OOF） | `analysis/control_exp_20261028/results/dummy_task_id_lopo/summary.csv` | 同折 OOF，只改 task_id 喂法 | film_on+**true** 0.8837 / 0.8715 / **0.9188**<br>film_on+random 0.7484 / 0.6837 / 0.8787<br>film_on+mean 0.6617 / 0.3982 / 0.8911<br>film_on+zero(0) 0.6638 / 0.3990 / 0.7593<br>film_disabled(γ=1,β=0) 0.6850 / 0.4777 / 0.8966<br>no_film 0.6808 / 0.4659 / 0.7114 | true task_id 本身 | **⚠️ 机制澄清**：true task_id 在 **within-task** 上贡献巨大（macro_f1 0.8715 vs random 0.6837，**是记忆不是泛化**）；一旦换 part（L1/L3）就没了 → 证明"FiLM 学到 part-specific 模式"只在 seen part 内成立 |

### 1.3 论文/备忘里已写明的"不要再做"清单（含出处）

`docs/NEXT_EXPERIMENTS_20261104.md` §4「明确不要再做的（9 条已实测证伪）」：

| 方向 | 实测 |
|---|---|
| 换学习器（LGBM / HGB / MoE / 集成） | 全部 ≤ RF（0.9255–0.9268 vs 0.9258） |
| 加特征（多尺度 14 维 / M4 强度 / 截止压力） | −0.0058 / −0.0013 / +0.0020(不显著) |
| 特征层残差化 | −0.027 |
| 直接优化主指标（LambdaRank） | −0.0229，P(Δ≤0)=99.05% |
| 序数回归（预测最终到第几段） | −0.0169 |
| 每段单独建模 | 0.9216 < pooled 0.9258 |
| 深度架构判别线 | MetaMamba −0.1302；CS2-SPC 0.79 vs RF 0.87；CS2-FWD 0.82 vs RF 0.91 |
| 段内多尺度/会话结构特征 | 坐下级内 0.8644 vs 0.8665（无增量） |
| 坐次级 FiLM 进度调制 | 0.7840 vs 0.7912（略负） |

其余已写明结论：
- `docs/paper_v8_generalization_zh.md:254-256`：B-plan 全场 ROC≈0.537≈随机 → "K=5 few-shot 不是万能的，必须配正确的 sampling strategy"；且"**Design contribution 而非 generalization evidence**"
- `docs/paper_v8_generalization_zh.md:366-369`：EXP-2 LOPO grand ROC 0.619；EXP-3 K=5 让 part1/2/3/4 F1>0.88 但 part7 仍崩（F1=0.018）；EXP-5/6 FiLM 在 far OOD+few-shot 是净伤害
- `docs/STAGE_TASK_REPORT_20261031.md` §4：「架构线关闭」——四项全开 MetaMamba 仍输给同口径 RF（−0.1302，P(Δ≤0)=100%）

### 1.4 实现缺陷审计（2026-11-05 本次盘点新增，全部已逐条复核）

> 复核方式：读源码 + 对照 `results/*/run.log` 的 part 人数 + 比对 `task_ids.npy` 分布 + md5 比对三处副本。
> 三处副本（本地仓库 / `246:/home/ubuntu/StudentRisk/` / `246:/home/ubuntu/studentrisk_stage/`）的 `eval_per_part_5shot.py` **md5 完全一致 = `96d4e62ac5a813ca807bc06b881f6881`**，所以 246 上跑出来的结果同样带这些缺陷。

#### D1 · 少样本适配是完全的 no-op（最严重）

```python
# models/meta_mamba_7d/eval_per_part_5shot.py
def fomaml_inner_adapt(model, ..., inner_lr=0.01, inner_steps=3, device='cpu'):
    theta = {k: v.clone() for k, v in model.state_dict().items()}   # L70  ← 适配【前】的快照
    for _ in range(inner_steps):
        ...
        with torch.no_grad():
            for p, g in zip(model.parameters(), grads):
                p.sub_(inner_lr * g)                                # L80-82 ← 就地适配 model
    return theta                                                   # L83  ← 返回【适配前】快照 ❌

# 调用处 L179-189
adapted_state = fomaml_inner_adapt(...)   # L179 → 拿到的是适配前的权重
q_prob = predict_with_state(model, adapted_state, ...)   # L185
#   predict_with_state 内部 L90: model.load_state_dict(state)  ← 加载【适配前】权重
#                            L97: model.load_state_dict(theta) ← theta 是 L88 拍的【适配后】状态
```

- **L83 应当 `return model.state_dict()`（适配后），实际 `return theta`（适配前）** → `predict_with_state` 把适配结果**原样覆盖掉**。
- **净效果：F1/F2/F3/F5 的"5-shot FoMAML"= 零样本、不适配**，3 步内层梯度（inner_lr=0.01）完全没生效。
- **额外副作用（D1b · 跨轮状态漂移）**：L88 的 `theta` 拍的是**已被就地适配过**的 model，所以 L97 还原回去的是"适配后"的权重 → 第 2 个 (part, seed) 迭代从"已被上一个支持集适配过"的权重继续 → **权重在 60 次迭代间持续漂移**。这解释了为什么 F1/F2 同一 part 的 10 个 seed 差异大到 0.63–0.75（原始 part 7：ROC 0.503→0.753）。
- **修复只需一行**：`return {k: v.detach().clone() for k, v in model.state_dict().items()}`（并把 `predict_with_state` 的还原改成适配前快照）。

#### D2 · part 编号 off-by-one

- `models/lstm_7d/data.py:46`：`task_ids[i] = int(ev['part'].mode().iloc[0]) - 1` → **0-indexed dominant part**。
- 实测分布（`results/meta_mamba_7d/task_ids.npy`）：`{0:131, 1:78, 2:24, 3:55, 4:14, 5:9, 6:162}`。
- 但三个脚本都写 `parts=tuple(range(1, n_tasks + 1))` = 1..7：
  `eval_per_part_5shot.py:289`、`eval_only_adaptive.py:78`、`train_and_eval_per_part_5shot.py:137`；`246:run_kshot_sweep.py` 里 `EVAL_PARTS=(1,3,4,7)` 也是**直接当 task_id 用**。
- **对照 run.log 验证**（`results/per_part_5shot_full/run.log`）：

  | 日志标签 | n | task_ids==? | 实际原始 part |
  |---|---:|---|---|
  | [Part 1] | 78 | task_ids==1 | **原始 part 2** |
  | [Part 2] | 24 | task_ids==2 | **原始 part 3** |
  | [Part 3] | 55 | task_ids==3 | **原始 part 4** |
  | [Part 4] | 14 | task_ids==4 | **原始 part 5** |
  | [Part 5] | 9 | task_ids==5 | **原始 part 6** |
  | [Part 6] | 162 | task_ids==6 | **原始 part 7** |
  | [Part 7] | 0 | task_ids==7（不存在） | — |
  | **（从未出现）** | **131** | **task_ids==0** | **原始 part 1 ← 从未评估** |

- **后果**：**原始 part 1（n=131，人数最多的近端 part）从未被评估过**；"Part 7" 是不存在的 part，它的 SKIP 被误读成"样本不够"。
- **口径正确的脚本**：`run_4_gen_experiments.py`（`part = tid + 1`）、`lopo_train_eval.py` / `lopo_v2.py` / `lopo_v3_multiseed*.py`（`held_out_part_1idx = held_out + 1`）、`per_student_lopo_v2.py`（`task_ids = part - 1`，报 `+1`）、`eval_participation_kshot.py`（用原始 part 1..7）。

#### D3 · 训练集泄漏进评估

- `train_and_eval_per_part_5shot.py:95-116`：循环 `for seed: for fold:`，**`final_model = model` 在每次 fold 后覆盖** → 最终 `final_model` = "最后一个 seed(777) 的最后一个 fold(4)" 的模型，**它已经见过 4/5 的学生**。
- 随后 `run_per_part_balanced_5shot(final_model, sequences, masks, task_ids, y, ...)` 在**全部 473 人**上抽支持集/query → **query 里含训练样本**。
- **对比基线是 OOF**（`results/generalization_4runs/summary.json:exp1_5fold_cv` = OOF 0.9188）→ **这四个实验的 query 指标是偏乐观的**。
- 即：**在"部分泄漏 + 不适配"的条件下它们仍然失败**（ROC 0.597 / 0.700 vs OOF 0.9188）——结论方向不变，但"失败幅度"被低估。

#### D4 · ROC-AUC 从不 pooled

- `eval_per_part_5shot.py:61`：`roc_auc_score(y_true, y_prob)` 对**每个 (part, seed) 的 query 子集**各调一次；聚合是 `df.groupby('part').agg(mean/std)`（`:216-231`）。
- **从未把 query 拼起来算一个 part 内 pooled AUC** → 与项目主指标（`stage_common.pooled()`，n 加权 part 内 AUC）**口径不同、不可比**。
- 单类 query → `roc_auc_score` 抛错被 catch → 记为 **NaN**（这就是 F2/F3 里 4~6 个 part ROC=NaN 的原因）。

#### D5 · 不可复现

- `git status --porcelain` 显示**所有少样本脚本与结果目录都未入库**：`?? models/meta_mamba_7d/eval_per_part_5shot.py`、`?? models/meta_mamba_7d/eval_participation_kshot.py`、`?? models/meta_mamba_7d/eval_only_adaptive.py`、`?? models/meta_mamba_7d/train_and_eval_per_part_5shot.py`、`?? reproduce_5shot_samples.py`、`?? results/per_part_5shot_full/`、`?? results/per_part_5shot_adaptive/`、`?? results/per_part_3shot_A/`、`?? results/per_part_kshot_B/`、`?? results/lopo/`、`?? results/lopo_3seed/`、`?? results/per_student_lopo/`、`?? results/generalization_4runs/`。
- **`results/per_part_5shot_full/final_model.ckpt` 不存在**（`ls` 确认）→ F2/F3 依赖的 ckpt 没了，**这两个实验无法从落盘产物复核**。
- 修复方向：把 `fomaml_inner_adapt` 的返回改对后，**重跑一次带 ckpt 的 F1**，同时把 `eval_per_part_5shot.py` 的 part 映射改成 `task_ids == part-1`，并把 ROC 改成 part 内 pooled。

### 1.5 尚未被证伪 / 尚未做完的少样本类（可作为新实验入口）

| 缺口 | 证据 | 备注 |
|---|---|---|
| **EXP-C task_emb-only 适配未跑完** | `246:results/few_shot_study/exp_c_5shot_emb.json` → `adapted_emb: null`（5/5） | 唯一带 zero-shot 对照的适配实验；补跑成本低（K=5、20 步、只更新 `model.film.task_emb`）。**现在优先级更高了**：D1 证明原来的 5-shot 实现是 no-op，F1/F2/F3/F5 全部不能作为"少样本有效/无效"的依据 |
| **修好 D1 后重跑 F1（5-shot FoMAML）** | `eval_per_part_5shot.py:83` | 一行修复 + 修 part 映射 + 改 part 内 pooled AUC → 才能拿到第一个**真正有效**的 5-shot 数字。原始 part 1（131 人）必须补上 |
| **LOPO −FiLM 学生级 3seed 未跑完** | `results/lopo_3seed_nofilm/` 为空；`results/lopo_3seed_nofilm_run.log` 中断 | 但 L7/L11 已用另外两条路径回答了同一问题（去掉 FiLM 救不回 LOPO）→ **优先级低** |
| **"新学生冷启动"少样本**（新 part 只有 5–10 个标注样本，且是 stage 级而非 student 级） | 现有全部少样本实验都在 **student 级序列**上做；阶段级只有 LOPO（0-shot） | 阶段级 LOPO 的 RF 只掉 0.013 → **"阶段级 + 少量新段标注"的少样本设定看起来几乎肯定能赢，但从未测过** |
| **按 part 分层的 K-shot（真·跨段少样本）** | F1–F8 的 support 全部来自**同一个 part 内部**；跨段迁移只测过 0-shot（LOPO） | 真正的"跨段 K-shot"= 在 part A 上标 K 个、在 part B 上测 —— **项目里一个都没有** |

---

## 2 · 每个少样本实验的「4 个数」速查表

| 实验 | ① shot 数 / 支持集抽样 | ② 主指标 | ③ 最终数字 | ④ 结论 & 差值 |
|---|---|---|---|---|
| F1 per-part 5-shot | K=5/class 平衡；段内 dominant 学生；query=段内其余≤50；10 seed | 段内 roc_auc（跨 part 取均值，**非 pooled**） | **日志[Part 3]**（原始 part4）0.700±0.059；**日志[Part 6]**（原始 part7）0.597±0.082；仅 2/7 part | 比 full-label 差 0.219 / 0.322；原始 part7 F1(F)=0（退化全 pass）。**⚠️ D1/D2/D3：适配实际未生效** |
| F2 adaptive 5-shot | k=min(5, min(n_pass,n_fail))，同上 | 同上 | [Part 3] 0.696；[Part 6] 0.593；4/6 part AUC=NaN | 不可用（单类 query）。**⚠️ D1/D2/D3** |
| F3 adaptive 3-shot | k 上限 3 | 同上 | [Part 2] 0.388；[Part 3] 0.696；[Part 5] 0.450；[Part 6] 0.628；[Part 1]/[Part 4] NaN | 差（[Part 2]/[Part 5] <0.5）+ 不可用。**⚠️ D1/D2/D3** |
| F4 Plan-B K-sweep | 参与式 pool（215–471 人）+ task_id 强制 | 段内 roc_auc | 24 组全在 0.483–0.628，中心 0.537 | ≈随机；对 K 无响应。**✅ 实现与 part 标签均正确** |
| F5 K-shot sweep | 段内平衡，n_query≤30；EVAL_PARTS=(1,3,4,7) 直接当 task_id | F1(F)/ROC | [Part 3]（原始 part4）K3 0.740 / K5 0.712；[Part 1]/[Part 7] 全 SKIP | 假设无法检验 + 无 K 趋势。**⚠️ D1/D2** |
| F6 EXP-3 A-plan | K=5，段内 dominant，n_query=10 | F1(F)/ROC | overall F1(F) 0.754；**p7 ROC 0.221** | p7 比随机还差；p1/p2 −0.20/−0.29。**✅ 实现正确，part 标签正确** |
| F7 EXP-4 B-plan | K=5，参与式 | ROC | overall 0.5369 | ≈随机。**✅ 实现正确** |
| F8 EXP-6 A-plan −FiLM | 同 F6，无 FiLM | F1(F)/ROC | overall F1(F) 0.761；**p7 ROC 0.701** | p7 +0.480（机制证据），p3/p4 变差。**✅ 实现正确** |
| F9 EXP-C task_emb | K=5，跨 task，只更新 task_emb | macro_f1/AUC | adapted 全部 null；仅 no_adapt 0.88/0.69/0.48/0.66/0.55 | **未完成** |
| F10 notc_fomaml | K=5, n_query=10 | F1 | 0.7673 ± **0.3858** | 不稳定 |
| L1 LOPO v1 | 留一 part，0-shot | ROC/F1(F) | mean ROC 0.6421 | −0.277 |
| L3 LOPO 3seed | 留一 part × 3 seed | ROC | mean ROC 0.6134 ± 0.0949 | −0.305 |
| L5 per-student LOPO | 留一 (student,part)，15 学生 | acc/macro_f1 | acc 0.400 (FiLM) / 0.438 (−FiLM)，ROC NaN | 抛硬币级 |
| L6 EXP-2 LOPO | 留一 part × 3 seed | ROC | grand 0.6191 | −0.300 |
| L7 LOPO −FiLM | 同上 | ROC | grand 0.6302 | −0.011（FiLM 不是原因） |
| L8 阶段级 LOPO | 留一 part（RF / MM 同口径） | n 加权 LOPO AUC | **RF 0.9128 vs MM 0.8681**，Δ=−0.0447，P=100% | **RF 只比 within-task (0.9258) 低 0.013** |
| L9 CS2-D LOPO | 留一 part（诊断通道） | top1/top3/MRR | 0.5721/0.7512/0.6876 | 只 −0.035，不判负 |

---

## 3 · 数据集与特征文件清单

### 3.1 原始事件级（唯一真源，246 + 网关备份，md5 一致）

| 绝对路径 | 大小 | md5 | 行数 / 形状 | 字段 |
|---|---|---|---|---|
| `/home/ubuntu/IDE_logs/IDE_logs.csv`（246）<br>`/root/.openclaw/data-backups/IDE_logs.csv`（网关，字节一致） | 1,360,455,804 B (1.27 GiB) | `d4b1304f08d7f662d26b81079900757e` | **28,588,309 行 × 6 列**（含表头 28,588,310 行） | `student`(int, 0–472)、`part`(int, **1–7**)、`exercise`(int, **1–30**)、`eventType`(str)、`timestamp`(str `YYYY-MM-DD HH:MM:SS`)、`timeToDeadline`(int, 秒, 单调递减)。473 学生；**2171 个 distinct (student,part) 组** |
| `/home/ubuntu/IDE_logs/passed.csv`（246）<br>`/root/.openclaw/data-backups/passed.csv`（网关） | 4,950 B | `183120003794f4ecc2d6afd017ed401d` | **473 行 × 2 列** | `student`(int64)、`passed`(bool)；passed=True **159** / False **314**（挂科率 0.6638） |
| `/home/ubuntu/IDE_logs/README.txt` | 797 B | — | — | 数据说明；**两个文件均 chmod 444，README 明令不得删除**（2026-10-28 曾误删一次） |

**eventType 全量分布（28.59 M 行）**：`text_insert 19,652,100 (68.76%)` > `text_remove 5,582,299 (19.53%)` > `focus_gained 1,427,004 (4.99%)` > `focus_lost 1,383,954 (4.84%)` > `run 323,749 (1.13%)` > `text_paste 166,601 (0.58%)` > `submit 52,602 (0.18%)`。

### 3.2 246 上的特征表（`/home/ubuntu/studentrisk_stage/data/`）

| 绝对路径 | 大小 | 形状 | 关键列 | 标签 / part 表示 |
|---|---|---|---|---|
| `data/stage_dataset.npz` | 2,010,427 B | `seq11 (2171,128,11) f32`、`seq9 (2171,128,9)`、`seq7 (2171,128,7)`、`mask (2171,128) f32`、`agg10 (2171,10) f32`、`task_ids (2171,) i64`、`part_ids (2171,) i64`、`sid (2171,) i64`、`drop (2171,) i64`、`final (2171,) i64`、`first_ts/last_ts (2171,) f64`、`event_types (7,)` str | `agg10` = `[text_insert, text_remove, text_paste, focus_gained, focus_lost, run, submit, total_events, distinct_exercises, span_hours]`，均值 `[9052.1, 2571.3, 76.7, 657.3, 637.5, 149.1, 24.2, 13168.3, **2.95**, 90.63]`；`seq11` = [et_oh(7) \| log_interval \| deadline_norm \| part_norm \| exercise_norm]；`seq9` 去 part/exercise 通道；`seq7` 只留事件 one-hot | **`drop` (i64 0/1)**：**258 个正例（11.88%）**。规则：该 (学生,part) 是他**到达过的最远 part** 且**从未到达 part7** → 做完这一段他就不再推进（从日志事实推导，**无新标注**）<br>**`final`**：课程级 failed，每行重复（1069 个正例行 = 49.24%）<br>**`part_ids` 1..7**（原始 part）；**`task_ids` = `part_ids - 1`（0..6，已逐行验证 `task_ids == part_ids-1`）**；`sid` 0..472 |
| `data/exercise_dataset.npz` | 1,048,958 B | `raw (44764,12) f32`、`sid (44764,)`、`part (44764,)`、`exercise (44764,)` 1..30、`sp_s (2171,)`、`sp_p (2171,)`、`sp_ex (2171,)` object、`part_tot (7,)`、`part_ids (7,)`、`event_types (7,)` | `raw` 12 列习题级原始聚合；`sp_ex[j]` = 第 j 个阶段做过的习题号列表（object 数组） | 习题级：行 = (学生, part, 习题)；`part` 1..7；`exercise` 1..30（但**每 part 可用习题集不同**，见 `exercise_structure.npz:part_ex_n` = 11..30） |
| `data/exercise_structure.npz` | 8,717 B | `touched_keys (2171,)` str `'sid\|part'`、`touched_vals (2171,)` object、`part_ex_n (7,)` | `part_ex_n` = 每 part 题量 11..30 | — |
| `data/deadline_feats.npz` | 64,025 B | `feats (2171,9) f32`、`sid (2171,)`、`part (2171,)` 1..7、`names (9,)` | `names` = `['ttd_start_h','ttd_end_h','utilization','submit_slack_h','submit_ratio','run_slack_h','rate_per_h','rush_ratio','no_submit']`（max 3109.3, mean 151.09） | 阶段级截止压力特征；**行顺序与 `stage_dataset.npz` 完全一致**（已验证 `sid`/`part` 相等）→ 可直接按行拼接 |
| `data/multiscale_feats.npz` | 87,171 B | `feats (2171,14) f32`、`sid (2171,)`、`part (2171,)`、`names (14,)` | `names` = `['sess_per_stage','events_per_sess','sess_with_sub_ratio','run_per_ins','edit_no_run_ratio','rework_ratio','focus_dwell_mean','active_days','span_hours',…]` | 阶段级会话/多尺度 14 维（**行顺序同样与 stage_dataset 一致**）；**已证伪**：加入特征集 −0.0058 |
| `data/student_event_seqs.npz` | 11,347 B | `ids (473,)` object、`lengths (473,)` 1..425、`win_sec = 3600` | 学生级事件序列（ragged int8 符号序列，1 小时窗） | 学生级 |
| `data/student_win_feats.npz` | 976,293 B | `win_ids (473,)` object、`win_cnt (473,)` object、`lengths (473,)`、`win_sec = 3600` | 1 小时窗特征序列（EDM25 口径） | 学生级；**1 小时窗**（P2-8 说的"小时级风险轨迹"数据源） |

**`stage_dataset.npz` 逐 part 结构（新建少样本实验必读）**：

| part | 行数 | drop 正例 | drop 率 | final 正例 | task_id |
|---|---:|---:|---:|---:|---:|
| 1 | 471 | 83 | 17.6% | 313 | 0 |
| 2 | 388 | 67 | 17.3% | 230 | 1 |
| 3 | 321 | 31 | 9.7% | 163 | 2 |
| 4 | 290 | 40 | 13.8% | 132 | 3 |
| 5 | 250 | 15 | **6.0%** | 93 | 4 |
| 6 | 236 | 22 | 9.3% | 79 | 5 |
| 7 | 215 | **0** | **0.0%** | 59 | 6 |
| **合计** | **2171** | **258** | 11.88% | 1069 | — |

> 每生行数：min 1 / p25 2 / 中位 5 / p75 7 / max 7；**213 名学生做满 7 个 part**（`per_student_lopo_v2.py` 只找到 15 个可评估学生，原因就在这里）。
> 序列填充率：`mask` 用到 276,077 / 277,888 格（**99.35%**）→ 128 长度下几乎不稀疏，且**左填充**（尾部为最新事件）。
> `agg10[:,8]`（distinct_exercises）均值仅 **2.95**（可用习题 30 个）→ 每段做的题很少，重度偏斜。

> ⚠️ **本地仓库没有这些 npz**。本地 `StudentRisk/data/` 只有 `__init__.py` + `data_loader.py`。所有特征表只在 246 的 `/home/ubuntu/studentrisk_stage/data/`（另有一份原始仓库 `/home/ubuntu/StudentRisk/` 在 246 上，数据路径相同）。

### 3.3 坐下级 / 习题级截断产物（246 results）

| 绝对路径 | 大小 | 形状 | 说明 |
|---|---|---|---|
| `246:results/sitting/sit_k1_feat.npz` | 21,378 B | `rows (2171,)`、`oof (2171,)`、`y (2171,)`、`part (2171,)` | 第 1 次坐下截断 OOF；**part 内 pooled AUC 实测 0.8644**（全局 AUC 0.8905） |
| `246:results/sitting/sit_k2_feat.npz` | 11,189 B | `rows (1054,)` | 第 2 次坐下，pooled 0.8000 |
| `246:results/sitting/sit_k3_feat.npz` | 5,715 B | `rows (487,)` | 第 3 次坐下，pooled 0.7980 |
| `246:results/sitting/sit_k5_feat.npz` | 1,914 B | `rows (101,)` | 第 5 次坐下，pooled 0.3032（样本太少） |
| `246:results/exercise/exercise_results.npz` | 52,309 B | `lab_final (44764,)`、`lab_at (44764,)`、`ex_idx (44764,)`、`sid (44764,)`、`part (44764,)`、`exercise (44764,)`、`ex_k{1,2,3,5,10,999}_ps (473,)`、`part_K{1,2,3}..._ps (473,)` | 习题级 hazard 全量结果；学生级分数（取 max 聚合） |
| `246:results/ipsd/ipsd_results.npz` | 125,553 B | — | IPSD（习题级截断增广）结果包 |
| `246:results/cs2_fwd{2,3}/oof_*.npz` | 11–18 KB 每个 | `oof (n,)`、`y (n,)`、`part (n,)` | CS2-FWD RF / 深度 base / peer，K∈{0,1,3,10} |
| `246:results/cs2_fwd_clean/oof_*.npz` | 11–18 KB | `oof,y,part` | 剔除自参照的纯净口径 |

**坐下级截断口径**：一次坐下 = 同一 (学生,段) 内第 i 题结束时刻与第 i+1 题开始时刻间隔 < **30 min**（`analysis/hazard_model/run_sitting_truncation.py:SIT_GAP_SEC = 30*60`）；44,764 题 / 4,102 次坐下，中位一次坐下 ≈ 11 题。

### 3.4 已算好的 CV 划分 / OOF

**（a）学生级旧协议的 fold assignment（本地已有，`int64 (473,)`，值 0–4）**

```
StudentRisk/results/rf7/fold_idx.npy
StudentRisk/results/meta_mamba_7d/fold_idx.npy
StudentRisk/results/meta_mamba_7d_notc_fomaml/fold_idx.npy
StudentRisk/results/{meta_mamba,bilstm_7d,lstm_7d,attention_7d,meta_mamba_7d_ml512,
                     meta_mamba_7d_bs_32,meta_mamba_7d_lambda2_0_{1,5},
                     meta_mamba_ablation_*,sweep_lr0_*/d{32,64,128}} /fold_idx.npy
```
> 同目录还有 `labels.npy (473,)`、`oof_probs.npy (473,)`、`student_ids.npy (473,)`、`task_ids.npy (473,)`、`fold_metrics.csv`、`results.json`、`config_used.json`。
> ⚠️ 这些是**学生级 5 折 StratifiedKFold**（`lopo_train_eval.py` 等使用），**不是**阶段级的"按学生分组"折。
> 示例 `results/meta_mamba_7d_notc_fomaml/fold_idx.npy` = `[4 2 1 0 4 0 2 3 ...]`。

**（b）阶段级 OOF（本地 + 246）**——折**不落盘**，每次运行由 `stage_common.group_folds()` 现场生成

| 路径 | keys |
|---|---|
| `StudentRisk/analysis/stage_experiment/results/stage_oof_rf.npz` | `stage_part_prior, rf_full, rf_no_active, rf_active_only, rf_completion_only, rf_peer_all`（各 `(2171,)` f64） |
| `StudentRisk/analysis/stage_experiment/results/stage_oof_mm11.npz` | `stage_part_prior, mm11` |
| `StudentRisk/analysis/stage_experiment/results/stage_oof_mm7.npz` / `stage_oof_mm9.npz` | `stage_part_prior, mm7` / `mm9` |
| `246:results/stage_experiment/stage_fair_oof_cpu.npz` | `stage_part_prior, rf_full, rf_full_resid0.25/0.5/1.0, rf_full_resid, rf_full_resid_resid0.25/0.5/1.0` |
| `246:results/stage_experiment/stage_lopo_oof_lopo_rf.npz` | `rf_full (2171,)` |
| `246:results/stage_experiment/stage_lopo_oof_lopo_mm11_np_ctxraw.npz` | `mm11_np_ctxraw (2171,)` |

**（c）CS2-D 诊断通道 OOF**

```
StudentRisk/results/cs2_diag/oof_scores.json        # dict: student_id → list[float]（每学生每段一个分数）
StudentRisk/results/cs2_diag_lopo/oof_scores.json    # 同上，LOPO 口径
```

**（d）其他现成结果索引**

- `StudentRisk/results/generalization_4runs/summary.json`（14,654 B）——`exp1_5fold_cv / exp2_lopo / exp3_5shot_A / exp4_5shot_B`
- `StudentRisk/results/stage_experiment/stage_fair_summary.json`（16,257 B，55 行模型对比 + bootstrap）
- `StudentRisk/results/stage_experiment/stage_lopo_summary.json`（816 B）
- `246:results/generalization_nofilm/summary.json`（4,878 B）——EXP-5/6
- `246:results/few_shot_study/exp_c_5shot_emb.json`（1,981 B）
- `246:results/kshot_sweep/summary.json` + `run.log` + `part{3,4}_K{3,5,10}.jsonl`
- `246:results/lopo_3seed_film/film_records.json`（438 KB）、`246:results/lopo_3seed_film_8d/film_records.json`（437 KB）、`246:results/lopo_3seed_no_film_8d/records.json`（2,182 B）

### 3.5 标签 / part 列的表示法（统一口径）

| 概念 | 表示 | 位置 |
|---|---|---|
| **课程级标签** | `y=1` 表示 **FAILED（挂科）**，`y=0` 表示 PASSED。473 学生中 failed 314 (0.6638) / passed 159 (0.3362) | `data/data_loader.py:50-55 to_failed_label()`；`labels_df['failed'] = (passed == False)` |
| **阶段级标签 `drop`** | `drop=1` ⇔ 该 (学生,part) 是他**到达过的最远 part**，且**从未到达 part7**（即：做完这一段之后他就不再推进）。**从日志事实推导，无新标注** | `analysis/stage_experiment/build_stage_dataset.py:119` |
| **阶段级课程标签 `final`** | = passed.csv 的课程级 failed（每行重复该学生最终是否挂科） | 同上 `:118` |
| **part 列** | 原始日志 `part` 为 **1..7**；`stage_dataset.npz` 里 `part_ids` = 1..7，**`task_ids` = part-1（0..6）**；MetaMamba 的 FiLM 用 `task_ids`（0..6），共 `n_tasks=7` | `build_stage_dataset.py:115-116` |
| **学生 id** | 原始 `student` 0..472；`sid` 列同口径 | — |
| **序列口径** | 学生级 7 维事件序列：`(473, 128, 7)`，mask_sum=60395（每生平均 ~128 条截断后事件）；阶段级：`(2171, 128, {7,9,11})` | `models/lstm_7d/data.py:build_event_sequences_7d` |

---

## 4 · 可复用代码骨架（可直接抄）

### 4.1 ⭐ 首选：`analysis/stage_experiment/stage_common.py`（**本地已入库，163 行**）

| 函数 | 行号 | 作用 | 直接用于 |
|---|---|---|---|
| `load_dataset(path)` | **L14-25** | 读 `stage_dataset.npz` → dict(agg10, mask, task_ids, part_ids, sid, drop, final, first_ts, last_ts, seq11, seq9, seq7) + 别名 `RAW` / `part` / `stage_idx` | 数据加载 |
| `group_folds(sid, student_label, n_splits=5, seed=42)` | **L100-118** | **按学生分组 + 按学生课程级标签分层**的 K 折（同一学生绝不跨折），返回 `(tr, te)` 索引列表 | **"5 折按学生分组 × 3 seeds"协议本体** |
| `per_part_auc(y, oof, group)` | **L121-128** | 逐 part AUC（自动跳过单类 part） | 逐 part 读数 |
| `pooled(pp)` | **L131-136** | **part 内 pooled AUC**（n 加权平均各 part AUC） | **主指标** |
| `judged(pp)` | **L139-141** | 只对"正例≥10 且负例≥10"的 part 求均值（judgeable 口径） | 稳健口径 |
| `metrics(y, oof, group, tag)` | **L144-147** | 一次性返回 `auc`（全局）/ `pooled` / `judged` / `per_part` | 汇总 |
| `paired_boot(y, oa, ob, group, n=2000, seed=20261030)` | **L163-173** | **配对 bootstrap**：part 内分层重采样，返回 `{obs, lo(2.5%), hi(97.5%), p_le0}` | **显著性判定** |
| `derive(ds)` | L49-63 | 9 维派生特征（log 强度 / 比率 / 进度） | RF 特征集 |
| `self_ref(ds)` | L66-81 | 自参照（相对自己前几阶段的 z 分） | 消融（已知边际 0.005 不显著） |
| `peer_z(ds, tr_rows, tgt_rows, group_full)` | L84-97 | 同伴相对（**只算训练折**的 median/MAD，clip ±5） | 冷启动变体（已知边际 0.028 显著） |
| `base_exercises(agg10, part_ids)` | L39-46 | 每 part 题量众数基准 | 进度特征 |
| `_stage_index(sid)` | L28-36 | 每个学生在自己的 part 序列中的序号 | 进度/阶段序 |
| `stage_prior_oof(y, group, folds)` | L150-160 | part-prior 基线 OOF（stage_part_prior） | 对照 |

### 4.2 LOPO / 汇总 / 判定

| 文件 | 行号 | 作用 |
|---|---|---|
| `analysis/stage_experiment/compare_lopo.py` | **L30-122**（bootstrap **L90-111**） | LOPO 对照：逐 part AUC + **n 加权 LOPO AUC** + part 内重采样配对 bootstrap → `results/stage_experiment/stage_lopo_summary.json`（含 `bootstrap.lo/hi/p_le0`） |
| `analysis/stage_experiment/compare_stage_fair.py` | **L112-125** | 决定性配对 bootstrap（part 内分层重采样，Δ = pooled AUC 差） |
| `analysis/stage_experiment/run_layer0_decomposition.py` | **L118** | 变体 vs full 的配对 bootstrap 2000 次（写 `results/layer0_decomp/`） |
| `analysis/stage_experiment/run_stage_fair.py` | 全文件（含 `--protocol lopo` 分支） | 阶段级"四项全开"主程序：RF / MetaMamba 同口径 + LOPO 模式；产出 `stage_fair_oof_*.npz` |
| `analysis/stage_experiment/build_stage_dataset.py` | **L38-169** | `IDE_logs.csv` → `stage_dataset.npz` 的构建器（含 drop 标签推导 `:119`） |
| `analysis/stage_experiment/make_stage_report.py` | L44-45, L85, L125-127 | 报告生成器（含 bootstrap 段） |

### 4.3 划分工具：`generalization/data_splits.py`（本地已入库，504 行）

| 函数 | 行号 | 作用 |
|---|---|---|
| `lopo_split(labels_df, ide_logs_df, n_parts=7)` | **L187-252** | dominant-part LOPO 切分；返回 `(tr_idx, te_idx, meta)`，meta 含 `left_out_part / n_test / n_train / method`；**part 学生数 < 20 时发 RuntimeWarning**（L242-248） |
| `temporal_split(...)` | L127 | 时间外推切分（P1-7 的现成骨架） |
| `learning_curve_split(...)` | L87 | 学习曲线切分（少样本曲线用） |
| `perturbation_split(...)` | L255 | 扰动切分 |
| `stratified_subsample(y, n, rng)` | L47 | 分层子采样（造"少样本"支持集用） |

### 4.4 少样本 / FoMAML 实现

| 文件 | 行号 | 作用 |
|---|---|---|
| `models/meta_mamba_7d/eval_per_part_5shot.py` | `full_metrics` **L45-66**；`fomaml_inner_adapt` **L67-84**（一阶梯度 `create_graph=False`）；`predict_with_state` **L86-99**；`run_per_part_balanced_5shot` **L101-230**（支持集抽样 **L155-170**，聚合 **L213-230**） | **最完整的少样本实现**：balanced 支持集 / adaptive k / query 截断 / 10 seed / 每 part 独立指标 |
| `models/meta_mamba_7d/train_and_eval_per_part_5shot.py` | 全文件 | 5-fold CV 训练 + 落 `final_model.ckpt`，再调上面那个 eval（可分两步跑，省 GPU） |
| `models/meta_mamba_7d/eval_only_adaptive.py` | 全文件 | 只做 eval（复用 ckpt），adaptive k；F2/F3 就是它跑的 |
| `models/meta_mamba_7d/eval_participation_kshot.py` | 全文件（**未入库**） | Plan-B 参与式抽样 + task_id 强制 + K 扫描 |
| `run_4_gen_experiments.py` | `full_metrics` **L45-65**；`exp1_5fold_cv` **L67-125**；`exp2_lopo_multi_seed` **L126-191**；`exp3_5shot_A_plan` **L193-259**；`exp4_5shot_B_plan` **L262-357** | 论文口径 4 实验全家桶（Full+FiLM） |
| `246:run_nofilm_lopo_fomaml.py` | `exp2_lopo_multi_seed_nofilm` / `exp3_fomaml_A_nofilm` | −FiLM 对照（EXP-5/6） |
| `246:exp_C_few_shot.py` | `adapt_task_emb` **（约 L130-175）**；`train_one_fold` L70-127 | **task_emb-only 适配**（只训 `model.film.task_emb`，其余冻结）——F9 未跑完的那条，补跑直接用 |
| `246:run_kshot_sweep.py` | 全文件 | K-shot sweep（K∈{1,3,5,10,20}） |
| `reproduce_5shot_samples.py` | **L40-77** | 复现每个 (seed, part) 的确切支持集学生 id（`np.random.seed(seed*1000+part)`）→ 可复现性检查 |
| `lopo_train_eval.py` / `lopo_v2.py` / `lopo_v3_multiseed.py` / `lopo_v3_multiseed_nofilm.py` | L73-82（split）；L96-152（训练+预测）；L161-212（聚合） | 学生级 LOPO 四个版本（v3 是最干净的 3-seed 版） |
| `per_student_lopo_v2.py` | `build_per_student_part_sequences` **L54-101**；`select_full_part_students` **L102-113**；主循环 **L165-255** | per-(student,part) 序列重构 + 每生留一 part |

### 4.5 数据加载 / 指标 / 显著性

| 文件 | 行号 | 作用 |
|---|---|---|
| `data/data_loader.py` | `EVENT_TYPES` **L18-21**；`load_ide_logs` **L24-34**；`load_passed_labels` **L37-47**；`to_failed_label` **L50-55**；`load_dataset` **L58-98** | 唯一数据入口（路径可用环境变量 `STUDENTRISK_IDE_LOGS` / `STUDENTRISK_PASSED` 覆盖） |
| `models/meta_mamba_7d/data.py` | L1-11 | 7 维事件序列构建（转发到 `models/lstm_7d/data.py:build_event_sequences_7d`） |
| `models/base.py` | `evaluate_predictions` **L79-82**；`save_results` **L85-91** | 统一指标与落盘 |
| `analysis/significance.py` | `holm_bonferroni` **L45-54**；`sig_marker` L57-61；配对 t 检验 **L99-110** | **配对 t 检验 + Holm-Bonferroni**（fold 级，8fold×3seed=24 对）——与 bootstrap 互补 |
| `analysis/hazard_model/run_calibration_selective.py` | 全文件 | 校准 + 选择性预测（ECE / Platt / 拒答带 / AURC） |
| `analysis/hazard_model/run_sitting_truncation.py` | `SIT_GAP_SEC` L42；`group_peer_z` L45-52；`peer_stats` L54-65；`fit_rf` L67+ | 坐下截断三组对照（IPSD 复现 / 坐下 / 坐下+会话结构） |
| `analysis/hazard_model/run_exercise_hazard.py` | 全文件 | 习题级 hazard + 与 part 级公平对照 |
| `configs/default.yaml` | `cv.n_splits=5` L7；`cv.seeds=[42,123,777]` L8；`threshold.default=0.5` L49 | 全局超参 |

---

## 5 · 给新实验的建议（一句话版）

1. **先修代码，再下结论**：`eval_per_part_5shot.py:83` 的 `return theta` → `return {k: v.detach().clone() for k, v in model.state_dict().items()}`；`parts=range(1, n_tasks+1)` → `range(0, n_tasks)`（或统一 `task_ids == part-1`）；ROC 改成 part 内 pooled。**不修就不能引用 F1/F2/F3/F5 的任何数字**（包括论文里"5-shot FOMAML K=5 让 part1/2/3/4 F1>0.88"这句）。
2. **别再在学生级序列上做少样本**：学生级 LOPO 0.61、per-student LOPO 抛硬币、FoMAML 近 OOD 也只 0.72 —— 这条线已被 4 组**实现正确**的实验证伪（F4/F6/F7/F8 + L1/L3/L5/L6）。
3. **要做就做「阶段级 + 跨段 K-shot」**：阶段级 RF 的 LOPO 只掉 0.013（0.9128 vs 0.9258），说明跨段迁移在阶段级几乎免费；给新段标 5–10 个样本应该能拿到接近 0.90 的结果。这是**项目里从未被测过**的格子，也是现在性价比最高的实验。
4. **协议上直接抄 `stage_common.group_folds` + `paired_boot`**，主指标用 `pooled()`（part 内 pooled AUC），不要用"跨 part 取均值"（F1/F6 用的就是后者，数字不可比且被单类 part 污染）。
5. **两个能立刻补上的洞**：① EXP-C task_emb-only 适配（`246:exp_C_few_shot.py`）——只有 zero-shot 对照存下来了；② `lopo_v3_multiseed_nofilm.py`（本地跑了一半）。按 L7/L11 的结论，② 的预期收益低。
6. **样本量硬约束**（所有 part 内少样本设计都要先算这张表）：
   - 段内 dominant（原始 part1–7）：n = 131 / 78 / 24 / 55 / 14 / 9 / 162，passed = 2 / 4 / 10 / 3 / 5 / 134 / 0 → **balanced K-shot 只在原始 part3 (K≤10) 和原始 part7 (K≤28) 可做；原始 part1 只有 2 个 passed → K≤2**
   - 参与式（participation）：pool = 215–471，K=5/10/15/20 全部可抽，但 F4/F7 证明这条线 ≈ 随机
   - **阶段级 (student,part) 2171 行 / 258 正例 / part1–6 各 471/388/321/290/250/236 行、正例 83/67/31/40/15/22 → 每 part 都有 ≥15 正例，K-shot 完全可行（且 part7 零正例天然排除）**

---

## 附录 · 路径速查

```
本地仓库   /root/.openclaw/workspace-staging/pc-ceo_assistant/StudentRisk   (git main @ efd97e8)
  ├── docs/                          ← 已有 39 个 md（论文 v1–v10 + 报告）
  ├── docs/STAGE_TASK_REPORT_20261031.md        ← 阶段级结案报告（LOPO 主证据）
  ├── docs/paper_v8_generalization_zh.md        ← 少样本/LOPO 章节 §5.4-5.5（EXP-2/3/4/5/6 全表）
  ├── docs/NEXT_EXPERIMENTS_20261104.md         ← 9 条已证伪清单 + 3 个新结果
  ├── analysis/stage_experiment/stage_common.py ← ⭐ 复用骨架（fold/pooled/bootstrap/peer/self-ref）
  ├── generalization/data_splits.py             ← ⭐ lopo_split / temporal_split / learning_curve_split
  ├── models/meta_mamba_7d/eval_per_part_5shot.py  ← ⭐ 最完整少样本实现
  ├── run_4_gen_experiments.py                  ← 论文口径 4 实验
  ├── lopo_{train_eval,v2,v3_multiseed,v3_multiseed_nofilm}.py
  ├── per_student_lopo_v2.py, reproduce_5shot_samples.py
  └── results/{lopo,lopo_3seed,per_student_lopo,generalization_4runs,
               per_part_5shot_full,per_part_5shot_adaptive,per_part_3shot_A,per_part_kshot_B,
               cs2_diag,cs2_diag_lopo,cs2_diag_cross,cs2_diag_e2l,cs2_diag_l2e,
               cs2_diag_randdrop,cs2_diag_clean,stage_experiment,meta_mamba_7d_notc_fomaml}

246 服务器  43.139.55.246  (Tesla T4 15.4 GiB, torch 2.11.0+cu128)
  /home/ubuntu/studentrisk_stage/                 ← 工作副本
    ├── data/{stage_dataset,exercise_dataset,exercise_structure,deadline_feats,
    │         multiscale_feats,student_event_seqs,student_win_feats}.npz
    ├── results/{kshot_sweep,generalization_nofilm,few_shot_study,lopo_3seed_film,
    │             lopo_3seed_film_8d,lopo_3seed_no_film_8d,zero_shot_a,perturbation_study,
    │             film_off_study,film_uniqueness_study,adapter_baselines_7d,
    │             task_id_switching,dummy_task_id_lopo,sitting,exercise,ipsd,
    │             stage_experiment,cs2_fwd2,cs2_fwd3,cs2_fwd_clean}
    ├── run_kshot_sweep.py, run_nofilm_lopo_fomaml.py, exp_C_few_shot.py,
    │   exp_B_8d_lopo.py, exp_B_8d_nofilm_lopo.py, lopo_3seed_save_film.py
  /home/ubuntu/StudentRisk/                       ← 原始作者仓库（generalization_4runs 的输出位置）
  /home/ubuntu/IDE_logs/{IDE_logs.csv,passed.csv,README.txt}   ← 唯一真源
```
