#!/usr/bin/env python3
"""把 §6.8 分层/配对分析结论回填进 RegimeMamba 中英文论文。

一次性脚本（幂等：若已打过补丁则跳过）。改动点：
  · 新增 §6.8「行为活动量混淆的分层与配对分析」，判据总表改为 §6.9
  · §6.7 末尾加口径提示
  · §7.3 / §7.5 表述收窄（发现 3 / 发现 5）
  · 摘要「结果(v)」与「结论」补混淆检验结论
  · §1.4 贡献 3、§8.1 不主张项、§9 结论项 5 与一句话结论
"""
import io
import os
import sys

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs')
ZH = os.path.join(DOCS, 'REGIMEMAMBA_PAPER_20261108_zh.md')
EN = os.path.join(DOCS, 'REGIMEMAMBA_PAPER_20261108_en.md')

PATCHES = []

# ─────────────────────────── ZH ───────────────────────────
PATCHES.append((ZH, '### 6.8 判据总表', '### 6.9 判据总表'))

ZH_68 = '''### 6.8 行为活动量混淆的分层与配对分析（修补 §6.7 的解释漏洞）

§6.7 报告的两个观察——R2 占用与挂科相关（单独 AUC 0.6386）、切换率与挂科**反向**相关（AUC 0.3806）——都只给出**未控制混淆**的读数。本节用四类方法检验"行为活动量"这一混淆因子：偏 Spearman（秩基，bootstrap 2000 次学生级重采样）、活动量分层（层内 AUC + 分层固定效应回归）、Mantel–Haenszel 合并 OR、以及**活动量配对分析**（每个通过生配一个活动量最接近的挂科生，z 空间最近邻，配对平衡 SMD<0.01）。

**口径提示（本文新发现的一条口径纪律）**：模型的序列表征是每个学生**最后 128 个事件**（左 padding），因此切换率与 regime 占用都在该窗口上计算；而 §6.7 表中的行为统计（paste/run/submit 计数）是**全轨迹**。窗口事件数几乎恒为 128（CV=0.046），**不能**用作活动量代理；真正与挂科强相关的是**全轨迹活动量**（事件总数单独判别挂科 AUC=0.159，反向）。

**混淆路径量化**：

| 活动量口径 | ρ(V, y) | ρ(V, switch) | ρ(switch, y) | 乘积 ρ(V,switch)·ρ(V,y) | 残差 | 混淆路径解释占比 |
|---|---|---|---|---|---|---|
| 全轨迹活动量 PCA1（主口径） | −0.5266 | +0.2594 | −0.1954 | −0.1366 | −0.0588 | 70% |
| 全轨迹事件总数 | −0.5581 | +0.2173 | −0.1954 | −0.1213 | −0.0741 | 62% |
| 末段跨度 | −0.1181 | +0.2356 | −0.1954 | −0.0278 | −0.1676 | 14% |

**控制活动量后 switch 的残余效应**：

| 控制集 | 偏 ρ(switch, y) | 95% CI | P |
|---|---|---|---|
| 无（原始） | −0.1954 | — | — |
| 全轨迹活动量 | −0.1002 | [−0.1883, −0.0059] | 0.030 |
| 全轨迹活动量 + 末段跨度 | −0.0835 | [−0.1722, +0.0086] | 0.070（ns） |
| 活动量区块（3 协变量） | −0.0817 | [−0.1722, +0.0088] | 0.077（ns） |
| RF-28d OOF（含全部活动量聚合） | −0.0443 | [−0.1391, +0.0505] | 0.337（ns） |

对应地：switch 的 OR 从未控制的 0.664（P=4.5e-05）降到控制活动量后 0.833（P=0.131）、分层固定效应后 0.837（P=0.145）、在 RF 之上 0.759（P=0.079）；按全轨迹活动量三分位分层的 MH 合并 OR=0.76（P=0.289，**分层后不再显著**）。机制线索：switch 与末段**跨度/间隔**正相关（ρ=+0.236）、与末段**强度**负相关（ρ=−0.229）、与末段活动量负相关（ρ=−0.385）→ **switch 实际上在量"末段行为有多稀疏/多不连续"**。

**活动量配对分析**（每个 PASS 生配一个活动量最接近的 FAIL 生，159 对，配对平衡 SMD<0.01）：

| 匹配口径 | switch 配对差(FAIL−PASS) | P | SMD | R2 占用配对差 | P | SMD |
|---|---|---|---|---|---|---|
| 仅全轨迹活动量 | −0.0197 | 0.26 | −0.104 | +0.0077 | 0.91 | −0.016 |
| 全轨迹活动量 + 末段新鲜度 | −0.0576 | 0.022 | −0.194 | +0.0096 | 0.65 | +0.025 |
| 仅窗口层行为量（与 regime 同窗口） | −0.0424 | 0.003 | −0.257 | +0.0707 | 1.5e-07 | +0.449 |
| 全量（全轨迹 + 窗口层） | −0.0029 | 0.441 | −0.110 | +0.0128 | 0.36 | +0.032 |
| 全轨迹活动量 + RF 分数 | −0.0755 | 0.0015 | −0.241 | −0.0367 | 0.0006 | −0.314 |

**读法**：把**全轨迹活动量**配平后，挂科生与通过生的 **switch 差异不再显著**（配对差 −0.020，P=0.26）、**R2 占用差异完全消失**（配对差 +0.008，P=0.91）。仅按窗口层行为量配对时 R2 差异很大（配对差 +0.071，SMD +0.45），但那种配对**没有约束全轨迹投入量**，R2 仍在替全轨迹投入量说话——把全轨迹量也纳入后差异即消失，两条证据互相印证。

**对 §6.7 的修订结论**：R2 占用的原始 AUC 0.6386 与切换率的反向关联，**主要都是"行为投入量"的重新编码**，而非独立的风险信号。偏相关口径同向：ρ(R2, y)=+0.2268 → 控制活动量后 +0.1056（P=0.022，约 53% 的原始关联被解释）；分层固定效应回归中 R2 的 OR=1.11（P=0.473）；MH 合并 OR=1.16（P=0.616）。

> **本文据此收窄两项表述**：① §7.5「switching is not risk」改为「switching is **not an independent risk signal**」；② §7.3「R2 是可直接使用的风险状态指标」改为「R2 是**行为投入强度的可解释化再编码**，不是独立风险信号」。详见 §7.3 与 §7.5。

'''
PATCHES.append((ZH, '### 6.9 判据总表', ZH_68 + '### 6.9 判据总表'))

ZH_67_note = '''
> **口径提示**：本节的 switch 与 regime 占用均在**每个学生最后 128 个事件**的窗口上计算，而上表的行为统计为**全轨迹**；这一口径差异及其后果见 §6.8。
'''
PATCHES.append((ZH,
                '这与 §6.4 的去混淆结果相互印证：若 regime 主要编码身份，其与行为总量的关联不应如此规整。',
                '这与 §6.4 的去混淆结果相互印证：若 regime 主要编码身份，其与行为总量的关联不应如此规整。' + ZH_67_note))

PATCHES.append((ZH,
'''### 7.3 发现 3：R2 是一个可直接使用的风险状态指标

R2 占用单独判别挂科 AUC 0.6386（Mann-Whitney P<0.0001），且与 R1 方向相反、互补。相比黑盒风险分，"处于高风险行为状态的时间占比"是一个**教师与辅导员可直接理解、并可按周追踪**的指标。它同时提供了一个反例式的提醒：**行为状态的"可解释"并不意味着"高风险 regime = 高活动量"**——切换率与行为总量强相关而与效率指标无关，说明 regime 捕捉的是活动形态与节奏，不是用功强度。''',
'''### 7.3 发现 3：R2 是行为投入强度的"可解释化再编码"，而不是独立风险信号

R2 占用单独判别挂科 AUC 0.6386（Mann-Whitney P<0.0001），与 R1 方向相反、互补，形式上是一个**教师与辅导员可直接理解、并可按周追踪**的指标。但 §6.8 的分层与配对分析表明：**该关联约 53% 由行为活动量解释，并且在活动量配平后消失**（配对差 +0.008，P=0.91；控制活动量后的偏 ρ=+0.106，P=0.022；分层固定效应回归 OR=1.11，P=0.473；MH 合并 OR=1.16，P=0.616）。因此本文把该发现的含义**收窄**为：R2 更接近"行为投入强度/节奏"的**可解释化再编码**，而不是独立于投入量的风险状态指标。

它同时提供了一个反例式的提醒：**行为状态的"可解释"并不意味着"高风险 regime = 高活动量"**——切换率与行为总量强相关而与效率指标无关，说明 regime 捕捉的是活动形态与节奏，不是用功强度。

**产品含义**：若要保留"R2 风险状态时长"这一产品形态，必须与投入强度**并列展示**，文案只能是"投入强度与节奏"，不能声称它比"做了多少"多提供了信息；在活动量配平后挂科与通过学生并无差异，因此**不能**把它作为独立的早期预警指标使用。'''))

PATCHES.append((ZH,
'''### 7.5 发现 5：Switching 不等于风险

切换率单独判别挂科 AUC 0.3806（反向）。直觉上"状态频繁切换 = 状态不稳 = 高风险"，但数据不支持：切换多的学生反而更可能通过。原因可能是**活跃度混淆**（切换率与行为总量 ρ≈+0.25，行为多的人本来就更容易切换），也可能说明真正的风险信号是**长期停留在某个状态**而非切换频繁。这一点应在后续以活动量分层（partial correlation / 分层回归）进一步检验。''',
'''### 7.5 发现 5：切换率不是独立风险信号（原文漏洞已修补）

切换率单独判别挂科 AUC 0.3806（反向）。直觉上"状态频繁切换 = 状态不稳 = 高风险"，但数据不支持：切换多的学生反而更可能通过。§6.8 的分层与配对分析把原文并列的两种解释**区分**开了：**活动量混淆成立，且几乎解释了全部效应**。

- 混淆路径：ρ(switch, 全轨迹活动量)=+0.2594，ρ(全轨迹活动量, 挂科)=−0.5266，两者乘积已覆盖原始关联的 70%；
- 控制全轨迹活动量后，偏 ρ 从 −0.1954 降到 −0.1002（P=0.030），再加控制末段跨度即不显著（P=0.070），在 RF-28d OOF 之上完全不显著（P=0.337，−0.044）；
- 按活动量三分位分层的 MH 合并 OR=0.76（P=0.289，分层后不再显著）；
- 活动量配对后（159 对，SMD<0.01），挂科生与通过生的 switch 差异不再显著（配对差 −0.020，P=0.26）。

机制上，switch 与末段**跨度/间隔**正相关（ρ=+0.236）、与末段**强度**负相关（ρ=−0.229）、与末段活动量负相关（ρ=−0.385）→ **switch 实际在量"末段行为有多稀疏/多不连续"**，而不是某种独立的"状态不稳定度"。

**因此本文的表述从"switching is not risk"改为"switching is not an independent risk signal"**：数据既不支持"切换少 = 危险"这条可交付规则，也不支持"切换与风险存在独立反向关联"；switch 是行为活动量/末段稀疏度的代理，不应作为独立风险指标使用。'''))

PATCHES.append((ZH,
                '说明 regime 切换刻画的是活动量而非效率。',
                '说明 regime 切换刻画的是活动量而非效率。**§6.8 的分层与配对分析进一步表明：R2 占用与切换率的上述关联主要都是"行为投入量"的重新编码——在活动量配平后，R2 在挂科生与通过生之间已无差异（配对差 +0.008，P=0.91），切换率的反向关联也不再显著（配对差 −0.020，P=0.26）**。'))

PATCHES.append((ZH,
                '诚实的后果是：RegimeMamba 的**可解释性**（R2 占用可直接作为"风险状态时长"指标）比它的**准确率**更有价值；本文主张该模型用于**全周期风险分层与状态解释**，而非早期预警或精度竞赛。',
                '诚实的后果是：RegimeMamba 的**可解释性**（把行为状态变成可读指标）比它的**准确率**更有价值，但这一可解释性经得起混淆检验的部分有限——**R2 占用在活动量配平后不再区分挂科与通过**（§6.8）；本文因此把该模型定位为**全周期风险分层与状态描述**，而非早期预警、精度竞赛或独立的"风险状态时长"产品。'))

PATCHES.append((ZH,
                '3. **经验发现**：regime 结构相对 K=1 的增益不显著（H6 判负），且部分依赖身份、增益部分依赖身份（去混淆臂）、且**regime 具有可解释的风险语义**（R2 占用单独 AUC 0.6386）。',
                '3. **经验发现**：regime 结构相对 K=1 的增益不显著（H6 判负），增益部分依赖身份（去混淆臂），**regime 具有可解释的风险语义**（R2 占用单独 AUC 0.6386）——但 §6.8 表明该语义**约一半由行为活动量解释**。'))

PATCHES.append((ZH,
'''5. **不主张 regime 语义已被充分解释**：R2/R1 的行为含义目前只是统计对应，尚未做人工编码的语义核验。''',
'''5. **不主张 regime 语义已被充分解释**：R2/R1 的行为含义目前只是统计对应，尚未做人工编码的语义核验。
6. **不主张 regime 统计量是独立风险信号**：控制行为投入量后，R2 占用在挂科生与通过生之间已无差异（配对差 +0.008，P=0.91）；切换率同样不携带残余信息（配对后 P=0.26，在 RF 基线之上 ns）。'''))

PATCHES.append((ZH,
                '5. **可解释性是它的真正价值**：R2 是一个与挂科显著相关、单独 AUC 0.6386 的行为状态指标，可直接产品化为"风险状态时长"。',
                '5. **可解释性是它最可能的真实价值，但要打折**：R2 占用单独 AUC 0.6386，但 §6.8 显示在活动量配平后其挂科/通过差异消失 → 它更接近"行为投入强度的可解释化再编码"，可作为描述性指标，不足以支撑独立的"风险状态时长"产品。'))

PATCHES.append((ZH,
                '一句话：**RegimeMamba 的价值在于把"行为状态"变成可测量、可解释、可产品化的对象，而不是把 AUC 抬高一个点。**',
                '一句话：**RegimeMamba 的价值在于把"行为状态"变成可测量、可解释的对象，而不是把 AUC 抬高一个点；但本轮证据显示，这个对象的信息量与行为投入量高度重合，必须与投入量并列解读。**'))

PATCHES.append((ZH,
                '| H4 regime 真实存在 | 占用≥5% / \\|ρ\\|≥0.3 / 至少一 regime 与挂科显著相关 | 最小占用 0.240；切换率-行为总量 ρ 最高 +0.281（P<0.0001，**未达 0.30 阈值**）；R2 占用与挂科显著（P<0.0001） | ⚠️ 部分通过 |',
                '| H4 regime 真实存在 | 占用≥5% / \\|ρ\\|≥0.3 / 至少一 regime 与挂科显著相关 | 最小占用 0.240；切换率-行为总量 ρ 最高 +0.281（P<0.0001，**未达 0.30 阈值**）；R2 占用与挂科显著（P<0.0001，**原始未控制**；§6.8：约一半由活动量解释） | ⚠️ 部分通过 |'))

# ─────────────────────────── EN ───────────────────────────
PATCHES.append((EN, '### 6.8 Summary of pre-registered criteria', '### 6.9 Summary of pre-registered criteria'))

EN_68 = '''### 6.8 Stratified and matched analysis of behavioural-volume confounding (closing the §6.7 loophole)

The two observations of §6.7 — R2 occupancy associated with failure (standalone AUC 0.6386) and switch rate associated with failure **in reverse** (AUC 0.3806) — were read without any confound control. This section tests the "behavioural volume" confound with four instruments: rank-based partial Spearman (bootstrap, 2000 student resamples), volume stratification (within-stratum AUC + stratified fixed-effects regression), Mantel–Haenszel pooled OR, and a **volume-matched pair design** (each passing student matched to the failing student with the closest behavioural volume in z-space; matched standardised mean differences < 0.01).

**Caliber note (a caliber discipline surfaced by this analysis).** The model's sequence representation is each student's **last 128 events** (left padding), so the switch rate and the regime occupancies are computed on that window, whereas the behavioural statistics of §6.7 (paste/run/submit counts) are **whole-trajectory**. The window event count is almost constant at 128 (CV = 0.046) and cannot serve as a volume proxy; the quantity that actually predicts failure is **whole-trajectory volume** (event count alone discriminates failure at AUC 0.159, reversed).

**Size of the confounding path**:

| Volume caliber | ρ(V, y) | ρ(V, switch) | ρ(switch, y) | product ρ(V,switch)·ρ(V,y) | residual | share explained by the confound path |
|---|---|---|---|---|---|---|
| whole-trajectory volume PCA1 (primary) | −0.5266 | +0.2594 | −0.1954 | −0.1366 | −0.0588 | 70% |
| whole-trajectory event count | −0.5581 | +0.2173 | −0.1954 | −0.1213 | −0.0741 | 62% |
| end-of-window span | −0.1181 | +0.2356 | −0.1954 | −0.0278 | −0.1676 | 14% |

**Residual effect of switch after controlling for volume**:

| Control set | partial ρ(switch, y) | 95% CI | P |
|---|---|---|---|
| none (raw) | −0.1954 | — | — |
| whole-trajectory volume | −0.1002 | [−0.1883, −0.0059] | 0.030 |
| volume + end-window span | −0.0835 | [−0.1722, +0.0086] | 0.070 (ns) |
| volume block (3 covariates) | −0.0817 | [−0.1722, +0.0088] | 0.077 (ns) |
| RF-28d OOF (contains all activity aggregates) | −0.0443 | [−0.1391, +0.0505] | 0.337 (ns) |

Correspondingly, switch's OR falls from 0.664 unadjusted (P = 4.5e-05) to 0.833 with volume controlled (P = 0.131), 0.837 with stratified fixed effects (P = 0.145) and 0.759 on top of RF (P = 0.079); the Mantel–Haenszel pooled OR after stratifying by whole-trajectory volume tertiles is 0.76 (P = 0.289, **no longer significant after stratification**). Mechanistically, switch correlates positively with the final window's **span / inter-event gaps** (ρ = +0.236) and negatively with its **intensity** (ρ = −0.229) and with window activity (ρ = −0.385) — **switch is effectively measuring how sparse/discontinuous the end-of-term behaviour is**.

**Volume-matched pair analysis** (each passing student matched to the closest-volume failing student; 159 pairs; matched SMD < 0.01):

| Matching set | switch paired diff (FAIL−PASS) | P | SMD | R2 occupancy paired diff | P | SMD |
|---|---|---|---|---|---|---|
| whole-trajectory volume only | −0.0197 | 0.26 | −0.104 | +0.0077 | 0.91 | −0.016 |
| volume + end-window recency | −0.0576 | 0.022 | −0.194 | +0.0096 | 0.65 | +0.025 |
| window-level behaviour only (same window as the regimes) | −0.0424 | 0.003 | −0.257 | +0.0707 | 1.5e-07 | +0.449 |
| all (volume + window level) | −0.0029 | 0.441 | −0.110 | +0.0128 | 0.36 | +0.032 |
| whole-trajectory volume + RF score | −0.0755 | 0.0015 | −0.241 | −0.0367 | 0.0006 | −0.314 |

**Reading.** Once **whole-trajectory volume** is balanced, the failing/passing difference in switch is no longer significant (paired diff −0.020, P = 0.26) and the difference in **R2 occupancy disappears entirely** (paired diff +0.008, P = 0.91). Matching on window-level behaviour alone leaves a large R2 difference (+0.071, SMD +0.45), but that design does **not** constrain whole-trajectory engagement, so R2 is still speaking for volume; once whole-trajectory volume enters the matching, the difference vanishes — the two lines of evidence corroborate each other.

**Consequence (a revision of §6.7).** R2 occupancy's raw AUC 0.6386 and the reversed switch–failure association are **largely re-encodings of behavioural engagement volume**, not independent risk signals. The partial-correlation caliber agrees: ρ(R2, y) = +0.2268 → +0.1056 after controlling volume (P = 0.022; about 53% of the raw association removed); stratified fixed-effects regression gives OR = 1.11 (P = 0.473); MH pooled OR = 1.16 (P = 0.616).

> **Two claims are therefore narrowed**: (i) §7.5 "switching is not risk" becomes "switching is **not an independent risk signal**"; (ii) §7.3 "R2 is a directly usable risk-state indicator" becomes "R2 is an **interpretable re-encoding of behavioural engagement**, not an independent risk signal". See §7.3 and §7.5.

'''
PATCHES.append((EN, '### 6.9 Summary of pre-registered criteria', EN_68 + '### 6.9 Summary of pre-registered criteria'))

PATCHES.append((EN,
                'Switching tracks **behavioural volume**, not efficiency — consistent with §6.4: had regimes mainly encoded identity, this volume-aligned association would not be so regular.',
                '''Switching tracks **behavioural volume**, not efficiency — consistent with §6.4: had regimes mainly encoded identity, this volume-aligned association would not be so regular.

> **Caliber note.** The switch rate and regime occupancies in this section are computed on each student's **last 128 events**, whereas the behavioural statistics above are **whole-trajectory**; see §6.8 for the consequences of this caliber mismatch.'''))

PATCHES.append((EN,
'''### 7.3 Finding 3: R2 is a directly usable risk-state indicator

R2 occupancy discriminates failure at AUC 0.6386 on its own (Mann-Whitney P<0.0001) and is complementary (opposite in sign) to R1. Unlike a black-box risk score, "fraction of time in a high-risk behavioural state" is understandable to teachers and counsellors and can be tracked weekly. It also delivers a cautionary note: **interpretable regimes do not mean "high-risk regime = high activity"** — switch rate correlates with behavioural volume but not with efficiency, so regimes capture activity shape and rhythm rather than effort intensity.''',
'''### 7.3 Finding 3: R2 is an interpretable re-encoding of behavioural engagement, not an independent risk signal

R2 occupancy discriminates failure at AUC 0.6386 on its own (Mann-Whitney P<0.0001), is complementary (opposite in sign) to R1, and is formally an indicator that teachers and counsellors can understand and track weekly. But the stratified and matched analysis of §6.8 shows that **about 53% of this association is explained by behavioural volume and it disappears once volume is balanced** (paired diff +0.008, P = 0.91; partial ρ = +0.106 after volume control, P = 0.022; stratified fixed-effects OR = 1.11, P = 0.473; MH pooled OR = 1.16, P = 0.616). We therefore **narrow** the claim: R2 is better read as an **interpretable re-encoding of behavioural engagement intensity and rhythm** than as a risk-state indicator independent of engagement.

It also delivers a cautionary note: **interpretable regimes do not mean "high-risk regime = high activity"** — switch rate correlates with behavioural volume but not with efficiency, so regimes capture activity shape and rhythm rather than effort intensity.

**Product implication.** If the "duration in a risky state" indicator is kept in any product form, it must be shown **next to** engagement intensity, its copy may only claim "engagement intensity and rhythm", and it must not be sold as an early-warning signal in its own right: at matched engagement level, failing and passing students do not differ in R2 occupancy.'''))

PATCHES.append((EN,
'''### 7.5 Finding 5: switching is not risk

Switch rate discriminates failure at AUC 0.3806 (reversed). Intuitively "frequent state switching = instability = risk", but the data disagree: students who switch more are more likely to pass. The likely cause is **activity-volume confounding** (switch rate correlates with behavioural volume at ρ≈+0.25, and busier students switch more), or the possibility that the true risk signal is **dwelling** in a state rather than switching often. This deserves follow-up with volume-stratified analyses (partial correlation, stratified regression).''',
'''### 7.5 Finding 5: the switch rate is not an independent risk signal (the original loophole is now closed)

Switch rate discriminates failure at AUC 0.3806 (reversed). Intuitively "frequent state switching = instability = risk", but the data disagree: students who switch more are more likely to pass. The stratified and matched analysis of §6.8 **separates** the two explanations that §6.7 left side by side: **the activity-volume confound is real and accounts for essentially the whole effect.**

- Confound path: ρ(switch, whole-trajectory volume) = +0.2594 and ρ(whole-trajectory volume, failure) = −0.5266; their product already covers 70% of the raw association.
- Controlling whole-trajectory volume reduces the partial ρ from −0.1954 to −0.1002 (P = 0.030); adding the end-window span makes it non-significant (P = 0.070); on top of the RF-28d OOF score it vanishes entirely (P = 0.337, −0.044).
- Mantel–Haenszel pooled OR after stratifying by volume tertiles: 0.76 (P = 0.289 — no longer significant after stratification).
- After volume matching (159 pairs, SMD < 0.01) the failing/passing difference in switch is no longer significant (paired diff −0.020, P = 0.26).

Mechanistically, switch correlates positively with the final window's **span / inter-event gaps** (ρ = +0.236) and negatively with its **intensity** (ρ = −0.229) and window activity (ρ = −0.385): **switch is effectively measuring how sparse/discontinuous the end-of-term behaviour is**, rather than an independent "state instability".

**The claim is therefore restated from "switching is not risk" to "switching is not an independent risk signal"**: the data support neither a deliverable rule ("fewer switches = danger") nor an independent reversed association; switch is a proxy for behavioural volume / end-of-term sparsity and must not be used as a risk indicator in its own right.'''))

PATCHES.append((EN,
                'regime switching tracks activity shape, not efficiency.',
                'regime switching tracks activity shape, not efficiency. **The stratified and matched analysis of §6.8 further shows that both associations are largely re-encodings of behavioural engagement: after matching engagement level, R2 occupancy no longer separates failing from passing students (paired diff +0.008, P = 0.91) and the reversed switch association is no longer significant (paired diff −0.020, P = 0.26).**'))

PATCHES.append((EN,
                "The honest consequence is that RegimeMamba's **interpretability** (R2 occupancy as a directly displayable \"duration in a risky state\" indicator) is worth more than its accuracy. We position the model for **whole-cycle risk stratification and state explanation**, not early warning and not an accuracy race.",
                "The honest consequence is that RegimeMamba's **interpretability** — turning behavioural state into a readable quantity — is worth more than its accuracy, but only part of that interpretability survives the confound check: **at matched engagement level, R2 occupancy no longer separates failing from passing students** (§6.8). We position the model for **whole-cycle risk stratification and state description**, not early warning, not an accuracy race, and not as a stand-alone \"duration in a risky state\" product."))

PATCHES.append((EN,
                '3. **Empirical findings.** the gain of regime structure over K=1 is not significant (H6 fails), and it partly relies on identity, part of the gain depends on identity, and the regimes carry **interpretable risk semantics** (R2 occupancy alone: AUC 0.6386).',
                '3. **Empirical findings.** the gain of regime structure over K=1 is not significant (H6 fails), it partly relies on identity, and the regimes carry **interpretable risk semantics** (R2 occupancy alone: AUC 0.6386) — but §6.8 shows that roughly half of that association is explained by behavioural engagement volume.'))

PATCHES.append((EN,
'''5. **No claim that regime semantics are fully explained.** R2/R1 currently correspond to behaviour only statistically; no manual coding validation has been performed.''',
'''5. **No claim that regime semantics are fully explained.** R2/R1 currently correspond to behaviour only statistically; no manual coding validation has been performed.
6. **No claim that regime statistics are independent risk signals.** Once behavioural engagement volume is controlled, R2 occupancy no longer separates failing from passing students (matched paired diff +0.008, P = 0.91), and the switch rate carries no residual information either (P = 0.26 after matching; ns on top of the RF baseline).'''))

PATCHES.append((EN,
                '5. **Interpretability is its real value.** R2 is a behaviour state significantly associated with failure (standalone AUC 0.6386) and can be productised as "duration in a risky state".',
                '5. **Interpretability is its most plausible real value, at a discount.** R2 occupancy reaches 0.6386 standalone, but §6.8 shows the failing/passing difference disappears once engagement is matched: it is better read as an **interpretable re-encoding of behavioural engagement** — usable descriptively, but not sufficient to support a stand-alone "duration in a risky state" product.'))

PATCHES.append((EN,
                'In one sentence: **RegimeMamba\'s value is that it turns "behavioural state" into a measurable, interpretable, productisable object — not that it raises AUC by a point.**',
                'In one sentence: **RegimeMamba\'s value is that it turns "behavioural state" into a measurable, interpretable object — not that it raises AUC by a point; but this round of evidence shows that object\'s information content overlaps heavily with behavioural engagement volume and must be read next to it.**'))

PATCHES.append((EN,
                '| H4 regimes are real | occupancy≥5% / \\|ρ\\|≥0.3 / one regime associated with failure | min occupancy 0.240; max ρ +0.281 (P<0.0001, **below the 0.30 threshold**); R2 occupancy associated with failure (P<0.0001) | ⚠️ partial pass |',
                '| H4 regimes are real | occupancy≥5% / \\|ρ\\|≥0.3 / one regime associated with failure | min occupancy 0.240; max ρ +0.281 (P<0.0001, **below the 0.30 threshold**); R2 occupancy associated with failure (P<0.0001, **uncontrolled**; §6.8: about half explained by activity volume) | ⚠️ partial pass |'))


def main():
    applied = 0
    skipped = 0
    for path, old, new in PATCHES:
        with io.open(path, encoding='utf-8') as f:
            s = f.read()
        if old not in s:
            # 幂等：若 new 已存在则视为已打过
            if new in s:
                skipped += 1
                continue
            print(f'[FAIL] 未找到锚点: {os.path.basename(path)} :: {old[:60]!r}')
            sys.exit(1)
        cnt = s.count(old)
        if cnt != 1:
            print(f'[FAIL] 锚点出现 {cnt} 次: {os.path.basename(path)} :: {old[:60]!r}')
            sys.exit(1)
        s = s.replace(old, new)
        with io.open(path, 'w', encoding='utf-8') as f:
            f.write(s)
        applied += 1
    print(f'applied={applied} skipped={skipped}')


if __name__ == '__main__':
    main()
