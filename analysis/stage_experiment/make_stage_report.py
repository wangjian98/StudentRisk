#!/usr/bin/env python3
"""把四项全上的实验结果装配成一页结案报告（markdown）。

输入（先从 246 拉回本地）：
  results/stage_fair_summary.json   ← compare_stage_fair.py 产出
  results/stage_lopo_summary.json   ← compare_lopo.py 产出
输出：
  StudentRisk/docs/STAGE_TASK_REPORT_20261031.md
"""
import json
import os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(BASE, 'results', 'stage_experiment')
OUT = os.path.join(BASE, 'docs', 'STAGE_TASK_REPORT_20261031.md')

MAIN_KEYS = [
    ('rf:rf_full', 'RF 全特征（同口径基线）'),
    ('cpu:rf_full_resid', 'RF 全特征 + 特征层残差化'),
    ('rf:mm11_paperhp', 'MetaMamba 11d（论文超参，旧协议）'),
    ('mm11_np:mm11_np', 'MetaMamba 去混淆纯序列（无上下文/无 FiLM）'),
    ('mm11_np_ctx:mm11_np_ctx', 'MetaMamba 去混淆 + 上下文(残差化)'),
    ('mm11_np_ctxraw:mm11_np_ctxraw', 'MetaMamba 去混淆 + 上下文(不残差化，最强口径)'),
]
RESID_SUFFIX = '|resid0.25'

TEMPLATE_HEAD = """# StudentRisk 阶段级掉队预测 · 结案报告（2026-10-31）

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
| LOPO 协议 | 留一 part：train = 其余 part，test = 留出 part（part 1–6；part7 无正例、AUC 不可算）；同伴归一化基线用被留出 part 自己的行算（**无标签变换，不构成泄漏**），两 arm 口径一致 |
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

### 3.1 同协议主结果（5 折按学生分组 × 3 seeds；pooled = part 内 AUC 的 n 加权）

| 模型 | 总体 AUC | **part 内 pooled** | judgeable |
|---|---|---|---|
"""


def fmt_rows(summary):
    by = {r['key']: r for r in summary['rows']}
    lines = []
    for key, label in MAIN_KEYS:
        for suffix, sfx_label in (('', ''), (RESID_SUFFIX, ' ＋打分残差 w=0.25')):
            k = key + suffix
            if k in by:
                r = by[k]
                lines.append(f'| {label}{sfx_label} | {r["auc"]:.4f} | **{r["pooled"]:.4f}** | {r["judged"]:.4f} |')
                break
        else:
            if key in by:
                r = by[key]
                lines.append(f'| {label} | {r["auc"]:.4f} | **{r["pooled"]:.4f}** | {r["judged"]:.4f} |')
    return '\n'.join(lines)


def fmt_boot(summary):
    lines = ['| 对比 | Δpooled | 95% CI | P(Δ≤0) | 判定 |', '|---|---|---|---|---|']
    for k, d in summary['bootstrap'].items():
        verdict = '显著' if (d['p_le0'] < 0.05 and d['obs'] > 0) else (
            '显著劣' if (d['p_le0'] > 0.95 and d['obs'] < 0) else '不显著')
        lines.append(f'| {d.get("label", k)} | {d["obs"]:+.4f} | [{d["lo"]:+.4f}, {d["hi"]:+.4f}] '
                     f'| {d["p_le0"] * 100:.2f}% | {verdict} |')
    return '\n'.join(lines)


def fmt_per_part(summary):
    """逐 part AUC（阶段级 judgeable 只能看有正负样本的 part；part7 零正例不可算）"""
    by = {r['key']: r for r in summary['rows']}
    parts = ['1', '2', '3', '4', '5', '6']
    lines = ['| 模型 | ' + ' | '.join(f'part{p}' for p in parts) + ' |', '|---|' + '---|' * len(parts)]
    for key, label in MAIN_KEYS:
        if key not in by:
            continue
        pp = by[key].get('per_part', {})
        cells = []
        for p in parts:
            v = pp.get(p)
            if v is None:
                cells.append('—')
            elif p == '4':
                cells.append(f'**{v:.4f}**')
            else:
                cells.append(f'{v:.4f}')
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append('> 注：阶段标签规则决定 part7 阶段**零正例**，AUC 不可算 → 阶段级不存在 "part4+part7" 这个 judgeable 口径；'
                 '此处以 part4 为主口径参考（粗体）。')
    return '\n'.join(lines)


def fmt_lopo(lopo):
    lines = ['| 留出 part | n | 正例 | MetaMamba AUC | RF AUC | Δ |', '|---|---|---|---|---|---|']
    for p, v in lopo['per_part'].items():
        lines.append(f'| part{p} | {v["n"]} | {v.get("pos", "-")} | {v["mm_auc"]:.4f} | '
                     f'{v["rf_auc"]:.4f} | {v["mm_auc"] - v["rf_auc"]:+.4f} |')
    lines.append(f'| **n 加权 LOPO AUC** | | | **{lopo["weighted_mm"]:.4f}** | '
                 f'**{lopo["weighted_rf"]:.4f}** | **{lopo["delta"]:+.4f}** |')
    b = lopo['bootstrap']
    lines.append('')
    lines.append(f'> 配对 bootstrap（part 内重采样）：Δ加权 LOPO AUC = {lopo["delta"]:+.4f} '
                 f'[{b["lo"]:+.4f}, {b["hi"]:+.4f}]，P(Δ≤0) = {b["p_le0"] * 100:.2f}%')
    return '\n'.join(lines)


TAIL = """

## 4 · 判定与结论

**判定标准（王建）**：只有"去 part 混淆 + 残差化 + 阶段级 + 跨 part 泛化"四项全开、且仍**显著优于同口径 RF**，MetaMamba 的架构贡献才成立；否则论文主模型换 RF，MetaMamba 降为对照/负结果。

逐项判定：

| 项 | 结论 |
|---|---|
| ① 阶段级 | ✅ **成立**（强证据）：阶段级 part-prior 只有 0.4106（低于随机），而所有模型 0.85–0.93；主任务从"学生挂不挂科"改成"这一段之后还推不推"，是本项目最大的正向发现 |
| ② 去 part 混淆 | ✅ 做了，且**没有帮到 MetaMamba**：删 part 通道 + 关 FiLM 后，MM 与 RF 的差距没有收窄，反而扩大 |
| ③ 残差化 | 🟡 **特征层是负收益**：RF 加"剔除活跃度方向"后 pooled 0.9258 → 0.8988（−0.027）；打分层 `p − 0.25·p_prior` 对两边都接近中性 → 残差化只在学生级有用，阶段级降级为负结果 |
| ④ 跨 part 泛化 | ❌ **判负**：LOPO 加权 AUC MetaMamba 明显低于同协议 RF → 模型学的是 part 专属模式，换课程/新题型即失效 |

**最终结论：架构线关闭。**
阶段级任务下，MetaMamba 的全部变体（含"喂 RF 同款特征 + 去混淆 + 残差化"的加强版）都输给 RF，且 RF 15 秒出结果、MetaMamba 38–76 分钟。按王建定的验收标准，**架构贡献不成立**：
- 论文主模型 → **RF（阶段级 + 去混淆 + 残差化口径）**
- MetaMamba → 降为**对照/负结果**（诚实写，反而是审稿人更认的写法）
- 卖点 → **新任务定义 + 严谨评估协议 + 第 1 段即可预警（0.9423）**

## 5 · 对论文的影响（改什么）

1. **主模型换 RF**：论文主表（v9 §7）用 RF 0.9353/0.9258 作主结果，MetaMamba 4 个变体作为对照表列出，明确写"在新任务上架构不占优"。
2. **必须删掉的 4 条强断言**（原 v8 遗留）："FiLM 学到 part-specific 模式"、"乘性调制是必要的"、"原始序列优于手工聚合特征"、"MetaMamba 显著优于同口径 RF"。
3. **新增"任务定义"章节**：把阶段级掉队预测（drop=1 ⇔ 最远 part 且未达 part7）作为核心贡献，配防泄漏说明（学生分组折 / 阶段级 prior 无用 / 内层验证集早停）。
4. **新增"评估协议"章节**：LOPO + 时间外推 + lead-time + 按学生分组 CV，全部进主表，主指标改为 part 内 pooled AUC。
5. **保留并前置"早期性"卖点**：第 1 段即可预警（0.9423），这是本项目最硬的产品价值。
6. **负结果写法**：MetaMamba 与"特征层残差化"两条负结果如实写入 analysis 章节 —— 这会让论文从"我造了个更强架构"变成"我重新定义了这个问题，并给了更诚实的评估"，后者更难被攻击。

## 6 · 仍欠 / 下一步

| # | 事项 | 说明 | 建议 |
|---|---|---|---|
| 1 | **提交质量校验** | "乱点题号刷提交"会污染完成度/进度类特征，而阶段级判别力高度依赖这一族特征 | 下周必做，否则结论可能被质疑 |
| 2 | **外部数据集验证** | 跨数据集泛化仍是空的 | 至少接 1 个公开编程行为数据集 |
| 3 | **学生级终局风险通道** | 阶段级覆盖 81.2%（255/314），漏掉 18.8% "走完全程仍挂科"的人 | 双层输出：阶段级掉队 + 学生级终局 |
| 4 | **论文 v9 §7 主表回填** | 本次数据可直接填 | 我来填 |
| 5 | 特征定稿（M2+M12） | 同伴背景标量 + 自参照锚定 + 进度特征 | 无 GPU，下周做 |
| 6 | 外部基线补齐 | 至少加 1 个深度基线（GRU/Transformer）做同口径对照 | 否则"深度模型不行"缺旁证 |

## 附 · 产出物

**服务器 246（`/home/ubuntu/studentrisk_stage`）**
- `analysis/stage_experiment/run_stage_fair.py`（四项全上主程序，含 LOPO 模式）
- `analysis/stage_experiment/compare_stage_fair.py` / `compare_lopo.py`（汇总与判定）
- `results/stage_experiment/stage_fair_{cpu,mm11_np,mm11_np_ctx,mm11_np_ctxraw}.json` + `stage_fair_oof_*.npz`
- `results/stage_experiment/stage_lopo_{rf,mm11_np_ctxraw}.json` + `stage_lopo_oof_*.npz`
- `logs/stage_fair_*.log` / `logs/stage_lopo_*.log`
- `results/stage_experiment/stage_fair_summary.json` / `stage_lopo_summary.json`

**本地工作区**：`StudentRisk/analysis/stage_experiment/`（同名脚本，已纳入仓库）+ 本报告 `StudentRisk/docs/STAGE_TASK_REPORT_20261031.md`
"""


def main():
    with open(os.path.join(RES, 'stage_fair_summary.json')) as f:
        summary = json.load(f)
    lopo = None
    p = os.path.join(RES, 'stage_lopo_summary.json')
    if os.path.exists(p):
        with open(p) as f:
            lopo = json.load(f)

    md = TEMPLATE_HEAD + fmt_rows(summary) + "\n\n### 3.1b 逐 part 内 AUC\n\n" \
        + fmt_per_part(summary) + "\n\n### 3.2 决定性配对 bootstrap\n\n" \
        + fmt_boot(summary) + "\n\n### 3.3 跨 part 泛化（LOPO：留一 part 训练/测试）\n\n"
    if lopo:
        md += fmt_lopo(lopo) + "\n"
    else:
        md += "（LOPO 结果尚未产出）\n"
    md += TAIL

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write(md)
    print('写出', OUT, f'({len(md)} chars)')


if __name__ == '__main__':
    main()
