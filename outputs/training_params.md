# StudentRisk — 全部模型的训练参数汇总

**项目**：StudentRisk（CS1 公开数据集，n=473，挂科率 fail_rate=0.6638）
**服务器**：`43.139.55.246`
**日期**：2026-08-22
**来源**：
- `configs/default.yaml`（yaml 中显式声明的参数）
- `models/<name>/train.py` 中 `mcfg.get(key, default)` 的硬编码默认值
- `models/meta_mamba/model.py` 中各模块（`S6Block` / `TaskFiLM` / `MetaMambaClassifier`）的内置参数
- `results/<name>/config_used.json`（实际跑本次实验时记录的运行时配置快照）

> ⚠️ **重要**：所有 `*_cfg` 字段在 `config_used.json` 中均为空 `{}`——这说明实际运行时**没有**读取 yaml 中的非通用配置（yaml 里也只配了 `rf7/lstm_7d/bilstm_7d/attention_7d` 四个段，没有 `meta_mamba`/`meta_mamba_7d` 段）。下表中的实际生效值全部来自 `train.py` 中 `mcfg.get(..., default)` 的硬编码默认。

---

## 0. 通用 CV 协议（全部模型共享）

| 参数 | 值 | 来源 |
|---|---|---|
| 交叉验证协议 | StratifiedKFold | `models/<name>/train.py` |
| **n_splits** | **8**（实际跑） / 5（yaml 默认） | `results/*/config_used.json` / `configs/default.yaml` |
| **seeds** | **[42, 123, 777]** | `configs/default.yaml` |
| **threshold** | **0.5** | `configs/default.yaml` |
| Threshold 搜索网格 | [0.30, 0.35, 0.39, 0.45, 0.50, 0.55] | `configs/default.yaml`（未实际使用） |
| **max_len**（事件序列长度） | **256** for MetaMamba；**128** for MetaMamba-7d / LSTM-7d / BiLSTM-7d / Attention-7d | `results/*/config_used.json` |
| **n_tasks**（problem part 数） | **7** | `results/*/config_used.json` |
| 数据集划分失败/通过 | Failed=1（正类），Passed=0 | `data/data_loader.py` |

**数据路径**
- IDE 日志：`/home/ubuntu/IDE_logs/IDE_logs.csv`（28,588,309 行，7 类事件）
- 通过标签：`/home/ubuntu/IDE_logs/passed.csv`（473 学生，passed=True=159，passed=False=314）

**事件类型 7-dim**：`text_insert`, `text_remove`, `text_paste`, `focus_gained`, `focus_lost`, `run`, `submit`

---

## 1. MetaMamba（d_event=11，主模型）

**架构参数**（`models/meta_mamba/model.py`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `d_event` | **11** | 7 维事件 one-hot + 4 维连续特征（time interval、deadline 等） |
| `d_model` | **64** | Event Embedding 输出维度 |
| `d_state` | **16** | S6 选择性 SSM 状态维度 |
| `n_layers` | **2** | MambaBlock 层数 |
| `n_tasks` | **7** | problem part 类别数 |
| `dropout` | **0.2** | Event Embedding + 各模块中的 dropout |

**S6Block 内部参数**（自实现的 Selective SSM，无 mamba-ssm 依赖）

| 参数 | 值 |
|---|---|
| `d_conv` | **4**（1D 因果卷积核大小） |
| `dt_rank` | `max(d_inner // 16, 1)` = **4**（d_inner=64 时） |
| `A_log` 初始化 | `Uniform(1, 16)` 后取负（保证 A 为负对角） |
| `D` 初始化 | `ones(d_inner)`（skip 参数） |

**TaskFiLM 参数**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `task_emb_dim` | **16** | problem part 嵌入维度 |
| `γ` (gamma) 激活 | **Sigmoid** → γ ∈ [0, 1] | 通道级缩放 |
| `β` (beta) | 无激活 | 通道级偏移 |

**优化器与训练循环**

| 参数 | 默认值 | 来源 |
|---|---|---|
| `epochs` | **40** | `models/meta_mamba/train.py` |
| `lr` | **1e-3** | `models/meta_mamba/train.py` |
| `weight_decay` | **1e-3** | `models/meta_mamba/train.py` |
| `batch_size` | **16** | `models/meta_mamba/train.py` |
| `patience` | **10**（早停） | `models/meta_mamba/train.py` |
| `contrastive_weight` | **0.3**（task-contrastive 损失权重） | `models/meta_mamba/train.py` |
| 优化器 | AdamW | `models/meta_mamba/train.py` |
| 梯度裁剪 | — | （未启用） |

**Task-Contrastive 损失（NT-Xent 风格）**

| 参数 | 默认值 |
|---|---|
| `temperature` (τ) | **0.1** |

**FOMAML 少样本评估**

| 参数 | 默认值 |
|---|---|
| `n_way` | **4**（4 个 task 同时评估，取 ≤ min(n_tasks, 4)） |
| `k_shot` | **5**（support 集样本数） |
| `n_query` | **10**（query 集样本数） |
| `inner_lr` | **0.01** |
| `inner_steps` | **3** |
| `seeds` | (42, 123, 777) |

**输入维度**：`d_event = 11`，`max_len = 256`
**参数量**：22,065
**训练耗时**（8 折 × 3 seed）：5,013 s ≈ 83.6 min

---

## 2. MetaMamba-7d（d_event=7，轻量版）

除以下差异外，与 MetaMamba 完全一致：

| 参数 | MetaMamba-7d | 与 MetaMamba 的差异 |
|---|---|---|
| `d_event` | **7** | 仅事件类型 one-hot，无连续特征 |
| `max_len` | **128** | 更短序列 |
| 其它全部架构 / 训练 / FOMAML 参数 | 同 MetaMamba | — |
| 参数量 | 21,809 | 略少于 MetaMamba |
| 训练耗时 | 2,205 s ≈ 36.8 min | 远少于 MetaMamba |

新增 CLI 选项（已合入 `train.py`，commit 3bab298）：
- `--no-film`：关闭 FiLM 调制
- `--no-tc`：关闭 Task-Contrastive 损失（`contrastive_weight` 强制为 0.0）

---

## 3. LSTM-7d

**架构参数**

| 参数 | 默认值 |
|---|---|
| `hidden_dim` (= `d_model`) | **64** |
| `num_layers` | **1** |
| `dropout` | **0.3** |
| 输入维度 `d_event` | **7** |
| `max_len` | **128** |

**优化器与训练循环**

| 参数 | 默认值 |
|---|---|
| `epochs` | **40** |
| `lr` | **1e-3** |
| `weight_decay` | **1e-3** |
| `batch_size` | **32** |
| `patience` | **10**（早停） |

**yaml 中的对应段**（注意 yaml 配的是 `epochs=60, patience=12`）

| yaml 值 | train.py 默认值 | **实际生效** |
|---|---|---|
| epochs: 60 | epochs: 40 | **40** |
| patience: 12 | patience: 10 | **10** |

> ⚠️ `results/lstm_7d/config_used.json` 中 `lstm_cfg: {}` 为空，所以**实际生效的是 train.py 的 hardcode 默认值（40/10），不是 yaml 的 60/12**。yaml 段可能是更早期版本的遗留。

**参数量**：33,857
**训练耗时**：29.6 s

---

## 4. BiLSTM-7d

与 LSTM-7d 完全一致的参数（同样的架构默认值、同样的训练超参）。差异仅在模型内部使用双向 LSTM。

| 参数 | 默认值 |
|---|---|
| `hidden_dim` | **64** |
| `num_layers` | **1** |
| `dropout` | **0.3** |
| 训练超参 | 同 LSTM-7d |

**参数量**：67,201（双向所以 ~2× LSTM-7d）
**训练耗时**：35.6 s

yaml 段同样存在 `epochs=60, patience=12` 与 train.py hardcode（`epochs=40, patience=10`）的差异，实际生效的是 train.py 默认值。

---

## 5. Attention-7d（Transformer 编码器）

**架构参数**

| 参数 | 默认值 |
|---|---|
| `d_model` | **64** |
| `n_heads` | **4** |
| `n_layers` | **2**（TransformerEncoder 层数） |
| `dim_feedforward` | **128** |
| `dropout` | **0.3** |
| 输入维度 `d_event` | **7** |
| `max_len` | **128** |

**优化器与训练循环**：同 LSTM-7d（`epochs=40, lr=1e-3, weight_decay=1e-3, batch_size=32, patience=10`）

yaml 段同样有 `epochs=60, patience=12` 与 train.py hardcode 的差异。

**参数量**：67,713
**训练耗时**：56.6 s

---

## 6. RF-7d（Random Forest — 当前是 stub 实现）

| 参数 | 值 |
|---|---|
| `n_estimators` | **200** |
| `max_depth` | **10** |
| `min_samples_split` | **5** |
| `class_weight` | **balanced** |

> ⚠️ **当前未实际运行**：`results/rf7/` 目录已从最新 commit 中删除（commit `358ef62`），因为 `models/rf7/train.py` 中的实现是占位桩（`_StubRFModel`），main.py 在 2026-08-21/22 跑时报错：`_StubRFModel() takes no arguments`。RF-7d 从对比表中排除。如需重新纳入，需要完成 `models/rf7/train.py` 的真实实现。

---

## 7. 模型间差异一览（横向对比表）

| | MetaMamba | MetaMamba-7d | LSTM-7d | BiLSTM-7d | Attention-7d | RF-7d |
|---|---|---|---|---|---|---|
| **输入维度 d_event** | 11 | 7 | 7 | 7 | 7 | 7（特征工程后） |
| **max_len** | 256 | 128 | 128 | 128 | 128 | n/a |
| **架构** | S6+FiLM+TC | S6+FiLM+TC | LSTM | BiLSTM | Transformer | Random Forest |
| **hidden_dim / d_model** | 64 | 64 | 64 | 64 | 64 | n/a |
| **num_layers / n_layers** | 2 | 2 | 1 | 1 | 2 | n/a |
| **dropout** | 0.2 | 0.2 | 0.3 | 0.3 | 0.3 | n/a |
| **epochs** | 40 | 40 | 40 | 40 | 40 | n/a |
| **lr** | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | n/a |
| **weight_decay** | 1e-3 | 1e-3 | 1e-3 | 1e-3 | 1e-3 | n/a |
| **batch_size** | 16 | 16 | 32 | 32 | 32 | n/a |
| **patience（早停）** | 10 | 10 | 10 | 10 | 10 | n/a |
| **contrastive_weight** | 0.3 | 0.3 | n/a | n/a | n/a | n/a |
| **TC temperature τ** | 0.1 | 0.1 | n/a | n/a | n/a | n/a |
| **FiLM task_emb_dim** | 16 | 16 | n/a | n/a | n/a | n/a |
| **FOMAML（few-shot eval）** | yes | yes | no | no | no | no |
| **n_heads (Attention)** | n/a | n/a | n/a | n/a | 4 | n/a |
| **dim_feedforward** | n/a | n/a | n/a | n/a | 128 | n/a |
| **n_estimators** | n/a | n/a | n/a | n/a | n/a | 200 |
| **max_depth** | n/a | n/a | n/a | n/a | n/a | 10 |
| **min_samples_split** | n/a | n/a | n/a | n/a | n/a | 5 |
| **class_weight** | n/a | n/a | n/a | n/a | n/a | balanced |
| **参数量** | 22,065 | 21,809 | 33,857 | 67,201 | 67,713 | n/a |
| **训练耗时（8 折 × 3 seed）** | 5,013 s | 2,205 s | 29.6 s | 35.6 s | 56.6 s | 未跑 |
| **Macro-F1（OOF）** | 0.8787 | 0.8783 | 0.3982 | 0.4822 | 0.5624 | n/a |

---

## 8. yaml vs train.py hardcode 不一致说明

`configs/default.yaml` 中存在一些与 `train.py` 内 hardcode 默认值**不一致**的字段。`results/*/config_used.json` 中所有 `*_cfg` 字段均为空 `{}`，说明**实际生效的是 train.py 的 hardcode 默认**：

| 字段 | yaml 值 | train.py hardcode | 实际生效 |
|---|---|---|---|
| `lstm_7d.epochs` | 60 | 40 | **40** |
| `lstm_7d.patience` | 12 | 10 | **10** |
| `bilstm_7d.epochs` | 60 | 40 | **40** |
| `bilstm_7d.patience` | 12 | 10 | **10** |
| `attention_7d.epochs` | 60 | 40 | **40** |
| `attention_7d.patience` | 12 | 10 | **10** |
| `meta_mamba.*` | 整段缺失 | 全部在 train.py | **train.py 默认** |
| `meta_mamba_7d.*` | 整段缺失 | 全部在 train.py | **train.py 默认** |

> 这是上游的一个轻微一致性 bug。yaml 中的 `epochs=60, patience=12` 应该是历史版本遗留——当前 run 的实际生效值是 `epochs=40, patience=10`。如果将来要在 yaml 中真正控制这些超参，需要在 `train.py` 中加 fallback：`mcfg = config.get('xxx', {})` 中允许 yaml 段缺失时合并默认。

---

## 9. MetaMamba 系列独有的可选开关（已合入 main train.py，commit 3bab298）

| CLI / 函数参数 | 默认 | 作用 |
|---|---|---|
| `--no-film` / `use_film=False` | use_film=True | 关闭 FiLM 调制（`x_film = x`，绕过 TaskFiLM） |
| `--no-tc` / `use_tc=False` | use_tc=True | 关闭 Task-Contrastive 损失（`contrastive_weight` 强制 0.0） |

这两个开关是为后续消融研究准备的：可直接在嵌套 CV 里加 `use_film=False` 或 `use_tc=False` 跑出 FiLM / TC 各自对最终 Macro-F1 的贡献。

---

## 10. 文件位置速查

| 文件 | 内容 |
|---|---|
| `configs/default.yaml` | yaml 配置（rf7 + 4 个 7-dim 模型段；缺 meta_mamba/meta_mamba_7d 段） |
| `models/<name>/train.py` | 每个模型的训练主循环，内含 `mcfg.get(key, default)` 的 hardcode 默认值 |
| `models/meta_mamba/model.py` | S6Block + TaskFiLM + MetaMambaClassifier 的内置参数 |
| `results/<name>/config_used.json` | 实际跑本次实验时的运行时配置快照（含 n_splits, seeds, threshold, max_len, n_tasks, n_params） |
| `results/<name>/results.json` | 含完整 per-fold 指标 + overall metrics + few-shot FOMAML |
| `results/<name>/fold_metrics.csv` | 每 (seed, fold) 的指标明细 |

---

*报告生成于 2026-08-22；所有数值均实地读取自 246 服务器当前 HEAD (`f99ecdf`) 下的源文件与 `results/` 快照。*