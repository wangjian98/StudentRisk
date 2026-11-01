"""StudentRisk: FiLM 推理时 part_id 依赖性实验

目标: 验证 FiLM 模块在推理时**依赖 part_id** 还是**已经退化成常数变换**

实验设计:
  训练: 全部 7 parts × 473 students × 5-fold StratifiedKFold × 3 seed
       每个 fold 训练一个 MetaMamba(max_len=128, FiLM ON, TC weight=0.3)

  推理: 每折 val set 用 4 种 FiLM context 喂法:
    A. one_hot_true    — 喂学生自己的 part_id (baseline, 最理想)
    B. one_hot_wrong   — 喂错的 part_id (其他 6 个 part 的平均)
    C. mean_context    — 训练 6 part embedding 的平均
    D. random_uniform  — [0,1)^d 随机向量

  评估: val_acc / val_auc / val_f1 / val_macro_f1 (4 context 各一份)
       per-student 预测概率变化 (delta_prob between contexts)

  预期:
    - 若 FiLM 真依赖 task context: A 显著好于 B/C/D (Acc 差距 >= 5%)
    - 若 FiLM 是 dead code: A/B/C/D 差异极小 (Acc 差距 < 1%)

输出: results/film_inference_ablation/
"""
import os, sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix,
)

_HERE = '/home/ubuntu/StudentRisk'
sys.path.insert(0, _HERE)

from data import load_dataset
from models.base import set_seed
from models.meta_mamba.model import MetaMambaClassifier
from models.meta_mamba.data import build_event_sequences
from models.meta_mamba.train import train_one_fold

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEEDS = [42, 123, 777]


# ─────────────────── FiLM context override wrapper ───────────────────
class FiLMWithOverride(nn.Module):
    """跟 zero_shot_a.py 一样: 推理时让 FiLM.task_emb 被 raw tensor 取代"""
    def __init__(self, base: MetaMambaClassifier):
        super().__init__()
        self.base = base

    def forward(self, sequences, mask, task_ids, override=None):
        if override is None:
            return self.base(sequences, mask, task_ids)
        d = self.base.film.task_emb.embedding_dim
        B = sequences.shape[0]
        if override.dim() == 1:
            ovr = override.unsqueeze(0).expand(B, -1).to(sequences.device)
        else:
            ovr = override.to(sequences.device)

        def patched(x, task_ids_):
            t = ovr
            gamma = self.base.film.gamma_mlp(t).unsqueeze(1)
            beta  = self.base.film.beta_mlp(t).unsqueeze(1)
            return gamma * x + beta
        orig = self.base.film.forward
        self.base.film.forward = patched
        try:
            return self.base(sequences, mask, task_ids)
        finally:
            self.base.film.forward = orig


@torch.no_grad()
def predict_with_context(model_wrap, sequences, masks, task_ids, override=None, batch_size=32):
    model_wrap.eval()
    n = len(sequences)
    probs = np.zeros(n, dtype=np.float32)
    for i in range(0, n, batch_size):
        s = torch.from_numpy(sequences[i:i+batch_size]).float().to(DEVICE)
        m = torch.from_numpy(masks[i:i+batch_size]).float().to(DEVICE)
        t = torch.from_numpy(task_ids[i:i+batch_size]).long().to(DEVICE)
        if override is not None and override.dim() == 2:
            ovr_batch = override[i:i+batch_size]
        else:
            ovr_batch = override
        logit = model_wrap(s, m, t, override=ovr_batch).squeeze(-1).ravel()
        probs[i:i+batch_size] = torch.sigmoid(logit).cpu().numpy()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_len', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--out_root', default='/home/ubuntu/StudentRisk/results/film_inference_ablation')
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = ap.parse_args()

    out_dir = Path(args.out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('FiLM inference-time part_id dependence ablation')
    print(f'max_len={args.max_len}, epochs={args.epochs}, seeds={args.seeds}')
    print('=' * 70)

    # 加载数据
    ide, labels_df, y_all, student_ids = load_dataset()
    sequences, masks, task_ids, _ = build_event_sequences(ide, student_ids, max_len=args.max_len)
    print(f'sequences={sequences.shape} n_tasks={int(task_ids.max())+1}')

    # 5-fold split (跟 meta_mamba_8d 一致)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_splits = list(skf.split(np.zeros(len(y_all)), y_all))

    all_rows = []
    t_start = time.time()

    for seed in args.seeds:
        for fold_idx, (tr, va) in enumerate(fold_splits):
            print(f'\n--- seed={seed} fold={fold_idx+1}/5 ---')
            set_seed(seed)
            t0 = time.time()
            model = MetaMambaClassifier(
                d_event=sequences.shape[-1],
                d_model=64, d_state=16, n_layers=2,
                n_tasks=int(task_ids.max())+1, dropout=0.2,
            ).to(DEVICE)
            model, best_val = train_one_fold(
                model,
                sequences[tr], masks[tr], task_ids[tr], y_all[tr],
                sequences[va], masks[va], task_ids[va], y_all[va],
                epochs=args.epochs, lr=1e-3, weight_decay=1e-3,
                batch_size=16, patience=10,
                contrastive_weight=0.3, use_film=True, device=DEVICE,
            )
            print(f'  train done in {time.time()-t0:.0f}s, best_val_loss={best_val:.4f}')

            # 准备 4 种 context
            trained_emb = model.film.task_emb.weight.detach().cpu()
            trained_ids = sorted(set(range(int(task_ids.max())+1)))  # 全部 7 个 part 都训过
            mean_emb = trained_emb[trained_ids].mean(dim=0)
            rand_emb = torch.rand_like(mean_emb)
            # one_hot_wrong: 用其他 part 的平均 one-hot — 但 FiLM 吃 task_emb, not one-hot
            # 实际: 用除该 student 自己 part 外所有其他 part 的 task_emb 平均
            contexts = {}

            # A. one_hot_true (用学生自己的 task_id)
            contexts['one_hot_true'] = None  # 表示正常 forward

            # B. one_hot_wrong (用其他 part 的 task_emb 平均 - 跟 mean 一样但只算非自己 part)
            wrong_per_student = []
            for i in range(len(va)):
                sid = int(task_ids[va[i]])
                other_ids = [j for j in trained_ids if j != sid]
                wrong_per_student.append(trained_emb[other_ids].mean(dim=0))
            wrong_tensor = torch.stack(wrong_per_student)  # (n_va, d)
            contexts['one_hot_wrong'] = wrong_tensor

            # C. mean_context (全部 7 part 平均)
            contexts['mean_context'] = mean_emb

            # D. random_uniform
            contexts['random_uniform'] = rand_emb

            # 4 种 context 推理 + 评估
            wrap = FiLMWithOverride(model)
            val_seq = sequences[va]; val_msk = masks[va]; val_t = task_ids[va]; val_y = y_all[va]

            for ctx_name, ovr in contexts.items():
                if ovr is None:
                    # one_hot_true
                    probs = predict_with_context(wrap, val_seq, val_msk, val_t, override=None)
                else:
                    probs = predict_with_context(wrap, val_seq, val_msk, val_t, override=ovr)

                pred = (probs >= 0.5).astype(int)
                acc = float(accuracy_score(val_y, pred))
                f1 = float(f1_score(val_y, pred, average='macro', zero_division=0))
                try:
                    auc = float(roc_auc_score(val_y, probs))
                except Exception:
                    auc = float('nan')
                try:
                    pr_auc = float(average_precision_score(val_y, probs))
                except Exception:
                    pr_auc = float('nan')

                row = {
                    'seed': seed, 'fold': fold_idx + 1,
                    'context': ctx_name,
                    'n_val': len(val_y), 'val_pos_rate': float(val_y.mean()),
                    'val_acc': acc, 'val_f1_macro': f1,
                    'val_auc': auc, 'val_pr_auc': pr_auc,
                    'mean_prob_pos': float(probs[val_y == 1].mean()) if val_y.sum() else float('nan'),
                    'mean_prob_neg': float(probs[val_y == 0].mean()) if (val_y==0).sum() else float('nan'),
                    'separation': float(probs[val_y==1].mean() - probs[val_y==0].mean()) if val_y.sum() and (val_y==0).sum() else float('nan'),
                }
                all_rows.append(row)
                print(f'    ctx={ctx_name:18s} acc={acc:.4f} auc={auc:.4f} f1={f1:.4f}', flush=True)

            # Save model
            torch.save(model.state_dict(), out_dir / f'seed{seed}_fold{fold_idx+1}.pt')

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(out_dir / 'all_runs.csv', index=False, encoding='utf-8')

    # Aggregate per-context
    summary_rows = []
    for ctx_name, grp in df_all.groupby('context'):
        summary_rows.append({
            'context': ctx_name,
            'mean_acc': grp.val_acc.mean(),
            'std_acc': grp.val_acc.std(ddof=1),
            'mean_f1': grp.val_f1_macro.mean(),
            'std_f1': grp.val_f1_macro.std(ddof=1),
            'mean_auc': grp.val_auc.mean(),
            'std_auc': grp.val_auc.std(ddof=1),
            'mean_sep': grp.separation.mean(),
            'std_sep': grp.separation.std(ddof=1),
            'n_runs': len(grp),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values('mean_acc', ascending=False)
    summary_df.to_csv(out_dir / 'summary.csv', index=False, encoding='utf-8')

    # Markdown report
    md = []
    md.append("# FiLM 推理时 part_id 依赖性分析\n")
    md.append(f"## 实验设计\n")
    md.append(f"- 数据: 473 students × 7 parts (in-distribution, 全部 7 part 训过)")
    md.append(f"- 训练: 5-fold StratifiedKFold × {len(args.seeds)} seeds × {args.epochs} epochs × max_len={args.max_len}")
    md.append(f"- 推理: 每折 val set 上用 **4 种 FiLM context** 喂入, 评估 val_acc / AUC / F1\n")
    md.append(f"### 4 种 FiLM context\n")
    md.append(f"| context | 含义 |")
    md.append(f"|---|---|")
    md.append(f"| **one_hot_true** | 学生自己的 part_id (baseline, 最理想) |")
    md.append(f"| **one_hot_wrong** | 用其他 part 的 task_emb 平均 (模拟 part_id 错配) |")
    md.append(f"| **mean_context** | 全部 7 part embedding 平均 (无 part 区分) |")
    md.append(f"| **random_uniform** | [0,1)^d 随机向量 (noise) |\n")
    md.append(f"## 📊 评测结果 ({len(args.seeds)} seeds × 5 folds = {len(args.seeds)*5} runs)\n")
    md.append(f"| context | val_acc (mean±std) | val_f1_macro | val_auc | separation |")
    md.append(f"|---|---:|---:|---:|---:|")
    for _, row in summary_df.iterrows():
        md.append(f"| **{row['context']}** | **{row['mean_acc']:.4f}±{row['std_acc']:.4f}** | "
                  f"{row['mean_f1']:.4f}±{row['std_f1']:.4f} | "
                  f"{row['mean_auc']:.4f}±{row['std_auc']:.4f} | "
                  f"{row['mean_sep']:.4f}±{row['std_sep']:.4f} |")
    md.append("")

    # Cross-context deltas
    acc_by_ctx = df_all.groupby('context').val_acc.mean().to_dict()
    if 'one_hot_true' in acc_by_ctx:
        ref = acc_by_ctx['one_hot_true']
        md.append(f"### Context 敏感度 (Δ vs one_hot_true = {ref:.4f})\n")
        md.append(f"| context | Δ acc | 结论 |")
        md.append(f"|---|---:|---|")
        for ctx_name, acc in acc_by_ctx.items():
            if ctx_name == 'one_hot_true':
                continue
            delta = acc - ref
            if abs(delta) < 0.005:
                verdict = "✅ FiLM 不敏感 (dead code)"
            elif delta < -0.02:
                verdict = "❌ FiLM 严重依赖正确 part_id"
            else:
                verdict = "⚠️ FiLM 部分依赖"
            md.append(f"| {ctx_name} | {delta:+.4f} | {verdict} |")
        md.append("")

    md.append(f"## 🔍 分析\n")
    md.append(f"### 关键发现\n")
    md.append(f"详见 `full_results_dump.md` 和 `summary.csv`\n")
    md.append(f"## 📁 输出文件\n")
    md.append(f"- `all_runs.csv` — {len(df_all)} runs 完整指标")
    md.append(f"- `summary.csv` — 4 context 聚合")
    md.append(f"- `film_inference_ablation_report.md` — 本报告")
    md.append(f"- `seed{{}}_fold{{}}.pt` — {len(args.seeds) * 5} 个 trained models\n")

    md.append(f"## ⏱ 训练时间\n")
    md.append(f"- 总耗时: {time.time() - t_start:.0f}s ({(time.time() - t_start)/60:.1f} min)")

    (out_dir / 'film_inference_ablation_report.md').write_text('\n'.join(md), encoding='utf-8')

    print('\n' + '=' * 70)
    print('🎯 Summary (mean over seeds × folds)')
    print('=' * 70)
    print(summary_df.to_string(index=False))
    print(f'\nTotal time: {(time.time()-t_start)/60:.1f} min')


if __name__ == '__main__':
    main()