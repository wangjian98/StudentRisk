import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, f1_score
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 60)

pred = pd.read_csv('results/concat_dummy_task_id/per_student_predictions.csv')


def g(cond, feed):
    return pred[(pred.conditioning == cond) & (pred.task_id_used == feed)].sort_values('student_idx')


print("=" * 100)
print("G) film vs concat 是不是真的两个不同模型? (逐学生概率相关性)")
print("=" * 100)
for feed in ['true', 'zero(0)', 'mean', 'random']:
    a = g('film', feed)['pred_prob'].values
    b = g('concat', feed)['pred_prob'].values
    print(f"  feed={feed:8s} pearson r={np.corrcoef(a, b)[0,1]:.4f}  "
          f"max|a-b|={np.max(np.abs(a-b)):.4f}  mean|diff|={np.mean(np.abs(a-b)):.4f}")

print()
print("=" * 100)
print("H) 逐 part 拆解 (film)")
print("=" * 100)
for feed in ['true', 'zero(0)']:
    d = g('film', feed)
    rows = []
    for part, sub in d.groupby('true_part'):
        y = sub['true_label'].values
        p = sub['pred_prob'].values
        yp = (p >= 0.5).astype(int)
        rows.append(dict(part=part, n=len(sub), fail_rate=y.mean(),
                         acc=(yp == y).mean(),
                         f1m=f1_score(y, yp, average='macro', zero_division=0),
                         auc=roc_auc_score(y, p) if len(set(y)) > 1 else np.nan,
                         mean_p=p.mean()))
    print(f"\n--- feed={feed} ---")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print()
print("=" *100)
print("I) 两套协议交叉验证")
print("=" * 100)
fia = pd.read_csv('results/film_inference_ablation/summary.csv')
print("[协议1] film_inference_ablation (10-11): val-set 逐折, 15 runs, 有 std")
print(fia[['context', 'mean_acc', 'std_acc', 'mean_f1', 'std_f1', 'mean_auc', 'std_auc',
           'mean_sep', 'std_sep', 'n_runs']].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n[协议2] concat_dummy_task_id (10-28): per-student OOF 池化, 同模型同fold")
s = pd.read_csv('results/concat_dummy_task_id/summary.csv')
print(s[s.conditioning == 'film'][['task_id_used', 'accuracy', 'f1_macro', 'roc_auc',
                                    'mean_prob', 'std_prob']].to_string(index=False,
                                                                       float_format=lambda x: f"{x:.4f}"))

print()
print("=" * 100)
print("J) '喂错 task_id' vs '完全不喂 task_id' 哪个更糟? (旧协议 10-12)")
print("=" * 100)
old = pd.read_csv('results/dummy_task_id_lopo/summary.csv')
print(old[['state', 'task_id_used', 'accuracy', 'f1_macro', 'roc_auc']].to_string(
    index=False, float_format=lambda x: f"{x:.4f}"))
print()
print("解读:")
print("  film_disabled (γ=1,β=0, FiLM 通路关掉, 但模型是带 FiLM 训练的) -> AUC 0.8966, 仍强")
print("  film_on + 错 task_id                                     -> AUC 0.5630, 崩塌到接近随机")
print("  => 喂错 task_id 比完全不喂 task_id 更糟")
print("     FiLM 学到的不是'可选修正', 而是主导判别的必需信号")
print()
print("  per_part_breakdown (旧协议, 错 task_id 时每个 part 的表现):")
bp = pd.read_csv('results/dummy_task_id_lopo/per_part_breakdown.csv')
print(bp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
