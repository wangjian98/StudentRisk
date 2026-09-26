#!/usr/bin/env bash
# Meta-Mamba-11d 消融实验 — 仅跑剩余未完成的 5 个变体（跳过 full + 已完成的 no_tc）
set -e
cd /home/ubuntu/StudentRisk
mkdir -p logs

LOG=logs/meta_mamba_ablation_remaining_$(date +%Y%m%d_%H%M%S).log
echo "[meta_mamba_ablation_remaining] start at $(date)" | tee -a $LOG

# 跳过 full 和 no_tc（已完成）
echo "[meta_mamba_ablation_remaining] skipping full (results/meta_mamba) and no_tc (results/meta_mamba_ablation_no_tc) — already done" | tee -a $LOG

VARIANTS=(
  "no_film:False:0.3"
  "no_film_no_tc:False:0.0"
  "lam_0_1:True:0.1"
  "lam_0_5:True:0.5"
  "lam_1_0:True:1.0"
)

for spec in "${VARIANTS[@]}"; do
  IFS=':' read -r variant use_film tcw <<< "$spec"
  OUT_DIR=/home/ubuntu/StudentRisk/results/meta_mamba_ablation_${variant}
  echo "" | tee -a $LOG
  echo "[meta_mamba_ablation_remaining] >>> variant=$variant use_film=$use_film tcw=$tcw start=$(date +%H:%M:%S)" | tee -a $LOG
  python3 run_meta_mamba_ablation.py --variant "$variant" --no-fewshot --out-dir "$OUT_DIR" 2>&1 | tee -a $LOG
  echo "[meta_mamba_ablation_remaining] <<< variant=$variant done=$(date +%H:%M:%S)" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "[meta_mamba_ablation_remaining] all 5 remaining variants done at $(date)" | tee -a $LOG
echo DONE_REMAINING > /home/ubuntu/StudentRisk/.ablation_remaining_done
