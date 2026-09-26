#!/usr/bin/env bash
# Meta-Mamba-11d 消融实验串行调度（6 个 run, 复用 full）
# 预计总时长 ~50 分钟
set -e
cd /home/ubuntu/StudentRisk
mkdir -p logs

LOG=logs/meta_mamba_ablation_$(date +%Y%m%d_%H%M%S).log
echo "[meta_mamba_ablation] start at $(date)" | tee -a $LOG

# 已存在的 full (FiLM=True, λ=0.3) 用 results/meta_mamba/results.json 当 baseline
echo "[meta_mamba_ablation] full run already exists at results/meta_mamba/results.json — skipping" | tee -a $LOG

# 6 个 ablation run
VARIANTS=(
  "no_tc:True:0.0"
  "no_film:False:0.3"
  "no_film_no_tc:False:0.0"
  "lam_0_1:True:0.1"
  "lam_0_5:True:0.5"
  "lam_1_0:True:1.0"
)

for spec in "${VARIANTS[@]}"; do
  IFS=':' read -r variant use_film tcw <<< "$spec"
  echo "" | tee -a $LOG
  echo "[meta_mamba_ablation] >>> variant=$variant use_film=$use_film tcw=$tcw start=$(date +%H:%M:%S)" | tee -a $LOG
  python3 run_meta_mamba_ablation.py --variant "$variant" --no-fewshot 2>&1 | tee -a $LOG
  echo "[meta_mamba_ablation] <<< variant=$variant done=$(date +%H:%M:%S)" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "[meta_mamba_ablation] all done at $(date)" | tee -a $LOG