#!/usr/bin/env bash
set -e
cd /home/ubuntu/StudentRisk
mkdir -p logs results

LOG=/home/ubuntu/StudentRisk/logs/maxlen_sweep_20260925_210337.log
echo "[maxlen_sweep] start at $(date)" | tee -a $LOG
echo "[maxlen_sweep] plan: 11d x {256,512,1024} + 7d x {256,512,1024} = 6 configs" | tee -a $LOG
echo "[maxlen_sweep] reuse existing 128 baselines from results/meta_mamba and results/meta_mamba_7d" | tee -a $LOG

# ===== MetaMamba-11d =====
for ML in 256 512 1024; do
  OUT=/home/ubuntu/StudentRisk/results/meta_mamba_ml$ML
  echo "" | tee -a $LOG
  echo "[maxlen_sweep] >>> 11d max_len=$ML start=$(date +%H:%M:%S)" | tee -a $LOG
  python3 run_meta_mamba_ablation.py --variant full --no-fewshot --max-len $ML --out-dir $OUT 2>&1 | tee -a $LOG
  echo "[maxlen_sweep] <<< 11d max_len=$ML done=$(date +%H:%M:%S)" | tee -a $LOG
done

# ===== MetaMamba-7d =====
for ML in 256 512 1024; do
  OUT=/home/ubuntu/StudentRisk/results/meta_mamba_7d_ml$ML
  echo "" | tee -a $LOG
  echo "[maxlen_sweep] >>> 7d max_len=$ML start=$(date +%H:%M:%S)" | tee -a $LOG
  python3 models/meta_mamba_7d/train.py --max-len $ML --no-fewshot --out-dir $OUT 2>&1 | tee -a $LOG
  echo "[maxlen_sweep] <<< 7d max_len=$ML done=$(date +%H:%M:%S)" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "[maxlen_sweep] ALL DONE at $(date)" | tee -a $LOG
echo DONE_MAXLEN_SWEEP > /home/ubuntu/StudentRisk/.maxlen_sweep_done
