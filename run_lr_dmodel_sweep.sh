#!/usr/bin/env bash
# MetaMamba-11d lr x d_model grid sweep (3x3 = 9 runs)
# Protocol: 5-fold StratifiedKFold x 3 seeds (42, 123, 777)
# Default config: FiLM=ON, TC=0.3 (paper defaults), max_len=256
# Skips FOMAML (--no-fewshot) to speed up; ablation-only metrics matter for grid.
# Estimated runtime: ~30min (d=32) + ~65min (d=64) + ~130min (d=128) per lr, 3 lrs total = ~10-11h
set -e
cd /home/ubuntu/StudentRisk
mkdir -p logs

LOG=logs/lr_dmodel_sweep_$(date +%Y%m%d_%H%M%S).log
echo "[lr_dmodel_sweep] start at $(date)" | tee -a $LOG

LRS=(0.0003 0.001 0.003)
DMS=(32 64 128)

for lr in "${LRS[@]}"; do
  for dm in "${DMS[@]}"; do
    LR_TAG=$(echo $lr | tr '.' '_')
    OUT_DIR=/home/ubuntu/StudentRisk/results/sweep_lr${LR_TAG}_d${dm}
    echo "" | tee -a $LOG
    echo "[lr_dmodel_sweep] >>> lr=$lr d_model=$dm start=$(date +%H:%M:%S)" | tee -a $LOG
    if [ -f "$OUT_DIR/results.json" ]; then
      echo "[lr_dmodel_sweep]    SKIP — already done" | tee -a $LOG
      continue
    fi
    python3 models/meta_mamba/train.py \
        --lr "$lr" --d-model "$dm" \
        --no-fewshot \
        --out-dir "$OUT_DIR" 2>&1 | tee -a $LOG
    echo "[lr_dmodel_sweep] <<< lr=$lr d_model=$dm done=$(date +%H:%M:%S)" | tee -a $LOG
  done
done

echo "" | tee -a $LOG
echo "[lr_dmodel_sweep] all 9 grid runs done at $(date)" | tee -a $LOG
echo DONE_SWEEP > /home/ubuntu/StudentRisk/.sweep_done
