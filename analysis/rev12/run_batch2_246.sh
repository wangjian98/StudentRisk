#!/bin/bash
# rev12 第二批（精简版）实验启动器 —— 246 / Tesla T4
cd /home/ubuntu/SR_rev12
mkdir -p logs
export PYTHONPATH=/home/ubuntu/SR_rev12
export SR12_CACHE=/home/ubuntu/SR12_cache
export OMP_NUM_THREADS=2

run () {
  local name=$1; shift
  nohup python3 -m "$@" > logs/${name}.log 2>&1 &
  echo "launched $name pid=$!"
  sleep 2
}

# 早期预警曲线（精简：4 个前缀点；MetaMamba 只跑 28d）
run exp2  analysis.rev12.exp2_prefix --prefixes 0.1,0.2,0.4,1.0 --cells metamamba_d28,rf_d28,lstm_d28,hmm_gmm_d28 --tag exp2_prefix
run exp2d analysis.rev12.exp2_prefix --prefixes 0.1,0.4,1.0 --cells metamamba_d28,rf_d28 --tag exp2_prefix_deconf --deconfound

# FiLM 多 seed（7d/11d 5 seed —— 意见要求"5 个以上种子"）
run exp3  analysis.rev12.exp3_film --dims 7,11 --seeds 42,123,777,2024,7 --tag exp3_film

# LOPO 跨任务迁移
run exp4  analysis.rev12.exp4_lopo --cells rf_d28,metamamba_d28,metamamba_d28_nofilm --tag exp4_lopo

echo "BATCH2 LAUNCHED"
sleep 3
ps aux | grep -c "[a]nalysis.rev12"
