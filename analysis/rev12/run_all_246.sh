#!/bin/bash
# rev12 全量实验启动器（246 / Tesla T4）
cd /home/ubuntu/SR_rev12
mkdir -p logs
export PYTHONPATH=/home/ubuntu/SR_rev12
export SR12_CACHE=/home/ubuntu/SR12_cache
export OMP_NUM_THREADS=2

run () {  # $1=name $2...=args
  local name=$1; shift
  nohup python3 -m "$@" > logs/${name}.log 2>&1 &
  echo "launched $name pid=$!"
  sleep 2
}

# 核心三件（矩阵 / 前缀曲线 / FiLM 多 seed）
run exp1  analysis.rev12.exp1_matrix --archs rf7_all,rf,lstm,bilstm,attention,metamamba,hmm_cat,hmm_gmm --spaces 7,11,28 --tag exp1_matrix
run exp2  analysis.rev12.exp2_prefix --prefixes 0.1,0.2,0.4,0.6,1.0 --cells metamamba_d11,metamamba_d28,rf_d28,lstm_d28,attention_d28,hmm_gmm_d28 --tag exp2_prefix
run exp3  analysis.rev12.exp3_film --dims 7,11,28 --seeds 42,123,777,2024,7 --tag exp3_film

# 去混淆臂 + LOPO（稍后启动，避免瞬时过载）
sleep 45
run exp1d analysis.rev12.exp1_matrix --archs rf7_all,rf,lstm,bilstm,attention,metamamba,hmm_cat,hmm_gmm --spaces 7,11,28 --tag exp1_matrix_deconf --deconfound
run exp2d analysis.rev12.exp2_prefix --prefixes 0.1,0.2,0.4,0.6,1.0 --cells metamamba_d28,rf_d28,lstm_d28 --tag exp2_prefix_deconf --deconfound
run exp4  analysis.rev12.exp4_lopo --cells rf_d28,metamamba_d7,metamamba_d11,metamamba_d28,metamamba_d28_nofilm --tag exp4_lopo

echo "ALL LAUNCHED"
sleep 5
ps aux | grep -c "[a]nalysis.rev12"
