#!/bin/bash
# RegimeMamba 实验队列（串行，246 后台）
# 协议与 rev12 对齐；E1 走完整超参搜索，其余格固定 E1 选出的超参（节省算力，预算对消融公平）
set -u
cd /home/ubuntu/SR_rev12
LOG=logs/regimemamba.log
mkdir -p logs results/regimemamba

stamp () { date '+%Y-%m-%d %H:%M:%S'; }

run () {
  tag="$1"; shift
  echo "[queue] $(stamp) START $tag :: $*" >> $LOG
  python3 analysis/regimemamba/run_exp.py "$@" >> $LOG 2>&1
  echo "[queue] $(stamp) END   $tag (exit=$?)" >> $LOG
}

# ── E1 主格：身份臂 28d 全程，4 候选超参搜索 ──
run E1_main_d28_p1.0 --tag main_d28_p1.0 --dim 28 --prefix 1.0

HP=$(python3 - <<'PY'
import json
try:
    print(json.dumps(json.load(open('results/regimemamba/main_d28_p1.0/results.json'))['hp']))
except Exception as e:
    print('{"d_model":64,"d_state":16,"n_layers":2,"dropout":0.2,"lr":0.001}')
PY
)
echo "[queue] $(stamp) fixed HP for ablations/derived cells: $HP" >> $LOG

# ── E2 早期预警：前缀 10% ──
run E2_main_d28_p0.1 --tag main_d28_p0.1 --dim 28 --prefix 0.1 --hp_fixed "$HP"
# ── E3 前缀 20% ──
run E3_main_d28_p0.2 --tag main_d28_p0.2 --dim 28 --prefix 0.2 --hp_fixed "$HP"
# ── E4 去身份臂 ──
run E4_deconf_d28_p1.0 --tag deconf_d28_p1.0 --dim 28 --prefix 1.0 --deconfound --hp_fixed "$HP"
# ── E5 K=1 无切换对照 ──
run E5_abl_k1 --tag abl_k1_d28_p1.0 --dim 28 --prefix 1.0 --n_regime 1 --hp_fixed "$HP"
# ── E6 去掉 regime 直通特征 ──
run E6_abl_nodirect --tag abl_nodirect_d28_p1.0 --dim 28 --prefix 1.0 --no_direct --hp_fixed "$HP"
# ── E7 去掉防塌缩正则 λ_u=0 ──
run E7_abl_nousage --tag abl_nousage_d28_p1.0 --dim 28 --prefix 1.0 --lam_usage 0.0 --hp_fixed "$HP"

echo "[queue] $(stamp) ALL DONE" >> $LOG
