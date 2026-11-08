#!/bin/bash
# RegimeMamba 论文收尾：拉回结果 → 回填论文 → 生成 docx → git 提交推送
set -u
BASE=/root/.openclaw/workspace-staging/pc-ceo_assistant/StudentRisk
LOG=$BASE/.regimemamba_finalize.log
REMOTE=/home/ubuntu/SR_rev12

echo "==== $(date '+%F %T') finalize start ====" >> $LOG

# 1) 拉回 246 上的结果与分析产物（结果 / 论文 / 报告 / CSV）
rsync235_pull -az ubuntu@43.139.55.246:$REMOTE/results/regimemamba/ "$BASE/results/regimemamba/" >> $LOG 2>&1
rsync235_pull -az ubuntu@43.139.55.246:$REMOTE/docs/REGIMEMAMBA_PAPER_20261108_zh.md "$BASE/docs/" >> $LOG 2>&1
rsync235_pull -az ubuntu@43.139.55.246:$REMOTE/docs/REGIMEMAMBA_PAPER_20261108_en.md "$BASE/docs/" >> $LOG 2>&1
rsync235_pull -az ubuntu@43.139.55.246:$REMOTE/docs/REGIMEMAMBA_RESULTS_20261108.md "$BASE/docs/" >> $LOG 2>&1
rsync235_pull -az ubuntu@43.139.55.246:$REMOTE/docs/REGIMEMAMBA_DATA_20261108.csv "$BASE/docs/" >> $LOG 2>&1
rsync235_pull -az ubuntu@43.139.55.246:$REMOTE/docs/REGIMEMAMBA_REGIME_STATS_20261108.csv "$BASE/docs/" >> $LOG 2>&1
rsync235_pull -az ubuntu@43.139.55.246:$REMOTE/docs/REGIMEMAMBA_ROUTING_20261108.csv "$BASE/docs/" >> $LOG 2>&1
rsync235_pull -az ubuntu@43.139.55.246:$REMOTE/logs/regimemamba_k1.log "$BASE/results/regimemamba/queue_k1.log" >> $LOG 2>&1
rsync235_pull -az ubuntu@43.139.55.246:$REMOTE/logs/regimemamba_lopo.log "$BASE/results/regimemamba/queue_lopo.log" >> $LOG 2>&1

# 2) docx 交付件
cd $BASE
python3 analysis/hazard_model/md2docx.py docs/REGIMEMAMBA_PAPER_20261108_zh.md docs/REGIMEMAMBA_PAPER_20261108_zh.docx >> $LOG 2>&1
python3 analysis/hazard_model/md2docx.py docs/REGIMEMAMBA_PAPER_20261108_en.md docs/REGIMEMAMBA_PAPER_20261108_en.docx >> $LOG 2>&1
python3 analysis/hazard_model/md2docx.py docs/REGIMEMAMBA_RESULTS_20261108.md docs/REGIMEMAMBA_RESULTS_20261108.docx >> $LOG 2>&1
python3 analysis/hazard_model/md2docx.py docs/REGIMEMAMBA_DESIGN_20261108.md docs/REGIMEMAMBA_DESIGN_20261108.docx >> $LOG 2>&1

# 3) 占位符检查（不应有残留）
LEFT=$(grep -c '{{' docs/REGIMEMAMBA_PAPER_20261108_zh.md docs/REGIMEMAMBA_PAPER_20261108_en.md | tr '\n' ' ')
echo "placeholder_left: $LEFT" >> $LOG

# 4) git 提交推送
git add -A analysis/regimemamba analysis/fusion docs results/regimemamba results/rev12 2>>$LOG
git -c user.name="wangjian98" -c user.email="wangjian98@example.com" commit -m "docs(regimemamba): RegimeMamba 论文中英文定稿 + 实验报告 + 数据表（含 K=1 除零缺陷修复后的重跑、LOPO 跨任务迁移、regime 语义分析）" >> $LOG 2>&1
COMMIT=$(git rev-parse --short HEAD)
git push origin main >> $LOG 2>&1
PUSH_RC=$?
echo "commit=$COMMIT push_rc=$PUSH_RC" >> $LOG
echo "$COMMIT" > $BASE/.regimemamba_finalized
echo "==== $(date '+%F %T') finalize done commit=$COMMIT push_rc=$PUSH_RC ====" >> $LOG
