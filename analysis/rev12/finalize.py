#!/usr/bin/env python3
"""
rev12.finalize — 无人值守收尾：等待队列完成 → 补跑去混淆臂 → 显著性 → 汇总表 → 论文更新

用法：
  python3 -m analysis.rev12.finalize --wait-queue --send-ready
  （--send-ready 只写 docs/REVISION_SUMMARY.md，不发消息；消息由上层 agent 读文件后发送）
"""
import os
import sys
import json
import time
import glob
import shutil
import argparse
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
RES = os.path.join(_ROOT, 'results', 'rev12')
LOGS = os.path.join(_ROOT, 'logs')
PY = sys.executable

ENV = dict(os.environ)
ENV['PYTHONPATH'] = _ROOT
ENV['SR12_CACHE'] = '/home/ubuntu/SR12_cache'
ENV['OMP_NUM_THREADS'] = '2'

# 这些格必须在"task 身份置零"修复后重跑（早期跑的是带 part 条件的 FiLM）
STALE = [
    'results/rev12/exp1_matrix_deconf/metamamba_d7',
    'results/rev12/exp1_matrix_deconf/metamamba_d11',
    'results/rev12/exp1_matrix_deconf/metamamba_d28',
]


def sh(cmd, **kw):
    print(f'[finalize] $ {cmd}', flush=True)
    return subprocess.run(cmd, shell=True, cwd=_ROOT, env=ENV, **kw)


def queue_running():
    out = subprocess.run("ps -eo args | grep '[a]nalysis.rev12' | grep -v run_queue",
                         shell=True, capture_output=True, text=True).stdout.strip()
    return len(out.splitlines()) > 0


def wait_queue(max_s=7200, poll=30):
    t0 = time.time()
    while time.time() - t0 < max_s:
        if not queue_running():
            logf = os.path.join(LOGS, 'queue.log')
            if os.path.exists(logf):
                with open(logf) as f:
                    if 'ALL DONE' in f.read():
                        return True
            time.sleep(10)
            if not queue_running():
                return True
        time.sleep(poll)
    return False


def drop_stale():
    for rel in STALE:
        d = os.path.join(_ROOT, rel)
        if os.path.exists(os.path.join(d, 'results.json')):
            print(f'[finalize] drop stale cell: {rel}', flush=True)
            shutil.rmtree(d, ignore_errors=True)


def headline():
    """输出紧凑摘要（供上层 agent 直接引用到飞书消息）。"""
    import glob
    lines = []
    for tag, label in (('exp1_matrix', '身份臂'), ('exp1_matrix_deconf', '去混淆臂')):
        rows = []
        for path in sorted(glob.glob(os.path.join(RES, tag, '*', 'results.json'))):
            with open(path) as f:
                r = json.load(f)
            s = r['summary']['per_fold_mean']
            rows.append((r['arch'], r['dim'], s['accuracy'], s['balanced_accuracy'],
                         s['macro_f1'], s['roc_auc']))
        if not rows:
            continue
        lines.append(f'## {label}（5 折 × 3 seed，均值）')
        lines.append('| 模型 | 空间 | acc | bal-acc | macro-F1 | ROC-AUC |')
        lines.append('|---|---|---:|---:|---:|---:|')
        for a, d, acc, bal, f1, auc in sorted(rows, key=lambda x: (x[1], -x[5])):
            lines.append(f'| {a} | {d}d | {acc:.4f} | {bal:.4f} | {f1:.4f} | {auc:.4f} |')
        lines.append('')
    out = os.path.join(_ROOT, 'docs', 'REVISION_SUMMARY.md')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wait-queue', action='store_true')
    ap.add_argument('--skip-stale-rerun', action='store_true')
    args = ap.parse_args()

    if args.wait_queue:
        print('[finalize] waiting for queue ...', flush=True)
        wait_queue()
        print('[finalize] queue done', flush=True)

    if not args.skip_stale_rerun:
        drop_stale()
        print('[finalize] re-running queue for missing/stale cells ...', flush=True)
        sh(f'{PY} -m analysis.rev12.run_queue --workers 8')
        # 二次清理（确保去混淆臂全部为新协议）
        drop_stale()
        sh(f'{PY} -m analysis.rev12.run_queue --workers 8')

    print('[finalize] significance ...', flush=True)
    sh(f'{PY} -m analysis.rev12.exp5_stats')
    print('[finalize] aggregate ...', flush=True)
    sh(f'{PY} -m analysis.rev12.aggregate --out docs/REVISION_EXPERIMENTS_20261107.md')
    print('[finalize] paper v12 ...', flush=True)
    sh(f'{PY} analysis/hazard_model/build_paper_v12_zh.py')
    sh(f'{PY} analysis/hazard_model/build_paper_v12_en.py')
    print('[finalize] headline ...', flush=True)
    try:
        headline()
    except Exception as e:
        print(f'[finalize] headline failed: {e}', flush=True)
    print('[finalize] ALL DONE', flush=True)


if __name__ == '__main__':
    main()
