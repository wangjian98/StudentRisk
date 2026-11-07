"""
rev12.run_queue — cell 级并行调度器（246 / 单 GPU）

GPU 上的 MetaMamba 训练是**延迟受限**（S6 逐时间步的 kernel launch），单进程跑不满
GPU；把实验拆到 **cell 级**并行，用 K 个 worker 共享一张 T4，整体吞吐可提升 2–3 倍。

用法：
  python3 -m analysis.rev12.run_queue --workers 8
每个 job = 一个 cell（一个 arch×dim 或一个 prefix×cell 或一个 FiLM 格），
已完成的 cell（marker 文件存在）自动跳过。
"""
import os
import sys
import json
import time
import argparse
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
RES = os.path.join(_ROOT, 'results', 'rev12')
LOG_DIR = os.path.join(_ROOT, 'logs')

PY = sys.executable


def _jobs(specs):
    """specs: list of (label, args, marker_relpath)"""
    out = []
    for label, args, marker in specs:
        out.append({'label': label, 'args': args, 'marker': os.path.join(_ROOT, marker)})
    return out


def build_jobs(scope='all'):
    J = []

    # ── 实验 1：模型 × 输入空间矩阵（身份 / 去混淆） ─────────────
    arch_sets = {
        'hmm_cat': 'hmm_cat', 'hmm_gmm': 'hmm_gmm',
    }
    cells_1 = ([f'rf_d{d}' for d in (7, 11, 28)]
               + [f'lstm_d{d}' for d in (7, 11, 28)]
               + [f'bilstm_d{d}' for d in (7, 11, 28)]
               + [f'attention_d{d}' for d in (7, 11, 28)]
               + [f'metamamba_d{d}' for d in (7, 11, 28)]
               + ['rf7_all_d7', 'hmm_cat_d7', 'hmm_gmm_d28'])
    for tag, extra in (('exp1_matrix', []), ('exp1_matrix_deconf', ['--deconfound'])):
        for cell in cells_1:
            dim = cell.split('_d')[1]
            J.append(_jobs([(f'{tag}:{cell}',
                             [PY, '-m', 'analysis.rev12.exp1_matrix', '--spaces', dim,
                              '--archs', cell.split('_d')[0], '--tag', tag] + extra,
                             f'results/rev12/{tag}/{cell}/results.json')])[0])

    # ── 实验 2：早期预警前缀曲线 ────────────────────────────────
    for cell, pref in [('metamamba_d28', '0.1'), ('metamamba_d28', '0.2'),
                       ('metamamba_d28', '0.4'), ('metamamba_d28', '1.0'),
                       ('rf_d28', '0.1'), ('rf_d28', '0.2'), ('rf_d28', '0.4'), ('rf_d28', '1.0'),
                       ('lstm_d28', '0.1'), ('lstm_d28', '0.2'), ('lstm_d28', '0.4'), ('lstm_d28', '1.0'),
                       ('hmm_gmm_d28', '0.1'), ('hmm_gmm_d28', '0.4'), ('hmm_gmm_d28', '1.0')]:
        J.append(_jobs([(f'exp2:{cell}@{pref}',
                         [PY, '-m', 'analysis.rev12.exp2_prefix', '--prefixes', pref,
                          '--cells', cell, '--tag', 'exp2_prefix'],
                         f'results/rev12/exp2_prefix/{cell}_p{pref}/results.json')])[0])

    # ── 实验 2d：去混淆前缀曲线 ────────────────────────────────
    for cell, pref in [('metamamba_d28', '0.1'), ('metamamba_d28', '0.4'), ('metamamba_d28', '1.0'),
                       ('rf_d28', '0.1'), ('rf_d28', '0.4'), ('rf_d28', '1.0')]:
        J.append(_jobs([(f'exp2d:{cell}@{pref}',
                         [PY, '-m', 'analysis.rev12.exp2_prefix', '--prefixes', pref,
                          '--cells', cell, '--tag', 'exp2_prefix_deconf', '--deconfound'],
                         f'results/rev12/exp2_prefix_deconf/{cell}_p{pref}/results.json')])[0])

    # ── 实验 3：FiLM 多 seed ───────────────────────────────────
    for dim, seeds in ((7, '42,123,777,2024,7'), (11, '42,123,777,2024,7'),
                       (28, '42,123,777')):
        for arm in ('on', 'off'):
            J.append(_jobs([(f'exp3:d{dim}_film{arm}',
                             [PY, '-m', 'analysis.rev12.exp3_film', '--dims', str(dim),
                              '--seeds', seeds, '--only', f'{dim}_{arm}',
                              '--tag', 'exp3_film'],
                             f'results/rev12/exp3_film/d{dim}_film{arm}/results.json')])[0])

    # ── 实验 4：LOPO ───────────────────────────────────────────
    for cell in ('metamamba_d28', 'metamamba_d28_nofilm'):
        J.append(_jobs([(f'exp4:{cell}',
                         [PY, '-m', 'analysis.rev12.exp4_lopo', '--cells', cell,
                          '--tag', 'exp4_lopo'],
                         f'results/rev12/exp4_lopo/{cell}/results.json')])[0])
    return J


def run_pool(jobs, workers=8):
    os.makedirs(LOG_DIR, exist_ok=True)
    pending = [j for j in jobs if not os.path.exists(j['marker'])]
    print(f'[queue] {len(jobs)} jobs, {len(jobs)-len(pending)} done, '
          f'{len(pending)} pending, workers={workers}', flush=True)
    running = []
    t0 = time.time()
    done = 0
    while pending or running:
        while pending and len(running) < workers:
            j = pending.pop(0)
            logf = open(os.path.join(LOG_DIR, f'q_{j["label"].replace(":", "_").replace("@", "_")}.log'), 'w')
            env = dict(os.environ)
            env['PYTHONPATH'] = _ROOT
            env['SR12_CACHE'] = '/home/ubuntu/SR12_cache'
            env['OMP_NUM_THREADS'] = '2'
            p = subprocess.Popen(j['args'], stdout=logf, stderr=subprocess.STDOUT,
                                 env=env, cwd=_ROOT)
            running.append((j, p, logf))
            print(f'[queue] start {j["label"]} (pid={p.pid})', flush=True)
            time.sleep(2)
        time.sleep(10)
        still = []
        for j, p, logf in running:
            if p.poll() is None:
                still.append((j, p, logf))
            else:
                logf.close()
                done += 1
                ok = os.path.exists(j['marker'])
                print(f'[queue] finish {j["label"]} rc={p.returncode} ok={ok} '
                      f'({done} done, {time.time()-t0:.0f}s elapsed)', flush=True)
        running = still
    print(f'[queue] ALL DONE in {time.time()-t0:.0f}s', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    jobs = build_jobs()
    if args.dry_run:
        for j in jobs:
            done = 'DONE' if os.path.exists(j['marker']) else 'TODO'
            print(f'{done:5s} {j["label"]}')
        return
    run_pool(jobs, args.workers)


if __name__ == '__main__':
    main()
