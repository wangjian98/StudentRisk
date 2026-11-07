"""
rev12.exp2_prefix — 实验 2：早期预警曲线（审稿意见 调整2 · 硬伤2 的正面回应）

问题：论文标题/引言主张"学期初前几周做出准确预测"，而原方法用"每生最后 128 个事件"
（≈期末数据），形成"用期末数据讲早期故事"的逻辑裂缝。

做法：对每名学生的事件序列取**前 p%**（p ∈ {10,20,40,60,80,100%}），
在每个 p 上独立训练并测试同协议模型，画"前缀比例 → 性能"曲线。
  * 序列模型输入 = 该前缀内的最后 128 个事件
  * RF 输入 = 同一窗口的聚合孪生
  * 标签不变（该生最终是否挂科）
输出：results/rev12/exp2_prefix/<arch>_d<dim>_p<frac>/
"""
import os
import sys
import time
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.rev12.common import (load_cache, make_folds, build_spaces_multi,
                                   build_tabulars_multi)
from analysis.rev12.exp1_matrix import run_cell, OUT_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefixes', default='0.1,0.2,0.4,0.6,0.8,1.0')
    ap.add_argument('--cells', default='metamamba_d11,metamamba_d28,rf_d28,lstm_d28,attention_d28,hmm_gmm_d28')
    ap.add_argument('--seeds', default='42,123,777')
    ap.add_argument('--n-splits', type=int, default=5)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--tag', default='exp2_prefix')
    ap.add_argument('--deconfound', action='store_true')
    ap.add_argument('--only', default='',
                    help='只跑指定 cell，逗号分隔（如 metamamba_d28,rf_d28）')
    args = ap.parse_args()

    prefixes = [float(p) for p in args.prefixes.split(',') if p]
    cells = [c for c in args.cells.split(',') if c]
    seeds = tuple(int(s) for s in args.seeds.split(','))

    cache = load_cache()
    y = cache['y']
    folds = make_folds(y, seeds, args.n_splits)
    print(f"[exp2] students={len(y)} folds={len(folds)}", flush=True)

    base = os.path.join(OUT_ROOT, args.tag)
    # 预构建：每个输入空间一次性编码 28.5M 事件，切出全部前缀窗口
    dims = sorted({int(c.split('_d')[1]) for c in cells})
    prebuilt = {}
    for dim in dims:
        seqs = build_spaces_multi(cache, dim, prefixes, args.max_len)
        tabs = build_tabulars_multi(cache, dim, prefixes, args.max_len)
        for p in prefixes:
            prebuilt[(dim, p)] = {'seq': seqs[p][0], 'mask': seqs[p][1], 'task': seqs[p][2],
                                  'X': tabs[p][0], 'names': tabs[p][1]}

    only = {c.strip() for c in args.only.split(',') if c.strip()}
    for p in prefixes:
        for cell in cells:
            if only and cell not in only:
                continue
            arch, dim = cell.split('_d')
            dim = int(dim)
            out_dir = os.path.join(base, f"{cell}_p{p}")
            if os.path.exists(os.path.join(out_dir, 'results.json')):
                print(f"[exp2] skip {cell} p{p} (done)", flush=True)
                continue
            print(f"[exp2] === {cell} prefix={p} ===", flush=True)
            try:
                run_cell(arch, dim, cache, y, folds, out_dir, device=args.device,
                         prefix_frac=p, max_len=args.max_len,
                         prebuilt=prebuilt.get((dim, p)),
                         deconfound=args.deconfound)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[exp2] FAILED {cell} p{p}: {e}", flush=True)
    print("[exp2] done", flush=True)


if __name__ == '__main__':
    main()
