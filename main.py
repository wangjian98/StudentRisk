"""StudentRisk main entry point.

Run all models with a single command:
    python main.py --model all
    python main.py --model rf lstm bilstm attention
    python main.py --model all --seeds 42
    python main.py --model all --threshold 0.5 --n-splits 5

After training, generate comparison + visualizations:
    python main.py --compare
    python main.py --viz
"""
import os
import sys
import argparse
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


MODEL_RUNNERS = {
    'rf':         ('models.rf.train', 'run'),
    'rf7':        ('models.rf7.train', 'run'),
    'lstm':       ('models.lstm.train', 'run'),
    'bilstm':     ('models.bilstm.train', 'run'),
    'attention':  ('models.attention.train', 'run'),
    'meta_mamba': ('models.meta_mamba.train', 'run'),
}


def run_models(names, seeds, n_splits, threshold, save_oof=True):
    """Run selected models sequentially."""
    results = {}
    overall_t0 = time.time()
    for name in names:
        if name not in MODEL_RUNNERS:
            print(f"[main] Unknown model: {name}, skipping", flush=True)
            continue
        mod_path, fn_name = MODEL_RUNNERS[name]
        mod = __import__(mod_path, fromlist=[fn_name])
        fn = getattr(mod, fn_name)
        print(f"\n{'='*70}", flush=True)
        print(f"[main] Running model: {name}", flush=True)
        print(f"{'='*70}", flush=True)
        try:
            payload = fn(seeds=seeds, n_splits=n_splits,
                         threshold=threshold, save_oof=save_oof)
            results[name] = payload
        except Exception as e:
            import traceback
            print(f"[main] ERROR running {name}: {e}", flush=True)
            traceback.print_exc()
            results[name] = {'error': str(e)}
    print(f"\n[main] All requested models done in {time.time()-overall_t0:.1f}s", flush=True)
    return results


def run_compare():
    from analysis.compare import run as run_compare_inner
    return run_compare_inner()


def run_viz():
    from analysis.visualize import generate_all_visualizations
    generate_all_visualizations()


def main():
    parser = argparse.ArgumentParser(
        description='StudentRisk — multi-model training/eval on CS1 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', nargs='+', default=['all'],
                        choices=['all', 'rf', 'rf7', 'lstm', 'bilstm', 'attention', 'meta_mamba'],
                        help='Model(s) to run. Use "all" for all 4 models.')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Random seeds for CV (default from configs/default.yaml)')
    parser.add_argument('--n-splits', type=int, default=None,
                        help='Number of CV folds (default from config)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Decision threshold for FAILED (default 0.5)')
    parser.add_argument('--no-oof', action='store_true',
                        help='Skip saving OOF probabilities')
    parser.add_argument('--compare', action='store_true',
                        help='Only run comparison (no training)')
    parser.add_argument('--viz', action='store_true',
                        help='Only run visualizations (no training)')
    args = parser.parse_args()

    if args.compare:
        run_compare()
        return
    if args.viz:
        run_viz()
        return

    # Determine which models to run
    if 'all' in args.model:
        names = ['rf', 'rf7', 'lstm', 'bilstm', 'attention', 'meta_mamba']
    else:
        names = [m for m in args.model if m != 'all']

    # Load default config for seeds/n_splits if not provided
    import yaml
    cfg_path = os.path.join(_ROOT, 'configs', 'default.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seeds = args.seeds if args.seeds else cfg.get('cv', {}).get('seeds', [42, 123, 777])
    n_splits = args.n_splits if args.n_splits else cfg.get('cv', {}).get('n_splits', 5)

    print(f"[main] Models to run: {names}", flush=True)
    print(f"[main] Seeds: {seeds}", flush=True)
    print(f"[main] n_splits: {n_splits}", flush=True)
    print(f"[main] Threshold: {args.threshold}", flush=True)

    run_models(names, seeds, n_splits, args.threshold, save_oof=not args.no_oof)

    # If all 4 ran successfully, automatically build comparison + viz
    if len(names) == 4:
        print(f"\n[main] All 4 models done. Building comparison + visualizations ...", flush=True)
        try:
            run_compare()
        except Exception as e:
            print(f"[main] Comparison failed: {e}", flush=True)
        try:
            run_viz()
        except Exception as e:
            print(f"[main] Visualization failed: {e}", flush=True)


if __name__ == '__main__':
    main()