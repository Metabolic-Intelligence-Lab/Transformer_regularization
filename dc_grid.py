"""
Grid search + final runs for the data-constrained language-modelling study (dcstudy/).

One method per invocation. The grid is the Cartesian product of the learning rates and
of the method's own hyperparameter values given on the command line; every point is
trained on --tune-seed and selected on best-checkpoint validation loss. Finals re-train the
winner on FRESH seeds (never the tuning seed), so the reported numbers carry no selection
bias. Every run is journaled; an interrupted invocation resumes without retraining, and a
re-invocation with an EXTENDED grid (one more --lrs or --grid value) retrains only the new
points, plus the finals if the winner moved (finals are journaled under the winner's
hyperparameters, not under the method name alone).

    python dc_grid.py --size S --unique-tokens 20000000 --epochs 16 --method AdamW \
        --lrs 3e-4,1e-3,3e-3 --grid wd=0.1,0.3,1.0 --finals 5
    python dc_grid.py --size S --unique-tokens 20000000 --epochs 16 --method Baseline \
        --lrs 1e-3 --finals 0                                   # single calibration run
    python dc_grid.py --smoke --method Const --lrs 3e-3 --grid rho=1e-4,3e-4   # CPU, seconds

Output: results/dc/<stem>.json and results/journal/dc_<stem>.jsonl, stem =
<size>_D<unique tokens in M>_E<epochs>_<method>[_<tag>].
"""
import argparse
import itertools
import json
import math
import os

import numpy as np
import torch

from dcstudy.methods import METHODS
from dcstudy.train import load_tokens, run
from experiment_utils import RunJournal

FINAL_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
VAL_EVAL_TOKENS = 1 << 20          # periodic validation on the first 1,048,576 tokens


def parse_grid(items):
    """['wd=0.1,0.3', 'delta=0.5'] -> {'wd': [0.1, 0.3], 'delta': [0.5]}"""
    out = {}
    for it in items or []:
        k, v = it.split('=', 1)
        out[k.strip()] = [float(x) for x in v.split(',') if x.strip()]
    return out


def grid_points(method, lrs, grid, fixed):
    keys = [k for k in METHODS[method]['hp'] if k not in fixed]
    missing = [k for k in keys if k not in grid]
    if missing:
        raise SystemExit(f"--grid must give values for {missing} (method {method})")
    extra = [k for k in grid if k not in keys]
    if extra:
        raise SystemExit(f"--grid has keys {extra} that {method} does not tune")
    pts = []
    for lr in lrs:
        for combo in itertools.product(*[grid[k] for k in keys]):
            pts.append({'lr': lr, **dict(zip(keys, combo)), **fixed})
    return pts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--size', default='S', choices=['tiny', 'XS', 'S', 'M'])
    ap.add_argument('--unique-tokens', type=int, default=20_000_000)
    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--method', required=True, choices=list(METHODS))
    ap.add_argument('--lrs', required=True, help='comma-separated peak learning rates')
    ap.add_argument('--grid', action='append', help='key=v1,v2,... (repeatable)')
    ap.add_argument('--fixed', action='append', help='key=value held fixed (e.g. aux_lr=1e-3)')
    ap.add_argument('--finals', type=int, default=5)
    ap.add_argument('--final-seeds', default='',
                    help='comma-separated final seeds; overrides the first --finals of FINAL_SEEDS '
                         '(protocol P2 uses the held-out seeds 2001-2006)')
    ap.add_argument('--tune-seed', type=int, default=42)
    ap.add_argument('--batch-seqs', type=int, default=64)
    ap.add_argument('--micro-seqs', type=int, default=16)
    ap.add_argument('--block', type=int, default=1024)
    ap.add_argument('--warmup-frac', type=float, default=0.02)
    ap.add_argument('--lr-floor', type=float, default=0.0)
    ap.add_argument('--evals-per-epoch', type=int, default=2)
    ap.add_argument('--data', default='data/dc')
    ap.add_argument('--tag', default='')
    ap.add_argument('--no-compile', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='synthetic data, tiny model, CPU/GPU')
    args = ap.parse_args(argv)

    fixed = {k: v[0] for k, v in parse_grid(args.fixed).items()}
    pts = grid_points(args.method, [float(x) for x in args.lrs.split(',')],
                      parse_grid(args.grid), fixed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.smoke:
        rng = np.random.default_rng(0)
        base = rng.integers(0, 300, 257)
        mk = lambda n, s: np.where(np.random.default_rng(s).random(n) < 0.8,  # noqa: E731
                                   np.tile(base, n // 257 + 1)[:n],
                                   np.random.default_rng(s + 7).integers(0, 300, n)).astype(np.uint16)
        train, val, test = mk(4000, 1), mk(800, 2), mk(800, 3)
        args.size, args.block, args.batch_seqs, args.micro_seqs, args.epochs = 'tiny', 32, 8, 4, 2
        args.unique_tokens = len(train)
    else:
        train = load_tokens(os.path.join(args.data, 'train.bin'), args.unique_tokens)
        if len(train) < args.unique_tokens:
            raise SystemExit(f"train.bin has only {len(train):,} tokens")
        val = load_tokens(os.path.join(args.data, 'val.bin'), VAL_EVAL_TOKENS + 1)
        test = load_tokens(os.path.join(args.data, 'test.bin'))

    steps_per_epoch = ((len(train) - 1) // args.block) // args.batch_seqs
    base_cfg = dict(size=args.size, epochs=args.epochs, block_size=args.block,
                    batch_seqs=args.batch_seqs, micro_seqs=args.micro_seqs,
                    warmup_frac=args.warmup_frac, lr_floor=args.lr_floor,
                    eval_every=max(1, steps_per_epoch // args.evals_per_epoch),
                    eval_batch=16, device=device,
                    amp='bf16' if device == 'cuda' else None,
                    compile=(device == 'cuda' and not args.no_compile), method=args.method)
    dtag = f"D{args.unique_tokens / 1e6:g}M" if not args.smoke else 'Dsmoke'
    stem = f"{args.size}_{dtag}_E{args.epochs}_{args.method}" + (f"_{args.tag}" if args.tag else '')
    os.makedirs(os.path.join('results', 'dc'), exist_ok=True)
    journal = RunJournal(os.path.join('results', 'journal', f'dc_{stem}.jsonl'))
    protocol = {k: base_cfg[k] for k in ('size', 'epochs', 'block_size', 'batch_seqs',
                                         'warmup_frac', 'lr_floor', 'eval_every')}
    protocol['unique_tokens'] = args.unique_tokens

    def key_of(hp, seed):
        return {**protocol, **hp, 'seed': seed}

    grid = []
    print(f"[DC] {stem}: {len(pts)} grid point(s), device={device}", flush=True)
    for hp in pts:
        k = key_of(hp, args.tune_seed)
        m = journal.lookup_pso(args.method, k)
        if m is None:
            m = run(dict(base_cfg, hp=hp, seed=args.tune_seed), train, val, test)
            journal.record_pso(args.method, k, m)
        grid.append({'hp': hp, **{kk: vv for kk, vv in m.items() if kk != 'curve'},
                     'curve': m['curve']})
        print(f"  [GRID] {hp} -> val {m['val_loss']:.4f} (best ep {m['best_epoch']:.2f}) "
              f"final val {m['final_val_loss']:.4f} tok/s {m['tokens_per_second']:,.0f}", flush=True)

    score = lambda g: g['val_loss'] if math.isfinite(g['val_loss']) else math.inf  # noqa: E731
    best = min(grid, key=score)
    edge = {}
    for kk in ['lr'] + [k for k in METHODS[args.method]['hp'] if k not in fixed]:
        vals = sorted({g['hp'][kk] for g in grid})
        edge[kk] = len(vals) > 1 and best['hp'][kk] in (vals[0], vals[-1])
    journal.record_best_hp(args.method, {'hp': best['hp'], 'val_loss': best['val_loss'], 'edge': edge})
    print(f"[DC] winner {best['hp']} val {best['val_loss']:.4f} on-edge {edge}", flush=True)

    # Finals are journaled under the WINNER's hyperparameters, not the method name alone,
    # so that an extended grid whose winner moved does not reuse the old winner's finals.
    fin_name = f"{args.method}@{RunJournal._key(best['hp'])}"
    finals = {}
    seeds = ([int(x) for x in args.final_seeds.split(',')] if args.final_seeds
             else FINAL_SEEDS[:args.finals])
    if args.tune_seed in seeds:
        raise SystemExit('a final seed must differ from the tuning seed')
    for s in seeds:
        m = journal.lookup_eval(fin_name, s)
        if m is None:
            m = run(dict(base_cfg, hp=best['hp'], seed=s), train, val, test)
            journal.record_eval(fin_name, s, m)
        finals[str(s)] = m
        print(f"  [FINAL] seed {s} test {m['test_loss']:.4f} (ppl {m['test_ppl']:.2f}) "
              f"final-test {m['final_test_loss']:.4f} best ep {m['best_epoch']:.2f}", flush=True)
    journal.close()

    out = {'stem': stem, 'method': args.method, 'protocol': protocol, 'tune_seed': args.tune_seed,
           'grid': grid, 'winner': {'hp': best['hp'], 'val_loss': best['val_loss'], 'edge': edge},
           'finals': finals,
           'data_meta': (json.load(open(os.path.join(args.data, 'meta.json')))
                         if not args.smoke and os.path.exists(os.path.join(args.data, 'meta.json'))
                         else None)}
    path = os.path.join('results', 'dc', stem + '.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"[DC] wrote {path}", flush=True)
    return out


if __name__ == '__main__':
    main()
