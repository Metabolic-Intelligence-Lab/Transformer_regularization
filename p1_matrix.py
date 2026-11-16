"""Driver of protocol P1 (protocols/P1_minibench.md): runs the method matrix exactly as the
protocol states it, through dc_grid.py, with the edge rule and the stacking methods applied
mechanically. Everything is journaled by dc_grid, so a stopped invocation resumes.

    python p1_matrix.py --phase r16 --size XS [--methods AdamW,CWD] [--dry-run]
    python p1_matrix.py --phase r1  --size XS            # needs the R16 results of that size
    python p1_matrix.py --phase s                        # AdamW + the 4 best XS methods, at S
    python p1_matrix.py --phase gate                     # the pilot gate (analysis/p1_gate.py)

Phases: r16 = the data-constrained arm (6 configurations per method, 3 finals, edge rule);
r1 = the control arm at each method's R16 winner plus one weaker-regularization variant
(2 finals); s = the R16 arm at size S for AdamW and the four best XS methods.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------- the protocol
LR_ADAM = [5e-4, 1.5e-3, 4.5e-3]
LR_MUON = [7e-3, 2e-2, 6e-2]
D_R16, E_R16 = 10_000_000, 16
D_R1, E_R1, EVALS_R1 = 159_383_553, 1, 32          # same 2,432 steps and tokens as R16
FINALS_R16, FINALS_R1, FINALS_S = 3, 2, 3
# method: learning-rate grid, tuned second axis (key, values), fixed keys, inherited keys
SPEC = {
    'Baseline':      dict(lrs=LR_ADAM, axis=None),
    'AdamW':         dict(lrs=LR_ADAM, axis=('wd', [0.1, 0.5])),
    'AdamW+dropout': dict(lrs=LR_ADAM, axis=('dropout', [0.1, 0.3]), inherit=('AdamW', 'wd')),
    'AdamW+EMA':     dict(lrs=LR_ADAM, axis=('ema', [0.99, 0.999]), inherit=('AdamW', 'wd')),
    'CWD':           dict(lrs=LR_ADAM, axis=('wd', [0.1, 0.5])),
    'AdamS':         dict(lrs=LR_ADAM, axis=('wd', [1e-4, 1e-3])),
    'CPR':           dict(lrs=LR_ADAM, axis=('warm_start', [0, 250])),
    'SF-AdamW':      dict(lrs=LR_ADAM, axis=('wd', [0.1, 0.5])),
    'Muon':          dict(lrs=LR_MUON, axis=('wd', [0.005, 0.02]), fixed={'aux_lr': 1.5e-3, 'aux_wd': 0.1}),
    'Muon+Const':    dict(lrs=LR_MUON, axis=('rho', [1e-4, 5e-4]), fixed={'aux_lr': 1.5e-3}),
    'Const':         dict(lrs=LR_ADAM, axis=('rho', [1e-4, 5e-4])),
    'Const-Fair':    dict(lrs=LR_ADAM, axis=('rho', [1e-4, 5e-4]), fixed={'delta': 0.1}),
    'Const-Huber':   dict(lrs=LR_ADAM, axis=('rho', [1e-4, 5e-4]), fixed={'delta': 0.1}),
}
ORDER = list(SPEC)                       # AdamW precedes the methods that inherit from it


def extend_value(method, axis, value, side):
    """The edge rule: the next grid value outward of `value` on `axis`, or None (no extension)."""
    up = side == 'high'
    if axis == 'lr':
        return value * 3 if up else value / 3
    if axis == 'wd':
        f = 10 if method == 'AdamS' else 5
        return value * f if up else value / f
    if axis == 'rho':
        return value * 5 if up else value / 5
    if axis == 'dropout':
        return 0.5 if up else 0.03
    if axis == 'ema':
        return 0.9999 if up else 0.9
    if axis == 'warm_start':
        return 1000 if up else None
    raise ValueError(axis)


def weaker_value(method, axis, value):
    """The weaker-regularization variant of the control arm (protocol, 'Control arm R1')."""
    if axis in ('wd', 'rho'):
        return value / 5
    if axis == 'dropout':
        return 0.1 if value >= 0.3 else 0.03
    if axis == 'ema':
        return 0.99 if value >= 0.999 else 0.9
    if axis == 'warm_start':
        return 1000 if value < 1000 else value * 4
    raise ValueError(axis)


def stem(size, d, e, method):
    return f"{size}_D{d / 1e6:g}M_E{e}_{method}"


def result_path(size, d, e, method, results='results/dc'):
    return os.path.join(results, stem(size, d, e, method) + '.json')


def load_winner(size, d, e, method, results='results/dc'):
    p = result_path(size, d, e, method, results)
    if not os.path.exists(p):
        return None
    return json.load(open(p))['winner']['hp']


def grid_args(size, d, e, method, lrs, grid, fixed, finals, evals_per_epoch=None):
    """Argument list of one dc_grid.py invocation."""
    args = ['--size', size, '--unique-tokens', str(d), '--epochs', str(e), '--method', method,
            '--lrs', ','.join(f'{v:g}' for v in lrs), '--finals', str(finals)]
    for k, vals in grid.items():
        args += ['--grid', f"{k}=" + ','.join(f'{v:g}' for v in vals)]
    for k, v in fixed.items():
        args += ['--fixed', f'{k}={v:g}']
    if evals_per_epoch is not None:
        args += ['--evals-per-epoch', str(evals_per_epoch)]
    return args


def r16_plan(method, size, results='results/dc'):
    """Initial lrs / grid / fixed of a method's R16 search (before any edge extension)."""
    spec = SPEC[method]
    fixed = dict(spec.get('fixed', {}))
    if 'inherit' in spec:
        src, key = spec['inherit']
        w = load_winner(size, D_R16, E_R16, src, results)
        if w is None:
            raise FileNotFoundError(f"{method} inherits {key} from {src}: run {src} first "
                                    f"({result_path(size, D_R16, E_R16, src, results)})")
        fixed[key] = w[key]
    grid = {spec['axis'][0]: list(spec['axis'][1])} if spec['axis'] else {}
    return list(spec['lrs']), grid, fixed


def run_search(method, size, d, e, lrs, grid, fixed, finals, runner, evals_per_epoch=None,
               edge_rule=True, log=print):
    """One dc_grid search with the edge rule: re-invoke with an extended axis while the
    winner sits on an edge that has not been extended yet (at most once per axis)."""
    lrs, grid = list(lrs), {k: list(v) for k, v in grid.items()}
    extended = {k: False for k in ['lr'] + list(grid)}
    while True:
        out = runner(grid_args(size, d, e, method, lrs, grid, fixed, finals, evals_per_epoch))
        if not edge_rule:
            return out
        w = out['winner']
        moves = []
        for ax in extended:
            vals = lrs if ax == 'lr' else grid[ax]
            if not w['edge'].get(ax) or extended[ax] or len(vals) < 2:
                continue
            side = 'high' if w['hp'][ax] == max(vals) else 'low'
            nv = extend_value(method, ax, w['hp'][ax], side)
            if nv is None:
                continue
            vals.append(nv)
            vals.sort()
            extended[ax] = True
            moves.append((ax, nv))
        if not moves:
            return out
        log(f"[P1] {method}: winner {w['hp']} on edge -> extending {moves}")


def phase_r16(size, methods, runner, results='results/dc', finals=FINALS_R16, log=print):
    outs = {}
    for m in methods:
        lrs, grid, fixed = r16_plan(m, size, results)
        log(f"[P1] R16 {size} {m}: lrs {lrs} grid {grid} fixed {fixed}")
        outs[m] = run_search(m, size, D_R16, E_R16, lrs, grid, fixed, finals, runner, log=log)
    return outs


def r1_plan(method, size, results='results/dc'):
    """lrs / grid / fixed of the control-arm search: the R16 winner and its weaker variant."""
    w = load_winner(size, D_R16, E_R16, method, results)
    if w is None:
        raise FileNotFoundError(f"R1 needs the R16 result of {method}: {result_path(size, D_R16, E_R16, method, results)}")
    spec = SPEC[method]
    fixed = {k: v for k, v in w.items() if k != 'lr' and not (spec['axis'] and k == spec['axis'][0])}
    grid = {}
    if spec['axis']:
        ax = spec['axis'][0]
        vals = sorted({w[ax], weaker_value(method, ax, w[ax])})
        grid[ax] = vals
    return [w['lr']], grid, fixed


def phase_r1(size, methods, runner, results='results/dc', finals=FINALS_R1, log=print):
    outs = {}
    for m in methods:
        lrs, grid, fixed = r1_plan(m, size, results)
        log(f"[P1] R1 {size} {m}: lrs {lrs} grid {grid} fixed {fixed}")
        outs[m] = run_search(m, size, D_R1, E_R1, lrs, grid, fixed, finals, runner,
                             evals_per_epoch=EVALS_R1, edge_rule=False, log=log)
    return outs


def top_methods(size='XS', results='results/dc', k=4):
    """AdamW plus the k best other methods by mean test loss over the R16 finals."""
    means = {}
    for m in ORDER:
        p = result_path(size, D_R16, E_R16, m, results)
        if not os.path.exists(p):
            raise FileNotFoundError(f"S phase needs every XS R16 result; missing {p}")
        fin = json.load(open(p))['finals']
        if not fin:
            raise ValueError(f"{p} has no finals")
        means[m] = sum(r['test_loss'] for r in fin.values()) / len(fin)
    best = sorted((m for m in means if m != 'AdamW'), key=lambda m: means[m])[:k]
    return ['AdamW'] + [m for m in ORDER if m in best], means


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--phase', required=True, choices=['r16', 'r1', 's', 'gate'])
    ap.add_argument('--size', default='XS', choices=['XS', 'S'])
    ap.add_argument('--methods', default=None, help='comma-separated subset, protocol order kept')
    ap.add_argument('--results', default='results/dc')
    ap.add_argument('--dry-run', action='store_true', help='print the dc_grid invocations only')
    args = ap.parse_args(argv)

    if args.phase == 'gate':
        from analysis import p1_gate
        return p1_gate.main(['--results', args.results])

    methods = ORDER if not args.methods else [m for m in ORDER if m in args.methods.split(',')]
    unknown = set(args.methods.split(',')) - set(ORDER) if args.methods else set()
    if unknown:
        raise SystemExit(f"unknown methods {sorted(unknown)}")

    if args.dry_run:
        def runner(a):
            print('  dc_grid.py ' + ' '.join(a))
            return {'winner': {'hp': {}, 'edge': {}}}
    else:
        import dc_grid

        def runner(a):
            print('[P1] dc_grid.py ' + ' '.join(a), flush=True)
            return dc_grid.main(a)

    if args.phase == 'r16':
        return phase_r16(args.size, methods, runner, args.results)
    if args.phase == 'r1':
        return phase_r1(args.size, methods, runner, args.results)
    if args.phase == 's':
        chosen, means = top_methods('XS', args.results)
        print('[P1] XS R16 mean test loss: ' + ', '.join(f'{m} {v:.4f}' for m, v in sorted(means.items(), key=lambda kv: kv[1])))
        print(f'[P1] S phase methods: {chosen}')
        return phase_r16('S', chosen, runner, args.results, finals=FINALS_S)


if __name__ == '__main__':
    main()
