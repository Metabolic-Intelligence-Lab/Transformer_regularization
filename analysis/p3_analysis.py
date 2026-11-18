"""Analysis of protocol P3 (protocols/P3_controls.md), fixed before the runs.

    python analysis/p3_analysis.py [--out results/p3_analysis.json] [--tex-dir paper/jmlr/generated]

Part A (S, 10M unique tokens x 16 epochs): CWD (P2 finals) against RandWD (H4, primary) and AntiCWD (H5),
paired over seeds 2001-2006; two-sided paired t-test, Holm over H4 and H5, t-based CI95 (bootstrap CI95
alongside). Each control is also contrasted with AdamW (P2 finals). Confirmed if p_holm < 0.05 and the t CI95
lies below zero. Total decay = wd x fraction of decayed coordinates (sign agreement for CWD, 0.40 by
construction for RandWD, 1 - agreement for AntiCWD).
Part B (XS, 159.4M unique tokens x 1 epoch, seeds 1001-1003): re-tuned AdamW, AdamW+EMA, AdamW+dropout.
Contrast with AdamW in R1 (paired t CI95), the R16 contrast from P1 (paired, same seeds) and the regime effect
(R16 contrast minus R1 contrast) with a Welch t CI95, the arms being independent runs.
Sensitivity P3x (protocols/P3x_extension.md, post-registration): the same part-B analysis on the P3x runs
(ema and dropout grids extended, finals on seeds 1001-1006) with R16 from the P1 six-seed runs (`_x6`).
"""
import argparse
import json
import os

import numpy as np
from scipy import stats

B = 10_000
S16, XS1, XS16 = 'S_D10M_E16', 'XS_D159.384M_E1', 'XS_D10M_E16'


def load(name):
    return json.load(open(f'results/dc/{name}.json'))


def finals(res):
    return {s: f['test_loss'] for s, f in res['finals'].items()}


def mean_agree(res):
    vals = [e['agree'] for f in res['finals'].values() for e in f['curve'] if e.get('agree') is not None]
    return float(np.mean(vals)) if vals else None


def t_interval(d):
    d = np.asarray(d)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return [float(d.mean() - half), float(d.mean() + half)]


def boot_mean(d, rng):
    d = np.asarray(d)
    return np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)], [2.5, 97.5]).tolist()


def paired(a, b, rng):
    """a - b over shared seeds."""
    seeds = sorted(set(a) & set(b), key=int)
    d = np.array([a[s] - b[s] for s in seeds])
    t = stats.ttest_1samp(d, 0.0)
    return {'seeds': [int(s) for s in seeds], 'diff': d.tolist(), 'mean': float(d.mean()),
            'sd': float(d.std(ddof=1)), 't_ci': t_interval(d), 'boot_ci': boot_mean(d, rng),
            't': float(t.statistic), 'p': float(t.pvalue), 'wins': int((d < 0).sum())}


def welch_diff(x, y):
    """mean(x) - mean(y) with a Welch t CI95."""
    x, y = np.asarray(x), np.asarray(y)
    vx, vy = x.var(ddof=1) / len(x), y.var(ddof=1) / len(y)
    se = np.sqrt(vx + vy)
    df = (vx + vy) ** 2 / (vx ** 2 / (len(x) - 1) + vy ** 2 / (len(y) - 1))
    m = float(x.mean() - y.mean())
    h = stats.t.ppf(0.975, df) * se
    return {'mean': m, 't_ci': [m - h, m + h], 'df': float(df)}


def holm(ps):
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def arm(res):
    f = finals(res)
    v = np.array(list(f.values()))
    return {'winner': res['winner']['hp'], 'edge': res['winner']['edge'], 'tune_val': res['winner']['val_loss'],
            'grid_size': len(res['grid']), 'test': f, 'mean': float(v.mean()), 'sd': float(v.std(ddof=1))}


def part_a(rng):
    cwd, adamw = load(f'{S16}_CWD_p2'), load(f'{S16}_AdamW_p2')
    out = {'CWD': {**arm(cwd), 'agree': mean_agree(cwd)}, 'AdamW': {**arm(adamw), 'agree': mean_agree(adamw)}}
    out['CWD']['total_decay'] = out['CWD']['agree'] * out['CWD']['winner']['wd']
    for name, frac in (('RandWD', lambda ag: 0.40), ('AntiCWD', lambda ag: 1 - ag)):
        r = load(f'{S16}_{name}_p3')
        out[name] = {**arm(r), 'agree': mean_agree(r)}
        out[name]['total_decay'] = frac(out[name]['agree']) * out[name]['winner']['wd']
        out[name]['cwd_minus'] = paired(finals(cwd), finals(r), rng)
        out[name]['minus_adamw'] = paired(finals(r), finals(adamw), rng)
    ph = holm([out['RandWD']['cwd_minus']['p'], out['AntiCWD']['cwd_minus']['p']])
    for name, p, h in (('RandWD', ph[0], 'H4'), ('AntiCWD', ph[1], 'H5')):
        c = out[name]['cwd_minus']
        c['p_holm'] = p
        c['confirmed'] = bool(p < 0.05 and c['t_ci'][1] < 0)
        out[h] = {'control': name, 'confirmed': c['confirmed']}
    return out


def part_b(rng):
    a1 = load(f'{XS1}_AdamW_p3')
    a16 = load(f'{XS16}_AdamW')
    out = {'AdamW': {**arm(a1), 'p1_r1': arm(load(f'{XS1}_AdamW'))}}
    for name in ('AdamW+EMA', 'AdamW+dropout'):
        r1, r16 = load(f'{XS1}_{name}_p3'), load(f'{XS16}_{name}')
        c1 = paired(finals(r1), finals(a1), rng)
        c16 = paired(finals(r16), finals(a16), rng)
        p1 = load(f'{XS1}_{name}')
        out[name] = {**arm(r1), 'r1_contrast': c1, 'r16_contrast': c16,
                     'regime_effect': welch_diff(c16['diff'], c1['diff']),
                     'p1_r1': {**arm(p1), 'contrast': paired(finals(p1), finals(load(f'{XS1}_AdamW')), rng)}}
    return out


def part_x(rng):
    a1, a16 = load(f'{XS1}_AdamW_p3x'), load(f'{XS16}_AdamW_x6')
    out = {'AdamW': arm(a1)}
    for name in ('AdamW+EMA', 'AdamW+dropout'):
        r1, r16 = load(f'{XS1}_{name}_p3x'), load(f'{XS16}_{name}_x6')
        c1 = paired(finals(r1), finals(a1), rng)
        c16 = paired(finals(r16), finals(a16), rng)
        out[name] = {**arm(r1), 'r1_contrast': c1, 'r16_contrast': c16,
                     'regime_effect': welch_diff(c16['diff'], c1['diff'])}
    return out


def tex_tables(a, b, d, x=None):
    f = lambda x: f'{x:g}'
    ci = lambda c: f"$[{c[0]:+.3f}, {c[1]:+.3f}]$"
    L = ['% generated by analysis/p3_analysis.py; do not edit by hand',
         '\\begin{tabular}{@{}llrrrrccrr@{}}', '\\toprule',
         'Test & Method & lr & wd & total decay & test loss & CWD $-$ method, t CI95 & bootstrap CI95 & $p$ & $p_{\\mathrm{Holm}}$ \\\\',
         '\\midrule']
    for name in ('CWD', 'AdamW'):
        r = a[name]
        td = f"{r['total_decay']:.1f}" if 'total_decay' in r else '--'
        L.append(f"-- & {name} & {f(r['winner']['lr'])} & {f(r['winner']['wd'])} & {td} & "
                 f"{r['mean']:.3f} & -- & -- & -- & -- \\\\")
    for h, name in (('H4', 'RandWD'), ('H5', 'AntiCWD')):
        r, c = a[name], a[name]['cwd_minus']
        L.append(f"{h} & {name} & {f(r['winner']['lr'])} & {f(r['winner']['wd'])} & {r['total_decay']:.1f} & "
                 f"{r['mean']:.3f} & ${c['mean']:+.3f}$ {ci(c['t_ci'])} & {ci(c['boot_ci'])} & "
                 f"{c['p']:.1e} & {c['p_holm']:.1e} \\\\")
    L += ['\\bottomrule', '\\end{tabular}', '']
    open(os.path.join(d, 'p3a.tex'), 'w').write('\n'.join(L))

    def hp(w):
        extra = [f'{k} {f(v)}' for k, v in w.items() if k not in ('lr', 'wd')]
        return ', '.join([f(w['lr']), f(w['wd'])] + extra)
    L = ['% generated by analysis/p3_analysis.py; do not edit by hand',
         '\\begin{tabular}{@{}llrcccc@{}}', '\\toprule',
         'Method & R1 winner (lr, wd, \\ldots) & R1 test loss & R1 contrast, t CI95 & R16 contrast (P1) & '
         'regime effect, Welch CI95 & P1 R1 contrast \\\\', '\\midrule',
         f"AdamW & {hp(b['AdamW']['winner'])} & {b['AdamW']['mean']:.3f} & -- & -- & -- & -- \\\\"]
    for name in ('AdamW+EMA', 'AdamW+dropout'):
        r = b[name]
        c1, c16, re_, p1 = r['r1_contrast'], r['r16_contrast'], r['regime_effect'], r['p1_r1']['contrast']
        L.append(f"{name} & {hp(r['winner'])} & {r['mean']:.3f} & ${c1['mean']:+.3f}$ {ci(c1['t_ci'])} & "
                 f"${c16['mean']:+.3f}$ & ${re_['mean']:+.3f}$ {ci(re_['t_ci'])} & ${p1['mean']:+.3f}$ \\\\")
    L += ['\\bottomrule', '\\end{tabular}', '']
    open(os.path.join(d, 'p3b.tex'), 'w').write('\n'.join(L))
    if x is None:
        return
    L = ['% generated by analysis/p3_analysis.py; do not edit by hand',
         '\\begin{tabular}{@{}llrccc@{}}', '\\toprule',
         'Method & R1 winner (lr, wd, \\ldots) & R1 test loss & R1 contrast, t CI95 & R16 contrast, t CI95 & '
         'regime effect, Welch CI95 \\\\', '\\midrule',
         f"AdamW & {hp(x['AdamW']['winner'])} & {x['AdamW']['mean']:.3f} & -- & -- & -- \\\\"]
    for name in ('AdamW+EMA', 'AdamW+dropout'):
        r = x[name]
        c1, c16, re_ = r['r1_contrast'], r['r16_contrast'], r['regime_effect']
        L.append(f"{name} & {hp(r['winner'])} & {r['mean']:.3f} & ${c1['mean']:+.3f}$ {ci(c1['t_ci'])} & "
                 f"${c16['mean']:+.3f}$ {ci(c16['t_ci'])} & ${re_['mean']:+.3f}$ {ci(re_['t_ci'])} \\\\")
    L += ['\\bottomrule', '\\end{tabular}', '']
    open(os.path.join(d, 'p3x.tex'), 'w').write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='results/p3_analysis.json')
    ap.add_argument('--tex-dir', default='paper/jmlr/generated')
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    a, b = part_a(rng), part_b(rng)
    x = part_x(np.random.default_rng(1))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({'A': a, 'B': b, 'X': x}, open(args.out, 'w'), indent=1)
    os.makedirs(args.tex_dir, exist_ok=True)
    tex_tables(a, b, args.tex_dir, x)
    r = lambda x: round(x, 4)
    summ = {'A': {n: {'winner': a[n]['winner'], 'edge': a[n]['edge'], 'mean': r(a[n]['mean']), 'sd': r(a[n]['sd']),
                      'agree': r(a[n]['agree']), 'total_decay': r(a[n].get('total_decay', float('nan'))),
                      **({'cwd_minus': {k: a[n]['cwd_minus'][k] for k in ('mean', 't_ci', 'boot_ci', 'p', 'p_holm',
                                                                          'wins', 'confirmed')},
                          'minus_adamw': {k: a[n]['minus_adamw'][k] for k in ('mean', 't_ci', 'p', 'wins')}}
                         if n in ('RandWD', 'AntiCWD') else {})}
                  for n in ('CWD', 'AdamW', 'RandWD', 'AntiCWD')},
            'B': {n: {'winner': b[n]['winner'], 'edge': b[n]['edge'], 'mean': r(b[n]['mean']), 'sd': r(b[n]['sd']),
                      'grid': b[n]['grid_size'], 'p1_r1_mean': r(b[n]['p1_r1']['mean']),
                      **({'r1': {k: b[n]['r1_contrast'][k] for k in ('mean', 't_ci', 'p')},
                          'r16': {k: b[n]['r16_contrast'][k] for k in ('mean', 't_ci')},
                          'regime': b[n]['regime_effect'], 'p1_r1_contrast': b[n]['p1_r1']['contrast']['mean']}
                         if n != 'AdamW' else {})}
                  for n in ('AdamW', 'AdamW+EMA', 'AdamW+dropout')},
            'X': {n: {'winner': x[n]['winner'], 'edge': x[n]['edge'], 'mean': r(x[n]['mean']), 'sd': r(x[n]['sd']),
                      **({'r1': {k: x[n]['r1_contrast'][k] for k in ('mean', 't_ci', 'p', 'wins')},
                          'r16': {k: x[n]['r16_contrast'][k] for k in ('mean', 't_ci', 'p')},
                          'regime': x[n]['regime_effect']} if n != 'AdamW' else {})}
                  for n in ('AdamW', 'AdamW+EMA', 'AdamW+dropout')}}
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
