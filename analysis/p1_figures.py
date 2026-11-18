"""Figures of the benchmark paper, from results/p1_analysis.json and results/dc/*.json.

    python analysis/p1_figures.py [--analysis results/p1_analysis.json] [--results results/dc]
                                  [--replicate results/dc_replicate_S_A4500] [--out paper/figures]

F2  paired contrasts with AdamW in R16 and R1 (XS), with t CI95
F3  regime effect, (method - AdamW in R16) - (method - AdamW in R1), with CI95
F4  validation-loss curves of the final runs in R16 (XS and S), best checkpoint marked
F5  contrasts with AdamW at S (R16), with CI95
F6  two-host replicate of the S phase, per method and seed
Writes <name>.pdf and <name>.png. Colours: a validated categorical palette, fixed order.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e3e2de'
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']   # slots 1-5, validated order
COLOR = {'AdamW': SERIES[0], 'AdamW+EMA': SERIES[1], 'Muon': SERIES[2], 'Const': SERIES[3],
         'Muon+Const': SERIES[4]}                              # colour follows the method
ORDER = ['Baseline', 'AdamW', 'AdamW+dropout', 'AdamW+EMA', 'CWD', 'AdamS', 'CPR', 'SF-AdamW',
         'Muon', 'Muon+Const', 'Const', 'Const-Fair', 'Const-Huber']

plt.rcParams.update({
    'font.size': 8.5, 'font.family': 'sans-serif', 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'legend.frameon': False, 'pdf.fonttype': 42, 'savefig.dpi': 300,
})


def tidy(ax, xgrid=True, ygrid=False):
    ax.grid(axis='x' if xgrid and not ygrid else ('y' if ygrid and not xgrid else 'both'),
            color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), bbox_inches='tight')
    plt.close(fig)
    print(f'[figure] {name}')


def contrasts_by_method(arm):
    return {r['method']: r for r in arm['contrasts_best']}


# ----------------------------------------------------------------------------- F1
def fig_protocol(out):
    """Schematic of the two arms (identical compute) and of the tuning pipeline."""
    fig = plt.figure(figsize=(6.4, 3.3))
    a1 = fig.add_axes([0.30, 0.62, 0.66, 0.30])
    a2 = fig.add_axes([0.01, 0.02, 0.98, 0.36])
    total = 159.4
    a1.set_xlim(0, total)
    a1.set_ylim(0, 2)
    for k in range(16):
        a1.barh(1.4, total / 16, left=k * total / 16, height=0.5, color=SERIES[0], edgecolor='white', linewidth=1.0)
    a1.barh(0.5, total, left=0, height=0.5, color=SERIES[1], edgecolor='white', linewidth=1.0)
    a1.text(-3, 1.4, 'R16: 10M unique tokens, 16 epochs', ha='right', va='center')
    a1.text(-3, 0.5, 'R1: 159.4M unique tokens, 1 epoch', ha='right', va='center')
    a1.set_yticks([])
    a1.spines['left'].set_visible(False)
    a1.set_xlabel('Training tokens seen (millions); 2,432 steps in both arms')
    a2.axis('off')
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1)
    steps = [('Tune', 'seed 42; learning rate\nx second axis'),
             ('Edge rule', 'extend an axis once\nif the winner sits on it'),
             ('Final runs', 'winner on seeds\n1001-1003, shared'),
             ('Analysis', 'paired vs AdamW; CI,\nHolm, TOST, regime')]
    w, gap = 0.21, 0.045
    x0 = (1 - (4 * w + 3 * gap)) / 2
    for i, (t, d) in enumerate(steps):
        x = x0 + i * (w + gap)
        a2.add_patch(plt.Rectangle((x, 0.08), w, 0.84, facecolor='white', edgecolor=INK2, linewidth=0.8))
        a2.text(x + w / 2, 0.70, t, ha='center', va='center', fontweight='bold')
        a2.text(x + w / 2, 0.36, d, ha='center', va='center', color=INK2, fontsize=7.5)
        if i < 3:
            a2.annotate('', xy=(x + w + gap - 0.005, 0.5), xytext=(x + w + 0.005, 0.5),
                        arrowprops=dict(arrowstyle='->', color=INK2, lw=0.8))
    save(fig, out, 'F1_protocol')


# ----------------------------------------------------------------------------- F7
def best_val_by(grid, axis):
    best = {}
    for g in grid:
        v = g['hp'][axis]
        best[v] = min(best.get(v, np.inf), g['val_loss'])
    xs = sorted(best)
    return xs, [best[x] for x in xs]


def fig_cwd(results_dir, out):
    """Tuning-seed validation loss (best over learning rates) against weight decay: AdamW and CWD."""
    cwd_color = INK2
    panels = [('XS, 30M parameters', 'XS_D10M_E16'), ('S, 51.5M parameters', 'S_D10M_E16')]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=False)
    for ax, (title, arm) in zip(axes, panels):
        found = False
        for m, color, marker, cands in [
                ('AdamW', SERIES[0], 'o', [f'{arm}_AdamW_edge', f'{arm}_AdamW_wdx', f'{arm}_AdamW']),
                ('CWD', cwd_color, 's', [f'{arm}_CWD_cwd', f'{arm}_CWD_edge', f'{arm}_CWD'])]:
            path = next((os.path.join(results_dir, c + '.json') for c in cands
                         if os.path.exists(os.path.join(results_dir, c + '.json'))), None)
            if path is None:
                continue
            xs, ys = best_val_by(json.load(open(path))['grid'], 'wd')
            ax.plot(xs, ys, color=color, marker=marker, markersize=4.5, linewidth=1.6, label=m)
            found = True
        ax.set_xscale('log')
        ax.set_title(title, loc='left', fontsize=8.5)
        ax.set_xlabel('Weight decay (wd)')
        tidy(ax, xgrid=False, ygrid=True)
        if found:
            ax.legend(loc='lower left')
        else:
            ax.text(0.5, 0.5, 'not run yet', transform=ax.transAxes, ha='center', color=INK2)
    axes[0].set_ylabel('Validation loss, tuning seed\n(best over learning rates)')
    fig.tight_layout()
    save(fig, out, 'F7_cwd_wd')


# ----------------------------------------------------------------------------- F2
def fig_contrasts_two_arms(res, out):
    r16 = contrasts_by_method(res['arms']['xs_r16'])
    r1 = contrasts_by_method(res['arms']['xs_r1'])
    methods = [m for m in sorted(r16, key=lambda m: r16[m]['diff'])]
    y = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.axvline(0, color=INK2, linewidth=0.8)
    for off, arm, label, col in ((0.17, r16, 'R16: 10M tokens x 16 epochs', SERIES[0]),
                                 (-0.17, r1, 'R1: 159M tokens x 1 epoch', SERIES[1])):
        for i, m in enumerate(methods):
            if m not in arm:
                continue
            lo, hi = arm[m]['t_ci']
            ax.plot([lo, hi], [y[i] + off] * 2, color=col, linewidth=1.6, solid_capstyle='butt')
            ax.plot(arm[m]['diff'], y[i] + off, 'o', color=col, markersize=5.5,
                    markeredgecolor='white', markeredgewidth=0.8)
        ax.plot([], [], 'o-', color=col, label=label, markersize=5.5, linewidth=1.6)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.invert_yaxis()
    ax.set_xlabel('Test loss difference from AdamW (nats), best-validation checkpoint')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2, handlelength=1.6)
    tidy(ax)
    ax.text(0.0, 1.02, 'XS, 30M parameters. Bars: t CI95 of the paired difference.',
            transform=ax.transAxes, color=INK2, fontsize=7.5)
    save(fig, out, 'F2_contrasts_r16_r1')


# ----------------------------------------------------------------------------- F3
def fig_regime_effect(res, out):
    rows = sorted(res['regime_effect'], key=lambda r: r['regime_effect'])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axvline(0, color=INK2, linewidth=0.8)
    for i, r in enumerate(rows):
        lo, hi = r['t_ci']
        ax.plot([lo, hi], [i, i], color=SERIES[0], linewidth=1.6, solid_capstyle='butt')
        ax.plot(r['regime_effect'], i, 'o', color=SERIES[0], markersize=5.5,
                markeredgecolor='white', markeredgewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r['method'] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel('Regime effect (nats): (method − AdamW in R16) − (method − AdamW in R1)')
    tidy(ax)
    ax.text(0.0, 1.02, 'Positive: the method loses ground to AdamW when the data are repeated. '
            'Bars: t CI95.', transform=ax.transAxes, color=INK2, fontsize=7.5)
    save(fig, out, 'F3_regime_effect')


# ----------------------------------------------------------------------------- F4
def mean_curve(doc):
    """Mean validation loss over the final seeds at each evaluation; the best checkpoint of the mean."""
    curves = [f['curve'] for f in doc['finals'].values() if f.get('curve')]
    n = min(len(c) for c in curves)
    ep = np.array([p['epoch'] for p in curves[0][:n]])
    val = np.mean([[p['val_loss'] for p in c[:n]] for c in curves], axis=0)
    return ep, val


def fig_curves(results_dir, out):
    panels = [('XS_D10M_E16_', 'XS, 30M parameters', ['AdamW', 'AdamW+EMA', 'Muon', 'Const']),
              ('S_D10M_E16_', 'S, 51.5M parameters', ['AdamW', 'AdamW+EMA', 'Muon', 'Muon+Const'])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
    for ax, (prefix, title, methods) in zip(axes, panels):
        for m in methods:
            path = os.path.join(results_dir, f'{prefix}{m}.json')
            if not os.path.exists(path):
                continue
            ep, val = mean_curve(json.load(open(path)))
            ax.plot(ep, val, color=COLOR[m], linewidth=1.6, label=m)
            j = int(np.argmin(val))
            ax.plot(ep[j], val[j], 'o', color=COLOR[m], markersize=5.5, markeredgecolor='white',
                    markeredgewidth=0.8)
        ax.set_xlabel('Epoch over the 10M-token corpus')
        ax.set_title(title, loc='left', fontsize=8.5, color=INK)
        ax.set_xlim(0, 16.5)
        ax.set_ylim(4.2, 6.0)
        ax.legend(loc='upper right', handlelength=1.6, fontsize=7.5)
        tidy(ax, xgrid=False, ygrid=True)
    axes[0].set_ylabel('Validation loss (nats)')
    fig.suptitle('Mean over the final seeds; dots mark the best checkpoint of the mean curve.',
                 x=0.01, ha='left', color=INK2, fontsize=7.5)
    fig.tight_layout()
    save(fig, out, 'F4_curves_r16')


# ----------------------------------------------------------------------------- F5
def fig_s_contrasts(res, out):
    if 's_r16' not in res['arms']:
        return
    rows = sorted(res['arms']['s_r16']['contrasts_best'], key=lambda r: r['diff'])
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    ax.axvline(0, color=INK2, linewidth=0.8)
    for i, r in enumerate(rows):
        lo, hi = r['t_ci']
        ax.plot([lo, hi], [i, i], color=SERIES[0], linewidth=1.6, solid_capstyle='butt')
        ax.plot(r['diff'], i, 'o', color=SERIES[0], markersize=5.5, markeredgecolor='white',
                markeredgewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r['method'] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel('Test loss difference from AdamW (nats), S, R16, best-validation checkpoint')
    tidy(ax)
    ax.text(0.0, 1.04, 'S, 51.5M parameters, n = 3 seeds. Bars: t CI95.',
            transform=ax.transAxes, color=INK2, fontsize=7.5)
    save(fig, out, 'F5_s_contrasts')


# ----------------------------------------------------------------------------- F6
def fig_replicate(results_dir, replicate_dir, out):
    pts = []
    for path in sorted(glob.glob(os.path.join(replicate_dir, 'S_D10M_E16_*.json'))):
        name = os.path.basename(path)
        main = os.path.join(results_dir, name)
        if not os.path.exists(main):
            continue
        a, b = json.load(open(path)), json.load(open(main))
        m = a['method']
        for s in sorted(set(a['finals']) & set(b['finals'])):
            pts.append((m, a['finals'][s]['test_loss'], b['finals'][s]['test_loss'],
                        a['winner']['hp'] == b['winner']['hp']))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    lo = min(min(p[1], p[2]) for p in pts) - 0.02
    hi = max(max(p[1], p[2]) for p in pts) + 0.02
    ax.plot([lo, hi], [lo, hi], color=GRID, linewidth=1.0, zorder=0)
    for m, x, yv, same in pts:
        ax.plot(x, yv, 'o' if same else 's', color=SERIES[0] if same else SERIES[1], markersize=6,
                markeredgecolor='white', markeredgewidth=0.8)
    offsets = {'AdamW': (9, -11), 'Muon+Const': (9, 7), 'Muon': (9, -3), 'AdamW+EMA': (9, -11),
               'AdamS': (-46, -16)}
    for m in dict.fromkeys(p[0] for p in pts):
        xs = [p[1] for p in pts if p[0] == m]
        ys = [p[2] for p in pts if p[0] == m]
        ax.annotate(m, (np.mean(xs), np.mean(ys)), xytext=offsets.get(m, (7, -2)),
                    textcoords='offset points', fontsize=7.5, color=INK)
    ax.plot([], [], 'o', color=SERIES[0], label='same tuning winner on both hosts', markersize=6)
    ax.plot([], [], 's', color=SERIES[1], label='tuning winner differed', markersize=6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel('Test loss, first run (RTX A4500)')
    ax.set_ylabel('Test loss, second run (RTX 3090)')
    ax.legend(loc='upper left', fontsize=7.5, handlelength=1.2)
    tidy(ax, xgrid=True, ygrid=True)
    ax.text(0.0, 1.02, 'S phase, one point per method and seed.', transform=ax.transAxes,
            color=INK2, fontsize=7.5)
    save(fig, out, 'F6_replicate')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--analysis', default='results/p1_analysis.json')
    ap.add_argument('--results', default='results/dc')
    ap.add_argument('--replicate', default='results/dc_replicate_S_A4500')
    ap.add_argument('--out', default='paper/figures')
    args = ap.parse_args()
    res = json.load(open(args.analysis))
    fig_contrasts_two_arms(res, args.out)
    fig_regime_effect(res, args.out)
    fig_curves(args.results, args.out)
    fig_s_contrasts(res, args.out)
    fig_replicate(args.results, args.replicate, args.out)
    fig_protocol(args.out)
    fig_cwd(args.results, args.out)


if __name__ == '__main__':
    main()
