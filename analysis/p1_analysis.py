"""Pre-registered analysis of protocol P1 (protocols/P1_minibench.md, section "Analysis").

    python analysis/p1_analysis.py                 # reads results/dc/, writes results/p1_analysis.json
                                                   # and results/p1_report.md (English, paper-ready)
    python analysis/p1_analysis.py --results DIR --out-json F --out-md F

Every number comes from the result files written by dc_grid.py; nothing is typed by hand.

Primary: XS, R16, test loss at the best-validation checkpoint, mean over the finals. Each method
is contrasted with AdamW, paired by seed: paired t-test, percentile bootstrap CI95 of the paired
difference (10,000 resamples of the seed tuples), Holm correction within the family of contrasts.
Parity with AdamW is declared only by a paired TOST with a margin of 0.01 nats.
Secondary: the same at the end-of-training checkpoint, the R1 arm, the regime effect per method
= (method - AdamW)_R16 - (method - AdamW)_R1, and the S confirmation.
Also reported per method: winner hyper-parameters and edge flags after the extension, best-validation
epoch, throughput relative to AdamW, and the full tuning grids.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stats_utils import paired_ttest, bootstrap_ci, holm, t_ci, welch_ci  # noqa: E402

MARGIN = 0.01          # TOST margin, nats
N_BOOT = 10_000
ALPHA = 0.05
STANDARD = 'AdamW'
ORDER = ['Baseline', 'AdamW', 'AdamW+dropout', 'AdamW+EMA', 'CWD', 'AdamS', 'CPR', 'SF-AdamW',
         'Muon', 'Muon+Const', 'Const', 'Const-Fair', 'Const-Huber']
ARMS = {                                      # key: (size, file tag regex, label)
    'xs_r16': ('XS', r'XS_D10M_E16_', 'XS, R16 (D = 10M x 16 epochs)'),
    'xs_r1': ('XS', r'XS_D159\.384M_E1_', 'XS, R1 (D = 159.4M x 1 epoch)'),
    's_r16': ('S', r'S_D10M_E16_', 'S, R16 (D = 10M x 16 epochs)'),
}
STEM_RE = re.compile(r'^(?P<size>XS|S|M|tiny)_D(?P<d>[\d.]+)M_E(?P<e>\d+)_(?P<method>.+)\.json$')


# ----------------------------------------------------------------------------- loading
def load_arm(results_dir, tag_re):
    """{method: result dict} for every file whose stem starts with the arm's tag."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        name = os.path.basename(path)
        if not re.match(tag_re, name):
            continue
        m = STEM_RE.match(name)
        if not m:
            continue
        if m.group('method') not in ORDER:      # tagged post-registration stems: see sensitivity()
            continue
        doc = json.load(open(path))
        if doc.get('finals'):
            out[m.group('method')] = doc
    return out


def seeds_of(doc, metric):
    """{seed: value} over the finals of one result file."""
    return {int(s): float(r[metric]) for s, r in doc['finals'].items()}


def paired(a, b):
    """Same-seed-aligned value lists (a, b, seeds)."""
    seeds = sorted(set(a) & set(b))
    return [a[s] for s in seeds], [b[s] for s in seeds], seeds


# ----------------------------------------------------------------------------- statistics
def paired_tost(a, b, margin=MARGIN, alpha=ALPHA):
    """Paired two one-sided tests: H0 'the mean difference is at least `margin` in absolute value'.

    Returns (equivalent, p_tost, ci_low, ci_high) with the (1 - 2 alpha) CI of mean(a - b).
    Equivalence is declared when both one-sided tests reject, i.e. the CI sits inside (-m, +m).
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.size
    if n < 2:
        return False, float('nan'), float('nan'), float('nan')
    mean, sd = d.mean(), d.std(ddof=1)
    if sd == 0:
        eq = abs(mean) < margin
        return bool(eq), 0.0 if eq else 1.0, float(mean), float(mean)
    se, df = sd / math.sqrt(n), n - 1
    p_lower = stats.t.sf((mean + margin) / se, df)     # tests mean > -margin
    p_upper = stats.t.cdf((mean - margin) / se, df)    # tests mean < +margin
    p = float(max(p_lower, p_upper))
    crit = stats.t.ppf(1 - alpha, df)
    return bool(p < alpha), p, float(mean - crit * se), float(mean + crit * se)


def contrasts(arm, metric):
    """Every method of the arm against the standard, paired by seed; Holm within the family."""
    std = seeds_of(arm[STANDARD], metric)
    rows = []
    for m in ORDER:
        if m == STANDARD or m not in arm:
            continue
        a, b, seeds = paired(seeds_of(arm[m], metric), std)
        t, p, diff, n = paired_ttest(a, b)
        lo, hi = bootstrap_ci(a, b, n_boot=N_BOOT, seed=0)
        eq, p_tost, tlo, thi = paired_tost(a, b)
        rows.append(dict(method=m, n_pairs=n, seeds=seeds, mean=float(np.mean(a)),
                         sd=float(np.std(a, ddof=1)) if n > 1 else float('nan'),
                         diff=diff, t=t, p=p, boot_ci=[lo, hi],
                         t_ci=list(t_ci(np.asarray(a) - np.asarray(b))),
                         tost_equivalent=eq, tost_p=p_tost, tost_ci=[tlo, thi]))
    rej = holm([r['p'] for r in rows], alpha=ALPHA) if rows else []
    adj = holm_adjusted([r['p'] for r in rows]) if rows else []
    for r, reject, p_adj in zip(rows, rej, adj):
        r['holm_reject'] = bool(reject)
        r['p_holm'] = float(p_adj)
        r['verdict'] = ('worse' if r['diff'] > 0 else 'better') if r['holm_reject'] else \
                       ('parity' if r['tost_equivalent'] else 'inconclusive')
    return rows


def holm_adjusted(pvals):
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(list(pvals), dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def per_method(arm):
    """Descriptive block per method: winner, edges, best epoch, throughput, overhead vs standard."""
    std_tps = np.mean([r['tokens_per_second'] for r in arm[STANDARD]['finals'].values()]) \
        if STANDARD in arm else float('nan')
    out = {}
    for m, doc in arm.items():
        fin = list(doc['finals'].values())
        tps = float(np.mean([r['tokens_per_second'] for r in fin]))
        out[m] = dict(
            winner=doc['winner']['hp'], edge=[k for k, v in doc['winner'].get('edge', {}).items() if v],
            tune_val_loss=float(doc['winner']['val_loss']),
            test_loss=float(np.mean([r['test_loss'] for r in fin])),
            test_loss_sd=float(np.std([r['test_loss'] for r in fin], ddof=1)) if len(fin) > 1 else float('nan'),
            final_test_loss=float(np.mean([r['final_test_loss'] for r in fin])),
            test_ppl=float(math.exp(np.mean([r['test_loss'] for r in fin]))),
            best_epoch=float(np.mean([r['best_epoch'] for r in fin])),
            epochs=int(doc['protocol']['epochs']), n_finals=len(fin),
            tokens_per_second=tps, time_overhead=float(std_tps / tps - 1.0) if tps else float('nan'),
            train_seconds=float(np.mean([r['train_seconds'] for r in fin])),
            grid=[dict(hp=g['hp'], val_loss=g['val_loss'], best_epoch=g['best_epoch']) for g in doc['grid']])
    return out


def regime_effect(r16, r1, metric='test_loss', n_boot=N_BOOT):
    """(method - AdamW) in R16 minus (method - AdamW) in R1, with a bootstrap CI that resamples the
    seeds of each arm independently (the two arms share no seed pairing)."""
    rng = np.random.default_rng(0)
    out = []
    for m in ORDER:
        if m == STANDARD or m not in r16 or m not in r1:
            continue
        a16, b16, _ = paired(seeds_of(r16[m], metric), seeds_of(r16[STANDARD], metric))
        a1, b1, _ = paired(seeds_of(r1[m], metric), seeds_of(r1[STANDARD], metric))
        d16 = np.asarray(a16) - np.asarray(b16)
        d1 = np.asarray(a1) - np.asarray(b1)
        eff = float(d16.mean() - d1.mean())
        if d16.size >= 2 and d1.size >= 2:
            i16 = rng.integers(0, d16.size, size=(n_boot, d16.size))
            i1 = rng.integers(0, d1.size, size=(n_boot, d1.size))
            boot = d16[i16].mean(axis=1) - d1[i1].mean(axis=1)
            lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
        else:
            lo = hi = float('nan')
        out.append(dict(method=m, diff_r16=float(d16.mean()), diff_r1=float(d1.mean()),
                        regime_effect=eff, boot_ci=[lo, hi], t_ci=list(welch_ci(d16, d1))))
    return out


# ----------------------------------------------------------------------------- report
TAG_RE = re.compile(r'^(?P<arm>(?:XS|S|M)_D[\d.]+M_E\d+)_(?P<method>.+?)_(?P<tag>x6|wdx|audit|tune43|x3|edge|r1s|add2|add|m|cwd)\.json$')
TAGS = {'x6': 'six final seeds (1004-1006 added) under the pre-registered winners',
        'wdx': 'second weight-decay extension of the AdamW grid (wd 7.5), beyond the one the protocol allows',
        'audit': 'tuning points beyond the protocol grid (no finals)',
        'tune43': 'the registered grid tuned again on seed 43 (no finals): variance of the tuning winner',
        'x3': 'a third final seed (1003) in the control arm',
        'edge': 'second extension on the axis where the registered winner still sits on the edge',
        'r1s': 'control arm at S: the R16 winner at S and its weaker variant, two finals',
        'add': 'method outside the protocol (AdEMAMix), tuned on the extended AdamW grid',
        'add2': 'AdEMAMix with alpha and beta3 tuned (reference values diverge at this horizon), wd 2.5',
        'm': 'spot check at 124.5M parameters: S-tuned second axis, learning rate re-tuned, three finals',
        'cwd': 'CWD follow-up after its edge winner (wd 7.5): grid closed with wd 37.5 at XS, search at S'}


def sensitivity(results_dir, loaded):
    """Post-registration runs (tagged stems <arm>_<method>_<tag>.json, written by pod_launch_P1_extra.sh).

    Per tag and arm: every tagged method's winner, grid and finals next to its pre-registered
    counterpart, and, when tagged finals exist, the paired contrast against AdamW (the tagged AdamW of
    the same arm when it has finals, e.g. x6, otherwise the pre-registered one). A tagged AdamW is also
    contrasted with the pre-registered AdamW on the seeds they share (wdx: did the wider grid move it?).
    """
    docs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        m = TAG_RE.match(os.path.basename(path))
        if m:
            docs.setdefault((m['tag'], m['arm']), {})[m['method']] = json.load(open(path))
    arm_key = {v[1].rstrip('_').replace('\\', ''): k for k, v in ARMS.items()}   # 'XS_D10M_E16' -> 'xs_r16'
    out = []
    for (tag, arm), methods in sorted(docs.items()):
        key = arm_key.get(arm)
        pre = loaded.get(key, {})
        entry = dict(tag=tag, arm=arm, label=ARMS[key][2] if key else arm, note=TAGS.get(tag, ''),
                     methods={}, contrasts=[])
        for m, d in methods.items():
            fin = list(d['finals'].values())
            p = pre.get(m)
            entry['methods'][m] = dict(
                winner=d['winner']['hp'], tune_val_loss=float(d['winner']['val_loss']),
                edge=[k for k, v in d['winner'].get('edge', {}).items() if v],
                n_finals=len(fin), seeds=sorted(int(s) for s in d['finals']),
                test_loss=float(np.mean([r['test_loss'] for r in fin])) if fin else None,
                test_loss_sd=float(np.std([r['test_loss'] for r in fin], ddof=1)) if len(fin) > 1 else None,
                pre_registered=(dict(winner=p['winner']['hp'], tune_val_loss=float(p['winner']['val_loss']),
                                     test_loss=float(np.mean([r['test_loss'] for r in p['finals'].values()])),
                                     n_finals=len(p['finals'])) if p else None),
                grid=[dict(hp=g['hp'], val_loss=g['val_loss'], best_epoch=g['best_epoch']) for g in d['grid']])
        with_finals = {m: d for m, d in methods.items() if d.get('finals')}
        std = with_finals.get(STANDARD) or pre.get(STANDARD)
        if std is not None:
            fin = list(std['finals'].values())
            entry['standard'] = 'tagged' if STANDARD in with_finals else 'pre-registered'
            entry['standard_desc'] = dict(
                test_loss=float(np.mean([r['test_loss'] for r in fin])),
                test_loss_sd=float(np.std([r['test_loss'] for r in fin], ddof=1)) if len(fin) > 1 else float('nan'),
                winner=std['winner']['hp'], edge=[k for k, v in std['winner'].get('edge', {}).items() if v],
                n_finals=len(fin))
            others = {m: d for m, d in with_finals.items() if m != STANDARD}
            if others:
                entry['contrasts'] = contrasts({STANDARD: std, **others}, 'test_loss')
        if STANDARD in with_finals and STANDARD in pre:
            a, b, seeds = paired(seeds_of(with_finals[STANDARD], 'test_loss'), seeds_of(pre[STANDARD], 'test_loss'))
            if len(seeds) >= 2:
                t, p, diff, n = paired_ttest(a, b)
                lo, hi = bootstrap_ci(a, b, n_boot=N_BOOT, seed=0)
                entry['standard_vs_pre'] = dict(
                    diff=diff, p=p, boot_ci=[lo, hi], t_ci=list(t_ci(np.asarray(a) - np.asarray(b))),
                    n_pairs=n, seeds=seeds,
                    same_winner=with_finals[STANDARD]['winner']['hp'] == pre[STANDARD]['winner']['hp'])
        out.append(entry)
    return out


def fmt(x, nd=4):
    return 'n/a' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{nd}f}'


def contrast_table(rows, desc, title, metric_label):
    lines = [f'### {title}', '',
             f'Methods ranked by mean {metric_label}; every contrast is paired by seed against AdamW. '
             'p: paired t-test; p(Holm): step-down adjusted within this family; CI95: percentile '
             'bootstrap of the paired difference (10,000 resamples; n is tiny, read the interval, '
             f'not the p-value); TOST: equivalence within +/-{MARGIN} nats (90 % CI inside the margin).', '',
             '| Method | mean | sd | PPL | diff vs AdamW | bootstrap CI95 | p | p(Holm) | TOST | verdict | winner hp | edge |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    std = desc[STANDARD]
    allrows = rows + [dict(method=STANDARD, mean=std['test_loss'], sd=std['test_loss_sd'], diff=0.0)]
    for r in sorted(allrows, key=lambda r: r['mean']):
        m = r['method']
        d = desc[m]
        hp = ', '.join(f'{k}={v:g}' for k, v in d['winner'].items())
        if m == STANDARD:
            lines.append(f"| **{m}** | {fmt(r['mean'])} | {fmt(r['sd'])} | {math.exp(r['mean']):.2f} | "
                         f"— | — | — | — | — | standard | {hp} | {','.join(d['edge']) or '-'} |")
            continue
        lines.append(f"| {m} | {fmt(r['mean'])} | {fmt(r['sd'])} | {math.exp(r['mean']):.2f} | "
                     f"{r['diff']:+.4f} | [{r['boot_ci'][0]:+.4f}, {r['boot_ci'][1]:+.4f}] | "
                     f"{fmt(r['p'], 3)} | {fmt(r['p_holm'], 3)} | "
                     f"{'yes' if r['tost_equivalent'] else 'no'} (p={fmt(r['tost_p'], 3)}) | "
                     f"{r['verdict']} | {hp} | {','.join(d['edge']) or '-'} |")
    return lines + ['']


def descriptive_table(desc, title):
    lines = [f'### {title}', '',
             '| Method | best-val epoch | final-ckpt test loss | tok/s | time overhead vs AdamW | finals |',
             '|---|---|---|---|---|---|']
    for m in sorted(desc, key=lambda m: desc[m]['test_loss']):
        d = desc[m]
        lines.append(f"| {m} | {d['best_epoch']:.2f} / {d['epochs']} | {fmt(d['final_test_loss'])} | "
                     f"{d['tokens_per_second']:,.0f} | {100 * d['time_overhead']:+.1f} % | {d['n_finals']} |")
    return lines + ['']


def grid_tables(desc, title):
    lines = [f'### {title}', '', 'Tuning-seed validation loss at the best checkpoint for every grid point '
             '(the winner is the minimum). Rows: learning rate; columns: the second axis.', '']
    for m in sorted(desc):
        g = desc[m]['grid']
        if not g:
            continue
        axes = [k for k in g[0]['hp'] if k != 'lr' and len({tuple(sorted(x['hp'].items())) for x in g}) > 1
                and len({x['hp'].get(k) for x in g}) > 1]
        # every non-lr axis that varies is a column dimension (AdEMAMix-T varies two)
        key = lambda x: tuple(x['hp'][k] for k in axes)
        lrs = sorted({x['hp']['lr'] for x in g})
        cols = sorted({key(x) for x in g}) if axes else [None]
        lab = lambda c: ', '.join(f'{v:g}' for v in c) if c is not None else 'val loss'
        lines.append(f'**{m}**' + (f" (columns: {', '.join(axes)})" if axes else ''))
        lines.append('')
        lines.append('| lr | ' + ' | '.join(lab(c) for c in cols) + ' |')
        lines.append('|---|' + '---|' * len(cols))
        for lr in lrs:
            cells = []
            for c in cols:
                hit = [x for x in g if x['hp']['lr'] == lr and (not axes or key(x) == c)]
                cells.append(f"{hit[0]['val_loss']:.4f} (ep {hit[0]['best_epoch']:g})" if hit else '')
            lines.append(f'| {lr:g} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    return lines


def build_report(res):
    L = ['# Protocol P1 — pre-registered analysis', '',
         f'Source: `results/dc/*.json` (dc_grid.py); margin for parity {MARGIN} nats; alpha {ALPHA}; '
         f'bootstrap {N_BOOT:,} resamples. Generated by `analysis/p1_analysis.py`.', '']
    arms = res['arms']
    if 'xs_r16' in arms:
        a = arms['xs_r16']
        L += ['## Primary outcome', ''] + contrast_table(a['contrasts_best'], a['methods'],
                                                         ARMS['xs_r16'][2] + ' — best-validation checkpoint',
                                                         'test loss at the best-validation checkpoint')
        L += ['## Secondary outcomes', ''] + contrast_table(a['contrasts_final'], a['methods'],
                                                            ARMS['xs_r16'][2] + ' — end-of-training checkpoint',
                                                            'test loss at the end of training')
    if 'xs_r1' in arms:
        a = arms['xs_r1']
        L += contrast_table(a['contrasts_best'], a['methods'], ARMS['xs_r1'][2] + ' — best-validation checkpoint',
                            'test loss at the best-validation checkpoint')
    if res.get('regime_effect'):
        L += ['### Regime effect: (method − AdamW) in R16 minus (method − AdamW) in R1', '',
              'Positive = the method loses ground relative to AdamW when the data are repeated. '
              'CI95: bootstrap over the seeds of each arm independently.', '',
              '| Method | diff R16 | diff R1 | regime effect | CI95 |', '|---|---|---|---|---|']
        for r in sorted(res['regime_effect'], key=lambda r: r['regime_effect']):
            L.append(f"| {r['method']} | {r['diff_r16']:+.4f} | {r['diff_r1']:+.4f} | {r['regime_effect']:+.4f} | "
                     f"[{r['boot_ci'][0]:+.4f}, {r['boot_ci'][1]:+.4f}] |")
        L.append('')
    if 's_r16' in arms:
        a = arms['s_r16']
        L += contrast_table(a['contrasts_best'], a['methods'], ARMS['s_r16'][2] + ' — confirmation at 51.5M params',
                            'test loss at the best-validation checkpoint')
    if res.get('sensitivity'):
        L += ['## Sensitivity analyses (post-registration)', '',
              'Runs added after the pre-registered matrix (tagged result files, `pod_launch_P1_extra.sh`). '
              'They never replace the tables above: each tagged search starts from the registered journal, '
              'so it differs from its registered counterpart only by the added grid points or seeds.', '']
        for e in res['sensitivity']:
            L += [f"### {e['label']} — `{e['tag']}`: {e['note']}", '']
            for m, d in sorted(e['methods'].items()):
                hp = ', '.join(f'{k}={v:g}' for k, v in d['winner'].items())
                line = (f"- **{m}**: winner {hp} (tuning val {d['tune_val_loss']:.4f}; "
                        f"edge {','.join(d['edge']) or '-'})")
                if d['n_finals']:
                    line += f"; finals n={d['n_finals']}, test {d['test_loss']:.4f} ± {fmt(d['test_loss_sd'])}"
                p = d.get('pre_registered')
                if p:
                    php = ', '.join(f'{k}={v:g}' for k, v in p['winner'].items())
                    line += (f" — pre-registered: winner {php} (val {p['tune_val_loss']:.4f}), "
                             f"test {p['test_loss']:.4f} (n={p['n_finals']})")
                L.append(line)
            L.append('')
            if e.get('standard_vs_pre'):
                s = e['standard_vs_pre']
                L += [f"Tagged AdamW vs pre-registered AdamW, paired on seeds {s['seeds']}: diff {s['diff']:+.4f}, "
                      f"bootstrap CI95 [{s['boot_ci'][0]:+.4f}, {s['boot_ci'][1]:+.4f}], p={fmt(s['p'], 3)}"
                      + (' (same winner: the finals are the registered ones).' if s['same_winner']
                         else ' (the winner moved).'), '']
            if e['contrasts']:
                desc = {m: dict(test_loss=d['test_loss'], test_loss_sd=d['test_loss_sd'], winner=d['winner'],
                                edge=d['edge']) for m, d in e['methods'].items() if d['n_finals']}
                desc[STANDARD] = e['standard_desc']
                L += contrast_table(e['contrasts'], desc,
                                    f"{e['label']} — `{e['tag']}`: contrasts vs {e['standard']} AdamW "
                                    f"(n = {e['standard_desc']['n_finals']})",
                                    'test loss at the best-validation checkpoint')
            grids = {m: dict(grid=d['grid']) for m, d in e['methods'].items() if d['grid']}
            if grids:
                L += grid_tables(grids, f"{e['label']} — `{e['tag']}`: tuning grids")
    L += ['## Descriptives', '']
    for key in ('xs_r16', 'xs_r1', 's_r16'):
        if key in arms:
            L += descriptive_table(arms[key]['methods'], ARMS[key][2])
    L += ['## Appendix: tuning grids', '']
    for key in ('xs_r16', 'xs_r1', 's_r16'):
        if key in arms:
            L += grid_tables(arms[key]['methods'], ARMS[key][2])
    return '\n'.join(L)


def analyse(results_dir):
    res = {'arms': {}, 'margin': MARGIN, 'alpha': ALPHA, 'n_boot': N_BOOT}
    loaded = {}
    for key, (size, tag, label) in ARMS.items():
        arm = load_arm(results_dir, tag)
        if STANDARD not in arm or len(arm) < 2:
            continue
        loaded[key] = arm
        res['arms'][key] = dict(label=label, methods=per_method(arm),
                                contrasts_best=contrasts(arm, 'test_loss'),
                                contrasts_final=contrasts(arm, 'final_test_loss'))
    if 'xs_r16' in loaded and 'xs_r1' in loaded:
        res['regime_effect'] = regime_effect(loaded['xs_r16'], loaded['xs_r1'])
    res['sensitivity'] = sensitivity(results_dir, loaded)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', default='results/dc')
    ap.add_argument('--out-json', default='results/p1_analysis.json')
    ap.add_argument('--out-md', default='results/p1_report.md')
    args = ap.parse_args()
    res = analyse(args.results)
    if not res['arms']:
        sys.exit(f'no arm with AdamW and at least one other method under {args.results}')
    report = build_report(res)
    os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
    json.dump(res, open(args.out_json, 'w'), indent=1)
    open(args.out_md, 'w').write(report + '\n')
    print(report)
    print(f'\n[written] {args.out_json}  {args.out_md}')


if __name__ == '__main__':
    main()
