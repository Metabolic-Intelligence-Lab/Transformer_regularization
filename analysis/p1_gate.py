"""Gate G1 of protocol P1 (protocols/P1_minibench.md): verdict of the pilot runs, plus the cost
projection of the full study from the measured throughput.

    python analysis/p1_gate.py                     # reads results/dc/, writes results/p1_gate.json
    python analysis/p1_gate.py --rate 0.5          # $/GPU-h for the money column
    python analysis/p1_gate.py --d 5               # the D = 5M fallback of the protocol

Rules, fixed in the protocol before the pilot ran:
  1. XS Baseline, R16 arm: best-validation epoch <= 60 % of the budget.
  2. XS Baseline, R1 arm:  best-validation checkpoint at >= 90 % of the single epoch.
  3. Projected GPU-hours of the full study <= 80.
"""
import argparse
import json
import os

RULES = {'r16_best_epoch_max_frac': 0.6, 'r1_best_epoch_min_frac': 0.9, 'gpu_h_max': 80.0}
# Run counts of the full study (protocol, section "Budget").
COUNTS = {'xs_r16': 114, 'xs_edge_reserve': 15, 'xs_r1': 51, 's_r16': 50}
OVERHEAD = 1.15      # evaluation, compilation and data loading on top of train_seconds


def stems(d_millions):
    """Result stems of the three pilot runs for a given R16 corpus size (dc_grid naming)."""
    d = int(d_millions * 1e6)
    steps = ((d - 1) // 1024 // 64) * 16                  # R16 steps = R1 steps
    d_ctrl = steps * 64 * 1024 + 1
    tag16, tag1 = f"D{d / 1e6:g}M", f"D{d_ctrl / 1e6:g}M"
    return {'xs_r16': f'XS_{tag16}_E16_Baseline', 'xs_r1': f'XS_{tag1}_E1_Baseline',
            's_r16': f'S_{tag16}_E16_Baseline'}


def load_pilot(results_dir, d_millions=10):
    out = {}
    for key, stem in stems(d_millions).items():
        path = os.path.join(results_dir, stem + '.json')
        if not os.path.exists(path):
            out[key] = None
            continue
        doc = json.load(open(path))
        g = doc['grid'][0]
        out[key] = {'stem': stem, 'best_epoch': g['best_epoch'], 'epochs': doc['protocol']['epochs'],
                    'train_seconds': g['train_seconds'], 'tokens_per_second': g['tokens_per_second'],
                    'val_loss': g['val_loss'], 'final_val_loss': g['final_val_loss'],
                    'test_ppl': g['test_ppl'], 'params': g['params']}
    return out


def evaluate(pilot, rate=None):
    r16, r1, s = pilot.get('xs_r16'), pilot.get('xs_r1'), pilot.get('s_r16')
    checks = {
        'r16_overfits_in_budget': bool(r16) and r16['best_epoch'] <= RULES['r16_best_epoch_max_frac'] * r16['epochs'],
        'r1_still_improving': bool(r1) and r1['best_epoch'] >= RULES['r1_best_epoch_min_frac'] * r1['epochs'],
    }
    xs = [x for x in (r16, r1) if x]
    proj = None
    if xs and s:
        t_xs = sum(x['train_seconds'] for x in xs) / len(xs) * OVERHEAD
        t_s = s['train_seconds'] * OVERHEAD
        xs_runs = COUNTS['xs_r16'] + COUNTS['xs_edge_reserve'] + COUNTS['xs_r1']
        gpu_h = (xs_runs * t_xs + COUNTS['s_r16'] * t_s) / 3600
        proj = {'xs_run_h': t_xs / 3600, 's_run_h': t_s / 3600, 'xs_runs': xs_runs,
                's_runs': COUNTS['s_r16'], 'gpu_h': gpu_h}
        if rate:
            proj['usd'] = gpu_h * rate
    checks['cost_within_budget'] = proj is not None and proj['gpu_h'] <= RULES['gpu_h_max']
    return {'rules': RULES, 'counts': COUNTS, 'pilot': pilot, 'checks': checks,
            'projection': proj, 'G1': 'PASS' if all(checks.values()) else 'FAIL'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--results', default='results/dc')
    ap.add_argument('--d', type=float, default=10, help='R16 unique tokens in millions (10, or 5 for the fallback)')
    ap.add_argument('--rate', type=float, default=None, help='$/GPU-h')
    ap.add_argument('--out', default='results/p1_gate.json')
    args = ap.parse_args(argv)
    verdict = evaluate(load_pilot(args.results, args.d), args.rate)
    for key, p in verdict['pilot'].items():
        if p is None:
            print(f"{key:7s} MISSING ({stems(args.d)[key]}.json)")
        else:
            print(f"{key:7s} {p['stem']}: best epoch {p['best_epoch']:.2f}/{p['epochs']}  "
                  f"val {p['val_loss']:.4f} (final {p['final_val_loss']:.4f})  test ppl {p['test_ppl']:.2f}  "
                  f"{p['tokens_per_second']:,.0f} tok/s  {p['train_seconds'] / 60:.1f} min")
    for k, v in verdict['checks'].items():
        print(f"  {k:24s} {'PASS' if v else 'FAIL'}")
    if verdict['projection']:
        pr = verdict['projection']
        print(f"  projection: XS run {pr['xs_run_h'] * 60:.1f} min x {pr['xs_runs']}, S run {pr['s_run_h'] * 60:.1f} min "
              f"x {pr['s_runs']} -> {pr['gpu_h']:.1f} GPU-h" + (f" ≈ {pr['usd']:.0f} $" if 'usd' in pr else ''))
    print(f"G1 = {verdict['G1']}")
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(verdict, fh, indent=1)
    return verdict


if __name__ == '__main__':
    main()
