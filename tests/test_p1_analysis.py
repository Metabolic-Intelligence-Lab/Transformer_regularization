"""CPU tests of analysis/p1_analysis.py on a synthetic results directory."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import p1_analysis as pa          # noqa: E402
from stats_utils import holm      # noqa: E402


def _doc(method, seeds_loss, grid=None, epochs=16, tps=1000.0):
    finals = {str(s): dict(val_loss=l, test_loss=l, test_ppl=math.exp(l), final_val_loss=l + 0.1,
                           final_test_loss=l + 0.1, final_test_ppl=math.exp(l + 0.1), best_step=100,
                           best_epoch=10.0, total_steps=160, tokens=1, params=1, train_seconds=10.0,
                           tokens_per_second=tps, curve=[]) for s, l in seeds_loss.items()}
    grid = grid or [dict(hp={'lr': 1e-3, 'wd': 0.1}, val_loss=1.0, best_epoch=8.0),
                    dict(hp={'lr': 3e-3, 'wd': 0.1}, val_loss=0.9, best_epoch=9.0)]
    return dict(stem='x', method=method, protocol=dict(epochs=epochs), tune_seed=42, grid=grid,
                winner=dict(hp=grid[-1]['hp'], val_loss=grid[-1]['val_loss'], edge={'lr': True, 'wd': False}),
                finals=finals, data_meta={})


def _write(tmp, tag, method, seeds_loss, **kw):
    json.dump(_doc(method, seeds_loss, **kw), open(os.path.join(tmp, f'{tag}{method}.json'), 'w'))


def test_holm_matches_hand_computation():
    # m = 4: sorted 0.001, 0.012, 0.03, 0.2 vs 0.0125, 0.0167, 0.025, 0.05 -> first two rejected
    rej = holm([0.03, 0.001, 0.2, 0.012])
    assert rej.tolist() == [False, True, False, True]
    assert holm([]).size == 0
    assert holm([0.04]).tolist() == [True]          # single test: plain alpha


def test_paired_tost_declares_parity_only_inside_margin():
    eq, p, lo, hi = pa.paired_tost([1.000, 1.001, 0.999], [1.000, 1.000, 1.000], margin=0.01)
    assert eq and p < 0.05 and -0.01 < lo < hi < 0.01
    eq, p, lo, hi = pa.paired_tost([1.05, 1.06, 1.04], [1.0, 1.0, 1.0], margin=0.01)
    assert not eq and p > 0.05
    eq, _, _, _ = pa.paired_tost([1.0, 1.0], [1.0, 1.0], margin=0.01)   # zero variance, inside
    assert eq


def test_analysis_end_to_end(tmp_path):
    tmp = str(tmp_path)
    r16, r1 = 'XS_D10M_E16_', 'XS_D159.384M_E1_'
    _write(tmp, r16, 'AdamW', {1001: 4.50, 1002: 4.52, 1003: 4.48}, tps=1000.0)
    _write(tmp, r16, 'Const', {1001: 4.70, 1002: 4.71, 1003: 4.69}, tps=800.0)     # clearly worse
    _write(tmp, r16, 'CWD', {1001: 4.501, 1002: 4.521, 1003: 4.479}, tps=1000.0)  # parity
    _write(tmp, r16, 'Muon', {1001: 4.30, 1002: 4.33, 1003: 4.31})                 # clearly better
    _write(tmp, r1, 'AdamW', {1001: 4.10, 1002: 4.12})
    _write(tmp, r1, 'Const', {1001: 4.20, 1002: 4.22})
    _write(tmp, r1, 'CWD', {1001: 4.10, 1002: 4.12})
    # post-registration (tagged) files: excluded from the arms, reported under sensitivity
    _write(tmp, r16, 'AdamW_x6', {1001: 4.50, 1002: 4.52, 1003: 4.48, 1004: 4.51, 1005: 4.49, 1006: 4.50})
    _write(tmp, r16, 'Muon_x6', {1001: 4.30, 1002: 4.33, 1003: 4.31, 1004: 4.32, 1005: 4.30, 1006: 4.31})
    _write(tmp, r16, 'AdamW_wdx', {1001: 4.45, 1002: 4.47, 1003: 4.43},
           grid=[dict(hp={'lr': 1e-3, 'wd': 0.1}, val_loss=1.0, best_epoch=8.0),
                 dict(hp={'lr': 3e-3, 'wd': 0.5}, val_loss=0.85, best_epoch=9.0)])
    _write(tmp, r16, 'SF-AdamW_audit', {}, grid=[dict(hp={'lr': 1e-2, 'wd': 2.5}, val_loss=4.9, best_epoch=16.0)])
    res = pa.analyse(tmp)
    assert set(res['arms']) == {'xs_r16', 'xs_r1'}
    assert set(res['arms']['xs_r16']['methods']) == {'AdamW', 'Const', 'CWD', 'Muon'}
    sens = {e['tag']: e for e in res['sensitivity']}
    assert set(sens) == {'x6', 'wdx', 'audit'}
    x6 = sens['x6']
    assert x6['standard'] == 'tagged' and x6['standard_desc']['n_finals'] == 6
    assert x6['contrasts'][0]['method'] == 'Muon' and x6['contrasts'][0]['n_pairs'] == 6
    assert x6['standard_vs_pre']['same_winner'] and abs(x6['standard_vs_pre']['diff']) < 1e-12
    wdx = sens['wdx']
    assert not wdx['contrasts'] and not wdx['standard_vs_pre']['same_winner']
    assert abs(wdx['standard_vs_pre']['diff'] + 0.05) < 1e-9
    assert wdx['methods']['AdamW']['pre_registered']['winner'] == {'lr': 3e-3, 'wd': 0.1}
    assert sens['audit']['methods']['SF-AdamW']['n_finals'] == 0 and 'standard' in sens['audit']
    by = {r['method']: r for r in res['arms']['xs_r16']['contrasts_best']}
    assert by['Const']['verdict'] == 'worse' and by['Const']['holm_reject']
    assert by['Muon']['verdict'] == 'better'
    assert by['CWD']['verdict'] == 'parity' and by['CWD']['tost_equivalent']
    assert by['Const']['n_pairs'] == 3 and by['Const']['seeds'] == [1001, 1002, 1003]
    assert by['Const']['boot_ci'][0] > 0                       # the whole interval above zero
    # Holm-adjusted p never below the raw p and monotone in the raw p
    raw = sorted(by.values(), key=lambda r: r['p'])
    assert all(a['p_holm'] <= b['p_holm'] + 1e-12 for a, b in zip(raw, raw[1:]))
    assert all(r['p_holm'] >= r['p'] for r in raw)
    desc = res['arms']['xs_r16']['methods']
    assert abs(desc['Const']['time_overhead'] - 0.25) < 1e-9        # 1000/800 - 1
    assert desc['AdamW']['edge'] == ['lr']
    # regime effect: Const +0.20 in R16, +0.10 in R1 -> +0.10; CWD ~0 - 0 -> ~0
    reg = {r['method']: r for r in res['regime_effect']}
    assert abs(reg['Const']['regime_effect'] - 0.10) < 1e-6
    assert abs(reg['CWD']['regime_effect']) < 1e-3
    assert 'Muon' not in reg                                       # no R1 file for Muon
    report = pa.build_report(res)
    assert '## Primary outcome' in report and '| **AdamW** |' in report and 'Regime effect' in report
    assert 'Appendix: tuning grids' in report
    assert 'Sensitivity analyses' in report and '`x6`: contrasts vs tagged AdamW (n = 6)' in report
    assert 'the winner moved' in report
