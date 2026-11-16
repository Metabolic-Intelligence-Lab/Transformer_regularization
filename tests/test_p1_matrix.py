"""Tests of the P1 matrix driver (p1_matrix.py): the protocol table, the edge rule, the
stacking methods and the control-arm variants, with a fake runner. CPU, instant."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import p1_matrix as P  # noqa: E402
from dcstudy.methods import METHODS  # noqa: E402


def test_protocol_table_matches_the_method_registry():
    assert set(P.SPEC) <= set(METHODS)
    for m, spec in P.SPEC.items():
        tuned = set(METHODS[m]['hp'])
        covered = set(spec.get('fixed', {})) | ({spec['axis'][0]} if spec['axis'] else set())
        if 'inherit' in spec:
            covered.add(spec['inherit'][1])
        assert tuned <= covered, f"{m}: tuned keys {tuned} not all covered by {covered}"
        assert len(spec['lrs']) == 3 and (spec['axis'] is None or len(spec['axis'][1]) == 2)
    assert P.ORDER.index('AdamW') < P.ORDER.index('AdamW+dropout') < P.ORDER.index('AdamW+EMA')
    assert len(P.SPEC) == 13


def test_r16_and_r1_token_budgets_are_identical():
    steps16 = ((P.D_R16 - 1) // 1024 // 64) * P.E_R16
    steps1 = ((P.D_R1 - 1) // 1024 // 64) * P.E_R1
    assert steps16 == steps1 == 2432
    assert P.stem('XS', P.D_R1, P.E_R1, 'Baseline') == 'XS_D159.384M_E1_Baseline'


def test_edge_rule_values():
    assert P.extend_value('AdamW', 'lr', 4.5e-3, 'high') == pytest.approx(1.35e-2)
    assert P.extend_value('AdamW', 'wd', 0.5, 'high') == pytest.approx(2.5)
    assert P.extend_value('AdamS', 'wd', 1e-4, 'low') == pytest.approx(1e-5)
    assert P.extend_value('Const', 'rho', 1e-4, 'low') == pytest.approx(2e-5)
    assert P.extend_value('AdamW+dropout', 'dropout', 0.3, 'high') == 0.5
    assert P.extend_value('AdamW+EMA', 'ema', 0.99, 'low') == 0.9
    assert P.extend_value('CPR', 'warm_start', 250, 'high') == 1000
    assert P.extend_value('CPR', 'warm_start', 0, 'low') is None


def test_grid_args_count_six_configurations():
    lrs, grid, fixed = P.r16_plan('Const-Fair', 'XS')
    a = P.grid_args('XS', P.D_R16, P.E_R16, 'Const-Fair', lrs, grid, fixed, 3)
    assert a[:8] == ['--size', 'XS', '--unique-tokens', '10000000', '--epochs', '16', '--method', 'Const-Fair']
    assert '--grid' in a and 'rho=0.0001,0.0005' in a and 'delta=0.1' in a and '--finals' in a
    assert len(lrs) * len(grid['rho']) == 6


def test_edge_rule_extends_once_per_axis_then_stops():
    calls = []

    def runner(args):
        calls.append(args)
        lrs = [float(x) for x in args[args.index('--lrs') + 1].split(',')]
        # the winner always sits at the highest LR: the lr axis is extended exactly once
        return {'winner': {'hp': {'lr': max(lrs), 'wd': 0.5}, 'edge': {'lr': True, 'wd': True}}}

    out = P.run_search('AdamW', 'XS', P.D_R16, P.E_R16, P.LR_ADAM, {'wd': [0.1, 0.5]}, {}, 3, runner, log=lambda *_: None)
    assert len(calls) == 2
    assert calls[1][calls[1].index('--lrs') + 1] == '0.0005,0.0015,0.0045,0.0135'
    assert calls[1][calls[1].index('--grid') + 1] == 'wd=0.1,0.5,2.5'
    assert out['winner']['hp']['lr'] == pytest.approx(0.0135)


def test_stacking_and_control_arm_use_the_winners(tmp_path):
    res = tmp_path / 'dc'
    res.mkdir()
    (res / 'XS_D10M_E16_AdamW.json').write_text(json.dumps(
        {'winner': {'hp': {'lr': 0.0015, 'wd': 0.5}, 'edge': {}}, 'finals': {'1001': {'test_loss': 3.0}}}))
    lrs, grid, fixed = P.r16_plan('AdamW+dropout', 'XS', results=str(res))
    assert fixed == {'wd': 0.5} and grid == {'dropout': [0.1, 0.3]} and lrs == P.LR_ADAM
    with pytest.raises(FileNotFoundError):
        P.r16_plan('AdamW+EMA', 'S', results=str(res))
    # control arm: the winner and its weaker variant at the winning LR; other keys fixed
    (res / 'XS_D10M_E16_Const-Fair.json').write_text(json.dumps(
        {'winner': {'hp': {'lr': 0.0045, 'rho': 0.0005, 'delta': 0.1}, 'edge': {}}, 'finals': {}}))
    lrs, grid, fixed = P.r1_plan('Const-Fair', 'XS', results=str(res))
    assert lrs == [0.0045] and grid == {'rho': [0.0001, 0.0005]} and fixed == {'delta': 0.1}
    (res / 'XS_D10M_E16_Baseline.json').write_text(json.dumps({'winner': {'hp': {'lr': 0.0015}, 'edge': {}}, 'finals': {}}))
    lrs, grid, fixed = P.r1_plan('Baseline', 'XS', results=str(res))
    assert lrs == [0.0015] and grid == {} and fixed == {}
    calls = []
    P.phase_r1('XS', ['Baseline', 'Const-Fair'], lambda a: calls.append(a) or {'winner': {'hp': {}, 'edge': {}}},
               results=str(res), log=lambda *_: None)
    assert all('--unique-tokens' in a and a[a.index('--unique-tokens') + 1] == str(P.D_R1) for a in calls)
    assert all(a[a.index('--evals-per-epoch') + 1] == '32' and a[a.index('--finals') + 1] == '2' for a in calls)


def test_top_methods_adds_adamw_and_keeps_protocol_order(tmp_path):
    res = tmp_path / 'dc'
    res.mkdir()
    losses = {m: 3.0 + i * 0.01 for i, m in enumerate(P.ORDER)}      # Baseline best, Const-Huber worst
    losses['AdamW'] = 3.5
    for m, v in losses.items():
        (res / f'XS_D10M_E16_{m}.json').write_text(json.dumps(
            {'winner': {'hp': {'lr': 0.0015}, 'edge': {}}, 'finals': {'1001': {'test_loss': v}, '1002': {'test_loss': v}}}))
    chosen, means = P.top_methods('XS', results=str(res))
    assert chosen == ['AdamW', 'Baseline', 'AdamW+dropout', 'AdamW+EMA', 'CWD']
    assert means['AdamW'] == pytest.approx(3.5)


def test_dry_run_prints_every_invocation(capsys):
    P.main(['--phase', 'r16', '--size', 'XS', '--methods', 'Baseline,AdamW,Const', '--dry-run'])
    out = capsys.readouterr().out
    assert out.count('dc_grid.py') == 3 and '--method Baseline' in out and '--grid rho=0.0001,0.0005' in out
