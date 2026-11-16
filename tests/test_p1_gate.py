"""Tests of the P1 pilot gate (analysis/p1_gate.py). CPU, instant."""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('p1_gate', os.path.join(HERE, '..', 'analysis', 'p1_gate.py'))
p1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p1)


def _run(best_epoch, epochs, train_seconds):
    return {'stem': 'x', 'best_epoch': best_epoch, 'epochs': epochs, 'train_seconds': train_seconds,
            'tokens_per_second': 1e5, 'val_loss': 3.0, 'final_val_loss': 3.1, 'test_ppl': 20.0, 'params': 1}


def test_stems_match_dc_grid_naming():
    s = p1.stems(10)
    assert s == {'xs_r16': 'XS_D10M_E16_Baseline', 'xs_r1': 'XS_D159.384M_E1_Baseline',
                 's_r16': 'S_D10M_E16_Baseline'}
    assert p1.stems(5)['xs_r1'] == 'XS_D79.6918M_E1_Baseline'


def test_gate_passes_when_all_rules_hold():
    v = p1.evaluate({'xs_r16': _run(8.0, 16, 600), 'xs_r1': _run(0.97, 1, 600), 's_r16': _run(6.0, 16, 1300)}, rate=0.5)
    assert v['checks'] == {'r16_overfits_in_budget': True, 'r1_still_improving': True, 'cost_within_budget': True}
    assert v['G1'] == 'PASS'
    pr = v['projection']
    assert pr['xs_runs'] == 180 and pr['s_runs'] == 50
    assert abs(pr['gpu_h'] - (180 * 600 + 50 * 1300) * 1.15 / 3600) < 1e-9
    assert abs(pr['usd'] - pr['gpu_h'] * 0.5) < 1e-9


def test_gate_fails_on_each_rule():
    ok = {'xs_r16': _run(8.0, 16, 600), 'xs_r1': _run(0.97, 1, 600), 's_r16': _run(6.0, 16, 1300)}
    assert p1.evaluate(dict(ok, xs_r16=_run(12.0, 16, 600)))['checks']['r16_overfits_in_budget'] is False
    assert p1.evaluate(dict(ok, xs_r1=_run(0.5, 1, 600)))['checks']['r1_still_improving'] is False
    slow = p1.evaluate(dict(ok, xs_r16=_run(8.0, 16, 3000), xs_r1=_run(0.97, 1, 3000)))
    assert slow['checks']['cost_within_budget'] is False and slow['G1'] == 'FAIL'
    assert p1.evaluate(dict(ok, s_r16=None))['projection'] is None
    assert p1.evaluate(dict(ok, s_r16=None))['G1'] == 'FAIL'


def test_main_reads_result_files(tmp_path):
    d = tmp_path / 'results' / 'dc'
    d.mkdir(parents=True)
    for key, stem in p1.stems(10).items():
        epochs = 1 if key == 'xs_r1' else 16
        best = 0.95 if key == 'xs_r1' else 7.5
        doc = {'protocol': {'epochs': epochs},
               'grid': [{'best_epoch': best, 'train_seconds': 700.0, 'tokens_per_second': 2e5, 'val_loss': 3.2,
                         'final_val_loss': 3.3, 'test_ppl': 25.0, 'params': 30357504}]}
        (d / f'{stem}.json').write_text(json.dumps(doc))
    out = tmp_path / 'g.json'
    v = p1.main(['--results', str(d), '--out', str(out), '--rate', '0.4'])
    assert v['G1'] == 'PASS' and json.load(open(out))['G1'] == 'PASS'
