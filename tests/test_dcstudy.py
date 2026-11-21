"""Tests for the data-constrained study package (dcstudy/). CPU, ~1 minute.

    pytest tests/test_dcstudy.py -q
"""
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dcstudy import methods as M  # noqa: E402
from dcstudy.model import build, decay_scope, n_params  # noqa: E402
from dcstudy.optim import EMA, DecayAdamW, post_step_decay, sign_agreement  # noqa: E402
from dcstudy.train import lr_multiplier, run  # noqa: E402

HAS_CPR = importlib.util.find_spec('pytorch_cpr') is not None
HAS_SF = importlib.util.find_spec('schedulefree') is not None


# ----------------------------------------------------------------------------- model
def test_parameter_counts_and_scope():
    for size, n in (('XS', 30_357_504), ('S', 51_500_032), ('M', 124_475_904), ('L', 354_871_296)):
        m = build(size)
        assert n_params(m) == n
        decay, no_decay = decay_scope(m)
        assert all(p.dim() >= 2 for _, p in decay)
        assert all(('bias' in name) or ('ln' in name) for name, _ in no_decay)
        assert m.lm_head.weight is m.wte.weight
        del m


# ----------------------------------------------------------------------------- optimizers
def _pair(seed=0):
    torch.manual_seed(seed)
    a = torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3))
    b = torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3))
    b.load_state_dict(a.state_dict())
    return a, b


def _groups(model, wd):
    dec = [p for p in model.parameters() if p.dim() >= 2]
    nod = [p for p in model.parameters() if p.dim() < 2]
    return [{'params': dec, 'weight_decay': wd}, {'params': nod, 'weight_decay': 0.0}]


def _train(model, opt, steps=6, post=None):
    torch.manual_seed(1)
    x, y = torch.randn(16, 6), torch.randn(16, 3)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        opt.step()
        if post:
            post()


def test_coupled_mode_equals_torch_adamw():
    a, b = _pair()
    _train(a, DecayAdamW(_groups(a, 0.3), lr=1e-2, betas=(0.9, 0.95), mode='coupled'))
    _train(b, torch.optim.AdamW(_groups(b, 0.3), lr=1e-2, betas=(0.9, 0.95)))
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-6, atol=1e-7)


def test_none_and_zero_decay_modes_equal_adam():
    a, b = _pair()
    _train(a, DecayAdamW(_groups(a, 0.0), lr=1e-2, mode='cautious'))
    _train(b, torch.optim.Adam(b.parameters(), lr=1e-2, betas=(0.9, 0.95)))
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-6, atol=1e-7)


def test_cautious_decays_only_where_update_agrees_in_sign():
    p = torch.nn.Parameter(torch.tensor([1.0, 1.0, -1.0, -1.0]))
    p.grad = torch.tensor([1.0, -1.0, -1.0, 1.0])     # first Adam update u = sign(grad)
    opt = DecayAdamW([{'params': [p], 'weight_decay': 0.5}], lr=0.1, mode='cautious')
    opt.step()
    # u*p >= 0 for coordinates 0 and 2 -> decayed; 1 and 3 -> not decayed
    u = torch.tensor([1.0, -1.0, -1.0, 1.0])
    mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    expected = torch.tensor([1.0, 1.0, -1.0, -1.0]) - 0.1 * (u + 0.5 * mask * torch.tensor([1.0, 1.0, -1.0, -1.0]))
    torch.testing.assert_close(p.detach(), expected, rtol=1e-5, atol=1e-6)


def test_post_step_decay_rules():
    w = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0])
    for mode, rho, delta, scale, f in [
        ('constant', 0.1, None, 1.0, lambda x: x * 0.9),
        ('constant_fair', 0.1, 0.5, 1.0, lambda x: x - 0.1 * x / (1 + x.abs() / 0.5)),
        ('constant_huber', 0.1, 0.5, 1.0, lambda x: x - 0.1 * x.clamp(-0.5, 0.5)),
        ('coupled_fair', 0.1, 0.5, 0.25, lambda x: x - 0.025 * x / (1 + x.abs() / 0.5)),
    ]:
        p = w.clone()
        post_step_decay([p], mode, rho, delta, lr_scale=scale)
        torch.testing.assert_close(p, f(w))
    p = w.clone()
    post_step_decay([p], 'coupled_fair', 0.1, 0.5, lr_scale=0.0)
    torch.testing.assert_close(p, w)
    with pytest.raises(ValueError):
        post_step_decay([w.clone()], 'coupled', 0.1)


def test_lr_multiplier():
    vals = [lr_multiplier(s, 10, 100, 0.0) for s in range(100)]
    assert vals[0] == pytest.approx(0.1) and vals[9] == pytest.approx(1.0)
    assert vals[10] == pytest.approx(1.0) and lr_multiplier(100, 10, 100) == 0.0
    assert lr_multiplier(100, 10, 100, 0.1) == pytest.approx(0.1)


# ----------------------------------------------------------------------------- end to end
def _tokens(n, seed):
    # A learnable synthetic stream: a repeated random "vocabulary walk" plus noise.
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 300, size=257)
    reps = np.tile(base, n // len(base) + 1)[:n]
    noise = rng.integers(0, 300, size=n)
    keep = rng.random(n) < 0.8
    return np.where(keep, reps, noise).astype(np.uint16)


CFG = dict(size='tiny', seed=42, epochs=2, block_size=32, batch_seqs=8, micro_seqs=4,
           warmup_frac=0.1, lr_floor=0.0, eval_every=5, eval_batch=16, device='cpu',
           amp=None, compile=False)

HP = {
    'Baseline': {'lr': 3e-3}, 'AdamW': {'lr': 3e-3, 'wd': 0.1},
    'AdamW+dropout': {'lr': 3e-3, 'wd': 0.1, 'dropout': 0.1}, 'CWD': {'lr': 3e-3, 'wd': 0.1},
    'AntiCWD': {'lr': 3e-3, 'wd': 0.1}, 'RandWD': {'lr': 3e-3, 'wd': 0.1},
    'AdamW+EMA': {'lr': 3e-3, 'wd': 0.1, 'ema': 0.9},
    'Const': {'lr': 3e-3, 'rho': 1e-4}, 'Const-Fair': {'lr': 3e-3, 'rho': 1e-4, 'delta': 0.1},
    'Const-Huber': {'lr': 3e-3, 'rho': 1e-4, 'delta': 0.1},
    'Coupled-Fair': {'lr': 3e-3, 'rho': 1e-4, 'delta': 0.1},
    'CPR': {'lr': 3e-3, 'warm_start': 5}, 'AdamS': {'lr': 3e-3, 'wd': 1e-3},
    'SF-AdamW': {'lr': 3e-3, 'wd': 0.1}, 'AdEMAMix': {'lr': 3e-3, 'wd': 0.1},
    'AdEMAMix-T': {'lr': 3e-3, 'wd': 0.1, 'alpha': 2.0, 'beta3': 0.999},
    'Muon': {'lr': 0.02, 'wd': 0.01, 'aux_lr': 3e-3, 'aux_wd': 0.1},
    'Muon+Const': {'lr': 0.02, 'rho': 1e-4, 'aux_lr': 3e-3},
}


def test_every_method_has_a_test_config():
    assert set(HP) == set(M.METHODS)


@pytest.mark.parametrize('method', sorted(M.METHODS))
def test_run_end_to_end(method):
    if method == 'CPR' and not HAS_CPR:
        pytest.skip('pytorch-cpr not installed')
    if method == 'SF-AdamW' and not HAS_SF:
        pytest.skip('schedulefree not installed')
    torch.set_num_threads(2)
    train, val, test = _tokens(2000, 0), _tokens(600, 1), _tokens(600, 2)
    out = run(dict(CFG, method=method, hp=HP[method]), train, val, test, log=lambda *_: None)
    assert out['total_steps'] == 2 * ((2000 - 1) // 32 // 8)
    assert out['tokens'] == out['total_steps'] * 8 * 32
    assert math.isfinite(out['test_loss']) and math.isfinite(out['final_test_loss'])
    assert out['val_loss'] == pytest.approx(min(c['val_loss'] for c in out['curve']), rel=5e-3)
    assert out['curve'][-1]['val_loss'] < math.log(50304)       # it learned something


def test_run_is_deterministic_on_cpu():
    train, val, test = _tokens(2000, 0), _tokens(600, 1), _tokens(600, 2)
    a = run(dict(CFG, method='Const', hp=HP['Const']), train, val, test, log=lambda *_: None)
    b = run(dict(CFG, method='Const', hp=HP['Const']), train, val, test, log=lambda *_: None)
    assert a['test_loss'] == b['test_loss'] and a['best_step'] == b['best_step']


# ----------------------------------------------------------------------------- EMA
def test_ema_closed_form_and_swap_is_an_involution():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(7))
    ema = EMA([p], beta=0.9)
    ref = p.detach().clone()
    for t in range(1, 9):
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.1)
        ema.update()
        b = min(0.9, (1 + t) / (10 + t))
        ref = b * ref + (1 - b) * p.detach()
    torch.testing.assert_close(ema.shadow[0], ref)
    live = p.detach().clone()
    ema.swap()
    torch.testing.assert_close(p.detach(), ref)          # the model now holds the average
    ema.swap()
    torch.testing.assert_close(p.detach(), live)         # and is back to the live weights
    torch.testing.assert_close(ema.shadow[0], ref)
    with pytest.raises(ValueError):
        EMA([p], beta=1.0)


def test_ema_with_beta_zero_reproduces_adamw_exactly():
    # beta = 0 makes the average equal to the live weights after every step, so the EMA hooks
    # (swap before each evaluation, swap back, swap at the end) must leave every reported
    # number identical to the plain AdamW run on the same seed.
    train, val, test = _tokens(2000, 0), _tokens(600, 1), _tokens(600, 2)
    a = run(dict(CFG, method='AdamW', hp={'lr': 3e-3, 'wd': 0.1}), train, val, test,
            log=lambda *_: None)
    b = run(dict(CFG, method='AdamW+EMA', hp={'lr': 3e-3, 'wd': 0.1, 'ema': 0.0}), train, val,
            test, log=lambda *_: None)
    assert a['test_loss'] == b['test_loss'] and a['final_test_loss'] == b['final_test_loss']
    assert a['best_step'] == b['best_step']
    assert [c['val_loss'] for c in a['curve']] == [c['val_loss'] for c in b['curve']]


# ----------------------------------------------------------------------------- dc_grid
def test_dc_grid_finals_follow_the_winner(tmp_path, monkeypatch):
    import dc_grid
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(2)
    out1 = dc_grid.main(['--smoke', '--no-compile', '--method', 'Const', '--lrs', '3e-3',
                         '--grid', 'rho=1e-4', '--finals', '1'])
    out2 = dc_grid.main(['--smoke', '--no-compile', '--method', 'Const', '--lrs', '3e-3,1e-2',
                         '--grid', 'rho=1e-4', '--finals', '1'])
    assert len(out1['grid']) == 1 and len(out2['grid']) == 2
    recs = [json.loads(line) for line in open(tmp_path / 'results' / 'journal' / 'dc_tiny_Dsmoke_E2_Const.jsonl')]
    grid_runs = [r for r in recs if r['kind'] == 'pso']
    evals = [r for r in recs if r['kind'] == 'eval']
    assert len(grid_runs) == 2                           # the first point was replayed, not retrained
    winners = {out1['winner']['hp']['lr'], out2['winner']['hp']['lr']}
    assert len(evals) == len(winners)                    # finals re-run only if the winner moved
    assert all('@' in r['method'] for r in evals)        # journaled under the winner's hyperparameters
    assert out2['winner']['edge']['lr'] is True          # two values: the winner is on an edge


def test_sign_agreement_is_read_only_and_matches_the_cautious_mask():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(64, 32))
    opt = DecayAdamW([{'params': [w], 'weight_decay': 2.5}], lr=1e-2, mode='coupled')
    assert sign_agreement(opt) is None or sign_agreement(opt) == 0  # no state yet
    for _ in range(3):
        w.grad = torch.randn_like(w)
        opt.step()
    before = w.detach().clone()
    a = sign_agreement(opt)
    assert torch.equal(w.detach(), before)               # reading the diagnostic changes nothing
    st = opt.state[w]
    u = (st['exp_avg'] / (1 - 0.9 ** 3)) / ((st['exp_avg_sq'] / (1 - 0.95 ** 3)).sqrt() + 1e-8)
    assert abs(a - float((u * w >= 0).float().mean())) < 1e-6
    assert sign_agreement(torch.optim.SGD([w], lr=0.1)) is None


def test_dc_grid_final_seeds_override(tmp_path, monkeypatch):
    import dc_grid
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(2)
    out = dc_grid.main(['--smoke', '--no-compile', '--method', 'CWD', '--lrs', '3e-3',
                        '--grid', 'wd=0.5', '--final-seeds', '2001,2002'])
    assert sorted(out['finals']) == ['2001', '2002']
    assert all('agree' in c for c in out['grid'][0]['curve'])
    with pytest.raises(SystemExit):
        dc_grid.main(['--smoke', '--no-compile', '--method', 'CWD', '--lrs', '3e-3',
                      '--grid', 'wd=0.5', '--final-seeds', '42'])


def test_p2_holm_adjustment():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        'p2a', os.path.join(os.path.dirname(__file__), '..', 'analysis', 'p2_analysis.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assert m.holm([0.055, 0.168]) == [0.11, 0.168]
    assert m.holm([0.3, 0.01]) == [0.3, 0.02]


def _one_step(mode, seed=0, wd=0.5, **kw):
    torch.manual_seed(seed)
    w = torch.nn.Parameter(torch.randn(256, 64))
    g = torch.randn_like(w)
    p0 = w.detach().clone()
    opt = DecayAdamW([{'params': [w], 'weight_decay': wd}], lr=1e-2, mode=mode, **kw)
    w.grad = g.clone()
    opt.step()
    return p0, w.detach().clone(), g


def test_cwd_control_masks():
    p0, none_w, _ = _one_step('cautious', wd=0.0)          # pure Adam step, no decay anywhere
    _, c, _ = _one_step('cautious')
    _, a, _ = _one_step('anticautious')
    _, r, _ = _one_step('random', mask_p=0.4)
    dec = lambda x: (none_w - x).abs() > 1e-6                          # coordinates that were decayed
    dc, da, dr = dec(c), dec(a), dec(r)
    assert not (dc & da).any() and (dc | da).float().mean() > 0.99   # the two masks partition the coordinates
    assert abs(dr.float().mean().item() - 0.4) < 0.03                 # random mask: the stated fraction
    _, r2, _ = _one_step('random', mask_p=0.4)
    assert torch.equal(r, r2)                                         # reproducible under the same seed
