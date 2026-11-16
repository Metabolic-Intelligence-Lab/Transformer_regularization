"""One training run of the data-constrained study.

A fixed set of D unique training tokens is cut into non-overlapping block_size+1 windows
and revisited for `epochs` passes, each pass in a fresh seeded order. Every run of a given
(size, D, epochs, batch) sees exactly the same number of tokens and optimizer steps, so
methods are compared at matched training compute. Validation loss is measured every
`eval_every` steps; the best-validation checkpoint is kept in memory and restored with the
verified restore of experiment_utils.EarlyStopping (no early termination: every run uses
its whole budget). Reported: test loss at the best-validation checkpoint and at the end.
Methods with an 'ema' hyperparameter are evaluated, snapshotted and reported on the
exponential moving average of their weights (dcstudy.optim.EMA); training itself is unchanged.
"""
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_utils import EarlyStopping, set_seed  # noqa: E402

from .methods import METHODS, build_optimizer  # noqa: E402
from .model import build, decay_scope, n_params  # noqa: E402
from .optim import EMA, post_step_decay, sign_agreement  # noqa: E402


def lr_multiplier(step, warmup_steps, total_steps, floor=0.0):
    """Linear warmup, then linear decay from 1 to `floor` at total_steps."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    frac = max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return floor + (1.0 - floor) * frac


def load_tokens(path, n=None):
    arr = np.memmap(path, dtype=np.uint16, mode='r')
    return arr if n is None else arr[:n]


def windows(n_tokens, block):
    return (n_tokens - 1) // block


def batch_from(tokens, starts, block, device):
    idx = starts[:, None] + np.arange(block + 1)[None, :]
    arr = torch.from_numpy(np.asarray(tokens[idx.ravel()], dtype=np.int64).reshape(idx.shape))
    if device.type == 'cuda':
        arr = arr.pin_memory().to(device, non_blocking=True)
    else:
        arr = arr.to(device)
    return arr[:, :-1], arr[:, 1:]


@torch.no_grad()
def eval_loss(model, tokens, block, batch, device, amp):
    model.eval()
    n = windows(len(tokens), block)
    starts_all = np.arange(n) * block
    tot, cnt = 0.0, 0
    for i in range(0, n, batch):
        x, y = batch_from(tokens, starts_all[i:i + batch], block, device)
        with amp():
            _, loss = model(x, y)
        tot += loss.item() * x.numel()
        cnt += x.numel()
    model.train()
    return tot / cnt


@torch.no_grad()
def weight_stats(params):
    sq = sum(float(p.float().pow(2).sum()) for p in params)
    mx = max(float(p.abs().max()) for p in params)
    return {'l2': math.sqrt(sq), 'max_abs': mx}


def run(cfg, train_tokens, val_tokens, test_tokens, log=print):
    """cfg keys: size, method, hp (dict incl. lr), seed, epochs, block_size, batch_seqs,
    micro_seqs, warmup_frac, lr_floor, eval_every, eval_batch, device, amp ('bf16'|None),
    compile (bool). Returns a metrics dict."""
    device = torch.device(cfg['device'])
    set_seed(cfg['seed'])
    hp = dict(cfg['hp'])
    spec = METHODS[cfg['method']]
    model = build(cfg['size'], dropout=hp.get('dropout', 0.0),
                  block_size=cfg['block_size']).to(device)
    n_par = n_params(model)
    decay_params = [p for _, p in decay_scope(model)[0]]
    fwd = torch.compile(model) if cfg.get('compile') else model

    if cfg.get('amp') == 'bf16' and device.type == 'cuda':
        def amp():
            return torch.autocast('cuda', dtype=torch.bfloat16)
    else:
        from contextlib import nullcontext

        def amp():
            return nullcontext()

    block, B, mb = cfg['block_size'], cfg['batch_seqs'], cfg['micro_seqs']
    assert B % mb == 0
    n_win = windows(len(train_tokens), block)
    steps_per_epoch = n_win // B
    total = steps_per_epoch * cfg['epochs']
    warmup = max(1, int(cfg['warmup_frac'] * total))
    opt, post_params, post_mode, sf = build_optimizer(cfg['method'], model, hp, total, warmup)
    ema = EMA(model.parameters(), hp['ema']) if 'ema' in spec['hp'] else None
    peak = [g['lr'] for g in opt.param_groups]
    sched = None if sf else torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_multiplier(s, warmup, total, cfg.get('lr_floor', 0.0)))
    tracker = EarlyStopping(patience=10 ** 9, min_delta=0.0, mode='min')
    rng = np.random.default_rng(cfg['seed'])
    curve, t_train, tok_seen, run_loss, n_loss = [], 0.0, 0, 0.0, 0
    if sf:
        opt.train()
    model.train()
    log(f"[RUN] {cfg['method']} hp={hp} seed={cfg['seed']} params={n_par:,} "
        f"steps={total} ({steps_per_epoch}/epoch x {cfg['epochs']}) warmup={warmup}")

    for step in range(total):
        if step % steps_per_epoch == 0:
            order = rng.permutation(n_win)[:steps_per_epoch * B] * block
        t0 = time.time()
        sel = order[(step % steps_per_epoch) * B:(step % steps_per_epoch + 1) * B]
        for j in range(0, B, mb):
            x, y = batch_from(train_tokens, sel[j:j + mb], block, device)
            with amp():
                _, loss = fwd(x, y)
            (loss * (mb / B)).backward()
            run_loss = run_loss + loss.detach() * (mb / B)     # no host sync per micro-batch
        n_loss += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr_scale = opt.param_groups[0]['lr'] / peak[0]         # the lr this step used
        opt.step()
        if sched is not None:
            sched.step()
        if post_mode:
            post_step_decay(post_params, post_mode, hp['rho'], hp.get('delta'), lr_scale=lr_scale)
        if ema is not None:
            ema.update()
        opt.zero_grad(set_to_none=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_train += time.time() - t0
        tok_seen += B * block

        if (step + 1) % cfg['eval_every'] == 0 or step + 1 == total:
            if sf:
                opt.eval()
            if ema is not None:
                ema.swap()                  # evaluate and snapshot the averaged weights
            v = eval_loss(model, val_tokens, block, cfg['eval_batch'], device, amp)
            tracker(v, model, len(curve))
            if ema is not None:
                ema.swap()                  # back to the live weights for training
            tr = float(run_loss) / max(1, n_loss)
            agree = sign_agreement(opt)
            curve.append({'step': step + 1, 'tokens': tok_seen, 'epoch': (step + 1) / steps_per_epoch,
                          'train_loss': tr, 'val_loss': v,
                          'train_s': t_train, **weight_stats(decay_params),
                          **({'agree': agree} if agree is not None else {})})
            log(f"  step {step + 1}/{total} ep {(step + 1) / steps_per_epoch:.2f} "
                f"train {tr:.4f} val {v:.4f} tok/s {tok_seen / max(t_train, 1e-9):,.0f}")
            run_loss, n_loss = 0.0, 0
            if sf and step + 1 < total:
                opt.train()

    final_val = curve[-1]['val_loss']
    if ema is not None:
        ema.swap()          # from here on the model holds the averaged weights, as the snapshots do
    final_test = eval_loss(model, test_tokens, block, cfg['eval_batch'], device, amp)
    tracker.restore_best_model(model)
    v_best = eval_loss(model, val_tokens, block, cfg['eval_batch'], device, amp)
    tracker.check_restored(v_best, context=f"{cfg['method']} seed={cfg['seed']}")
    best_test = eval_loss(model, test_tokens, block, cfg['eval_batch'], device, amp)
    best_i = tracker.best_epoch
    return {
        'val_loss': v_best, 'test_loss': best_test, 'test_ppl': math.exp(best_test),
        'final_val_loss': final_val, 'final_test_loss': final_test,
        'final_test_ppl': math.exp(final_test),
        'best_step': curve[best_i]['step'], 'best_epoch': curve[best_i]['epoch'],
        'total_steps': total, 'tokens': tok_seen, 'params': n_par,
        'train_seconds': t_train, 'tokens_per_second': tok_seen / max(t_train, 1e-9),
        'curve': curve,
    }
