"""Optimizers and decay rules of the data-constrained study.

Every Adam-family method runs on ONE Adam core (`DecayAdamW`), so that methods differ only
in their decay rule and wall-clock comparisons are not confounded by kernels:

  in-step decay (inside DecayAdamW.step, before the parameter moves)
    'none'      no decay
    'coupled'   p <- p * (1 - lr_t * wd), then the Adam update     (= torch.optim.AdamW)
    'cautious'  p <- p - lr_t * (u + wd * 1[u * p >= 0] * p)       (Cautious Weight Decay,
                Chen et al., ICLR 2026: decay only coordinates whose update agrees in sign)

  post-step decay (`post_step_decay`, applied after optimizer.step on the decay scope)
    'constant'        p <- p - rho * p                                  lr-independent
    'constant_fair'   p <- p - rho * p / (1 + |p|/delta)                tau-decay (Fair)
    'constant_huber'  p <- p - rho * clamp(p, -delta, delta)            Huber / AdamHD rule
    'coupled_fair'    p <- p - (lr_t/lr_peak) * rho * p / (1 + |p|/delta)

The decay scope (2-D weights: linear layers and embeddings; never biases or LayerNorm) is
the same for every method; see dcstudy.model.decay_scope.

`EMA` keeps an exponential moving average of ALL parameters for the methods that tune an
'ema' hyperparameter (AdamW+EMA): the averaged weights are what gets evaluated, snapshotted
and reported; training continues on the live weights.
"""
import math

import torch
from torch.optim import Optimizer

POST_STEP_MODES = ('constant', 'constant_fair', 'constant_huber', 'coupled_fair')
IN_STEP_MODES = ('none', 'coupled', 'cautious')


class DecayAdamW(Optimizer):
    """Adam with a decoupled in-step decay rule chosen per parameter group ('mode')."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                 mode='coupled'):
        if mode not in IN_STEP_MODES:
            raise ValueError(f"in-step mode must be one of {IN_STEP_MODES}, got {mode!r}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, mode=mode))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd, mode = group['lr'], group['weight_decay'], group['mode']
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st['step'] = 0
                    st['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                st['step'] += 1
                m, v = st['exp_avg'], st['exp_avg_sq']
                m.lerp_(p.grad, 1 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                bc1 = 1 - beta1 ** st['step']
                bc2 = 1 - beta2 ** st['step']
                denom = (v.sqrt() / math.sqrt(bc2)).add_(group['eps'])
                if mode == 'cautious' and wd != 0:
                    u = (m / bc1) / denom
                    mask = (u * p >= 0).to(p.dtype)
                    p.add_(u + wd * mask * p, alpha=-lr)
                    continue
                if mode == 'coupled' and wd != 0:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(m, denom, value=-lr / bc1)
        return loss


@torch.no_grad()
def sign_agreement(opt):
    """Diagnostic, read-only: the fraction of decayed coordinates whose current Adam update u
    satisfies u * p >= 0, the coordinates that Cautious Weight Decay decays. u is rebuilt from
    the optimizer state (bias-corrected m / (sqrt(v_hat) + eps)), so the call changes nothing
    in training. Returns None for an optimizer other than DecayAdamW or with no decayed group."""
    if not isinstance(opt, DecayAdamW):
        return None
    hit, tot = 0.0, 0
    for group in opt.param_groups:
        if group['weight_decay'] == 0:
            continue
        beta1, beta2 = group['betas']
        for p in group['params']:
            st = opt.state.get(p)
            if not st:
                continue
            bc1 = 1 - beta1 ** st['step']
            bc2 = 1 - beta2 ** st['step']
            u = (st['exp_avg'] / bc1) / ((st['exp_avg_sq'].sqrt() / math.sqrt(bc2)).add_(group['eps']))
            hit += float((u * p >= 0).sum())
            tot += p.numel()
    return hit / tot if tot else None


@torch.no_grad()
def post_step_decay(params, mode, rho, delta=None, lr_scale=1.0):
    """Decoupled decay applied after optimizer.step to `params` (the decay scope)."""
    if rho == 0 or mode is None:
        return
    if mode not in POST_STEP_MODES:
        raise ValueError(f"post-step mode must be one of {POST_STEP_MODES}, got {mode!r}")
    if mode == 'coupled_fair':
        rho = rho * lr_scale
        if rho == 0:
            return
    for p in params:
        if mode == 'constant':
            p.mul_(1 - rho)
        elif mode in ('constant_fair', 'coupled_fair'):
            p.sub_(rho * p / (1 + p.abs() / delta))
        elif mode == 'constant_huber':
            p.sub_(rho * p.clamp(-delta, delta))


class EMA:
    """Exponential moving average of a list of parameters, kept as a shadow copy.

    The coefficient is warmed up as beta_t = min(beta, (1 + t) / (10 + t)), so the average is
    unbiased from the first step and a beta close to 1 does not pin it to the initialisation
    on a short run. `swap()` exchanges the live and the averaged parameters in place; a second
    swap undoes the first exactly, so the model can be evaluated (and snapshotted) with the
    averaged weights and go on training with the live ones.
    """

    def __init__(self, params, beta):
        beta = float(beta)
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"EMA beta must be in [0, 1), got {beta}")
        self.params = [p for p in params]
        self.beta = beta
        self.t = 0
        self.shadow = [p.detach().clone() for p in self.params]
        self.swapped = False

    def coefficient(self, t):
        return min(self.beta, (1.0 + t) / (10.0 + t))

    @torch.no_grad()
    def update(self):
        self.t += 1
        w = 1.0 - self.coefficient(self.t)
        for p, s in zip(self.params, self.shadow):
            s.lerp_(p.detach(), w)

    @torch.no_grad()
    def swap(self):
        for p, s in zip(self.params, self.shadow):
            tmp = p.detach().clone()
            p.copy_(s)
            s.copy_(tmp)
        self.swapped = not self.swapped
