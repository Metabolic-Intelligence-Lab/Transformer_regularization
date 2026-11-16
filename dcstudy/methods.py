"""Method registry: how each compared method builds its optimizer and post-step decay.

Hyperparameters every method receives: `lr` (peak learning rate; for Muon the Muon-group
learning rate, with the auxiliary Adam groups at `aux_lr`) plus the method-specific keys
listed in METHODS[name]['hp'].
"""
from .optim import DecayAdamW
from .model import decay_scope

ADAM_BETAS = (0.9, 0.95)

METHODS = {
    # name:            optimizer family, in-step mode, post-step mode, tuned keys (besides lr)
    'Baseline':      dict(family='adam', mode='none',     post=None,             hp=[]),
    'AdamW':         dict(family='adam', mode='coupled',  post=None,             hp=['wd']),
    'AdamW+dropout': dict(family='adam', mode='coupled',  post=None,             hp=['wd', 'dropout']),
    'AdamW+EMA':     dict(family='adam', mode='coupled',  post=None,             hp=['wd', 'ema']),
    'CWD':           dict(family='adam', mode='cautious', post=None,             hp=['wd']),
    'Const':         dict(family='adam', mode='none',     post='constant',       hp=['rho']),
    'Const-Fair':    dict(family='adam', mode='none',     post='constant_fair',  hp=['rho', 'delta']),
    'Const-Huber':   dict(family='adam', mode='none',     post='constant_huber', hp=['rho', 'delta']),
    'Coupled-Fair':  dict(family='adam', mode='none',     post='coupled_fair',   hp=['rho', 'delta']),
    'CPR':           dict(family='cpr',  mode=None,       post=None,             hp=['warm_start']),
    'AdamS':         dict(family='adams', mode=None,      post=None,             hp=['wd']),
    'SF-AdamW':      dict(family='sf',   mode=None,       post=None,             hp=['wd']),
    'AdEMAMix':      dict(family='ademamix', mode=None,   post=None,             hp=['wd']),
    # post-registration: AdEMAMix with its mixing weight and slow-momentum coefficient tuned
    # (the reference values alpha 8, beta3 0.9999 diverge at 2,432 steps)
    'AdEMAMix-T':    dict(family='ademamix', mode=None,   post=None,             hp=['alpha', 'beta3']),
    'Muon':          dict(family='muon', mode=None,       post=None,             hp=['wd']),
    'Muon+Const':    dict(family='muon', mode=None,       post='constant',       hp=['rho']),
}


def _groups(model, wd):
    decay, no_decay = decay_scope(model)
    return ([p for _, p in decay], [p for _, p in no_decay],
            [{'params': [p for _, p in decay], 'weight_decay': wd},
             {'params': [p for _, p in no_decay], 'weight_decay': 0.0}])


def build_optimizer(name, model, hp, total_steps, warmup_steps):
    """Return (optimizer, post_params, post_mode, schedule_free)."""
    spec = METHODS[name]
    fam, lr = spec['family'], hp['lr']
    wd = hp.get('wd', 0.0)
    decay_p, no_decay_p, groups = _groups(model, wd)
    post_params = decay_p if spec['post'] else []

    if fam == 'adam':
        opt = DecayAdamW(groups, lr=lr, betas=ADAM_BETAS, mode=spec['mode'])
        return opt, post_params, spec['post'], False

    if fam == 'cpr':
        from pytorch_cpr import AdamCPR
        ws = int(hp['warm_start'])
        cpr_groups = [{'params': decay_p, 'regularize': True},
                      {'params': no_decay_p, 'regularize': False}]
        kw = (dict(kappa_init_method='inflection_point') if ws <= 0 else
              dict(kappa_init_method='warm_start', kappa_init_param=ws))
        opt = AdamCPR(cpr_groups, lr=lr, betas=ADAM_BETAS, **kw)
        return opt, [], None, False

    if fam == 'adams':
        from .third_party.adams import AdamS
        return AdamS(groups, lr=lr, betas=ADAM_BETAS), [], None, False

    if fam == 'sf':
        from schedulefree import AdamWScheduleFree
        opt = AdamWScheduleFree(groups, lr=lr, betas=ADAM_BETAS, warmup_steps=warmup_steps)
        return opt, [], None, True

    if fam == 'ademamix':
        from .third_party.ademamix import AdEMAMix
        # Recommended settings of the reference implementation (README): betas
        # (0.9, 0.999, 0.9999), alpha 8, alpha/beta3 warmup over the whole run.
        opt = AdEMAMix(groups, lr=lr, betas=(0.9, 0.999, float(hp.get('beta3', 0.9999))),
                       alpha=float(hp.get('alpha', 8.0)),
                       beta3_warmup=total_steps, alpha_warmup=total_steps)
        return opt, [], None, False

    if fam == 'muon':
        from .third_party.muon import SingleDeviceMuonWithAuxAdam
        # Muon on hidden 2-D matrices; embeddings (tied head), biases and LayerNorms on
        # the auxiliary Adam, as the reference implementation prescribes.
        hidden = [p for n, p in model.named_parameters() if p.dim() == 2 and n.startswith('h.')]
        hidden_ids = {id(p) for p in hidden}
        emb = [p for n, p in model.named_parameters() if p.dim() == 2 and id(p) not in hidden_ids]
        muon_wd = 0.0 if spec['post'] else wd
        param_groups = [
            dict(params=hidden, lr=lr, momentum=0.95, weight_decay=muon_wd, use_muon=True),
            dict(params=emb, lr=hp['aux_lr'], betas=ADAM_BETAS, eps=1e-8,
                 weight_decay=0.0 if spec['post'] else hp.get('aux_wd', wd), use_muon=False),
            dict(params=no_decay_p, lr=hp['aux_lr'], betas=ADAM_BETAS, eps=1e-8,
                 weight_decay=0.0, use_muon=False),
        ]
        return SingleDeviceMuonWithAuxAdam(param_groups), post_params, spec['post'], False

    raise ValueError(f"unknown family {fam!r} for {name}")
