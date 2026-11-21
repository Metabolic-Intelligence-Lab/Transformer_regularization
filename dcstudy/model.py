"""GPT-2-style decoder for the data-constrained study.

Pre-LayerNorm blocks, learned absolute positions, GELU MLP (4x), biases on linear layers
and LayerNorms (as in GPT-2), tied input/output embeddings, attention through
F.scaled_dot_product_attention (flash kernels on GPU). Initialisation follows GPT-2:
N(0, 0.02) everywhere, residual output projections scaled by 1/sqrt(2 * n_layer).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304          # GPT-2 BPE (50257) padded to a multiple of 64
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


SIZES = {
    # tied embeddings; parameter counts asserted in tests/test_dcstudy.py
    'tiny': dict(n_layer=2, n_head=2, n_embd=64, block_size=128),     # CPU tests only
    'XS':   dict(n_layer=6, n_head=6, n_embd=384),                    # ~30.4M
    'S':    dict(n_layer=8, n_head=8, n_embd=512),                    # ~51.5M
    'M':    dict(n_layer=12, n_head=12, n_embd=768),                  # ~124.4M (GPT-2 small)
    'L':    dict(n_layer=24, n_head=16, n_embd=1024),                 # ~355M (GPT-2 medium)
}


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.residual = True
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.residual = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x), approximate='tanh')))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight            # tied
        self.apply(self._init)
        for m in self.modules():
            if getattr(m, 'residual', False):
                nn.init.normal_(m.weight, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for blk in self.h:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def build(size, dropout=0.0, block_size=None):
    kw = dict(SIZES[size])
    if block_size is not None:
        kw['block_size'] = block_size
    return GPT(GPTConfig(dropout=dropout, **kw))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def decay_scope(model):
    """Parameters every decay method acts on: 2-D weight tensors (linear weights and the
    tied token embedding, plus the position embedding); biases and LayerNorm gains are
    never decayed. This is the standard AdamW practice of nanoGPT/GPT-2 training, and it
    is the SAME scope for every method, so the methods differ only in how they decay."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append((name, p))
    return decay, no_decay
