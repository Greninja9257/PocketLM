"""The three 50K candidates: a tiny Transformer, a GRU, and a hybrid.

Shared discipline across all three so the comparison is fair:
  * identical 512-token vocabulary and tied input/output embeddings
  * bias-free linear layers, RMSNorm instead of LayerNorm
  * the same <=50,000 trainable-parameter budget, asserted at construction

Every model exposes forward(ids) -> logits [B, T, V]; nothing else about the
internals leaks into training, generation, or evaluation.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    """Norm without the mean subtraction or the bias: d params instead of 2d."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype)) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device=None):
    """Rotary position embeddings: positional information for zero parameters."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv)                       # [T, head_dim/2]
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, hd]
    t = x.shape[-2]
    cos, sin = cos[:t].unsqueeze(0).unsqueeze(0), sin[:t].unsqueeze(0).unsqueeze(0)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    """Causal attention with optional grouped/multi-query key-value sharing."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, self.n_heads, self.n_kv = cfg.d_model, cfg.n_heads, cfg.n_kv_heads
        self.hd = cfg.head_dim
        assert self.n_heads % self.n_kv == 0, "n_heads must be divisible by n_kv_heads"
        self.q = nn.Linear(d, self.n_heads * self.hd, bias=False)
        self.k = nn.Linear(d, self.n_kv * self.hd, bias=False)
        self.v = nn.Linear(d, self.n_kv * self.hd, bias=False)
        self.o = nn.Linear(self.n_heads * self.hd, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.ff = RMSNorm(cfg.d_model), SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.ff(self.n2(x))


class BaseLM(nn.Module):
    """Shared embedding/tied-head plumbing and the budget assertion."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        # Tied: the output projection *is* the embedding table, transposed.
        return F.linear(h, self.embed.weight)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def loss(self, ids: torch.Tensor, targets: torch.Tensor,
             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Cross-entropy, optionally restricted to masked (assistant) positions."""
        logits = self(ids)
        flat = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        )
        if mask is None:
            return flat.mean()
        m = mask.reshape(-1).float()
        return (flat * m).sum() / m.sum().clamp(min=1.0)


class TinyTransformer(BaseLM):
    """Model A: 4-layer decoder, RMSNorm + RoPE + SwiGLU, 48,416 params."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        cos, sin = build_rope_cache(cfg.context_length, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        return self.head(self.norm(x))


class TinyGRU(BaseLM):
    """Model B: 2-layer GRU, no attention, 48,000 params."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.rnn = nn.GRU(cfg.d_model, cfg.gru_hidden, cfg.gru_layers, batch_first=True)
        self.proj = (nn.Linear(cfg.gru_hidden, cfg.d_model, bias=False)
                     if cfg.gru_hidden != cfg.d_model else nn.Identity())
        self.norm = RMSNorm(cfg.d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.embed(ids))
        return self.head(self.norm(self.proj(h)))


class TinyHybrid(BaseLM):
    """Model C: GRU for local structure, one attention block for lookback.

    The bet: a recurrent stack is a cheaper way to buy short-range fluency than
    four attention layers, leaving budget for a single block that can actually
    reach back across the conversation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        self.rnn = nn.GRU(cfg.d_model, cfg.gru_hidden, cfg.gru_layers, batch_first=True)
        self.proj = (nn.Linear(cfg.gru_hidden, cfg.d_model, bias=False)
                     if cfg.gru_hidden != cfg.d_model else nn.Identity())
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.ff = RMSNorm(cfg.d_model), SwiGLU(cfg.d_model, cfg.d_ff)
        self.norm = RMSNorm(cfg.d_model)
        cos, sin = build_rope_cache(cfg.context_length, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.embed(ids))
        x = self.proj(h)
        x = x + self.attn(self.n1(x), self.cos, self.sin)
        x = x + self.ff(self.n2(x))
        return self.head(self.norm(x))


ARCHES = {"transformer": TinyTransformer, "gru": TinyGRU, "hybrid": TinyHybrid}


def build_model(cfg: ModelConfig, budget: Optional[int] = None,
                strict: bool = True) -> BaseLM:
    """Build a model and refuse to return one that busts its budget."""
    budget = budget if budget is not None else cfg.budget
    model = ARCHES[cfg.arch](cfg)
    actual, predicted = model.n_params(), cfg.n_params()
    if actual != predicted:
        raise AssertionError(
            f"config.n_params() says {predicted:,} but the module has {actual:,}; "
            "the analytic counter and the code have drifted apart"
        )
    if strict and actual > budget:
        raise ValueError(f"{cfg.name} has {actual:,} params, over the {budget:,} budget")
    return model


def describe(model: BaseLM) -> str:
    cfg = model.cfg
    n = model.n_params()
    groups = {}
    for name, p in model.named_parameters():
        groups[name.split(".")[0]] = groups.get(name.split(".")[0], 0) + p.numel()
    lines = [f"PocketLM-{cfg.name} ({cfg.arch}): {n:,} params "
             f"({n / cfg.budget:.1%} of {cfg.budget:,} budget), "
             f"{n * 2 / 1024:.1f} KB at fp16",
             f"  vocab {cfg.vocab_size}  d_model {cfg.d_model}  layers {cfg.n_layers}  "
             f"heads {cfg.n_heads}  d_ff {cfg.d_ff}  ctx {cfg.context_length}"]
    for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {k:<10} {v:>7,}  {v / n:5.1%}")
    return "\n".join(lines)
