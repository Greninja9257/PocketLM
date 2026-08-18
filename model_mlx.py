"""MLX implementation of the PocketLM transformer, for Apple Silicon.

MLX is roughly 1.6-1.9x faster than PyTorch on an M1 for models this size
(measured, batch 64: 50k 20.8 vs 13.4 it/s; 1m 1.9 vs 1.1 it/s). The gap comes
from kernel-launch overhead rather than arithmetic -- these models are so small
that framework overhead dominates, which is exactly the regime MLX's lazy
graph and unified memory handle well.

This module is a *training accelerator only*. Every parameter is given the same
name as its PyTorch counterpart, so a model trained here saves a checkpoint
that chat.py, eval/, export.py and runtime_numpy.py load without knowing MLX
was ever involved. `assert_parity` in this file is what keeps that promise
honest.

Only the transformer architecture is implemented, which covers all seven
members of the family. The GRU and hybrid variants exist at the 50K size only,
where PyTorch trains them in ~15 minutes anyway.
"""

import math
from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d, d_ff, bias=False)
        self.up = nn.Linear(d, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    """Causal attention with optional grouped/multi-query key-value sharing."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q = nn.Linear(d, self.n_heads * self.hd, bias=False)
        self.k = nn.Linear(d, self.n_kv * self.hd, bias=False)
        self.v = nn.Linear(d, self.n_kv * self.hd, bias=False)
        self.o = nn.Linear(self.n_heads * self.hd, d, bias=False)
        # traditional=True is the interleaved (GPT-J) rotation, which is what
        # the PyTorch apply_rope in model.py does. Getting this wrong produces
        # a model that trains fine and then decodes garbage under the other
        # runtime, so it is asserted in assert_parity.
        self.rope = nn.RoPE(self.hd, traditional=True, base=cfg.rope_theta)

    def __call__(self, x):
        B, T, _ = x.shape
        q = self.q(x).reshape(B, T, self.n_heads, self.hd).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, T, self.n_kv, self.hd).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, self.n_kv, self.hd).transpose(0, 2, 1, 3)
        q, k = self.rope(q), self.rope(k)
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = mx.repeat(k, rep, axis=1)
            v = mx.repeat(v, rep, axis=1)
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.hd), mask="causal")
        return self.o(y.transpose(0, 2, 1, 3).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n1 = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.attn = Attention(cfg)
        self.n2 = nn.RMSNorm(cfg.d_model, eps=1e-5)
        self.ff = SwiGLU(cfg.d_model, cfg.d_ff)

    def __call__(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ff(self.n2(x))


class TinyTransformerMLX(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layers)]
        self.norm = nn.RMSNorm(cfg.d_model, eps=1e-5)

    def __call__(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x)
        return self.embed.as_linear(self.norm(x))       # tied head

    def n_params(self) -> int:
        return sum(v.size for _, v in tree_flatten(self.parameters()))

    def loss(self, ids, targets, mask):
        logits = self(ids)
        V = logits.shape[-1]
        ce = nn.losses.cross_entropy(
            logits.reshape(-1, V).astype(mx.float32), targets.reshape(-1), reduction="none")
        m = mask.reshape(-1).astype(mx.float32)
        return (ce * m).sum() / mx.maximum(m.sum(), 1.0)


def build_model_mlx(cfg: ModelConfig, strict: bool = True) -> TinyTransformerMLX:
    if cfg.arch != "transformer":
        raise ValueError(
            f"the MLX backend implements the transformer only; {cfg.name!r} is "
            f"{cfg.arch!r}. Train it with --backend torch.")
    model = TinyTransformerMLX(cfg)
    mx.eval(model.parameters())
    actual, predicted = model.n_params(), cfg.n_params()
    if actual != predicted:
        raise AssertionError(f"config says {predicted:,} params, MLX module has {actual:,}")
    if strict and actual > cfg.budget:
        raise ValueError(f"{cfg.name} has {actual:,} params, over the {cfg.budget:,} budget")
    return model


# ------------------------------------------------------- torch interop

def to_torch_state_dict(model: TinyTransformerMLX) -> Dict:
    """Convert MLX parameters into a PyTorch state_dict.

    Names already match by construction, so this is a pure dtype/container
    conversion. The one real transform is the RoPE cache: PyTorch keeps
    cos/sin as non-persistent buffers, and MLX computes them on the fly, so
    there is nothing to carry across.
    """
    import numpy as np
    import torch
    out = {}
    for name, value in tree_flatten(model.parameters()):
        out[name] = torch.from_numpy(np.array(value, copy=True)).float()
    return out


def from_torch_state_dict(model: TinyTransformerMLX, state_dict) -> TinyTransformerMLX:
    """Load PyTorch weights into the MLX module (for resuming or fine-tuning)."""
    import numpy as np
    from mlx.utils import tree_unflatten
    params = [(k, mx.array(np.asarray(v.detach().cpu().float().numpy())))
              for k, v in state_dict.items() if not k.endswith((".cos", ".sin"))]
    model.update(tree_unflatten(params))
    mx.eval(model.parameters())
    return model


def assert_parity(cfg: ModelConfig, atol: float = 2e-4) -> float:
    """Build both models, copy MLX weights into torch, compare logits.

    This is the load-bearing test for the whole backend. If it passes, an
    MLX-trained checkpoint is indistinguishable from a torch-trained one to
    every consumer downstream.
    """
    import numpy as np
    import torch
    from model import build_model

    mlx_model = build_model_mlx(cfg)
    torch_model = build_model(cfg)
    torch_model.load_state_dict(to_torch_state_dict(mlx_model), strict=True)
    torch_model.eval()

    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg.vocab_size, size=(2, min(cfg.context_length, 48)))
    with torch.no_grad():
        t_out = torch_model(torch.tensor(ids, dtype=torch.long)).numpy()
    m_out = np.array(mlx_model(mx.array(ids)))

    diff = float(np.abs(t_out - m_out).max())
    agree = float((t_out.argmax(-1) == m_out.argmax(-1)).mean())
    if diff > atol or agree < 1.0:
        raise AssertionError(
            f"{cfg.name}: MLX and PyTorch disagree — max|diff| {diff:.2e}, "
            f"argmax agreement {agree:.1%}. Check the RoPE convention.")
    return diff


if __name__ == "__main__":
    from config import FAMILY
    print("MLX <-> PyTorch parity (same weights, same logits):")
    for name, cfg in FAMILY.items():
        print(f"  {name:>5}: max|diff| = {assert_parity(cfg):.2e}  ok")
