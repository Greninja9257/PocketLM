"""Sampling for a very small model.

Tiny models get worse when you let them ramble, so the defaults are tight:
temperature 0.7, top-p 0.9, a repetition penalty, and a hard cap of 64 new
tokens. Short replies do not just hide weakness -- they are what the training
data looks like, so they are also what the model is best at.

There is no KV cache. The context is 128 tokens and the model is 48K
parameters; recomputing the whole window each step costs less than the code to
avoid it would.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F


@dataclass
class SamplingConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0                     # 0 disables
    repetition_penalty: float = 1.1
    max_new_tokens: int = 64
    min_new_tokens: int = 1


def _apply_repetition_penalty(logits: torch.Tensor, prev: Sequence[int],
                              penalty: float) -> torch.Tensor:
    if penalty == 1.0 or not prev:
        return logits
    idx = torch.tensor(sorted(set(prev)), dtype=torch.long, device=logits.device)
    vals = logits[idx]
    logits[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def _filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k and top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0 < top_p < 1.0:
        ordered, order = torch.sort(logits, descending=True)
        probs = F.softmax(ordered, dim=-1).cumsum(dim=-1)
        cut = probs > top_p
        cut[..., 1:] = cut[..., :-1].clone()
        cut[..., 0] = False             # always keep the argmax
        logits = logits.masked_fill(torch.zeros_like(cut).scatter(0, order, cut), float("-inf"))
    return logits


@torch.no_grad()
def generate(model, prompt_ids: Sequence[int], cfg: SamplingConfig,
             stop_ids: Sequence[int] = (), banned_ids: Sequence[int] = (),
             device: Optional[torch.device] = None) -> List[int]:
    """Continue prompt_ids and return only the newly generated tokens."""
    model.eval()
    device = device or next(model.parameters()).device
    ctx = model.cfg.context_length
    ids = list(prompt_ids)
    out: List[int] = []
    stop, banned = set(stop_ids), list(banned_ids)

    for step in range(cfg.max_new_tokens):
        window = ids[-ctx:]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float()

        if banned:
            logits[torch.tensor(banned, dtype=torch.long, device=device)] = float("-inf")
        if step < cfg.min_new_tokens:
            for s in stop:
                logits[s] = float("-inf")
        logits = _apply_repetition_penalty(logits, out, cfg.repetition_penalty)

        if cfg.temperature <= 0:
            nxt = int(logits.argmax())
        else:
            logits = _filter(logits / cfg.temperature, cfg.top_k, cfg.top_p)
            nxt = int(torch.multinomial(F.softmax(logits, dim=-1), 1))

        if nxt in stop:
            break
        ids.append(nxt)
        out.append(nxt)
    return out


def respond(model, tok, turns: Sequence[dict], memory: Optional[str] = None,
            cfg: Optional[SamplingConfig] = None, device=None) -> str:
    """Full path from conversation history to a decoded assistant reply."""
    cfg = cfg or SamplingConfig()
    ctx = model.cfg.context_length
    turns = list(turns)
    while turns:
        ids, _ = tok.encode_turns(turns, memory=memory, open_reply=True)
        if len(ids) <= ctx:
            break
        turns = turns[1:]               # drop oldest turn, same rule as training
    if not turns:
        ids, _ = tok.encode_turns([], memory=memory, open_reply=True)

    new = generate(
        model, ids, cfg,
        # <eos> ends the reply; <user> means it started writing our lines for us.
        stop_ids=(tok.eos_id, tok.user_id, tok.bos_id),
        banned_ids=(tok.pad_id, tok.mem_id, tok.assistant_id, tok.unk_id),
        device=device,
    )
    return tok.decode(new).strip()
