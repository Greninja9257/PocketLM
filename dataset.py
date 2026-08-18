"""Turning conversations into masked training batches.

The important idea here is section 8 of the plan: loss masking. Given

    <user> what's your name? <eos> <assistant> I'm PocketLM! <eos>

the model is scored only on `I'm PocketLM! <eos>`. Predicting what the *user*
will type is a different and much harder task, and at 48K parameters there is
no capacity to spare for it. Phase 1 is the exception -- learning English at
all requires every token.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from tokenizer import BPETokenizer


def load_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def encode_conversation(row: dict, tok: BPETokenizer, ctx: int) -> Optional[Tuple[List[int], List[int]]]:
    """Encode one conversation, dropping oldest turns until it fits the window.

    Truncating from the front rather than the back is deliberate: the reply is
    the training signal, so it must survive. Losing the start of a long
    conversation is exactly what the runtime does too, which keeps training and
    inference seeing the same shape of context.
    """
    turns = row["turns"]
    memory = row.get("memory")
    while turns:
        ids, mask = tok.encode_turns(turns, memory=memory)
        if len(ids) <= ctx + 1:
            return (ids, mask) if any(mask) else None
        turns = turns[1:] if len(turns) > 2 else turns[:-1] if len(turns) > 1 else []
    return None


def pack_text(rows: Sequence[dict], tok: BPETokenizer, ctx: int) -> List[Tuple[List[int], List[int]]]:
    """Phase-1 packing: concatenate documents and cut into full windows."""
    stream: List[int] = []
    for row in rows:
        stream.append(tok.bos_id)
        stream.extend(tok.encode(row["text"]))
        stream.append(tok.eos_id)
    out = []
    step = ctx + 1
    for i in range(0, len(stream) - step, step):
        chunk = stream[i:i + step]
        out.append((chunk, [1] * len(chunk)))
    return out


class Source:
    """One named pool of examples, pre-encoded to tensors."""

    def __init__(self, name: str, examples: List[Tuple[List[int], List[int]]],
                 ctx: int, pad_id: int):
        self.name = name
        self.ctx = ctx
        self.pad_id = pad_id
        n = len(examples)
        self.ids = torch.full((n, ctx + 1), pad_id, dtype=torch.long)
        self.mask = torch.zeros((n, ctx + 1), dtype=torch.bool)
        for i, (seq, m) in enumerate(examples):
            L = min(len(seq), ctx + 1)
            self.ids[i, :L] = torch.tensor(seq[:L], dtype=torch.long)
            self.mask[i, :L] = torch.tensor([bool(v) for v in m[:L]], dtype=torch.bool)

    def __len__(self) -> int:
        return self.ids.shape[0]

    def batch(self, idx: torch.Tensor, loss_on: str):
        seq, msk = self.ids[idx], self.mask[idx]
        x, y = seq[:, :-1], seq[:, 1:]
        # "all" scores every real token (phase 1); otherwise only assistant tokens.
        m = (y != self.pad_id) if loss_on == "all" else msk[:, 1:]
        return x, y, m


class Corpus:
    """The full training corpus: named sources plus weighted sampling."""

    def __init__(self, sources: Dict[str, Source], pad_id: int, seed: int = 0):
        self.sources = sources
        self.pad_id = pad_id
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)

    def available(self, names: Sequence[str]) -> List[str]:
        return [n for n in names if n in self.sources and len(self.sources[n]) > 0]

    def sample(self, names: Sequence[str], batch_size: int, loss_on: str,
               weights: Optional[Dict[str, float]] = None):
        names = self.available(names)
        if not names:
            raise ValueError(f"no data for any of {names}; run scripts/build_corpus.py")
        w = [float((weights or {}).get(n, 1.0)) * len(self.sources[n]) for n in names]
        total = sum(w)
        xs, ys, ms = [], [], []
        for name, weight in zip(names, w):
            k = max(1, round(batch_size * weight / total))
            src = self.sources[name]
            idx = torch.randint(0, len(src), (k,), generator=self.gen)
            x, y, m = src.batch(idx, loss_on)
            xs.append(x); ys.append(y); ms.append(m)
        x = torch.cat(xs)[:batch_size]
        y = torch.cat(ys)[:batch_size]
        m = torch.cat(ms)[:batch_size]
        if not m.any():                 # degenerate batch; retry rather than divide by zero
            return self.sample(names, batch_size, loss_on, weights)
        return x, y, m

    def stats(self) -> str:
        lines = []
        for name, src in self.sources.items():
            if not len(src):
                continue
            real = int((src.ids != self.pad_id).sum())
            scored = int(src.mask.sum())
            lines.append(f"  {name:<12} {len(src):>7,} examples  {real:>10,} tokens  "
                         f"{scored:>9,} scored ({scored / max(real, 1):.0%})")
        return "\n".join(lines)


def build_corpus(data_dir: str, tok: BPETokenizer, ctx: int, split: str = "train",
                 seed: int = 0) -> Corpus:
    root = Path(data_dir)
    sources: Dict[str, Source] = {}
    for name in ("language", "dialogue", "behavior", "personality", "synthetic"):
        rows = load_rows(root / name / f"{split}.jsonl")
        if not rows:
            continue
        if name == "language":
            examples = pack_text(rows, tok, ctx)
        else:
            examples = [e for e in (encode_conversation(r, tok, ctx) for r in rows) if e]
        if examples:
            sources[name] = Source(name, examples, ctx, tok.pad_id)
    return Corpus(sources, tok.pad_id, seed)
