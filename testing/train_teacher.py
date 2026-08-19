#!/usr/bin/env python3
"""Train a mid-sized teacher that shares the student's vocabulary.

KD computes a KL over the output distribution, so teacher and student must have
the same vocabulary. None of the released models qualify -- the 1K student uses
a 40-token vocabulary that no family member shares -- so the teacher is trained
from scratch here.
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import ModelConfig, tokenizer_path
from dataset import build_corpus
from model import build_model
from tokenizer import BPETokenizer
from train import lr_at, pick_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vocab", type=int, default=40)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--n-layers", type=int, default=5)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=144)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--data", default="testing/data")
    ap.add_argument("--tokenizer-dir", default="testing/checkpoints")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    tok = BPETokenizer.load(str(Path(args.tokenizer_dir) /
                                Path(tokenizer_path(args.vocab)).name))
    cfg = ModelConfig(name=f"teacher-v{args.vocab}", vocab_size=args.vocab,
                      context_length=args.ctx, d_model=args.d_model,
                      n_layers=args.n_layers, n_heads=args.n_heads,
                      n_kv_heads=args.n_heads, d_ff=args.d_ff,
                      budget=10**9)
    model = build_model(cfg).to(device)
    print(f"teacher {cfg.name}: {model.n_params():,} params")

    corpus = build_corpus(args.data, tok, args.ctx, "train", seed=args.seed)
    val = build_corpus(args.data, tok, args.ctx, "val", seed=args.seed)
    sources = corpus.available(["dialogue", "behavior", "personality"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, 2e-4, 200)
        x, y, m = corpus.sample(sources, args.batch_size, "assistant")
        loss = model.loss(x.to(device), y.to(device), m.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"  step {step+1:>5}/{args.steps}  loss {loss.item():.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    model.eval()
    with torch.no_grad():
        v = sum(model.loss(*[t.to(device) for t in val.sample(sources, args.batch_size,
                "assistant")]).item() for _ in range(20)) / 20
    print(f"teacher val loss {v:.4f}  ppl {math.exp(min(v,20)):.2f}")

    out = args.out or f"testing/checkpoints/teacher-v{args.vocab}.pt"
    torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict(),
                "tokenizer": str(Path(args.tokenizer_dir) /
                                 Path(tokenizer_path(args.vocab)).name)}, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
