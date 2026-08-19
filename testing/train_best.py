#!/usr/bin/env python3
"""Train and save the configurations the sweep found best.

sweep.py ranks configs but throws the weights away — it trains each candidate
briefly and only keeps the score. This retrains the winners at full length so
they can be evaluated against the released family on the same 900 prompts.
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

# The winners, and the shipped configs they beat, so both can be evaluated.
BEST = {
    "1k-best":  dict(vocab=40,  d=8,  L=1, ff=16, h=2, ctx=64,  budget=1_000),
    "10k-best": dict(vocab=192, d=16, L=2, ff=48, h=2, ctx=128, budget=10_000),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", default="all", choices=["all", *BEST])
    ap.add_argument("--data", default="testing/data")
    ap.add_argument("--tokenizer-dir", default="testing/checkpoints")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.device)
    names = list(BEST) if args.which == "all" else [args.which]
    for name in names:
        spec = BEST[name]
        torch.manual_seed(args.seed)
        cfg = ModelConfig(name=name, vocab_size=spec["vocab"], context_length=spec["ctx"],
                          d_model=spec["d"], n_layers=spec["L"], n_heads=spec["h"],
                          n_kv_heads=spec["h"], d_ff=spec["ff"], budget=spec["budget"])
        tok_path = str(Path(args.tokenizer_dir) / Path(tokenizer_path(cfg.vocab_size)).name)
        tok = BPETokenizer.load(tok_path)
        model = build_model(cfg).to(device)
        print(f"\n=== {name}: {model.n_params():,} params "
              f"({model.n_params()/cfg.budget:.1%} of {cfg.budget:,})")

        corpus = build_corpus(args.data, tok, cfg.context_length, "train", seed=args.seed)
        val = build_corpus(args.data, tok, cfg.context_length, "val", seed=args.seed)
        sources = corpus.available(["dialogue", "behavior", "personality"])
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01,
                                betas=(0.9, 0.95))
        t0 = time.time()
        for step in range(args.steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, 3e-3, 3e-4, 200)
            x, y, m = corpus.sample(sources, args.batch_size, "assistant")
            loss = model.loss(x.to(device), y.to(device), m.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (step + 1) % 1000 == 0:
                print(f"  step {step+1:>5}/{args.steps}  loss {loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

        model.eval()
        with torch.no_grad():
            v = sum(model.loss(*[t.to(device) for t in
                    val.sample(sources, args.batch_size, "assistant")]).item()
                    for _ in range(20)) / 20
        print(f"  val loss {v:.4f}  ppl {math.exp(min(v, 20)):.2f}")
        out = f"testing/checkpoints/{name}.pt"
        torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict(),
                    "tokenizer": tok_path}, out)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
