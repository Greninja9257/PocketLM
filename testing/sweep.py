#!/usr/bin/env python3
"""Architecture search at a fixed parameter budget.

The family configs in config.py were chosen by hand: pick a width, pick a
depth, solve d_ff for whatever budget is left. That is one point in a large
space, and at 984 parameters the choice of where to spend them matters far more
than it does at a billion. This searches the space instead of guessing.

The dominant tension is the embedding table. It costs vocab x d_model, so at
1K parameters a 64-token vocabulary already eats 52% of everything. Shrinking
the vocabulary frees parameters for layers but lengthens sequences (fewer
characters per token), which makes every dependency longer-range. There is no
way to reason that tradeoff out from first principles at this scale -- it has
to be measured.

    python testing/sweep.py --budget 1k  --steps 1200
    python testing/sweep.py --budget 10k --steps 1500

Scores by validation loss on assistant tokens only, which is what the model is
actually asked to produce.
"""

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import ModelConfig, TrainConfig, tokenizer_path
from dataset import build_corpus
from model import build_model
from tokenizer import BPETokenizer
from train import lr_at, pick_device

# Search spaces. Vocabulary is included because it is the single largest lever
# at these sizes, not because it is a hyperparameter in the usual sense.
SPACES = {
    "1k": dict(budget=1_000,
               vocab=[32, 40, 48, 56, 64, 80, 96],
               d_model=[8, 12, 16],
               n_layers=[1, 2, 3],
               d_ff=[4, 8, 12, 16, 20, 24, 32],
               n_heads=[1, 2],
               ctx=[64]),
    "10k": dict(budget=10_000,
                vocab=[96, 128, 160, 192, 256, 320, 384],
                d_model=[16, 24, 32],
                n_layers=[2, 3, 4, 5],
                d_ff=[16, 24, 32, 40, 48, 56, 64],
                n_heads=[2, 4],
                ctx=[128]),
}


def candidates(space):
    """Every config that is valid and fits, sorted by how much budget it uses."""
    out = []
    for v, d, L, ff, h, ctx in itertools.product(
            space["vocab"], space["d_model"], space["n_layers"],
            space["d_ff"], space["n_heads"], space["ctx"]):
        if d % h or (d // h) % 2:            # RoPE needs an even head dimension
            continue
        cfg = ModelConfig(name=f"v{v}-d{d}-L{L}-ff{ff}-h{h}", vocab_size=v,
                          context_length=ctx, d_model=d, n_layers=L, n_heads=h,
                          n_kv_heads=h, d_ff=ff, budget=space["budget"])
        n = cfg.n_params()
        if n > space["budget"]:
            continue
        # Anything leaving more than 12% of the budget unspent is dominated by
        # a bigger sibling; there is no reason to train it.
        if n < space["budget"] * 0.88:
            continue
        out.append((cfg, n))
    return sorted(out, key=lambda t: -t[1])


def train_and_score(cfg, tok, data_dir, steps, device, batch_size, seed):
    torch.manual_seed(seed)
    model = build_model(cfg).to(device)
    corpus = build_corpus(data_dir, tok, cfg.context_length, "train", seed=seed)
    val = build_corpus(data_dir, tok, cfg.context_length, "val", seed=seed)
    sources = corpus.available(["dialogue", "behavior", "personality"])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01,
                            betas=(0.9, 0.95))
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, 3e-3, 3e-4, min(100, steps // 10))
        x, y, m = corpus.sample(sources, batch_size, "assistant")
        loss = model.loss(x.to(device), y.to(device), m.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(20):
            x, y, m = val.sample(sources, batch_size, "assistant")
            losses.append(model.loss(x.to(device), y.to(device), m.to(device)).item())
    return sum(losses) / len(losses), model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", default="1k", choices=list(SPACES))
    ap.add_argument("--data", default="testing/data")
    ap.add_argument("--tokenizer-dir", default="testing/checkpoints")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top", type=int, default=0, help="only train the N largest configs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    space = SPACES[args.budget]
    cands = candidates(space)
    if args.top:
        cands = cands[:args.top]
    print(f"{len(cands)} candidate configs fit the {space['budget']:,} budget "
          f"(>=85% used)\n")

    device = pick_device(args.device)
    toks, skipped = {}, set()
    for v in sorted({c.vocab_size for c, _ in cands}):
        p = Path(args.tokenizer_dir) / Path(tokenizer_path(v)).name
        if p.exists():
            toks[v] = BPETokenizer.load(str(p))
        else:
            skipped.add(v)
    if skipped:
        # make_tokenizers.py rejects vocabularies too small to spell "PocketLM".
        print(f"skipping vocab sizes with no viable tokenizer: {sorted(skipped)}")
        cands = [(c, n) for c, n in cands if c.vocab_size not in skipped]
        print(f"{len(cands)} candidates remain\n")
    if not cands:
        raise SystemExit("no candidates left — run testing/make_tokenizers.py first")

    results = []
    t0 = time.time()
    for i, (cfg, n) in enumerate(cands, 1):
        tok = toks[cfg.vocab_size]
        loss, _ = train_and_score(cfg, tok, args.data, args.steps, device,
                                  args.batch_size, args.seed)
        results.append((loss, cfg, n))
        print(f"  [{i:>3}/{len(cands)}] {cfg.name:<26} {n:>6,}p  "
              f"val {loss:.4f}  ppl {math.exp(min(loss, 20)):>7.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    results.sort(key=lambda t: t[0])
    print(f"\n=== best configs for the {space['budget']:,} budget")
    print(f"  {'rank':<6}{'config':<26}{'params':>8}{'val loss':>10}{'ppl':>9}")
    for r, (loss, cfg, n) in enumerate(results[:10], 1):
        print(f"  {r:<6}{cfg.name:<26}{n:>8,}{loss:>10.4f}{math.exp(min(loss,20)):>9.2f}")

    out = Path(args.tokenizer_dir).parent / f"sweep-{args.budget}.json"
    out.write_text(json.dumps(
        [{"config": c.to_dict(), "params": n, "val_loss": l} for l, c, n in results],
        indent=1))
    print(f"\nwrote {out}")
    best = results[0][1]
    print(f"\nbest: {best.name}  vocab={best.vocab_size} d={best.d_model} "
          f"L={best.n_layers} ff={best.d_ff} heads={best.n_heads}")


if __name__ == "__main__":
    main()
