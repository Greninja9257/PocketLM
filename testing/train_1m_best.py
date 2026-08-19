#!/usr/bin/env python3
"""The best 1M-parameter PocketLM this project knows how to build.

Every choice here is taken from a measurement made elsewhere in the repo rather
than from taste:

  data          The dev branch showed real dialogue is worth more than any
                architecture change -- 500k went from 3.06 to 1.30 bits/char.
                This uses 26 MB of it, 2.4x what dev used, with 105,555
                distinct user turns against the synthetic generator's 280.

  steps         The shipped 1m runs at STEPS_SCALE 0.35 (4,375 steps) and has
                the family's lowest perplexity but regresses on the behavioural
                eval -- it is undertrained, not worse. This runs the full
                curriculum.

  depth vs FFN  The 10K architecture sweep found two layers beating three and
                the budget better spent on FFN width. Applied moderately here:
                five layers instead of six, d_ff 256 instead of 192.

  vocabulary    At 1M the embedding table is only ~13% of the budget, so a
                larger vocabulary is affordable and buys compression on real
                text, where the templated corpus needed far less.

    python testing/train_1m_best.py
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from config import DEV_PHASES, ModelConfig, TrainConfig, tokenizer_path
from dataset import build_corpus
from model import build_model
from tokenizer import BPETokenizer
from train import pick_backend, pick_device

# 984,448 params, 98.4% of the 1,000,000 budget.
CONFIG = ModelConfig(name="1m-best", vocab_size=1280, context_length=256,
                     d_model=128, n_layers=5, n_heads=8, n_kv_heads=8,
                     d_ff=256, budget=1_000_000)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="testing/data")
    ap.add_argument("--tokenizer-dir", default="testing/checkpoints")
    ap.add_argument("--out", default="testing/checkpoints/1m-best.pt")
    ap.add_argument("--steps-scale", type=float, default=1.0)
    ap.add_argument("--lr-scale", type=float, default=0.4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    CONFIG.validate()
    torch.manual_seed(args.seed)
    tok_path = Path(args.tokenizer_dir) / Path(tokenizer_path(CONFIG.vocab_size)).name
    if not tok_path.exists():
        raise SystemExit(f"no tokenizer at {tok_path} — train one first:\n"
                         f"  python scripts/train_tokenizer.py --data {args.data} "
                         f"--out-dir {args.tokenizer_dir} --vocab-size {CONFIG.vocab_size}")
    tok = BPETokenizer.load(str(tok_path))

    backend = pick_backend(args.backend, CONFIG.arch)
    device = pick_device("auto") if backend == "torch" else None
    print(f"PocketLM-1m-best: {CONFIG.n_params():,} params "
          f"({CONFIG.n_params()/CONFIG.budget:.1%} of budget)")
    print(f"  vocab {CONFIG.vocab_size}  d_model {CONFIG.d_model}  "
          f"layers {CONFIG.n_layers}  heads {CONFIG.n_heads}  d_ff {CONFIG.d_ff}  "
          f"ctx {CONFIG.context_length}")
    print(f"  backend {backend}   tokenizer {tok_path}")

    corpus = build_corpus(args.data, tok, CONFIG.context_length, "train", seed=args.seed)
    val = build_corpus(args.data, tok, CONFIG.context_length, "val", seed=args.seed)
    print("\ncorpus:")
    print(corpus.stats())

    tcfg = TrainConfig(batch_size=args.batch_size, seed=args.seed)
    log = []
    if backend == "mlx":
        from model_mlx import build_model_mlx, to_torch_state_dict
        import mlx.core as mx
        import train_mlx
        mx.random.seed(args.seed)
        model = build_model_mlx(CONFIG)
        runner = train_mlx.run_phase
    else:
        from train import run_phase
        model = build_model(CONFIG).to(device)
        runner = None

    t0 = time.time()
    for phase in DEV_PHASES:
        scaled = type(phase)(phase.name, phase.sources,
                             max(1, int(phase.steps * args.steps_scale)),
                             phase.lr * args.lr_scale, phase.loss_on, phase.weights)
        if backend == "mlx":
            runner(model, scaled, corpus, val, tcfg, log)
        else:
            from train import run_phase
            run_phase(model, scaled, corpus, val, tcfg, device, log)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    state = (to_torch_state_dict(model) if backend == "mlx"
             else model.state_dict())
    torch.save({"config": CONFIG.to_dict(), "state_dict": state,
                "tokenizer": str(tok_path), "log": log, "backend": backend}, args.out)
    print(f"\nsaved -> {args.out}  ({CONFIG.n_params():,} params, "
          f"{(time.time()-t0)/60:.0f} min)")


if __name__ == "__main__":
    main()
