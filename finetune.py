#!/usr/bin/env python3
"""Fine-tune a trained checkpoint on new conversations.

Use this to push a personality further, or to fold in distilled data that
arrived after the main run:

    python finetune.py --arch transformer --source personality --steps 800
    python finetune.py --arch transformer --data-file mine.jsonl --steps 500

Loss is always assistant-only here, and the learning rate is an order of
magnitude below phase 1 -- at 48K parameters it is very easy to fine-tune the
English back out of the model.
"""

import argparse
import json
from pathlib import Path

import torch

from chat import load
from config import TrainConfig
from dataset import Corpus, Source, build_corpus, encode_conversation, load_rows
from model import describe
from train import evaluate, pick_device, run_phase
from config import Phase


def corpus_from_file(path: str, tok, ctx: int, seed: int) -> Corpus:
    rows = load_rows(Path(path))
    examples = [e for e in (encode_conversation(r, tok, ctx) for r in rows) if e]
    if not examples:
        raise SystemExit(f"no usable conversations in {path}")
    src = Source("custom", examples, ctx, tok.pad_id)
    return Corpus({"custom": src}, tok.pad_id, seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="50k")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default="data")
    ap.add_argument("--source", default="personality",
                    help="named split in data/, e.g. personality or synthetic")
    ap.add_argument("--data-file", default=None, help="a .jsonl of conversations instead")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    ckpt_path = args.checkpoint or f"checkpoints/{args.model}.pt"
    out = args.out or ckpt_path.replace(".pt", "-ft.pt")

    model, tok = load(ckpt_path, device)
    model.train()
    print(describe(model))
    ctx = model.cfg.context_length

    if args.data_file:
        corpus = corpus_from_file(args.data_file, tok, ctx, args.seed)
        val_corpus, sources = corpus, ["custom"]
    else:
        corpus = build_corpus(args.data, tok, ctx, "train", seed=args.seed)
        val_corpus = build_corpus(args.data, tok, ctx, "val", seed=args.seed)
        sources = [args.source]
    print("\ncorpus:")
    print(corpus.stats())

    tcfg = TrainConfig(batch_size=args.batch_size, warmup_steps=min(100, args.steps // 5),
                       seed=args.seed)
    before = evaluate(model, val_corpus, sources, "assistant", args.batch_size, 20, device)
    log = []
    run_phase(model, Phase(f"finetune:{sources[0]}", sources, args.steps, args.lr),
              corpus, val_corpus, tcfg, device, log)
    after = evaluate(model, val_corpus, sources, "assistant", args.batch_size, 20, device)
    print(f"\nval loss {before:.3f} -> {after:.3f}")

    torch.save({"config": model.cfg.to_dict(), "state_dict": model.state_dict(),
                "tokenizer": "checkpoints/tokenizer.json", "log": log}, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
