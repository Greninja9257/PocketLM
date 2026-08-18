#!/usr/bin/env python3
"""Export a checkpoint to fp16 weights + the tokenizer, and check the size.

    python export.py --model 50k
    -> checkpoints/50k.pocketlm.npz  (~99 KB)

The point of the exercise: 48,416 params x 2 bytes is 94.6 KB. Anything much
larger than that in the exported file is overhead worth knowing about.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chat import load


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="50k")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = args.checkpoint or f"checkpoints/{args.model}.pt"
    out = Path(args.out or ckpt.replace(".pt", ".pocketlm.npz"))

    model, tok = load(ckpt, torch.device("cpu"))
    arrays = {k: v.detach().cpu().numpy().astype(np.float16)
              for k, v in model.state_dict().items()}
    meta = {"config": model.cfg.to_dict(),
            "tokenizer": {"specials": tok.specials, "alphabet": tok.alphabet,
                          "merges": [list(m) for m in tok.merges],
                          "lowercase": tok.lowercase}}
    np.savez_compressed(out, __meta__=np.frombuffer(
        json.dumps(meta).encode(), dtype=np.uint8), **arrays)

    n_params = sum(a.size for a in arrays.values())
    size = out.stat().st_size
    print(f"{ckpt} -> {out}")
    print(f"  {n_params:,} params  =  {n_params * 2 / 1024:.1f} KB of fp16 weights")
    print(f"  file on disk: {size / 1024:.1f} KB (weights + tokenizer + config, compressed)")
    print(f"\nrun it with no torch installed:  python runtime_numpy.py --model {out}")


if __name__ == "__main__":
    main()
