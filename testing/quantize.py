#!/usr/bin/env python3
"""Post-training quantization for PocketLM checkpoints.

The family measures itself in parameters, but what actually ships is a *file*,
and the two stop tracking each other once the weights are not fp32. This asks
how few bits per weight a model this small can survive on.

Method: symmetric per-output-channel quantization of the 2D weight matrices.
Per-channel rather than per-tensor because one badly scaled row otherwise sets
the step size for the whole matrix. LayerNorm gains and the embedding table are
selectable separately, since they behave very differently under the same
treatment -- see --skip.

    python testing/quantize.py --bits 8 4 3 2
    python testing/quantize.py --bits 4 --skip embed --out models/testing/1m-best-q4.npz
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def quantize(w: np.ndarray, bits: int):
    """Symmetric per-row quantization. Returns (codes, scales, dequantized)."""
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(w).max(axis=1, keepdims=True) / qmax
    scale[scale == 0] = 1.0                      # a dead row would divide by zero
    codes = np.clip(np.rint(w / scale), -qmax - 1, qmax)
    return codes.astype(np.int8), scale.astype(np.float32), codes * scale


def packed_bytes(codes: np.ndarray, bits: int) -> int:
    """Bytes the codes occupy once packed, with no wasted high bits."""
    return math.ceil(codes.size * bits / 8)


def pack(codes: np.ndarray, bits: int) -> np.ndarray:
    """Bit-pack signed codes into bytes. 4-bit really costs 4 bits, not 8."""
    off = codes.astype(np.int16) + (1 << (bits - 1))          # to unsigned
    flat = off.reshape(-1).astype(np.uint8)
    bit_planes = ((flat[:, None] >> np.arange(bits - 1, -1, -1)) & 1).astype(np.uint8)
    return np.packbits(bit_planes.reshape(-1))


def unpack(buf: np.ndarray, bits: int, shape) -> np.ndarray:
    n = int(np.prod(shape))
    planes = np.unpackbits(buf)[: n * bits].reshape(n, bits).astype(np.int16)
    vals = (planes << np.arange(bits - 1, -1, -1)).sum(axis=1)
    return (vals - (1 << (bits - 1))).reshape(shape)


def write_npz(ckpt: str, out: str, bits: int, skip=()) -> int:
    """Write a quantized model the numpy runtime can load: codes + scales."""
    import json as _json
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    arrays, meta = {}, {"quant_bits": bits, "quant": {}}
    for k, v in d["state_dict"].items():
        w = v.detach().cpu().numpy().astype(np.float32)
        if w.ndim < 2 or any(sk in k for sk in skip):
            arrays[k] = w.astype(np.float16)
            continue
        codes, scale, _ = quantize(w, bits)
        arrays[k + ".q"] = pack(codes, bits)
        arrays[k + ".s"] = scale.astype(np.float16)
        meta["quant"][k] = list(w.shape)

    cfg = d["config"] if isinstance(d["config"], dict) else d["config"].to_dict()
    meta["config"] = cfg
    meta["tokenizer"] = _json.loads(Path(d["tokenizer"]).read_text())
    arrays["__meta__"] = np.frombuffer(_json.dumps(meta).encode(), dtype=np.uint8)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    return Path(out).stat().st_size


def quantize_state(sd, bits: int, skip=()):
    """Quantize every 2D weight, leaving 1D gains (LayerNorm) alone."""
    out, stats = {}, {"quantized": 0, "kept_fp16": 0, "bytes": 0}
    for k, v in sd.items():
        w = v.detach().cpu().numpy().astype(np.float32)
        skip_this = w.ndim < 2 or any(s in k for s in skip)
        if skip_this:
            # 1D gains are 128 values each and set the scale of everything
            # downstream of them; quantizing them buys nothing and costs a lot.
            out[k] = w.astype(np.float16).astype(np.float32)
            stats["kept_fp16"] += w.size
            stats["bytes"] += w.size * 2
        else:
            codes, scale, deq = quantize(w, bits)
            out[k] = deq
            stats["quantized"] += w.size
            stats["bytes"] += packed_bytes(codes, bits) + scale.size * 4
    return out, stats


def load_dequantized(ckpt, bits, skip=()):
    """A model with quantize->dequantize applied, ready to evaluate."""
    from config import ModelConfig
    from model import build_model
    d = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**d["config"]) if isinstance(d["config"], dict) else d["config"]
    qsd, stats = quantize_state(d["state_dict"], bits, skip)
    model = build_model(cfg)
    model.load_state_dict({k: torch.tensor(v) for k, v in qsd.items()})
    model.eval()
    return model, stats, d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="testing/checkpoints/1m-best.pt")
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4, 3, 2])
    ap.add_argument("--skip", nargs="*", default=[],
                    help="substrings of tensor names to leave in fp16, e.g. embed")
    ap.add_argument("--data", default="testing/data")
    ap.add_argument("--out", default=None, help="write one quantized .npz")
    args = ap.parse_args()

    from chat import load
    from scripts.benchmark_external import bpc_pocketlm, dialogue_turns

    turns = dialogue_turns(args.data)
    _, tok = load(args.checkpoint, torch.device("cpu"))
    d = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    n_params = sum(v.numel() for v in d["state_dict"].values())

    def bpc(model):
        import types
        from scripts import benchmark_external as be
        # bpc_pocketlm loads from a path; reuse its maths against a live model.
        total, ctx = 0.0, model.cfg.context_length
        ids = []
        for conv in turns:
            cid, _ = tok.encode_turns(conv)
            ids.extend(cid)
        with torch.no_grad():
            for i in range(0, len(ids) - 1, ctx):
                chunk = ids[i:i + ctx + 1]
                if len(chunk) < 2:
                    break
                x = torch.tensor([chunk[:-1]]); y = torch.tensor([chunk[1:]])
                lg = model(x)
                total += float(torch.nn.functional.cross_entropy(
                    lg.reshape(-1, lg.size(-1)), y.reshape(-1), reduction="sum"))
        chars = len(" ".join(t["text"] for c in turns for t in c))
        return total / math.log(2) / chars

    base_model, _ = load(args.checkpoint, torch.device("cpu"))[0], None
    base = bpc(base_model)
    print(f"{args.checkpoint}   {n_params:,} params")
    print(f"  fp32 baseline: {base:.3f} bits/char\n")
    print(f'{"bits":>5}{"weights KB":>12}{"vs fp16":>9}{"bits/char":>11}{"delta":>9}')
    print(f'{"fp16":>5}{n_params*2/1024:>12.0f}{"1.0x":>9}{base:>11.3f}{"—":>9}')

    for b in args.bits:
        model, stats, _ = load_dequantized(args.checkpoint, b, args.skip)
        v = bpc(model)
        kb = stats["bytes"] / 1024
        print(f'{b:>5}{kb:>12.0f}{n_params*2/stats["bytes"]:>8.1f}x'
              f'{v:>11.3f}{v-base:>+9.3f}')

    if args.out:
        b = args.bits[0]
        size = write_npz(args.checkpoint, args.out, b, args.skip)
        import os
        ref = "models/testing/1m-best.npz"
        base_size = os.path.getsize(ref) if os.path.exists(ref) else 0
        print(f"\nwrote {args.out}  ({size:,} bytes at {b}-bit"
              + (f", {base_size/size:.1f}x smaller than the fp16 export)" if base_size else ")"))


if __name__ == "__main__":
    main()
