#!/usr/bin/env python3
"""Inference with numpy only -- no torch, no framework.

    python runtime_numpy.py --model checkpoints/transformer.pocketlm.npz

This exists to keep the deployment claim honest. If a 48K chatbot is supposed
to run on cheap hardware, there should be a path that does not drag in a
300 MB training framework to multiply a 32-wide matrix. Everything below is
the forward pass rewritten against the exported fp16 arrays.
"""

import argparse
import json
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from tokenizer import BPETokenizer      # pure python, no torch


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def rmsnorm(x, w, eps=1e-5):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * w


def rope(x, cos, sin):
    # x: [H, T, hd]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def _dequantize(buf: np.ndarray, scale: np.ndarray, bits: int, shape) -> np.ndarray:
    """Undo testing/quantize.py: unpack the codes, then rescale per row."""
    n = int(np.prod(shape))
    planes = np.unpackbits(buf)[: n * bits].reshape(n, bits).astype(np.int16)
    vals = (planes << np.arange(bits - 1, -1, -1)).sum(axis=1) - (1 << (bits - 1))
    return (vals.reshape(shape).astype(np.float32)
            * scale.astype(np.float32).reshape(-1, 1))


class NumpyPocketLM:
    def __init__(self, path: str):
        blob = np.load(path)
        meta = json.loads(bytes(blob["__meta__"]).decode())
        self.cfg = meta["config"]
        self.w: Dict[str, np.ndarray] = {
            k: blob[k].astype(np.float32) for k in blob.files
            if k != "__meta__" and not k.endswith((".q", ".s"))}
        # Quantized exports store bit-packed codes plus a per-row scale; unpack
        # once here so every path downstream still sees ordinary float32.
        qbits = meta.get("quant_bits")
        for name, shape in meta.get("quant", {}).items():
            self.w[name] = _dequantize(blob[name + ".q"], blob[name + ".s"],
                                       qbits, tuple(shape))
        t = meta["tokenizer"]
        self.tok = BPETokenizer(t["alphabet"], [tuple(m) for m in t["merges"]],
                                t["specials"], t.get("lowercase", False))
        hd = self.cfg["d_model"] // self.cfg["n_heads"]
        inv = 1.0 / (self.cfg["rope_theta"] ** (np.arange(0, hd, 2) / hd))
        pos = np.arange(self.cfg["context_length"])[:, None]
        self.cos, self.sin = np.cos(pos * inv), np.sin(pos * inv)

    # ------------------------------------------------------------ pieces

    def _attention(self, x, prefix: str):
        cfg, w = self.cfg, self.w
        T, d = x.shape
        H, KV = cfg["n_heads"], cfg["n_kv_heads"]
        hd = d // H
        q = (x @ w[f"{prefix}q.weight"].T).reshape(T, H, hd).transpose(1, 0, 2)
        k = (x @ w[f"{prefix}k.weight"].T).reshape(T, KV, hd).transpose(1, 0, 2)
        v = (x @ w[f"{prefix}v.weight"].T).reshape(T, KV, hd).transpose(1, 0, 2)
        cos, sin = self.cos[:T], self.sin[:T]
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        if KV != H:
            k, v = np.repeat(k, H // KV, axis=0), np.repeat(v, H // KV, axis=0)
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(hd)
        scores += np.triu(np.full((T, T), -1e30, dtype=np.float32), 1)
        y = (softmax(scores) @ v).transpose(1, 0, 2).reshape(T, d)
        return y @ w[f"{prefix}o.weight"].T

    def _swiglu(self, x, prefix: str):
        w = self.w
        return (silu(x @ w[f"{prefix}gate.weight"].T) *
                (x @ w[f"{prefix}up.weight"].T)) @ w[f"{prefix}down.weight"].T

    def _gru(self, x):
        w, h_size = self.w, self.cfg["gru_hidden"]
        out = x
        for layer in range(self.cfg["gru_layers"]):
            Wi, Wh = w[f"rnn.weight_ih_l{layer}"], w[f"rnn.weight_hh_l{layer}"]
            bi, bh = w[f"rnn.bias_ih_l{layer}"], w[f"rnn.bias_hh_l{layer}"]
            h = np.zeros(h_size, dtype=np.float32)
            seq = []
            for t in range(out.shape[0]):
                gi, gh = Wi @ out[t] + bi, Wh @ h + bh
                # PyTorch gate order is reset, update, new
                r = sigmoid(gi[:h_size] + gh[:h_size])
                z = sigmoid(gi[h_size:2 * h_size] + gh[h_size:2 * h_size])
                n = np.tanh(gi[2 * h_size:] + r * gh[2 * h_size:])
                h = (1 - z) * n + z * h
                seq.append(h)
            out = np.stack(seq)
        return out

    # ----------------------------------------------------------- forward

    def logits(self, ids: List[int]) -> np.ndarray:
        cfg, w = self.cfg, self.w
        ids = ids[-cfg["context_length"]:]
        x = w["embed.weight"][np.asarray(ids)]

        if cfg["arch"] == "transformer":
            for i in range(cfg["n_layers"]):
                x = x + self._attention(rmsnorm(x, w[f"blocks.{i}.n1.weight"]),
                                        f"blocks.{i}.attn.")
                x = x + self._swiglu(rmsnorm(x, w[f"blocks.{i}.n2.weight"]),
                                     f"blocks.{i}.ff.")
        else:
            x = self._gru(x)
            if "proj.weight" in w:
                x = x @ w["proj.weight"].T
            if cfg["arch"] == "hybrid":
                x = x + self._attention(rmsnorm(x, w["n1.weight"]), "attn.")
                x = x + self._swiglu(rmsnorm(x, w["n2.weight"]), "ff.")

        return rmsnorm(x, w["norm.weight"]) @ w["embed.weight"].T

    # ---------------------------------------------------------- sampling

    def reply(self, turns, memory=None, temperature=0.7, top_p=0.9,
              repetition_penalty=1.1, max_new_tokens=64, rng=None) -> str:
        rng = rng or np.random.default_rng(0)
        tok = self.tok
        ids, _ = tok.encode_turns(turns, memory=memory, open_reply=True)
        stop = {tok.eos_id, tok.user_id, tok.bos_id}
        banned = [tok.pad_id, tok.mem_id, tok.assistant_id, tok.unk_id]
        out: List[int] = []
        for _ in range(max_new_tokens):
            lg = self.logits(ids)[-1].copy()
            lg[banned] = -np.inf
            for t in set(out):
                lg[t] = lg[t] / repetition_penalty if lg[t] > 0 else lg[t] * repetition_penalty
            lg /= max(temperature, 1e-6)
            p = softmax(lg)
            order = np.argsort(-p)
            keep = np.searchsorted(np.cumsum(p[order]), top_p) + 1
            mask = np.zeros_like(p, dtype=bool)
            mask[order[:keep]] = True
            p = np.where(mask, p, 0.0)
            nxt = int(rng.choice(len(p), p=p / p.sum()))
            if nxt in stop:
                break
            ids.append(nxt)
            out.append(nxt)
        return tok.decode(out).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/main/500k.npz")
    ap.add_argument("--prompt", default=None, help="one-shot instead of a REPL")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    lm = NumpyPocketLM(args.model)
    n = sum(a.size for a in lm.w.values())
    print(f"numpy runtime: PocketLM-{lm.cfg['name']} ({lm.cfg['arch']}), "
          f"{n:,} params, no torch")

    if args.prompt:
        print(lm.reply([{"role": "user", "text": args.prompt}], temperature=args.temperature))
        return

    turns = []
    rng = np.random.default_rng(0)
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        turns.append({"role": "user", "text": line})
        out = lm.reply(turns[-8:], temperature=args.temperature, rng=rng)
        turns.append({"role": "assistant", "text": out})
        print(f"pocketlm> {out}")


if __name__ == "__main__":
    main()
