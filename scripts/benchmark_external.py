#!/usr/bin/env python3
"""Compare PocketLM against other small language models.

Comparing models with different tokenizers is a trap: perplexity is per-token,
so a model with a bigger vocabulary looks better for free. **Bits per character**
is the fix — total negative log-likelihood divided by the number of characters,
which is invariant to how the text was split.

Reference models span two kinds: the TinyStories family (small models trained
on synthetic children's stories) and general-purpose LMs at the small end
(TinyLlama, Pythia). PocketLM is trained on dialogue, so every model is measured
on both domains rather than only the one that flatters it.
"""

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

# (model id, family). Families become series on the chart, so the grouping is
# by what a model is for: story models, general-purpose LMs, and PocketLM.
REFERENCE = [
    ("roneneldan/TinyStories-1M",  "TinyStories"),
    ("roneneldan/TinyStories-3M",  "TinyStories"),
    ("roneneldan/TinyStories-8M",  "TinyStories"),
    ("roneneldan/TinyStories-33M", "TinyStories"),
    ("Maykeye/TinyLLama-v0",       "general LM"),
    ("nickypro/tinyllama-15M",     "general LM"),
    ("EleutherAI/pythia-70m",      "general LM"),
]

STORY = (
    "Once upon a time there was a little girl named Lily. She liked to play "
    "outside with her dog. One day the sun was bright and warm. Lily and her dog "
    "ran to the park. They saw a big tree with red apples. Lily wanted an apple "
    "but she was too small to reach. Her friend Tom came to help her. He was "
    "tall and he picked one for her. Lily said thank you and smiled. They sat "
    "under the tree and shared the apple. It was a very happy day."
)


def dialogue_turns(data_dir="data", limit=120):
    """Held-out conversations, kept as turns so each model can frame them."""
    p = Path(data_dir) / "dialogue" / "val.jsonl"
    return [json.loads(l)["turns"] for l in p.read_text().splitlines()[:limit]]


def flatten(turns):
    return " ".join(t["text"] for conv in turns for t in conv)


@torch.no_grad()
def bpc_pocketlm(ckpt, turns):
    """Bits per character for a PocketLM checkpoint.

    The framing matters more than it looks. PocketLM is trained on
    <bos><user>...<eos><assistant>...<eos>, and feeding it the same dialogue as
    one raw string is out of distribution: it expects a control token and gets
    a word, is confidently wrong, and scores *worse than uniform random* --
    19.7 bits/token against a 8-bit ceiling for a 256-token vocabulary. That
    measures the missing framing, not the model.

    So each model gets its native format -- PocketLM sees the chat structure,
    the raw-text models see raw text -- and **every token is charged**,
    control tokens included.

    Charging for them matters. Read as compression, bits per character is what
    it costs to transmit the text, and PocketLM's format encodes the turn
    boundaries: excusing <user>/<assistant> would hand it that structure for
    free while TinyStories has to infer boundaries from words alone. Its
    control tokens are cheap because they are predictable, which is the correct
    outcome, not a free pass.
    """
    from chat import load
    model, tok = load(ckpt, torch.device("cpu"))
    ids = []
    for conv in turns:
        cid, _ = tok.encode_turns(conv)
        ids.extend(cid)

    ctx = model.cfg.context_length
    total_nll = 0.0
    for i in range(0, len(ids) - 1, ctx):
        chunk = ids[i:i + ctx + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]])
        y = torch.tensor([chunk[1:]])
        logits = model(x)
        total_nll += float(torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"))

    text = " ".join(t["text"] for conv in turns for t in conv)
    chars = len(text.lower() if tok.lowercase else text)
    return total_nll / math.log(2) / chars


@torch.no_grad()
def bpc_hf(model_id, text):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).eval()
    n_params = sum(p.numel() for p in model.parameters())
    ids = tok(text, return_tensors="pt").input_ids[0]
    ctx = min(getattr(model.config, "max_position_embeddings", 512), 512)
    total_nll = 0.0
    for i in range(0, len(ids) - 1, ctx):
        chunk = ids[i:i + ctx + 1]
        if len(chunk) < 2:
            break
        out = model(chunk[:-1].unsqueeze(0))
        nll = torch.nn.functional.cross_entropy(
            out.logits.reshape(-1, out.logits.size(-1)), chunk[1:], reduction="sum")
        total_nll += float(nll)
    return total_nll / math.log(2) / len(text), n_params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="eval/results/external.json")
    args = ap.parse_args()

    turns = dialogue_turns()
    story_turns = [[{"role": "user", "text": STORY}]]
    # PocketLM gets turns; raw-text models get the same content flattened.
    native = {"dialogue": turns, "stories": story_turns}
    raw = {"dialogue": flatten(turns), "stories": STORY}
    print(f"dialogue sample: {len(raw['dialogue']):,} chars   "
          f"stories sample: {len(raw['stories']):,} chars\n")

    from config import FAMILY

    # Every branch's models, so the chart can show what real data and a config
    # search do to the same budgets.
    LOCAL = [
        ("PocketLM", "10k",  "checkpoints/10k.pt",   FAMILY["10k"].n_params()),
        ("PocketLM", "50k",  "checkpoints/50k.pt",   FAMILY["50k"].n_params()),
        ("PocketLM", "100k", "checkpoints/100k.pt",  FAMILY["100k"].n_params()),
        ("PocketLM", "500k", "checkpoints/500k.pt",  FAMILY["500k"].n_params()),
        ("PocketLM", "1m",   "checkpoints/1m.pt",    FAMILY["1m"].n_params()),
        ("dev branch", "10k-real",  "dev/checkpoints/hybrid/10k-transformer.pt", 9_584),
        ("dev branch", "50k-real",  "dev/checkpoints/50k-dev.pt",  48_416),
        ("dev branch", "500k-real", "dev/checkpoints/500k-dev.pt", 491_040),
        ("testing branch", "1k-best",  "testing/checkpoints/1k-best.pt",  984),
        ("testing branch", "10k-best", "testing/checkpoints/10k-best.pt", 9_808),
    ]
    results = []
    for family, name, ckpt, n in LOCAL:
        if not Path(ckpt).exists():
            print(f"  {name:<28} skipped: no checkpoint")
            continue
        row = {"model": f"PocketLM-{name}", "family": family, "params": n}
        for domain, t in native.items():
            row[domain] = bpc_pocketlm(ckpt, t)
        results.append(row)
        print(f"  {row['model']:<28}{row['params']:>11,}  "
              f"dialogue {row['dialogue']:.2f}  stories {row['stories']:.2f}")

    for mid, family in REFERENCE:
        row = {"model": mid.split("/")[-1], "family": family}
        try:
            for domain, text in raw.items():
                bpc, n = bpc_hf(mid, text)
                row[domain], row["params"] = bpc, n
        except Exception as exc:
            print(f"  {mid:<28} skipped: {str(exc)[:60]}")
            continue
        results.append(row)
        print(f"  {row['model']:<28}{row['params']:>11,}  "
              f"dialogue {row['dialogue']:.2f}  stories {row['stories']:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}   (bits per character, lower is better)")


if __name__ == "__main__":
    main()
