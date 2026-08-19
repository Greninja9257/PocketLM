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


def dialogue_text(data_dir="data", limit=120):
    rows, out = [], []
    p = Path(data_dir) / "dialogue" / "val.jsonl"
    for line in p.read_text().splitlines()[:limit]:
        r = json.loads(line)
        out.append(" ".join(t["text"] for t in r["turns"]))
    return " ".join(out)


@torch.no_grad()
def bpc_pocketlm(ckpt, text):
    from chat import load
    model, tok = load(ckpt, torch.device("cpu"))
    ids = [tok.bos_id] + tok.encode(text)
    ctx = model.cfg.context_length
    total_nll, n_pred = 0.0, 0
    for i in range(0, len(ids) - 1, ctx):
        chunk = ids[i:i + ctx + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]])
        y = torch.tensor([chunk[1:]])
        logits = model(x)
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_nll += float(nll)
        n_pred += y.numel()
    # Characters the tokenizer can actually represent (lowercase models fold case).
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

    texts = {"dialogue": dialogue_text(), "stories": STORY}
    print(f"dialogue sample: {len(texts['dialogue']):,} chars   "
          f"stories sample: {len(texts['stories']):,} chars\n")

    results = []
    for name in ["10k", "50k", "100k", "500k", "1m"]:
        row = {"model": f"PocketLM-{name}", "family": "PocketLM"}
        from config import FAMILY
        row["params"] = FAMILY[name].n_params()
        for domain, text in texts.items():
            row[domain] = bpc_pocketlm(f"checkpoints/{name}.pt", text)
        results.append(row)
        print(f"  {row['model']:<28}{row['params']:>11,}  "
              f"dialogue {row['dialogue']:.2f}  stories {row['stories']:.2f}")

    for mid, family in REFERENCE:
        row = {"model": mid.split("/")[-1], "family": family}
        try:
            for domain, text in texts.items():
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
