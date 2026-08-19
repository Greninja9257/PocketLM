#!/usr/bin/env python3
"""Run one fixed prompt set through every trained checkpoint.

Produces the comparison table in the README. Covers all three branches, because
the interesting comparisons are cross-branch: the same budget trained on
templates vs real dialogue, or a hand-picked config vs a searched one.

    python scripts/make_examples.py --out /tmp/examples.md
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from chat import load
from manager import ConversationManager
from memory import Memory

PROMPTS = ["hey", "what's your name?", "I'm having a rough day",
           "tell me a joke", "what's the capital of Chad?"]

# (checkpoint, label, branch, note). Order is the reading order of the table.
MODELS = [
    ("checkpoints/1k.pt",    "1k",    "main", "released"),
    ("checkpoints/5k.pt",    "5k",    "main", "released"),
    ("checkpoints/10k.pt",   "10k",   "main", "released"),
    ("checkpoints/50k.pt",   "50k",   "main", "released"),
    ("checkpoints/100k.pt",  "100k",  "main", "released"),
    ("checkpoints/500k.pt",  "500k",  "main", "released"),
    ("checkpoints/1m.pt",    "1m",    "main", "released"),
    ("testing/checkpoints/1k-best.pt",  "1k-best",  "testing", "config found by sweep"),
    ("testing/checkpoints/10k-best.pt", "10k-best", "testing", "config found by sweep"),
    ("testing/checkpoints/kd-student-v40.pt", "1k-kd", "testing", "distilled (failed)"),
    ("dev/checkpoints/hybrid/10k-transformer.pt", "10k-real", "dev", "real data"),
    ("dev/checkpoints/hybrid/10k-hybrid.pt", "10k-hybrid", "dev", "GRU+attention, real data"),
    ("dev/checkpoints/50k-dev.pt",  "50k-real",  "dev", "real data + noise"),
    ("dev/checkpoints/500k-dev.pt", "500k-real", "dev", "real data + noise"),
]


def cell(text: str) -> str:
    """Escape for a markdown table. Never truncate — a clipped reply hides
    exactly the rambling that distinguishes a small model from a good one."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/examples.md")
    ap.add_argument("--seed", type=int, default=4)
    args = ap.parse_args()

    rows = []
    for path, label, branch, note in MODELS:
        if not Path(path).exists():
            print(f"  skip {label}: no checkpoint at {path}")
            continue
        model, tok = load(path, torch.device("cpu"))
        torch.manual_seed(0)
        mgr = ConversationManager(model, tok, memory=Memory(), seed=args.seed)
        replies = [mgr.reply(p) for p in PROMPTS]
        rows.append((label, branch, note, model.n_params(), replies))
        print(f"  {label:<12} {branch:<8} {model.n_params():>9,}p  ok", flush=True)

    head = ["model", "branch", "params"] + PROMPTS
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for label, branch, note, n, replies in rows:
        lines.append("| `" + label + "` | " + branch + " | " + f"{n:,}" + " | "
                     + " | ".join(cell(r) for r in replies) + " |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
