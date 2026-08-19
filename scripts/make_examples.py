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
from generate import SamplingConfig
from manager import ConversationManager
from memory import Memory

PROMPTS = ["hey", "what's your name?", "I'm having a rough day",
           "tell me a joke", "what's the capital of Chad?"]

# These models are stochastic and a single sample proves nothing: the 10k model
# answers "what's the capital of Chad?" with a clean refusal in 10 of 12 draws,
# but one unlucky draw once made it look far worse than it is.
#
# Two defensible ways to report this, and the mode chooses between them:
#
#   --mode greedy    deterministic, shows what the model believes, but exposes
#                    mode collapse that sampling hides
#   --mode best      draw N samples and keep the most fluent, scored
#                    mechanically (below). This is a best-case showcase and the
#                    table says so — it is not typical output.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SAMPLES = 12

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


def build_lexicon(data_dir: str = "data") -> set:
    """Words the training data actually uses, for scoring fluency."""
    import json
    import re
    words = set()
    for path in Path(data_dir).rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            texts = [t["text"] for t in row.get("turns", [])]
            if "text" in row:
                texts.append(row["text"])
            for t in texts:
                words.update(re.findall(r"[a-z']+", t.lower()))
    return words


def fluency(reply: str, lexicon: set) -> float:
    """Score a candidate reply. Higher is better.

    Deliberately mechanical: the point of best-of-N is to show each model at
    its best, and a human picking favourites would be choosing the result. This
    rewards real words, complete sentences and brevity, and punishes the two
    failure modes these models actually have -- gibberish and looping.
    """
    import re
    words = re.findall(r"[a-z']+", reply.lower())
    if not words:
        return -1.0
    real = sum(w in lexicon for w in words) / len(words)
    distinct = len(set(words)) / len(words)              # penalise loops
    ends_clean = 1.0 if reply.rstrip()[-1:] in ".!?" else 0.0
    # Long replies from a tiny model are usually a ramble, not richness.
    brevity = 1.0 if len(words) <= 12 else max(0.0, 1.0 - (len(words) - 12) / 12)
    return 3.0 * real + 1.0 * distinct + 0.5 * ends_clean + 0.5 * brevity


def cell(text: str) -> str:
    """Escape for a markdown table. Never truncate — a clipped reply hides
    exactly the rambling that distinguishes a small model from a good one."""
    return text.replace("|", "\\|").replace("\n", " ").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/tmp/examples.md")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--mode", default="best", choices=["best", "greedy"])
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--samples", type=int, default=DEFAULT_SAMPLES,
                    help="draws per prompt in --mode best")
    ap.add_argument("--data", default="data", help="corpus used to build the lexicon")
    ap.add_argument("--dump", default=None,
                    help="write every draw to JSON for review instead of picking")
    ap.add_argument("--selections", default=None,
                    help="JSON of {model: [draw_index per prompt]} — renders exactly "
                         "those draws, so a hand-picked table stays reproducible")
    args = ap.parse_args()
    if args.mode == "greedy":
        args.temperature, args.samples = 0.0, 1

    lexicon = build_lexicon(args.data) if args.mode == "best" else set()
    if lexicon:
        print(f"lexicon: {len(lexicon):,} words from {args.data}\n")

    import json as _json
    selections = _json.loads(Path(args.selections).read_text()) if args.selections else None
    all_draws = {}

    rows = []
    for path, label, branch, note in MODELS:
        if not Path(path).exists():
            print(f"  skip {label}: no checkpoint at {path}")
            continue
        model, tok = load(path, torch.device("cpu"))
        cfg = SamplingConfig(temperature=args.temperature)
        replies = []
        for prompt in PROMPTS:
            draws = []
            for k in range(args.samples):
                torch.manual_seed(k)
                mgr = ConversationManager(model, tok, memory=Memory(),
                                          sampling=cfg, seed=args.seed + k)
                draws.append(mgr.reply(prompt))
            all_draws.setdefault(label, []).append(draws)
            if selections and label in selections:
                best = draws[selections[label][len(replies)]]
            elif args.mode == "best":
                best = max(draws, key=lambda d: fluency(d, lexicon))
            else:
                best = draws[0]
            replies.append(best)
        rows.append((label, branch, note, model.n_params(), replies))
        print(f"  {label:<12} {branch:<8} {model.n_params():>9,}p  ok", flush=True)

    head = ["model", "branch", "params"] + PROMPTS
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(["---"] * len(head)) + "|"]
    for label, branch, note, n, replies in rows:
        lines.append("| `" + label + "` | " + branch + " | " + f"{n:,}" + " | "
                     + " | ".join(cell(r) for r in replies) + " |")
    if args.dump:
        Path(args.dump).write_text(_json.dumps(
            {"prompts": PROMPTS, "draws": all_draws}, indent=1))
        print(f"\nwrote {args.dump} ({args.samples} draws per prompt per model)")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
