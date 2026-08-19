#!/usr/bin/env python3
"""Train one BPE tokenizer per vocabulary size the family needs.

    python scripts/train_tokenizer.py            # all sizes: 64 .. 1024
    python scripts/train_tokenizer.py --vocab-size 512

Sizes cannot share a tokenizer: the embedding table costs vocab x d_model, so
the 1K model can afford 64 tokens and the 1M model can afford 1,024. Each
lands at checkpoints/tokenizer-<vocab>.json.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (ASSISTANT_NAME, LOWERCASE_BELOW_VOCAB, VOCAB_SIZES,
                    tokenizer_path)
from tokenizer import BPETokenizer

PROBE = "hey! what's your favorite color? I'm PocketLM."


def iter_text(data_dir: Path):
    for path in sorted(data_dir.rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if "text" in row:
                yield row["text"]
            for turn in row.get("turns", []):
                yield turn["text"]
            if row.get("memory"):
                yield row["memory"]


def train_one(texts, chars, vocab_size: int, min_symbol_count: int, quiet: bool,
              out_dir: str = "checkpoints") -> None:
    lowercase = vocab_size < LOWERCASE_BELOW_VOCAB
    print(f"\n=== vocab {vocab_size}{'  (lowercase)' if lowercase else ''}")
    tok = BPETokenizer.train(texts, vocab_size, min_symbol_count=min_symbol_count,
                             lowercase=lowercase, verbose=not quiet)
    out = str(Path(out_dir) / Path(tokenizer_path(vocab_size)).name)

    # Validate BEFORE writing. Saving first left rejected tokenizers on disk,
    # where a later run would find the file, report "already present", and
    # happily train a model that cannot spell its own name.
    if ASSISTANT_NAME.lower() not in tok.decode(tok.encode(ASSISTANT_NAME)).lower():
        raise SystemExit(f"vocab {vocab_size} cannot represent {ASSISTANT_NAME!r} — "
                         f"its alphabet is capped too hard; not saving")
    tok.save(out)

    n_tokens = sum(len(tok.encode(t)) for t in texts)
    print(f"  {len(tok)} tokens = {len(tok.specials)} special + {len(tok.alphabet)} "
          f"alphabet + {len(tok.merges)} merges")
    print(f"  compression: {chars / max(n_tokens, 1):.2f} chars/token")
    if tok.merges:
        longest = sorted(tok.itos[-len(tok.merges):], key=len, reverse=True)[:8]
        print("  longest merges: " + ", ".join(repr(t) for t in longest))

    ids = tok.encode(PROBE)
    expected = PROBE.lower() if lowercase else PROBE
    ok = tok.decode(ids) == expected
    print(f"  probe -> {len(ids)} tokens, roundtrip {'ok' if ok else 'LOSSY'}")
    if not ok:
        print(f"    expected {expected!r}\n    got      {tok.decode(ids)!r}")
    print(f"  saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out-dir", default="checkpoints",
                    help="where to write tokenizers (branches use their own dir)")
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="train just this one size (default: every size in the family)")
    ap.add_argument("--min-symbol-count", type=int, default=20)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    texts = list(iter_text(Path(args.data)))
    chars = sum(len(t) for t in texts)
    print(f"corpus: {len(texts):,} strings / {chars:,} chars")

    for vocab_size in ([args.vocab_size] if args.vocab_size else VOCAB_SIZES):
        train_one(texts, chars, vocab_size, args.min_symbol_count, args.quiet,
                  args.out_dir)


if __name__ == "__main__":
    main()
