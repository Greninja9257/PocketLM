#!/usr/bin/env python3
"""Corrupt user turns so the model survives real typing.

The released models answer "hello" correctly and "Hello" with nonsense, because
the corpus contains 95,054 user turns of which only 280 are distinct, all
lowercase, all correctly spelled, all punctuated the same way. The model
therefore learned exact-string matching rather than intent.

This is unusually cheap to fix here, because of loss masking: user tokens are
never scored (see dataset.py), so corrupting them costs nothing in reply
quality. The gradient only ever flows through assistant text, which stays
pristine. Noise on the input side is free robustness.

Assistant turns are never touched -- they are the training target, and teaching
the model to produce typos would be actively harmful.

    python dev/augment.py --in dev/data/real --out dev/data/real_noisy --rate 0.6
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List

# Keyboard neighbours, for typos that look like real slips rather than random
# character noise. "helllo" and "hrllo" are the mistakes people actually make.
_NEIGHBOURS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx",
}

# Chat shorthand, in the direction real users type it.
_SHORTHAND = [
    (r"\byou\b", "u"), (r"\byour\b", "ur"), (r"\byou're\b", "ur"),
    (r"\bare\b", "r"), (r"\bplease\b", "pls"), (r"\bbecause\b", "cuz"),
    (r"\bokay\b", "ok"), (r"\bwhat\b", "wat"), (r"\bwant to\b", "wanna"),
    (r"\bgoing to\b", "gonna"), (r"\bkind of\b", "kinda"), (r"\bto\b", "2"),
    (r"\bfor\b", "4"), (r"\bthanks\b", "thx"), (r"\bwith\b", "w/"),
    (r"\bI don't know\b", "idk"), (r"\bsee you\b", "cya"),
]


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 3:
        return word
    i = rng.randrange(len(word))
    kind = rng.random()
    if kind < 0.3 and i < len(word) - 1:                 # transpose
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind < 0.55:                                      # drop a character
        return word[:i] + word[i + 1:]
    if kind < 0.8:                                       # double a character
        return word[:i] + word[i] + word[i:]
    near = _NEIGHBOURS.get(word[i].lower())              # hit the wrong key
    return word[:i] + rng.choice(near) + word[i + 1:] if near else word


def _casing(text: str, rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.45:
        return text[:1].upper() + text[1:]               # Sentence case
    if roll < 0.60:
        return text.upper()                              # SHOUTING
    if roll < 0.70:
        return "".join(c.upper() if rng.random() < 0.3 else c for c in text)
    return text.lower()


def _punct(text: str, rng: random.Random) -> str:
    roll = rng.random()
    if roll < 0.4:
        return text.rstrip("?!.,")                       # drop terminal punctuation
    if roll < 0.55:
        return text.rstrip("?!.,") + rng.choice(["!!", "??", "...", "!?"])
    if roll < 0.7:
        return text.replace("'", "")                     # whats, im, dont
    if roll < 0.8:
        return text + " "
    return re.sub(r" ", "  ", text, count=1)


def noisify(text: str, rng: random.Random, rate: float = 0.6) -> str:
    """Apply 1-3 realistic corruptions to a user utterance."""
    if rng.random() > rate:
        return text                                       # leave it clean
    ops = rng.sample(["case", "punct", "typo", "shorthand"], k=rng.randint(1, 3))
    for op in ops:
        if op == "case":
            text = _casing(text, rng)
        elif op == "punct":
            text = _punct(text, rng)
        elif op == "shorthand":
            pat, rep = rng.choice(_SHORTHAND)
            text = re.sub(pat, rep, text, count=1, flags=re.I)
        elif op == "typo":
            words = text.split(" ")
            idxs = [i for i, w in enumerate(words) if len(w) > 3]
            if idxs:
                for i in rng.sample(idxs, k=min(len(idxs), rng.randint(1, 2))):
                    words[i] = _typo(words[i], rng)
                text = " ".join(words)
    return text


def augment_row(row: Dict, rng: random.Random, rate: float) -> Dict:
    out = dict(row)
    out["turns"] = [
        # Assistant turns are the training target and stay untouched.
        t if t["role"] == "assistant" else {**t, "text": noisify(t["text"], rng, rate)}
        for t in row["turns"]
    ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", default="dev/data/real")
    ap.add_argument("--out", dest="dst", default="dev/data/real_noisy")
    ap.add_argument("--rate", type=float, default=0.6,
                    help="fraction of user turns corrupted (default 0.6)")
    ap.add_argument("--keep-clean", type=float, default=0.35,
                    help="fraction of conversations copied through untouched, so "
                         "the canonical forms are still learned well")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        path = src / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        out: List[Dict] = []
        for r in rows:
            out.append(r if rng.random() < args.keep_clean else augment_row(r, rng, args.rate))
        with (dst / f"{split}.jsonl").open("w") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        users = [t["text"] for r in out for t in r["turns"] if t["role"] == "user"]
        print(f"{split}: {len(out):,} conversations, {len(set(users)):,} distinct user turns")

    print("\nexamples:")
    demo = random.Random(0)
    for s in ["hello", "what's your name?", "I'm having a rough day",
              "do you want to play a game?", "thanks, that was helpful"]:
        variants = [noisify(s, demo, rate=1.0) for _ in range(3)]
        print(f"  {s!r}\n     -> {variants}")


if __name__ == "__main__":
    main()
