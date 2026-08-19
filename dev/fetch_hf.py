#!/usr/bin/env python3
"""Fetch real dialogue corpora from Hugging Face and convert them to PocketLM JSONL.

The bootstrap corpus in scripts/build_corpus.py is slot-filled: 95,054 user
turns but only 280 distinct ones. A model trained on it learns exact-string
matching, which is why the released models answer "hello" correctly and "Hello"
with nonsense. This script replaces that with human-written conversation.

Sources, and why each is here:

  better_daily_dialog     DailyDialog as parquet. Everyday human chit-chat with
                          naturally short turns -- the closest match to what
                          PocketLM is trying to be.
  empathetic_dialogues_v2 Human conversations about feelings. Covers the
                          "acknowledge, then open a door" behaviour directly.
  persona-chat            Crowdworker conversations with a consistent persona.
  oasst1                  Human assistant-style dialogue, Apache-2.0.

LICENCES -- read before redistributing anything:

  oasst1                  Apache-2.0        redistributable
  better_daily_dialog     CC BY-NC-SA 4.0   NON-COMMERCIAL, do not redistribute
  empathetic_dialogues    CC BY-NC 4.0      NON-COMMERCIAL, do not redistribute
  persona-chat            unspecified       treat as non-redistributable

Everything this script writes lands under dev/data/, never in the shared data/
directory, so the dev branch cannot pollute main's working tree. dev/data/ is
gitignored, so fetched corpora stay local and nothing here is republished. Use
--permissive-only to restrict the build to Apache/MIT sources.

Deliberately NOT included: Anthropic/hh-rlhf. It is MIT and human-written, but
it is red-teaming data whose content is adversarial and frequently abusive by
construction -- the wrong thing to point a small friendly chatbot at.

    python dev/fetch_hf.py --target-mb 14      # -> dev/data/real/
"""

import argparse
import ast
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

Turns = List[Dict[str, str]]

# PocketLM replies must stay short: it is trained to end turns, and long targets
# teach it to ramble at a size where rambling reads as broken.
MAX_ASSISTANT_WORDS = 24
MAX_USER_WORDS = 40
MAX_TURNS = 12
MIN_TURNS = 2

_BAD = re.compile(r"(https?://|www\.|[*_#`~<>{}\[\]|]|\\n|@\w+|\b(?:as an ai|language model)\b)", re.I)
_WS = re.compile(r"\s+")


def clean(text: str) -> Optional[str]:
    """Normalise one utterance, or reject it."""
    if not isinstance(text, str):
        return None
    text = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("—", "-")
    text = _WS.sub(" ", text).strip()
    if not text or not text.isascii() or _BAD.search(text):
        return None
    # DailyDialog spaces its punctuation: "how about it ? " -> "how about it?"
    text = re.sub(r"\s+([,.!?;:'])", r"\1", text)
    text = re.sub(r"([,.!?;:])(?=[A-Za-z])", r"\1 ", text)
    return text.strip() or None


def valid(turns: Turns) -> bool:
    if not MIN_TURNS <= len(turns) <= MAX_TURNS:
        return False
    if turns[0]["role"] != "user" or turns[-1]["role"] != "assistant":
        return False
    for i, t in enumerate(turns):
        if t["role"] != ("user" if i % 2 == 0 else "assistant"):
            return False
        n = len(t["text"].split())
        if n < 1 or n > (MAX_ASSISTANT_WORDS if t["role"] == "assistant" else MAX_USER_WORDS):
            return False
    return True


def windows(utts: List[str], rng: random.Random) -> Iterator[Turns]:
    """Cut a long transcript into short alternating conversations.

    Every window starts on a user turn and ends on an assistant turn, so each
    one is a complete training example rather than a fragment.
    """
    for start in range(0, len(utts) - 1, 2):
        span = rng.choice([2, 4, 4, 6, 6, 8])
        chunk = utts[start:start + span]
        if len(chunk) % 2:
            chunk = chunk[:-1]
        if len(chunk) < 2:
            continue
        turns = [{"role": "user" if i % 2 == 0 else "assistant", "text": t}
                 for i, t in enumerate(chunk)]
        if valid(turns):
            yield turns


# ------------------------------------------------------------------ adapters

def from_daily_dialog(rng, limit) -> Iterator[Turns]:
    from datasets import load_dataset
    ds = load_dataset("pixelsandpointers/better_daily_dialog", split="train", streaming=True)
    current, dialog_id = [], None
    for i, row in enumerate(ds):
        if limit and i > limit:
            break
        if row["dialog_id"] != dialog_id:
            if current:
                yield from windows(current, rng)
            current, dialog_id = [], row["dialog_id"]
        u = clean(row["utterance"])
        if u:
            current.append(u)
        else:
            if current:
                yield from windows(current, rng)
            current = []
    if current:
        yield from windows(current, rng)


def from_empathetic(rng, limit) -> Iterator[Turns]:
    from datasets import load_dataset
    ds = load_dataset("Adapting/empathetic_dialogues_v2", split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit and i > limit:
            break
        try:
            history = ast.literal_eval(row["chat_history"])
        except (ValueError, SyntaxError):
            continue
        utts = [clean(h) for h in history] + [clean(row["sys_response"])]
        if any(u is None for u in utts):
            continue
        if len(utts) % 2:                       # must end on the assistant
            utts = utts[1:]
        turns = [{"role": "user" if j % 2 == 0 else "assistant", "text": u}
                 for j, u in enumerate(utts)]
        if valid(turns):
            yield turns


def from_persona_chat(rng, limit) -> Iterator[Turns]:
    from datasets import load_dataset
    ds = load_dataset("AlekseyKorshuk/persona-chat", split="train", streaming=True)
    for i, row in enumerate(ds):
        if limit and i > limit:
            break
        for utt in row["utterances"][-3:]:      # later entries have longer history
            hist = [clean(h) for h in utt["history"] if h != "__ SILENCE __"]
            reply = clean(utt["candidates"][-1])   # last candidate is the true reply
            if reply is None or any(h is None for h in hist) or not hist:
                continue
            utts = hist + [reply]
            if len(utts) % 2:
                utts = utts[1:]
            turns = [{"role": "user" if j % 2 == 0 else "assistant", "text": u}
                     for j, u in enumerate(utts)]
            if valid(turns):
                yield turns


def from_oasst1(rng, limit) -> Iterator[Turns]:
    """Reconstruct conversation threads from oasst1's message tree."""
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    msgs, children = {}, defaultdict(list)
    for row in ds:
        if row["lang"] != "en":
            continue
        msgs[row["message_id"]] = row
        children[row["parent_id"]].append(row["message_id"])

    def walk(mid, path):
        row = msgs.get(mid)
        if row is None:
            return
        text = clean(row["text"])
        if text is None:
            return
        role = "user" if row["role"] == "prompter" else "assistant"
        path = path + [{"role": role, "text": text}]
        if valid(path):
            yield list(path)
        if len(path) < MAX_TURNS:
            for child in children.get(mid, []):
                yield from walk(child, path)

    count = 0
    for root in children.get(None, []):
        for conv in walk(root, []):
            yield conv
            count += 1
            if limit and count > limit:
                return


SOURCES = {
    "daily_dialog":  (from_daily_dialog, "CC BY-NC-SA 4.0", False),
    "empathetic":    (from_empathetic,   "CC BY-NC 4.0",    False),
    "persona_chat":  (from_persona_chat, "unspecified",     False),
    "oasst1":        (from_oasst1,       "Apache-2.0",      True),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dev/data/real")
    ap.add_argument("--target-mb", type=float, default=12.0)
    ap.add_argument("--sources", default=",".join(SOURCES))
    ap.add_argument("--permissive-only", action="store_true",
                    help="Apache/MIT sources only, i.e. safe to redistribute")
    ap.add_argument("--limit-per-source", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    wanted = [s for s in args.sources.split(",") if s in SOURCES]
    if args.permissive_only:
        wanted = [s for s in wanted if SOURCES[s][2]]

    rows, seen, bytes_out = [], set(), 0
    per_source = {}
    # Per-source budgets, not one shared pot. Drawing sequentially let
    # DailyDialog take 33,376 of 33,418 conversations and starved every other
    # source, which is a monoculture dressed up as a mixed corpus.
    budget = args.target_mb * 1024 * 1024
    share = budget / max(len(wanted), 1)

    for idx, name in enumerate(wanted):
        fn, licence, permissive = SOURCES[name]
        kept = 0
        # Anything earlier sources left unspent is redistributed forward.
        allowance = share * (idx + 1) + max(0.0, share * idx - bytes_out)
        print(f"\n=== {name}  ({licence}{'' if permissive else ', local use only'})")
        try:
            for turns in fn(rng, args.limit_per_source):
                key = turns[0]["text"][:60] + "|" + turns[-1]["text"][:60]
                if key in seen:
                    continue
                seen.add(key)
                row = {"id": len(rows), "category": f"real:{name}",
                       "memory": None, "turns": turns}
                line = json.dumps(row, ensure_ascii=False)
                rows.append(row)
                bytes_out += len(line) + 1
                kept += 1
                if kept % 20000 == 0:
                    print(f"  {kept:,} conversations, {bytes_out/1024/1024:.1f} MB")
                if bytes_out >= min(allowance, budget):
                    break
        except Exception as exc:
            print(f"  failed: {exc}")
        per_source[name] = kept
        print(f"  kept {kept:,} conversations  (running total {bytes_out/1024/1024:.1f} MB)")
        if bytes_out >= budget:
            print(f"  reached {args.target_mb} MB target")
            break

    if not rows:
        raise SystemExit("nothing fetched")

    rng.shuffle(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cut = max(1, int(len(rows) * args.val_frac))
    for split, part in (("val", rows[:cut]), ("train", rows[cut:])):
        with (out / f"{split}.jsonl").open("w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    turns_total = sum(len(r["turns"]) for r in rows)
    users = {t["text"] for r in rows for t in r["turns"] if t["role"] == "user"}
    print(f"\n{'source':<16}{'conversations':>15}")
    for k, v in per_source.items():
        print(f"  {k:<14}{v:>15,}")
    print(f"\ntotal: {len(rows):,} conversations, {turns_total:,} turns, "
          f"{bytes_out/1024/1024:.1f} MB")
    print(f"distinct user turns: {len(users):,}  "
          f"(the synthetic corpus has 280)")
    print(f"wrote {out}/train.jsonl and {out}/val.jsonl")


if __name__ == "__main__":
    main()
