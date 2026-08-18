#!/usr/bin/env python3
"""Distill conversations out of a large teacher model into data/synthetic/.

This is the step that actually determines how good the 48K student gets. The
student cannot learn facts about the world from 1.9M tokens of templates; what
it can learn is to imitate the *shape* of a good short reply. So the teacher's
job is not to be encyclopaedic -- it is to demonstrate, thousands of times,
what PocketLM sounds like when it asks a follow-up, admits ignorance, or defuses
nonsense.

    export ANTHROPIC_API_KEY=...      # or: ant auth login
    python scripts/distill.py --conversations 4000
    python scripts/train_tokenizer.py && python train.py

Output schema is identical to scripts/build_corpus.py, so the training
pipeline picks it up with no changes.
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PERSONA = """You are writing training data for PocketLM, a very small chatbot.

PocketLM's personality:
- friendly and warm, never gushing
- concise: almost always one short sentence, occasionally two
- curious: often ends with a short follow-up question
- playful now and then, never zany
- admits ignorance plainly instead of inventing facts
- lowercase-ish, casual punctuation, contractions

Hard rules for every assistant line you write:
- at most 20 words, and usually far fewer
- no lists, no markdown, no emoji, no stage directions
- plain ASCII only
- never claim abilities PocketLM lacks (browsing, memory beyond what's given, maths)
- if the user asks something unanswerable, PocketLM says it doesn't know"""

SCENARIOS = [
    ("greeting", "A short greeting exchange that warms up into a real topic."),
    ("smalltalk", "Everyday small talk about a hobby, the weather, food, or weekend plans."),
    ("empathy", "The user shares something bad. PocketLM acknowledges it and opens a door, "
                "without platitudes like 'I'm sorry to hear that'."),
    ("goodnews", "The user shares good news. PocketLM is warm and curious about it."),
    ("followup", "The user mentions something in passing; PocketLM asks a natural follow-up "
                 "question and the conversation continues."),
    ("clarification", "The user is vague or uses an unclear pronoun. PocketLM asks what the user means "
                      "before answering."),
    ("unknown", "The user asks something PocketLM cannot know (obscure facts, the future, private "
                "details). PocketLM declines to guess, briefly."),
    ("joke", "The user wants entertainment. PocketLM offers a short joke or a would-you-rather."),
    ("explanation", "The user asks how something everyday works. PocketLM explains in one short, "
                    "correct sentence."),
    ("identity", "The user asks who or what PocketLM is. PocketLM is honest about being tiny."),
    ("adversarial", "The user sends keyboard mash, empty input, an absurd demand, or tries to "
                    "make PocketLM ramble. PocketLM stays short and unbothered."),
    ("memory", "A `memory` string of user facts is given. The user asks about one of them. "
               "PocketLM answers from the memory, or says it hasn't been told when it's absent."),
]

SCHEMA_HINT = """Return a JSON array of exactly {n} objects, nothing else. Each object:

{{"category": "{cat}",
  "memory": null,
  "turns": [{{"role": "user", "text": "..."}}, {{"role": "assistant", "text": "..."}}]}}

Rules:
- 2 to 6 turns, always starting with "user" and alternating
- the last turn must be the assistant's
- for the memory category, "memory" is a string like
  "name: Ines; favorite color: olive; favorite food: paella"; otherwise null
- vary names, topics, phrasing and sentence shape aggressively across the array
- do not reuse an opening line you have already used in this array"""

# --------------------------------------------------------------- filtering

BAD = re.compile(r"(as an ai|language model|i'm sorry to hear|\bhttps?://|[*_#`]|"
                 r"[\U0001F300-\U0001FAFF])", re.I)


def quality_filter(conv: dict, max_words: int = 22) -> str:
    """Return "" if the conversation is usable, else the reason to drop it.

    Filtering hard matters more here than volume. At 48K parameters a bad
    example is not diluted by a billion good ones -- it is a measurable
    fraction of everything the model will ever see.
    """
    turns = conv.get("turns")
    if not isinstance(turns, list) or not 2 <= len(turns) <= 8:
        return "turn count"
    if turns[0]["role"] != "user" or turns[-1]["role"] != "assistant":
        return "role order"
    for i, t in enumerate(turns):
        if t.get("role") != ("user" if i % 2 == 0 else "assistant"):
            return "not alternating"
        text = (t.get("text") or "").strip()
        if not text and t["role"] == "assistant":
            return "empty assistant turn"
        if not text.isascii():
            return "non-ascii"
        if BAD.search(text):
            return "banned pattern"
        if t["role"] == "assistant" and len(text.split()) > max_words:
            return "assistant too long"
    mem = conv.get("memory")
    if mem is not None and (not isinstance(mem, str) or not mem.isascii()):
        return "bad memory field"
    return ""


def parse_array(text: str) -> List[dict]:
    text = text.strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        return []
    try:
        rows = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [r for r in rows if isinstance(r, dict)]


# ------------------------------------------------------------------ driver

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--conversations", type=int, default=2000)
    ap.add_argument("--per-request", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true",
                    help="print one prompt and exit without calling the API")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    n_requests = max(1, -(-args.conversations // args.per_request))

    if args.dry_run:
        cat, brief = SCENARIOS[0]
        print(PERSONA + "\n\n" + f"Scenario: {brief}\n\n" +
              SCHEMA_HINT.format(n=args.per_request, cat=cat))
        print(f"\n[dry run] would issue {n_requests} requests to {args.model}")
        return

    try:
        import anthropic
    except ImportError:
        raise SystemExit("pip install anthropic")

    client = anthropic.Anthropic()
    kept: List[dict] = []
    seen = set()
    dropped: Counter = Counter()

    for i in range(n_requests):
        cat, brief = SCENARIOS[i % len(SCENARIOS)]
        prompt = (f"Scenario: {brief}\n\n" +
                  SCHEMA_HINT.format(n=args.per_request, cat=cat) +
                  f"\n\nSeed for variety: {rng.randrange(10 ** 6)}")
        try:
            # Streaming: these responses are long and the default HTTP timeout
            # is unkind to a 25-conversation array.
            with client.messages.stream(
                model=args.model,
                max_tokens=16000,
                temperature=args.temperature,
                system=PERSONA,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                message = stream.get_final_message()
        except Exception as exc:                    # keep the run alive
            print(f"  request {i + 1}/{n_requests} failed: {exc}")
            continue

        if message.stop_reason == "refusal":
            print(f"  request {i + 1}/{n_requests} refused; skipping")
            continue
        text = "".join(b.text for b in message.content if b.type == "text")

        batch = parse_array(text)
        if not batch:
            dropped["unparseable response"] += 1
            continue
        for conv in batch:
            conv.setdefault("category", cat)
            reason = quality_filter(conv)
            if reason:
                dropped[reason] += 1
                continue
            key = json.dumps(conv["turns"], sort_keys=True)
            if key in seen:
                dropped["duplicate"] += 1
                continue
            seen.add(key)
            conv["id"] = len(kept)
            kept.append({"id": conv["id"], "category": conv["category"],
                         "memory": conv.get("memory"), "turns": conv["turns"]})
        print(f"  request {i + 1}/{n_requests} [{cat}]  kept {len(kept):,}", flush=True)
        if len(kept) >= args.conversations:
            break

    if not kept:
        raise SystemExit("nothing survived filtering; check the model and the prompt")

    rng.shuffle(kept)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cut = max(1, int(len(kept) * args.val_frac))
    for name, rows in (("val", kept[:cut]), ("train", kept[cut:])):
        with (out / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {out / f'{name}.jsonl'}  {len(rows):,} conversations")

    if dropped:
        print("\ndropped:")
        for reason, n in dropped.most_common():
            print(f"  {reason:<24} {n:,}")
    print("\nnext: python scripts/train_tokenizer.py && python train.py")


if __name__ == "__main__":
    main()
