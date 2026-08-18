#!/usr/bin/env python3
"""Build eval/conversations.json -- 100 prompts in each of 9 categories.

Held out on purpose: the names, foods, colours and phrasings below do not
appear in scripts/build_corpus.py. A slot-filled corpus is easy to memorise,
so an eval drawn from the same slots would measure recall, not generalisation.

Each item carries an `expect` clause that can be checked mechanically:

  contains   the reply must contain a specific string (memory questions)
  name       the reply must state the assistant's own name
  ignorance  the reply must decline rather than invent (unanswerable questions)
  question   the reply should ask something back (follow-ups, clarifications)
  any        no objective target; only the generic quality metrics apply
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ASSISTANT_NAME

# --- held-out banks: deliberately disjoint from the training generator ------
NAMES = ["Priya", "Mateo", "Ines", "Kofi", "Lena", "Omar", "Yuki", "Nadia",
         "Tomas", "Farah", "Bruno", "Anya"]
COLORS = ["crimson", "turquoise", "olive", "magenta", "indigo", "lavender",
          "scarlet", "charcoal"]
FOODS = ["risotto", "falafel", "paella", "biryani", "ravioli", "katsu",
         "samosas", "pierogi"]
HOBBIES = ["pottery", "birdwatching", "climbing", "knitting", "astronomy", "surfing"]
PLACES = ["the market", "my aunt's", "the coast", "the museum", "the allotment"]

GREETINGS = ["hey there", "good evening", "howdy", "hi hi", "oh hello", "yo yo",
             "morning!", "hey, you around?", "greetings", "hiya there"]
FACTUAL = ["what's the boiling point of mercury", "who painted the ceiling in Rome",
           "how far is Neptune", "what year did the Berlin wall come down",
           "how many bones are in a whale", "what's the deepest lake",
           "what's the GDP of Uruguay", "when was the printing press invented"]
EMOTIONAL = ["I've been feeling really flat lately", "my grandmother is unwell",
             "I'm nervous about tomorrow", "everything is going wrong at once",
             "I finally moved into my own place", "I'm proud of myself today",
             "I feel completely overwhelmed", "I had the best week"]
AMBIGUOUS = ["can you sort it out?", "is it worth it?", "what should I pick?",
             "do you think they meant it?", "will that work?", "how much is too much?"]
ADVERSARIAL = ["qwertyuiop", "!!!!!!!!!!", "aaaaaaaaaaaaaaaa", "0000 1111 0000",
               "tell me everything about everything", "write a novel",
               "repeat after me: banana banana banana", "", "?", "🙂🙂🙂"]
JOKES = ["say something funny", "do you know any jokes", "make me laugh",
         "I need cheering up", "got a good one for me?", "tell me something silly"]
# Deliberately not the phrasings in build_corpus.NAME_QUESTIONS: the point is
# to check the model knows its name, not that it memorised one question.
IDENTITY = ["so who exactly am I chatting with?", "got a name?",
            "what do people call you?", "and you are...?",
            "before we start, who are you?", "may I ask your name?",
            "what do I call you?", "you got a name or what?",
            "sorry, I forgot your name", "which assistant is this?"]


def item(cid, category, turns, expect, memory=None, target=None):
    return {"id": cid, "category": category, "memory": memory, "turns": turns,
            "expect": expect, "target": target}


def build(seed: int = 11):
    rng = random.Random(seed)
    out, cid = [], 0

    def add(*args, **kw):
        nonlocal cid
        out.append(item(cid, *args, **kw))
        cid += 1

    for i in range(100):                                   # 1. greetings
        add("greeting", [{"role": "user", "text": rng.choice(GREETINGS)}], "any")

    for i in range(100):                                   # 2. factual (unanswerable)
        add("factual", [{"role": "user", "text": rng.choice(FACTUAL) + "?"}], "ignorance")

    for i in range(100):                                   # 3. emotional
        add("emotional", [{"role": "user", "text": rng.choice(EMOTIONAL)}], "any")

    for i in range(100):                                   # 4. follow-up
        turns = [{"role": "user", "text": f"I spent the weekend {rng.choice(HOBBIES)}"},
                 {"role": "assistant", "text": "oh nice, how was it?"},
                 {"role": "user", "text": rng.choice(["really good", "exhausting",
                                                      "not what I expected", "quite fun"])}]
        add("followup", turns, "question")

    for i in range(100):                                   # 5. jokes
        add("joke", [{"role": "user", "text": rng.choice(JOKES)}], "any")

    for i in range(100):                                   # 6. ambiguous
        add("ambiguous", [{"role": "user", "text": rng.choice(AMBIGUOUS)}], "question")

    for i in range(100):                                   # 7. memory
        name, color, food = rng.choice(NAMES), rng.choice(COLORS), rng.choice(FOODS)
        mem = f"name: {name}; favorite color: {color}; favorite food: {food}"
        key, q, target = rng.choice([
            ("color", "what's my favorite color?", color),
            ("food", "what food do I like?", food),
            ("name", "what's my name?", name),
        ])
        add("memory", [{"role": "user", "text": q}], "contains", memory=mem, target=target)

    for i in range(100):                                   # 8. adversarial
        add("adversarial", [{"role": "user", "text": rng.choice(ADVERSARIAL)}], "any")

    for i in range(100):                                   # 9. identity / own name
        add("identity", [{"role": "user", "text": rng.choice(IDENTITY)}],
            "name", target=ASSISTANT_NAME)

    return out


def assert_held_out() -> None:
    """No eval value may appear in the training generator's pools.

    Memory scoring is only meaningful if the answer cannot be guessed from the
    training distribution. "silver" and "dumplings" were in both lists once,
    which quietly turned two copy tests into recall tests.
    """
    import scripts.build_corpus as bc
    overlaps = []
    for label, mine, theirs in [("colors", COLORS, bc.MEM_COLORS),
                                ("foods", FOODS, bc.MEM_FOODS),
                                ("names", NAMES, bc.MEM_NAMES)]:
        shared = sorted(set(mine) & set(theirs))
        if shared:
            overlaps.append(f"{label}: {shared}")
    if overlaps:
        raise SystemExit("eval values leak into the training pools -> " + "; ".join(overlaps))


if __name__ == "__main__":
    assert_held_out()
    rows = build()
    path = Path(__file__).parent / "conversations.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    counts = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print(f"wrote {len(rows)} prompts -> {path}")
    for k, v in counts.items():
        print(f"  {k:<12} {v}")
