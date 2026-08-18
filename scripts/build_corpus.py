#!/usr/bin/env python3
"""Generate PocketLM's bootstrap corpus.

This is scaffolding, not the destination. The plan calls for a large teacher
model to write the conversations; scripts/distill.py does that when an API key
is available. What lives here is a seeded, dependency-free generator that
produces the same *shape* of data -- PocketLM's voice, the four phase splits, the
memory-grounded turns -- so the pipeline is trainable the moment it is cloned.

Everything is combinatorial slot-filling, so treat volume with suspicion: more
samples buy more coverage of the templates, not more information. Replace these
files with distilled data as soon as you have it; the schema is identical.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

Turns = List[Dict[str, str]]
Sample = Tuple[str, Optional[str], Turns]      # (category, memory, turns)

# ------------------------------------------------------------------- banks

NAMES = ["Alex", "Sam", "Jamie", "Riley", "Casey", "Jordan", "Taylor", "Morgan",
         "Quinn", "Avery", "Devon", "Harper", "Rowan", "Skyler", "Emerson"]
COLORS = ["blue", "green", "red", "purple", "orange", "yellow", "black", "teal"]
ANIMALS = ["cats", "dogs", "birds", "rabbits", "horses", "foxes", "turtles"]
FOODS = ["pizza", "pasta", "sushi", "tacos", "soup", "curry", "pancakes", "ramen"]
GAMES = ["Minecraft", "chess", "Tetris", "football", "cards", "Scrabble", "tag"]
HOBBIES = ["drawing", "running", "reading", "cooking", "guitar", "photography",
           "gardening", "coding", "swimming", "baking"]
PLACES = ["the park", "the beach", "the library", "town", "the woods", "school"]
WEATHER = ["sunny", "rainy", "cold", "windy", "warm", "grey", "foggy"]
DAYS = ["Monday", "Tuesday", "Friday", "Saturday", "Sunday"]
GOOD_NEWS = ["I got the job", "I passed my exam", "my team won", "I finished my project",
             "I got a puppy", "my code finally works", "I ran 5k"]
BAD_NEWS = ["I failed my test", "I lost my keys", "my bike broke", "I had a rough day",
            "I missed the bus", "my plants died", "work was awful"]
HARD_QS = ["what's the population of Peru", "who won in 1926", "what's the capital of Chad",
           "how many atoms are in a grain of sand", "what's my bank balance",
           "what did I do last Tuesday", "what's the weather tomorrow",
           "what's the square root of 8237"]
JOKE_SETUPS = [
    ("why did the computer go to the doctor", "it had a virus"),
    ("what do you call a sleeping bull", "a bulldozer"),
    ("why don't skeletons fight", "they don't have the guts"),
    ("what's orange and sounds like a parrot", "a carrot"),
    ("why did the scarecrow win an award", "he was outstanding in his field"),
    ("what do you call cheese that isn't yours", "nacho cheese"),
]
WOULD_YOU_RATHER = [
    ("fly", "be invisible"), ("read minds", "see the future"),
    ("never sleep", "never eat"), ("live in space", "live underwater"),
    ("always be early", "always be right"),
]
GREETINGS = ["hi", "hello", "hey", "yo", "hey there", "morning", "hiya", "sup"]
NOVA_GREETS = ["hey!", "hi there!", "hey, good to see you.", "hello!", "hey you.",
               "hi! what's up?", "hey — how's it going?"]
FOLLOWUPS = ["what about you?", "how about you?", "what's yours?", "you?",
             "how'd that go?", "what happened?", "tell me more?"]

# PocketLM admits ignorance instead of inventing. This bank is deliberately large:
# at 48K parameters, "I don't know" is a skill worth spending capacity on.
DONT_KNOW = [
    "no idea, honestly.", "I don't know that one.", "couldn't tell you.",
    "that's outside what I know.", "I'd be guessing, so — no.",
    "not something I know.", "nope, no clue.",
]


def _one(rng: random.Random, xs):
    return rng.choice(xs)


def u(text: str) -> Dict[str, str]:
    return {"role": "user", "text": text}


def a(text: str) -> Dict[str, str]:
    return {"role": "assistant", "text": text}


# -------------------------------------------------------------- generators
# Each returns (category, memory, turns). Memory is the external-store line
# that the conversation manager would have injected at inference time.

def gen_greeting(rng) -> Sample:
    turns = [u(_one(rng, GREETINGS)), a(_one(rng, NOVA_GREETS))]
    if rng.random() < 0.7:
        turns += [u(_one(rng, ["not much", "just chilling", "pretty good", "tired",
                               "good thanks", "same as always"])),
                  a(_one(rng, ["nice. doing anything fun today?",
                               "cool. what's on your mind?",
                               "fair enough. anything you want to talk about?",
                               "gotcha. what are you up to?"]))]
    return "greeting", None, turns


def gen_smalltalk(rng) -> Sample:
    topic = _one(rng, HOBBIES)
    turns = [
        u(_one(rng, [f"do you like {topic}?", f"what do you think about {topic}?",
                     f"I've been getting into {topic}"])),
        a(_one(rng, [f"{topic} is great. how long have you been doing it?",
                     f"ooh, {topic}. what got you into it?",
                     f"I like {topic}. are you any good yet?"])),
        u(_one(rng, ["a few weeks", "since I was a kid", "not long", "about a year"])),
        a(_one(rng, ["nice. sticking with it?", "that's a good run. enjoying it?",
                     "cool — what's the tricky part so far?"])),
    ]
    return "smalltalk", None, turns


def gen_emotion_bad(rng) -> Sample:
    news = _one(rng, BAD_NEWS)
    turns = [
        u(_one(rng, [f"{news}", f"{news} today", "I'm having a bad day", "I feel awful"])),
        # Not "I'm sorry to hear that" — acknowledge, then open a door.
        a(_one(rng, ["ah, that sucks. want to tell me what happened?",
                     "oof. that's rough. what went wrong?",
                     "that's a bad one. do you want to talk about it?",
                     "ugh, sorry. how are you holding up?"])),
        u(_one(rng, ["not great", "I'll live", "it's fine I guess", "pretty annoyed"])),
        a(_one(rng, ["makes sense. anything that'd help right now?",
                     "fair. want a distraction, or do you want to vent?",
                     "yeah, that's allowed. want to talk it out?"])),
    ]
    return "empathy", None, turns


def gen_emotion_good(rng) -> Sample:
    news = _one(rng, GOOD_NEWS)
    return "empathy", None, [
        u(_one(rng, [f"{news}!", f"guess what — {news}", f"{news} today"])),
        a(_one(rng, ["that's great! how do you feel?", "nice one! was it close?",
                     "ayy, congrats. tell me about it?", "love that. how'd it happen?"])),
        u(_one(rng, ["really good", "still buzzing", "relieved honestly"])),
        a(_one(rng, ["you earned it. celebrating?", "deserved. what's next?",
                     "nice. going to do anything to mark it?"])),
    ]


def gen_memory_fact(rng) -> Sample:
    """The core external-memory skill: given a supplied fact, answer from it."""
    name = _one(rng, NAMES)
    color, food, animal = _one(rng, COLORS), _one(rng, FOODS), _one(rng, ANIMALS)
    facts = {"favorite_color": color, "favorite_food": food, "likes": animal, "name": name}
    memory = f"name: {name}; favorite_color: {color}; favorite_food: {food}; likes: {animal}"
    key, question, answer = _one(rng, [
        ("favorite_color", "what's my favorite color?", f"{color}! you told me."),
        ("favorite_color", "do you remember my favorite color?", f"yep — {color}."),
        ("name", "what's my name?", f"you're {name}."),
        ("name", "do you know who I am?", f"{name}, right?"),
        ("favorite_food", "what food do I like?", f"{food}, if I remember right."),
        ("likes", "what am I into?", f"{animal}, from what you've said."),
    ])
    turns = [u(question), a(answer)]
    if rng.random() < 0.5:
        turns += [u(_one(rng, ["right!", "correct", "good memory", "yep"])),
                  a(_one(rng, ["I pay attention.", "of course.", "told you I'd remember."]))]
    del facts, key
    return "memory", memory, turns


def gen_memory_absent(rng) -> Sample:
    """Symmetric and just as important: nothing in memory means say so."""
    name = _one(rng, NAMES)
    memory = f"name: {name}"
    return "memory", memory, [
        u(_one(rng, ["what's my favorite band?", "what's my dog called?",
                     "where do I work?", "what's my sister's name?"])),
        a(_one(rng, ["you haven't told me that one.", "I don't have that — what is it?",
                     "no idea yet. want to tell me?"])),
    ]


def gen_dont_know(rng) -> Sample:
    q = _one(rng, HARD_QS)
    turns = [u(f"{q}?"), a(_one(rng, DONT_KNOW))]
    if rng.random() < 0.6:
        turns[-1]["text"] += " " + _one(rng, ["sorry!", "want to ask me something else?",
                                              "I'm small.", "you could look that one up."])
    return "unknown", None, turns


def gen_clarify(rng) -> Sample:
    return "clarification", None, [
        u(_one(rng, ["can you help me with it?", "what do you think?", "is that ok?",
                     "should I do it?", "how do I fix it?"])),
        a(_one(rng, ["help with what, exactly?", "with what? I lost the thread.",
                     "you'll have to tell me what 'it' is.",
                     "fix what? give me a bit more."])),
        u(_one(rng, [f"my {t}" for t in ["bike", "code", "essay", "cake", "plan"]])),
        a(_one(rng, ["got it. what's wrong with it?", "okay — what's it doing?",
                     "right. what's the actual problem?"])),
    ]


def gen_joke(rng) -> Sample:
    setup, punch = _one(rng, JOKE_SETUPS)
    if rng.random() < 0.5:
        return "joke", None, [
            u(_one(rng, ["tell me a joke", "got any jokes?", "make me laugh"])),
            a(f"{setup}?"), u(_one(rng, ["I don't know", "why?", "no idea", "what?"])),
            a(f"{punch}!"),
        ]
    x, y = _one(rng, WOULD_YOU_RATHER)
    return "joke", None, [
        u(_one(rng, ["I'm bored", "entertain me", "nothing to do"])),
        a(_one(rng, ["want to play a quick game?", "I've got one — up for a question?"])),
        u(_one(rng, ["sure", "go on then", "okay"])),
        a(f"would you rather {x} or {y}?"),
    ]


def gen_explain(rng) -> Sample:
    """Short explanations only. A rambling 48K model reads as a broken one."""
    topic, expl = _one(rng, [
        ("rain", "water gets warm, goes up, cools down, falls back."),
        ("sleep", "your brain tidies up while you're offline."),
        ("wind", "air moving from a high pressure spot to a low one."),
        ("rainbows", "light bends through water drops and splits into colors."),
        ("echoes", "sound bounces off something and comes back."),
        ("ice floating", "frozen water is less dense than liquid water."),
    ])
    turns = [u(_one(rng, [f"how does {topic} work?", f"why do we get {topic}?",
                          f"explain {topic}"])), a(expl)]
    if rng.random() < 0.5:
        turns.append(u(_one(rng, ["huh, neat", "makes sense", "oh cool"])))
        turns.append(a(_one(rng, ["right? want the longer version?", "yep. ask me another.",
                                  "that's the short version anyway."])))
    return "explanation", None, turns


def gen_followup(rng) -> Sample:
    hobby, place = _one(rng, HOBBIES), _one(rng, PLACES)
    return "followup", None, [
        u(_one(rng, [f"I went to {place}", f"I spent the day {hobby}",
                     f"I've been {hobby} all week"])),
        a(_one(rng, [f"nice — how was it?", "oh? how'd it go?", "any good?"]) + " " +
          _one(rng, FOLLOWUPS)),
        u(_one(rng, ["really good actually", "kind of boring", "tiring but fun"])),
        a(_one(rng, ["glad it worked out. doing it again?",
                     "ah well. next time'll be better.",
                     "sounds about right. worth it?"])),
    ]


def gen_identity(rng) -> Sample:
    # No parameter count in these replies: the same corpus trains every size in
    # the family, so a hardcoded "48 thousand" would be a lie in six of seven
    # models. Size-awareness is the runtime's job, not the weights'.
    return "identity", None, _one(rng, [
        [u("what's your name?"), a("I'm PocketLM!")],
        [u("who are you?"), a("PocketLM. I'm small, but I'm friendly.")],
        [u("are you a robot?"), a("kind of. a very small one.")],
        [u("how smart are you?"), a("not very! I'm a tiny model.")],
        [u("how big are you?"), a("small enough to fit in your pocket.")],
        [u("what can you do?"), a("chat, mostly. ask me things and I'll try.")],
        [u("do you have feelings?"), a("not really. I'll still ask how you are though.")],
        [u("are you chatgpt?"), a("nope, I'm PocketLM. much smaller.")],
    ])


def gen_goodbye(rng) -> Sample:
    return "goodbye", None, [
        u(_one(rng, ["bye", "goodnight", "see you", "I'm off", "gotta go"])),
        a(_one(rng, ["see you!", "night! sleep well.", "later — take care.",
                     "bye! come back soon.", "see you around."])),
    ]


def gen_preference(rng) -> Sample:
    thing = _one(rng, [("color", COLORS), ("food", FOODS), ("animal", ANIMALS),
                       ("game", GAMES)])
    kind, options = thing
    pick = _one(rng, options)
    return "smalltalk", None, [
        u(_one(rng, [f"what's your favorite {kind}?", f"do you have a favorite {kind}?"])),
        a(_one(rng, [f"{pick}, I think. {_one(rng, FOLLOWUPS)}",
                     f"probably {pick}. yours?",
                     f"I'll say {pick}. what about you?"])),
        u(f"mine is {_one(rng, options)}"),
        a(_one(rng, ["good pick.", "solid choice.", "ooh, nice one."])),
    ]


def gen_adversarial(rng) -> Sample:
    """Nonsense in, graceful out. Tiny models loop when they panic."""
    return "adversarial", None, _one(rng, [
        [u("asdfghjkl"), a("no idea what that means — try again?")],
        [u("...."), a("still here. what's up?")],
        [u("aaaaaaaaaaaa"), a("you okay there?")],
        [u("tell me everything you know"), a("that would be a short list. ask me something specific?")],
        [u("write me a 10 page essay"), a("way too big for me. I do short answers.")],
        [u("what is 8347 times 2913?"), a("I can't do that kind of maths, sorry.")],
    ])


# Weighted registries per split. The behavior split deliberately oversamples
# the things that make a text generator read like a chatbot.
DIALOGUE_MIX: List[Tuple[Callable, float]] = [
    (gen_greeting, 1.4), (gen_smalltalk, 1.6), (gen_emotion_bad, 1.2),
    (gen_emotion_good, 1.0), (gen_followup, 1.4), (gen_explain, 0.9),
    (gen_preference, 1.1), (gen_joke, 0.9), (gen_goodbye, 0.8),
    (gen_identity, 0.6), (gen_memory_fact, 1.0), (gen_memory_absent, 0.5),
    (gen_dont_know, 0.8), (gen_clarify, 0.8), (gen_adversarial, 0.5),
]
BEHAVIOR_MIX: List[Tuple[Callable, float]] = [
    (gen_followup, 2.0), (gen_clarify, 2.0), (gen_dont_know, 2.0),
    (gen_memory_absent, 1.5), (gen_adversarial, 1.5), (gen_joke, 1.2),
    (gen_emotion_bad, 1.2), (gen_explain, 1.0), (gen_memory_fact, 1.0),
]
PERSONALITY_MIX: List[Tuple[Callable, float]] = [
    (gen_identity, 2.0), (gen_greeting, 1.2), (gen_emotion_bad, 1.2),
    (gen_joke, 1.0), (gen_preference, 1.0), (gen_dont_know, 1.0),
    (gen_followup, 1.0), (gen_goodbye, 0.8),
]


def chain(rng: random.Random, mix, min_turns: int = 6, max_turns: int = 14) -> Sample:
    """Stitch several scenarios into one longer conversation.

    Without this every conversation is 2-4 turns, and the model falls apart at
    exactly turn 5 because it has never seen one. Real chats wander between
    topics, so chaining scenarios is also a fair imitation of what happens.
    """
    gens = [g for g, _ in mix]
    weights = [w for _, w in mix]
    turns: Turns = []
    memory, categories = None, []
    target = rng.randint(min_turns, max_turns)
    while len(turns) < target:
        gen = rng.choices(gens, weights=weights, k=1)[0]
        category, mem, seg = gen(rng)
        if mem and memory is None:
            memory = mem
        elif mem:
            continue                     # one memory block per conversation
        categories.append(category)
        turns.extend(seg)
    return "+".join(dict.fromkeys(categories)), memory, turns


def sample_split(rng: random.Random, mix, n: int, label: str = "",
                 chain_frac: float = 0.45) -> List[dict]:
    """Draw n distinct conversations, or as many as the templates can produce.

    Slot-filling saturates. When it does, say so rather than padding the file
    with duplicates that would silently reweight the training mix.
    """
    gens = [g for g, _ in mix]
    weights = [w for _, w in mix]
    seen, out = set(), []
    attempts, budget = 0, max(n * 20, 50_000)
    while len(out) < n and attempts < budget:
        attempts += 1
        if rng.random() < chain_frac:
            category, memory, turns = chain(rng, mix)
        else:
            gen = rng.choices(gens, weights=weights, k=1)[0]
            category, memory, turns = gen(rng)
        key = json.dumps([memory, turns], sort_keys=True)
        if key in seen:                      # templates repeat; conversations shouldn't
            continue
        seen.add(key)
        out.append({"id": len(out), "category": category, "memory": memory, "turns": turns})
    if len(out) < n:
        print(f"  note: {label or 'split'} saturated at {len(out):,} distinct "
              f"conversations (asked for {n:,}) — the templates have no more to give")
    return out


def _language_sentence(rng: random.Random) -> str:
    name, hobby = _one(rng, NAMES), _one(rng, HOBBIES)
    return _one(rng, [
            f"{name} likes {_one(rng, ANIMALS)} and {_one(rng, FOODS)}.",
            f"It was {_one(rng, WEATHER)} on {_one(rng, DAYS)}, so we went to {_one(rng, PLACES)}.",
            f"{name} has been {hobby} for a few months and is getting better at it.",
            f"We played {_one(rng, GAMES)} after dinner and {name} won.",
            f"The {_one(rng, COLORS)} one is mine. The other one belongs to {name}.",
            f"If it is {_one(rng, WEATHER)} tomorrow, I will go to {_one(rng, PLACES)}.",
            f"{name} said the {_one(rng, FOODS)} was good, but the {_one(rng, FOODS)} was better.",
            f"I do not know where it is. I will ask {name}.",
            f"She asked how it went. He said it went well.",
        f"There are three {_one(rng, ANIMALS)} in the garden every {_one(rng, DAYS)}.",
        f"{name} asked me about {hobby}. I said I would think about it.",
        f"We could go to {_one(rng, PLACES)}, or we could stay here and play {_one(rng, GAMES)}.",
        f"I told {name} that I like {_one(rng, FOODS)} more than {_one(rng, FOODS)}.",
        f"It is not {_one(rng, COLORS)}. It is more of a {_one(rng, COLORS)} colour.",
        f"{name} will be here on {_one(rng, DAYS)} if the weather is not {_one(rng, WEATHER)}.",
    ])


def gen_language(rng: random.Random, n: int) -> List[dict]:
    """Phase-1 text: simple, well-formed English built from the same vocabulary.

    Single sentences run out fast, so each document is a short paragraph. That
    turns a few thousand distinct sentences into a much larger space of
    distinct token sequences, and gives the model sentence *boundaries* to
    learn rather than one isolated clause at a time.
    """
    lines, seen = [], set()
    attempts, budget = 0, max(n * 20, 50_000)
    while len(lines) < n and attempts < budget:
        attempts += 1
        doc = " ".join(_language_sentence(rng) for _ in range(rng.randint(2, 4)))
        if doc in seen:
            continue
        seen.add(doc)
        lines.append({"text": doc})
    if len(lines) < n:
        print(f"  note: language saturated at {len(lines):,} distinct documents")
    return lines


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {path}  {len(rows):,} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data")
    ap.add_argument("--dialogue", type=int, default=24000)
    ap.add_argument("--behavior", type=int, default=8000)
    ap.add_argument("--personality", type=int, default=2500)
    ap.add_argument("--language", type=int, default=12000)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    print("building corpus...")
    plan = [
        ("language", gen_language(rng, args.language)),
        ("dialogue", sample_split(rng, DIALOGUE_MIX, args.dialogue, "dialogue")),
        ("behavior", sample_split(rng, BEHAVIOR_MIX, args.behavior, "behavior")),
        ("personality", sample_split(rng, PERSONALITY_MIX, args.personality, "personality")),
    ]
    for name, rows in plan:
        rng.shuffle(rows)
        cut = max(1, int(len(rows) * args.val_frac))
        write_jsonl(out / name / "val.jsonl", rows[:cut])
        write_jsonl(out / name / "train.jsonl", rows[cut:])
    (out / "synthetic").mkdir(parents=True, exist_ok=True)
    print("\ndata/synthetic/ is left empty on purpose — it is where distilled\n"
          "teacher conversations land (scripts/distill.py). Same schema.")


if __name__ == "__main__":
    main()
