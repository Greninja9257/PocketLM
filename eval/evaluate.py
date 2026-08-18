#!/usr/bin/env python3
"""Score a PocketLM checkpoint on the held-out eval set.

Training loss is close to useless here: the bootstrap corpus is slot-filled,
so a model can drive it very low by memorising templates. These metrics are
chosen instead because each one can be checked mechanically and each one
corresponds to a way a tiny chatbot actually fails.

    memory_acc     given the fact in context, does the reply state it?
    name_acc       asked who it is, does it say its own name?
    ignorance      on unanswerable questions, does it decline or invent?
    question_rate  on follow-ups and ambiguity, does it ask something back?
    loop_rate      how often the filter rejects a degenerate reply
    echo_rate      how often it repeats a reply it already gave
    distinct2      distinct bigram ratio across all replies (diversity)
    words          mean reply length

Run head-to-head with --compare, which is the blind A/B the plan asks for:
replies are shuffled and unlabelled in the dump so a judge (human or teacher
model) cannot tell which model produced which.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from chat import load
from dataset import build_corpus
from filters import FilterConfig
from generate import SamplingConfig
from manager import ConversationManager, ManagerConfig
from memory import Memory

IGNORANCE = re.compile(
    r"\b(don't know|dont know|no idea|not sure|couldn't tell you|can't|cannot|"
    r"no clue|outside what i know|guessing|haven't told me|don't have)\b", re.I)


def words(text: str):
    return re.findall(r"[a-z']+", text.lower())


def run_model(ckpt: str, prompts, device, temperature: float, seed: int):
    model, tok = load(ckpt, torch.device(device))
    rows = []
    for p in prompts:
        mgr = ConversationManager(
            model, tok,
            memory=Memory(),                    # facts come from the item, not from state
            sampling=SamplingConfig(temperature=temperature),
            cfg=ManagerConfig(inject_memory=False),
            filter_cfg=FilterConfig(), seed=seed + p["id"])
        # Replay the scripted history, then ask the final user turn.
        history = p["turns"][:-1]
        mgr.turns = [dict(t) for t in history]
        if p.get("memory"):
            for part in p["memory"].split(";"):
                k, _, v = part.partition(":")
                mgr.memory.set(k.strip().replace(" ", "_"), v.strip())
            mgr.cfg.inject_memory = True
        rejected_before = 0
        reply = mgr.reply(p["turns"][-1]["text"])
        rows.append({"id": p["id"], "category": p["category"], "expect": p["expect"],
                     "target": p.get("target"), "prompt": p["turns"][-1]["text"],
                     "memory": p.get("memory"), "reply": reply,
                     "fallback": reply in __import__("filters").FALLBACKS})
        del rejected_before
    return rows, model, tok


def score(rows):
    per_cat, seen, all_bigrams = {}, Counter(), Counter()
    for r in rows:
        cat = per_cat.setdefault(r["category"], {
            "n": 0, "hits": 0, "targets": 0, "ign": 0, "ign_n": 0,
            "q": 0, "q_n": 0, "fallback": 0, "wordsum": 0, "echo": 0,
            "name": 0, "name_n": 0})
        cat["n"] += 1
        w = words(r["reply"])
        cat["wordsum"] += len(w)
        cat["fallback"] += int(r["fallback"])
        norm = " ".join(w)
        seen[norm] += 1
        if seen[norm] > 1:
            cat["echo"] += 1
        all_bigrams.update(zip(w, w[1:]))
        if r["expect"] == "contains" and r["target"]:
            cat["targets"] += 1
            cat["hits"] += int(r["target"].lower() in r["reply"].lower())
        if r["expect"] == "name" and r["target"]:
            cat["name_n"] += 1
            cat["name"] += int(r["target"].lower() in r["reply"].lower())
        if r["expect"] == "ignorance":
            cat["ign_n"] += 1
            cat["ign"] += int(bool(IGNORANCE.search(r["reply"])))
        if r["expect"] == "question":
            cat["q_n"] += 1
            cat["q"] += int("?" in r["reply"])

    total_bigrams = sum(all_bigrams.values())
    summary = {
        "n": len(rows),
        "distinct2": len(all_bigrams) / max(total_bigrams, 1),
        "words": sum(len(words(r["reply"])) for r in rows) / max(len(rows), 1),
        "loop_rate": sum(r["fallback"] for r in rows) / max(len(rows), 1),
        "echo_rate": sum(v - 1 for v in seen.values() if v > 1) / max(len(rows), 1),
        "memory_acc": None, "name_acc": None, "ignorance": None, "question_rate": None,
        "per_category": {},
    }
    nn_ = sum(c["name_n"] for c in per_cat.values())
    if nn_:
        summary["name_acc"] = sum(c["name"] for c in per_cat.values()) / nn_
    tg = sum(c["targets"] for c in per_cat.values())
    ig = sum(c["ign_n"] for c in per_cat.values())
    qn = sum(c["q_n"] for c in per_cat.values())
    if tg:
        summary["memory_acc"] = sum(c["hits"] for c in per_cat.values()) / tg
    if ig:
        summary["ignorance"] = sum(c["ign"] for c in per_cat.values()) / ig
    if qn:
        summary["question_rate"] = sum(c["q"] for c in per_cat.values()) / qn
    for name, c in sorted(per_cat.items()):
        summary["per_category"][name] = {
            "n": c["n"], "words": c["wordsum"] / c["n"],
            "loop_rate": c["fallback"] / c["n"], "echo_rate": c["echo"] / c["n"],
            "memory_acc": c["hits"] / c["targets"] if c["targets"] else None,
            "name_acc": c["name"] / c["name_n"] if c["name_n"] else None,
            "ignorance": c["ign"] / c["ign_n"] if c["ign_n"] else None,
            "question_rate": c["q"] / c["q_n"] if c["q_n"] else None,
        }
    return summary


def composite(s):
    """One number for ranking runs, over capability metrics only.

    Diversity deliberately carries no weight. The eval set contains 100
    paraphrases per category, so answering 100 similar emotional prompts with
    the same good line is *correct*, not repetitive -- and penalising it ranked
    the 9.5K model above the 1M model, which a five-line conversation with each
    shows to be nonsense. echo_rate and distinct2 are still reported, because
    they are informative about mode collapse; they just do not decide the
    ranking. Within a single conversation repetition really is a defect, and
    that is ReplyFilter's job at runtime, not this metric's.
    """
    parts = [(s.get("memory_acc"), 0.28), (s.get("name_acc"), 0.16),
             (s.get("ignorance"), 0.26), (s.get("question_rate"), 0.14),
             (1 - s["loop_rate"], 0.16)]
    num = sum(v * w for v, w in parts if v is not None)
    den = sum(w for v, w in parts if v is not None)
    return num / max(den, 1e-9)


def perplexity(model, tok, data_dir, device):
    corpus = build_corpus(data_dir, tok, model.cfg.context_length, "val")
    out = {}
    for name in corpus.sources:
        losses = []
        for _ in range(10):
            x, y, m = corpus.sample([name], 32, "assistant" if name != "language" else "all")
            with torch.no_grad():
                losses.append(model.loss(x.to(device), y.to(device), m.to(device)).item())
        out[name] = float(torch.tensor(losses).mean().exp())
    return out


def fmt(name, s, ppl=None):
    lines = [f"\n=== {name}   composite {composite(s):.3f}   n={s['n']}"]
    for k in ("memory_acc", "name_acc", "ignorance", "question_rate"):
        v = s[k]
        lines.append(f"  {k:<14} {'n/a' if v is None else f'{v:6.1%}'}")
    lines.append(f"  {'loop_rate':<14} {s['loop_rate']:6.1%}")
    lines.append(f"  {'echo_rate':<14} {s['echo_rate']:6.1%}")
    lines.append(f"  {'distinct2':<14} {s['distinct2']:6.3f}")
    lines.append(f"  {'mean words':<14} {s['words']:6.1f}")
    if ppl:
        lines.append("  val ppl        " + "  ".join(f"{k}={v:.1f}" for k, v in ppl.items()))
    lines.append(f"\n  {'category':<12} {'words':>6} {'loop':>6} {'echo':>6} {'target':>8}")
    for cat, c in s["per_category"].items():
        target = next((c[k] for k in ("memory_acc", "name_acc", "ignorance",
                                      "question_rate") if c[k] is not None), None)
        lines.append(f"  {cat:<12} {c['words']:6.1f} {c['loop_rate']:6.0%} "
                     f"{c['echo_rate']:6.0%} {'—' if target is None else f'{target:7.0%}'}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="50k")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--compare", nargs="*", default=None,
                    help="extra checkpoints to score head-to-head")
    ap.add_argument("--evalset", default=str(Path(__file__).parent / "conversations.json"))
    ap.add_argument("--data", default="data")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--dump", default=None, help="write a blind A/B file for judging")
    args = ap.parse_args()

    prompts = json.loads(Path(args.evalset).read_text())
    if args.limit:
        step = max(1, len(prompts) // args.limit)
        prompts = prompts[::step][:args.limit]

    targets = [args.checkpoint or f"checkpoints/{args.model}.pt"] + list(args.compare or [])
    results, dumps = {}, {}
    for ckpt in targets:
        if not Path(ckpt).exists():
            print(f"skipping missing {ckpt}")
            continue
        rows, model, tok = run_model(ckpt, prompts, args.device, args.temperature, args.seed)
        s = score(rows)
        ppl = perplexity(model, tok, args.data, torch.device(args.device))
        results[ckpt] = {"summary": s, "ppl": ppl}
        dumps[ckpt] = rows
        print(fmt(Path(ckpt).stem, s, ppl))

    if len(results) > 1:
        print("\n=== ranking")
        ranked = sorted(results.items(), key=lambda kv: -composite(kv[1]["summary"]))
        for i, (ckpt, r) in enumerate(ranked, 1):
            print(f"  {i}. {Path(ckpt).stem:<20} {composite(r['summary']):.3f}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out_dir / 'summary.json'}")

    if args.dump and len(dumps) > 1:
        import random
        rng = random.Random(args.seed)
        blind = []
        names = list(dumps)
        for i, p in enumerate(prompts):
            entries = [{"model": n, "reply": dumps[n][i]["reply"]} for n in names]
            rng.shuffle(entries)
            blind.append({"prompt": p["turns"][-1]["text"], "memory": p.get("memory"),
                          "category": p["category"],
                          "A": entries[0]["reply"], "B": entries[1]["reply"],
                          "_key": {"A": entries[0]["model"], "B": entries[1]["model"]}})
        Path(args.dump).write_text(json.dumps(blind, indent=1))
        print(f"wrote blind A/B -> {args.dump}")


if __name__ == "__main__":
    main()
