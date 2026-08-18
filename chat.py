#!/usr/bin/env python3
"""Talk to PocketLM.

    python chat.py --arch transformer

Commands: /reset  /mem  /forget [key]  /remember k=v  /temp 0.8  /debug  /quit
"""

import argparse
from pathlib import Path

import torch

from generate import SamplingConfig
from manager import ConversationManager, ManagerConfig
from memory import Memory
from model import build_model, describe
from config import ModelConfig
from tokenizer import BPETokenizer

BANNER = """PocketLM
{desc}
type /help for commands, /quit to leave
"""

HELP = """  /reset          clear the conversation (memory survives)
  /mem            show what PocketLM remembers
  /remember k=v   set a fact by hand
  /forget [key]   drop one fact, or all of them
  /temp 0.8       change sampling temperature
  /debug          toggle prompt/memory inspection
  /quit           exit"""


def load(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tok = BPETokenizer.load(ckpt.get("tokenizer") or cfg.tokenizer)
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="50k", help="family member, e.g. 1k 5k 10k 50k 100k 500k 1m")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--memory", default="checkpoints/memory.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--no-memory", action="store_true")
    args = ap.parse_args()

    ckpt_path = args.checkpoint or f"checkpoints/{args.model}.pt"
    if not Path(ckpt_path).exists():
        raise SystemExit(f"no checkpoint at {ckpt_path} — run: python train.py --model {args.model}")

    device = torch.device(args.device)
    model, tok = load(ckpt_path, device)
    mgr = ConversationManager(
        model, tok,
        memory=Memory() if args.no_memory else Memory.load(args.memory),
        sampling=SamplingConfig(temperature=args.temperature, top_p=args.top_p,
                                repetition_penalty=args.repetition_penalty,
                                max_new_tokens=args.max_new_tokens),
        cfg=ManagerConfig(inject_memory=not args.no_memory),
    )
    print(BANNER.format(desc=describe(model)))
    debug = False

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line[1:].partition(" ")
            rest = rest.strip()
            if cmd in ("quit", "exit", "q"):
                break
            if cmd == "help":
                print(HELP)
            elif cmd == "reset":
                mgr.reset()
                print("  (conversation cleared)")
            elif cmd == "mem":
                print("  " + (str(mgr.memory.facts) if mgr.memory.facts else "(nothing yet)"))
            elif cmd == "remember" and "=" in rest:
                k, _, v = rest.partition("=")
                mgr.memory.set(k.strip(), v.strip())
                print(f"  remembered {k.strip()}={v.strip()}")
            elif cmd == "forget":
                mgr.memory.forget(rest or None)
                print(f"  forgot {rest or 'everything'}")
            elif cmd == "temp":
                try:
                    mgr.sampling.temperature = float(rest)
                    print(f"  temperature = {mgr.sampling.temperature}")
                except ValueError:
                    print("  usage: /temp 0.8")
            elif cmd == "debug":
                debug = not debug
                print(f"  debug {'on' if debug else 'off'}")
            else:
                print("  unknown command; /help")
            continue

        if debug:
            info = mgr.debug(line)
            print(f"  [mem: {info['memory']}]")
            print(f"  [prompt: {info['prompt_tokens']}/{model.cfg.context_length} tokens]")
            print(f"pocketlm> {info['reply']}")
        else:
            print(f"pocketlm> {mgr.reply(line)}")


if __name__ == "__main__":
    main()
