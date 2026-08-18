"""The conversation manager -- the system around the 48K model.

    user -> manager -> external memory -> 48K model -> filter -> response

The neural network is one component. History windowing, fact injection,
resampling on rejection, and fallbacks all live out here, where they cost no
parameters. This is what makes the whole thing feel bigger than its weights.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from filters import FALLBACKS, FilterConfig, ReplyFilter
from generate import SamplingConfig, respond
from memory import Memory


@dataclass
class ManagerConfig:
    max_turns: int = 8                 # history kept before the window trims it anyway
    retries: int = 3                   # resamples allowed before falling back
    inject_memory: bool = True
    max_facts: int = 4


class ConversationManager:
    def __init__(self, model, tok, memory: Optional[Memory] = None,
                 sampling: Optional[SamplingConfig] = None,
                 cfg: Optional[ManagerConfig] = None,
                 filter_cfg: Optional[FilterConfig] = None,
                 seed: int = 0):
        self.model, self.tok = model, tok
        self.memory = memory if memory is not None else Memory()
        self.sampling = sampling or SamplingConfig()
        self.cfg = cfg or ManagerConfig()
        self.filter = ReplyFilter(filter_cfg or FilterConfig())
        self.turns: List[Dict[str, str]] = []
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.turns.clear()
        self.filter.recent.clear()

    def reply(self, user_text: str) -> str:
        self.memory.observe(user_text)
        self.turns.append({"role": "user", "text": user_text})
        self.turns = self.turns[-self.cfg.max_turns:]

        mem = (self.memory.render(user_text, self.cfg.max_facts)
               if self.cfg.inject_memory else None)

        text = None
        for attempt in range(self.cfg.retries):
            cfg = self.sampling
            if attempt:                 # first resample is a nudge, not a repeat
                cfg = SamplingConfig(**{**cfg.__dict__,
                                        "temperature": cfg.temperature + 0.15 * attempt})
            candidate = respond(self.model, self.tok, self.turns, memory=mem, cfg=cfg)
            text = self.filter.check(candidate)
            if text:
                break

        if not text:
            text = self.rng.choice(FALLBACKS)
        self.filter.accept(text)
        self.turns.append({"role": "assistant", "text": text})
        return text

    def debug(self, user_text: str) -> Dict:
        """Same as reply(), but returns what every stage did. Used by eval."""
        mem = self.memory.render(user_text, self.cfg.max_facts)
        ids, _ = self.tok.encode_turns(
            self.turns + [{"role": "user", "text": user_text}],
            memory=mem, open_reply=True)
        return {"memory": mem, "prompt_tokens": len(ids), "reply": self.reply(user_text)}
