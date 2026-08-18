"""The last stage before a reply reaches the user.

A 48K model fails in predictable ways: it loops ("what what what"), it trails
off mid-word, it repeats its previous line, it occasionally produces nothing.
None of those need to be fixed in the weights -- they are cheaper to catch
here, which is the entire argument for the wrapper system.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

_BLOCKLIST = re.compile(
    r"\b(kill yourself|kys|suicide|self[- ]harm)\b", re.I
)
SAFE_DEFLECTION = "that's not something I can talk about. want to change the subject?"

# Words a reply should never end on: the model ran out of steam mid-clause.
_DANGLING = {"and", "but", "or", "so", "the", "a", "an", "to", "of", "with", "for",
             "that", "is", "was", "if", "my", "your", "i", "it's", "in", "on", "at"}

FALLBACKS = [
    "I didn't quite get that — say it another way?",
    "hmm, I'm not sure. what do you mean?",
    "you've lost me there. can you rephrase?",
]


@dataclass
class FilterConfig:
    max_words: int = 30
    max_repeat_ngram: int = 3          # an n-gram may not appear more than twice
    history_window: int = 3            # how many past replies count as "recent"
    min_chars: int = 1


@dataclass
class ReplyFilter:
    cfg: FilterConfig = field(default_factory=FilterConfig)
    recent: List[str] = field(default_factory=list)

    def check(self, text: str) -> Optional[str]:
        """Return a cleaned reply, or None if it should be rejected."""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) < self.cfg.min_chars:
            return None
        if _BLOCKLIST.search(text):
            return SAFE_DEFLECTION

        words = text.split()
        if len(words) > self.cfg.max_words:                 # truncate at a boundary
            clipped = " ".join(words[:self.cfg.max_words])
            cut = max(clipped.rfind(c) for c in ".!?")
            text = clipped[:cut + 1] if cut > 10 else clipped
            words = text.split()

        if self._loops(words):
            return None
        if self._echoes(text):
            return None
        return self._tidy(text)

    def _loops(self, words: List[str]) -> bool:
        n = self.cfg.max_repeat_ngram
        if len(words) < 4:
            return False
        for size in range(1, n + 1):
            grams = [tuple(words[i:i + size]) for i in range(len(words) - size + 1)]
            if not grams:
                continue
            counts = {}
            for g in grams:
                counts[g] = counts.get(g, 0) + 1
            # allow ordinary repetition of short function words
            worst = max(counts.values())
            if worst > 2 and (size > 1 or len(set(words)) < len(words) / 2):
                return True
        return False

    def _echoes(self, text: str) -> bool:
        norm = re.sub(r"[^a-z ]", "", text.lower()).strip()
        return any(norm == re.sub(r"[^a-z ]", "", r.lower()).strip()
                   for r in self.recent[-self.cfg.history_window:])

    @staticmethod
    def _tidy(text: str) -> str:
        text = re.sub(r"\s+([,.!?'])", r"\1", text)
        words = text.split()
        while words and words[-1].strip(",.;:").lower() in _DANGLING:
            words.pop()
        text = " ".join(words)
        if text and text[-1] not in ".!?":
            # cut a dangling half-sentence rather than shipping it
            cut = max(text.rfind(c) for c in ".!?")
            text = text[:cut + 1] if cut > 8 else text.rstrip(" ,;:-") + "."
        return text

    def accept(self, text: str) -> str:
        self.recent.append(text)
        self.recent = self.recent[-8:]
        return text
