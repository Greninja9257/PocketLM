"""External memory: the cheapest capability upgrade available.

The network does not need to store that Alex likes cats. It needs to learn the
much smaller skill of turning a supplied fact into a sentence. So facts live in
a plain dict on disk, and the manager injects the relevant ones into the 128
token window as a <mem> prefix before every reply.

This is why the corpus contains memory-grounded conversations in both
directions: fact present -> use it, fact absent -> say you don't have it.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

# Ordered: the first pattern that matches a sentence wins, so the more specific
# ones ("my favorite colour is X") must precede the general ones ("I like X").
_RULES = [
    ("name", re.compile(r"\b(?:my name is|i am|i'm|call me)\s+([A-Z][a-z]{1,14})\b")),
    ("favorite_color", re.compile(r"\bfa[vf]ou?rite colou?r (?:is|=)\s+([a-z]{3,12})\b", re.I)),
    ("favorite_food", re.compile(r"\bfa[vf]ou?rite food (?:is|=)\s+([a-z ]{3,20})\b", re.I)),
    ("favorite_animal", re.compile(r"\bfa[vf]ou?rite animal (?:is|=)\s+([a-z ]{3,20})\b", re.I)),
    ("job", re.compile(r"\bi (?:work as|am)\s+an?\s+([a-z ]{3,20})\b", re.I)),
    ("location", re.compile(r"\bi live in\s+([A-Za-z ]{3,20})\b", re.I)),
]
_LIKES = re.compile(r"\bi (?:really )?(?:like|love|enjoy)\s+([a-z ]{3,24})\b", re.I)
_DISLIKES = re.compile(r"\bi (?:really )?(?:hate|dislike|can't stand)\s+([a-z ]{3,24})\b", re.I)

_STOP = {"it", "that", "this", "you", "them", "him", "her", "to", "the", "a", "when"}


def format_fact(key: str, value) -> str:
    """Render one fact for the <mem> line.

    This is the single source of truth for the format, imported by
    scripts/build_corpus.py so training data and inference cannot drift. They
    did drift once: the corpus wrote "favorite_color: blue" while inference
    rendered "favorite color: blue", and the model -- which had only ever seen
    the underscore form -- scored 0% on every memory question.
    """
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value)
    return f"{key}: {value}"


def format_facts(facts: Dict[str, object]) -> str:
    return "; ".join(format_fact(k, v) for k, v in facts.items())


def _clean(value: str, lower: bool = True) -> str:
    value = value.strip().strip(".!,")
    if lower:
        value = value.lower()
    for tail in (" and ", " but ", " because ", " so "):
        value = value.split(tail)[0]
    return value.strip()


class Memory:
    """A small key/value store with list-valued 'likes' and 'dislikes'."""

    MAX_LIST = 5

    def __init__(self, facts: Optional[Dict] = None, path: Optional[str] = None):
        self.facts: Dict[str, object] = dict(facts or {})
        self.path = Path(path) if path else None

    # ------------------------------------------------------------- updating

    def observe(self, text: str) -> List[str]:
        """Pull facts out of a user turn. Returns the keys that changed."""
        changed = []
        for key, rx in _RULES:
            m = rx.search(text)
            if m:
                # Names keep their capitalisation; the corpus trains on "you're Alex."
                value = _clean(m.group(1), lower=(key != "name"))
                if value and value not in _STOP and self.facts.get(key) != value:
                    self.facts[key] = value
                    changed.append(key)
        for key, rx in (("likes", _LIKES), ("dislikes", _DISLIKES)):
            m = rx.search(text)
            if not m:
                continue
            value = _clean(m.group(1))
            if not value or value in _STOP:
                continue
            bucket = list(self.facts.get(key, []))
            if value not in bucket:
                bucket.append(value)
                self.facts[key] = bucket[-self.MAX_LIST:]
                changed.append(key)
        if changed:
            self.save()
        return changed

    def set(self, key: str, value) -> None:
        self.facts[key] = value
        self.save()

    def forget(self, key: Optional[str] = None) -> None:
        if key is None:
            self.facts.clear()
        else:
            self.facts.pop(key, None)
        self.save()

    # ------------------------------------------------------------ rendering

    _fmt = staticmethod(format_fact)

    def render(self, query: str = "", max_facts: int = 4) -> Optional[str]:
        """Render the facts most relevant to the query, most relevant first.

        Relevance matters because the whole conversation has to fit in 128
        tokens. A question about colour should not spend the window on a job
        title.
        """
        if not self.facts:
            return None
        words = set(re.findall(r"[a-z]+", query.lower()))
        scored = []
        for key, value in self.facts.items():
            key_words = set(key.replace("_", " ").split())
            score = len(key_words & words) * 2
            text = self._fmt(key, value)
            score += len(set(re.findall(r"[a-z]+", text.lower())) & words)
            if key == "name":
                score += 1                       # identity is cheap and often useful
            scored.append((score, key, text))
        scored.sort(key=lambda t: -t[0])
        return "; ".join(text for _, _, text in scored[:max_facts])

    # ------------------------------------------------------------------- io

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.facts, indent=1))

    @classmethod
    def load(cls, path: str) -> "Memory":
        p = Path(path)
        return cls(json.loads(p.read_text()) if p.exists() else {}, path)
