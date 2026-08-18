"""Character-level BPE for the PocketLM family, implemented in plain Python.

The embedding table is the dominant cost at every size in this family: it is
vocab_size x d_model, so at 1K parameters a 512-token vocabulary would be 4x
the entire model. Vocabulary therefore scales with the model, and one
tokenizer is trained per size: 64, 128, 256, 512, 640 and 1024 tokens.

Two details matter more than they look:

* The alphabet always includes the core ASCII letters, however rare. Learned
  purely from frequency, a capital that appears only in "PocketLM" falls below
  the threshold and the model becomes literally unable to spell its own name.

* Below 256 tokens the tokenizer folds case away. A model that small has no
  capacity to spend on knowing that "Hey" and "hey" are the same word, and the
  alphabet slots are worth far more as merges.
"""

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from config import SPECIAL_TOKENS, PAD, BOS, EOS, USER, ASSISTANT, MEM, UNK

# Words, numbers one digit at a time, punctuation runs, whitespace. Merges
# never cross these boundaries, which keeps a tiny vocabulary from wasting
# slots on cross-word noise like "you.I".
_SPLIT = re.compile(r"[ ]?[A-Za-z]+|[ ]?[0-9]|[ ]?[^\sA-Za-z0-9]+|\s+")

# Always in the alphabet, however rare. Learned purely from frequency, a
# capital that appears only in "PocketLM" falls below the threshold and the bot
# becomes unable to spell its own name -- which is exactly what happened.
CORE_ALPHABET = sorted(set(string.ascii_letters + string.digits + " .,!?'\"-:;()"))


class BPETokenizer:
    def __init__(self, alphabet: Sequence[str], merges: Sequence[Tuple[str, str]],
                 specials: Sequence[str] = SPECIAL_TOKENS, lowercase: bool = False):
        self.specials = list(specials)
        self.lowercase = lowercase
        self.alphabet = list(alphabet)
        self.merges = [tuple(m) for m in merges]
        self.itos: List[str] = self.specials + self.alphabet + [a + b for a, b in self.merges]
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)}
        self.ranks: Dict[Tuple[str, str], int] = {m: i for i, m in enumerate(self.merges)}
        self._alphabet_set = set(self.alphabet)
        self._cache: Dict[str, List[int]] = {}
        for name in ("pad", "bos", "eos", "user", "assistant", "mem", "unk"):
            setattr(self, f"{name}_id", self.stoi[f"<{name}>"])

    def __len__(self) -> int:
        return len(self.itos)

    # ------------------------------------------------------------- training

    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int,
              specials: Sequence[str] = SPECIAL_TOKENS,
              min_symbol_count: int = 20, lowercase: bool = False,
              min_merges_frac: float = 0.25, verbose: bool = True) -> "BPETokenizer":
        word_freq: Counter = Counter()
        for text in texts:
            if lowercase:
                text = text.lower()
            word_freq.update(_SPLIT.findall(text))

        sym_freq: Counter = Counter()
        for word, n in word_freq.items():
            for ch in word:
                sym_freq[ch] += n
        core = {c for c in CORE_ALPHABET if not (lowercase and c.isupper())}
        alphabet = sorted(core | {c for c, n in sym_freq.items()
                                  if n >= min_symbol_count and not (lowercase and c.isupper())})

        # At tiny vocab sizes the alphabet alone would consume everything, so
        # cap it and keep the most frequent symbols. What is dropped becomes
        # <unk> -- an explicit, visible loss rather than a silent one.
        cap = vocab_size - len(specials) - int(vocab_size * min_merges_frac)
        if len(alphabet) > cap:
            if cap < 1:
                raise ValueError(f"vocab_size {vocab_size} is too small for "
                                 f"{len(specials)} special tokens")
            keep = sorted(alphabet, key=lambda c: -sym_freq.get(c, 0))[:cap]
            dropped = sorted(set(alphabet) - set(keep))
            if verbose:
                print(f"  vocab {vocab_size}: alphabet capped at {cap}, "
                      f"{len(dropped)} symbols folded into <unk>: {''.join(dropped)!r}")
            alphabet = sorted(keep)

        budget = vocab_size - len(specials) - len(alphabet)

        known = set(alphabet)
        splits = {w: [c if c in known else UNK for c in w] for w in word_freq}
        merges: List[Tuple[str, str]] = []

        pairs = cls._count_pairs(splits, word_freq)
        while len(merges) < budget and pairs:
            best, count = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
            if count < 2:
                break
            merges.append(best)
            new_sym = best[0] + best[1]
            for word in [w for w, s in splits.items() if best[0] in s]:
                splits[word] = cls._apply(splits[word], best, new_sym)
            pairs = cls._count_pairs(splits, word_freq)
            if verbose and len(merges) % 50 == 0:
                print(f"  merge {len(merges):>4}/{budget}  {new_sym!r:>14}  x{count:,}")

        return cls(alphabet, merges, specials, lowercase)

    @staticmethod
    def _count_pairs(splits: Dict[str, List[str]], freq: Counter) -> Counter:
        pairs: Counter = Counter()
        for word, parts in splits.items():
            n = freq[word]
            for a, b in zip(parts, parts[1:]):
                pairs[(a, b)] += n
        return pairs

    @staticmethod
    def _apply(parts: List[str], pair: Tuple[str, str], new_sym: str) -> List[str]:
        out, i = [], 0
        while i < len(parts):
            if i < len(parts) - 1 and (parts[i], parts[i + 1]) == pair:
                out.append(new_sym)
                i += 2
            else:
                out.append(parts[i])
                i += 1
        return out

    # -------------------------------------------------------------- encoding

    def _encode_word(self, word: str) -> List[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        parts = [c if c in self._alphabet_set else UNK for c in word]
        while len(parts) > 1:
            ranked = [(self.ranks.get(p, 1 << 30), i)
                      for i, p in enumerate(zip(parts, parts[1:]))]
            rank, i = min(ranked)
            if rank == 1 << 30:
                break
            parts = self._apply(parts, (parts[i], parts[i + 1]), parts[i] + parts[i + 1])
        ids = [self.stoi.get(p, self.unk_id) for p in parts]
        self._cache[word] = ids
        return ids

    def encode(self, text: str) -> List[int]:
        if self.lowercase:
            text = text.lower()
        out: List[int] = []
        for word in _SPLIT.findall(text):
            out.extend(self._encode_word(word))
        return out

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        pieces = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                continue
            tok = self.itos[i]
            if skip_special and tok in self.specials:
                continue
            pieces.append(tok)
        return "".join(pieces)

    # ------------------------------------------------------- chat framing

    def encode_turns(self, turns: Sequence[dict], add_bos: bool = True,
                     memory: Optional[str] = None,
                     open_reply: bool = False) -> Tuple[List[int], List[int]]:
        """Render a conversation to (ids, assistant_mask).

        assistant_mask[i] == 1 marks a token the model is scored on: assistant
        content plus the <eos> that ends it. Everything the user typed, and
        every control token, is context only.
        """
        ids: List[int] = [self.bos_id] if add_bos else []
        mask: List[int] = [0] * len(ids)
        if memory:
            ids.append(self.mem_id)
            mask.append(0)
            body = self.encode(memory)
            ids.extend(body)
            mask.extend([0] * len(body))
        for turn in turns:
            role = turn["role"]
            head = self.user_id if role == "user" else self.assistant_id
            ids.append(head)
            mask.append(0)
            body = self.encode(turn["text"])
            ids.extend(body)
            mask.extend([1 if role == "assistant" else 0] * len(body))
            ids.append(self.eos_id)
            mask.append(1 if role == "assistant" else 0)
        if open_reply:                       # prompt ends ready for generation
            ids.append(self.assistant_id)
            mask.append(0)
        return ids, mask

    # ---------------------------------------------------------------- io

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "specials": self.specials,
            "lowercase": self.lowercase,
            "alphabet": self.alphabet,
            "merges": [list(m) for m in self.merges],
        }, ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text())
        return cls(d["alphabet"], [tuple(m) for m in d["merges"]], d["specials"],
                   d.get("lowercase", False))
