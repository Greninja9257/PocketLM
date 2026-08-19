"""Model configuration and the parameter budgets for the PocketLM family.

Seven sizes, from 1K to 1M parameters, all the same architecture and all
trained on the same data with the same curriculum. The point of the family is
to make the scaling behaviour visible: what does a chatbot gain, concretely,
between 1,000 parameters and 1,000,000?

Every size is budget-checked. config.n_params() computes the count
analytically and model.py asserts the real module agrees, so a config that
busts its budget cannot be built.

The single hardest constraint at the small end is the embedding table: it
costs vocab_size x d_model, so at 1K parameters a 512-token vocabulary is
already 4x the entire budget. Vocabulary therefore scales with the model,
from 64 tokens (essentially character-level) up to 1,024.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

ASSISTANT_NAME = "PocketLM"

# ---------------------------------------------------------------- tokenizer

PAD, BOS, EOS, USER, ASSISTANT, MEM, UNK = (
    "<pad>", "<bos>", "<eos>", "<user>", "<assistant>", "<mem>", "<unk>",
)
SPECIAL_TOKENS: List[str] = [PAD, BOS, EOS, USER, ASSISTANT, MEM, UNK]

# Below this vocabulary size the tokenizer folds case away entirely. A model
# this small has no capacity to spend on knowing that "Hey" and "hey" are the
# same word, and the alphabet slots are worth more as merges.
LOWERCASE_BELOW_VOCAB = 256


def tokenizer_path(vocab_size: int) -> str:
    return f"checkpoints/tokenizer-{vocab_size}.json"


# ------------------------------------------------------------------ configs

@dataclass
class ModelConfig:
    name: str = "50k"
    arch: str = "transformer"          # transformer | gru | hybrid
    vocab_size: int = 512
    context_length: int = 128
    d_model: int = 32
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4                # < n_heads gives grouped-query attention
    d_ff: int = 40
    gru_hidden: int = 32
    gru_layers: int = 2
    rope_theta: float = 10_000.0
    budget: int = 50_000

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def n_params(self) -> int:
        """Exact trainable parameter count, computed without building anything."""
        d, V = self.d_model, self.vocab_size
        total = V * d                              # tied token embedding + LM head
        if self.arch == "transformer":
            total += self.n_layers * (self._attn() + self._swiglu(self.d_ff) + 2 * d)
            total += d                             # final norm
        elif self.arch == "gru":
            h = self.gru_hidden
            for i in range(self.gru_layers):
                total += self._gru(d if i == 0 else h, h)
            if h != d:
                total += h * d
            total += d
        elif self.arch == "hybrid":
            h = self.gru_hidden
            for i in range(self.gru_layers):
                total += self._gru(d if i == 0 else h, h)
            if h != d:
                total += h * d
            total += self._attn() + d              # one attention block + its norm
            total += self._swiglu(self.d_ff) + d   # one SwiGLU block + its norm
            total += d
        else:
            raise ValueError(f"unknown arch {self.arch!r}")
        return total

    def _attn(self) -> int:
        d, hd = self.d_model, self.head_dim
        return d * d + 2 * (d * hd * self.n_kv_heads) + d * d   # q, k, v, o, bias-free

    def _swiglu(self, ff: int) -> int:
        return 3 * self.d_model * ff                            # gate, up, down

    @staticmethod
    def _gru(n_in: int, h: int) -> int:
        return 3 * h * n_in + 3 * h * h + 6 * h                 # torch nn.GRU layout

    def check_budget(self, budget: Optional[int] = None) -> int:
        budget = budget or self.budget
        n = self.n_params()
        if n > budget:
            raise ValueError(
                f"config {self.name!r} needs {n:,} params, budget is {budget:,} "
                f"(over by {n - budget:,})"
            )
        return n

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"{self.name}: d_model {self.d_model} not divisible by "
                             f"n_heads {self.n_heads}")
        if self.head_dim % 2:
            raise ValueError(f"{self.name}: head_dim {self.head_dim} must be even for RoPE")
        if self.n_heads % self.n_kv_heads:
            raise ValueError(f"{self.name}: n_heads must be divisible by n_kv_heads")
        self.check_budget()

    @property
    def tokenizer(self) -> str:
        return tokenizer_path(self.vocab_size)

    @property
    def lowercase(self) -> bool:
        return self.vocab_size < LOWERCASE_BELOW_VOCAB

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ModelConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ------------------------------------------------------------------- family
# Each entry is the largest configuration that fits under its budget. Vocab,
# width, depth and context all scale together; the architecture never changes.

FAMILY: Dict[str, ModelConfig] = {
    # 984 params. One layer, 8 dims, 64-token (character-level) vocabulary.
    "1k":   ModelConfig(name="1k",   budget=1_000,     vocab_size=64,   context_length=64,
                        d_model=8,   n_layers=1, n_heads=2, n_kv_heads=2, d_ff=8),
    # 4,944 params.
    "5k":   ModelConfig(name="5k",   budget=5_000,     vocab_size=128,  context_length=96,
                        d_model=16,  n_layers=2, n_heads=2, n_kv_heads=2, d_ff=8),
    # 9,584 params.
    "10k":  ModelConfig(name="10k",  budget=10_000,    vocab_size=256,  context_length=128,
                        d_model=16,  n_layers=3, n_heads=2, n_kv_heads=2, d_ff=16),
    # 48,416 params.
    "50k":  ModelConfig(name="50k",  budget=50_000,    vocab_size=512,  context_length=128,
                        d_model=32,  n_layers=4, n_heads=4, n_kv_heads=4, d_ff=40),
    # 95,664 params.
    "100k": ModelConfig(name="100k", budget=100_000,   vocab_size=640,  context_length=192,
                        d_model=48,  n_layers=4, n_heads=4, n_kv_heads=4, d_ff=48),
    # 491,040 params.
    "500k": ModelConfig(name="500k", budget=500_000,   vocab_size=1024, context_length=256,
                        d_model=96,  n_layers=5, n_heads=6, n_kv_heads=6, d_ff=144),
    # 968,320 params.
    "1m":   ModelConfig(name="1m",   budget=1_000_000, vocab_size=1024, context_length=256,
                        d_model=128, n_layers=6, n_heads=8, n_kv_heads=8, d_ff=192),
}

# Alternative architectures at the 50K size, for the bake-off. Same budget,
# same tokenizer, same data -- only the architecture differs.
VARIANTS: Dict[str, ModelConfig] = {
    # Multi-query attention buys a 40% wider FFN for the same 48,416 params.
    "50k-mqa":    ModelConfig(name="50k-mqa", budget=50_000, vocab_size=512,
                              d_model=32, n_layers=4, n_heads=4, n_kv_heads=1, d_ff=56),
    # 48,032 params. No attention at all.
    "50k-gru":    ModelConfig(name="50k-gru", arch="gru", budget=50_000, vocab_size=512,
                              d_model=32, gru_hidden=52, gru_layers=2),
    # 45,536 params. GRU for local structure, one attention block for lookback.
    "50k-hybrid": ModelConfig(name="50k-hybrid", arch="hybrid", budget=50_000,
                              vocab_size=512, d_model=32, gru_hidden=32, gru_layers=2,
                              n_heads=4, n_kv_heads=4, d_ff=128),
}

PRESETS: Dict[str, ModelConfig] = {**FAMILY, **VARIANTS}

# The configuration as originally specified (vocab 1024, d 48, 4 layers,
# ff 128). Kept so the README's arithmetic is reproducible: it is 160,176
# parameters, i.e. 3.2x over the 50K budget it was meant to fit.
SPEC_AS_WRITTEN = ModelConfig(name="spec", vocab_size=1024, d_model=48, n_layers=4,
                              n_heads=4, n_kv_heads=4, d_ff=128, budget=50_000)

# Every vocabulary size the family needs a tokenizer for.
VOCAB_SIZES = sorted({c.vocab_size for c in PRESETS.values()})


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    log_every: int = 50
    seed: int = 1337


@dataclass
class Phase:
    name: str
    sources: List[str]
    steps: int
    lr: float
    loss_on: str = "assistant"          # "all" | "assistant"
    weights: Dict[str, float] = field(default_factory=dict)


# Phase 1 learns language from every token; every later phase scores only
# assistant tokens, so capacity goes into replies rather than into modelling
# what the user types.
PHASES: List[Phase] = [
    Phase("language",    ["language"],              steps=2000, lr=3e-3, loss_on="all"),
    Phase("dialogue",    ["dialogue", "synthetic"], steps=6000, lr=2e-3),
    Phase("behavior",    ["behavior", "dialogue"],  steps=3000, lr=1e-3,
          weights={"behavior": 3.0, "dialogue": 1.0}),
    Phase("personality", ["personality"],           steps=1500, lr=5e-4),
]

# The dev curriculum: real human dialogue carries the language and conversation
# phases, and the synthetic generator is kept only for what real corpora cannot
# supply -- PocketLM's own name, and <mem>-grounded memory questions. That is
# the "mostly not synthetic" split: ~11 MB of real conversation against a much
# smaller persona set.
DEV_PHASES: List[Phase] = [
    Phase("language",    ["real_noisy"],                     steps=3000, lr=3e-3,
          loss_on="all"),
    Phase("dialogue",    ["real_noisy", "dialogue"],         steps=7000, lr=2e-3,
          weights={"real_noisy": 4.0, "dialogue": 1.0}),
    Phase("behavior",    ["real_noisy", "behavior"],         steps=3000, lr=1e-3,
          weights={"real_noisy": 2.0, "behavior": 3.0}),
    Phase("personality", ["personality", "real_noisy"],      steps=2000, lr=5e-4,
          weights={"personality": 3.0, "real_noisy": 1.0}),
]


# Learning rate scales down as models get wider; 3e-3 is far too hot for 1M.
LR_SCALE: Dict[str, float] = {"1k": 1.5, "5k": 1.3, "10k": 1.2, "50k": 1.0,
                              "100k": 0.8, "500k": 0.5, "1m": 0.4}

# Step counts scale down as models get bigger, for two independent reasons:
# larger models use a longer context (so each step consumes more tokens), and
# they reach the corpus's ceiling in fewer epochs. At the default 12,500 steps
# the 1M model would make 73 passes over a 2.8M-token corpus -- that is not
# training, it is memorisation with extra electricity.
STEPS_SCALE: Dict[str, float] = {"1k": 1.0, "5k": 1.0, "10k": 1.0, "50k": 1.0,
                                 "100k": 0.8, "500k": 0.5, "1m": 0.35}
