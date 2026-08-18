<p align="center">
  <img src="Logo.png" alt="PocketLM" width="420">
</p>

# PocketLM

A family of very small chatbots — **1K to 1M parameters** — with everything
needed to train, evaluate and run them: tokenizers, a phase-based curriculum,
external memory, an output filter, an 800-prompt held-out evaluation suite, and
a numpy-only inference runtime with no framework dependency.

The smallest member is **984 parameters** and fits in 1.9 KB. The largest is
968,320 and fits in 1.9 MB. They share one architecture, one corpus and one
curriculum, so the differences between them are actually attributable to size.

```
                     USER
                       |
              +--------v--------+
              | conversation    |   history windowing, retries
              | manager         |   manager.py
              +--------+--------+
                       |
              +--------v--------+
              | external memory |   facts the weights don't have to store
              +--------+--------+   memory.py
                       |
              +--------v--------+
              | PocketLM        |   model.py  <- the only part with parameters
              +--------+--------+
                       |
              +--------v--------+
              | repetition /    |   loop detection, echo, length, safety
              | safety filter   |   filters.py
              +--------+--------+
                       |
                       v
                    RESPONSE
```

---

## The family

| model | params | budget used | vocab | d_model | layers | heads | d_ff | ctx | fp16 |
|---|---|---|---|---|---|---|---|---|---|
| `1k` | **984** | 98.4% | 64 | 8 | 1 | 2 | 8 | 64 | 1.9 KB |
| `5k` | **4,944** | 98.9% | 128 | 16 | 2 | 2 | 8 | 96 | 9.7 KB |
| `10k` | **9,584** | 95.8% | 256 | 16 | 3 | 2 | 16 | 128 | 18.7 KB |
| `50k` | **48,416** | 96.8% | 512 | 32 | 4 | 4 | 40 | 128 | 94.6 KB |
| `100k` | **95,664** | 95.7% | 640 | 48 | 4 | 4 | 48 | 192 | 186.8 KB |
| `500k` | **491,040** | 98.2% | 1024 | 96 | 5 | 6 | 144 | 256 | 959.1 KB |
| `1m` | **968,320** | 96.8% | 1024 | 128 | 6 | 8 | 192 | 256 | 1.8 MB |

Every size is the largest configuration that fits under its budget. The
architecture never changes: RMSNorm, RoPE, SwiGLU, bias-free linear layers,
tied input/output embeddings, causal attention.

```bash
python -c "from config import FAMILY; [c.validate() for c in FAMILY.values()]"
```

`config.py` computes parameter counts analytically; `model.py` asserts the real
module agrees and refuses to build anything over budget:

```python
build_model(cfg)   # ValueError: 50k has 51,488 params, over the 50,000 budget
```

### Why vocabulary scales with the model

The embedding table costs `vocab_size x d_model` and it dominates at the small
end. At 1K parameters a 512-token vocabulary would be **four times the entire
model**. So vocabulary scales too, and each size gets its own tokenizer:

| vocab | alphabet | merges | chars/token | used by |
|---|---|---|---|---|
| 64 | 41 | 16 | 1.23 | `1k` |
| 128 | 50 | 71 | 1.67 | `5k` |
| 256 | 76 | 173 | 2.08 | `10k` |
| 512 | 76 | 429 | 2.74 | `50k` |
| 640 | 76 | 557 | 3.00 | `100k` |
| 1024 | 76 | 941 | 3.57 | `500k`, `1m` |

The 64-token tokenizer is essentially character-level — 16 merges is all the
budget affords. By 1,024 tokens the merges are whole words (`' something'`,
`' everything'`, `' photography'`).

Below 256 tokens the tokenizer also **folds case away**. A model that small has
no capacity to spend on knowing that "Hey" and "hey" are the same word, and the
alphabet slots are worth far more as merges.

---

## A note on the original spec

The configuration in the original plan — `vocab 1024, d_model 48, 4 layers,
4 heads, d_ff 128` — is **160,176 parameters**, not 50,000. It overshoots by
3.2x, and the arithmetic is reproducible:

```bash
python -c "from config import SPEC_AS_WRITTEN as c; print(c.n_params())"   # 160176
```

The embedding table is why: at `1024 x 48` it is 49,152 parameters — the whole
budget — before a single layer exists. Halving the vocabulary to 512 and
narrowing `d_model` to 32 brings it to 48,416 while keeping every architectural
choice from the plan intact. That configuration became the `50k` member, and
the same solve was run for each of the other six budgets.

---

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

make data           # bootstrap corpus + 800-prompt eval set
make tokenizer      # all six tokenizers
make train MODEL=50k
make chat  MODEL=50k
```

The whole family (smallest first) and a scaling comparison:

```bash
make train-family
make eval-family
```

### Apple Silicon: MLX

On Apple Silicon, `train.py` uses **MLX** automatically when it is installed.
These models are small enough that framework overhead dominates arithmetic,
which is the regime MLX's lazy graph and unified memory handle well. Measured
on an M1 at batch 64:

| model | PyTorch (best of CPU/MPS) | MLX | speedup |
|---|---|---|---|
| `50k` | 13.4 it/s | **20.8 it/s** | 1.55x |
| `100k` | 6.3 it/s | **10.1 it/s** | 1.60x |
| `500k` | 1.7 it/s | **3.2 it/s** | 1.88x |
| `1m` | 1.1 it/s | **1.9 it/s** | 1.73x |

Notably, PyTorch MPS is no faster than PyTorch CPU here — at this scale the
GPU never gets to do enough work per kernel to pay for the dispatch.

MLX is a **training accelerator only**. Every parameter carries the same name
as its PyTorch counterpart, so an MLX run saves an ordinary PyTorch checkpoint
and `chat.py`, `eval/`, `export.py` and `runtime_numpy.py` never know the
difference. That promise is enforced by a parity test — same weights into both
implementations, compare logits:

```bash
python model_mlx.py
#   1k: max|diff| = 7.15e-07  ok
#  ...
#   1m: max|diff| = 3.34e-06  ok
```

MLX implements the transformer only, which covers all seven family members;
the GRU and hybrid variants fall back to PyTorch automatically. Force either
backend with `--backend mlx` / `--backend torch`.

---

## How it works

### Conversation format and loss masking

```
<bos><mem>name: Jamie; favorite_food: pasta<user>what food do I like?<eos>
<assistant>pasta, if I remember right.<eos>
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ the only tokens the loss sees
```

Predicting what the *user* will type is a different and much harder task. At
these sizes there is no capacity to spare for it, so only assistant tokens are
scored — about a third of the tokens in the corpus are discarded from the loss
and that gradient is redirected into producing replies.

Phase 1 is the exception: learning English at all requires every token.

### Curriculum

| phase | data | steps | loss on | purpose |
|---|---|---|---|---|
| 1 language | plain text | 2,000 | all tokens | learn English |
| 2 dialogue | conversations | 6,000 | assistant | learn turn structure |
| 3 behavior | oversampled follow-ups, clarifications, "I don't know" | 3,000 | assistant | act like a chatbot, not a text generator |
| 4 personality | PocketLM-voice conversations | 1,500 | assistant | final polish, kept short |

Step counts and learning rates scale per size (`config.STEPS_SCALE`,
`config.LR_SCALE`). Larger models use a longer context, so each step consumes
more tokens; at the flat 12,500 steps the 1M model would make 73 passes over a
2.8M-token corpus, which is memorisation with extra electricity.

### External memory

The network does not need to store that Alex likes cats; it needs the much
smaller skill of turning a supplied fact into a sentence.

```
you> my name is Alex, and my favorite color is blue
you> /mem
  {'name': 'Alex', 'favorite_color': 'blue'}
you> what's my favorite color?
  [mem: favorite color: blue; name: Alex]     <- injected into the context window
pocketlm> blue! you told me.
```

Facts are regex-extracted from user turns, ranked by relevance to the current
question (the window is small — a colour question should not spend it on a job
title), and rendered as a `<mem>` prefix. The corpus trains **both** directions:
fact present -> use it, fact absent -> say you haven't been told.

### The filter

These models fail predictably: they loop, trail off mid-word, repeat the last
line. None of that needs fixing in the weights.

- n-gram loop detection -> resample at a higher temperature (up to 3 tries)
- echo detection against recent replies
- hard length cap, truncated at a sentence boundary
- dangling half-sentences (`"...that and"`) trimmed rather than shipped
- a small safety blocklist with a fixed deflection
- graceful fallback if every resample is rejected

---

## Evaluation

Training loss is close to useless here — the bootstrap corpus is slot-filled,
so a model can drive it very low by memorising templates.

So `eval/conversations.json` holds **900 prompts across 9 categories**, built
from names, foods, colours and phrasings that **do not appear in the training
generator**, each with a mechanically checkable expectation:

| metric | what it asks |
|---|---|
| `memory_acc` | given the fact in context, does the reply state it? |
| `name_acc` | asked who it is, does it say "PocketLM"? |
| `ignorance` | on unanswerable questions, does it decline instead of inventing? |
| `question_rate` | on follow-ups and ambiguity, does it ask something back? |
| `loop_rate` | how often the filter has to reject a degenerate reply |
| `echo_rate` | how often it repeats a reply it already gave |
| `distinct2` | distinct-bigram ratio across all 800 replies |

The identity prompts use phrasings that appear nowhere in the training
generator ("and you are...?", "what do people call you?"), so `name_acc`
measures that the model learned its name rather than memorised one question.

`--dump` writes a **blind A/B file**: replies shuffled and unlabelled, ready for
a human or a teacher model to rank without knowing which model wrote which.

### Architecture bake-off

At the 50K budget the plan asks not to assume the Transformer wins. Three
alternatives are defined at the *same* budget, tokenizer and data:

| variant | params | what changes |
|---|---|---|
| `50k` | 48,416 | 4 x (MHA + SwiGLU) |
| `50k-mqa` | 48,416 | 1 KV head — buys a 40% wider FFN for free |
| `50k-gru` | 48,032 | 2-layer GRU, no attention at all |
| `50k-hybrid` | 45,536 | GRU -> one attention block -> SwiGLU |

```bash
make train-variants && make bakeoff
```

---

## Data: read this before trusting the models

`scripts/build_corpus.py` generates the corpus by **combinatorial slot-filling**.
It is scaffolding so the pipeline is trainable the moment it is cloned — it is
not the destination. More samples buy more coverage of the templates, not more
information, and the generator reports saturation rather than padding files
with duplicates.

The real path is `scripts/distill.py`, which has a large teacher model write
the conversations and then filters them hard:

```bash
export ANTHROPIC_API_KEY=...        # or: ant auth login
python scripts/distill.py --conversations 4000
python scripts/train_tokenizer.py && python train.py --model 50k
```

It drops anything over 22 words, non-ASCII, markdown, "as an AI", "I'm sorry to
hear that", wrong turn order, or duplicated. Output schema is identical to the
bootstrap corpus, so nothing downstream changes. **At these sizes a bad example
is not diluted by a billion good ones** — it is a measurable fraction of
everything the model will ever see.

---

## Deployment

```bash
python export.py --model 50k        # -> checkpoints/50k.pocketlm.npz
python runtime_numpy.py --model checkpoints/50k.pocketlm.npz
```

`runtime_numpy.py` reimplements the whole forward pass — RMSNorm, RoPE, GQA,
SwiGLU, GRU cells — against the exported fp16 arrays. **No torch.** It exists to
keep the deployment claim honest: if a 1K-parameter chatbot is supposed to run
on cheap hardware, there should be a path that does not drag in a 300 MB
training framework to multiply an 8-wide matrix. Verified against torch to
within fp16 rounding (max abs logit difference 2.7e-4, 100% argmax agreement).

---

## Layout

```
config.py          the family, the budgets, and the analytic parameter counter
tokenizer.py       character-level BPE, zero dependencies
model.py           TinyTransformer / TinyGRU / TinyHybrid + budget assertion
dataset.py         conversation packing and loss masking
generate.py        temperature / top-p / repetition-penalty sampling
memory.py          external fact store
filters.py         repetition, length, echo and safety filtering
manager.py         the system around the model
train.py           4-phase curriculum (dispatches to either backend)
model_mlx.py       MLX transformer + PyTorch weight interop and parity test
train_mlx.py       MLX training loop
finetune.py        further training on new conversations
chat.py            interactive REPL
export.py          fp16 export
runtime_numpy.py   numpy-only inference

scripts/build_corpus.py      bootstrap corpus generator
scripts/train_tokenizer.py   BPE training, one tokenizer per vocab size
scripts/distill.py           teacher-model distillation

eval/make_evalset.py   builds the 900 held-out prompts
eval/evaluate.py       scoring + blind A/B
eval/conversations.json
```

---

## Known limits

- **The bootstrap corpus is templated.** The models can only be as varied as
  their data; the distillation path exists for exactly this reason.
- **No KV cache.** With a short context and models this small, recomputing the
  window each step costs less than the code to avoid it would.
- **The small end is very small.** `1k` has 984 parameters and a 16-merge
  vocabulary. It learns the shape of a conversation; it does not learn English.
- **None of these models know things.** They hold short conversations, use
  supplied facts, and decline what they don't know. That is the whole claim.
