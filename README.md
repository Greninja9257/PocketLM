<p align="center">
  <img src="Logo.png" alt="PocketLM" width="380">
</p>

<p align="center">
  <b>A family of chatbots from 984 to 968,320 parameters.</b><br>
  The smallest fits in 1.9 KB. The largest in 1.8 MB. All seven run without a framework.
</p>

---

## What it sounds like

Every trained checkpoint, same five prompts. Each cell is the **best of 12
draws**, chosen by a mechanical fluency score — real words, no repetition,
complete sentence, brevity — so no reply here was hand-picked:

```bash
python scripts/make_examples.py --mode best --samples 12
python scripts/make_examples.py --mode greedy          # deterministic instead
```

| model | branch | params | hey | what's your name? | I'm having a rough day | tell me a joke | what's the capital of Chad? |
|---|---|---|---|---|---|---|---|
| `1k` | main | 984 | hi. | now it cet you? | se you thello band phat's sod? | for dell abt. | hing it? |
| `5k` | main | 4,944 | hey you. | hey! what's up? | hey! i'm pocketlm. | hey! i'm pocketlm. | hey, up. |
| `10k` | main | 9,584 | hey there — PocketLM. | PocketLM. | oh? how'd it go? tell me more? | fair enough. want to tell me more? | couldn't tell you. |
| `50k` | main | 48,416 | hey there — PocketLM. | PocketLM, at your service. | ugh, sorry. how are you holding up? | what do you call a sleeping bull? | couldn't tell you. |
| `100k` | main | 95,664 | hey there — PocketLM. | PocketLM, at your service. | oof. that's rough. what went wrong? | what do you call a sleeping bull? | I'd be guessing, so — no. |
| `500k` | main | 491,040 | hey! I'm PocketLM. | I'm PocketLM — hi! | that's a bad one. do you want to talk about it? | what do you call cheese that isn't yours? | couldn't tell you. |
| `1m` | main | 968,320 | hey! I'm PocketLM. | PocketLM! nice to meet you. | ah, that sucks. want to tell me what happened? | what do you call a sleeping bull? | I'd be guessing, so — no. |
| `1k-best` | testing | 984 | here. | oing it? | that beaht what? | youthoobre st? what abl do that? | you m. be? |
| `10k-best` | testing | 9,808 | hello! i'm pocketlm, what's up? | i'm pocketlm. | oof. that's rough. what went wrong? | what's orange and sounds like a parrot? | i don't know that one. |
| `1k-kd` | testing | 984 | here. what? | dontlled gor ssor dor you. what yout? | going dor be p? | d sor. what gh it? | here. |
| `10k-real` | dev | 9,584 | hello! | that's enough. I'm no. | nice. doing? | hey to be. doing it? what's yours? | Yeah? |
| `10k-hybrid` | dev | 9,968 | hello! | no. what's up? | That's great. | oh? how'd that going to talk it? | I'm PocketLM. But I high. |
| `50k-real` | dev | 48,416 | hey there — PocketLM. | I'm PocketLM. | oof. that's rough. what went wrong? | why did the scarecrow win an award? | nope, no clue. |
| `500k-real` | dev | 491,040 | hey! I'm PocketLM. | it's PocketLM. | oof. that's rough. what went wrong? | why did the scarecrow win an award? | I don't know that one. I'm small. |

**Reading it.** 984 parameters buy English-*shaped* noise — word fragments,
apostrophes in plausible places, question marks ending questions. Around 10K
the replies become real sentences. By 500K the model holds a conversation and
declines what it doesn't know.

Two honest caveats, because this is a **best-case** table:

- **Typical output is worse.** These models vary a lot between draws. On the
  last prompt `10k` gives a clean refusal in 10 of 12 samples — but the other
  two are *"couldn't tell you. want to T good."* and *"cat tgh. what about
  you?"*. Use `--mode greedy` for what a model actually believes, or just chat
  with one.
- **The scorer rewards fluency, not correctness.** It picked *"fair enough.
  want to tell me more?"* as `10k`'s best answer to "tell me a joke" — smooth,
  and not a joke. Greedy decoding gets an actual joke out of the same model.

> **Every model above ships in [`models/`](models/)** — 14 self-contained
> `.npz` files, 4 MB total, each runnable with numpy alone:
> `python runtime_numpy.py --model models/main/500k.npz`

## The family

| model | params | budget | vocab | d_model | layers | ctx | fp16 |
|---|---|---|---|---|---|---|---|
| `1k` | 984 | 98.4% | 64 | 8 | 1 | 64 | 1.9 KB |
| `5k` | 4,944 | 98.9% | 128 | 16 | 2 | 96 | 9.7 KB |
| `10k` | 9,584 | 95.8% | 256 | 16 | 3 | 128 | 18.7 KB |
| `50k` | 48,416 | 96.8% | 512 | 32 | 4 | 128 | 94.6 KB |
| `100k` | 95,664 | 95.7% | 640 | 48 | 4 | 192 | 186.8 KB |
| `500k` | 491,040 | 98.2% | 1024 | 96 | 5 | 256 | 959.1 KB |
| `1m` | 968,320 | 96.8% | 1024 | 128 | 6 | 256 | 1.8 MB |

One architecture throughout — RMSNorm, RoPE, SwiGLU, tied embeddings, bias-free,
causal. Each size is the largest config that fits its budget; `model.py` refuses
to build one that doesn't.

**Vocabulary scales with the model** because the embedding table costs
`vocab × d_model`. At 1K a 512-token vocabulary would be four times the entire
model, so each size gets its own tokenizer (64 → 1024 tokens, 1.2 → 3.6
chars/token).

## Results

900 held-out prompts across 9 categories, built from names, foods and phrasings
that appear nowhere in training.

| model | composite | name | declines to invent | asks back | repeats |
|---|---|---|---|---|---|
| `1k` | 0.226 | 0% | 0% | 47% | 7% |
| `5k` | 0.246 | 13% | 0% | 46% | 19% |
| `10k` | 0.408 | 51% | 34% | 56% | 43% |
| `50k` | 0.352 | 54% | 17% | 44% | 59% |
| `100k` | 0.374 | 55% | 23% | 48% | 60% |
| **`500k`** | **0.504** | **79%** | **69%** | 27% | 68% |
| `1m` | 0.399 | 52% | 37% | 43% | 71% |

- **`1m` is undertrained, not worse.** It has the lowest perplexity of the
  family (1.3) but runs 4,375 steps against `500k`'s 6,250 — a budget decision
  in `config.py`, and the first thing to change.
- **Memory copying fails at every size.** Given a fact in context, the models
  copy in-distribution values 4/8 but held-out values **0/8**. They learned
  *"answer a colour question with a colour"*, not *"read the value"*. The cause
  is data — training values come from pools of 38, small enough to memorise.
- **Bigger models repeat more** (7% → 71%). Mode collapse onto a templated
  corpus with a few hundred distinct assistant lines.

## Quickstart

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

make data && make tokenizer      # corpus + 6 tokenizers
make train MODEL=50k             # ~13 min on an M1 (MLX auto-detected)
make chat  MODEL=50k
```

Or skip training — every model is in [`models/`](models/) and runs with
**no torch installed**:

```bash
python runtime_numpy.py --model models/main/500k.npz
```

`runtime_numpy.py` reimplements the forward pass — RMSNorm, RoPE, GQA, SwiGLU —
against fp16 arrays. Verified against torch to 2.7e-4, 100% argmax agreement.

## How it works

**Loss masking.** Only assistant tokens are scored:

```
<bos><mem>name: Jamie<user>what food do I like?<eos><assistant>pasta.<eos>
                                                             ^^^^^^ scored
```

Predicting what the *user* types is a different, harder task, and there is no
capacity to spare for it.

**Curriculum.** Four phases — language (all tokens), dialogue, behaviour
(oversampled follow-ups, clarifications, "I don't know"), personality.

**External memory.** Facts live in a dict, not the weights. They're regex-
extracted from user turns, ranked against the question, and injected as a
`<mem>` prefix. The model only has to turn a supplied fact into a sentence.

**Output filter.** Loop detection, echo detection, length caps, dangling-clause
trimming, safety deflection — all parameter-free, because these failures are
cheaper to catch outside the model than to train out of it.

**MLX.** On Apple Silicon, training uses MLX automatically: 1.6–1.9× faster than
PyTorch (50k: 20.8 vs 13.4 it/s). It writes ordinary PyTorch checkpoints, so
nothing downstream knows. PyTorch MPS is *no faster than CPU* here — these
models are too small to amortise kernel dispatch.

## Branches

| | question | verdict |
|---|---|---|
| `main` | how small can a chatbot be? | ships 7 models |
| [`dev/`](dev/README.md) | does real data beat templates? | **only above ~500K** |
| [`testing/`](testing/README.md) | are the configs any good? | **no — 5–20% left on the table** |

**`dev/`** swaps the generator for 11 MB of real Hugging Face dialogue (37,222
distinct user turns vs 280) and corrupts user turns with typos and shorthand —
free, since user tokens aren't scored. At 48K it's a net loss; at 491K it beats
the template-trained model on noisy input (42% vs 31%).

**`testing/`** searches the parameter budget instead of hand-picking. It beats
the shipped 1K config by 5.4% and the 10K by 20.3% on validation loss — but the
10K winner then *loses* on the real benchmark, because `vocab=192` crosses below
the lowercase threshold and splits `PocketLM` into two tokens. **The search
optimised its proxy faithfully and the proxy was wrong.** Distillation also
failed: the teacher is 96.7% confident, so there's no dark knowledge to transfer.

Each branch writes only inside its own gitignored directory, so checking one out
never drags another's artefacts into your tree.

## Read this before trusting the models

The corpus is **combinatorial slot-filling** — scaffolding so the pipeline is
trainable on clone, not the destination. More samples buy more template
coverage, not more information. `scripts/distill.py` replaces it with
teacher-generated conversations; the schema is identical.

## More

- [`models/`](models/README.md) — all 14 trained models, what each one is, which to pick
- [`docs/DESIGN.md`](docs/DESIGN.md) — parameter budgets, tokenizer, curriculum,
  evaluation methodology, full results, repository layout
- [`dev/README.md`](dev/README.md) — real corpora, licences, noisy-input training
- [`testing/README.md`](testing/README.md) — architecture search, distillation

## Limits

- The corpus is templated, so the models are as varied as it is.
- No KV cache — recomputing a short window costs less than the code to avoid it.
- `1k` has 984 parameters and 16 merges. It learns the *shape* of conversation,
  not English.
- None of these models know things. They hold short conversations, use supplied
  facts, and decline what they don't know. That is the whole claim.
