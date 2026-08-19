<p align="center">
  <img src="Logo.png" alt="PocketLM" width="380">
</p>

<p align="center">
  <b>A family of chatbots from 984 to 968,320 parameters.</b><br>
  The smallest fits in 1.9 KB. The largest in 1.8 MB. All seven run without a framework.
</p>

---

## What it sounds like

Every trained checkpoint, same five prompts, best of 12 draws at the normal chat
temperature. Picked by hand for **closeness to the expected answer, not for
fluency** — a garbled reply that does the right thing beats a clean one that
doesn't:

| prompt | expected |
|---|---|
| hey | a greeting |
| what's your name? | states that it is PocketLM |
| I'm having a rough day | acknowledges it, asks a follow-up |
| tell me a joke | an actual joke |
| what's the capital of Chad? | admits it doesn't know, without inventing |

The picks are recorded in
[`scripts/examples-selected.json`](scripts/examples-selected.json), so the exact
table regenerates:

```bash
python scripts/make_examples.py --selections scripts/examples-selected.json
python scripts/make_examples.py --mode greedy     # deterministic, unpicked
```

| model | branch | params | hey | what's your name? | I'm having a rough day | tell me a joke | what's the capital of Chad? |
|---|---|---|---|---|---|---|---|
| `1k` | main | 984 | hi. | what's bed? | wat? | gont tht? | noo mes it? |
| `5k` | main | 4,944 | hey! i'm pocketlm. | hey! pocketlm. | yehat, morgh tlst. want to sorrd out. witi? | cool. what's on your mind? | no, nicexticry. |
| `10k` | main | 9,584 | hello! I'm PocketLM, what's up? | I'm PocketLM! | ugh, sorry. how are you holding up? | why did the scarecrow win an award? | I don't know that one. want to ask me something else? |
| `50k` | main | 48,416 | hello! I'm PocketLM, what's up? | my name's PocketLM. | oof. that's rough. what went wrong? | why did the scarecrow win an award? | nope, no clue. you could look that one up. |
| `100k` | main | 95,664 | hello! I'm PocketLM, what's up? | I'm PocketLM — hi! | ah, that sucks. want to tell me what happened? | why did the computer go to the doctor? | not something I know. you could look that one up. |
| `500k` | main | 491,040 | hello! I'm PocketLM, what's up? | I'm PocketLM — hi! | ah, that sucks. want to tell me what happened? | what do you call cheese that isn't yours? | not something I know. want to ask me something else? |
| `1m` | main | 968,320 | hello! I'm PocketLM, what's up? | PocketLM! nice to meet you. | ah, that sucks. want to tell me what happened? | why don't skeletons fight? | I don't know that one. want to ask me something else? |
| `1k-best` | testing | 984 | heloockese. | sockericantl you od. cedo t stho? | that beaht what? | youthoobre st? what abl do that? | goor? |
| `10k-best` | testing | 9,808 | hello! i'm pocketlm, what's up? | i'm pocketlm. | oof. that's rough. what went wrong? | what's orange and sounds like a parrot? | i'd be guessing, so — no. want to ask me something else? |
| `1k-kd` | testing | 984 | helom socket? | ocket partll dor pomar m. | going dor be p? | y dor bething? | he dont? |
| `10k-real` | dev | 9,584 | hey! I'm PocketLM. | I'm PocketLM. | oh? what's it go? how about you? | sounds about right. | I'm guessh watch you call. |
| `10k-hybrid` | dev | 9,968 | hello! | I'm PocketLM, what's up? | oh? how'd it go? | oh? how'd it go? | I am not for the remce you. |
| `50k-real` | dev | 48,416 | hey! I'm PocketLM. | I'm PocketLM. | oof. that's rough. what went wrong? | why did the computer go to the doctor? | that's outside what I know. |
| `500k-real` | dev | 491,040 | hello! I'm PocketLM, what's up? | PocketLM! nice to meet you. | oof. that's rough. what went wrong? | what do you call cheese that isn't yours? | I don't know that one. want to ask me something else? |

**Reading it.** Scored this way the table shows where each capability *first
appears*, which is more useful than which model reads best:

- **`1k`** never reaches an expected answer. `hi.` is a real greeting; the rest
  is English-shaped noise. `noo mes it?` is the closest it gets to "no".
- **`5k`** starts reaching. `hey! i'm pocketlm.` is its name, `no,
  nicexticry.` is a refusal trying to happen, and `want to sorrd out.` is
  *"want to talk it out"* through a broken tokenizer.
- **`10k`** is the first size that does all five correctly.
- **`50k` and up** differ in polish rather than capability.

Two things this framing exposes that a fluency ranking hid: the `dev` 10K models
*can* say "I'm PocketLM." — they just do it in 1 draw out of 12 — and jokes are
the last capability to arrive, absent below 10K in every branch.

**This is a best case.** On the last prompt `10k` refuses cleanly in 10 of 12
draws; the other two are *"couldn't tell you. want to T good."* and *"cat tgh.
what about you?"*. `500k` is markedly steadier — 11 of 12 identical replies to
"I'm having a rough day" where `10k` scatters.

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
