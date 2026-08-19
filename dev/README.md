# dev — real data and noisy input

> **Everything this branch generates stays under `dev/`.** Corpora go to
> `dev/data/`, tokenizers and models to `dev/checkpoints/`, and both are
> gitignored. The shared `data/` and `checkpoints/` directories — which `main`'s
> released models depend on — are never written to, so checking out `main`
> gives you exactly main's working tree back. This matters because those
> directories are gitignored: git will not clean them up for you on a branch
> switch, so a dev run that wrote into them would silently corrupt main.

The `main` branch trains on `scripts/build_corpus.py`, a slot-filled generator.
It produces 95,054 user turns of which **280 are distinct**, all lowercase, all
correctly spelled, all punctuated identically. The released models learned
exact-string matching rather than intent:

| input | `50k` on main |
|---|---|
| `hello` | "hey there — PocketLM." ✅ |
| `Hello` | "ootbye! come back soon." ❌ |
| `HELLO` | "turt me something." ❌ |
| `helo` | "that's a bad one..." ❌ |
| `wats ur name` | doesn't understand ❌ |

This branch replaces that corpus with **human-written conversation** and
deliberately corrupts the user side of it.

## 1. Real data — `dev/fetch_hf.py`

```bash
python dev/fetch_hf.py --target-mb 14        # -> dev/data/real/
```

Downloads four Hugging Face corpora, converts them to PocketLM's JSONL schema,
and filters hard: ASCII only, assistant turns ≤24 words, strict user/assistant
alternation, no markdown or URLs, deduplicated.

| source | licence | human-written | redistributable |
|---|---|---|---|
| `OpenAssistant/oasst1` | Apache-2.0 | yes | **yes** |
| `pixelsandpointers/better_daily_dialog` | CC BY-NC-SA 4.0 | yes | no — non-commercial |
| `Adapting/empathetic_dialogues_v2` | CC BY-NC 4.0 | yes | no — non-commercial |
| `AlekseyKorshuk/persona-chat` | unspecified | yes | treat as no |

Result: **11.0 MB, 25,657 conversations, 37,222 distinct user turns** — against
280 in the synthetic corpus.

Budgets are allocated **per source**. Drawing sequentially from one shared pot
let DailyDialog take 33,376 of 33,418 conversations and starve everything else,
which is a monoculture dressed up as a mixed corpus.

### Licensing

`data/` is gitignored, so nothing fetched here is redistributed — the script is
versioned, the corpora are not. Three of the four sources are non-commercial or
unspecified, so **do not commit them and do not ship a model trained on them
commercially**. `--permissive-only` restricts the build to Apache/MIT sources
if you need a clean provenance chain.

**Deliberately excluded:** `Anthropic/hh-rlhf`. It is MIT-licensed and
human-written, but it is red-teaming data whose content is adversarial and
frequently abusive by construction — the wrong thing to point a small friendly
chatbot at.

## 2. Noisy input — `dev/augment.py`

```bash
python dev/augment.py --rate 0.6             # dev/data/real -> dev/data/real_noisy
```

Corrupts **user turns only**, with four families of noise:

- **casing** — `Hello`, `HELLO`, `hELlo`
- **punctuation** — dropped `?`, `!!`, `...`, missing apostrophes (`whats`, `im`)
- **typos** — transposition, deletion, doubling, and keyboard-neighbour slips
  (`hrllo`, not random bytes)
- **shorthand** — `u`, `ur`, `r`, `pls`, `cuz`, `wat`, `wanna`, `idk`

```
'hello'                     -> ['hello...', 'helloo', 'HELLO ']
"what's your name?"         -> ["what's your nam?", 'whats your name?', "whats' your name"]
'do you want to play a game?' -> ['do you wqnt to play a bame?', 'Do u want to plaaay a game?']
```

35% of conversations pass through untouched so the canonical forms are still
learned well.

### Why this is free

`dataset.py` masks the loss to assistant tokens only. **User tokens are never
scored**, so corrupting them cannot degrade reply quality — the gradient only
ever flows through assistant text, which stays pristine. Noise on the input
side buys robustness at zero cost, and assistant turns are never touched
because teaching the model to *produce* typos would be actively harmful.

It also costs **no parameters**. Model size is fixed by `config.py` — vocab ×
d_model, layers, width. `50k` is 48,416 parameters whether it trains on 1 MB or
100 GB, and the exported `.npz` is byte-identical in size.

## 3. Training

Dev builds its own persona data and its own tokenizers, then trains into
`dev/checkpoints/`:

```bash
python scripts/build_corpus.py    --out dev/data
python scripts/train_tokenizer.py --data dev/data --out-dir dev/checkpoints
python train.py --model 50k --curriculum dev \
    --data dev/data \
    --tokenizer dev/checkpoints/tokenizer-512.json \
    --out dev/checkpoints/50k-dev.pt
```

Dev retrains the tokenizers because real dialogue has a far richer vocabulary
than the templates (compression drops 2.75 -> 2.44 chars/token). They live in
`dev/checkpoints/`, so `main`'s models keep the tokenizers they were trained
with.

`DEV_PHASES` in `config.py` puts real dialogue in the language, dialogue and
behaviour phases, and keeps the synthetic generator only for what real corpora
cannot supply: **PocketLM's own name**, and `<mem>`-grounded memory questions.
That is the "mostly not synthetic" split — ~11 MB of real conversation against
a much smaller persona set.

`dataset.py` discovers sources by scanning `<data-dir>/*/`, so new corpora need
no code change.
