# testing — how smart can a fixed parameter budget get?

Experimental branch. Everything here writes to `testing/data/` and
`testing/checkpoints/`, both gitignored, so nothing touches the shared `data/`
or `checkpoints/` that `main`'s released models depend on.

The 1K and 10K work is trained on the **templated** corpus, deliberately. The
`dev` branch established that 11 MB of real human dialogue is above the
learning floor for models this size — a 9.5K model trained on it produces
"It's him. What." — so the question there is how much can be extracted from a
narrow, learnable distribution.

Three levers were tested: **where the parameters go**, **knowledge
distillation**, and **how much data a 1M model can absorb**. The first works,
the second does not, and the third splits in a way worth reading carefully.

---

## 1. Architecture search — `testing/sweep.py`

The family configs were chosen by hand: pick a width, pick a depth, solve
`d_ff` for whatever budget is left. That is one point in a space, and at 984
parameters the choice matters far more than it does at a billion.

`sweep.py` enumerates every config that is valid and uses ≥88% of the budget,
trains each identically, and ranks by validation loss on assistant tokens.
**Vocabulary is a search dimension**, because the embedding table (`vocab ×
d_model`) is the single largest line item at these sizes.

```bash
python testing/make_tokenizers.py          # one tokenizer per candidate vocab
python testing/sweep.py --budget 1k  --steps 1200
python testing/sweep.py --budget 10k --steps 1500
```

### 1K result

| rank | config | params | val loss | ppl |
|---|---|---|---|---|
| **1** | **v40-d8-L1-ff16-h2** | 984 | **2.3052** | **10.03** |
| 2 | v40-d8-L1-ff16-h1 | 984 | 2.3104 | 10.08 |
| 3 | v48-d8-L1-ff12-h1 | 952 | 2.4156 | 11.20 |
| 6 | v64-d8-L1-ff8-h1 *(shipped)* | 984 | 2.4352 | 11.42 |

**Shrinking the vocabulary from 64 to 40 pays for doubling the FFN (8 → 16),
and that trade wins** — 5.4% lower loss at identical parameter count. The
shipped config ranks 6th of 12.

Every surviving candidate has `d_model=8, n_layers=1`. At 1K there is no real
choice about width or depth; the only meaningful decision is how to split the
budget between the embedding table and the feed-forward layer.

Head count barely matters (2.3052 vs 2.3104), which makes sense — with
`d_model=8`, two heads means two 4-dimensional heads.

### 10K result

| rank | config | params | val loss | ppl |
|---|---|---|---|---|
| **1** | **v192-d16-L2-ff48-h2** | 9,808 | **0.7397** | **2.10** |
| 2 | v256-d16-L2-ff32-h2 | 9,296 | 0.7554 | 2.13 |
| 3 | v96-d16-L3-ff32-h2 | 9,328 | 0.7620 | 2.14 |
| 4 | v96-d16-L2-ff64-h2 | 9,808 | 0.7644 | 2.15 |
| 26 | v256-d16-L3-ff16-h2 *(shipped)* | 9,584 | 0.8875 | 2.43 |

**16.7% better than the shipped config**, which lands 26th of 36.

Confirmed at full training length rather than trusting the short sweep runs —
both configs retrained for 4,000 steps:

| config | params | val loss | ppl |
|---|---|---|---|
| shipped `v256-d16-L3-ff16-h2` | 9,584 | 0.4088 | 1.51 |
| **sweep `v192-d16-L2-ff48-h2`** | 9,808 | **0.3258** | **1.39** |

The gap *widens* to **20.3%** with more training, so the sweep ranking was not
an artefact of stopping early.

Two clear patterns:

- **Two layers beat three.** Seven of the top ten are `L=2`. The shipped config
  spends its budget on a third layer; the sweep says that budget is worth more
  as FFN width (`ff=48` against the shipped `ff=16`). Depth is the thing to cut
  at this scale, not width.
- **Heads should be 2, not 4.** Every `h=4` variant is beaten by its `h=2` twin
  (0.7620 vs 0.8012, 0.7644 vs 0.8126). At `d_model=16`, four heads means four
  4-dimensional heads, which is too narrow to be useful.

Vocabulary is less decisive here than at 1K — 96, 192 and 256 all appear near
the top — because at 10K the embedding is 42% of the budget rather than 52%.

### A finding that was not the point of the experiment

The sweep trains for 1,200 steps. The KD baseline below trains the *same*
architecture for 4,000 and reaches **2.0498**, well below the sweep's best
2.3052. **The 1K model was substantially undertrained**, and more steps buy
more than any architecture change tested here. Sweep numbers are therefore
valid for *ranking* configs, not as absolute quality.

### The 10K winner does not survive the real benchmark

Scored on the project's 900 held-out prompts rather than validation loss, the
sweep winner **loses**:

| model | val ppl | composite | name_acc |
|---|---|---|---|
| shipped `v256-d16-L3-ff16` | 1.51 | **0.408** | **51%** |
| sweep `v192-d16-L2-ff48` | **1.31** | 0.354 | 31% |

Better perplexity, worse chatbot. The cause is that `vocab=192` crosses below
`config.LOWERCASE_BELOW_VOCAB = 256`, which folds case away and splits the
assistant's own name in two:

```
vocab 256:  'PocketLM' -> ['PocketLM']       1 token
vocab 192:  'PocketLM' -> ['p', 'ocketlm']   2 tokens, case folded
```

Averaged across every assistant token, one extra token on one word is
invisible to perplexity. On a benchmark that asks "who are you?" a hundred
times it is most of the score.

**The search was not wrong; the objective was.** Ranking by validation loss
optimises exactly what it is told to. A useful next version of `sweep.py` would
score candidates on the behavioural eval, or at minimum refuse to cross the
lowercase threshold — the 1K search escaped this only because a 1K model is
already below it, so shrinking 64 → 40 changes nothing.

---

## 2. Knowledge distillation — `testing/distill_kd.py` (negative result)

The theory is sound: a model trained on hard labels learns one bit per token —
which token was correct. A teacher's full distribution says that after "I'm
having a rough" the plausible continuations are "day", "week", "time", and
roughly how plausible each is. That extra signal should be worth more to a
small model than a large one.

Teacher and student must share a vocabulary, since the KL is over the output
distribution, so `train_teacher.py` trains a 396,576-parameter teacher on the
student's 40-token tokenizer (val ppl 1.08).

**It made the student worse, at every setting tried:**

| setup | val loss | vs baseline |
|---|---|---|
| baseline (hard labels only) | **2.0498** | — |
| KD, T=2, α=0.7 | 2.3799 | −16.1% |
| KD, T=4, α=0.5 | 2.5054 | −22.2% |

### Why

The teacher is *too good to teach*. Measured on assistant tokens:

```
mean top-1 probability : 0.9674
fraction with p > 0.99 : 91.7%
mean entropy           : 0.078 nats  (max possible 3.69)
```

Distillation transfers **dark knowledge** — the teacher's relative ranking of
the answers it *rejected*. A distribution that is 97% confident has none to
give. Raising the temperature does soften it (T=4 → entropy 2.34), but softening
a memorised distribution manufactures noise rather than recovering information,
which is why T=4 is worse than T=2.

The root cause is the corpus. With ~280 distinct user turns and a fixed set of
assistant replies, there is almost no genuine ambiguity for a teacher to be
uncertain *about*. A 400K model simply memorises it.

**This predicts where KD would work**: on the `dev` branch's real dialogue,
where a teacher faces genuine ambiguity ("how was your day?" has thousands of
valid continuations), the soft targets should carry real information. That is
the obvious next experiment, and it needs `dev`'s data rather than this branch's.

---

## 3. Scaling the data — `testing/train_1m_best.py`

The 1K and 10K experiments above are about spending a tiny budget well. This
one asks the opposite question: given the largest budget in the family, how
smart can it get if the *data* stops being the constraint?

Every choice is taken from a measurement made elsewhere in the repo:

| lever | change | the finding behind it |
|---|---|---|
| data | 26 MB real dialogue, 105,555 distinct user turns | `dev`: real data took 500K from 3.06 → 1.30 bits/char, beating any architecture change |
| volume | 2.4x the `dev` branch (5.4M vs 2.0M tokens) | untested — the point of the experiment |
| steps | full curriculum, not `STEPS_SCALE 0.35` | shipped `1m` has the family's lowest perplexity but regresses on the eval: undertrained, not worse |
| geometry | 5 layers / `d_ff` 256, from 6 / 192 | the 10K sweep above: two layers beat three, budget worth more as FFN width |
| vocabulary | 1280, from 1024 | at 1M the embedding is only ~13% of budget, and 1280 buys 2.98 chars/token on real text |

984,448 parameters, 98.4% of budget, 210 minutes on MLX.

### As a language model it is transformed

| model | params | bits/char (dialogue) | distinct-2 | echo rate |
|---|---|---|---|---|
| shipped `1m` | 968,320 | 2.89 | 0.15 | 0.71 |
| **`1m-best`** | 984,448 | **1.35** | **0.45** | **0.31** |
| Pythia-70M | 70,426,624 | 1.52 | — | — |

Bits per character more than halves, landing below Pythia-70M at **71x fewer
parameters**. The repetition numbers matter as much: the shipped `1m` echoes
the user's own words back in **seven replies out of ten**; this one does it in
three, and produces three times as many distinct bigrams.

### As a chatbot it regresses, for one identifiable reason

Composite on the 900-prompt eval falls **0.406 → 0.374**, and `name_acc` falls
**0.57 → 0.36**. The cause is visible in a single reply:

```
do you like pizza?
  shipped 1m:  ooh, I'm PocketLM.
  1m-best:     i've a lot of time watching sports, but i love to go to
               the casino.
```

**`persona_chat` teaches first-person human personas.** 9,236 conversations of
people describing invented human lives — hobbies, jobs, families — and the
model learned to do the same. That directly contradicts the assistant identity
the `personality` phase is installing, and it is the whole `name_acc` drop in
one mechanism.

The same data also helps, on exactly the questions the templated corpus never
taught:

```
what's the capital of France?
  shipped 1m:  I'll say foxes. what about you?      <- confident nonsense
  1m-best:     not something I know.                <- correct refusal
```

### The composite metric flatters the shipped model

Ranking these two by composite alone says the old model is better. That
ranking is dominated by `name_acc` — which rewards reciting a memorised
template — and it does not penalise the echoing and repetition that `1m-best`
halves. One model recites and the other converses, and the number hides it.

This is the *same failure* as the 10K sweep in section 1, inverted. There, a
config with better perplexity lost the behavioural eval. Here, a model with far
better language modelling loses a behavioural eval that is measuring template
compliance. **Both times the objective was the problem, not the model.**

### What the extra scale did not buy

| model | params | bits/char |
|---|---|---|
| `500k-dev` | 491,040 | **1.30** |
| `1m-best` | 984,448 | 1.35 |

`1m-best` does not beat the `dev` branch's 500K despite **2x the parameters and
2.4x the data**. The comparison is partly confounded — the test text is drawn
from the templated dialogue set, and `1m-best` saw proportionally less of it —
but there is no evidence here that the extra scale paid for itself.

### Next

Drop `persona_chat` and retrain. It is the one source actively fighting the
identity training, it is ~4 MB of the 26 MB, and the other three sources carry
no first-person human personas. The expectation is that `name_acc` recovers
most of the way to 0.57 while the language-modelling gains survive, since they
come from volume and variety rather than from that source specifically.

---

## Practical recommendation

For a better 1K model today, in order of measured effect:

1. **Train longer.** 1,200 → 4,000 steps moved val loss 2.31 → 2.05. Largest
   single effect measured, and it costs nothing but time.
2. **Change the geometry.** Both are free at identical parameter count:
   - 1K: `vocab=40, d=8, L=1, ff=16, heads=2` (from `vocab=64 … ff=8`) — 5.4%
   - 10K: `vocab=192, d=16, L=2, ff=48, heads=2` (from `vocab=256, L=3, ff=16`) — 16.7%
3. **Do not distill** from a teacher trained on this corpus.

The two sweeps agree on the underlying lesson: **the shipped configs spend too
much on the embedding table and on depth, and too little on FFN width.** That
is a systematic bias in how the family was hand-designed, and it very likely
applies to the larger members too — worth sweeping 50K and 100K next.

For a better **1M** model, the order is different, because at that size the
data is the binding constraint rather than the geometry:

1. **Use real dialogue.** 2.89 → 1.35 bits/char, larger than every
   architecture effect measured on this branch combined.
2. **Train the full curriculum.** The shipped `1m` is undertrained at
   `STEPS_SCALE 0.35`.
3. **Filter the sources for identity conflicts.** `persona_chat` costs 21
   points of `name_acc`. More data is not automatically better data.

The recurring lesson across all three experiments is about **objectives, not
models**: twice now the metric being optimised has disagreed with the thing
actually wanted. Validation loss picked a config that could not spell its own
name; the behavioural composite prefers a model that recites over one that
converses. Any future search on this branch should score candidates on the
behaviour it wants, and read more than one number.
