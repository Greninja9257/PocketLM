# testing — how smart can 1K and 10K get?

Experimental branch. Everything here writes to `testing/data/` and
`testing/checkpoints/`, both gitignored, so nothing touches the shared `data/`
or `checkpoints/` that `main`'s released models depend on.

Trained on the **templated** corpus, deliberately. The `dev` branch established
that 11 MB of real human dialogue is above the learning floor for models this
size — a 9.5K model trained on it produces "It's him. What." — so the question
here is how much can be extracted from a narrow, learnable distribution.

Two levers were tested: **where the parameters go**, and **knowledge
distillation**. One worked.

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

### A finding that was not the point of the experiment

The sweep trains for 1,200 steps. The KD baseline below trains the *same*
architecture for 4,000 and reaches **2.0498**, well below the sweep's best
2.3052. **The 1K model was substantially undertrained**, and more steps buy
more than any architecture change tested here. Sweep numbers are therefore
valid for *ranking* configs, not as absolute quality.

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

## Practical recommendation

For a better 1K model today, in order of measured effect:

1. **Train longer.** 1,200 → 4,000 steps moved val loss 2.31 → 2.05. Largest
   single effect measured.
2. **Use `vocab=40, d=8, L=1, ff=16, heads=2`** instead of the shipped
   `vocab=64 … ff=8`. Free, 5.4%.
3. **Do not distill** from a teacher trained on this corpus.
