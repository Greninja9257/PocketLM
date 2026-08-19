# Models

Every trained PocketLM, in one place. Each `.npz` is **self-contained** —
weights, tokenizer and config in a single file — and runs with **numpy alone**,
no PyTorch, no framework:

```bash
python runtime_numpy.py --model models/main/500k.npz
python runtime_numpy.py --model models/main/1k.npz --prompt "hey"
```

To chat with one through the full system (external memory, output filter):

```bash
python chat.py --checkpoint checkpoints/500k.pt        # main models
```

| folder | what it is |
|---|---|
| `main/` | the released family — one architecture, seven budgets |
| `dev/` | trained on 11 MB of real human dialogue instead of templates |
| `testing/` | experimental geometries and a distillation attempt |


## main/ — the released family

| model | params | file | what it is |
|---|---|---|---|
| [`1k.npz`](main/1k.npz) | 984 | 4.8 KB | Smallest possible. Learns the shape of conversation, not English. |
| [`5k.npz`](main/5k.npz) | 4,944 | 14.5 KB | Word-like fragments; occasional real phrases. |
| [`10k.npz`](main/10k.npz) | 9,584 | 25.4 KB | First size with reliably grammatical replies. |
| [`50k.npz`](main/50k.npz) | 48,416 | 97.9 KB | The original 50K target. Holds a short conversation. |
| [`100k.npz`](main/100k.npz) | 95,664 | 184.4 KB | Steadier than 50k; longer 192-token context. |
| [`500k.npz`](main/500k.npz) | 491,040 | 897.6 KB | **Best overall.** Highest score on the 900-prompt suite. |
| [`1m.npz`](main/1m.npz) | 968,320 | 1.71 MB | Lowest perplexity, but undertrained — see Results. |

## dev/ — real dialogue

| model | params | file | what it is |
|---|---|---|---|
| [`10k-real.npz`](dev/10k-real.npz) | 9,584 | 25.8 KB | 10k architecture on real human dialogue. Collapses — below the floor. |
| [`10k-hybrid.npz`](dev/10k-hybrid.npz) | 9,968 | 23.7 KB | GRU + one attention block, real dialogue. Same budget as 10k-real. |
| [`50k-real.npz`](dev/50k-real.npz) | 48,416 | 99.1 KB | Real dialogue + typo/casing noise. Net loss at this size. |
| [`500k-real.npz`](dev/500k-real.npz) | 491,040 | 899.1 KB | **Most robust.** Survives `Hello`, `HELLO`, `wats ur name`. |

## testing/ — experiments

| model | params | file | what it is |
|---|---|---|---|
| [`1k-best.npz`](testing/1k-best.npz) | 984 | 4.8 KB | 1k geometry found by budget search. Beats `1k` on the suite. |
| [`10k-best.npz`](testing/10k-best.npz) | 9,808 | 23.5 KB | Better perplexity than `10k`, worse benchmark — lowercase-locked. |
| [`1k-kd.npz`](testing/1k-kd.npz) | 984 | 4.8 KB | Distilled from a 396K teacher. Worse than `1k` — a failed experiment. |

## Which one should I use?

- **Just want it to work:** `main/500k.npz` — the best scores on the held-out suite.
- **Smallest usable:** `main/10k.npz` at 25 KB is the first size with reliably
  grammatical replies.
- **Typing on a phone / expecting typos:** `dev/500k-real.npz` — the only model
  that handles `Hello`, `HELLO` and `wats ur name`.
- **Curiosity:** `main/1k.npz` is 4.8 KB and produces English-shaped noise.
  That is the point of it.

## What they can't do

None of these models know facts. They hold short conversations, use facts you
supply through external memory, and decline what they don't know. Memory
*copying* fails at every size — given "favourite colour: turquoise" they will
answer with a colour, but usually the wrong one. See the main README.
