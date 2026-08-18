"""MLX training loop for Apple Silicon.

Same curriculum, same data, same checkpoints as train.py -- only the inner
loop differs. Data loading is deliberately reused from dataset.py rather than
reimplemented: batches are built once as torch tensors and handed to MLX as
arrays, which costs a memcpy on a machine with unified memory and saves an
entire duplicated pipeline.

Invoked through train.py --backend mlx; not usually run directly.
"""

import math
import time
from typing import List

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from config import Phase, TrainConfig
from model_mlx import TinyTransformerMLX


def to_mx(x, y, m):
    return (mx.array(x.numpy().astype("int32")),
            mx.array(y.numpy().astype("int32")),
            mx.array(m.numpy().astype("float32")))


def lr_at(step: int, total: int, peak: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))


def evaluate(model, corpus, sources, loss_on, batch_size, n_batches) -> float:
    losses = []
    for _ in range(n_batches):
        try:
            batch = corpus.sample(sources, batch_size, loss_on)
        except ValueError:
            return float("nan")
        x, y, m = to_mx(*batch)
        losses.append(float(model.loss(x, y, m)))
    return sum(losses) / max(len(losses), 1)


def run_phase(model: TinyTransformerMLX, phase: Phase, corpus, val_corpus,
              tcfg: TrainConfig, log: List[dict]) -> None:
    sources = corpus.available(phase.sources)
    if not sources:
        print(f"  skipping phase {phase.name!r}: none of {phase.sources} has data")
        return

    opt = optim.AdamW(learning_rate=phase.lr, betas=[0.9, 0.95],
                      weight_decay=tcfg.weight_decay)

    def loss_fn(m, x, y, msk):
        return m.loss(x, y, msk)

    step_fn = nn.value_and_grad(model, loss_fn)

    print(f"\n=== phase {phase.name}  sources={sources}  steps={phase.steps}  "
          f"lr={phase.lr:.2e}  loss_on={phase.loss_on}  [mlx]")
    t0, running = time.time(), []
    for step in range(phase.steps):
        opt.learning_rate = lr_at(step, phase.steps, phase.lr,
                                  tcfg.min_lr, tcfg.warmup_steps)
        x, y, m = to_mx(*corpus.sample(sources, tcfg.batch_size,
                                       phase.loss_on, phase.weights))
        loss, grads = step_fn(model, x, y, m)
        grads, _ = optim.clip_grad_norm(grads, tcfg.grad_clip)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state)
        running.append(float(loss))

        if (step + 1) % tcfg.log_every == 0:
            train_loss = sum(running) / len(running)
            running = []
            msg = (f"  {phase.name:<12} step {step + 1:>5}/{phase.steps}  "
                   f"loss {train_loss:.3f}  lr {opt.learning_rate.item():.2e}  "
                   f"{(step + 1) / (time.time() - t0):.0f} it/s")
            if (step + 1) % tcfg.eval_every == 0:
                val = evaluate(model, val_corpus, sources, phase.loss_on,
                               tcfg.batch_size, tcfg.eval_batches)
                msg += f"  val {val:.3f}  ppl {math.exp(min(val, 20)):.1f}"
                log.append({"phase": phase.name, "step": step + 1,
                            "train": train_loss, "val": val})
            print(msg, flush=True)
