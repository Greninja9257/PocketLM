#!/usr/bin/env python3
"""Phase-based training for any model in the PocketLM family.

    python train.py --model 50k
    python train.py --model 1m                 # uses MLX automatically on Apple Silicon
    python train.py --model 50k --backend torch
    python train.py --model 50k-gru            # variant at the same budget (torch only)


    phase 1  language      every token scored, plain text -- learn English
    phase 2  dialogue      assistant tokens only -- learn the turn structure
    phase 3  behavior      oversample follow-ups, clarifications, "I don't know"
    phase 4  personality   a short final pass in PocketLM's voice

Phases run in sequence on one set of weights, each with its own LR schedule.
The split exists because the model is too small to learn all of it at once:
early phases buy fluency, later ones spend it on being a chatbot.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

from config import (DEV_PHASES, LR_SCALE, PHASES, PRESETS, STEPS_SCALE,
                    ModelConfig, TrainConfig)
from dataset import build_corpus
from model import build_model, describe
from tokenizer import BPETokenizer


def pick_backend(requested: str, arch: str) -> str:
    """Choose MLX when it is available and applicable, else PyTorch.

    MLX is ~1.6-1.9x faster than PyTorch for these models on Apple Silicon, so
    it is the default when it can be used. It implements the transformer only,
    which covers the whole family; the GRU and hybrid variants fall back.
    """
    if requested != "auto":
        return requested
    if arch != "transformer":
        return "torch"
    try:
        import mlx.core as mx
        if mx.default_device().type == mx.DeviceType.gpu:
            return "mlx"
    except Exception:
        pass
    return "torch"


def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lr_at(step: int, total: int, peak: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (peak - min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def evaluate(model, corpus, sources, loss_on, batch_size, n_batches, device):
    model.eval()
    losses = []
    for _ in range(n_batches):
        try:
            x, y, m = corpus.sample(sources, batch_size, loss_on)
        except ValueError:
            return float("nan")
        losses.append(model.loss(x.to(device), y.to(device), m.to(device)).item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def run_phase(model, phase, corpus, val_corpus, tcfg, device, log):
    sources = corpus.available(phase.sources)
    if not sources:
        print(f"  skipping phase {phase.name!r}: none of {phase.sources} has data")
        return
    opt = torch.optim.AdamW(model.parameters(), lr=phase.lr,
                            weight_decay=tcfg.weight_decay, betas=(0.9, 0.95))
    print(f"\n=== phase {phase.name}  sources={sources}  steps={phase.steps}  "
          f"lr={phase.lr}  loss_on={phase.loss_on}")
    t0, running = time.time(), []
    for step in range(phase.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, phase.steps, phase.lr, tcfg.min_lr, tcfg.warmup_steps)
        x, y, m = corpus.sample(sources, tcfg.batch_size, phase.loss_on, phase.weights)
        loss = model.loss(x.to(device), y.to(device), m.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        running.append(loss.item())

        if (step + 1) % tcfg.log_every == 0:
            train_loss = sum(running) / len(running)
            running = []
            msg = (f"  {phase.name:<12} step {step + 1:>5}/{phase.steps}  "
                   f"loss {train_loss:.3f}  lr {opt.param_groups[0]['lr']:.2e}  "
                   f"{(step + 1) / (time.time() - t0):.0f} it/s")
            if (step + 1) % tcfg.eval_every == 0:
                val = evaluate(model, val_corpus, sources, phase.loss_on,
                               tcfg.batch_size, tcfg.eval_batches, device)
                msg += f"  val {val:.3f}  ppl {math.exp(min(val, 20)):.1f}"
                log.append({"phase": phase.name, "step": step + 1,
                            "train": train_loss, "val": val})
            print(msg, flush=True)


def save(model, tok_path, out, log):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.cfg.to_dict(),
                "state_dict": model.state_dict(),
                "tokenizer": str(tok_path),
                "log": log}, out)
    print(f"\nsaved -> {out}  ({model.n_params():,} params)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="50k", choices=list(PRESETS),
                    help="which member of the family to train")
    ap.add_argument("--data", default="data")
    ap.add_argument("--tokenizer", default=None,
                    help="default: the tokenizer matching the model's vocab size")
    ap.add_argument("--out", default=None, help="default checkpoints/<model>.pt")
    ap.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"],
                    help="auto picks MLX on Apple Silicon for transformer models")
    ap.add_argument("--device", default="auto", help="torch backend only")
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--steps-scale", type=float, default=None,
                    help="multiply every phase's step count (0.05 for a smoke test); "
                         "defaults to the per-size scale in config.STEPS_SCALE")
    ap.add_argument("--phases", default=None, help="comma-separated subset of phase names")
    ap.add_argument("--curriculum", default="default", choices=["default", "dev"],
                    help="'dev' trains on real dialogue (data/real_noisy) instead "
                         "of the synthetic generator")
    ap.add_argument("--lr-scale", type=float, default=None,
                    help="override the per-size learning-rate multiplier")
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = args.out or f"checkpoints/{args.model}.pt"

    cfg = PRESETS[args.model]
    cfg.validate()
    backend = pick_backend(args.backend, cfg.arch)
    device = pick_device(args.device) if backend == "torch" else None
    tok_path = args.tokenizer or cfg.tokenizer
    if not Path(tok_path).exists():
        raise SystemExit(f"no tokenizer at {tok_path} — run scripts/train_tokenizer.py")
    tok = BPETokenizer.load(tok_path)
    if len(tok) != cfg.vocab_size:
        raise SystemExit(f"tokenizer {tok_path} has {len(tok)} tokens but "
                         f"{args.model} expects {cfg.vocab_size}")

    if backend == "mlx":
        import mlx.core as mx
        from model_mlx import build_model_mlx, to_torch_state_dict
        mx.random.seed(args.seed)
        model = build_model_mlx(cfg)
        print(describe(build_model(cfg)))
        print(f"backend: mlx ({mx.default_device()})   tokenizer: {tok_path}")
    else:
        model = build_model(cfg).to(device)
        print(describe(model))
        print(f"backend: torch ({device})   tokenizer: {tok_path}")

    ctx = cfg.context_length
    corpus = build_corpus(args.data, tok, ctx, "train", seed=args.seed)
    val_corpus = build_corpus(args.data, tok, ctx, "val", seed=args.seed)
    print("\ncorpus:")
    print(corpus.stats())

    tcfg = TrainConfig(batch_size=args.batch_size, seed=args.seed)
    wanted = set(args.phases.split(",")) if args.phases else None
    # Wider models need a cooler learning rate; 3e-3 diverges at 1M.
    lr_scale = args.lr_scale if args.lr_scale is not None else LR_SCALE.get(args.model, 1.0)
    steps_scale = (args.steps_scale if args.steps_scale is not None
                   else STEPS_SCALE.get(args.model, 1.0))
    print(f"lr scale: {lr_scale}x   steps scale: {steps_scale}x")
    log = []
    if backend == "mlx":
        import train_mlx
    curriculum = DEV_PHASES if args.curriculum == "dev" else PHASES
    for phase in curriculum:
        if wanted and phase.name not in wanted:
            continue
        scaled = type(phase)(phase.name, phase.sources,
                             max(1, int(phase.steps * steps_scale)),
                             phase.lr * lr_scale, phase.loss_on, phase.weights)
        if backend == "mlx":
            train_mlx.run_phase(model, scaled, corpus, val_corpus, tcfg, log)
        else:
            run_phase(model, scaled, corpus, val_corpus, tcfg, device, log)

    if backend == "mlx":
        # Save as a PyTorch checkpoint so chat.py, eval/, export.py and
        # runtime_numpy.py never need to know MLX was involved.
        torch.save({"config": cfg.to_dict(), "state_dict": to_torch_state_dict(model),
                    "tokenizer": tok_path, "log": log, "backend": "mlx"}, out)
        print(f"\nsaved -> {out}  ({cfg.n_params():,} params, trained with mlx)")
    else:
        save(model, tok_path, out, log)


if __name__ == "__main__":
    main()
