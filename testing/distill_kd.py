#!/usr/bin/env python3
"""Knowledge distillation into a tiny student.

A 984-parameter model trained on hard labels sees one bit of information per
token: which token was correct. A teacher's full distribution says much more --
that after "I'm having a rough" the plausible continuations are "day", "week",
"time", and roughly how plausible each is. That extra signal is worth more to a
small model than to a large one, because the small model cannot afford to
rediscover the structure itself.

The constraint is that teacher and student must share a vocabulary: the KL is
computed over the output distribution, so the two must be the same shape. So
the teacher here is not one of the released models -- it is a mid-sized model
trained from scratch on the student's tokenizer.

    python testing/distill_kd.py --teacher testing/checkpoints/teacher-v40.pt \\
        --vocab 40 --d-model 8 --n-layers 1 --d-ff 16 --n-heads 2

Loss is Hinton KD: KL against temperature-softened teacher logits, blended with
ordinary cross-entropy, both restricted to assistant tokens.
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from config import ModelConfig, tokenizer_path
from dataset import build_corpus
from model import build_model
from tokenizer import BPETokenizer
from train import lr_at, pick_device


def kd_loss(student_logits, teacher_logits, targets, mask, temperature, alpha):
    """Blend soft-target KL with hard-label cross-entropy, on masked positions."""
    m = mask.reshape(-1).float()
    denom = m.sum().clamp(min=1.0)

    s = student_logits.reshape(-1, student_logits.size(-1))
    t = teacher_logits.reshape(-1, teacher_logits.size(-1))
    # T^2 keeps the soft-loss gradient magnitude comparable to the hard loss.
    soft = F.kl_div(F.log_softmax(s / temperature, dim=-1),
                    F.log_softmax(t / temperature, dim=-1),
                    reduction="none", log_target=True).sum(-1)
    soft = (soft * m).sum() / denom * (temperature ** 2)

    hard = F.cross_entropy(s, targets.reshape(-1), reduction="none")
    hard = (hard * m).sum() / denom
    return alpha * soft + (1 - alpha) * hard, hard.item()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data", default="testing/data")
    ap.add_argument("--tokenizer-dir", default="testing/checkpoints")
    ap.add_argument("--out", default=None)
    # Student geometry, defaulting to the 1K sweep winner.
    ap.add_argument("--vocab", type=int, default=40)
    ap.add_argument("--d-model", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=1)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--d-ff", type=int, default=16)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.7, help="weight on the soft loss")
    ap.add_argument("--baseline", action="store_true",
                    help="also train an identical student on hard labels only")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = BPETokenizer.load(str(Path(args.tokenizer_dir) /
                                Path(tokenizer_path(args.vocab)).name))

    ckpt = torch.load(args.teacher, map_location=device, weights_only=False)
    teacher = build_model(ModelConfig.from_dict(ckpt["config"]), strict=False).to(device)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.eval()
    # Count before freezing: n_params() only counts requires_grad parameters,
    # so reading it after this loop reports zero.
    teacher_params = teacher.n_params()
    for p in teacher.parameters():
        p.requires_grad_(False)
    if teacher.cfg.vocab_size != args.vocab:
        raise SystemExit(f"teacher vocab {teacher.cfg.vocab_size} != student vocab "
                         f"{args.vocab}; KD needs a shared output space")

    student_cfg = ModelConfig(name="kd-student", vocab_size=args.vocab,
                              context_length=args.ctx, d_model=args.d_model,
                              n_layers=args.n_layers, n_heads=args.n_heads,
                              n_kv_heads=args.n_heads, d_ff=args.d_ff,
                              budget=args.budget)
    print(f"teacher: {teacher_params:,} params   "
          f"student: {student_cfg.n_params():,} params "
          f"({student_cfg.n_params()/teacher_params:.2%} of teacher)")

    corpus = build_corpus(args.data, tok, args.ctx, "train", seed=args.seed)
    val = build_corpus(args.data, tok, args.ctx, "val", seed=args.seed)
    sources = corpus.available(["dialogue", "behavior", "personality"])

    def evaluate(model):
        model.eval()
        out = []
        with torch.no_grad():
            for _ in range(20):
                x, y, m = val.sample(sources, args.batch_size, "assistant")
                out.append(model.loss(x.to(device), y.to(device), m.to(device)).item())
        model.train()
        return sum(out) / len(out)

    def run(distill: bool, label: str):
        torch.manual_seed(args.seed)
        model = build_model(student_cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                                betas=(0.9, 0.95))
        t0 = time.time()
        for step in range(args.steps):
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.lr, 3e-4, 200)
            x, y, m = corpus.sample(sources, args.batch_size, "assistant")
            x, y, m = x.to(device), y.to(device), m.to(device)
            if distill:
                with torch.no_grad():
                    tl = teacher(x)
                loss, hard = kd_loss(model(x), tl, y, m, args.temperature, args.alpha)
            else:
                loss = model.loss(x, y, m)
                hard = loss.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (step + 1) % 500 == 0:
                print(f"  {label:<10} step {step+1:>5}/{args.steps}  "
                      f"hard-CE {hard:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        v = evaluate(model)
        print(f"  {label:<10} final val loss {v:.4f}  ppl {math.exp(min(v,20)):.2f}")
        return v, model

    print(f"\nteacher val loss: {evaluate(teacher):.4f}\n")
    kd_val, kd_model = run(True, "distilled")

    if args.baseline:
        base_val, _ = run(False, "baseline")
        delta = base_val - kd_val
        print(f"\n  baseline  {base_val:.4f}\n  distilled {kd_val:.4f}\n"
              f"  improvement {delta:+.4f} ({delta/base_val:+.1%})")

    out = args.out or f"testing/checkpoints/kd-student-v{args.vocab}.pt"
    torch.save({"config": student_cfg.to_dict(), "state_dict": kd_model.state_dict(),
                "tokenizer": str(Path(args.tokenizer_dir) /
                                 Path(tokenizer_path(args.vocab)).name)}, out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
