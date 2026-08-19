#!/usr/bin/env python3
"""Train a tokenizer for every vocabulary size the sweep explores.

The sweep treats vocabulary as a search dimension, so each candidate size needs
its own tokenizer before any model can be trained.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "testing"))

from sweep import SPACES, candidates  # noqa: E402

vocabs = sorted({cfg.vocab_size for space in SPACES.values()
                 for cfg, _ in candidates(space)})
print(f"training {len(vocabs)} tokenizers: {vocabs}")
viable, rejected = [], []
for v in vocabs:
    out = ROOT / "testing" / "checkpoints" / f"tokenizer-{v}.json"
    if out.exists():
        print(f"  vocab {v:>4}: already present")
        viable.append(v)
        continue
    r = subprocess.run([sys.executable, str(ROOT / "scripts/train_tokenizer.py"),
                        "--data", "testing/data", "--out-dir", "testing/checkpoints",
                        "--vocab-size", str(v), "--quiet"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  vocab {v:>4}: ok")
        viable.append(v)
    else:
        # train_tokenizer.py asserts the model can spell its own name. Below
        # roughly 48 tokens the alphabet is capped so hard that most letters
        # fold into <unk> and "PocketLM" is unrepresentable. That is a real
        # floor on vocabulary size, so record it rather than crashing.
        why = "cannot represent 'PocketLM'" if "PocketLM" in r.stderr else \
              r.stderr.strip().splitlines()[-1][:70] if r.stderr.strip() else "failed"
        print(f"  vocab {v:>4}: REJECTED — {why}")
        rejected.append(v)

print(f"\nviable vocabularies:  {viable}")
if rejected:
    print(f"rejected (too small): {rejected}")
