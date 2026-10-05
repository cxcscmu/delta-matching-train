#!/usr/bin/env python
"""Generate a tiny raw token .bin file for tests / dry-runs.

Token ids are random in [0, vocab). Random data has no learnable structure across
the whole set, but a small dataset is memorizable: which is exactly what the smoke
test exploits (loss must drop when the model overfits a small token pool).
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

DT = {"uint16": np.uint16, "uint32": np.uint32}


def write_tokens(out: str, n_tokens: int, vocab: int, dtype: str = "uint16", seed: int = 0) -> str:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, vocab, size=n_tokens, dtype=DT[dtype])
    arr.tofile(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), f"pretrain_smoke_{os.getuid()}",
                                                  "data", "tokens.bin"))
    ap.add_argument("--n_tokens", type=int, default=1_000_000)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--dtype", choices=list(DT), default="uint16")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    write_tokens(a.out, a.n_tokens, a.vocab, a.dtype, a.seed)
    print(f"wrote {a.n_tokens} {a.dtype} tokens (vocab={a.vocab}) -> {a.out}")


if __name__ == "__main__":
    main()
