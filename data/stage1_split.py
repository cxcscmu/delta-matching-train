#!/usr/bin/env python
"""Stage 1, step 3: the representative random *document-level* holdout split of the tokenized stream
(data/pretrain_stage1.md). Called by data/build_stage1.py; runnable on its own:

    python data/stage1_split.py --bin <root>/nemotron_cc_v2d1_hq_dqa/nemotron_hqdqa.bin \
        --doc-lengths <root>/nemotron_cc_v2d1_hq_dqa/nemotron_hqdqa.doc_lengths.npy \
        --out <root>/nemotron_cc_v2d1_hq_dqa/nemotron_rs --val-frac 0.02 --seed 11785

WHY doc-level + shuffled val: a contiguous byte/tail split puts a single corpus region (e.g. one domain, or the HQ-then-
DQA ordering of the stream) entirely in val -> an unrepresentative held-out set. A per-document Bernoulli(val_frac)
holdout samples every region proportionally; shuffling the val document order makes any contiguous window set of the
val-CE scorer representative too.

Writes  <out>_train.bin / <out>_train.doc_lengths.npy   (original document order)
        <out>_val.bin   / <out>_val.doc_lengths.npy     (val document order SHUFFLED)
Invariant: sum(train_lengths)+sum(val_lengths) == sum(doc_lengths); byte totals match.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

DT = {"uint16": np.uint16, "uint32": np.uint32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--doc-lengths", required=True)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=11785)
    ap.add_argument("--dtype", default="uint16", choices=list(DT))
    a = ap.parse_args()

    dl = np.load(a.doc_lengths).astype(np.int64)
    n = len(dl)
    starts = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(dl, out=starts[1:])
    n_tok = int(starts[-1])

    mm = np.memmap(a.bin, dtype=DT[a.dtype], mode="r")
    assert mm.shape[0] == n_tok, f"bin tokens {mm.shape[0]} != sum(doc_lengths) {n_tok}"

    rng = np.random.RandomState(a.seed)
    is_val = rng.random(n) < a.val_frac
    val_idx = np.nonzero(is_val)[0]
    val_order = rng.permutation(val_idx)               # shuffle val doc order
    print(f"[split] {n:,} docs / {n_tok:,} tok -> val={len(val_idx):,} docs "
          f"({100*len(val_idx)/n:.2f}%), seed={a.seed}", flush=True)

    t0 = time.time()
    # TRAIN: every non-val doc, in original order, copied as contiguous runs (RLE of
    # the non-val mask -> one block write per maximal run, ~#val_docs writes total).
    notval = (~is_val).astype(np.int8)
    d = np.diff(np.concatenate([[0], notval, [0]]))
    run_s = np.nonzero(d == 1)[0]                       # doc index where a keep-run starts
    run_e = np.nonzero(d == -1)[0]                      # one-past-end doc index
    train_len_parts = []
    with open(a.out + "_train.bin", "wb") as ft:
        for s, e in zip(run_s, run_e):
            mm[starts[s]:starts[e]].tofile(ft)
            train_len_parts.append(dl[s:e])
    train_lens = (np.concatenate(train_len_parts) if train_len_parts
                  else np.zeros(0, dtype=np.int64))
    np.save(a.out + "_train.doc_lengths.npy", train_lens)

    # VAL: val docs in shuffled order, one block per doc.
    with open(a.out + "_val.bin", "wb") as fv:
        for g in val_order:
            mm[starts[g]:starts[g + 1]].tofile(fv)
    val_lens = dl[val_order]
    np.save(a.out + "_val.doc_lengths.npy", val_lens)

    # invariants
    itembytes = np.dtype(DT[a.dtype]).itemsize
    nt = os.path.getsize(a.out + "_train.bin") // itembytes
    nv = os.path.getsize(a.out + "_val.bin") // itembytes
    assert nt == int(train_lens.sum()) and nv == int(val_lens.sum()), "byte/len mismatch"
    assert nt + nv == n_tok, f"train+val tokens {nt+nv} != {n_tok}"
    print(f"[split] DONE  train={nt:,} tok ({nt*itembytes/1e9:.1f} GB, {len(train_lens):,} docs) "
          f"val={nv:,} tok ({nv*itembytes/1e9:.2f} GB, {len(val_lens):,} docs)  "
          f"sum-check OK  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
