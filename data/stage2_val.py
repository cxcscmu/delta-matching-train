#!/usr/bin/env python
"""Stage 2, step 3: carve the held-out validation windows out of the stage-2 training stream (data/pretrain_stage2.md).
Called by data/build_stage2.py; runnable on its own:

    python data/stage2_val.py --train <root>/prolong_64k_llama2/prolong64k_train \
        --out <root>/prolong_64k_llama2/prolong64k_val

The training stream is domain-ordered, so a tail slice would be code or textbooks only. The training sampler
(src/pretrain/data/dist_sampler.py) draws windows in the order PCG64(data.seed + epoch).permutation(n_windows) and a run's
data.chunk_subset (lo, hi) keeps order[int(lo * n):int(hi * n)]. The validation windows are the first 256 of the same
seed-11785 permutation's last 2 % (order[int(0.98 n):]), sorted, copied whole: a random sample of every domain that no
run with data.seed 11785 and a chunk_subset ending at or below 0.98 ever trains on (stage 2 uses [0, 0.85]). Each
validation "document" is one 65,536-token window, so the evaluator scores exactly the training chunking."""
import argparse

import numpy as np

CHUNK = 65536


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="training prefix: <train>.bin + <train>.doc_lengths.npy")
    ap.add_argument("--out", required=True, help="output prefix: <out>.bin + <out>.doc_lengths.npy")
    ap.add_argument("--seed", type=int, default=11785, help="data.seed of the training runs")
    ap.add_argument("--start", type=float, default=0.98, help="the held-out part of the permutation starts here")
    ap.add_argument("--windows", type=int, default=256)
    a = ap.parse_args()
    n_tok = int(np.load(a.train + ".doc_lengths.npy").sum())
    n = n_tok // CHUNK                                   # the sampler's dataset length
    order = np.random.Generator(np.random.PCG64(a.seed)).permutation(n)
    held = order[int(a.start * n):n]
    assert len(held) >= a.windows, f"only {len(held)} held-out windows"
    pick = np.sort(held[:a.windows])
    mm = np.memmap(a.train + ".bin", dtype=np.uint16, mode="r")
    assert mm.size == n_tok, f"{a.train}.bin holds {mm.size} tokens, its document lengths sum to {n_tok}"
    out = np.empty(a.windows * CHUNK, dtype=np.uint16)
    for i, c in enumerate(pick):
        out[i * CHUNK:(i + 1) * CHUNK] = mm[int(c) * CHUNK:(int(c) + 1) * CHUNK]
    out.tofile(a.out + ".bin")
    np.save(a.out + ".doc_lengths.npy", np.full(a.windows, CHUNK, dtype=np.int64))
    print(f"[stage2] validation: {a.windows} of {len(held):,} held-out windows (of {n:,}; ids {int(pick[0]):,} .. "
          f"{int(pick[-1]):,}) -> {a.out}.bin ({out.size:,} tokens)")


if __name__ == "__main__":
    main()
