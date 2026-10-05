"""Resumable distributed sampler.

Determinism contract: the global sample order is a pure function of (seed, epoch), and
rank r takes every `num_replicas`-th sample of it starting at r. `iteration` is the
number of samples each rank has already consumed in the current epoch: resuming drops
the first `iteration * num_replicas` samples of the order, then takes this rank's
strided slice. One optimizer step consumes one loader batch per rank, i.e.
`num_replicas * batch_size` samples, so the trainer recovers (epoch, iteration) from
`global_step` alone (BaseTrainer.jump_to_step) and pins the exact next batch with no
extra state.
"""
from __future__ import annotations

import math

import numpy as np
from torch.utils.data import Sampler


class ResumableDistSampler(Sampler):
    def __init__(self, dataset_len: int, num_replicas: int, rank: int,
                 seed: int = 42, shuffle: bool = True, drop_last: bool = True,
                 chunk_subset=None):
        self.dataset_len = int(dataset_len)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.iteration = 0
        # chunk_subset = (start_frac, end_frac): restrict sampling to a contiguous slice of the FULL
        # (seed-fixed) permutation. Disjoint fraction windows give disjoint random subsets of all chunks
        # spanning all domains (scripts/campaign/stage2_arm.sh trains on [0, 0.85] of the stage-2
        # permutation; data/stage2_val.py draws the held-out windows from its last 2 %).
        if chunk_subset is not None:
            lo, hi = float(chunk_subset[0]), float(chunk_subset[1])
            self._sub_lo = int(lo * self.dataset_len)
            self._sub_hi = int(hi * self.dataset_len)
            assert 0 <= self._sub_lo < self._sub_hi <= self.dataset_len, chunk_subset
        else:
            self._sub_lo, self._sub_hi = 0, self.dataset_len
        self._eff_len = self._sub_hi - self._sub_lo
        if self.drop_last:
            self.num_samples = self._eff_len // self.num_replicas
        else:
            self.num_samples = math.ceil(self._eff_len / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def set_epoch(self, epoch: int, iteration: int | None = None) -> None:
        self.epoch = int(epoch)
        self.iteration = int(iteration) if iteration is not None else 0

    def _order(self) -> np.ndarray:
        if self.shuffle:
            g = np.random.Generator(np.random.PCG64(self.seed + self.epoch))
            idx = g.permutation(self.dataset_len)
        else:
            idx = np.arange(self.dataset_len)
        idx = idx[self._sub_lo:self._sub_hi]          # restrict to this split's disjoint subset
        if len(idx) >= self.total_size:
            idx = idx[: self.total_size]
        else:  # pad by repeating from the front to make it evenly divisible
            pad = self.total_size - len(idx)
            idx = np.concatenate([idx, idx[:pad]])
        return idx

    def __iter__(self):
        idx = self._order()
        if self.iteration > 0:
            idx = idx[self.iteration * self.num_replicas:]
        idx = idx[self.rank:: self.num_replicas]
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples - self.iteration
