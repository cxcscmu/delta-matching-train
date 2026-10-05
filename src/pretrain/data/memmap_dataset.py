"""Memmap token dataset: flat token-id files -> fixed-length chunks.

On-disk format: a directory or list of raw token files. Each file
is a contiguous array of token ids in uint16 (vocab <= 65535) or uint32. We chunk
each file into floor(n_tokens / chunk_size) sequences of exactly `chunk_size` tokens
(packed pretraining; the trailing remainder is dropped). Document EOS separators
must already be baked into the token stream.

Supports `.npy` (loaded via np.load mmap_mode) and raw `.bin` (np.memmap). Memmaps
are opened lazily and cached per-process so the dataset pickles cleanly to
DataLoader workers.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

DTYPES = {"uint16": np.uint16, "uint32": np.uint32}


def _resolve_paths(paths) -> list[str]:
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out: list[str] = []
    for p in paths:
        p = str(p)
        if os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "*.bin")) + glob.glob(os.path.join(p, "*.npy")))
            out.extend(files)
        else:
            out.append(p)
    if not out:
        raise FileNotFoundError(f"No token files found under {paths}")
    return out


class MemmapTokenDataset(Dataset):
    def __init__(self, paths, chunk_size: int, dtype: str = "uint16", prepend_bos: bool = False, bos_token_id: int = 1):
        assert dtype in DTYPES, f"dtype must be one of {list(DTYPES)}"
        self.chunk_size = int(chunk_size)
        # BOS-anchored windows: [bos] + chunk[:-1] (same chunks, order and length; see DataConfig.prepend_bos)
        self.prepend_bos, self.bos_token_id = bool(prepend_bos), int(bos_token_id)
        self.np_dtype = DTYPES[dtype]
        self.itemsize = np.dtype(self.np_dtype).itemsize
        self.paths = _resolve_paths(paths)
        # chunk counts from file sizes (no need to open the memmap at init)
        chunks_per = []
        for p in self.paths:
            n_tokens = os.path.getsize(p) // self.itemsize
            if p.endswith(".npy"):
                # .npy carries a header; subtract it by trusting np.load's shape instead
                n_tokens = int(np.load(p, mmap_mode="r").shape[0])
            chunks_per.append(int(n_tokens) // self.chunk_size)
        self._cum = np.concatenate([[0], np.cumsum(chunks_per)]).astype(np.int64)
        self.n_chunks = int(self._cum[-1])
        if self.n_chunks <= 0:
            raise ValueError(f"Dataset has 0 chunks of size {chunk_size} (files too small): {self.paths}")
        self.base_name = os.path.basename(os.path.dirname(self.paths[0])) or os.path.basename(self.paths[0])
        self._cache: dict[int, np.ndarray] = {}

    def _mmap(self, fi: int) -> np.ndarray:
        m = self._cache.get(fi)
        if m is None:
            p = self.paths[fi]
            m = np.load(p, mmap_mode="r") if p.endswith(".npy") else np.memmap(p, dtype=self.np_dtype, mode="r")
            self._cache[fi] = m
        return m

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.n_chunks:
            raise IndexError(idx)
        fi = int(np.searchsorted(self._cum, idx, side="right") - 1)
        local = idx - int(self._cum[fi])
        start = local * self.chunk_size
        arr = np.asarray(self._mmap(fi)[start:start + self.chunk_size]).astype(np.int64)
        if self.prepend_bos:
            arr = np.concatenate([[self.bos_token_id], arr[:-1]]).astype(np.int64)
        return {"input_ids": torch.from_numpy(arr), "index": idx}
