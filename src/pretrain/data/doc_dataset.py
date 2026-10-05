"""Document-boundary-aware token dataset for ProLong-style packed long-context data.

Same fixed-`chunk_size` chunking as `MemmapTokenDataset` (so the resumable sampler and
`global_step`-based resume are unchanged) over a single flat token `.bin`, PLUS a sidecar
`doc_lengths` array (int64, one entry per document) that records how the stream is split
into documents. From it each chunk additionally carries:

- `position_ids`: RoPE positions. `position_mode="packed"` (default) keeps continuous
  0..T-1 positions (cross-document long-context: what ProLong is packed *for*);
  `"reset"` restarts positions at each document start (use this only together with
  intra-document attention masking).
- `cu_seqlens`: the document-boundary offsets *within* the chunk, as a cumulative
  `[0, b1, b2, ..., T]` int32 tensor (the flash-attn-varlen convention). This is the
  32k-feasible representation of "respect boundaries": a dense [T,T] intra-doc mask is
  4 GB at T=32768, so masking must be done varlen from `cu_seqlens`, not a materialized
  mask. (The varlen wiring in the GQA/SSM mixers is the remaining step: see
  `docs/train_engine.md`; today `cu_seqlens` is emitted and `position_ids` is honored.)

Padding is avoided entirely: documents are concatenated and chunked exactly as the packed
stream already is; boundaries are tracked, not enforced by splitting/padding.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .memmap_dataset import DTYPES


class DocAwareMemmapDataset(Dataset):
    def __init__(self, path: str, chunk_size: int, doc_lengths, dtype: str = "uint16",
                 position_mode: str = "packed", prepend_bos: bool = False, bos_token_id: int = 1):
        assert dtype in DTYPES, f"dtype must be one of {list(DTYPES)}"
        assert position_mode in ("packed", "reset"), position_mode
        # BOS-anchored windows: [bos] + chunk[:-1] (same chunks, order and length; see DataConfig.prepend_bos). The BOS
        # joins the first document's segment of cu_seqlens; per-document ("reset") positions are not defined for it.
        if prepend_bos and position_mode != "packed":
            raise NotImplementedError("prepend_bos is implemented for position_mode='packed' only")
        self.prepend_bos, self.bos_token_id = bool(prepend_bos), int(bos_token_id)
        self.path = str(path)
        self.chunk_size = int(chunk_size)
        self.np_dtype = DTYPES[dtype]
        self.position_mode = position_mode
        # document start offsets in the flat stream: cumsum([0, len0, len1, ...])
        dl = np.load(doc_lengths) if isinstance(doc_lengths, str) else np.asarray(doc_lengths)
        self.doc_starts = np.concatenate([[0], np.cumsum(dl.astype(np.int64))]).astype(np.int64)
        self.n_tokens = int(self.doc_starts[-1])
        # the sidecar must describe exactly this stream: catches a wrong dtype, a sidecar of another file and a .npy
        # (whose header np.memmap would read as tokens)
        n_bytes, itemsize = os.path.getsize(self.path), np.dtype(self.np_dtype).itemsize
        if n_bytes != self.n_tokens * itemsize:
            raise ValueError(f"{self.path} holds {n_bytes} bytes, but doc_lengths sums to {self.n_tokens} tokens = "
                             f"{self.n_tokens * itemsize} bytes as {dtype}: check data.dtype, that the sidecar belongs "
                             f"to this file, and that the file is a raw token .bin")
        self.n_chunks = self.n_tokens // self.chunk_size
        if self.n_chunks <= 0:
            raise ValueError(f"0 chunks of size {chunk_size} for {self.n_tokens} tokens")
        self.base_name = "docaware"
        self._mm = None

    def _mmap(self) -> np.ndarray:
        if self._mm is None:  # lazy + per-worker (pickles cleanly)
            self._mm = np.memmap(self.path, dtype=self.np_dtype, mode="r")
        return self._mm

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.n_chunks:
            raise IndexError(idx)
        start = idx * self.chunk_size
        T = self.chunk_size
        arr = np.asarray(self._mmap()[start:start + T]).astype(np.int64)
        g = np.arange(start, start + T)
        # which document each token belongs to (largest doc_start <= g)
        di = np.searchsorted(self.doc_starts, g, side="right") - 1
        if self.prepend_bos:   # shift right by one: the BOS takes position 0 (first document's segment), the last token drops
            arr = np.concatenate([[self.bos_token_id], arr[:-1]]).astype(np.int64)
            di = np.concatenate([di[:1], di[:-1]])
        if self.position_mode == "reset":
            pos = (g - self.doc_starts[di]).astype(np.int64)
        else:  # packed: continuous positions across the chunk
            pos = np.arange(T, dtype=np.int64)
        # cu_seqlens: chunk-local offsets where the document index changes, + endpoints
        change = np.nonzero(np.diff(di))[0] + 1            # local indices of doc starts
        cu = np.concatenate([[0], change, [T]]).astype(np.int32)
        return {
            "input_ids": torch.from_numpy(arr),
            "position_ids": torch.from_numpy(pos),
            "cu_seqlens": torch.from_numpy(cu),            # ragged length; not collated by default
            "index": idx,
        }
