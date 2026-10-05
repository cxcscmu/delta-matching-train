"""Small, dependency-free utilities shared across the pretraining engine.

`Peekable` replaces `more_itertools.peekable` (not a declared dependency): the FSDP2/DDP
trainers need a one-element look-ahead to snapshot the *next* batch for the resume
sanity check.
"""
from __future__ import annotations

import os
import random
from typing import Any, Iterable, Iterator

import numpy as np
import torch

_SENTINEL = object()


class Peekable:
    """Wrap an iterator so the next item can be inspected without consuming it.

    Drop-in for the single feature we use from more_itertools.peekable: `.peek()`
    plus normal iteration. `peek()` raises StopIteration when exhausted (same as
    more_itertools), which the trainers catch to roll the epoch over.
    """

    def __init__(self, iterable: Iterable):
        self._it: Iterator = iter(iterable)
        self._cache: Any = _SENTINEL

    def __iter__(self) -> "Peekable":
        return self

    def __next__(self) -> Any:
        if self._cache is not _SENTINEL:
            val, self._cache = self._cache, _SENTINEL
            return val
        return next(self._it)

    def peek(self) -> Any:
        if self._cache is _SENTINEL:
            self._cache = next(self._it)  # may raise StopIteration
        return self._cache


def seed_all(seed: int) -> int:
    """Seed python / numpy / torch (CPU+CUDA). Returns the seed for logging."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def get_amp_dtype(bf16: bool, fp16: bool) -> torch.dtype:
    """Resolve the autocast compute dtype. bf16 wins if both set (it is strictly
    safer than fp16 for LM pretraining: no loss scaling needed)."""
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return torch.float32


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts.

    NOTE: with weight tying (lm_head.weight is embed.weight) the shared tensor is
    a single nn.Parameter, so `model.parameters()` already counts it once. Callers
    that sum over `named_parameters()` instead would double count: use this.
    """
    seen: set[int] = set()
    total = trainable = 0
    for p in model.parameters():
        if id(p) in seen:  # tied weights share storage; count once
            continue
        seen.add(id(p))
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return total, trainable


def human_count(n: int) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)


def str_to_dir_format(path) -> str:
    """Coerce a Path/str into the str form torch.distributed.checkpoint wants as a
    checkpoint_id (it rejects Path on some torch versions)."""
    return os.fspath(path)
