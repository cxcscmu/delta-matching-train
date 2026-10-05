"""Per-step engine context threaded into the rebound layer forwards.

A single forward pass is either all-prefill (one sequence, variable length) or
all-decode (N sequences, one token each): phase is per-forward, so the engine
never needs the HF cache's global `has_previous_state` flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:   # annotations only: importing a pool here would tie this module to one engine path
    from .cache import HybridStatePool
    from .dm_cache import DMStatePool

PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"


@dataclass
class EngineContext:
    pool: HybridStatePool | DMStatePool   # Nemotron-H path | dmlm path
    phase: str
    # decode: int32 (N,) mapping compact batch row -> pool slot
    slot_ids: torch.Tensor | None = None
    # prefill: the single pool slot being initialized
    prefill_slot: int | None = None
    # RoPE tables for the dmlm engine (the attention mixers): prefill = cos/sin for positions 0..L-1
    # ([1,L,D]); decode = cos/sin gathered at each running slot's current position ([N,1,D]); the identity
    # rotation (cos = 1, sin = 0) for a NoPE model. The Nemotron-H path leaves these None (NoPE attention).
    cos: torch.Tensor | None = None
    sin: torch.Tensor | None = None
