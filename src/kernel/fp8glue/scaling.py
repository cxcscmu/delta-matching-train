"""P3: scaling state.  Two modes, selected by env DM_FP8GLUE_SCALING (default `dynamic`):

dynamic  torchao's dynamic tensorwise scaling, two passes per cast (exact amax, then cast); the
         codes and scales are bitwise torchao's (common.py).  No state.
delayed  TE-class delayed tensorwise scaling (a REALIZATION CHANGE of every activation/gradient
         operand it touches; weights keep the precomputed-scale path of the fp8 all-gather):
           * per (layer, role) an amax history ring of HIST_LEN entries (fp32, device), algo max,
             margin 0; per rank, no cross-rank reduction;
           * scale for the current cast = FP8_MAX / max(history)  (fp64 division then fp32, the
             torchao rule applied to the history max), computed BEFORE the current amax enters;
           * the cast kernel emits the exact current amax (fp32 atomic max) into the role's
             `cur` slot; at the NEXT block-forward entry of a new microbatch all `cur` slots of
             the layer are pushed into the ring (one kernel per layer per microbatch) and zeroed;
           * bootstrap: the first cast of a role runs the dynamic two-pass and its amax seeds
             the history (no scale=1 first-step transient);
           * casts are saturating (clamp to +-FP8_MAX before conversion): load-bearing when the
             current amax exceeds the history max;
           * (scale, inv) of a role are cached per microbatch so an activation-checkpoint
             RECOMPUTE of the same microbatch reuses the identical scale (bitwise replay).
Microbatch and optimizer-step counters come from hooks installed by modeling/mlp_fp8glue.py's
install_fp8glue, called from parallel/wrap.py (a forward pre-hook on the root model and a global
optimizer post-step hook); the trainer is untouched.

LIMITATION (delayed mode only): this state (the history rings, their pointer, the seeded flags,
the counters) is NOT part of the trainer checkpoint.  `LayerScalingState.state_dict()` exists for
inspection, but nothing saves or restores it, so a resumed delayed-scaling run re-bootstraps every
role (the dynamic two-pass on its first cast after the restart) and is not bitwise the
uninterrupted run.  The default dynamic mode keeps no cross-step state and is unaffected.

API for a cast of role r with fp8 max X:
    p = state.plan(r, X)
    if p.two_pass:   run the amax kernel into p.amax  -> (scale, inv) = state.finalize(p)
                     then the cast kernel with scale (no amax emission)
    else:            run the ONE cast kernel with p.scale, emitting the amax into p.amax
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import triton
import triton.language as tl

from .common import scale_from_amax, scale_from_hist

MODE = os.environ.get("DM_FP8GLUE_SCALING", "dynamic")
if MODE not in ("dynamic", "delayed"):
    raise ValueError(f"DM_FP8GLUE_SCALING must be dynamic|delayed, got {MODE!r}")
HIST_LEN = int(os.environ.get("DM_FP8GLUE_HIST", "16"))
assert HIST_LEN > 0 and (HIST_LEN & (HIST_LEN - 1)) == 0, "DM_FP8GLUE_HIST must be a power of two"

_MB = [0]       # microbatch index (root-model forward pre-hook)
_EPOCH = [0]    # optimizer-step epoch (weights static within an epoch)


def begin_microbatch() -> int:
    _MB[0] += 1
    return _MB[0]


def bump_epoch() -> int:
    _EPOCH[0] += 1
    return _EPOCH[0]


def microbatch() -> int:
    return _MB[0]


def epoch() -> int:
    return _EPOCH[0]


@triton.jit
def _fp8glue_hist_push_kernel(hist_ptr, cur_ptr, R: tl.constexpr, L: tl.constexpr, slot):
    r = tl.arange(0, R)
    v = tl.load(cur_ptr + r)
    tl.store(hist_ptr + r * L + slot, v)
    tl.store(cur_ptr + r, tl.zeros([R], dtype=tl.float32))


@dataclass
class CastPlan:
    role: int
    fp8_max: float
    two_pass: bool
    amax: torch.Tensor                 # where the amax goes (fresh zeros | the role's cur slot)
    scale: Optional[torch.Tensor]      # set for one_pass (history) or after finalize
    inv: Optional[torch.Tensor]
    seed: bool = False                 # two_pass that seeds the delayed history


class LayerScalingState:
    """Scaling state of ONE decoder layer (all of its cast roles)."""

    def __init__(self, roles: List[str], device, mode: str = MODE, hist_len: int = HIST_LEN):
        self.mode = mode
        self.L = hist_len
        self.roles = {name: i for i, name in enumerate(roles)}
        R = len(roles)
        self.R_pad = 1 << max(0, (R - 1).bit_length())
        self.device = device
        self.hist = torch.zeros(self.R_pad, hist_len, dtype=torch.float32, device=device)
        self.cur = torch.zeros(self.R_pad, dtype=torch.float32, device=device)
        self.ptr = 0
        self.seeded = [False] * R
        self.last_mb = -1
        self.pending = False
        self.mb_cache: Dict[int, tuple] = {}   # role -> (mb, scale, inv)
        self.pushes = 0

    def enter_microbatch(self) -> None:
        """At the layer's forward entry: push the previous microbatch's amaxes once per
        microbatch (idempotent under checkpoint recompute of the same microbatch)."""
        if self.mode != "delayed":
            return
        mb = microbatch()
        if mb == self.last_mb:
            return
        if self.pending:
            _fp8glue_hist_push_kernel[(1,)](self.hist, self.cur, R=self.R_pad, L=self.L, slot=self.ptr)
            self.ptr = (self.ptr + 1) % self.L
            self.pushes += 1
        self.last_mb = mb
        self.pending = True
        self.mb_cache.clear()

    def plan(self, role: str, fp8_max: float) -> CastPlan:
        r = self.roles[role]
        if self.mode == "dynamic":
            return CastPlan(r, fp8_max, True, torch.zeros(1, dtype=torch.float32, device=self.device), None, None)
        slot = self.cur[r:r + 1]
        mb = microbatch()
        c = self.mb_cache.get(r)
        if c is not None and c[0] == mb:          # recompute of this microbatch: same scale
            return CastPlan(r, fp8_max, c[3], slot, c[1], c[2], seed=c[3])
        if not self.seeded[r]:
            return CastPlan(r, fp8_max, True, slot, None, None, seed=True)
        scale, inv = scale_from_hist(self.hist[r], fp8_max, slot)
        self.mb_cache[r] = (mb, scale, inv, False)
        return CastPlan(r, fp8_max, False, slot, scale, inv)

    def finalize(self, p: CastPlan):
        """After the amax pass of a two_pass plan: the scale from that amax."""
        if p.scale is not None:
            return p.scale, p.inv
        scale, inv = scale_from_amax(p.amax, p.fp8_max)
        p.scale, p.inv = scale, inv
        if self.mode == "delayed":
            self.seeded[p.role] = True
            self.mb_cache[p.role] = (microbatch(), scale, inv, True)
        return scale, inv

    def state_dict(self) -> dict:
        """A snapshot for inspection; there is no load method and no checkpoint saves it (see the
        LIMITATION note in the module docstring)."""
        return {"hist": self.hist.clone(), "cur": self.cur.clone(), "ptr": self.ptr,
                "seeded": list(self.seeded), "pushes": self.pushes}
