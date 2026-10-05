"""Shared pieces: fp8 constants, the Fp8Pack container, the scale-rule kernels.

THE SCALE RULE (dynamic mode) mirrors the COMPILED torchao Float8Linear byte for byte (the
Inductor kernel triton_per_fused__to_copy_abs_clamp_max_mul_reciprocal_* of the compiled fp8 MLP):
    scale     = fp32( (fp32(1.0) / max(fp64(amax), 1e-12)) * fp64(FP8_MAX) )
    inv_scale = fp32(1.0) / scale                       (Triton fp32 '/', i.e. div.full.f32)
Eager torchao writes the same thing as fp32(FP8_MAX / clamp(amax_f64, 1e-12)) and
scale.reciprocal() (IEEE); the two agree except at fp64->fp32 rounding boundaries (never
observed in test_fp8glue.py, which asserts the eager rule and probes the compiled one).
The cast is  x.to(fp32) * scale, clamp to +-FP8_MAX, .to(fp8) (RNE), exactly torchao's
_ToFloat8ConstrFunc / to_fp8_saturated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0
E5M2_MAX = 57344.0
EPS = 1e-12   # torchao.float8.float8_utils.EPS


def fp8_max_of(dtype: torch.dtype) -> float:
    if dtype == torch.float8_e4m3fn:
        return E4M3_MAX
    if dtype == torch.float8_e5m2:
        return E5M2_MAX
    raise ValueError(f"unsupported fp8 dtype {dtype}")


@dataclass
class Fp8Pack:
    """One fp8-cast operand in both GEMM layouts.
    data: fp8 [M, N] row-major (the 'rowwise' operand: mat1 of a forward/dgrad GEMM)
    twin: fp8 [N, M] row-major (the 'columnwise' operand: twin.t() is the column-major [M, N]
          that torch._scaled_mm wants as mat2, i.e. the wgrad operand); may be None
    scale / inv_scale: fp32 [1] tensors (torchao convention: x8 = x * scale; _scaled_mm takes inv)
    amax: fp32 [1] (the exact max|x| of THIS tensor; in delayed mode it is the side output that
          feeds the history, NOT the value the scale was derived from)
    carrier: an optional bf16 tensor that autograd routes the gradient through (see ops.py)
    """
    data: torch.Tensor
    twin: Optional[torch.Tensor]
    scale: torch.Tensor
    inv_scale: torch.Tensor
    amax: torch.Tensor
    fp8_dtype: torch.dtype
    lead_shape: tuple
    carrier: Optional[torch.Tensor] = None

    @property
    def M(self) -> int:
        return self.data.shape[0]

    @property
    def N(self) -> int:
        return self.data.shape[1]


# ----------------------------------------------------------------------------- scale kernels
# THE COMPILED-GRAPH AMAX QUIRK (found when gating against the compiled torchao MLP): in the
# compiled torchao MLP the amax scalar is a bf16-TYPED graph value (torchao computes
# torch.max(torch.abs(h)) on the bf16-typed h), so Inductor stores it as bf16 and the CAST
# kernel re-derives its scale from the bf16-ROUNDED amax, while the `_scaled_mm` inverse scale
# comes from the fp32 amax inside the scale kernel: quant and dequant use scales that differ by
# up to 2^-9 for the fp32-chain operands (h, dg, du; a no-op for bf16-valued tensors whose amax
# is exactly representable). Eager torchao is self-consistent. QUANT_BF16=True mirrors the
# compiled graph (env DM_FP8GLUE_AMAX_BF16=1: bitwise vs the compiled torchao MLP); the default
# uses ONE consistent fp32-amax scale (torchao's intended semantics).
AMAX_BF16 = os.environ.get("DM_FP8GLUE_AMAX_BF16", "0") == "1"


@triton.jit
def _scale_from_amax_kernel(amax_ptr, scale_ptr, inv_ptr, FP8_MAX: tl.constexpr, QUANT_BF16: tl.constexpr):
    """Inductor's exact expression (see module docstring)."""
    a = tl.load(amax_ptr).to(tl.float32)
    one = tl.full([], 1.0, tl.float32)
    a64 = a.to(tl.float64)
    a64 = tl.maximum(a64, tl.full([], 1e-12, tl.float64))
    r = one / a64                      # fp64 division (div.rn.f64)
    s64 = r * tl.full([], FP8_MAX, tl.float64)
    s = s64.to(tl.float32)
    inv = one / s                      # fp32 Triton division = div.full.f32 (what Inductor emits)
    if QUANT_BF16:
        ab = a.to(tl.bfloat16).to(tl.float32)
        b64 = ab.to(tl.float64)
        b64 = tl.maximum(b64, tl.full([], 1e-12, tl.float64))
        rb = one / b64
        s = (rb * tl.full([], FP8_MAX, tl.float64)).to(tl.float32)
    tl.store(scale_ptr, s)
    tl.store(inv_ptr, inv)


@triton.jit
def _scale_from_hist_kernel(hist_ptr, cur_ptr, L: tl.constexpr, scale_ptr, inv_ptr, FP8_MAX: tl.constexpr):
    """Delayed scaling: scale from max(history, cur) with the same arithmetic as the dynamic rule.
    `cur` (the role's not-yet-pushed amax) is normally 0 at the cast (the push happens at the
    block entry); folding it in keeps the scale sane if a push was skipped and reproduces the
    bootstrap scale under an activation-checkpoint recompute."""
    offs = tl.arange(0, L)
    h = tl.load(hist_ptr + offs)
    c = tl.load(cur_ptr)
    a = tl.maximum(tl.max(h, axis=0), c)   # order-free
    a64 = a.to(tl.float64)
    a64 = tl.maximum(a64, tl.full([], 1e-12, tl.float64))
    one = tl.full([], 1.0, tl.float32)
    r = one / a64
    s64 = r * tl.full([], FP8_MAX, tl.float64)
    s = s64.to(tl.float32)
    inv = one / s
    tl.store(scale_ptr, s)
    tl.store(inv_ptr, inv)


@triton.jit
def _recip_kernel(x_ptr, y_ptr):
    """fp32 reciprocal the way the compiled torchao graph does it (Triton '/', div.full.f32),
    for the WEIGHT scales that arrive precomputed with the fp8 all-gather."""
    x = tl.load(x_ptr).to(tl.float32)
    one = tl.full([], 1.0, tl.float32)
    tl.store(y_ptr, one / x)


def scale_from_amax(amax: torch.Tensor, fp8_max: float, quant_bf16: Optional[bool] = None):
    """amax: fp32 tensor with one element (device). Returns (scale, inv_scale), fp32 [1] each.
    scale is what the cast multiplies by; inv_scale is what torch._scaled_mm receives."""
    scale = torch.empty(1, dtype=torch.float32, device=amax.device)
    inv = torch.empty(1, dtype=torch.float32, device=amax.device)
    q = AMAX_BF16 if quant_bf16 is None else quant_bf16
    _scale_from_amax_kernel[(1,)](amax, scale, inv, FP8_MAX=fp8_max, QUANT_BF16=bool(q))
    return scale, inv


def scale_from_hist(hist: torch.Tensor, fp8_max: float, cur: Optional[torch.Tensor] = None):
    """hist: fp32 [L] (contiguous row of the ring); cur: fp32 [1] pending amax (0 if none).
    Returns (scale, inv_scale)."""
    L = hist.numel()
    assert (L & (L - 1)) == 0, "history length must be a power of two"
    scale = torch.empty(1, dtype=torch.float32, device=hist.device)
    inv = torch.empty(1, dtype=torch.float32, device=hist.device)
    if cur is None:
        cur = torch.zeros(1, dtype=torch.float32, device=hist.device)
    _scale_from_hist_kernel[(1,)](hist, cur, L, scale, inv, FP8_MAX=fp8_max)
    return scale, inv


def recip_like_inductor(scale: torch.Tensor) -> torch.Tensor:
    y = torch.empty(1, dtype=torch.float32, device=scale.device)
    _recip_kernel[(1,)](scale, y)
    return y


# ------------------------------------------------------------------- torch-side references
@torch.no_grad()
def ref_scale_torchao_eager(amax: torch.Tensor, fp8_max: float) -> torch.Tensor:
    """torchao.float8.float8_utils.amax_to_scale, verbatim arithmetic."""
    a = amax.to(torch.float64)
    return (fp8_max / torch.clamp(a, min=EPS)).to(torch.float32)


@torch.no_grad()
def ref_cast_torchao(x: torch.Tensor, scale: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    """torchao _ToFloat8ConstrFunc + to_fp8_saturated, verbatim."""
    mx = fp8_max_of(fp8_dtype)
    return (x.to(torch.float32) * scale).clamp(min=-mx, max=mx).to(fp8_dtype)
