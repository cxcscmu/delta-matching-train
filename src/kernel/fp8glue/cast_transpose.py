"""P4 (TE `cast_transpose`): bf16 -> fp8 ROWWISE + COLUMNWISE twin in ONE read, amax side output.

Input addressing is a general 4-D (B, T, H, D) view with arbitrary strides so the attention-side
gradients can be read in the op's native (B, H, T, D) layout (no permute copy): the logical
[M, N] matrix has m = b*T + t and n = h*D + d.  A plain row-major [M, N] tensor is B=1, H=1.
Tiles are [BM x BN] over (t, d) for a fixed (b, h).

TWIN STORE MODES (constexpr TWIN_MODE): 0 none | 1 `tl.trans` through shared memory | 2 the
Inductor idiom (untransposed register tile stored to transposed addresses).  WHY TWO: with the
bundled ptxas 12.8 (PTX 8.7) a kernel that stores BOTH layouts writes garbage into some twin
rows for certain (tile, shape) combinations (the SAME PTX assembled by CUDA 13.0's ptxas is
correct; single-store kernels are always correct).
test_fp8glue.py checks the launch configs below bitwise against torchao at the training shapes
(M = 16384, N in {1024, 2560, 4096, 8192}).  The fused two-store kernel runs only when M and N are
both multiples of 16 (`fused_twin_ok`); every other shape takes the two-kernel path (rowwise cast,
then `transpose_fp8` on the codes).  The first fused launch of each (shape, config) in a process
self-checks twin == rowwise.t() (DM_FP8GLUE_SELFCHECK=0 disables).

Numerics: q = fp32(x) * scale, clamp(+-FP8_MAX), RNE to fp8 == torchao's cast, elementwise, so
the codes are bitwise vs torchao for the same scale; the amax (fp32 atomic max of |x|) is exact
and order-free.  Modes: dynamic = amax pass + cast pass with the scale from that amax; delayed =
one cast pass with a history scale and EMIT_AMAX=True.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from .common import Fp8Pack, fp8_max_of, scale_from_amax

SELFCHECK = os.environ.get("DM_FP8GLUE_SELFCHECK", "1") == "1"
_checked: set = set()


@triton.jit
def _fp8glue_amax_kernel(x_ptr, amax_ptr, T, D, H, sb, st, sh, sd,
                         BM: tl.constexpr, BN: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_bh = tl.program_id(2)
    b = pid_bh // H
    h = pid_bh % H
    offs_t = pid_t * BM + tl.arange(0, BM)
    offs_d = pid_d * BN + tl.arange(0, BN)
    mask = (offs_t[:, None] < T) & (offs_d[None, :] < D)
    base = x_ptr + b.to(tl.int64) * sb + h.to(tl.int64) * sh
    x = tl.load(base + offs_t[:, None].to(tl.int64) * st + offs_d[None, :] * sd,
                mask=mask, other=0.0).to(tl.float32)
    m = tl.max(tl.abs(x))
    tl.atomic_max(amax_ptr, m)


@triton.jit
def _fp8glue_cast_transpose_kernel(x_ptr, out_ptr, outT_ptr, scale_ptr, amax_ptr,
                                   T, D, H, M, N, sb, st, sh, sd,
                                   FP8_MAX: tl.constexpr, WRITE_ROW: tl.constexpr,
                                   TWIN_MODE: tl.constexpr, EMIT_AMAX: tl.constexpr,
                                   BM: tl.constexpr, BN: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_bh = tl.program_id(2)
    b = pid_bh // H
    h = pid_bh % H
    offs_t = pid_t * BM + tl.arange(0, BM)
    offs_d = pid_d * BN + tl.arange(0, BN)
    mask = (offs_t[:, None] < T) & (offs_d[None, :] < D)
    base = x_ptr + b.to(tl.int64) * sb + h.to(tl.int64) * sh
    x = tl.load(base + offs_t[:, None].to(tl.int64) * st + offs_d[None, :] * sd,
                mask=mask, other=0.0).to(tl.float32)
    if EMIT_AMAX:
        tl.atomic_max(amax_ptr, tl.max(tl.abs(x)))
    s = tl.load(scale_ptr)
    q = x * s
    q = tl.maximum(q, tl.full([1, 1], -FP8_MAX, tl.float32))
    q = tl.minimum(q, tl.full([1, 1], FP8_MAX, tl.float32))
    q8 = q.to(out_ptr.dtype.element_ty)
    m_idx = (b * T + offs_t).to(tl.int64)
    n_idx = h * D + offs_d
    if WRITE_ROW:
        tl.store(out_ptr + m_idx[:, None] * N + n_idx[None, :], q8, mask=mask)
    if TWIN_MODE == 1:
        q8t = tl.trans(q8)
        maskT = (offs_d[:, None] < D) & (offs_t[None, :] < T)
        tl.store(outT_ptr + n_idx[:, None].to(tl.int64) * M + m_idx[None, :], q8t, mask=maskT)
    if TWIN_MODE == 2:
        tl.store(outT_ptr + n_idx[None, :].to(tl.int64) * M + m_idx[:, None], q8, mask=mask)


@triton.jit
def _fp8glue_transpose_kernel(x_ptr, y_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    """fp8 [M, N] row-major -> fp8 [N, M] row-major (single store: always correct)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], mask=mask)
    xt = tl.trans(x)
    maskT = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(y_ptr + offs_n[:, None].to(tl.int64) * M + offs_m[None, :], xt, mask=maskT)


# ----------------------------------------------------------------------------- launch config
# (BM, BN, num_warps, twin_mode): set by a micro-bench at the training shapes
# and checked bitwise there by kernel/fp8glue/test_fp8glue.py.
CFG_CAST = {"BM": 64, "BN": 128, "num_warps": 4, "twin_mode": 2}
CFG_CAST_ROW = {"BM": 128, "BN": 128, "num_warps": 8}
CFG_AMAX = {"BM": 128, "BN": 128, "num_warps": 8}
CFG_TRANS = {"BM": 128, "BN": 128, "num_warps": 4}


def _view4(x: torch.Tensor, D: Optional[int]) -> Tuple[int, int, int, int, int, int, int, int]:
    """Describe x as (B, T, H, D) with strides. x may be 2-D [M, N] (any strides), 3-D [B, T, N]
    or 4-D [B, T, H, D] (any strides). Returns (B, T, H, D, sb, st, sh, sd)."""
    if x.dim() == 2:
        M, N = x.shape
        sm, sn = x.stride()
        if D is None or D == N:
            return 1, M, 1, N, 0, sm, 0, sn
        assert N % D == 0
        return 1, M, N // D, D, 0, sm, D * sn, sn
    if x.dim() == 3:
        B, T, N = x.shape
        sb, st, sn = x.stride()
        if D is None or D == N:
            return B, T, 1, N, sb, st, 0, sn
        assert N % D == 0
        return B, T, N // D, D, sb, st, D * sn, sn
    if x.dim() == 4:
        B, T, H, Dd = x.shape
        sb, st, sh, sd = x.stride()
        return B, T, H, Dd, sb, st, sh, sd
    raise ValueError(f"cast_transpose: unsupported rank {x.dim()}")


def fused_twin_ok(M: int, N: int) -> bool:
    """The fused two-store kernel is used only when M % 16 == 0 and N % 16 == 0."""
    return M % 16 == 0 and N % 16 == 0


@torch.no_grad()
def amax_of(x: torch.Tensor, out: Optional[torch.Tensor] = None, D: Optional[int] = None) -> torch.Tensor:
    """Exact max|x| into `out` (fp32 [1], zero-initialised here unless given) via atomic max."""
    B, T, H, Dd, sb, st, sh, sd = _view4(x, D)
    if out is None:
        out = torch.zeros(1, dtype=torch.float32, device=x.device)
    c = CFG_AMAX
    grid = (triton.cdiv(T, c["BM"]), triton.cdiv(Dd, c["BN"]), B * H)
    _fp8glue_amax_kernel[grid](x, out, T, Dd, H, sb, st, sh, sd,
                               BM=c["BM"], BN=c["BN"], num_warps=c["num_warps"])
    return out


@torch.no_grad()
def transpose_fp8(x8: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """fp8 [M, N] contiguous -> fp8 [N, M] contiguous."""
    assert x8.dim() == 2 and x8.is_contiguous()
    M, N = x8.shape
    if out is None:
        out = torch.empty(N, M, dtype=x8.dtype, device=x8.device)
    c = CFG_TRANS
    grid = (triton.cdiv(M, c["BM"]), triton.cdiv(N, c["BN"]))
    _fp8glue_transpose_kernel[grid](x8, out, M, N, BM=c["BM"], BN=c["BN"], num_warps=c["num_warps"])
    return out


def _selfcheck(tag, out, outT):
    if not SELFCHECK or tag in _checked:
        return
    _checked.add(tag)
    ref = out.t().contiguous()
    if not torch.equal(outT.view(torch.uint8), ref.view(torch.uint8)):
        raise RuntimeError(f"[fp8glue] SELF-CHECK FAILED for {tag}: the fused twin store is wrong on this "
                           "toolchain (see cast_transpose.py docstring); set DM_FP8GLUE_FUSED_TWIN=0")
    print(f"[fp8glue] self-check OK: {tag}", flush=True)


FUSED_TWIN = os.environ.get("DM_FP8GLUE_FUSED_TWIN", "1") == "1"


@torch.no_grad()
def cast_with_scale(x: torch.Tensor, scale: torch.Tensor, fp8_dtype: torch.dtype, *,
                    want_row: bool = True, want_twin: bool = True,
                    amax_out: Optional[torch.Tensor] = None, D: Optional[int] = None,
                    out: Optional[torch.Tensor] = None, outT: Optional[torch.Tensor] = None,
                    cfg: Optional[dict] = None):
    """One pass: fp8 rowwise [M, N] (+ twin [N, M]) with a GIVEN scale; optional amax side output
    (atomic max into amax_out, which must be zeroed by the caller)."""
    B, T, H, Dd, sb, st, sh, sd = _view4(x, D)
    M, N = B * T, H * Dd
    fused = want_twin and FUSED_TWIN and fused_twin_ok(M, N) and (cfg is None or cfg.get("twin_mode", 0) != 0)
    if want_twin and not fused:
        want_row_eff = True
    else:
        want_row_eff = want_row
    if want_row_eff and out is None:
        out = torch.empty(M, N, dtype=fp8_dtype, device=x.device)
    if want_twin and outT is None:
        outT = torch.empty(N, M, dtype=fp8_dtype, device=x.device)
    dummy = out if out is not None else outT
    c = cfg if cfg is not None else (CFG_CAST if fused else CFG_CAST_ROW)
    twin_mode = c.get("twin_mode", 2) if fused else 0
    grid = (triton.cdiv(T, c["BM"]), triton.cdiv(Dd, c["BN"]), B * H)
    _fp8glue_cast_transpose_kernel[grid](
        x, out if out is not None else dummy, outT if outT is not None else dummy, scale,
        amax_out if amax_out is not None else scale,
        T, Dd, H, M, N, sb, st, sh, sd,
        FP8_MAX=fp8_max_of(fp8_dtype), WRITE_ROW=bool(want_row_eff), TWIN_MODE=twin_mode,
        EMIT_AMAX=amax_out is not None, BM=c["BM"], BN=c["BN"], num_warps=c["num_warps"])
    if want_twin and not fused:
        transpose_fp8(out, out=outT)
    elif fused and want_row_eff:
        _selfcheck(("cast", M, N, str(fp8_dtype), c["BM"], c["BN"], c["num_warps"], twin_mode), out, outT)
    return (out if want_row else None), outT


@torch.no_grad()
def cast_dynamic(x: torch.Tensor, fp8_dtype: torch.dtype, *, want_twin: bool = True,
                 want_row: bool = True, D: Optional[int] = None,
                 amax: Optional[torch.Tensor] = None) -> Fp8Pack:
    """torchao-bitwise dynamic tensorwise cast in two passes (amax, then cast+twin)."""
    if amax is None:
        amax = amax_of(x, D=D)
    scale, inv = scale_from_amax(amax, fp8_max_of(fp8_dtype))
    out, outT = cast_with_scale(x, scale, fp8_dtype, want_row=want_row, want_twin=want_twin, D=D)
    lead = tuple(x.shape[:-1]) if x.dim() in (2, 3) else (x.shape[0], x.shape[1])
    return Fp8Pack(out, outT, scale, inv, amax, fp8_dtype, lead)


# ----------------------------------------------------------------------------- fused 3-way add
@triton.jit
def _fp8glue_add3_kernel(a_ptr, b_ptr, c_ptr, out_ptr, n, ORDER: tl.constexpr, BLOCK: tl.constexpr):
    """out = a + b + c summed ONCE in fp32 (a fixed association) and rounded once to bf16: the
    dgrad sum into x of the shared-x8 q/k/v projection, as Inductor fuses it in the dm mixer's
    compiled shared q/k/v function (pairwise bf16 adds would round twice: the accumulation-order class)."""
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    c = tl.load(c_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if ORDER == 0:
        s = (a + b) + c
    elif ORDER == 1:
        s = (a + c) + b
    else:
        s = (b + c) + a
    tl.store(out_ptr + offs, s.to(out_ptr.dtype.element_ty), mask=mask)


ADD3_ORDER = int(os.environ.get("DM_FP8GLUE_ADD3_ORDER", "0"))


@torch.no_grad()
def add3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, order: Optional[int] = None) -> torch.Tensor:
    assert a.shape == b.shape == c.shape and a.is_contiguous() and b.is_contiguous() and c.is_contiguous()
    out = torch.empty_like(a)
    n = a.numel()
    BLOCK = 4096
    _fp8glue_add3_kernel[(triton.cdiv(n, BLOCK),)](a, b, c, out, n, ORDER=ADD3_ORDER if order is None else order,
                                                   BLOCK=BLOCK, num_warps=4)
    return out
