"""P1 (TE `gated_act_kernel` / `dgated_act_kernel`): SwiGLU fused with the fp8 casts.

Forward:  g, u (bf16 [M, F]) -> h8 = e4m3(silu(g) * u) rowwise [M, F] (+ twin [F, M]); h is NEVER
          materialized in bf16.  Dynamic: pass 1 = exact amax of the fp32 expression (order-free
          max), pass 2 = cast (+ twin).  Delayed: one pass with the history scale, amax side output.
Backward: dh, g, u -> dg8, du8 (e5m2) rowwise + twins (+ optionally h8^T recomputed with the
          forward's scale, the no-save variant), one read of dh/g/u per pass.

THE EXPRESSION IS THE COMPILED torchao MLP's, OP FOR OP (transcribed from its Inductor kernels
triton_red_fused_abs_max_mul_silu*, *_silu_silu_backward*):
    e = libdevice.exp(-g);  s = g / (1.0 + e)          (Triton fp32 '/', div.full.f32)
    h = s * u
    du = dh * s
    sig = 1.0 / (1.0 + e);  sig = sig * 1.0
    dg = ((dh * u) * sig) * (g * (1.0 - sig) + 1.0)
all in fp32 with NO intermediate bf16 rounding (Inductor keeps fused pointwise chains in fp32 when
torch._inductor.config.emulate_precision_casts is False, its default), then q = v * scale, clamp,
RNE to fp8.  What test_fp8glue.py checks: the amaxes against the IEEE fp32 torch expression
(ref_swiglu_fp32 / ref_swiglu_bwd_fp32 below) within a small fp32 ulp bound (Triton's '/' is not IEEE),
the codes against that expression's codes (same scale) to within one fp8 step, and the twins, the
recomputed h8^T and the fused dg|du buffer bitwise.  The Inductor kernels themselves are not
part of the shipped test.

TWIN_MODE / fallback: see cast_transpose.py (the two-store ptxas 12.8 issue).  The fused
multi-store kernels run only when M and F are both multiples of 16 (`fused_twin_ok`); the test
runs them at M = 16384 (F = 8192 and 2560), and each (shape, config) self-checks its first launch.
Other shapes take the fallback (rowwise stores, then transpose_fp8 on the codes).
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .cast_transpose import FUSED_TWIN, fused_twin_ok, transpose_fp8

_E4MAX = tl.constexpr(448.0)
_E5MAX = tl.constexpr(57344.0)
SELFCHECK = os.environ.get("DM_FP8GLUE_SELFCHECK", "1") == "1"
_checked: set = set()


@triton.jit
def _fp8glue_swiglu_fwd_amax_kernel(g_ptr, u_ptr, amax_ptr, M, F, sg, su,
                                    BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < F)
    g = tl.load(g_ptr + offs_m[:, None] * sg + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs_m[:, None] * su + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    e = libdevice.exp(-g)
    one = tl.full([1, 1], 1.0, tl.float32)
    den = e + one
    s = g / den
    h = s * u
    tl.atomic_max(amax_ptr, tl.max(tl.abs(h)))


@triton.jit
def _fp8glue_swiglu_fwd_cast_kernel(g_ptr, u_ptr, h8_ptr, h8t_ptr, scale_ptr, amax_ptr, M, F, sg, su,
                                    TWIN_MODE: tl.constexpr, EMIT_AMAX: tl.constexpr,
                                    BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < F)
    g = tl.load(g_ptr + offs_m[:, None] * sg + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs_m[:, None] * su + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    e = libdevice.exp(-g)
    one = tl.full([1, 1], 1.0, tl.float32)
    den = e + one
    s = g / den
    h = s * u
    if EMIT_AMAX:
        tl.atomic_max(amax_ptr, tl.max(tl.abs(h)))
    sc = tl.load(scale_ptr)
    q = h * sc
    q = tl.maximum(q, tl.full([1, 1], -_E4MAX, tl.float32))
    q = tl.minimum(q, tl.full([1, 1], _E4MAX, tl.float32))
    q8 = q.to(tl.float8e4nv)
    tl.store(h8_ptr + offs_m[:, None] * F + offs_n[None, :], q8, mask=mask)
    if TWIN_MODE == 1:
        maskT = (offs_n[:, None] < F) & (offs_m[None, :] < M)
        tl.store(h8t_ptr + offs_n[:, None].to(tl.int64) * M + offs_m[None, :], tl.trans(q8), mask=maskT)
    if TWIN_MODE == 2:
        tl.store(h8t_ptr + offs_n[None, :].to(tl.int64) * M + offs_m[:, None], q8, mask=mask)


@triton.jit
def _fp8glue_swiglu_bwd_amax_kernel(g_ptr, u_ptr, dh_ptr, amax_dg_ptr, amax_du_ptr, M, F, sg, su, sdh,
                                    BM: tl.constexpr, BN: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < F)
    g = tl.load(g_ptr + offs_m[:, None] * sg + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs_m[:, None] * su + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    dh = tl.load(dh_ptr + offs_m[:, None] * sdh + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    e = libdevice.exp(-g)
    one = tl.full([1, 1], 1.0, tl.float32)
    den = e + one
    s = g / den
    du = dh * s
    sig = one / den
    sig = sig * one
    dg = ((dh * u) * sig) * (g * (one - sig) + one)
    tl.atomic_max(amax_du_ptr, tl.max(tl.abs(du)))
    tl.atomic_max(amax_dg_ptr, tl.max(tl.abs(dg)))


@triton.jit
def _fp8glue_swiglu_bwd_cast_kernel(g_ptr, u_ptr, dh_ptr, dg8_ptr, dg8t_ptr, du8_ptr, du8t_ptr, h8t_ptr,
                                    scale_dg_ptr, scale_du_ptr, scale_h_ptr, amax_dg_ptr, amax_du_ptr,
                                    M, F, sg, su, sdh, ROW_STRIDE_OUT, COL_OFF_DU,
                                    HT_MODE: tl.constexpr, TWIN_MODE: tl.constexpr, EMIT_AMAX: tl.constexpr,
                                    BM: tl.constexpr, BN: tl.constexpr):
    """dg8/du8 rowwise are written with row stride ROW_STRIDE_OUT (== F for two separate
    tensors, == 2F with COL_OFF_DU == F for one [M, 2F] buffer holding [dg8 | du8]).
    HT_MODE: 0 none | 1/2 h8^T twin (like TWIN_MODE) | 3 h8 ROWWISE into h8t_ptr (fallback)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < F)
    maskT = (offs_n[:, None] < F) & (offs_m[None, :] < M)
    g = tl.load(g_ptr + offs_m[:, None] * sg + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + offs_m[:, None] * su + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    dh = tl.load(dh_ptr + offs_m[:, None] * sdh + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    e = libdevice.exp(-g)
    one = tl.full([1, 1], 1.0, tl.float32)
    den = e + one
    s = g / den
    if HT_MODE != 0:
        h = s * u
        sch = tl.load(scale_h_ptr)
        qh = h * sch
        qh = tl.maximum(qh, tl.full([1, 1], -_E4MAX, tl.float32))
        qh = tl.minimum(qh, tl.full([1, 1], _E4MAX, tl.float32))
        qh8 = qh.to(tl.float8e4nv)
        if HT_MODE == 1:
            tl.store(h8t_ptr + offs_n[:, None].to(tl.int64) * M + offs_m[None, :], tl.trans(qh8), mask=maskT)
        if HT_MODE == 2:
            tl.store(h8t_ptr + offs_n[None, :].to(tl.int64) * M + offs_m[:, None], qh8, mask=mask)
        if HT_MODE == 3:
            tl.store(h8t_ptr + offs_m[:, None] * F + offs_n[None, :], qh8, mask=mask)
    du = dh * s
    sig = one / den
    sig = sig * one
    dg = ((dh * u) * sig) * (g * (one - sig) + one)
    if EMIT_AMAX:
        tl.atomic_max(amax_du_ptr, tl.max(tl.abs(du)))
        tl.atomic_max(amax_dg_ptr, tl.max(tl.abs(dg)))
    scu = tl.load(scale_du_ptr)
    qu = du * scu
    qu = tl.maximum(qu, tl.full([1, 1], -_E5MAX, tl.float32))
    qu = tl.minimum(qu, tl.full([1, 1], _E5MAX, tl.float32))
    qu8 = qu.to(tl.float8e5)
    scg = tl.load(scale_dg_ptr)
    qg = dg * scg
    qg = tl.maximum(qg, tl.full([1, 1], -_E5MAX, tl.float32))
    qg = tl.minimum(qg, tl.full([1, 1], _E5MAX, tl.float32))
    qg8 = qg.to(tl.float8e5)
    tl.store(dg8_ptr + offs_m[:, None] * ROW_STRIDE_OUT + offs_n[None, :], qg8, mask=mask)
    tl.store(du8_ptr + offs_m[:, None] * ROW_STRIDE_OUT + COL_OFF_DU + offs_n[None, :], qu8, mask=mask)
    if TWIN_MODE == 1:
        tl.store(dg8t_ptr + offs_n[:, None].to(tl.int64) * M + offs_m[None, :], tl.trans(qg8), mask=maskT)
        tl.store(du8t_ptr + offs_n[:, None].to(tl.int64) * M + offs_m[None, :], tl.trans(qu8), mask=maskT)
    if TWIN_MODE == 2:
        tl.store(dg8t_ptr + offs_n[None, :].to(tl.int64) * M + offs_m[:, None], qg8, mask=mask)
        tl.store(du8t_ptr + offs_n[None, :].to(tl.int64) * M + offs_m[:, None], qu8, mask=mask)


# launch configs: set by a micro-bench at the training shapes; test_fp8glue.py runs them there (twins bitwise)
CFG_FWD = {"BM": 64, "BN": 128, "num_warps": 4, "twin_mode": 2}
CFG_BWD = {"BM": 32, "BN": 128, "num_warps": 4, "twin_mode": 2}
CFG_AMAX = {"BM": 128, "BN": 128, "num_warps": 8}
CFG_ROW = {"BM": 128, "BN": 128, "num_warps": 8}


def _2d(x: torch.Tensor):
    assert x.dim() == 2 and x.stride(1) == 1, "expects [M, F] with unit column stride"
    return x.shape[0], x.shape[1], x.stride(0)


def _check(tag, row, twin):
    if not SELFCHECK or tag in _checked:
        return
    _checked.add(tag)
    if not torch.equal(twin.view(torch.uint8), row.t().contiguous().view(torch.uint8)):
        raise RuntimeError(f"[fp8glue] SELF-CHECK FAILED for {tag}: fused twin store wrong on this toolchain; "
                           "set DM_FP8GLUE_FUSED_TWIN=0")
    print(f"[fp8glue] self-check OK: {tag}", flush=True)


@torch.no_grad()
def swiglu_fwd_amax(g: torch.Tensor, u: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    M, F, sg = _2d(g)
    _, _, su = _2d(u)
    if out is None:
        out = torch.zeros(1, dtype=torch.float32, device=g.device)
    c = CFG_AMAX
    grid = (triton.cdiv(M, c["BM"]), triton.cdiv(F, c["BN"]))
    _fp8glue_swiglu_fwd_amax_kernel[grid](g, u, out, M, F, sg, su, BM=c["BM"], BN=c["BN"],
                                          num_warps=c["num_warps"])
    return out


@torch.no_grad()
def swiglu_fwd_cast(g: torch.Tensor, u: torch.Tensor, scale: torch.Tensor, *, want_twin: bool = True,
                    amax_out: Optional[torch.Tensor] = None, cfg: Optional[dict] = None):
    M, F, sg = _2d(g)
    _, _, su = _2d(u)
    h8 = torch.empty(M, F, dtype=torch.float8_e4m3fn, device=g.device)
    h8t = torch.empty(F, M, dtype=torch.float8_e4m3fn, device=g.device) if want_twin else None
    fused = want_twin and FUSED_TWIN and fused_twin_ok(M, F) and (cfg is None or cfg.get("twin_mode", 0) != 0)
    c = cfg if cfg is not None else (CFG_FWD if fused else CFG_ROW)
    tm = c.get("twin_mode", 2) if fused else 0
    grid = (triton.cdiv(M, c["BM"]), triton.cdiv(F, c["BN"]))
    _fp8glue_swiglu_fwd_cast_kernel[grid](
        g, u, h8, h8t if h8t is not None else h8, scale, amax_out if amax_out is not None else scale,
        M, F, sg, su, TWIN_MODE=tm, EMIT_AMAX=amax_out is not None,
        BM=c["BM"], BN=c["BN"], num_warps=c["num_warps"])
    if want_twin and not fused:
        transpose_fp8(h8, out=h8t)
    elif fused:
        _check(("swiglu_fwd", M, F, c["BM"], c["BN"], c["num_warps"], tm), h8, h8t)
    return h8, h8t


@torch.no_grad()
def swiglu_bwd_amax(g, u, dh, amax_dg: Optional[torch.Tensor] = None, amax_du: Optional[torch.Tensor] = None):
    M, F, sg = _2d(g)
    _, _, su = _2d(u)
    _, _, sdh = _2d(dh)
    if amax_dg is None:
        amax_dg = torch.zeros(1, dtype=torch.float32, device=g.device)
    if amax_du is None:
        amax_du = torch.zeros(1, dtype=torch.float32, device=g.device)
    c = CFG_AMAX
    grid = (triton.cdiv(M, c["BM"]), triton.cdiv(F, c["BN"]))
    _fp8glue_swiglu_bwd_amax_kernel[grid](g, u, dh, amax_dg, amax_du, M, F, sg, su, sdh,
                                          BM=c["BM"], BN=c["BN"], num_warps=c["num_warps"])
    return amax_dg, amax_du


@torch.no_grad()
def swiglu_bwd_cast(g, u, dh, scale_dg, scale_du, *, scale_h: Optional[torch.Tensor] = None,
                    amax_dg: Optional[torch.Tensor] = None, amax_du: Optional[torch.Tensor] = None,
                    fused_dgdu: bool = False, cfg: Optional[dict] = None):
    """Returns (dg8, dg8t, du8, du8t, h8t_or_None, dgdu8 buffer).  With fused_dgdu the rowwise
    codes land in ONE [M, 2F] e5m2 buffer ([dg8 | du8]) and dg8/du8 are column views of it."""
    M, F, sg = _2d(g)
    _, _, su = _2d(u)
    _, _, sdh = _2d(dh)
    dev = g.device
    if fused_dgdu:
        buf = torch.empty(M, 2 * F, dtype=torch.float8_e5m2, device=dev)
        dg8, du8 = buf[:, :F], buf[:, F:]
        rs, coff = 2 * F, F
    else:
        buf = None
        dg8 = torch.empty(M, F, dtype=torch.float8_e5m2, device=dev)
        du8 = torch.empty(M, F, dtype=torch.float8_e5m2, device=dev)
        rs, coff = F, 0
    dg8t = torch.empty(F, M, dtype=torch.float8_e5m2, device=dev)
    du8t = torch.empty(F, M, dtype=torch.float8_e5m2, device=dev)
    want_ht = scale_h is not None
    fused = FUSED_TWIN and fused_twin_ok(M, F) and (cfg is None or cfg.get("twin_mode", 0) != 0)
    c = cfg if cfg is not None else (CFG_BWD if fused else CFG_ROW)
    tm = c.get("twin_mode", 2) if fused else 0
    if want_ht:
        h8t = torch.empty(F, M, dtype=torch.float8_e4m3fn, device=dev)
        h8tmp = None if fused else torch.empty(M, F, dtype=torch.float8_e4m3fn, device=dev)
        ht_mode = tm if fused else 3
        ht_ptr = h8t if fused else h8tmp
    else:
        h8t, h8tmp, ht_mode, ht_ptr = None, None, 0, dg8t
    emit = amax_dg is not None
    if emit:
        assert amax_du is not None
    grid = (triton.cdiv(M, c["BM"]), triton.cdiv(F, c["BN"]))
    _fp8glue_swiglu_bwd_cast_kernel[grid](
        g, u, dh, dg8 if buf is None else buf, dg8t, du8 if buf is None else buf, du8t, ht_ptr,
        scale_dg, scale_du, scale_h if scale_h is not None else scale_dg,
        amax_dg if emit else scale_dg, amax_du if emit else scale_du,
        M, F, sg, su, sdh, rs, coff,
        HT_MODE=ht_mode, TWIN_MODE=tm, EMIT_AMAX=emit, BM=c["BM"], BN=c["BN"], num_warps=c["num_warps"])
    if not fused:
        transpose_fp8(dg8.contiguous() if buf is not None else dg8, out=dg8t)
        transpose_fp8(du8.contiguous() if buf is not None else du8, out=du8t)
        if want_ht:
            transpose_fp8(h8tmp, out=h8t)
    else:
        _check(("swiglu_bwd", M, F, c["BM"], c["BN"], c["num_warps"], tm, want_ht, fused_dgdu),
               dg8.contiguous() if buf is not None else dg8, dg8t)
    return dg8, dg8t, du8, du8t, h8t, buf


# --------------------------------------------------------------------------- torch reference
@torch.no_grad()
def ref_swiglu_fp32(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """The compiled-torchao expression in torch fp32 (IEEE exp/div: the 'ulp-class' reference)."""
    g32, u32 = g.float(), u.float()
    return (g32 / (1.0 + torch.exp(-g32))) * u32


@torch.no_grad()
def ref_swiglu_bwd_fp32(g, u, dh):
    """(dg, du) of the same expression in torch fp32 (IEEE exp/div), op for op as the backward kernels."""
    g32, u32, dh32 = g.float(), u.float(), dh.float()
    e = torch.exp(-g32)
    den = 1.0 + e
    s = g32 / den
    sig = 1.0 / den
    du = dh32 * s
    dg = ((dh32 * u32) * sig) * (g32 * (1.0 - sig) + 1.0)
    return dg, du
