"""The fused Q producer of the dm mixer: partial RoPE + block e4m3 quantize for Q (Triton).

Replaces, in ONE kernel launch, the reference composition for the Q operand of the fp8
attention kernel:

    eager partial RoPE on q          (pretrain/modeling/layers.py apply_rotary,
                                      _split_rot: rotate lanes 0..RD-1 in fp32
                                      from bf16 q and bf16-VALUED cos/sin, round
                                      to bf16, pass lanes RD..D-1 through)
    qs32 = (q.float() * sm) [BH,N,D] (the reference quantizer's input,
                                      sm = D**-0.5, exact 2^-4 at D=256)
    Q8, Q8s = block_quant(qs32, BM)  (the reference block quantizer: per [BM, D]
                                      block absmax, s = max(amax, 1e-30)/448 fp32
                                      division, x/s RN divide, clamp +-448, RTNE
                                      e4m3 cast; BM = 64 at head_dim 256, 128 at 128)

BITWISE CONTRACT: the last bf16
intermediate is the RoPE output. This kernel therefore:
  1. upcasts the rotated lanes to fp32 and computes xr*cos + rotate_half(xr)*sin
     with EXPLICITLY ROUNDED libdevice mul_rn/add_rn (torch eager runs mul, mul,
     add as three separate RN kernels; a plain Triton `a*c + b*s` may contract
     into FMA and flip the low bit),
  2. ROUNDS THE ROTATED LANES TO BF16 IN-REGISTER (`.to(tl.bfloat16)`, RTNE,
     same as torch's `.to(q.dtype)`); the tail lanes stay the loaded bf16,
  3. upcasts all D lanes to fp32, multiplies by sm (the reference quantizer's own fp32
     multiply; exact at D=256),
  4. block absmax over [BLK, D]; s = max(amax, 1e-30) / 448.0 (the same scalar
     `/` lowering the reference quantizer delta_matching/quant.py uses),
  5. codes = div_rn(x, s), clamp +-448, `.to(tl.float8e4nv)` (RTNE saturating),
     mirroring delta_matching/quant.py verbatim.

K is NOT quantized here (smooth-K needs a sequence-global mean before any row block can
quantize; delta_matching.producers does it). The k RoPE reuses this kernel's rotated-output
path (m13_rope_bwd.k_rope_fused, codes discarded). The YaRN temperature is NOT folded
here: under rope_scaling the dm mixer multiplies the pre-RoPE q by mscale^2 before this
kernel runs (modeling/dm_attention.py), so the kernel sees the tempered q.

delta_matching is imported only for the default block size.

Outputs: Q8 e4m3 [B*H, N, D] dense C-contiguous, Q8s fp32 [B*H, N//BLK] dense,
exactly the flat-offset layout the kernel consumes (`bh*N*D + n*D + d`, no strides).
Optionally also the rotated bf16 q [B, H, N, D] (contiguous), bitwise equal to
apply_rotary's q output, for autograd / anything downstream that still wants it.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tld

E4M3_MAX = tl.constexpr(448.0)


@triton.jit
def _fneg(x):
    """IEEE sign-bit flip. Triton's unary `-x` lowers as (0 - x), which maps +0.0 to +0.0;
    torch's neg (rotate_half's `-x2`) flips the bit, giving -0.0. On an all-zero row the two
    disagree in the sign of zero, `x/s` then keeps that sign, and the e4m3 code differs (0x00
    vs 0x80), as observed on an all-zero block. The xor is exact for every value, zero
    included."""
    return (x.to(tl.int32, bitcast=True) ^ -2147483648).to(tl.float32, bitcast=True)


@triton.jit
def _ropeq_kernel(Q, COS, SIN, Q8, Q8S, QROT, N, H,
                  sqb, sqh, sqn,          # q element strides (B, H, N); lane stride == 1
                  scb, sct,               # cos element strides (B, T); lane stride == 1
                  ssb, sst,               # sin element strides (B, T); lane stride == 1
                  SM: tl.constexpr, BLK: tl.constexpr, D: tl.constexpr,
                  RD: tl.constexpr, HALF: tl.constexpr,
                  WANT_QROT: tl.constexpr):
    """One program per (row-block, bh). Rows are token positions n..n+BLK-1."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    rm = pid_m * BLK + tl.arange(0, BLK)
    rd = tl.arange(0, D)
    r1 = tl.arange(0, HALF)

    qrow = Q + b.to(tl.int64) * sqb + h.to(tl.int64) * sqh \
        + rm.to(tl.int64)[:, None] * sqn
    # rotated lanes: x1 = lanes 0..HALF-1, x2 = lanes HALF..RD-1 (rotate_half pairs)
    x1 = tl.load(qrow + r1[None, :]).to(tl.float32)
    x2 = tl.load(qrow + (HALF + r1)[None, :]).to(tl.float32)
    crow = COS + b.to(tl.int64) * scb + rm.to(tl.int64)[:, None] * sct
    srow = SIN + b.to(tl.int64) * ssb + rm.to(tl.int64)[:, None] * sst
    c1 = tl.load(crow + r1[None, :]).to(tl.float32)
    c2 = tl.load(crow + (HALF + r1)[None, :]).to(tl.float32)
    s1 = tl.load(srow + r1[None, :]).to(tl.float32)
    s2 = tl.load(srow + (HALF + r1)[None, :]).to(tl.float32)

    # torch eager: t1 = xr*cos; t2 = rotate_half(xr)*sin = [-x2, x1]*sin; out = t1+t2,
    # three separate fp32 RN ops. mul_rn/add_rn are libdevice explicitly-rounded
    # intrinsics: never FMA-contracted, bitwise the same three roundings.
    o1 = tld.add_rn(tld.mul_rn(x1, c1), tld.mul_rn(_fneg(x2), s1)).to(tl.bfloat16)
    o2 = tld.add_rn(tld.mul_rn(x2, c2), tld.mul_rn(x1, s2)).to(tl.bfloat16)

    # full row (tail lanes RD..D-1 pass through as loaded bf16)
    x_all = tl.load(qrow + rd[None, :])
    tail = rd[None, :] >= RD

    v1 = o1.to(tl.float32) * SM
    v2 = o2.to(tl.float32) * SM
    vt = x_all.to(tl.float32) * SM
    amax = tl.maximum(tl.maximum(tl.max(tl.abs(v1)), tl.max(tl.abs(v2))),
                      tl.max(tl.where(tail, tl.abs(vt), 0.0)))
    s = tl.maximum(amax, 1e-30) / E4M3_MAX
    tl.store(Q8S + pid_bh.to(tl.int64) * (N // BLK) + pid_m, s)

    obase = pid_bh.to(tl.int64) * N * D + rm.to(tl.int64)[:, None] * D
    # div_rn: IEEE RN division; a plain `/` may lower to reciprocal-multiply and
    # flip ~4e-4 of codes (the same lesson is recorded in delta_matching/quant.py).
    sden = tl.full([1, 1], 1.0, tl.float32) * s
    q1 = tl.math.div_rn(v1, sden)
    q1 = tl.minimum(tl.maximum(q1, -E4M3_MAX), E4M3_MAX).to(tl.float8e4nv)
    tl.store(Q8 + obase + r1[None, :], q1)
    q2 = tl.math.div_rn(v2, sden)
    q2 = tl.minimum(tl.maximum(q2, -E4M3_MAX), E4M3_MAX).to(tl.float8e4nv)
    tl.store(Q8 + obase + (HALF + r1)[None, :], q2)
    qt = tl.math.div_rn(vt, sden)
    qt = tl.minimum(tl.maximum(qt, -E4M3_MAX), E4M3_MAX).to(tl.float8e4nv)
    tl.store(Q8 + obase + rd[None, :], qt, mask=tail)

    if WANT_QROT:
        tl.store(QROT + obase + r1[None, :], o1)
        tl.store(QROT + obase + (HALF + r1)[None, :], o2)
        tl.store(QROT + obase + rd[None, :], x_all, mask=tail)


def ropeq_q(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
            blk: int | None = None, want_qrot: bool = False):
    """Fused partial-RoPE + sm-bake + [blk, D] e4m3 quantize for Q.

    q:        bf16 [B, H, N, D], post-q_norm PRE-RoPE, lane stride 1 (any dense
              batch/head/row strides, e.g. the mixer's transpose(1, 2) layout).
    cos, sin: bf16 [B, N, RD], the backbone's already-bf16 cos/sin (cast once in
              Transformer.forward, modeling/model.py).
    blk:      quant row-block; default blocks(D)[0] of the kernel's block-scale
              contract (64 at D=256, 128 at D=128).
    Returns (Q8 e4m3 [B*H, N, D], Q8s fp32 [B*H, N//blk]) and, when want_qrot,
    additionally the rotated bf16 q [B, H, N, D] (bitwise apply_rotary's output).
    """
    B, H, N, D = q.shape
    RD = cos.shape[-1]
    if blk is None:
        from delta_matching.core import blocks   # the block-scale contract per head dim: (64, 32) at 256, (128, 64) at 128
        blk = blocks(D)[0]
    assert q.dtype == torch.bfloat16, "q must be bf16 (post-q_norm, pre-RoPE)"
    assert cos.dtype == torch.bfloat16 and sin.dtype == torch.bfloat16, \
        "cos/sin must be the backbone's bf16-cast tensors (Transformer.forward, modeling/model.py)"
    assert cos.shape == (B, N, RD) and sin.shape == (B, N, RD)
    assert q.stride(-1) == 1 and cos.stride(-1) == 1 and sin.stride(-1) == 1
    assert N % blk == 0, f"N={N} must divide by blk={blk}"
    assert RD % 2 == 0 and (RD // 2) & (RD // 2 - 1) == 0, \
        f"rotary_dim={RD}: half must be a power of two"
    assert RD <= D

    dev = q.device
    Q8 = torch.empty(B * H, N, D, device=dev, dtype=torch.float8_e4m3fn)
    Q8s = torch.empty(B * H, N // blk, device=dev, dtype=torch.float32)
    QROT = torch.empty(B * H, N, D, device=dev, dtype=torch.bfloat16) \
        if want_qrot else Q8
    sm = D ** -0.5
    _ropeq_kernel[(N // blk, B * H)](
        q, cos, sin, Q8, Q8s, QROT, N, H,
        q.stride(0), q.stride(1), q.stride(2),
        cos.stride(0), cos.stride(1), sin.stride(0), sin.stride(1),
        SM=sm, BLK=blk, D=D, RD=RD, HALF=RD // 2,
        WANT_QROT=want_qrot, num_warps=8)
    if want_qrot:
        return Q8, Q8s, QROT.view(B, H, N, D)
    return Q8, Q8s
