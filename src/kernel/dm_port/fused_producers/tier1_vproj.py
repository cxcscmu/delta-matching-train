"""The V quantizer of the dm mixer: `v8_quantize` turns bf16 V into e4m3 codes plus one fp32 scale per [blk, D] block,
per KV head before the GQA expansion. The dm mixer runs v_proj through torchao and calls `v8_quantize` on its bf16
output (modeling/dm_attention.py).

WHAT IT REPLACES (the reference V producer):
    v32 = v.float().reshape(BH,N,D).contiguous()    fp32 upcast copy
    V8, V8s = block_quant(v32, BN)                  the reference block quantizer (BN = 32 at head_dim 256, 64 at 128)

QUANT REALIZATION: `_v8_quant_nsub` performs the ops of delta_matching/quant.py `_quant_kernel` MODE 2 (the V mode)
in the same order: fp32 upcast (exact for bf16 input), absmax per block (exact, order-free), s = maximum(amax, 1e-30)
/ 448.0 with Triton's fp32 `/` (div.full.f32, the same lowering as quant.py; NOT a multiply by fp32(1/448): one ulp of
scale flips codes), codes via tl.math.div_rn (a plain `/` may lower to a reciprocal multiply and flip codes), clamp
+-448, RTNE `.to(tl.float8e4nv)`. The codes and scales are therefore the reference quantizer's, bit for bit.

Layout: codes dense C-contiguous e4m3 [BH, N, D], scales dense fp32 [BH, N // blk], flat offsets.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

E4M3_MAX = tl.constexpr(448.0)   # instance, not annotation: jit globals must be constexpr


# --------------------------------------------------------------------------------------
# The V8 block quantizer: NSUB blocks of [BLK, D] rows per call, one scale per block.
# --------------------------------------------------------------------------------------
@triton.jit
def _v8_quant_nsub(x32, NSUB: tl.constexpr, BLK: tl.constexpr, D: tl.constexpr):
    """x32: (NSUB*BLK, D) fp32 values that ARE fp32-of-bf16. Returns
    (codes e4m3 (NSUB*BLK, D), scales fp32 (NSUB,)), one scale per [BLK, D] block.
    Ops of delta_matching.quant._quant_kernel MODE 2 (V): absmax -> maximum(.,1e-30) ->
    Triton fp32 `/` by 448.0 (div.full.f32, the same lowering) -> div_rn codes -> clamp -> RTNE."""
    xb = tl.reshape(x32, (NSUB, BLK, D))
    amax = tl.max(tl.max(tl.abs(xb), axis=2), axis=1)          # (NSUB,) exact (order-free)
    s = tl.maximum(amax, 1e-30) / E4M3_MAX                     # same lowering as quant.py's scale
    q = tl.math.div_rn(xb, s[:, None, None])                   # NOT `/`: no recip-multiply
    q = tl.minimum(tl.maximum(q, -E4M3_MAX), E4M3_MAX)
    return tl.reshape(q.to(tl.float8e4nv), (NSUB * BLK, D)), s


@triton.jit
def _v8_from_bf16_kernel(V, V8, V8S, N, D: tl.constexpr, BLK: tl.constexpr):
    """bf16 [BH, N, D] -> V8 codes + per-[BLK,D] scales. One quant block per program."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    rm = pid_m * BLK + tl.arange(0, BLK)
    rd = tl.arange(0, D)
    base = pid_bh.to(tl.int64) * N * D + rm.to(tl.int64)[:, None] * D + rd[None, :]
    x = tl.load(V + base).to(tl.float32)                       # fp32-of-bf16, exact
    q8, s = _v8_quant_nsub(x, 1, BLK, D)
    tl.store(V8 + base, q8)
    tl.store(V8S + pid_bh.to(tl.int64) * (N // BLK) + pid_m + tl.arange(0, 1), s)


def v8_quantize(v: torch.Tensor, blk: int = 32):
    """bf16 v [BH, N, D] (or [B, H, N, D]) -> (V8 e4m3 [BH,N,D], V8s fp32 [BH,N//blk]).
    The reference block quantizer's ops at blk rows per scale, hence its codes and scales."""
    if v.dim() == 4:
        B, H, N, D = v.shape
        v = v.reshape(B * H, N, D)
    v = v.contiguous()
    BH, N, D = v.shape
    assert v.dtype == torch.bfloat16 and N % blk == 0
    V8 = torch.empty(BH, N, D, device=v.device, dtype=torch.float8_e4m3fn)
    V8s = torch.empty(BH, N // blk, device=v.device, dtype=torch.float32)
    _v8_from_bf16_kernel[(N // blk, BH)](v, V8, V8s, N, D=D, BLK=blk, num_warps=4)
    return V8, V8s
