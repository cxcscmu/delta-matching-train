"""The auxiliary-output delta (dm_mode "aux_out"): Attn-QAT's backward correction ported onto the Delta-Matching fp8
kernels, as a baseline for the matched delta.

Attn-QAT keeps a second, higher-precision forward output O' = P V-hat, with P the softmax probabilities BEFORE their fp8
cast (cast to bf16 for this one product) and V-hat the dequantized fp8 V, and forms the backward's softmax-gradient
centering term from it:  delta_i = < dO-hat_i , O'_i >.  In exact arithmetic sum_j P_ij dP_ij = < dO_i , P_i V >, so this
delta equals the matched delta (pass 0 of the kernel); in the kernel's arithmetic it differs by the bf16 rounding of P, the
bf16 storage of O', and the fp8 rounding of P inside the backward's own dP / dS tiles, which O' does not see.

What this module computes, per (batch, query head, row i):
    S_ij  = (sum_d q8_id k8_jd) * q8s(i) * k8s(j)        the kernel's OWN codes and block scales, QK^T as an fp8 x fp8
                                                          tensor-core product with fp32 accumulation like the forward
                                                          kernel's (the softmax scale is baked into the Q codes and K is
                                                          centred by its sequence mean: the kernel's convention)
    P_ij  = exp(S_ij - LSE_i)                            fp32, LSE = the forward's natural-log LSE (causal: j <= i); the
                                                          forward's denominator is the sum of the UNQUANTIZED p, so these
                                                          rows sum to 1 (the WANT_ROWSUM self-check measures it)
    O'_i  = sum_j bf16(P_ij) * (v8_jd * v8s(j))          bf16(P) times the EXACT dequantized V: the V codes enter the bf16
                                                          matmul as codes (e4m3 is exact in bf16) and each 64-key tile's
                                                          block scale multiplies that tile's fp32 product
    stored in bf16, the forward's (B, N, H, D) layout.
It is a separate pass launched right after the forward kernel (the forward kernel itself is unchanged): it re-reads the
codes and recomputes QK^T.  The backward forms delta = rowsum(dequant(dO8) * O') in the dO quantizer (doq_m17 with
delta_from_codes, the consistent-dO arm's realization) and then runs the unchanged fp8 backward (passes 1 and 2) with it;
no pass 0.

Realization: the V codes are converted to bf16 once per call (exact), the Q / K / V scales are read as one scalar per
tile (the 64-key tile is one K / V block and the row tile lies inside one Q block), each tile's bf16 product lands in a
fresh fp32 accumulator, is multiplied by the tile's V scale and added to the output accumulator.  Every launch
configuration gives bitwise the same O' (key tiles always in increasing order); training uses DM_AUX_CFG (see _FIXED).

Entry points
    aux_out(Q8, Q8s, K8, K8s, V8, V8s, Lse, B, H, Hkv, N, D, causal=True) -> O' (B, N, H, D) bf16
    aux_out_debug(...)      O' plus the per-row sum of the fp32 P (the LSE-consistency self-check), and the variants
                            used to price the pass (benchmarks only)
    aux_out_reference(...)  the same quantity in float64 torch from the codes (S exact, so not the tensor core's
                            rounding of QK^T; P rounded fp64 -> fp32 -> bf16), for tests
Layouts (the orchestrator's): Q8 [B*H, N, D] e4m3, Q8s [B*H, N // qblk]; K8 / V8 [B*Hkv, N, D] e4m3, K8s / V8s
[B*Hkv, N // kblk]; Lse (B, H, N) fp32 natural log; query head h reads KV head h // (H // Hkv).  head_dim 128 (qblk 128,
kblk 64) and 256 (qblk 64, kblk 32).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

__all__ = ["aux_out", "aux_out_debug", "aux_out_reference", "AUX_OUT_VERSION", "AUX_OUT_NORM"]

AUX_OUT_VERSION = "aux-out-v2"   # bump on any arithmetic change of the production configuration
LOG2E = 1.4426950408889634

# The production launch configuration: BLOCK_M, num_warps, num_stages (identical on every rank, never autotuned in
# training; the offline sweep, dm-workspace/auxout/scripts/t3_auxtune.py / t4_auxtune2.py, found every configuration
# bitwise identical, so this choice changes speed only).
_FIXED = os.environ.get("DM_AUX_CFG", "128,8,3")
# DM_AUX_NORM=1 (switch test, 2026-11-14; recorded in the run identity): divide each row of O' by the sum of its
# bf16-cast weights, sum_j bf16(P_ij) accumulated in fp32, so the cast weights of a row sum to one (Attn-QAT divides by
# a sum it builds from the weights it uses).  Unset / 0: unchanged (bitwise the aux-out-v2 output).
_NORM = os.environ.get("DM_AUX_NORM", "0").strip() == "1"
AUX_OUT_NORM = _NORM


def _fixed():
    bm, w, st = (int(x) for x in _FIXED.split(","))
    return bm, w, st


@triton.jit
def _aux_tile(acc, q, c_q, lse2, offs_m, start_n, kbase, ksbase, K8, K8S, VB, V8S, rsum, rs,
              D: tl.constexpr, BLOCK_N: tl.constexpr, KBLK: tl.constexpr, MASK: tl.constexpr,
              WANT_ROWSUM: tl.constexpr, SKIP_QK: tl.constexpr, SKIP_PV: tl.constexpr, NORM: tl.constexpr):
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    if SKIP_QK:                                                      # pricing variant: P = a constant
        p = tl.full([q.shape[0], BLOCK_N], 1.0 / 4096.0, tl.float32)
    else:
        k = tl.load(K8 + (kbase + offs_n.to(tl.int64))[:, None] * D + offs_d[None, :])
        ks = tl.load(K8S + ksbase + start_n // KBLK)                 # one K scale per tile (BLOCK_N == KBLK)
        s = tl.dot(q, tl.trans(k))                                   # fp8 x fp8 -> fp32 (the forward's QK^T)
        p = tl.math.exp2(s * (c_q * ks) - lse2[:, None])             # exp(S - LSE), S descaled in fp32
    if MASK:
        p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)
    if WANT_ROWSUM:
        rsum += tl.sum(p, axis=1)
    if NORM:                                                         # the row sum of the bf16-cast weights (fp32)
        rs += tl.sum(p.to(tl.bfloat16).to(tl.float32), axis=1)
    if SKIP_PV:                                                      # pricing variant: keep p live, no P.V-hat
        acc += tl.sum(p.to(tl.bfloat16).to(tl.float32), axis=1)[:, None]
    else:
        v = tl.load(VB + (kbase + offs_n.to(tl.int64))[:, None] * D + offs_d[None, :])   # V codes, exact in bf16
        vs = tl.load(V8S + ksbase + start_n // KBLK)                 # one V scale per tile
        acc += tl.dot(p.to(tl.bfloat16), v) * vs                     # bf16(P) @ codes, then the tile's V scale (fp32)
    return acc, rsum, rs


@triton.jit
def _aux_out_kernel(Q8, Q8S, K8, K8S, VB, V8S, LSE, OUT, RSUM, N, H, HKV, NREP,
                    D: tl.constexpr, BLOCK_N: tl.constexpr, QBLK: tl.constexpr, KBLK: tl.constexpr,
                    CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr, WANT_ROWSUM: tl.constexpr,
                    SKIP_QK: tl.constexpr, SKIP_PV: tl.constexpr, NORM: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    nbm = N // BLOCK_M
    mb = (nbm - 1 - pid_m) if CAUSAL else pid_m                      # longest rows first under causal
    b = pid_bh // H
    h = pid_bh % H
    bhk = b * HKV + h // NREP
    offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    bh64 = pid_bh.to(tl.int64)
    q = tl.load(Q8 + (bh64 * N + offs_m)[:, None] * D + offs_d[None, :])
    c_q = tl.load(Q8S + bh64 * (N // QBLK) + (mb * BLOCK_M) // QBLK) * 1.4426950408889634   # the tile's q scale x log2 e
    lse2 = tl.load(LSE + bh64 * N + offs_m) * 1.4426950408889634
    kbase = bhk.to(tl.int64) * N
    ksbase = bhk.to(tl.int64) * (N // KBLK)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    rsum = tl.zeros([BLOCK_M], dtype=tl.float32)
    rs = tl.zeros([BLOCK_M], dtype=tl.float32)
    hi = mb * BLOCK_M if CAUSAL else N                               # unmasked key tiles
    for start_n in range(0, hi, BLOCK_N):
        acc, rsum, rs = _aux_tile(acc, q, c_q, lse2, offs_m, start_n, kbase, ksbase, K8, K8S, VB, V8S, rsum, rs,
                                  D, BLOCK_N, KBLK, False, WANT_ROWSUM, SKIP_QK, SKIP_PV, NORM)
    if CAUSAL:                                                       # the diagonal tiles, masked
        for start_n in range(mb * BLOCK_M, (mb + 1) * BLOCK_M, BLOCK_N):
            acc, rsum, rs = _aux_tile(acc, q, c_q, lse2, offs_m, start_n, kbase, ksbase, K8, K8S, VB, V8S, rsum, rs,
                                      D, BLOCK_N, KBLK, True, WANT_ROWSUM, SKIP_QK, SKIP_PV, NORM)
    if NORM:                                                         # the cast weights of each row sum to one
        acc = acc / rs[:, None]
    o_ptrs = OUT + ((b.to(tl.int64) * N + offs_m)[:, None] * H + h) * D + offs_d[None, :]
    tl.store(o_ptrs, acc.to(tl.bfloat16))
    if WANT_ROWSUM:
        tl.store(RSUM + bh64 * N + offs_m, rsum)


def _blocks(D):
    from delta_matching.core import blocks
    return blocks(D)                                                  # (Q rows, K/V keys) per scale


def _check(Q8, Q8s, K8, K8s, V8, V8s, Lse, B, H, Hkv, N, D):
    qblk, kblk = _blocks(D)
    assert D in (128, 256) and H % Hkv == 0 and N % 128 == 0, (D, H, Hkv, N)
    assert Q8.dtype == K8.dtype == V8.dtype == torch.float8_e4m3fn
    assert Q8.shape == (B * H, N, D) and K8.shape == (B * Hkv, N, D) and V8.shape == (B * Hkv, N, D), \
        (tuple(Q8.shape), tuple(K8.shape), tuple(V8.shape))
    assert Q8s.numel() == B * H * (N // qblk) and K8s.numel() == B * Hkv * (N // kblk) and V8s.numel() == B * Hkv * (N // kblk)
    assert Lse.numel() == B * H * N and Lse.dtype == torch.float32
    for t in (Q8, K8, V8):
        assert t.is_contiguous()
    return qblk, kblk


def _launch(Q8, Q8s, K8, K8s, V8, V8s, Lse, B, H, Hkv, N, D, causal, cfg, want_rowsum=False, skip_qk=False,
            skip_pv=False, norm=None):
    norm = _NORM if norm is None else bool(norm)
    qblk, kblk = _check(Q8, Q8s, K8, K8s, V8, V8s, Lse, B, H, Hkv, N, D)
    bm, w, st = cfg[:3]
    assert N % bm == 0 and bm % kblk == 0 and bm <= qblk and qblk % bm == 0, (bm, qblk, kblk)
    out = torch.empty(B, N, H, D, device=Q8.device, dtype=torch.bfloat16)
    rsum = torch.empty(B * H, N, device=Q8.device, dtype=torch.float32) if want_rowsum else out
    Q8s, K8s, V8s, Lse = (t.contiguous() for t in (Q8s, K8s, V8s, Lse))
    with torch.cuda.device(Q8.device):
        VB = V8.to(torch.bfloat16)                                    # the V codes, exact in bf16 (once per call)
        _aux_out_kernel[(N // bm, B * H)](Q8, Q8s, K8, K8s, VB, V8s, Lse, out, rsum, N, H, Hkv, H // Hkv,
                                          D=D, BLOCK_N=kblk, QBLK=qblk, KBLK=kblk, CAUSAL=bool(causal), BLOCK_M=bm,
                                          WANT_ROWSUM=bool(want_rowsum), SKIP_QK=bool(skip_qk), SKIP_PV=bool(skip_pv),
                                          NORM=norm, num_warps=w, num_stages=st)
    return out, (rsum if want_rowsum else None)


def aux_out(Q8, Q8s, K8, K8s, V8, V8s, Lse, B, H, Hkv, N, D, causal: bool = True):
    """O' = bf16(P) @ V-hat from the kernel's own codes, scales and LSE (see the module docstring): (B, N, H, D) bf16."""
    return _launch(Q8, Q8s, K8, K8s, V8, V8s, Lse, B, H, Hkv, N, D, causal, _fixed())[0]


def aux_out_debug(Q8, Q8s, K8, K8s, V8, V8s, Lse, B, H, Hkv, N, D, causal=True, cfg=None, want_rowsum=True,
                  skip_qk=False, skip_pv=False, norm=None):
    """Benchmarks / tests: (O', rowsum of the fp32 P [B*H, N] or None) under an explicit cfg (BLOCK_M, warps, stages)
    and the pricing variants (skip_qk: P constant, no QK^T; skip_pv: no P.V-hat product)."""
    return _launch(Q8, Q8s, K8, K8s, V8, V8s, Lse, B, H, Hkv, N, D, causal, cfg or _fixed(), want_rowsum, skip_qk,
                   skip_pv, norm)


@torch.no_grad()
def aux_out_reference(Q8, Q8s, K8, K8s, V8, V8s, Lse, B, H, Hkv, N, D, causal: bool = True, heads=None, norm=None):
    """The same O' in float64 torch from the codes (S and P.V-hat in fp64; P rounded fp64 -> fp32 -> bf16 like the
    kernel's fp32 P).  heads: an iterable of (b, h) to compute (default all); returns (B, N, H, D) bf16 with only those
    heads filled (zeros elsewhere).  NOTE: S here is exact, while the forward kernel's QK^T is an fp8 tensor-core
    product whose accumulation keeps fewer bits, so this is a reference for the P.V-hat arithmetic, not for S."""
    qblk, kblk = _check(Q8, Q8s, K8, K8s, V8, V8s, Lse, B, H, Hkv, N, D)
    rep = H // Hkv
    out = torch.zeros(B, N, H, D, device=Q8.device, dtype=torch.bfloat16)
    ar = torch.arange(N, device=Q8.device)
    q8s, k8s, v8s, lse = Q8s.reshape(B * H, -1), K8s.reshape(B * Hkv, -1), V8s.reshape(B * Hkv, -1), Lse.reshape(B * H, N)
    for (b, h) in (heads if heads is not None else [(b, h) for b in range(B) for h in range(H)]):
        bh, bhk = b * H + h, b * Hkv + h // rep
        q = Q8[bh].double() * q8s[bh, ar // qblk].double()[:, None]
        k = K8[bhk].double() * k8s[bhk, ar // kblk].double()[:, None]
        v = V8[bhk].double() * v8s[bhk, ar // kblk].double()[:, None]
        s = q @ k.t()
        p = torch.exp(s - lse[bh].double()[:, None])
        if causal:
            p = p.masked_fill(ar[None, :] > ar[:, None], 0.0)
        p = p.float().to(torch.bfloat16).double()
        o = p @ v
        if (_NORM if norm is None else norm):
            o = o / p.sum(1, keepdim=True)
        out[b, :, h] = o.to(torch.bfloat16)
        del q, k, v, s, p
    return out
