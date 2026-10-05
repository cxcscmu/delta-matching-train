"""The fused-producer orchestrator of the `dm` mixer (modeling/dm_attention.py): the fp8 attention core of the
Delta-Matching training arms on the kernel package `delta_matching`, at head_dim 128 (the Delta-Matching-128 block-scale
contract, Q/dO scales per 128 rows and K/V scales per 64 keys) and head_dim 256 (64 rows / 32 keys).

Delta modes (the arm's `dm_mode`):
  stale     delta = rowsum(dO * O) from the forward's O, dP quantized       (the paper's "naive fp8" arm)
  matched   delta from the backward's own realized fp8 P / dP pair (pass 0 of the kernel, delta_fp8), dP quantized
            (Delta-Matching)
  bf16_dp   the stale delta with the single dP = dO V^T matmul kept in bf16 (SageBwd's exemption among the seven
            attention matmuls, arXiv 2505.11594 Algorithm 3 line 8, with the delta still Algorithm 3 line 2's
            rowsum(dO o O) over the quantized forward's O).  Needs the kernel built with DM_BF16_DP=1
            (build_bf16dp() == 1); the build flag is asserted, never inferred from the arm name.  head_dim 128
            only (the kernel's bf16-dP backward, bwd_fp8_bf16dp; the dm mixer refuses other head dims at build).

Producers (every code and scale bitwise the reference quantizer's rule at these blocks):
  Q   fused_producers/tier3a_ropeq.ropeq_q: partial RoPE + softmax-scale bake + block e4m3 quant, plus the pi-ordered
      Q twin (delta_matching.twins.twin_pi) the backward's pass 1 loads from global memory;
  K   delta_matching.producers.quant_k_m17: the frozen block quantizer with the split k-mean, the K twin from the same
      launch;
  V   V8kv / V8skv handed in by the mixer (fused_producers/tier1_vproj.v8_quantize, per KV head before the GQA
      expansion);
  dO  delta_matching.producers.doq_m17 in every mode: dO read in its native (B, N, H, D) layout, the twin in the same
      pass; stale / bf16_dp: rowsum(dO * O) in-kernel; matched: no O read;
  dq  the one-pass q-RoPE backward (fused_producers/m13_rope_bwd.rope_bwd_q), dk/dv folded to the KV heads exactly as
      the aten repeat_interleave backward does.
The kernel ops dispatch on head_size inside the package; the backward is split per KV-head group / per query head at
long context (delta_matching.split: DM_BWD_HEAD_GROUPS, DM_BWD_WS_LIMIT_GB, DM_BWD_STREAMS), bitwise the single call.
DM_STALE_DOQ2S (the launcher exports it for the stale arms) is accepted and inert here: the strided native-layout dO
quantizer is the only realization of this route.

Both head dims share this one orchestrator through delta_matching.core.blocks(D); in every mode the dO codes and
scales come from doq_m17 (the reference dO quantizer's amax / scale / div_rn / clamp / RTNE rule).
"""
from __future__ import annotations

import os as _os

import torch

try:
    import delta_matching  # noqa: F401
except ImportError as _e:  # pragma: no cover
    raise ImportError("kernel.dm_port.orchestrator needs the Delta-Matching kernel package: install the delta-matching "
                      "kernel package (see docs/SETUP.md section 3: git clone --recursive, then "
                      "pip install --no-build-isolation .)") from _e

from delta_matching.attn import _ops_gt
from delta_matching.core import _check_hd, blocks
from delta_matching.producers import doq_m17, quant_k_m17
from delta_matching import split as _split
from delta_matching.twins import twin_pi

from kernel.dm_port import diag
from kernel.dm_port.fused_producers.m13_rope_bwd import rope_bwd_q
from kernel.dm_port.fused_producers.tier3a_ropeq import ropeq_q

# What the EXEMPT dP GEMM of the bf16_dp build sees, chosen per process by DM_BF16DP_MODE (default raw, SageBwd's
# exemption); the other three are study variants that isolate one operand's rounding:
#   raw    raw V, raw dO                      SageBwd's exemption (the campaign's bf16_dp arm)
#   v8     raw dO, V = its own e4m3 codes     isolates V's rounding in dP
#   do8    raw V, dO = its own e4m3 codes     isolates dO's rounding in dP (the flushed rows of the 128-row block)
#   do8sr  raw V, dO rounded STOCHASTICALLY   dO's rounding without the flush bias (unbiased per element)
# Every mode keeps the stale delta and the fp8 dV/dQ/dK paths; only the operands of the bf16 dP GEMM change.
_BF16DP_MODE = _os.environ.get("DM_BF16DP_MODE", "raw").strip().lower()
_DELTA_Q8 = _os.environ.get("DM_DELTA_Q8", "0").strip() == "1"   # 2026-09-17: stale delta from the dO CODES (a study flag)
_delta_q8_announced = [False]
_BF16DP_BANNER = [False]
_bwd_chunk_announced = [False]


def _dequant_rows(codes, scales, blk):
    """codes (G, N, D) e4m3, scales (G, N // blk) -> (G, N, D) fp32, one scale per blk rows."""
    G, N, D = codes.shape
    idx = torch.arange(N, device=codes.device) // blk
    return codes.float() * scales.reshape(G, N // blk)[:, idx].unsqueeze(-1)


def _sr_e4m3(y):
    """Stochastic rounding of y (already divided by its block scale, |y| <= 448) to the e4m3 grid: add
    (u - 1/2) * ulp(y) with u ~ U[0, 1) and round to nearest; unbiased inside each rounding cell, so the small
    values that round-to-nearest flushes to zero survive with probability |y| / ulp."""
    a = y.abs().clamp_min(2.0 ** -9)
    e = torch.floor(torch.log2(a)).clamp_min(-6.0) - 3.0            # normals: 3 mantissa bits; subnormals: ulp 2^-9
    ulp = torch.exp2(e)
    yp = (y + (torch.rand_like(y) - 0.5) * ulp).clamp(-448.0, 448.0)
    return yp.to(torch.float8_e4m3fn).float()


def _bf16dp_operands(v_raw, dO, V8kv, V8skv, dO8, dO8s, B, H, Hkv, N, D, BM, BN):
    mode = _BF16DP_MODE
    if mode not in ("raw", "v8", "do8", "do8sr"):
        raise ValueError(f"DM_BF16DP_MODE must be raw|v8|do8|do8sr, got {mode!r}")
    if not _BF16DP_BANNER[0]:
        _BF16DP_BANNER[0] = True
        print(f"[dm_port] bf16 dP operands: mode={mode}  (raw = SageBwd's exemption; v8 / do8 / do8sr = study variants "
              f"that round one operand)", flush=True)
    if mode == "v8":
        v_in = (_dequant_rows(V8kv.view(B * Hkv, N, D), V8skv, BN).to(v_raw.dtype)
                .view(B, Hkv, N, D).permute(0, 2, 1, 3).contiguous())
    else:
        v_in = v_raw.permute(0, 2, 1, 3).contiguous()                    # (B, N, Hkv, D), V8kv's kernel layout
    if mode in ("raw", "v8"):
        do_in = dO.contiguous()                                          # (B, N, H, D) bf16
    elif mode == "do8":
        do_in = (_dequant_rows(dO8.view(B * H, N, D), dO8s, BM).to(dO.dtype)
                 .view(B, H, N, D).permute(0, 2, 1, 3).contiguous())
    else:  # do8sr
        idx = torch.arange(N, device=dO.device) // BM
        sc = dO8s.reshape(B * H, N // BM)[:, idx].unsqueeze(-1)          # (B*H, N, 1)
        y = dO.permute(0, 2, 1, 3).reshape(B * H, N, D).float() / sc.clamp_min(1e-30)
        do_in = (_sr_e4m3(y) * sc).to(dO.dtype).view(B, H, N, D).permute(0, 2, 1, 3).contiguous()
    return v_in, do_in


ORCHESTRATOR_VERSION = "dm-orchestrator-v1"
# stale     stale delta, dP QUANTIZED     (the paper's "naive fp8" arm)
# matched   matched delta, dP quantized   (Delta-Matching)
# bf16_dp   stale delta, dP in RAW BF16   (SageBwd's exemption; needs a DM_BF16_DP=1 build, asserted below)
_MATCHED = {"stale": False, "matched": True, "bf16_dp": False}
_BF16DP = {"stale": False, "matched": False, "bf16_dp": True}
# Memory option (default OFF, bitwise: the twin is a layout copy of the saved Q8): DM_QT_RECOMPUTE=1 does not save the Q
# twin for the backward and re-makes it there (one short pass).
_QT_RECOMPUTE = _os.environ.get("DM_QT_RECOMPUTE", "0") == "1"


def _ops_for(D):
    """The kernel ops with the global-memory twins compiled in (build_gt() == 1); head_dim 128 additionally needs the
    Delta-Matching-128 instantiations (build_hd128() == 1). Loud otherwise."""
    ops = _ops_gt()
    _check_hd(D, ops)
    return ops


class _FusedDMAttn(torch.autograd.Function):
    """q_pre (B, H, N, D) bf16 post-qk-norm PRE-RoPE; k_rot (B, Hkv, N, D) post-RoPE; v_t (B, Hkv, N, D) (the dv graph
    edge); V8kv / V8skv the pre-expansion V codes / scales; cos / sin (B, N, rotary_dim) bf16.  Returns O (B, N, H, D)."""

    @staticmethod
    def forward(ctx, q_pre, k_rot, v_t, V8kv, V8skv, cos, sin, n_rep, matched, bf16_dp=False, diag_layer=None):
        B, H, N, D = q_pre.shape
        Hkv = k_rot.shape[1]
        assert Hkv * n_rep == H and k_rot.shape == (B, Hkv, N, D)
        assert D in (128, 256), f"the dm orchestrator runs head_dim 128 or 256 (got {D})"
        assert N % 128 == 0, f"the Delta-Matching kernel needs N % 128 == 0 (got {N})"
        BM, BN = blocks(D)
        assert V8kv.shape == (B * Hkv, N, D) and V8skv.shape == (B * Hkv, N // BN), \
            f"V8kv/V8skv must be [B*Hkv, N, {D}] / [B*Hkv, N/{BN}] (got {tuple(V8kv.shape)} {tuple(V8skv.shape)})"
        ops = _ops_for(D)
        sm = D ** -0.5
        Q8, Q8s = ropeq_q(q_pre, cos, sin, blk=BM)                         # q RoPE + sm bake + block quant
        Qt = Q8 if _QT_RECOMPUTE else twin_pi(Q8)                          # the pass-1 Q twin
        K8kv, K8skv, kTtkv = quant_k_m17(k_rot, blk_k=BN)                  # split k-mean; the frozen block quantizer

        def bshd(x, h):
            return x.view(B, h, N, D).permute(0, 2, 1, 3)

        O, Lse = ops.fwd_fp8(bshd(Q8, H), bshd(K8kv, Hkv), bshd(V8kv, Hkv),
                             Q8s.view(B, H, -1), K8skv.view(B, Hkv, -1),
                             V8skv.view(B, Hkv, -1), sm, True)
        # bf16_dp also needs the RAW bf16 V for its exempt matmul; v_t is already an input (the dv graph edge), so
        # this only retains it.  ~17 MB per attention layer per rank at the M-tier hd128 cell.
        ctx.save_for_backward(Q8, Q8s, Qt, K8kv, K8skv, V8kv, V8skv, kTtkv, O, Lse, cos, sin,
                              *((v_t,) if bf16_dp else ()), *((k_rot,) if diag_layer is not None else ()))
        ctx.meta = (B, H, Hkv, int(n_rep), N, D, bool(matched), q_pre.dtype, bool(bf16_dp), diag_layer)
        return O  # (B, N, H, D) bf16 contiguous

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dO):
        saved = ctx.saved_tensors
        Q8, Q8s, Qt, K8kv, K8skv, V8kv, V8skv, kTtkv, O, Lse, cos, sin = saved[:12]
        B, H, Hkv, n_rep, N, D, matched, in_dtype, bf16_dp, diag_layer = ctx.meta
        BM, BN = blocks(D)
        # side channel only: no_grad, detached, its own files, never fed back (kernel/dm_port/diag.py)
        dg = diag.begin(diag_layer) if diag_layer is not None else False
        v_raw = saved[12] if bf16_dp else None
        k_rot_diag = saved[13 if bf16_dp else 12] if diag_layer is not None else None
        ops = _ops_for(D)
        if bf16_dp and int(getattr(ops, "build_bf16dp", lambda: 0)()) != 1:
            raise RuntimeError("dm_mode bf16_dp needs the delta_matching kernel built with DM_BF16_DP=1 (build_bf16dp() == 1)")
        if _QT_RECOMPUTE:
            Qt = twin_pi(Q8)
        sm = D ** -0.5
        # dO codes/scales per [BM, D] block (+ the twin); the stale delta only when needed.
        # On a logged diag step take the stale delta in every arm (it is the cheap one) so that eps can be
        # formed for BOTH arms; off the logged steps this is exactly the original call.
        want_stale = dg or not matched
        if _DELTA_Q8:
            if matched:
                raise ValueError("DM_DELTA_Q8=1 is a STALE-delta study flag (delta = rowsum(dequant(dO8) * O)); "
                                 "it has no meaning under the matched arm (the arm name decides the numerics)")
            if not _delta_q8_announced[0]:
                _delta_q8_announced[0] = True
                print("[dm_port] STALE DELTA FROM THE dO CODES: delta = rowsum(dequant(dO8) * O), the operand dP sees "
                      "(DM_DELTA_Q8=1; the raw-dO rowsum is off)", flush=True)
        dO8, dO8s, Dlt, dOt = doq_m17(dO, O if want_stale else None, H, BM,
                                      want_delta=want_stale, want_twin=True, delta_from_codes=_DELTA_Q8)
        Dlt_stale = Dlt if want_stale else None

        def bshd(x, h):
            return x.view(B, h, N, D).permute(0, 2, 1, 3)

        Dlt_matched = None
        _hp = _split._bwd_heads_per_call(B, H, Hkv, n_rep, N) if not bf16_dp else H
        _chunked_matched = matched and not dg and _hp < H and _split._BWD_STREAMS > 1   # pass 0 per group, only on the stream path
        if (matched or dg) and not _chunked_matched:
            Dlt_matched = ops.delta_fp8(bshd(dO8, H), bshd(Q8, H), bshd(K8kv, Hkv), bshd(V8kv, Hkv),
                                        Lse.view(B, H, N).contiguous(), kTtkv.view(B, Hkv, D, N),
                                        Q8s.view(B, H, -1), K8skv.view(B, Hkv, -1),
                                        V8skv.view(B, Hkv, -1), dO8s.view(B, H, -1), True).view(B * H, N)
        if matched and Dlt_matched is not None:
            Dlt = Dlt_matched
        # on the per-group matched path (_chunked_matched) no all-heads delta exists yet: pass 0 runs per group inside
        # _bwd_by_head_groups, which ignores this slot
        bwd_args = (bshd(dO8, H), bshd(Q8, H), bshd(K8kv, Hkv), bshd(V8kv, Hkv),
                    Lse, Dlt.view(B, H, N) if Dlt is not None else None, kTtkv.view(B, Hkv, D, N),
                    Qt.view(B, H, D, N), dOt.view(B, H, D, N),
                    Q8s.view(B, H, -1), K8skv.view(B, Hkv, -1), V8skv.view(B, Hkv, -1),
                    dO8s.view(B, H, -1))
        if bf16_dp:
            # raw V as (B, N, Hkv, D) to match V8kv's kernel layout; dO is already (B, N, H, D) bf16.
            # DM_BF16DP_MODE (default raw) may swap either operand for its own e4m3-rounded values (study variants).
            v_in, do_in = _bf16dp_operands(v_raw, dO, V8kv, V8skv, dO8, dO8s, B, H, Hkv, N, D, BM, BN)
            dq, dk, dv = ops.bwd_fp8_bf16dp(*bwd_args, v_in, do_in, sm, True)
        else:
            if _hp >= H:
                dq, dk, dv = ops.bwd_fp8_t(*bwd_args, sm, True)
            else:
                if not _bwd_chunk_announced[0]:
                    _bwd_chunk_announced[0] = True
                    print(f"[dm_port] BACKWARD PER HEAD GROUP: {_hp} query head(s) per call ({H // _hp} calls; dS' workspace "
                          f"{B * _hp * N * N / 2 ** 30:.1f} GiB per call instead of {B * H * N * N / 2 ** 30:.1f} GiB; "
                          f"DM_BWD_HEAD_GROUPS={_split._BWD_GROUPS}, limit {_split._BWD_WS_LIMIT / 2 ** 30:.0f} GiB; "
                          f"streams {_split._BWD_STREAMS}; pass 0 {'per group' if _chunked_matched else 'all heads'})", flush=True)
                dq, dk, dv = _split._bwd_by_head_groups(ops, bwd_args, sm, B, H, Hkv, n_rep, N, D, _hp,
                                                        matched_delta=_chunked_matched)
        if dg:
            # eps = delta_matched - delta_stale, in true dP units: the invariant residual itself.
            # delta_matched is by construction sum_j P-hat dP-hat over the backward's OWN realized
            # quantities, so eps is exactly what sum_j dS_ij becomes under the arm's delta.
            _eps = Dlt_matched - Dlt_stale
            _rows = diag.row_table(Q8.view(B * H, N, D), Q8s, K8kv.view(B * Hkv, N, D), K8skv,
                                   _eps.view(B * H, N), B, H, Hkv, N, D, BM, BN, sm)
            # The per-channel gain gradient the optimizer sees (kernel frame, from the arm's own dk) and the
            # residual's force split by flushed vs intact dO rows, so links 2 and 3 of the chain are read in training.
            _force = diag.gain_force(Q8.view(B * H, N, D), Q8s, K8kv.view(B * Hkv, N, D), K8skv, k_rot_diag,
                                     dk, _eps.view(B * H, N), dO8.view(B * H, N, D), B, H, Hkv, N, D, BM, BN)
            # k_dc from the RAW post-RoPE keys: the K codes are mean-centered per (b, h) by the smooth-K quantizer,
            # so their DC statistic is ~0 by construction
            diag.record(diag_layer, dk=dk, dv=dv, dq=dq, eps=_eps, q8=Q8, lse=Lse, rows=_rows,
                        kdc=diag.dc_raw(k_rot_diag.reshape(B * Hkv, N, D)),
                        qdc=diag.dc(Q8.view(B * H, N, D), Q8s, B * H, BM), force=_force)
        dq_pre = rope_bwd_q(dq, cos, sin, H)          # the one-pass q-RoPE backward (bitwise the autograd replay)
        dk = dk.permute(0, 2, 1, 3)
        dv = dv.permute(0, 2, 1, 3)
        if n_rep > 1:   # EXACT aten repeat_interleave backward: dk/dv per query head folded to the KV heads
            dk = dk.reshape(B, Hkv, n_rep, N, D).sum(2)
            dv = dv.reshape(B, Hkv, n_rep, N, D).sum(2)
        return dq_pre, dk, dv, None, None, None, None, None, None, None, None


def fused_dm_attention(q_pre, k_rot, v_t, V8kv, V8skv, cos, sin, n_rep: int, mode: str = "matched", diag_layer=None):
    """The dm mixer's attention core: mode in stale | matched | bf16_dp (see the module docstring)."""
    if mode not in _MATCHED:
        raise ValueError(f"fused_dm_attention supports stale|matched|bf16_dp, got {mode!r}")
    return _FusedDMAttn.apply(q_pre, k_rot, v_t, V8kv, V8skv, cos, sin, int(n_rep), _MATCHED[mode], _BF16DP[mode], diag_layer)
