"""The fused, BITWISE q-RoPE backward (rope_bwd_q) and the fused k RoPE (k_rope_fused) of the dm mixer.

WHAT IT REPLACES. The autograd replay of the q-RoPE backward, op for op: a
`.contiguous()` copy of dq, an fp32 upcast, two fp32 multiplies, two zero-filled fp32 temps with
slice assignments, two fp32 adds, a bf16 downcast, two more zero-filled bf16 temps with slice
assignments and a bf16 add: about eleven full passes over a [B, H, N, D] tensor per layer per
microbatch.  This kernel does the same arithmetic in ONE
pass: it reads dq in the kernel's native (B, N, H, D) layout (no contiguous copy), cos/sin, and
writes dq_pre DENSE (B, H, N, D): the same strides `zeros_like(dq_post)` produced, so q_norm's
weight-grad reduction, whose summation order follows the gradient's strides, sees identical strides
(a permuted layout there flips low-order bf16 bits of the q_norm weight gradient).

BITWISE CONTRACT (gated in `selftest()` below against the verbatim replay): every op is the same
IEEE fp32 RN op in the same order.  Per rotated lane j (HALF = RD // 2), with g = fp32(dq):
    lane j <  HALF:  d = RN(RN(0 + RN(g[j+HALF] * sin[j+HALF])) + RN(g[j] * cos[j]))
    lane j >= HALF:  d = RN(RN(neg(RN(g[j-HALF] * sin[j-HALF])) + 0) + RN(g[j] * cos[j]))
    dq_pre[j] = bf16_RTNE(d)          (the final `zeros + d_xr_bf` bf16 add is exact: 0 + x = x)
    tail lanes j >= RD: dq_pre[j] = dq[j]  (bf16 pass-through, partial rotary)
`neg` is torch's sign-bit flip (so -0.0 survives into the `+ 0` add, which then normalizes it
exactly as torch does); mul_rn/add_rn are libdevice explicitly-rounded intrinsics, never
FMA-contracted (the same rule as tier3a_ropeq.py).  cos/sin are the backbone's bf16 tensors,
upcast exactly.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tld


@triton.jit
def _fneg(x):
    """IEEE sign-bit flip (torch.neg); Triton's unary minus is (0 - x) and loses -0.0."""
    return (x.to(tl.int32, bitcast=True) ^ -2147483648).to(tl.float32, bitcast=True)


@triton.jit
def _rope_bwd_kernel(DQ, COS, SIN, OUT, N, H,
                     sdb, sdn, sdh,            # dq element strides (B, N, H); lane stride 1
                     scb, sct, ssb, sst,       # cos/sin element strides (B, T); lane stride 1
                     BLK: tl.constexpr, D: tl.constexpr, RD: tl.constexpr, HALF: tl.constexpr):
    """One program per (row block of BLK tokens, b*H + h).  OUT is dense (B, H, N, D)."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    rm = pid_m * BLK + tl.arange(0, BLK)
    r1 = tl.arange(0, HALF)
    grow = DQ + b.to(tl.int64) * sdb + h.to(tl.int64) * sdh + rm.to(tl.int64)[:, None] * sdn
    g1 = tl.load(grow + r1[None, :]).to(tl.float32)             # lanes 0..HALF-1
    g2 = tl.load(grow + (HALF + r1)[None, :]).to(tl.float32)    # lanes HALF..RD-1
    crow = COS + b.to(tl.int64) * scb + rm.to(tl.int64)[:, None] * sct
    srow = SIN + b.to(tl.int64) * ssb + rm.to(tl.int64)[:, None] * sst
    c1 = tl.load(crow + r1[None, :]).to(tl.float32)
    c2 = tl.load(crow + (HALF + r1)[None, :]).to(tl.float32)
    s1 = tl.load(srow + r1[None, :]).to(tl.float32)
    s2 = tl.load(srow + (HALF + r1)[None, :]).to(tl.float32)
    zero = tl.zeros([BLK, HALF], dtype=tl.float32)
    # gc = g*cos ; gs = g*sin  (fp32 RN multiplies, as torch's fp32 mul)
    gc1 = tld.mul_rn(g1, c1)
    gc2 = tld.mul_rn(g2, c2)
    gs1 = tld.mul_rn(g1, s1)
    gs2 = tld.mul_rn(g2, s2)
    # replay: d_xr = (d_xr + z1) + gc with d_xr[HALF:] = -gs[:HALF], z1[:HALF] = gs[HALF:]
    d1 = tld.add_rn(tld.add_rn(zero, gs2), gc1)                  # lanes < HALF: (0 + gs2) + gc1
    d2 = tld.add_rn(tld.add_rn(_fneg(gs1), zero), gc2)           # lanes >= HALF: (-gs1 + 0) + gc2
    obase = OUT + pid_bh.to(tl.int64) * N * D + rm.to(tl.int64)[:, None] * D
    tl.store(obase + r1[None, :], d1.to(tl.bfloat16))
    tl.store(obase + (HALF + r1)[None, :], d2.to(tl.bfloat16))
    if RD < D:                                                   # partial rotary: pass-through tail
        rd = tl.arange(0, D)
        tail = rd[None, :] >= RD
        x_all = tl.load(grow + rd[None, :], mask=tail, other=0.0)
        tl.store(obase + rd[None, :], x_all, mask=tail)


def rope_bwd_q(dq_bnhd: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, H: int,
               blk: int = 64) -> torch.Tensor:
    """dq_bnhd: bf16 [B, N, H, D] (the kernel's native gradient layout; any dense strides with
    lane stride 1); cos/sin: bf16 [B, N, RD].  Returns dq_pre bf16 DENSE [B, H, N, D], bitwise the
    orchestrator's replayed q-RoPE backward (selftest)."""
    B, N, Hd, D = dq_bnhd.shape
    assert Hd == H
    RD = cos.shape[-1]
    assert dq_bnhd.dtype == torch.bfloat16 and cos.dtype == torch.bfloat16 and sin.dtype == torch.bfloat16
    assert cos.shape == (B, N, RD) and sin.shape == (B, N, RD)
    assert dq_bnhd.stride(-1) == 1 and cos.stride(-1) == 1 and sin.stride(-1) == 1
    assert N % blk == 0 and RD % 2 == 0 and (RD // 2) & (RD // 2 - 1) == 0 and RD <= D
    out = torch.empty(B, H, N, D, device=dq_bnhd.device, dtype=torch.bfloat16)
    _rope_bwd_kernel[(N // blk, B * H)](
        dq_bnhd, cos, sin, out, N, H,
        dq_bnhd.stride(0), dq_bnhd.stride(1), dq_bnhd.stride(2),
        cos.stride(0), cos.stride(1), sin.stride(0), sin.stride(1),
        BLK=blk, D=D, RD=RD, HALF=RD // 2, num_warps=8)
    return out


class _KRopeFusedFn(torch.autograd.Function):
    """The k-RoPE of the dm mixer as ONE kernel each way.  Forward = the rotated-output
    (WANT_QROT) path of tier3a_ropeq.py's `_ropeq_kernel` (apply_rotary's rotation op for op:
    fp32 mul_rn/add_rn, RTNE to bf16; its Q8/Q8s side outputs are discarded here); backward
    = `rope_bwd_q` (bitwise the autograd backward of `kf*cos + rotate_half(kf)*sin`).  Both
    directions are checked bitwise against the eager k-RoPE by `selftest_krope()` below.
    Replaces the eager fp32 k-RoPE (six fp32 passes forward, about nine backward, plus the
    fp32 kf saved for autograd)."""

    @staticmethod
    def forward(ctx, k, cos, sin):
        from kernel.dm_port.fused_producers.tier3a_ropeq import ropeq_q
        B, Hkv, N, D = k.shape
        _, _, k_rot = ropeq_q(k, cos, sin, blk=64, want_qrot=True)   # dense (B, Hkv, N, D)
        ctx.save_for_backward(cos, sin)
        ctx.hkv = Hkv
        return k_rot

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dk):
        cos, sin = ctx.saved_tensors
        # rope_bwd_q reads (B, N, H, D)-shaped strides; dk is (B, Hkv, N, D) (any dense layout)
        return rope_bwd_q(dk.permute(0, 2, 1, 3), cos, sin, ctx.hkv), None, None


def k_rope_fused(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """k: bf16 [B, Hkv, N, D] (post k_norm, pre-RoPE, any dense strides, lane stride 1);
    cos/sin: the backbone's bf16 [B, N, RD].  Returns the rotated bf16 k, dense (B, Hkv, N, D)."""
    assert k.dim() == 4 and k.dtype == torch.bfloat16
    return _KRopeFusedFn.apply(k, cos, sin)


def eager_k_rope(k, cos, sin):
    """The eager k-RoPE (apply_rotary's k branch): the reference of selftest_krope."""
    from pretrain.modeling.layers import rotate_half
    cos_u, sin_u = cos.unsqueeze(1), sin.unsqueeze(1)
    rd = cos_u.shape[-1]
    D = k.shape[-1]
    if rd == D:
        kf = k.float()
        return (kf * cos_u + rotate_half(kf) * sin_u).to(k.dtype)
    kr, kp = k[..., :rd].float(), k[..., rd:]
    return torch.cat([(kr * cos_u + rotate_half(kr) * sin_u).to(k.dtype), kp], dim=-1)


def selftest_krope(verbose: bool = True) -> bool:
    ok = True
    torch.manual_seed(1)
    for (B, Hkv, N, D, RD) in [(2, 4, 8192, 256, 256), (1, 2, 1024, 256, 256), (2, 4, 2048, 256, 64)]:
        kflat = (torch.randn(B, N, Hkv * D, device="cuda") * 2).to(torch.bfloat16)
        pos = torch.arange(N, device="cuda").float()
        inv = 1.0 / (1e6 ** (torch.arange(0, RD, 2, device="cuda").float() / RD))
        fr = torch.einsum("t,d->td", pos, inv)
        emb = torch.cat([fr, fr], -1)
        cos = emb.cos().to(torch.bfloat16)[None].expand(B, N, RD).contiguous()
        sin = emb.sin().to(torch.bfloat16)[None].expand(B, N, RD).contiguous()
        dk = (torch.randn(B, Hkv, N, D, device="cuda") * 1e-3).to(torch.bfloat16)
        k1 = kflat.clone().requires_grad_(True)
        k1v = k1.view(B, N, Hkv, D).transpose(1, 2)           # the mixer's transposed view
        r1 = eager_k_rope(k1v, cos, sin)
        r1.backward(dk)
        k2 = kflat.clone().requires_grad_(True)
        k2v = k2.view(B, N, Hkv, D).transpose(1, 2)
        r2 = k_rope_fused(k2v, cos, sin)
        r2.backward(dk)
        f_eq = torch.equal(r1.view(torch.int16), r2.view(torch.int16))
        b_eq = torch.equal(k1.grad.view(torch.int16), k2.grad.view(torch.int16))
        ok &= f_eq and b_eq
        if verbose:
            print(f"k_rope_fused B{B} Hkv{Hkv} N{N} D{D} RD{RD}: fwd_bitwise={f_eq} bwd_bitwise={b_eq} "
                  f"flips_fwd={(r1.view(torch.int16) != r2.view(torch.int16)).sum().item()} "
                  f"flips_bwd={(k1.grad.view(torch.int16) != k2.grad.view(torch.int16)).sum().item()}")
    return ok


def replay_reference(dq_post: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                     in_dtype=torch.bfloat16) -> torch.Tensor:
    """The autograd q-RoPE backward written out op for op (the reference of selftest)."""
    rd = cos.shape[-1]
    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    half = rd // 2
    g_rot = dq_post[..., :rd]
    g32 = g_rot.float()
    gc = g32 * cos_u
    gs = g32 * sin_u
    d_xr = torch.zeros_like(g32)
    d_xr[..., half:] = -gs[..., :half]
    z1 = torch.zeros_like(g32)
    z1[..., :half] = gs[..., half:]
    d_xr = d_xr + z1 + gc
    d_xr_bf = d_xr.to(in_dtype)
    dq_pre = torch.zeros_like(dq_post)
    dq_pre[..., rd:] = dq_post[..., rd:]
    z2 = torch.zeros_like(dq_post)
    z2[..., :rd] = d_xr_bf
    dq_pre = dq_pre + z2
    return dq_pre


def selftest(verbose: bool = True) -> bool:
    ok = True
    torch.manual_seed(0)
    for (B, H, N, D, RD) in [(2, 16, 8192, 256, 256), (1, 8, 1024, 256, 256), (2, 4, 2048, 256, 64),
                             (1, 16, 512, 256, 128)]:
        dq = (torch.randn(B, N, H, D, device="cuda") * 3).to(torch.bfloat16)
        dq[0, :64, 0, :] = 0  # zero rows: the -0.0 corner
        pos = torch.arange(N, device="cuda").float()
        inv = 1.0 / (1e6 ** (torch.arange(0, RD, 2, device="cuda").float() / RD))
        fr = torch.einsum("t,d->td", pos, inv)
        emb = torch.cat([fr, fr], -1)
        cos = emb.cos().to(torch.bfloat16)[None].expand(B, N, RD).contiguous()
        sin = emb.sin().to(torch.bfloat16)[None].expand(B, N, RD).contiguous()
        ref = replay_reference(dq.permute(0, 2, 1, 3).contiguous(), cos, sin)
        got = rope_bwd_q(dq, cos, sin, H)
        eq = torch.equal(ref.view(torch.int16), got.view(torch.int16))
        ok &= eq and got.is_contiguous() and got.stride() == ref.stride()
        if verbose:
            nflip = (ref.view(torch.int16) != got.view(torch.int16)).sum().item()
            print(f"rope_bwd_q B{B} H{H} N{N} D{D} RD{RD}: bitwise={eq} flips={nflip} strides_equal={got.stride() == ref.stride()}")
    return ok


if __name__ == "__main__":
    import time
    ok = selftest()
    ok2 = selftest_krope()
    print("SELFTEST", "PASS" if (ok and ok2) else "FAIL")
    B, H, N, D = 2, 16, 8192, 256
    dq = torch.randn(B, N, H, D, device="cuda").to(torch.bfloat16)
    cos = torch.randn(B, N, D, device="cuda").to(torch.bfloat16)
    sin = torch.randn(B, N, D, device="cuda").to(torch.bfloat16)

    def bench(f, n=20):
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            f()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3
    print(f"fused rope_bwd_q: {bench(lambda: rope_bwd_q(dq, cos, sin, H)):.3f} ms  |  "
          f"replay (incl. contiguous copy): {bench(lambda: replay_reference(dq.permute(0, 2, 1, 3).contiguous(), cos, sin)):.3f} ms")
    raise SystemExit(0 if (ok and ok2) else 1)
