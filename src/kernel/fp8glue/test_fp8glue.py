#!/usr/bin/env python
"""fp8glue kernel unit tests (single GPU).  usage: python -m kernel.fp8glue.test_fp8glue [--quick]
(A script, not a pytest module: the sections are run_* functions and the exit code is the verdict.)
The cast_transpose and RMSNorm claims are BITWISE (torch.equal on uint8 views of the codes, fp32
scales exact); 'delayed' is checked against the same recipe written in torch.  The SwiGLU kernels
evaluate the compiled torchao MLP's expression with Triton's fp32 '/' (not IEEE), so they are
checked against the IEEE fp32 torch expression of swiglu_cast.py: amaxes within an ulp bound,
codes within one fp8 step; their twins, the recomputed h8^T and the fused dg|du buffer bitwise."""
from __future__ import annotations

import argparse
import sys
import time

import torch

from kernel.fp8glue import cast_transpose as CT
from kernel.fp8glue import rmsnorm_cast as RN
from kernel.fp8glue import swiglu_cast as SW
from kernel.fp8glue.common import (E4M3_MAX, E5M2_MAX, ref_cast_torchao, ref_scale_torchao_eager, scale_from_amax,
                                   scale_from_hist)
from kernel.fp8glue.scaling import LayerScalingState

E4, E5 = torch.float8_e4m3fn, torch.float8_e5m2
DEV = "cuda"
FAILS = []


def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {extra}")
    if not ok:
        FAILS.append(name)


def bits_eq(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def torchao_cast(x, dt):
    from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
    from torchao.float8.float8_training_tensor import LinearMMConfig
    t = hp_tensor_to_float8_dynamic(x, dt, LinearMMConfig())
    return t._data, t._scale


def compiled_cast_fn():
    from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
    from torchao.float8.float8_training_tensor import LinearMMConfig

    def f(x, dt):
        t = hp_tensor_to_float8_dynamic(x, dt, LinearMMConfig())
        return t._data, t._scale, t._scale.reciprocal()
    return torch.compile(f, dynamic=False)


# --------------------------------------------------------------------------- 1. scale rule
def run_scale_rule():
    print("1. scale rule (dynamic) vs torchao eager and vs torch.compile'd torchao")
    torch.manual_seed(0)
    am = torch.cat([torch.rand(4000, device=DEV) * 10, torch.rand(2000, device=DEV) * 1e-3,
                    torch.rand(2000, device=DEV) * 3e4, torch.tensor([0.0, 1e-13, 1e-12, 448.0, 57344.0, 1.0, 3.0, 7.0],
                                                                     device=DEV)])
    cf = compiled_cast_fn()
    n_eq_eager = n_eq_comp = n_inv_comp = 0
    for mx in (E4M3_MAX, E5M2_MAX):
        dt = E4 if mx == E4M3_MAX else E5
        ref = ref_scale_torchao_eager(am, mx)
        for i in range(am.numel()):
            s, inv = scale_from_amax(am[i:i + 1], mx)
            n_eq_eager += int(torch.equal(s, ref[i:i + 1]))
            # compiled torchao on a tensor whose amax is exactly am[i]
            x = torch.zeros(64, 32, device=DEV, dtype=torch.bfloat16)
            x[0, 0] = am[i].to(torch.bfloat16)
            if x[0, 0].float() != am[i]:
                continue   # not bf16-representable; the compiled path can only be probed on those
            d, sc, ric = cf(x, dt)
            n_eq_comp += int(torch.equal(s.view(()), sc.view(())))
            n_inv_comp += int(torch.equal(inv.view(()), ric.view(())))
    check("scale == torchao eager", n_eq_eager == 2 * am.numel(), f"{n_eq_eager}/{2 * am.numel()}")
    print(f"     compiled probe: scale eq {n_eq_comp}, inv eq {n_inv_comp} (of the bf16-representable amaxes)")


# --------------------------------------------------------------------------- 2. cast_transpose
def run_cast_transpose(quick):
    print("2. cast_transpose: codes bitwise vs torchao (eager), twin == data.t(), strided 4-D views")
    shapes = [(16384, 2560), (16384, 8192), (16384, 4096), (16384, 1024), (1001, 2560), (37, 640), (130, 96)]
    if quick:
        shapes = shapes[:1] + shapes[4:]
    for (M, N) in shapes:
        for dt in (E4, E5):
            torch.manual_seed(M * 31 + N)
            x = (torch.randn(M, N, device=DEV) * (3.0 if dt == E4 else 0.02)).to(torch.bfloat16)
            x[0, 0] = 1234.0 if dt == E4 else 0.0
            p = CT.cast_dynamic(x, dt, want_twin=True)
            rd, rs = torchao_cast(x, dt)
            ok = bits_eq(p.data, rd) and torch.equal(p.scale.view(()), rs.view(()))
            okT = bits_eq(p.twin, rd.t().contiguous())
            am_ok = torch.equal(p.amax.view(()), x.float().abs().max())
            check(f"cast {M}x{N} {str(dt)[-6:]}", ok and okT and am_ok,
                  f"codes={ok} twin={okT} amax={am_ok}")
    # strided (B, H, T, D)-contiguous grad viewed as (B, T, H, D): no permute copy
    B, H, T, D = 2, 16, 512, 256
    g = (torch.randn(B, H, T, D, device=DEV) * 0.01).to(torch.bfloat16)
    g4 = g.transpose(1, 2)                       # (B, T, H, D) view, strides (HTD, D, TD, 1)
    p = CT.cast_dynamic(g4, E5, want_twin=True, D=D)
    ref2d = g4.reshape(B * T, H * D)
    rd, rs = torchao_cast(ref2d, E5)
    check("strided (B,H,T,D) view cast", bits_eq(p.data, rd) and bits_eq(p.twin, rd.t().contiguous())
          and torch.equal(p.scale.view(()), rs.view(())))
    # 3-D contiguous [B, T, N] (the residual-stream grad)
    y = (torch.randn(2, 777, 2560, device=DEV) * 0.05).to(torch.bfloat16)
    p = CT.cast_dynamic(y, E5, want_twin=True)
    rd, rs = torchao_cast(y.view(-1, 2560), E5)
    check("3-D [B,T,N] cast", bits_eq(p.data, rd) and bits_eq(p.twin, rd.t().contiguous()))
    # fp8 transpose kernel
    x8 = torch.randint(0, 255, (1001, 2560), device=DEV, dtype=torch.uint8).view(E4)
    check("transpose_fp8", bits_eq(CT.transpose_fp8(x8), x8.t().contiguous()))
    # delayed-style one pass with a given scale + amax emission
    x = (torch.randn(4097, 2560, device=DEV)).to(torch.bfloat16)
    sc = torch.tensor([37.0], device=DEV)
    am = torch.zeros(1, device=DEV)
    out, outT = CT.cast_with_scale(x, sc, E4, amax_out=am)
    check("cast_with_scale(given scale)+amax", bits_eq(out, ref_cast_torchao(x, sc, E4)) and
          torch.equal(am.view(()), x.float().abs().max()) and bits_eq(outT, out.t().contiguous()))


# --------------------------------------------------------------------------- 3. swiglu
# The kernels use Triton's fp32 '/' (div.full.f32) and libdevice exp, so their fp32 values sit a few ulps from the
# IEEE torch expression (SW.ref_swiglu_fp32 / SW.ref_swiglu_bwd_fp32). dg has cancellation near its zero, so its
# elementwise ulp gap can be larger; its amax (a large |dg|) cannot.
H_ULP, DU_ULP, DG_ULP = 16, 16, 64   # fp32 ulps allowed between a kernel amax and the reference's
CODE_FRAC = 1e-4                      # fraction of fp8 codes allowed to differ from the reference's (one step each)


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> int:
    """Distance in fp32 ulps between two fp32 scalars (how many representable floats apart they are)."""
    def ordered(x):
        i = x.detach().reshape(1).float().view(torch.int32).to(torch.int64)
        return torch.where(i < 0, -(i & 0x7FFFFFFF), i)
    return int((ordered(a) - ordered(b)).abs().item())


def code_steps(a8: torch.Tensor, b8: torch.Tensor):
    """Two fp8 code tensors of one dtype -> (number of differing codes, largest distance in code steps).
    fp8 codes are sign-magnitude with the magnitude monotone in the low 7 bits, so on this signed integer
    line neighbouring values are one step apart (+0 and -0 coincide)."""
    def ordered(x8):
        c = x8.contiguous().view(torch.uint8).to(torch.int16)
        m = c & 0x7F
        return torch.where(c >= 0x80, -m, m)
    d = (ordered(a8) - ordered(b8)).abs()
    return int((d > 0).sum().item()), (int(d.max().item()) if d.numel() else 0)


def codes_close(name, got8, ref8):
    n, step = code_steps(got8, ref8)
    lim = max(16, int(CODE_FRAC * got8.numel()))
    check(f"{name} codes vs fp32 reference", step <= 1 and n <= lim,
          f"(differing {n}/{got8.numel()}, max {step} step, limit {lim})")


def run_swiglu(quick):
    print("3. SwiGLU fwd/bwd casts vs the IEEE fp32 torch expression: amaxes within an ulp bound, codes within one "
          "step; twins, recomputed h8^T and the fused dg|du buffer bitwise")
    import torch.nn.functional as F
    shapes = [(16384, 8192), (1001, 8192), (37, 256), (16384, 2560)]
    if quick:
        shapes = shapes[1:3]
    for (M, F_) in shapes:
        torch.manual_seed(M)
        g = (torch.randn(M, F_, device=DEV) * 2.0).to(torch.bfloat16)
        u = (torch.randn(M, F_, device=DEV) * 1.5).to(torch.bfloat16)
        dh = (torch.randn(M, F_, device=DEV) * 1e-3).to(torch.bfloat16)
        # ---- forward
        h32 = SW.ref_swiglu_fp32(g, u)
        am = SW.swiglu_fwd_amax(g, u)
        d = ulp_distance(am, h32.abs().max())
        check(f"swiglu fwd {M}x{F_} amax vs fp32 reference", d <= H_ULP, f"({d} ulp)")
        sc, inv = scale_from_amax(am, E4M3_MAX)
        h8, h8t = SW.swiglu_fwd_cast(g, u, sc, want_twin=True)
        codes_close(f"swiglu fwd {M}x{F_} h8", h8, ref_cast_torchao(h32, sc, E4))
        check(f"swiglu fwd {M}x{F_} twin", bits_eq(h8t, h8.t().contiguous()))
        del h32
        ed, _ = torchao_cast(F.silu(g) * u, E4)
        nde = (h8.view(torch.uint8) != ed.view(torch.uint8)).sum().item()
        print(f"     vs EAGER torchao (bf16-rounded h, its own scale): differing codes {nde}/{h8.numel()} "
              f"({100.0 * nde / h8.numel():.3f}%)")
        # ---- backward
        dg32, du32 = SW.ref_swiglu_bwd_fp32(g, u, dh)
        amg, amu = SW.swiglu_bwd_amax(g, u, dh)
        dgd, dud = ulp_distance(amg, dg32.abs().max()), ulp_distance(amu, du32.abs().max())
        check(f"swiglu bwd {M}x{F_} amaxes vs fp32 reference", dgd <= DG_ULP and dud <= DU_ULP,
              f"(dg {dgd} ulp, du {dud} ulp)")
        sg, _ = scale_from_amax(amg, E5M2_MAX)
        su, _ = scale_from_amax(amu, E5M2_MAX)
        dg8, dg8t, du8, du8t, h8t2, _ = SW.swiglu_bwd_cast(g, u, dh, sg, su, scale_h=sc)
        codes_close(f"swiglu bwd {M}x{F_} dg8", dg8, ref_cast_torchao(dg32, sg, E5))
        codes_close(f"swiglu bwd {M}x{F_} du8", du8, ref_cast_torchao(du32, su, E5))
        del dg32, du32
        check(f"swiglu bwd {M}x{F_} twins", bits_eq(dg8t, dg8.t().contiguous()) and bits_eq(du8t, du8.t().contiguous()))
        check(f"swiglu bwd {M}x{F_} recomputed h8^T == fwd twin", bits_eq(h8t2, h8t))
        dg8b, dg8tb, du8b, du8tb, _, buf = SW.swiglu_bwd_cast(g, u, dh, sg, su, fused_dgdu=True)
        check(f"swiglu bwd {M}x{F_} fused dg|du buffer", bits_eq(dg8b.contiguous(), dg8) and bits_eq(du8b.contiguous(), du8)
              and bits_eq(dg8tb, dg8t) and bits_eq(du8tb, du8t) and buf.shape == (M, 2 * F_))


# --------------------------------------------------------------------------- 4. rmsnorm
def run_rmsnorm(quick):
    print("4. RMSNorm: bf16 Y and rstd BITWISE vs Liger; amax exact; Function fwd/bwd bitwise vs Liger")
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction, rms_norm_forward
    shapes = [(16384, 2560), (1001, 2560), (4096, 1280), (37, 512), (16384, 4096)]
    if quick:
        shapes = shapes[1:3]
    for (M, N) in shapes:
        torch.manual_seed(N + M)
        x = (torch.randn(M, N, device=DEV) * 1.7).to(torch.bfloat16)
        w = (1.0 + 0.1 * torch.randn(N, device=DEV)).to(torch.bfloat16)
        eps = 1e-6
        Yl, _, RSTDl, BS, NW, _ = rms_norm_forward(x, w, eps, 0.0, "llama", None)
        am = torch.zeros(1, device=DEV)
        Y, RSTD = RN.rmsnorm_fwd(x, w, eps, amax_out=am)
        check(f"rmsnorm {M}x{N} Y bitwise vs Liger", bits_eq(Y, Yl),
              f"(diff {(Y.view(torch.int16) != Yl.view(torch.int16)).sum().item()})")
        check(f"rmsnorm {M}x{N} rstd bitwise", torch.equal(RSTD, RSTDl))
        check(f"rmsnorm {M}x{N} amax exact", torch.equal(am.view(()), Yl.float().abs().max()))
        # the norm-emitted pack (dynamic): cast of Y with the scale from the norm's amax == torchao(Y)
        sc, inv = scale_from_amax(am, E4M3_MAX)
        Q, QT = CT.cast_with_scale(Y, sc, E4, want_twin=True)
        rd, rs = torchao_cast(Yl, E4)
        check(f"rmsnorm {M}x{N} emitted x8 == torchao(Y), twin", bits_eq(Q, rd) and torch.equal(sc.view(()), rs.view(()))
              and bits_eq(QT, rd.t().contiguous()))
        # autograd Function: forward == Liger Y; backward == Liger backward bitwise (Liger writes dX
        # in place into dY, so every run gets its own copy of dy)
        dy = (torch.randn(M, N, device=DEV) * 0.01).to(torch.bfloat16)
        xa = x.clone().requires_grad_(True)
        wa = w.clone().requires_grad_(True)
        ya = LigerRMSNormFunction.apply(xa, wa, eps, 0.0, "llama")
        ya.backward(dy.clone())
        xb = x.clone().requires_grad_(True)
        wb = w.clone().requires_grad_(True)
        yb = RN.RMSNormCastFn.apply(xb, wb, eps, None)
        yb.backward(dy.clone())
        check(f"rmsnorm {M}x{N} Fn fwd/bwd bitwise vs Liger", bits_eq(ya, yb) and bits_eq(xa.grad, xb.grad)
              and bits_eq(wa.grad, wb.grad))


# --------------------------------------------------------------------------- 5. delayed scaling
def run_delayed():
    print("5. delayed scaling: history/scale semantics vs a torch reference of the same recipe")
    L = 16
    st = LayerScalingState(["a", "b"], DEV, mode="delayed", hist_len=L)
    from kernel.fp8glue import scaling as S
    torch.manual_seed(5)
    hist_ref = [[0.0] * L, [0.0] * L]
    ptr = 0
    seeded = [False, False]
    pending = False
    cur_ref = [0.0, 0.0]
    n_ok = 0
    n_sat = 0
    for mb in range(40):
        S.begin_microbatch()
        st.enter_microbatch()
        if pending:                      # reference push
            for r in range(2):
                hist_ref[r][ptr] = cur_ref[r]
                cur_ref[r] = 0.0
            ptr = (ptr + 1) % L
        pending = True
        for r, role in enumerate(("a", "b")):
            amp = 3.0 * (1.0 + 0.5 * torch.sin(torch.tensor(mb / 3.0 + r))) + (40.0 if mb == 25 and r == 0 else 0.0)
            x = (torch.randn(333, 2560, device=DEV) * amp).to(torch.bfloat16)
            out, outT, sc, inv = __import__("kernel.fp8glue.ops", fromlist=["_cast_role"])._cast_role(st, role, x, E4)
            amax_true = x.float().abs().max().item()
            if not seeded[r]:
                s_ref = ref_scale_torchao_eager(torch.tensor(amax_true, device=DEV), E4M3_MAX)
                seeded[r] = True
            else:
                s_ref = ref_scale_torchao_eager(torch.tensor(max(hist_ref[r]), device=DEV), E4M3_MAX)
            cur_ref[r] = max(cur_ref[r], amax_true)
            ref_codes = ref_cast_torchao(x, s_ref, E4)
            ok = bits_eq(out, ref_codes) and torch.equal(sc.view(()), s_ref.view(())) and bits_eq(outT, out.t().contiguous())
            n_ok += int(ok)
            if (x.float().abs().max() * s_ref).item() > E4M3_MAX:
                n_sat += 1
        # the device state must match the reference state
        hist_dev = st.hist[:2].cpu().tolist()
        if any(abs(hist_dev[r][i] - hist_ref[r][i]) > 0 for r in range(2) for i in range(L)):
            check(f"delayed history mismatch at mb {mb}", False)
            break
    check("delayed: codes/scales == torch recipe over 40 microbatches x 2 roles", n_ok == 80, f"{n_ok}/80, saturating casts seen: {n_sat}")
    check("delayed: seeded + pushes", all(st.seeded) and st.pushes == 39, f"pushes={st.pushes}")
    # recompute idempotence: casting the same role twice in one microbatch reuses the scale
    S.begin_microbatch()
    st.enter_microbatch()
    x = (torch.randn(200, 2560, device=DEV) * 5).to(torch.bfloat16)
    o1 = __import__("kernel.fp8glue.ops", fromlist=["_cast_role"])._cast_role(st, "a", x, E4)
    o2 = __import__("kernel.fp8glue.ops", fromlist=["_cast_role"])._cast_role(st, "a", x, E4)
    check("delayed: recompute (same mb) is bitwise", bits_eq(o1[0], o2[0]) and torch.equal(o1[2], o2[2]))
    # scale_from_hist == torch rule on random histories
    n = 0
    for i in range(200):
        h = torch.rand(L, device=DEV) * (10 ** torch.randint(-6, 5, (1,), device=DEV).float())
        s, inv = scale_from_hist(h, E5M2_MAX)
        n += int(torch.equal(s.view(()), ref_scale_torchao_eager(h.max(), E5M2_MAX).view(())))
    check("scale_from_hist == torchao rule on max(history)", n == 200, f"{n}/200")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    run_scale_rule()
    run_cast_transpose(a.quick)
    run_swiglu(a.quick)
    run_rmsnorm(a.quick)
    run_delayed()
    print(f"\nfp8glue unit tests: {'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)} ({time.time() - t0:.0f}s)")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
