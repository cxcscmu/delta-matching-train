"""Parity check: fused (Liger Triton) RMSNorm / RoPE / SwiGLU vs the eager primitives.

Verifies the opt-in fused paths (cfg.fused_rmsnorm / fused_rope / fused_swiglu) match the
eager math within bf16 fp tolerance, on forward AND backward (grad wrt inputs + weights),
including the per-head qk-norm shape and the YaRN cos/sin layout. Run on one GPU:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m pretrain.scripts.test_fused_layers
"""
import torch
import torch.distributed.tensor  # noqa: F401  (liger RMSNorm references torch.distributed.tensor.DTensor)

from pretrain.modeling.layers import RMSNorm, SwiGLUMLP, apply_rotary, RotaryEmbedding

dev = torch.device("cuda:0")
DT = torch.bfloat16


def relerr(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def maxerr(a, b):
    return (a.float() - b.float()).abs().max().item()


def check(name, eager_out, fused_out, eager_g, fused_g, tol=2e-2):
    ofe, ofg = relerr(fused_out, eager_out), max(relerr(fg, eg) for fg, eg in zip(fused_g, eager_g))
    ok = ofe < tol and ofg < tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name:18s} fwd relL2={ofe:.2e} (max {maxerr(fused_out,eager_out):.2e})  "
          f"bwd relL2={ofg:.2e}")
    return ok


def run_rmsnorm(dim, shape, tag):
    torch.manual_seed(0)
    x = torch.randn(*shape, dim, device=dev, dtype=DT, requires_grad=True)
    w = torch.randn(dim, device=dev, dtype=DT)
    ea = RMSNorm(dim, 1e-5, in_fp32=True, fused=False).to(dev); ea.weight.data = w.clone()
    fu = RMSNorm(dim, 1e-5, in_fp32=True, fused=True).to(dev);  fu.weight.data = w.clone()
    assert fu.fused, "fused RMSNorm did not enable (liger missing?)"
    xe = x.clone().detach().requires_grad_(True); xf = x.clone().detach().requires_grad_(True)
    oe = ea(xe); of = fu(xf)
    g = torch.randn_like(oe)
    oe.backward(g); of.backward(g)
    return check(f"RMSNorm {tag}", oe, of, [xe.grad, ea.weight.grad], [xf.grad, fu.weight.grad])


def run_rope():
    torch.manual_seed(1)
    B, H, Hkv, T, D = 2, 16, 4, 1024, 64
    rot = RotaryEmbedding(D, 65536, 10000.0,
                          rope_scaling={"type": "yarn", "factor": 8.0,
                                        "original_max_position_embeddings": 8192,
                                        "beta_fast": 32, "beta_slow": 1}).to(dev)
    pos = torch.arange(T, device=dev).unsqueeze(0).expand(B, -1)
    cos, sin = rot(pos); cos, sin = cos.to(DT), sin.to(DT)
    q = torch.randn(B, H, T, D, device=dev, dtype=DT)
    k = torch.randn(B, Hkv, T, D, device=dev, dtype=DT)
    qe = q.clone().requires_grad_(True); ke = k.clone().requires_grad_(True)
    qf = q.clone().requires_grad_(True); kf = k.clone().requires_grad_(True)
    qoe, koe = apply_rotary(qe, ke, cos, sin, fused=False)
    qof, kof = apply_rotary(qf, kf, cos, sin, fused=True)
    gq, gk = torch.randn_like(qoe), torch.randn_like(koe)
    (qoe * gq).sum().add_((koe * gk).sum()).backward()
    (qof * gq).sum().add_((kof * gk).sum()).backward()
    ok1 = check("RoPE q", qoe, qof, [qe.grad], [qf.grad])
    ok2 = check("RoPE k", koe, kof, [ke.grad], [kf.grad])
    return ok1 and ok2


def run_swiglu():
    torch.manual_seed(2)
    d, dff, B, T = 1024, 3584, 2, 1024
    x = torch.randn(B, T, d, device=dev, dtype=DT)
    ea = SwiGLUMLP(d, dff, 0.0, fused=False).to(dev).to(DT)
    fu = SwiGLUMLP(d, dff, 0.0, fused=True).to(dev).to(DT)
    assert fu.fused, "fused SwiGLU did not enable"
    fu.gate_proj.weight.data = ea.gate_proj.weight.data.clone()
    fu.up_proj.weight.data = ea.up_proj.weight.data.clone()
    fu.down_proj.weight.data = ea.down_proj.weight.data.clone()
    xe = x.clone().requires_grad_(True); xf = x.clone().requires_grad_(True)
    oe = ea(xe); of = fu(xf)
    g = torch.randn_like(oe)
    oe.backward(g); of.backward(g)
    return check("SwiGLU", oe, of, [xe.grad, ea.gate_proj.weight.grad, ea.down_proj.weight.grad],
                 [xf.grad, fu.gate_proj.weight.grad, fu.down_proj.weight.grad])


if __name__ == "__main__":
    oks = []
    oks.append(run_rmsnorm(1024, (4, 512), "d_model"))
    oks.append(run_rmsnorm(64, (2, 16, 512), "qk_norm/head"))
    oks.append(run_rope())
    oks.append(run_swiglu())
    print("\nALL PASS" if all(oks) else "\nSOME FAILED")
    raise SystemExit(0 if all(oks) else 1)
