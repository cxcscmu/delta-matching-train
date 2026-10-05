"""GPU checks of the compute stack: the Delta-Matching-128 kernel (the delta_matching package) against an fp32
reference (both delta modes), its determinism, the structural identities between the modes, torchao's fp8 linear, a tiny
hybrid model's forward and backward (GatedDeltaNet via fla + causal_conv1d, attention, RMSNorm, the loss), and the dm
mixer's build from the package."""
import math, os, subprocess
import pytest
import torch
from _helpers import require_kernel, require_standard_kernel, kernel_bf16dp, python, base_env, dm_env, dm_route, LN_VOCAB

pytestmark = pytest.mark.gpu
# Relative-norm bounds against fp32 attention. Every operand (q, k, v, P, dO) is rounded to e4m3, so the error is the
# fp8 rounding floor: measured with these Gaussian inputs O 0.036, dV 0.038, dQ 0.055, dK 0.056 (real activations sit
# lower). The bounds are 2x the measurement; a layout, scale or descale mistake reads 0.5 to 1.4 here, far above them.
TOL = {"O": 8e-2, "dV": 8e-2, "dQ": 1.2e-1, "dK": 1.2e-1}


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def _inputs(seed, B=1, H=4, Hk=1, N=1024, D=128):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(B, H, N, D, device="cuda", generator=g) * 0.5).bfloat16()
    k = (torch.randn(B, Hk, N, D, device="cuda", generator=g) * 0.5).bfloat16()
    v = (torch.randn(B, Hk, N, D, device="cuda", generator=g) * 0.5).bfloat16()
    go = (torch.randn(B, N, H, D, device="cuda", generator=g) * 0.1).bfloat16()
    return q, k, v, go


def _kernel(mode, seed):
    require_kernel()
    require_standard_kernel(f"the {mode} kernel through dm_attn_func_bhnd")
    from delta_matching import dm_attn_func_bhnd
    q, k, v, go = _inputs(seed)
    q, k, v = q.clone().requires_grad_(), k.clone().requires_grad_(), v.clone().requires_grad_()
    o = dm_attn_func_bhnd(q, k, v, causal=True, delta_mode=mode)
    assert o.shape == go.shape, (o.shape, go.shape)
    o.backward(go)
    return o.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()


def _reference(seed):
    q, k, v, go = _inputs(seed)
    rep = q.shape[1] // k.shape[1]
    q32 = q.float().requires_grad_(); k32 = k.float().requires_grad_(); v32 = v.float().requires_grad_()
    kk = k32.repeat_interleave(rep, dim=1); vv = v32.repeat_interleave(rep, dim=1)
    o = torch.nn.functional.scaled_dot_product_attention(q32, kk, vv, is_causal=True).transpose(1, 2)
    o.backward(go.float())
    return o.detach(), q32.grad, k32.grad, v32.grad


@pytest.mark.parametrize("mode", ["stale", "matched"])
def test_dm128_matches_fp32_reference(mode):
    o, dq, dk, dv = _kernel(mode, seed=0)
    ro, rdq, rdk, rdv = _reference(0)
    for t in (o, dq, dk, dv):
        assert torch.isfinite(t.float()).all()
    errs = {"O": _rel(o, ro), "dQ": _rel(dq, rdq), "dK": _rel(dk, rdk), "dV": _rel(dv, rdv)}
    bad = {k: e for k, e in errs.items() if e > TOL[k]}
    print(f"{mode}: relative errors vs fp32 {errs}")
    assert not bad, f"{mode}: relative errors {errs} exceed {TOL}"


def test_dm128_is_deterministic():
    a = _kernel("stale", seed=1); b = _kernel("stale", seed=1)
    for x, y, name in zip(a, b, ("O", "dQ", "dK", "dV")):
        assert torch.equal(x, y), f"{name} differs between two identical runs"


def test_delta_modes_share_forward_and_dv():
    """The two deltas differ only in the backward's dQ/dK: the forward and dV = P^T dO never read the delta."""
    s = _kernel("stale", seed=2); m = _kernel("matched", seed=2)
    assert torch.equal(s[0], m[0]), "forward O should be bitwise identical across delta modes"
    assert torch.equal(s[3], m[3]), "dV should be bitwise identical across delta modes"
    assert not torch.equal(s[1], m[1]), "dQ should differ between the stale and the matched delta"


def test_torchao_float8_linear_runs():
    from torchao.float8 import convert_to_float8_training
    torch.manual_seed(0)
    ref = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.Linear(512, 256)).cuda().bfloat16()
    m = convert_to_float8_training(__import__("copy").deepcopy(ref))
    x = torch.randn(64, 256, device="cuda").bfloat16()
    y8 = m(x); y = ref(x)
    # two chained dynamic-scaled e4m3 GEMMs on Gaussian weights: measured 0.051 relative to bf16
    assert torch.isfinite(y8.float()).all() and _rel(y8, y) < 1e-1, _rel(y8, y)
    y8.float().pow(2).mean().backward()
    assert all(torch.isfinite(p.grad.float()).all() for p in m.parameters())


def test_tiny_hybrid_model_forward_backward():
    from pretrain.modeling import build_model
    from pretrain.modeling.config import ModelConfig
    cfg = ModelConfig(vocab_size=32000, d_model=256, n_layers=4, n_heads=4, n_kv_heads=1, head_dim=128, max_seq_len=512,
                      layer_mixers=["gated_deltanet", "gated_deltanet", "gated_deltanet", "gqa"], intermediate_size=512,
                      gdn_head_dim=64, gdn_num_heads=4, attn_output_gate=True, partial_rotary_factor=0.25, qk_norm=True,
                      attn_impl="sdpa", rope_theta=1e7, norm_eps=1e-6, tie_embeddings=True)
    torch.manual_seed(0)
    model = build_model(cfg).cuda()
    ids = torch.randint(0, 32000, (2, 512), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids, labels=ids)
    loss = out["loss"].float()
    assert math.isfinite(loss.item()) and abs(loss.item() - LN_VOCAB) < 1.0, loss.item()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


_MIXER_CODE = ("from pretrain.modeling.config import ModelConfig; from pretrain.modeling.mixers import build_mixer; "
               "cfg = ModelConfig(vocab_size=32000, d_model=256, n_layers=4, n_heads=4, n_kv_heads=1, head_dim=128, max_seq_len=512, "
               "layer_mixers=['gated_deltanet','gated_deltanet','gated_deltanet','dm'], intermediate_size=512, gdn_head_dim=64, gdn_num_heads=4, "
               "attn_output_gate=True, partial_rotary_factor=0.25, qk_norm=True, dm_mode='{mode}', dm_fused_producers=True); "
               "m = build_mixer('dm', cfg, 3); print('BUILT', type(m).__name__)")


@pytest.mark.parametrize("mode", ["stale", "matched"])
def test_dm_mixer_builds_from_package(mode):
    """The dm (fp8) mixer builds against the installed delta_matching package under the launcher's environment and
    announces its route (the `[dm] DMAttention` banner names the kernel build, the orchestrator version and dm_mode)."""
    require_kernel()
    require_standard_kernel(f"the dm mixer in dm_mode {mode}")
    env = base_env(os.environ.get("DM_TEST_GPU", "0")); env.update(dm_env(stale=(mode == "stale")))
    p = subprocess.run([python(), "-c", _MIXER_CODE.format(mode=mode)], capture_output=True, text=True, env=env, timeout=600)
    assert p.returncode == 0 and "BUILT DMAttention" in p.stdout, f"rc={p.returncode}\n{p.stderr[-2500:]}"
    assert dm_route(p.stdout).get("dm") == mode, f"the dm route was not announced with dm_mode={mode}:\n{p.stdout[-2000:]}"


def test_dm_mixer_checks_the_bf16dp_build_flag():
    """dm_mode bf16_dp needs a library built with DM_BF16_DP=1 (build_bf16dp() == 1) and the other modes need one built
    without it: the mixer asserts the flag against the mode at model build instead of inferring the numerics from the
    arm name, so a mismatch fails loudly."""
    require_kernel()
    flag = int(kernel_bf16dp())            # the loaded library's build flag (DM_SO honored), read once per session
    env = base_env(os.environ.get("DM_TEST_GPU", "0")); env.update(dm_env(stale=True))
    p = subprocess.run([python(), "-c", _MIXER_CODE.format(mode="bf16_dp")], capture_output=True, text=True, env=env, timeout=600)
    if flag == 1:   # the bf16-dP variant: bf16_dp builds, a stale mixer is refused (the other direction)
        assert p.returncode == 0 and "BUILT DMAttention" in p.stdout, f"rc={p.returncode}\n{p.stderr[-2500:]}"
        assert dm_route(p.stdout).get("dm") == "bf16_dp", p.stdout[-2000:]
        p = subprocess.run([python(), "-c", _MIXER_CODE.format(mode="stale")], capture_output=True, text=True, env=env, timeout=600)
        assert p.returncode != 0 and "DM_BF16_DP=0" in p.stderr, \
            f"a stale mixer on a DM_BF16_DP=1 library must refuse; rc={p.returncode}\n{p.stderr[-2500:]}"
    else:
        assert p.returncode != 0 and "DM_BF16_DP=1" in p.stderr, \
            f"a bf16_dp mixer on a library without DM_BF16_DP=1 must refuse; rc={p.returncode}\n{p.stderr[-2500:]}"
