"""Autograd glue: the fp8 MLP (P1+P4), the shared-x8 q/k/v projection and the generic fp8 linear
(attention-side P4), all consuming Fp8Pack operands and torch._scaled_mm exactly as torchao's
addmm_float8_unwrapped does (reciprocal scales, out_dtype bf16, use_fast_accum True for the output
GEMM and False for grad_input / grad_weight), with the codes and twins SAVED instead of the bf16
inputs: no re-cast and no .t().contiguous() copies in backward.

Weights arrive as FSDP2 delivers them: a Float8TrainingTensor (the fp8 all-gather: `_data`
[out, in] e4m3 + `_scale`) or a bf16 tensor (fp8 all-gather off, DM_FP8_FSDP_AG=0: cast here as
torchao would).  The dgrad needs
the weight in the OTHER layout (W^T [in, out] row-major); it is derived once per optimizer-step
epoch and cached (weights are static within a step; env DM_FP8GLUE_WCACHE=0 disables).  The live
`_data` of an unsharded FSDP param is never cached (its storage is resized to 0 at reshard).
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch

from .cast_transpose import add3, amax_of, cast_dynamic, cast_with_scale, transpose_fp8
from .common import E4M3_MAX, E5M2_MAX, Fp8Pack, recip_like_inductor
from .scaling import LayerScalingState, epoch
from .swiglu_cast import swiglu_bwd_amax, swiglu_bwd_cast, swiglu_fwd_amax, swiglu_fwd_cast

E4 = torch.float8_e4m3fn
E5 = torch.float8_e5m2
BF16 = torch.bfloat16


# ------------------------------------------------------------------------------ weights
def _unwrap_weight(w: torch.Tensor):
    from torchao.float8.float8_training_tensor import Float8TrainingTensor
    from torchao.float8.fsdp_utils import WeightWithDynamicFloat8CastTensor
    import torch.distributed._functional_collectives as funcol
    t = w
    for _ in range(6):
        if isinstance(t, Float8TrainingTensor):
            return "fp8", t._data, t._scale
        if isinstance(t, funcol.AsyncCollectiveTensor):
            t = t.elem
            continue
        if isinstance(t, WeightWithDynamicFloat8CastTensor):
            t = t._tensor
            continue
        if isinstance(t, torch.distributed.tensor.DTensor):
            t = t._local_tensor
            continue
        break
    return "bf16", t, None


class WeightCache:
    def __init__(self, enabled: Optional[bool] = None):
        self.d: Dict[object, tuple] = {}
        self.enabled = (os.environ.get("DM_FP8GLUE_WCACHE", "1") == "1") if enabled is None else enabled
        self.hits = 0
        self.misses = 0

    def get(self, key, ep):
        e = self.d.get(key)
        if e is not None and e[0] == ep:
            self.hits += 1
            return e[1]
        self.misses += 1
        return None

    def put(self, key, ep, val):
        self.d[key] = (ep, val)


def weight_operands(w: torch.Tensor, key, cache: WeightCache, need_bwd: bool = True):
    """-> (W8 [out, in] fp8 row-major for the forward GEMM (mat2 = W8.t()),
           W8T [in, out] fp8 row-major for the dgrad GEMM (mat2 = W8T.t()) or None,
           inv fp32 [1] = 1/scale the way the compiled torchao graph forms it)."""
    kind, data, scale = _unwrap_weight(w)
    ep = epoch() if cache.enabled else -1
    ent = cache.get(key, ep) if cache.enabled else None
    if kind == "fp8":
        W8 = data if data.is_contiguous() else data.contiguous()
        if ent is None or (need_bwd and ent[0] is None):
            inv = recip_like_inductor(scale.reshape(1).to(torch.float32))
            W8T = transpose_fp8(W8) if need_bwd else None
            ent = (W8T, inv)
            if cache.enabled:
                cache.put(key, ep, ent)
        return W8, ent[0], ent[1]
    # bf16 weight (fp8 all-gather off): torchao's dynamic tensorwise e4m3 cast, both layouts in one pass
    if ent is None:
        p = cast_dynamic(data.detach(), E4, want_twin=True)
        ent = (p.data, p.twin, p.inv_scale)
        if cache.enabled:
            cache.put(key, ep, ent)
    return ent[0], ent[1], ent[2]


class GlueCtl:
    """Per-layer control block handed to the autograd Functions (not a tensor)."""

    def __init__(self, state: LayerScalingState, wcache: WeightCache, layer_idx: int,
                 save_ht: bool = True, fused_dgdu: bool = False):
        self.state = state
        self.wcache = wcache
        self.layer_idx = layer_idx
        self.save_ht = save_ht
        self.fused_dgdu = fused_dgdu

    def key(self, name: str):
        return (self.layer_idx, name)


# ------------------------------------------------------------------------------ casts
def _cast_role(st: LayerScalingState, role: str, x: torch.Tensor, fp8_dtype, *, want_twin=True,
               D: Optional[int] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """x -> (codes rowwise [M, N], twin [N, M] | None, scale, inv) under the role's scaling plan."""
    fp8_max = E4M3_MAX if fp8_dtype == E4 else E5M2_MAX
    p = st.plan(role, fp8_max)
    if p.two_pass:
        amax_of(x, p.amax, D=D)
        scale, inv = st.finalize(p)
        out, outT = cast_with_scale(x, scale, fp8_dtype, want_twin=want_twin, D=D)
    else:
        scale, inv = p.scale, p.inv
        out, outT = cast_with_scale(x, scale, fp8_dtype, want_twin=want_twin, D=D, amax_out=p.amax)
    return out, outT, scale, inv


def _mm(a8, b8_colmajor, inv_a, inv_b, fast):
    return torch._scaled_mm(a8, b8_colmajor, scale_a=inv_a.view(()), scale_b=inv_b.view(()), bias=None,
                            scale_result=None, out_dtype=BF16, use_fast_accum=fast)


# ------------------------------------------------------------------------------ MLP
class Fp8GlueMLPFn(torch.autograd.Function):
    """y = down(silu(gate(x)) * up(x)) on an Fp8Pack x (codes + twin + scale), fp8 everywhere."""

    @staticmethod
    def forward(ctx, carrier, x8, x8t, x_inv, wg, wu, wd, ctl: GlueCtl):
        st = ctl.state
        Wg8, _, g_inv = weight_operands(wg, ctl.key("g"), ctl.wcache, need_bwd=ctl.wcache.enabled)
        Wu8, _, u_inv = weight_operands(wu, ctl.key("u"), ctl.wcache, need_bwd=ctl.wcache.enabled)
        Wd8, _, d_inv = weight_operands(wd, ctl.key("d"), ctl.wcache, need_bwd=ctl.wcache.enabled)
        g = _mm(x8, Wg8.t(), x_inv, g_inv, True)
        u = _mm(x8, Wu8.t(), x_inv, u_inv, True)
        p = st.plan("h", E4M3_MAX)
        if p.two_pass:
            swiglu_fwd_amax(g, u, p.amax)
            scale_h, inv_h = st.finalize(p)
            h8, h8t = swiglu_fwd_cast(g, u, scale_h, want_twin=ctl.save_ht)
        else:
            scale_h, inv_h = p.scale, p.inv
            h8, h8t = swiglu_fwd_cast(g, u, scale_h, want_twin=ctl.save_ht, amax_out=p.amax)
        y = _mm(h8, Wd8.t(), inv_h, d_inv, True)
        ctx.ctl = ctl
        ctx.lead = tuple(carrier.shape[:-1])
        ctx.save_for_backward(x8t, x_inv, g, u, h8t, inv_h, scale_h, wg, wu, wd)
        return y.view(*ctx.lead, y.shape[-1])

    @staticmethod
    def backward(ctx, dy):
        x8t, x_inv, g, u, h8t, inv_h, scale_h, wg, wu, wd = ctx.saved_tensors
        ctl: GlueCtl = ctx.ctl
        st = ctl.state
        d, M = x8t.shape
        dy2 = dy.contiguous().view(M, -1)
        # 1. dy -> e5m2 rowwise (dgrad) + twin (down wgrad)
        dy8, dy8t, _, inv_dy = _cast_role(st, "dy", dy2, E5)
        # 2. dh = dy8 @ Wd
        _, Wd8T, d_inv = weight_operands(wd, ctl.key("d"), ctl.wcache)
        dh = _mm(dy8, Wd8T.t(), inv_dy, d_inv, False)
        # 3. dg/du (e5m2, both layouts) in one read of dh,g,u (+ h8^T recomputed if not saved)
        pg = st.plan("dg", E5M2_MAX)
        pu = st.plan("du", E5M2_MAX)
        assert pg.two_pass == pu.two_pass
        sh = None if h8t is not None else scale_h
        if pg.two_pass:
            swiglu_bwd_amax(g, u, dh, pg.amax, pu.amax)
            sg, inv_dg = st.finalize(pg)
            su, inv_du = st.finalize(pu)
            dg8, dg8t, du8, du8t, h8t_rec, _ = swiglu_bwd_cast(g, u, dh, sg, su, scale_h=sh,
                                                                fused_dgdu=ctl.fused_dgdu)
        else:
            sg, inv_dg, su, inv_du = pg.scale, pg.inv, pu.scale, pu.inv
            dg8, dg8t, du8, du8t, h8t_rec, _ = swiglu_bwd_cast(g, u, dh, sg, su, scale_h=sh,
                                                                amax_dg=pg.amax, amax_du=pu.amax,
                                                                fused_dgdu=ctl.fused_dgdu)
        if h8t is None:
            h8t = h8t_rec
        # 4. wgrad down: dWd [d, F] = dy8^T [d, M] @ h8 ([M, F] column-major = h8t.t())
        dWd = _mm(dy8t, h8t.t(), inv_dy, inv_h, False)
        # 5. dgrads into x: dx = dg8 @ Wg + du8 @ Wu  (bf16 add, fp32 inside, as Inductor's)
        _, Wg8T, g_inv = weight_operands(wg, ctl.key("g"), ctl.wcache)
        _, Wu8T, u_inv = weight_operands(wu, ctl.key("u"), ctl.wcache)
        dx = _mm(dg8, Wg8T.t(), inv_dg, g_inv, False)
        dx = dx + _mm(du8, Wu8T.t(), inv_du, u_inv, False)
        # 6. wgrads gate/up: dW [F, d] = dg8^T [F, M] @ x8 ([M, d] column-major = x8t.t())
        dWg = _mm(dg8t, x8t.t(), inv_dg, x_inv, False)
        dWu = _mm(du8t, x8t.t(), inv_du, x_inv, False)
        return dx.view(*ctx.lead, d), None, None, None, dWg, dWu, dWd, None


def fp8glue_mlp(pack: Fp8Pack, wg, wu, wd, ctl: GlueCtl) -> torch.Tensor:
    return Fp8GlueMLPFn.apply(pack.carrier, pack.data, pack.twin, pack.inv_scale, wg, wu, wd, ctl)


# ------------------------------------------------------------------------------ q/k/v
class Fp8GlueQKVFn(torch.autograd.Function):
    """q, k, v = x8 @ Wq^T, Wk^T, Wv^T on the norm-emitted pack; outputs shaped [B, T, H, D] so
    the gradients arrive as strided (B, H, T, D) views that the cast reads in place."""

    @staticmethod
    def forward(ctx, carrier, x8, x8t, x_inv, wq, wk, wv, ctl: GlueCtl, Hq: int, Hkv: int, D: int):
        Wq8, _, q_inv = weight_operands(wq, ctl.key("q"), ctl.wcache, need_bwd=ctl.wcache.enabled)
        Wk8, _, k_inv = weight_operands(wk, ctl.key("k"), ctl.wcache, need_bwd=ctl.wcache.enabled)
        Wv8, _, v_inv = weight_operands(wv, ctl.key("v"), ctl.wcache, need_bwd=ctl.wcache.enabled)
        q = _mm(x8, Wq8.t(), x_inv, q_inv, True)
        k = _mm(x8, Wk8.t(), x_inv, k_inv, True)
        v = _mm(x8, Wv8.t(), x_inv, v_inv, True)
        lead = tuple(carrier.shape[:-1])
        ctx.ctl, ctx.lead, ctx.D = ctl, lead, D
        ctx.save_for_backward(x8t, x_inv, wq, wk, wv)
        return q.view(*lead, Hq, D), k.view(*lead, Hkv, D), v.view(*lead, Hkv, D)

    @staticmethod
    def backward(ctx, dq, dk, dv):
        x8t, x_inv, wq, wk, wv = ctx.saved_tensors
        ctl: GlueCtl = ctx.ctl
        st = ctl.state
        D = ctx.D
        d = x8t.shape[0]
        dq8, dq8t, _, inv_q = _cast_role(st, "dq", dq, E5, D=D)
        dk8, dk8t, _, inv_k = _cast_role(st, "dk", dk, E5, D=D)
        dv8, dv8t, _, inv_v = _cast_role(st, "dv", dv, E5, D=D)
        _, Wq8T, q_inv = weight_operands(wq, ctl.key("q"), ctl.wcache)
        _, Wk8T, k_inv = weight_operands(wk, ctl.key("k"), ctl.wcache)
        _, Wv8T, v_inv = weight_operands(wv, ctl.key("v"), ctl.wcache)
        # the three dgrads into x summed once in fp32 (single bf16 rounding), as the compiled
        # shared fn of the dm mixer does; the association is DM_FP8GLUE_ADD3_ORDER (gated)
        dx = add3(_mm(dq8, Wq8T.t(), inv_q, q_inv, False), _mm(dk8, Wk8T.t(), inv_k, k_inv, False),
                  _mm(dv8, Wv8T.t(), inv_v, v_inv, False))
        dWq = _mm(dq8t, x8t.t(), inv_q, x_inv, False)
        dWk = _mm(dk8t, x8t.t(), inv_k, x_inv, False)
        dWv = _mm(dv8t, x8t.t(), inv_v, x_inv, False)
        return dx.view(*ctx.lead, d), None, None, None, dWq, dWk, dWv, None, None, None, None


def fp8glue_qkv(pack: Fp8Pack, wq, wk, wv, ctl: GlueCtl, Hq: int, Hkv: int, D: int):
    return Fp8GlueQKVFn.apply(pack.carrier, pack.data, pack.twin, pack.inv_scale, wq, wk, wv, ctl, Hq, Hkv, D)


# ------------------------------------------------------------------------------ linear (o_proj)
class Fp8GlueLinearFn(torch.autograd.Function):
    """y = x @ W^T with x cast ONCE (rowwise + twin; the twin is saved for the wgrad, so the backward
    never re-casts x) and dy cast once in both layouts."""

    @staticmethod
    def forward(ctx, x, w, ctl: GlueCtl, role_x: str, role_dy: str):
        lead = tuple(x.shape[:-1])
        x2 = x.contiguous().view(-1, x.shape[-1])
        x8, x8t, _, inv_x = _cast_role(ctl.state, role_x, x2, E4)
        W8, _, w_inv = weight_operands(w, ctl.key(role_x), ctl.wcache, need_bwd=ctl.wcache.enabled)
        y = _mm(x8, W8.t(), inv_x, w_inv, True)
        ctx.ctl, ctx.lead, ctx.roles = ctl, lead, (role_x, role_dy)
        ctx.save_for_backward(x8t, inv_x, w)
        return y.view(*lead, y.shape[-1])

    @staticmethod
    def backward(ctx, dy):
        x8t, inv_x, w = ctx.saved_tensors
        ctl: GlueCtl = ctx.ctl
        role_x, role_dy = ctx.roles
        n_in, M = x8t.shape
        dy2 = dy.contiguous().view(M, -1)
        dy8, dy8t, _, inv_dy = _cast_role(ctl.state, role_dy, dy2, E5)
        _, W8T, w_inv = weight_operands(w, ctl.key(role_x), ctl.wcache)
        dx = _mm(dy8, W8T.t(), inv_dy, w_inv, False)
        dW = _mm(dy8t, x8t.t(), inv_dy, inv_x, False)
        return dx.view(*ctx.lead, n_in), dW, None, None, None


def fp8glue_linear(x, w, ctl: GlueCtl, role_x: str, role_dy: str):
    return Fp8GlueLinearFn.apply(x, w, ctl, role_x, role_dy)
