"""The fp8 "cast glue" wired into the model: fused norm / activation + cast kernels, the amax taken inside the producing
kernel instead of a separate pass, one cast writing both GEMM layouts, and one shared q/k/v input cast (the mechanisms
Transformer Engine uses, re-implemented in Triton). DEFAULT ON (DM_FP8GLUE=1, the campaign's setting; DM_FP8GLUE=0
opts out).

DM_FP8GLUE=1 selects, at model build, `Fp8GlueSwiGLUMLP` (same parameters and FQNs as
SwiGLUMLP) and, after torchao's apply_fp8, `install_fp8glue` (parallel/wrap.py) wires:
  * the block's post-attention RMSNorm to EMIT an Fp8Pack (Liger-bitwise bf16 x_norm + its exact
    amax from the same kernel, then ONE cast pass writing the e4m3 codes and their transposed twin)
    that the MLP consumes directly: gate/up/down run as torch._scaled_mm on the codes, the fused
    SwiGLU+cast kernels replace the compiled torchao amax/cast passes, and the backward consumes
    the SAVED codes/twins (no re-cast, no .t().contiguous() copies);
  * on the dm mixer (DM_SHARED_X8=1) the input
    RMSNorm likewise emits the pack for the q/k/v projections (kernel/fp8glue/ops.py::Fp8GlueQKVFn),
    o_proj runs through Fp8GlueLinearFn (O cast once, both layouts saved), and dq/dk/dv are
    cast in their native (B, H, T, D) layout (no permute copies). DM_FP8GLUE_ATTN=0 keeps the
    attention side on the torchao path (attribution).
Scaling: DM_FP8GLUE_SCALING=dynamic (the default, torchao-bitwise) | delayed (an amax history as in Transformer
Engine's delayed scaling; a REALIZATION CHANGE, see kernel/fp8glue/scaling.py).  Other knobs: DM_FP8GLUE_SAVE_HT
(1: save h8^T from the forward; 0, the default: recompute it in the backward cast pass), DM_FP8GLUE_WCACHE (1: cache
the transposed fp8 weight per optimizer step), DM_FP8GLUE_SELFCHECK, DM_FP8GLUE_FUSED_TWIN.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn

from .layers import RMSNorm, SwiGLUMLP

ROLES = ["x_attn", "o", "dy_o", "dq", "dk", "dv", "x_mlp", "h", "dy", "dg", "du"]


def fp8glue_enabled() -> bool:
    return os.environ.get("DM_FP8GLUE", "1") == "1"


def _pack_cls():
    from kernel.fp8glue.common import Fp8Pack
    return Fp8Pack


class Fp8GlueSwiGLUMLP(SwiGLUMLP):
    """SwiGLUMLP whose forward takes the norm-emitted Fp8Pack once installed (else the plain path)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fp8glue = None

    def forward(self, x):
        if isinstance(x, torch.Tensor):   # the plain path (bf16 / CPU / eval builds): no Triton import
            return super().forward(x)
        Fp8Pack = _pack_cls()
        if isinstance(x, Fp8Pack):
            if self._fp8glue is None:
                x = x.carrier
            else:
                from kernel.fp8glue.ops import fp8glue_mlp
                return self.drop(fp8glue_mlp(x, self.gate_proj.weight, self.up_proj.weight,
                                             self.down_proj.weight, self._fp8glue))
        return super().forward(x)


def build_mlp(cfg) -> nn.Module:
    cls = Fp8GlueSwiGLUMLP if fp8glue_enabled() else SwiGLUMLP
    return cls(cfg.d_model, cfg.intermediate_size, cfg.resid_dropout, fused=cfg.fused_swiglu)


def norm_pack_forward(norm: RMSNorm, x: torch.Tensor):
    """RMSNorm forward that returns an Fp8Pack (codes + twin + scale) for an fp8 consumer.
    The bf16 carrier is Liger-bitwise x_norm; autograd routes the consumer's dx_norm through it
    into Liger's backward.  Dynamic / bootstrap: the norm kernel's amax IS the amax pass."""
    from kernel.fp8glue.cast_transpose import cast_with_scale
    from kernel.fp8glue.common import E4M3_MAX, Fp8Pack
    from kernel.fp8glue.rmsnorm_cast import RMSNormCastFn
    ctl, role = norm._fp8glue
    st = ctl.state
    p = st.plan(role, E4M3_MAX)
    Y = RMSNormCastFn.apply(x, norm.weight, norm.eps, p.amax)
    if p.two_pass:
        scale, inv = st.finalize(p)
    else:
        scale, inv = p.scale, p.inv
    with torch.no_grad():
        out, outT = cast_with_scale(Y.view(-1, Y.shape[-1]), scale, torch.float8_e4m3fn, want_twin=True)
    return Fp8Pack(out, outT, scale, inv, p.amax, torch.float8_e4m3fn, tuple(x.shape[:-1]), carrier=Y)


def mixer_qkv(mixer, pack):
    from kernel.fp8glue.ops import fp8glue_qkv
    return fp8glue_qkv(pack, mixer.q_proj.weight, mixer.k_proj.weight, mixer.v_proj.weight, mixer._fp8glue,
                       mixer.n_heads, mixer.n_kv_heads, mixer.head_dim)


def mixer_oproj(mixer, o):
    from kernel.fp8glue.ops import fp8glue_linear
    return fp8glue_linear(o, mixer.o_proj.weight, mixer._fp8glue, "o", "dy_o")


def install_fp8glue(model: nn.Module) -> dict:
    """Wire the ported path into every decoder block whose MLP linears are torchao Float8Linear
    (and, opt-out via DM_FP8GLUE_ATTN=0, the dm mixers with the shared x8 cast).  Also installs the
    microbatch / optimizer-step counters the scaling state needs.  Returns a summary dict."""
    from torchao.float8.float8_linear import Float8Linear
    from torch.optim.optimizer import register_optimizer_step_post_hook
    from kernel.fp8glue import FP8GLUE_VERSION
    from kernel.fp8glue.ops import GlueCtl, WeightCache
    from kernel.fp8glue.scaling import HIST_LEN, MODE, LayerScalingState, begin_microbatch, bump_epoch
    from .dm_attention import DMAttention

    attn_on = os.environ.get("DM_FP8GLUE_ATTN", "1") == "1"
    save_ht = os.environ.get("DM_FP8GLUE_SAVE_HT", "0") == "1"
    fused_dgdu = os.environ.get("DM_FP8GLUE_FUSED_DGDU", "0") == "1"
    wcache = WeightCache()
    dev = next(model.parameters()).device
    n_mlp = n_attn = 0
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if not isinstance(mlp, Fp8GlueSwiGLUMLP):
            continue
        if not all(isinstance(getattr(mlp, n), Float8Linear) for n in ("gate_proj", "up_proj", "down_proj")):
            continue
        st = LayerScalingState(ROLES, dev)
        ctl = GlueCtl(st, wcache, layer.layer_idx, save_ht=save_ht, fused_dgdu=fused_dgdu)
        mlp._fp8glue = ctl
        layer.post_attention_layernorm._fp8glue = (ctl, "x_mlp")
        layer._fp8glue = ctl
        n_mlp += 1
        mixer = layer.mixer
        if (attn_on and isinstance(mixer, DMAttention) and getattr(mixer, "_shared_x8", False)
                and not mixer.attn_output_gate
                and all(isinstance(getattr(mixer, n), Float8Linear) for n in ("q_proj", "k_proj", "v_proj", "o_proj"))):
            mixer._fp8glue = ctl
            layer.input_layernorm._fp8glue = (ctl, "x_attn")
            n_attn += 1
    if n_mlp:
        debug = os.environ.get("DM_FP8GLUE_DEBUG", "0") == "1"
        rank0 = int(os.environ.get("RANK", "0")) == 0

        def _pre(m, args):          # must return None (a non-None return REPLACES the forward args)
            from kernel.fp8glue.scaling import epoch
            mb = begin_microbatch()
            if debug and rank0 and mb % 8 == 1:
                print(f"[fp8glue] microbatch {mb} epoch {epoch()} wcache hits/misses {wcache.hits}/{wcache.misses}", flush=True)

        def _post_step(opt, args, kwargs):
            bump_epoch()
        model.register_forward_pre_hook(_pre)
        register_optimizer_step_post_hook(_post_step)
    info = {"version": FP8GLUE_VERSION, "scaling": MODE, "hist_len": HIST_LEN, "mlp_layers": n_mlp,
            "attn_layers": n_attn, "save_ht": save_ht, "wcache": wcache.enabled, "fused_dgdu": fused_dgdu}
    print(f"[fp8glue] installed: {info}", flush=True)
    return info
