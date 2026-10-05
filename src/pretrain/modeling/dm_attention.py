"""The `dm` mixer: GQA attention whose core is the fp8 Delta-Matching kernel (the `delta_matching` package) fed by the
fused fp8 producers of src/kernel/dm_port.  The paper's fp8 attention arms differ only in `dm_mode`, the backward's
delta realization:
    stale     delta = rowsum(dO * O) from the forward's O          (the "naive fp8" arm)
    matched   delta from the backward's own realized fp8 P / dP     (Delta-Matching)
    bf16_dp   the stale delta with the dP = dO V^T matmul kept in bf16 (the one-matmul exemption studied as the paper's
              bf16-dP arm; head_dim 128 only; the kernel built with DM_BF16_DP=1, checked through build_bf16dp())

Parameter-identical to GQAAttention (same submodules, same registration ORDER, so the same seed gives bit-identical
init across arms: the Standard arm is plain `mixer: gqa`).  Only the attention call differs.

The execution path (one; the one the paper's runs used):
  * q/k/v projections through torchao Float8Linear (fp8_scope ffn_qkvo): the input x cast to e4m3 ONCE and shared by
    the three GEMMs and their wgrads (DM_SHARED_X8=1, the default; 0 = three separate torchao casts, bitwise the same
    codes), or, with the fp8 cast glue installed (DM_FP8GLUE=1, the default; modeling/mlp_fp8glue.py), the
    RMSNorm-emitted Fp8Pack feeding the three GEMMs (kernel/fp8glue/ops.py::Fp8GlueQKVFn);
  * qk-norm; the k RoPE as one fused bitwise kernel each way (dm_port/fused_producers/m13_rope_bwd.k_rope_fused);
  * V quantized per KV head before the GQA expansion (dm_port/fused_producers/tier1_vproj.v8_quantize: 64 keys per
    scale at head_dim 128, 32 at 256);
  * the orchestrator (kernel/dm_port/orchestrator.py): the q RoPE + Q codes / scales / twin in one pass, the K codes /
    scales / twin, fwd_fp8; in the backward the dO codes / scales / twin, the arm's delta, bwd_fp8_t (bwd_fp8_bf16dp for
    bf16_dp), the one-pass q-RoPE backward, the exact dk/dv fold;
  * o_proj through torchao Float8Linear (or the glue's Fp8GlueLinearFn, the O cast once in both layouts).
v_proj and o_proj are torchao-owned: fp8 forward, fp8 dgrad and fp8 wgrad.

Optional side channels: DM_MODE_LAYERS="23:stale,3:matched" runs the named layers in another mode (the mechanism
study's layer-sufficiency legs); DM_DIAG=<interval> records the stale-vs-matched residual statistics
(kernel/dm_port/diag.py).  Both default off and leave the numerics untouched.
"""
from __future__ import annotations

import os
from functools import partial

import torch
import torch.nn as nn

from .config import ModelConfig
from .layers import apply_output_gate, q_proj_out_features, RMSNorm, split_q_gate, yarn_mscale

DM_MODES = ("stale", "matched", "bf16_dp")


def require_delta_matching():
    """Import the kernel package, loudly."""
    try:
        import delta_matching
    except ImportError as e:
        raise ImportError("the dm mixer needs the Delta-Matching kernel package: install the delta-matching kernel package "
                          "(see docs/SETUP.md section 3: git clone --recursive, then pip install --no-build-isolation .)") from e
    return delta_matching


def _diag_stash(m):
    """DM_DIAG: snapshot this mixer's qk-norm gains and W_Q / W_K norms into the diag side channel on logged steps."""
    if m._diag_kw:
        from kernel.dm_port import diag as _diag
        _diag.stash_params(int(m.layer_idx),
                           m.q_norm.weight if getattr(m, "qk_norm", False) else None,   # no-qk-norm arms carry no gains
                           m.k_norm.weight if getattr(m, "qk_norm", False) else None,
                           getattr(m.q_proj, "weight", None), getattr(m.k_proj, "weight", None))


class DMAttention(nn.Module):
    _announced = False
    _noqk_announced = False

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_rep
        self.head_dim = cfg.head_dim
        self.dm_mode = cfg.dm_mode
        # Per-layer delta override (the mechanism study's layer-sufficiency legs). Default OFF.
        # DM_MODE_LAYERS="23:stale,3:matched" makes the named layers run the named dm_mode while every other layer
        # keeps cfg.dm_mode, so one layer can carry the stale delta inside an otherwise-matched model (or the reverse).
        # Applied here, before any route gating reads self.dm_mode, so the delta mode and the build check follow it.
        _ovr = os.environ.get("DM_MODE_LAYERS", "")
        if _ovr:
            _m = dict(kv.split(":") for kv in _ovr.split(",") if kv)
            if str(layer_idx) in _m:
                self.dm_mode = _m[str(layer_idx)]
                print(f"[dm] LAYER OVERRIDE: layer {layer_idx} dm_mode -> {self.dm_mode!r} "
                      f"(DM_MODE_LAYERS={_ovr!r}; other layers keep {cfg.dm_mode!r})", flush=True)
        if self.dm_mode not in DM_MODES:
            raise ValueError(f"dm_mode must be one of {DM_MODES}, got {self.dm_mode!r}")
        if not bool(getattr(cfg, "dm_fused_producers", True)):
            raise ValueError("the dm mixer runs the fused-producer route only (dm_fused_producers must stay true); the plain "
                             "kernel route is the gqa mixer with attn_impl='dm'")
        if self.head_dim not in (128, 256):
            raise ValueError(f"the dm mixer runs head_dim 128 or 256 (the Delta-Matching kernels), got {self.head_dim}")
        if self.dm_mode == "bf16_dp" and self.head_dim != 128:
            # refused here, not at the first backward: the kernel's bf16-dP backward (bwd_fp8_bf16dp) is head_dim 128 only
            raise ValueError(f"dm_mode bf16_dp runs at head_dim 128 only (the kernel's bf16-dP backward), got head_dim "
                             f"{self.head_dim}")
        if yarn_mscale(cfg.rope_scaling) != 1.0 and self.head_dim != 128:
            raise ValueError("the dm mixer folds the YaRN temperature at head_dim 128 only; disable rope_scaling at head_dim 256")
        if cfg.fused_rope:
            raise ValueError("the dm mixer owns the q RoPE (the fused Q producer); fused_rope must be off")
        # The q/k/v input cast, at the campaign's value by default (v_proj / o_proj belong to torchao: fp8 forward,
        # dgrad and wgrad under fp8_scope ffn_qkvo).
        self._shared_x8 = os.environ.get("DM_SHARED_X8", "1") == "1"
        require_delta_matching()
        from delta_matching import kernel_version
        from delta_matching.attn import _ops_gt
        from delta_matching.core import _ops, blocks
        ops = _ops()          # load now, so a broken build fails here and not mid-step
        _ops_gt()             # refuses a build without the global-memory twins loudly
        if self.head_dim == 128 and int(ops.build_hd128()) != 1:
            raise RuntimeError("head_dim 128 needs a Delta-Matching-128 build of the delta_matching package (build_hd128() == 1)")
        # The arm name decides the numerics and the BUILD must agree, in both directions.
        _bf16dp = int(getattr(ops, "build_bf16dp", lambda: 0)())
        if (self.dm_mode == "bf16_dp") != bool(_bf16dp):
            raise RuntimeError(f"dm_mode={self.dm_mode!r} contradicts the delta_matching build (build_bf16dp()={_bf16dp}): "
                               "bf16_dp needs a DM_BF16_DP=1 build and every other mode needs a DM_BF16_DP=0 build")
        from kernel.dm_port import diag as _diag
        from kernel.dm_port.fused_producers.tier1_vproj import v8_quantize
        from kernel.dm_port.orchestrator import ORCHESTRATOR_VERSION, fused_dm_attention
        self._v8q = partial(v8_quantize, blk=blocks(self.head_dim)[1])   # V scales per 64 keys at hd128, 32 at hd256
        self._attn = fused_dm_attention
        # side-channel diagnostics (kernel/dm_port/diag.py), only when DM_DIAG is set: no_grad, detached, own files
        self._diag_kw: dict = {"diag_layer": int(self.layer_idx)} if _diag.enabled() else {}
        if not DMAttention._announced:   # once per process (the dm layer may not be layer 0 in a hybrid)
            DMAttention._announced = True
            kv = kernel_version()
            print(f"[dm] DMAttention: {ORCHESTRATOR_VERSION}  kernel={kv}  head_dim={self.head_dim} "
                  f"blocks={list(blocks(self.head_dim))}  dm_mode={self.dm_mode}  shared_x8={self._shared_x8}  "
                  f"build_hd128={int(ops.build_hd128())} build_gt={int(ops.build_gt())} build_bf16dp={_bf16dp}", flush=True)
            try:  # stamp wandb so logs show which kernel trained which arm
                import wandb
                if wandb.run is not None:
                    wandb.config.update({"dm_kernel": kv, "dm_orchestrator": ORCHESTRATOR_VERSION,
                                         "dm_mode": self.dm_mode}, allow_val_change=True)
            except Exception:
                pass
        # YaRN attention temperature: SAME placement as GQAAttention (post-RoPE q, mscale^2).
        self.attn_mscale = yarn_mscale(cfg.rope_scaling)

        # SAME modules, SAME order as GQAAttention -> identical seeded init across arms.
        self.attn_output_gate = cfg.attn_output_gate
        self.q_proj = nn.Linear(cfg.d_model, q_proj_out_features(cfg), bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.d_model, bias=False)
        self.qk_norm = cfg.qk_norm
        self.fused_rope = cfg.fused_rope
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps, cfg.rms_norm_in_fp32, fused=cfg.fused_rmsnorm)
            self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps, cfg.rms_norm_in_fp32, fused=cfg.fused_rmsnorm)
        elif not DMAttention._noqk_announced:      # the no-qk-norm ablation: one banner per process
            DMAttention._noqk_announced = True
            print(f"[dm] qk_norm OFF: no q_norm / k_norm on the attention layers (config qk_norm=false; "
                  f"head_dim {self.head_dim})", flush=True)

    def _qkv_shared_fp8(self, x):
        """DM_SHARED_X8: cast x to e4m3 ONCE (torchao's dynamic tensorwise cast, the q_proj module's own config) and run
        the three projection GEMMs on that one Float8TrainingTensor; torchao then also reuses it for the three wgrads
        (no per-Linear backward re-cast).  Bypasses Float8Linear.forward's autocast `.to()` (which would DEQUANTIZE an
        already-cast input) by calling torchao's autograd Function directly, the same call Float8Linear.forward makes.
        Compiled once per mixer (lazily: the projections become Float8Linear only after apply_fp8)."""
        fn = getattr(self, "_qkv_shared_fn", None)
        if fn is None:
            from torchao.float8.float8_linear import Float8Linear, matmul_with_hp_or_float8_args
            from torchao.float8.float8_scaling_utils import hp_tensor_to_float8_dynamic
            from torchao.float8.float8_training_tensor import GemmInputRole
            mods = (self.q_proj, self.k_proj, self.v_proj)
            if not all(isinstance(m, Float8Linear) for m in mods):
                raise RuntimeError("DM_SHARED_X8=1 needs torchao q/k/v_proj (fp8 true, fp8_scope ffn_qkvo)")
            cfg = self.q_proj.config
            mmc = self.q_proj.linear_mm_config
            kcfg, kmmc = self.k_proj.config, self.k_proj.linear_mm_config
            vcfg, vmmc = self.v_proj.config, self.v_proj.linear_mm_config

            def _shared(x, wq_t, wk_t, wv_t):
                if torch.is_autocast_enabled():
                    x = x.to(torch.get_autocast_dtype("cuda"))
                x8 = hp_tensor_to_float8_dynamic(
                    x, cfg.cast_config_input.target_dtype, mmc,
                    gemm_input_role=GemmInputRole.INPUT,
                    scaling_granularity=cfg.cast_config_input.scaling_granularity,
                    axiswise_dim=None,
                    round_scales_to_power_of_2=cfg.round_scales_to_power_of_2)
                q = matmul_with_hp_or_float8_args.apply(x8, wq_t, mmc, cfg)
                k = matmul_with_hp_or_float8_args.apply(x8, wk_t, kmmc, kcfg)
                v = matmul_with_hp_or_float8_args.apply(x8, wv_t, vmmc, vcfg)
                return q, k, v

            fn = torch.compile(_shared)
            self._qkv_shared_fn = fn
        return fn(x, self.q_proj.weight.t(), self.k_proj.weight.t(), self.v_proj.weight.t())

    def forward(self, x, cos, sin, attention_mask=None):
        assert attention_mask is None, "the dm mixer uses packed windows (no padding)"
        H, Hkv, D = self.n_heads, self.n_kv_heads, self.head_dim
        from kernel.dm_port.fused_producers.m13_rope_bwd import k_rope_fused
        glue = getattr(self, "_fp8glue", None) is not None and not isinstance(x, torch.Tensor)
        if glue:
            # DM_FP8GLUE=1: x is the norm-emitted Fp8Pack; ONE set of codes (+ twin) feeds the three projection GEMMs
            # and their wgrads (kernel/fp8glue/ops.py::Fp8GlueQKVFn); q/k/v come back as [B, T, H, D] views so
            # dq/dk/dv reach the cast as strided views.
            from .mlp_fp8glue import mixer_qkv
            B, T = x.lead_shape
            q4, k4, v4 = mixer_qkv(self, x)
            gate = None
            q = q4.transpose(1, 2)                                # [B, H, T, D]
            k = k4.transpose(1, 2)                                # [B, Hkv, T, D]
        else:
            B, T, _ = x.shape
            if self._shared_x8:
                qg, kflat, v_flat = self._qkv_shared_fp8(x)
            else:
                qg, kflat, v_flat = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            q, gate = split_q_gate(qg, H, D, self.attn_output_gate)
            q = q.transpose(1, 2)                                 # [B, H, T, D]
            k = kflat.view(B, T, Hkv, D).transpose(1, 2)          # [B, Hkv, T, D]
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        k = k_rope_fused(k, cos, sin)                             # fused bitwise k RoPE (forward and backward)
        if glue:
            v_t = v4.transpose(1, 2)                              # graph edge for dv
        else:
            v_t = v_flat.view(B, T, Hkv, D).transpose(1, 2)       # graph edge for dv
        V8kv, V8skv = self._v8q(v_t)                              # pre-expansion V codes / scales (the reference rule)
        _diag_stash(self)
        if self.attn_mscale != 1.0:               # YaRN temperature: logits x mscale^2 as in the bf16 mixer, applied
            q = q * (self.attn_mscale * self.attn_mscale)   # to the pre-RoPE q (RoPE and the quantizer are linear in q)
        o = self._attn(q, k, v_t, V8kv, V8skv, cos, sin, n_rep=self.n_rep, **self._diag_kw,
                       mode=self.dm_mode)                         # bf16 [B, T, H, D]
        o = o.reshape(B, T, H * D)                                # free: the op's output layout
        if glue:
            from .mlp_fp8glue import mixer_oproj
            return mixer_oproj(self, o)                           # O cast once, both layouts saved
        return self.o_proj(apply_output_gate(o, gate))            # torchao Float8Linear
