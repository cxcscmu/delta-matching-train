"""attn_impl='te_fp8': the GQA mixer's attention CORE taken directly from NVIDIA
Transformer Engine's fp8 fused attention (cuDNN FusedAttention sub-backend 2, DelayedScaling), with
everything around it ours (torchao Float8Linear q/k/v/o projections, our qk-norm, our RoPE, our
output gate, the fp8-glue MLP, FSDP2, the DCP checkpointer, the HF export).

WHAT RUNS.  `transformer_engine.pytorch.DotProductAttention(num_attention_heads=n_heads,
kv_channels=head_dim, num_gqa_groups=n_kv_heads, attention_dropout=0, qkv_format="bshd",
attn_mask_type="causal")` called inside `te.autocast(enabled=True, recipe=DelayedScaling(fp8_dpa=True,
fp8_mha=False, fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max"))`
(`te.fp8_autocast(enabled=True, fp8_recipe=...)` is the deprecated alias of exactly that call in
TE 2.20).  Inside the core TE quantizes Q/K/V (ONE shared tensorwise e4m3 scale, META_QKV), O and S
(e4m3) in the forward and dO/dP (e5m2) plus dQ/dK/dV in the backward with its delayed-scaling
amax histories; the boundaries stay bf16 (fp8_mha=False), so our projections keep their own
torchao dynamic tensorwise scaling.

GEOMETRY.  TE/cuDNN fp8 attention TRAINING on sm90 exists only at head_dim 128 (TE's
fused_attn.cpp limits sm90 fp8 training to head_dim_qk == head_dim_v == 128; hd256 = NoBackend),
so the route refuses any other head_dim.  The M-tier te_fp8 arm runs 16q/4kv hd128 (same q_dim
2048 / kv_dim 512 as the campaign's 8q/2kv hd256).

LAYOUT (no extra copies).  q/k/v are kept in the projection's natural [B, T, H, D] ("bshd")
layout end to end: per-head qk-norm over the last dim, RoPE in fp32 with the same NeoX
rotate_half math as layers.apply_rotary (partial RoPE honoured), YaRN mscale, then the core.
TE's layout detector sees three contiguous bshd tensors ("bshd_bshd_bshd") and copies nothing.
The core returns [B, T, H*D], i.e. already the o_proj input.  fused_rope is not used on this route
(the config force-disables it under partial RoPE anyway; the eager math is identical).

BACKWARD SCALES.  TE's DPA never issues the backward delayed-scaling update itself (only TE's
GEMM modules do); `run_core`/`_arm_bwd_update` below mirror that contract so a DPA-only model
gets real dO/dP/dQKV scales instead of unit ones.

AUTOCAST REGION = ONE PER MODEL FORWARD, never per layer.  TE's `autocast_exit` runs the
delayed-scaling amax reduction + scale update for EVERY module registered under the recipe at the
exit of the outermost region, and `is_first_fp8_module` (which decides who triggers the backward
update) is per region.  Per-layer regions would roll each core's 16-slot amax history once per
attention layer (six times per microbatch in the 24-layer 3:1 hybrid) and fire one backward update
per attention layer.  The mixer therefore publishes `mixer_forward_ctx` and the trainer
(trainer/base_trainer.py::_micro_step) wraps the whole model forward of every microbatch in that
single region; the core REFUSES to run outside it.

BACKEND PROOF, FAIL LOUDLY.  NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1 NVTE_UNFUSED_ATTN=0 are set
before TE is imported (defaults set by this route in _te(); an explicit conflicting value is an
error); after the first training forward the route reads TE's own selection cache
(`dot_product_attention._attention_backends`) and raises
unless the selection is FusedAttention with `FusedAttnBackend.FP8` (sub-backend 2) AND the core
is in fp8 mode with an fp8_dpa recipe; every later training forward re-checks the cache (a
re-selection to anything else raises too).  The proof line `[te_fp8] backend proof: ...` goes to
the log next to TE's own `Running with FusedAttention backend (sub-backend 2)` (NVTE_DEBUG=1
NVTE_DEBUG_LEVEL=1).  Nondeterminism: TE's fp8 backward is nondeterministic on sm90;
NVTE_ALLOW_NONDETERMINISTIC_ALGO defaults to 1.

CHECKPOINTS.  The DPA has no parameters or buffers; its fp8 metadata (amax histories, scales)
would otherwise ride in `state_dict()` as `..._extra_state` (a variable-length pickled byte
tensor), which the DCP loader cannot size-match before the first forward and the HF export would
cast to bf16.  `_TeFp8Core` restores nn.Module's no-op extra-state methods, so the state dict
carries NO TE entries: DCP save/load and the safetensors export are byte-for-byte the flash
arm's key set, and on resume the fp8 metadata is RE-INITIALISED deterministically at the first
forward (fresh recipe state on every rank: unit scales for the first microbatch, the 16-slot
history refills over the next 16 microbatches), the same transient every TE run has at step 0.

DEFAULT-INERT.  Nothing here imports transformer_engine until a mixer is built with
attn_impl='te_fp8'; in the campaign env (no TE) that build fails with a clear message and every
other route is untouched.  Eval of a te_fp8 export uses attn_impl='sdpa' (the launcher's export
tail patches config.json); the core has no weights, so the export loads anywhere.  TE's
DelayedScaling recipe is the paper's cuDNN / Transformer Engine FP8 baseline, distinct from the
dm mixer's bf16_dp mode (the single dP matmul kept in bf16).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

from .layers import apply_output_gate, rotate_half, split_q_gate

TE_FP8_IMPL = "te_fp8"
TE_ENV_HINT = "a Python environment with Transformer Engine installed (DM_TE_PY / DM_TE_TORCHRUN in scripts/campaign/train_arm.sh)"

# Backend switches: TE reads NVTE_FLASH_ATTN at import (a module constant) and all three at every
# backend selection.  Set BEFORE the import; a conflicting explicit value is an error, never a
# silent fallback.  NVTE_DEBUG=1/LEVEL=1 makes TE print `Running with ... backend` (INFO) once per
# selection on every rank; LEVEL=2 adds the per-selection reasons (and, for layer_number 1 only,
# per-call quantizer dumps: layer_number = layer_idx + 1, so only a model whose layer 0 is attention,
# e.g. pure GQA, triggers them; the 3:1 hybrid's first attention layer is layer_number 4).
_NVTE_REQUIRED = {"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "1", "NVTE_UNFUSED_ATTN": "0"}
_NVTE_DEFAULTS = {**_NVTE_REQUIRED, "NVTE_DEBUG": "1", "NVTE_DEBUG_LEVEL": "1"}

_S = {"te": None, "recipe": None, "core_cls": None, "proof_printed": False, "n_calls": 0}


def te_available() -> bool:
    """Probe without importing (importing TE before the NVTE_* pins is what _te() warns about)."""
    import importlib.util
    return importlib.util.find_spec("transformer_engine") is not None


def _te():
    """Lazy TE import with the backend env pinned first; a clear error where TE is absent."""
    if _S["te"] is not None:
        return _S["te"]
    preloaded = "transformer_engine" in sys.modules
    for k, v in _NVTE_DEFAULTS.items():
        os.environ.setdefault(k, v)
    bad = {k: os.environ.get(k) for k, v in _NVTE_REQUIRED.items() if os.environ.get(k) != v}
    if bad:
        raise RuntimeError(
            f"attn_impl='te_fp8' pins the cuDNN FusedAttention fp8 backend via {_NVTE_REQUIRED}; "
            f"the environment contradicts it: {bad}. Unset them (the route sets the defaults) or "
            "export exactly these values; the arm name decides the kernel.")
    try:
        import transformer_engine
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import DelayedScaling, Format
    except Exception as e:  # ImportError, or a broken build
        raise RuntimeError(
            "attn_impl='te_fp8' needs NVIDIA Transformer Engine (fp8 fused attention), which this "
            f"environment cannot import ({type(e).__name__}: {e}). Run the arm in the TE env "
            f"{TE_ENV_HINT}; a te_fp8 export is evaluated with attn_impl='sdpa' (its config.json "
            "is patched by the launcher's export tail).") from e
    if preloaded:
        print("[te_fp8] NOTE: transformer_engine was imported before this route pinned the NVTE_* "
              "env; the backend selection still reads the env at call time", flush=True)
    _S["te"] = (transformer_engine, te, DelayedScaling, Format)
    return _S["te"]


def te_recipe():
    """ONE DelayedScaling recipe object for every core and for the trainer's autocast (TE keys
    its global amax buffers by the recipe, so all cores must share the instance)."""
    if _S["recipe"] is None:
        _, _, DelayedScaling, Format = _te()
        _S["recipe"] = DelayedScaling(fp8_dpa=True, fp8_mha=False, fp8_format=Format.HYBRID,
                                      amax_history_len=16, amax_compute_algo="max")
    return _S["recipe"]


def te_forward_ctx():
    """The single TE autocast region that must wrap the WHOLE model forward of one microbatch
    (published by the mixer as `mixer_forward_ctx`; the trainer enters it in _micro_step).
    == te.fp8_autocast(enabled=True, fp8_recipe=te_recipe()) (deprecated alias in TE 2.20)."""
    _, te, _, _ = _te()
    return te.autocast(enabled=True, recipe=te_recipe())


def _state_manager():
    try:
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager
    except ImportError:  # older layout
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    return FP8GlobalStateManager


def _core_class():
    if _S["core_cls"] is None:
        _, te, _, _ = _te()

        class _TeFp8Core(te.DotProductAttention):
            """TE's DotProductAttention minus its extra-state serialization (see module doc:
            no fp8 metadata in the state dict; deterministic re-init on resume)."""
            get_extra_state = nn.Module.get_extra_state
            set_extra_state = nn.Module.set_extra_state

        _TeFp8Core.__name__ = "TeFp8Core"
        _S["core_cls"] = _TeFp8Core
    return _S["core_cls"]


def build_te_core(cfg, layer_idx: int) -> nn.Module:
    """The core module for one GQA mixer (no parameters, no buffers)."""
    if cfg.head_dim != 128:
        raise ValueError(
            f"attn_impl='te_fp8' is head_dim-128-only (got {cfg.head_dim}): TE/cuDNN fp8 fused "
            "attention TRAINING on sm90 exists only at head_dim 128. Run "
            "the same q/kv widths at hd128, e.g. 16q/4kv for the M-tier's 8q/2kv hd256.")
    if cfg.attn_dropout != 0.0:
        raise ValueError("attn_impl='te_fp8' runs attention_dropout=0 only")
    cls = _core_class()
    core = cls(num_attention_heads=cfg.n_heads, kv_channels=cfg.head_dim,
               num_gqa_groups=cfg.n_kv_heads, attention_dropout=0.0, qkv_format="bshd",
               attn_mask_type="causal", layer_number=layer_idx + 1,
               softmax_scale=cfg.head_dim ** -0.5)
    if len(list(core.parameters())) or len(list(core.buffers())) or core.state_dict():
        raise RuntimeError("te_fp8 core unexpectedly carries state: "
                           f"{list(core.state_dict().keys())}")
    return core


def _rope_bshd(q, k, cos, sin):
    """RoPE on [B, T, H, D] tensors; cos/sin [B, T, R] (R == rotary_dim <= D). Same fp32 NeoX
    math as layers.apply_rotary (only the broadcast axis differs), partial RoPE honoured."""
    cos = cos.unsqueeze(2)  # [B, T, 1, R]
    sin = sin.unsqueeze(2)
    rd = cos.shape[-1]
    if rd == q.shape[-1]:
        qf, kf = q.float(), k.float()
        return ((qf * cos + rotate_half(qf) * sin).to(q.dtype),
                (kf * cos + rotate_half(kf) * sin).to(k.dtype))

    def _rot(x):
        xr, xp = x[..., :rd].float(), x[..., rd:]
        out = xr * cos + rotate_half(xr) * sin
        return torch.cat([out.to(x.dtype), xp], dim=-1)
    return _rot(q), _rot(k)


def _arm_bwd_update(q, k, v, M):
    """BACKWARD delayed scaling for a DPA-only model.  In TE 2.20 the backward amax reduction +
    scale update (`FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)`) is issued
    ONLY from the backward of TE's GEMM modules (Linear / LayerNormLinear / LayerNormMLP /
    GroupedLinear / ops fuser) when they were the region's first fp8 module; DotProductAttention
    never issues it.  With DPAs as the only TE modules (this route) the dO/dP/dQKV scales would
    stay at 1.0 for the whole run (measured: bwd amax_history[0] filled, never rolled, scale_bwd
    all 1.0 after 8 fwd+bwd), a silent departure from a TE reference model where TE Linears do
    the update.  Mirror Linear's contract: the region's FIRST core
    (the first attention layer, whose backward is the LAST attention backward of the microbatch)
    fires the update once its own input gradients exist, i.e. after every core's backward has
    written its amaxes.  A tensor hook on one of q/k/v suffices: all three grads come out of the
    one fused backward node."""
    fired = {"done": False}

    def hook(_g):
        if not fired["done"]:
            fired["done"] = True
            M.reduce_and_update_fp8_tensors(forward=False)
        return None
    for t in (q, k, v):
        if t.requires_grad:
            t.register_hook(hook)
            return True
    return False


def run_core(core, q, k, v):
    """Call one te_fp8 core inside the (already entered) region with TE's first-module contract
    (see _arm_bwd_update).  q [B,T,H,D], k/v [B,T,Hkv,D] bshd; returns [B, T, H*D]."""
    M = _state_manager()
    first = M.is_first_fp8_module() if torch.is_grad_enabled() else False
    out = core(q, k, v, qkv_format="bshd", attn_mask_type="causal")
    if first:
        _arm_bwd_update(q, k, v, M)
    return out


def _require_region(mixer):
    M = _state_manager()
    if not M.is_fp8_enabled():
        raise RuntimeError(
            "te_fp8 core called OUTSIDE the TE autocast region: the whole model forward must run "
            "under the mixer's `mixer_forward_ctx` (the trainer installs it; a standalone caller "
            "enters `te_attn.te_forward_ctx()` itself). Refusing to run the core in bf16.")
    rec = M.get_fp8_recipe()
    if rec is not te_recipe():
        raise RuntimeError(f"te_fp8: a foreign TE recipe is active ({rec!r}); the route's own "
                           "DelayedScaling(fp8_dpa=True, fp8_mha=False) recipe must wrap the forward")


def _check_backend(mixer):
    """Read TE's selection cache; raise unless the cuDNN fp8 fused backend is what runs."""
    from transformer_engine.pytorch.attention.dot_product_attention import dot_product_attention as dpa_mod
    from transformer_engine.pytorch.attention.dot_product_attention.utils import FusedAttnBackend
    ab = dpa_mod._attention_backends
    fb = ab.get("fused_attention_backend")
    fb_name = getattr(fb, "name", str(fb))
    core = mixer._te_core
    fp8_on = bool(getattr(core, "fp8", False))
    rec = core.fp8_meta.get("recipe") if hasattr(core, "fp8_meta") else None
    ok = (bool(ab.get("use_fused_attention")) and fb == FusedAttnBackend["FP8"]
          and not ab.get("use_flash_attention") and not ab.get("use_unfused_attention")
          and fp8_on and bool(getattr(rec, "fp8_dpa", False)))
    if not _S["proof_printed"] or not ok:
        transformer_engine, te, _, _ = _te()
        line = (f"[te_fp8] backend proof: use_fused_attention={ab.get('use_fused_attention')} "
                f"fused_attention_backend={fb_name}({int(fb) if fb is not None else 'none'}) "
                f"use_flash_attention={ab.get('use_flash_attention')} "
                f"use_unfused_attention={ab.get('use_unfused_attention')} core.fp8={fp8_on} "
                f"recipe={type(rec).__name__ if rec is not None else None} fp8_dpa={getattr(rec, 'fp8_dpa', None)} "
                f"fp8_mha={getattr(rec, 'fp8_mha', None)} fp8_format={getattr(getattr(rec, 'fp8_format', None), 'name', None)} "
                f"amax_history_len={getattr(rec, 'amax_history_len', None)} qkv_format=bshd "
                f"heads={mixer.n_heads}q/{mixer.n_kv_heads}kv hd={mixer.head_dim} "
                f"TE={transformer_engine.__version__} cuDNN={te.get_cudnn_version()} sm={te.get_device_compute_capability()} "
                f"env NVTE_FLASH_ATTN={os.environ.get('NVTE_FLASH_ATTN')} NVTE_FUSED_ATTN={os.environ.get('NVTE_FUSED_ATTN')} "
                f"NVTE_UNFUSED_ATTN={os.environ.get('NVTE_UNFUSED_ATTN')} "
                f"NVTE_ALLOW_NONDETERMINISTIC_ALGO={os.environ.get('NVTE_ALLOW_NONDETERMINISTIC_ALGO', '1 (default)')}")
        print(line, flush=True)
        if not ok:
            raise RuntimeError("te_fp8: the cuDNN fp8 FusedAttention backend is NOT what ran "
                               "(never silently fall back to bf16/unfused attention): " + line)
        _S["proof_printed"] = True
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update({"te_fp8_backend": f"FusedAttention {fb_name} (sub-backend {int(fb)})",
                                     "te_fp8_recipe": repr(rec), "te_version": transformer_engine.__version__,
                                     "te_cudnn": str(te.get_cudnn_version())}, allow_val_change=True)
        except Exception:
            pass


def te_fp8_forward(mixer, x, cos, sin, attention_mask=None):
    """GQAAttention.forward for attn_impl='te_fp8' (see the module doc)."""
    if attention_mask is not None:
        raise RuntimeError("attn_impl='te_fp8' uses packed causal windows (no padding mask)")
    B, T, _ = x.shape
    H, Hkv, D = mixer.n_heads, mixer.n_kv_heads, mixer.head_dim
    q, gate = split_q_gate(mixer.q_proj(x), H, D, mixer.attn_output_gate)   # [B, T, H, D]
    k = mixer.k_proj(x).view(B, T, Hkv, D)
    v = mixer.v_proj(x).view(B, T, Hkv, D)
    if mixer.qk_norm:
        q, k = mixer.q_norm(q), mixer.k_norm(k)      # over head_dim, then rotate (as the flash route)
    q, k = _rope_bshd(q, k, cos, sin)
    if mixer.attn_mscale != 1.0:
        q = q * (mixer.attn_mscale * mixer.attn_mscale)
    _require_region(mixer)
    out = run_core(mixer._te_core, q, k, v)              # [B, T, H*D] bf16 (fp8 inside)
    _S["n_calls"] += 1
    if mixer.training:
        _check_backend(mixer)
    out = out.reshape(B, T, H * D)
    return mixer.o_proj(apply_output_gate(out, gate))
