"""Primitive layers: RMSNorm, rotary embeddings, SwiGLU MLP.

Math notes (kept terse; full derivation in docs/train_engine.md):
  RMSNorm(x) = x / sqrt(mean(x^2) + eps) * w        (no mean-subtraction; scale-only)
  RoPE: rotate (q,k) by position-dependent angle so that <R_m q, R_n k> depends only
        on (m-n) -> relative-position attention. rotate_half is the NeoX/Llama layout.
  SwiGLU(x) = down( silu(gate(x)) ⊙ up(x) ),  silu(z)=z*sigmoid(z).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- optional fused (Liger) kernels; lazily imported, cached, None if unavailable ---
def _liger():
    """Return the (RMSNormFn, RopeFn, SiLUMulFn) Liger ops or (None,None,None)."""
    cache = getattr(_liger, "_cache", "unset")
    if cache != "unset":
        return cache
    try:
        import torch.distributed.tensor  # noqa: F401 (liger RMSNorm references torch.distributed.tensor.DTensor)
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        from liger_kernel.ops.rope import LigerRopeFunction
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction
        cache = (LigerRMSNormFunction, LigerRopeFunction, LigerSiLUMulFunction)
    except Exception as e:  # keep training runnable without the optional dep
        import warnings
        warnings.warn(f"fused layer kernels requested but liger_kernel unavailable ({e}); using eager.")
        cache = (None, None, None)
    _liger._cache = cache
    return cache


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, in_fp32: bool = True, fused: bool = False):
        super().__init__()
        self.eps = eps
        self.in_fp32 = in_fp32
        self.weight = nn.Parameter(torch.ones(dim))
        # Liger's "llama" casting_mode == our in_fp32 path (reduce in fp32, cast back, then scale).
        # Only fuse when in_fp32 (the configured default) so the numerics match exactly.
        self.fused = bool(fused) and in_fp32 and _liger()[0] is not None
        # default None: set by modeling/mlp_fp8glue.install_fp8glue to (ctl, role)
        # when this norm feeds an fp8 consumer; the forward then returns an Fp8Pack (Liger-bitwise
        # bf16 x_norm + e4m3 codes/twin/scale). Inert unless DM_FP8GLUE=1.
        self._fp8glue = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_fp8glue", None) is not None:
            from .mlp_fp8glue import norm_pack_forward
            return norm_pack_forward(self, x)
        if self.fused:
            RMSNormFn = _liger()[0]
            return RMSNormFn.apply(x, self.weight, self.eps, 0.0, "llama")
        dtype = x.dtype
        if self.in_fp32:
            x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x.to(dtype)) if self.in_fp32 else (self.weight * x)


# --------------------------------------------------------------------------------------
# YaRN RoPE context extension (NTK-by-parts): DeepSeek-V3 long-context recipe.
#
# Goal: a model trained with theta RoPE at L_orig (e.g. 8192) is continued/zero-shot run at
# factor*L_orig (e.g. 64k) WITHOUT changing theta. YaRN interpolates inv_freq per dimension:
#   - high-frequency dims (short wavelength; many rotations over L_orig): keep the ORIGINAL
#     inv_freq (extrapolate) -> preserves local positional resolution.
#   - low-frequency dims  (long wavelength; few rotations over L_orig): scale inv_freq by
#     1/factor (interpolate) -> long-range positions fall back inside the trained angle range.
#   - a linear ramp over a [beta_fast, beta_slow]-rotation band blends the two.
# Plus an attention temperature mscale = 0.1*ln(factor)+1 to counter the softmax sharpening
# from the denser angles; applied by scaling q in the mixers (NOT folded into cos/sin, so the
# fp8 attention kernel's sm_scale stays untouched and k is unchanged).
# --------------------------------------------------------------------------------------

def _is_yarn(rope_scaling: dict | None) -> bool:
    return bool(rope_scaling) and str(rope_scaling.get("type", rope_scaling.get("rope_type", ""))).lower() == "yarn"


def yarn_mscale(rope_scaling: dict | None) -> float:
    """Attention temperature multiplier applied to (post-RoPE) q. 1.0 == no scaling.

    mscale = 0.1*ln(factor) + 1.0   (DeepSeek/YaRN get_mscale with the default 1.0 coefficient).
    An explicit rope_scaling['mscale'] overrides the formula if provided.
    """
    if not _is_yarn(rope_scaling):
        return 1.0
    if rope_scaling.get("mscale") is not None:
        return float(rope_scaling["mscale"])
    factor = float(rope_scaling["factor"])
    if factor <= 1.0:
        return 1.0
    return 0.1 * math.log(factor) + 1.0


# _yarn_find_dim / yarn_inv_freq are adapted from Hugging Face transformers' `_compute_yarn_parameters`
# (modeling_rope_utils.py, Copyright 2024 The HuggingFace Team, Apache-2.0).
def _yarn_find_dim(num_rotations: float, head_dim: int, theta: float, max_pos: int) -> float:
    # head_dim index d whose RoPE wavelength gives exactly `num_rotations` full turns over max_pos.
    return (head_dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(theta))


def yarn_inv_freq(head_dim: int, theta: float, rope_scaling: dict,
                  device, dtype=torch.float32) -> torch.Tensor:
    """NTK-by-parts interpolated inv_freq, shape [head_dim/2] (fp32)."""
    factor = float(rope_scaling["factor"])
    orig_max = int(rope_scaling.get("original_max_position_embeddings", 8192))
    beta_fast = float(rope_scaling.get("beta_fast", 32))
    beta_slow = float(rope_scaling.get("beta_slow", 1))

    pos_freqs = theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    inv_freq_extrap = 1.0 / pos_freqs                # original (high-freq / local): untouched
    inv_freq_interp = 1.0 / (factor * pos_freqs)     # scaled    (low-freq / long-range)

    # correction band over which to ramp from extrapolation -> interpolation
    low = math.floor(_yarn_find_dim(beta_fast, head_dim, theta, orig_max))
    high = math.ceil(_yarn_find_dim(beta_slow, head_dim, theta, orig_max))
    low = max(low, 0)
    high = min(high, head_dim // 2 - 1)
    if low == high:
        high += 0.001  # guard div-by-zero

    ramp = (torch.arange(head_dim // 2, device=device, dtype=torch.float32) - low) / (high - low)
    ramp = torch.clamp(ramp, 0.0, 1.0)
    extrap_factor = 1.0 - ramp                       # 1 -> keep original, 0 -> fully scaled
    inv = inv_freq_interp * (1.0 - extrap_factor) + inv_freq_extrap * extrap_factor
    return inv.to(dtype)


class RotaryEmbedding(nn.Module):
    """Compute cos/sin in forward from inv_freq (NeoX layout, fp32).

    Computing on-the-fly (rather than caching a [max_seq_len, head_dim] table at init)
    keeps it robust to HF's meta-device construction in `from_pretrained` (a cached
    buffer there would stay uninitialized -> NaN) and supports any eval sequence length.
    The backbone calls this once per forward and shares cos/sin across layers, so the
    cost is negligible.

    When `rope_scaling={"type":"yarn",...}` is set, inv_freq is NTK-by-parts interpolated
    (see yarn_inv_freq); the YaRN attention temperature (yarn_mscale) is applied to q in the
    mixers, not here, so cos/sin carry only the rotation and the fp8 attention kernel is untouched.
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0,
                 rope_scaling: dict | None = None, rotary_dim: int | None = None):
        super().__init__()
        # `rotary_dim` (partial RoPE, Qwen3.5) is the width cos/sin are built at; the inverse
        # frequencies are laid out over rotary_dim, NOT head_dim, so the rotated slice sees the
        # same frequency ladder a full head of that width would.
        head_dim = int(rotary_dim) if rotary_dim else head_dim
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"
        self.head_dim = head_dim
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.rope_scaling = rope_scaling
        self._use_yarn = _is_yarn(rope_scaling)
        # NOTE: no inv_freq buffer on purpose. HF `from_pretrained` materializes
        # non-persistent buffers to *uninitialized* memory (they're not in the
        # checkpoint), which silently corrupts a cached inv_freq -> NaN RoPE. inv_freq
        # is head_dim/2 elements, so recomputing it per forward is free.

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [B, T] -> cos/sin: [B, T, head_dim]
        if self._use_yarn:
            inv = yarn_inv_freq(self.head_dim, self.theta, self.rope_scaling, position_ids.device)
        else:
            inv = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, device=position_ids.device,
                                                     dtype=torch.float32) / self.head_dim))
        freqs = torch.einsum("bt,d->btd", position_ids.float(), inv)   # [B, T, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)                        # [B, T, head_dim]
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                 fused: bool = False):
    """q,k: [B, H, T, D]; cos,sin: [B, T, D]. Returns rotated (q,k) in q's dtype.

    `fused=True` routes through Liger's fused RoPE Triton kernel (liger_rotary_pos_emb takes
    cos/sin [B, T, D] and unsqueezes the head dim itself)."""
    if fused and _liger()[1] is not None:
        RopeFn = _liger()[1]
        return RopeFn.apply(q, k, cos, sin, None, 1)
    cos = cos.unsqueeze(1)  # [B,1,T,R] broadcast over heads (R == rotary_dim <= head_dim)
    sin = sin.unsqueeze(1)
    rd = cos.shape[-1]
    if rd == q.shape[-1]:
        qf, kf = q.float(), k.float()
        q_out = qf * cos + rotate_half(qf) * sin
        k_out = kf * cos + rotate_half(kf) * sin
        return q_out.to(q.dtype), k_out.to(k.dtype)
    # PARTIAL RoPE: rotate the leading `rd` channels, pass the tail through untouched.
    # rotate_half must be taken WITHIN the rotated slice: applying it across the full head
    # would pair a rotated channel with an unrotated one and silently scramble the tail.
    def _split_rot(x):
        xr, xp = x[..., :rd].float(), x[..., rd:]
        out = xr * cos + rotate_half(xr) * sin
        return torch.cat([out.to(x.dtype), xp], dim=-1)
    return _split_rot(q), _split_rot(k)


def q_proj_out_features(cfg) -> int:
    """Width of q_proj, doubled when `attn_output_gate` is on (Qwen3.5 / Qwen3-Next idiom).

    EVERY attention mixer must call this rather than using `cfg.q_dim` directly. The mixers
    deliberately duplicate their projections in a fixed registration ORDER so that one seed
    gives bit-identical init across arms: that identity is what makes the fp8-vs-clean
    comparison seed-matched, and it breaks the moment two mixers size q_proj differently.
    """
    return cfg.q_dim * (2 if cfg.attn_output_gate else 1)


def split_q_gate(qg: torch.Tensor, n_heads: int, head_dim: int, gated: bool):
    """[B, T, q_dim*(2 if gated)] -> (q, gate) each [B, T, n_heads, head_dim], gate None if off.

    The split is PER-HEAD (view as [.., n_heads, 2*head_dim] then chunk the last axis), matching
    Qwen3-Next's layout rather than halving the flat vector. From scratch the two layouts are a
    permutation of randomly-initialized output rows and train identically, but matching the
    reference keeps any future weight import honest.
    """
    if not gated:
        return qg.view(*qg.shape[:2], n_heads, head_dim), None
    q, g = qg.view(*qg.shape[:2], n_heads, 2 * head_dim).chunk(2, dim=-1)
    return q, g


def apply_output_gate(o: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    """o: [B, T, q_dim] attention output BEFORE o_proj; gate: [B, T, H, D] or None."""
    if gate is None:
        return o
    return o * torch.sigmoid(gate.reshape(o.shape).to(o.dtype))


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, intermediate_size: int, dropout: float = 0.0,
                 fused: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # fuse only the elementwise silu(gate)*up (the matmuls stay cuBLAS); avoids
        # materializing two [B,T,d_ff] activations.
        self.fused = bool(fused) and _liger()[2] is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused:
            SiLUMulFn = _liger()[2]
            return self.drop(self.down_proj(SiLUMulFn.apply(self.gate_proj(x), self.up_proj(x))))
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))
