"""DMLM for `transformers` (remote code): the hybrid decoder of the Delta-Matching study.

Self-contained copy of delta-matching-train's inference path (`src/pretrain/modeling` + `src/pretrain/hf`) for the mixers the
published checkpoints use: GQA and MLA attention, Mamba2, GatedDeltaNet, KDA. Training-only paths (FP8 Delta-Matching
attention, fused FP8 producers, TransformerEngine, the Liger fused kernels) are left out; what remains is the bf16 code
of the training repository's HF loader, with both of its attention paths (SDPA by default, flash-attn with
`attn_implementation="flash_attention_2"`), so on the same attention path the logits match that loader bit for bit
(scripts/hf/verify_standalone.py checks the SDPA path). The study's evaluation loaded models with
`attn_implementation="flash_attention_2"`; pass the same to reproduce its numerics (SDPA differs from it by attention
rounding only). Parameter names are the checkpoint's: `model.embed_tokens`,
`model.layers.N.{input_layernorm,mixer,post_attention_layernorm,mlp}`, `model.norm`, `lm_head` (tied to the embedding).

Requirements: torch and transformers; Mamba2 layers also need `mamba_ssm` + `causal_conv1d`, GatedDeltaNet and KDA
layers `flash-linear-attention` (`fla`). Each is imported only by the layers that use it, with an install hint if missing.

`generate()` keeps a cache (`DMLMCache`: K/V of the attention layers, mamba_ssm's conv / SSM states, fla's
recurrent and conv states), so each new token costs one step. It supports greedy decoding and sampling (num_beams=1);
a batch of prompts must share a length, since the recurrent layers have no padding mask. An all-ones attention mask
is the same as none. A plain forward (no `use_cache`) is the uncached evaluation path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_dmlm import DMLMConfig, ModelConfig


# ----------------------------------------------------------------------------------------------- primitive layers
class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * w, reduced in fp32 when `in_fp32` (then cast back before the scale)."""

    def __init__(self, dim: int, eps: float = 1e-5, in_fp32: bool = True):
        super().__init__()
        self.eps = eps
        self.in_fp32 = in_fp32
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        if self.in_fp32:
            x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x.to(dtype)) if self.in_fp32 else (self.weight * x)


def _is_yarn(rope_scaling: dict | None) -> bool:
    return bool(rope_scaling) and str(rope_scaling.get("type", rope_scaling.get("rope_type", ""))).lower() == "yarn"


def yarn_mscale(rope_scaling: dict | None) -> float:
    """YaRN attention temperature applied to post-RoPE q (1.0 == no scaling)."""
    if not _is_yarn(rope_scaling):
        return 1.0
    if rope_scaling.get("mscale") is not None:
        return float(rope_scaling["mscale"])
    factor = float(rope_scaling["factor"])
    if factor <= 1.0:
        return 1.0
    return 0.1 * math.log(factor) + 1.0


# The YaRN helpers below are adapted from transformers' `_compute_yarn_parameters` (src/transformers/modeling_rope_utils.py;
# Copyright 2024 The HuggingFace Team, Apache-2.0).
def _yarn_find_dim(num_rotations: float, head_dim: int, theta: float, max_pos: int) -> float:
    return (head_dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(theta))


def yarn_inv_freq(head_dim: int, theta: float, rope_scaling: dict, device, dtype=torch.float32) -> torch.Tensor:
    """NTK-by-parts interpolated inv_freq, shape [head_dim / 2] (fp32)."""
    factor = float(rope_scaling["factor"])
    orig_max = int(rope_scaling.get("original_max_position_embeddings", 8192))
    beta_fast = float(rope_scaling.get("beta_fast", 32))
    beta_slow = float(rope_scaling.get("beta_slow", 1))
    pos_freqs = theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    inv_freq_extrap = 1.0 / pos_freqs
    inv_freq_interp = 1.0 / (factor * pos_freqs)
    low = math.floor(_yarn_find_dim(beta_fast, head_dim, theta, orig_max))
    high = math.ceil(_yarn_find_dim(beta_slow, head_dim, theta, orig_max))
    low = max(low, 0)
    high = min(high, head_dim // 2 - 1)
    if low == high:
        high += 0.001
    ramp = (torch.arange(head_dim // 2, device=device, dtype=torch.float32) - low) / (high - low)
    ramp = torch.clamp(ramp, 0.0, 1.0)
    extrap_factor = 1.0 - ramp
    inv = inv_freq_interp * (1.0 - extrap_factor) + inv_freq_extrap * extrap_factor
    return inv.to(dtype)


class RotaryEmbedding(nn.Module):
    """cos / sin computed per forward from inv_freq (NeoX layout, fp32), over the rotated width `rotary_dim`.
    No inv_freq buffer on purpose: `from_pretrained` would materialize a non-persistent buffer uninitialized."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0,
                 rope_scaling: dict | None = None, rotary_dim: int | None = None):
        super().__init__()
        head_dim = int(rotary_dim) if rotary_dim else head_dim
        assert head_dim % 2 == 0, "RoPE needs an even head_dim"
        self.head_dim = head_dim
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.rope_scaling = rope_scaling
        self._use_yarn = _is_yarn(rope_scaling)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._use_yarn:
            inv = yarn_inv_freq(self.head_dim, self.theta, self.rope_scaling, position_ids.device)
        else:
            inv = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, device=position_ids.device,
                                                     dtype=torch.float32) / self.head_dim))
        freqs = torch.einsum("bt,d->btd", position_ids.float(), inv)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q, k: [B, H, T, D]; cos, sin: [B, T, R] with R <= D (partial RoPE rotates the leading R channels)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rd = cos.shape[-1]
    if rd == q.shape[-1]:
        qf, kf = q.float(), k.float()
        q_out = qf * cos + rotate_half(qf) * sin
        k_out = kf * cos + rotate_half(kf) * sin
        return q_out.to(q.dtype), k_out.to(k.dtype)

    def _split_rot(x):
        xr, xp = x[..., :rd].float(), x[..., rd:]
        out = xr * cos + rotate_half(xr) * sin
        return torch.cat([out.to(x.dtype), xp], dim=-1)
    return _split_rot(q), _split_rot(k)


def q_proj_out_features(cfg: ModelConfig) -> int:
    """q_proj width, doubled when the attention output gate is on (Qwen3.5 / Qwen3-Next idiom)."""
    return cfg.q_dim * (2 if cfg.attn_output_gate else 1)


def split_q_gate(qg: torch.Tensor, n_heads: int, head_dim: int, gated: bool):
    """[B, T, q_dim * (2 if gated)] -> (q, gate), each [B, T, n_heads, head_dim]; the split is per head."""
    if not gated:
        return qg.view(*qg.shape[:2], n_heads, head_dim), None
    q, g = qg.view(*qg.shape[:2], n_heads, 2 * head_dim).chunk(2, dim=-1)
    return q, g


def apply_output_gate(o: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    if gate is None:
        return o
    return o * torch.sigmoid(gate.reshape(o.shape).to(o.dtype))


class SwiGLUMLP(nn.Module):
    """down(silu(gate(x)) * up(x))."""

    def __init__(self, d_model: int, intermediate_size: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ------------------------------------------------------------------------------------------------ generation cache
class DMLMCache:
    """Decoding state. `kv`: K/V of the attention layers (post norm and RoPE). The recurrent layers' states live in
    their libraries' own containers, created on first use: mamba_ssm's InferenceParams (Mamba2 conv / SSM states,
    updated in place) and fla's FLACache (GatedDeltaNet / KDA recurrent and conv states, indexed by layer)."""
    is_compileable = False

    def __init__(self, batch_size: int, max_seqlen: int):
        self.kv = {}
        self.batch_size = batch_size
        self.max_seqlen = max_seqlen
        self.seen = 0
        self._mamba = None
        self._fla = None

    @property
    def mamba_params(self):
        if self._mamba is None:
            try:
                from mamba_ssm.utils.generation import InferenceParams
            except ImportError as e:
                raise ImportError("this checkpoint's Mamba2 layers need `pip install mamba-ssm causal-conv1d`") from e
            self._mamba = InferenceParams(max_seqlen=self.max_seqlen, max_batch_size=self.batch_size)
            self._mamba.seqlen_offset = self.seen
        return self._mamba

    @property
    def fla(self):
        if self._fla is None:
            try:
                from fla.models.utils import FLACache
            except ImportError as e:
                raise ImportError("this checkpoint's GatedDeltaNet / KDA layers need `pip install flash-linear-attention`") from e
            self._fla = FLACache()
        return self._fla

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seen

    def advance(self, n: int) -> None:
        self.seen += n
        if self._mamba is not None:
            self._mamba.seqlen_offset += n


# ---------------------------------------------------------------------------------------------------- attention
def _pick_flash():
    """flash-attn's flash_attn_func if installed (accepted only if it takes `dropout_p`), else None."""
    import inspect
    for mod in ("flash_attn_interface", "flash_attn"):
        try:
            fn = getattr(__import__(mod, fromlist=["flash_attn_func"]), "flash_attn_func")
            if "dropout_p" in inspect.signature(fn).parameters:
                return fn
        except Exception:
            continue
    return None


_FLASH_FN = _pick_flash()


class GQAAttention(nn.Module):
    """Grouped-query attention: qk-norm before (partial) RoPE, optional sigmoid output gate, SDPA or flash-attn."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_rep
        self.head_dim = cfg.head_dim
        self.attn_dropout = cfg.attn_dropout
        impl = "flash" if (cfg.attn_impl == "auto" and _FLASH_FN is not None) else cfg.attn_impl
        if impl not in ("flash", "sdpa", "auto"):
            raise ValueError(f"this checkpoint's remote code supports attn_impl sdpa | flash, got {impl!r}")
        if impl == "flash" and _FLASH_FN is None:
            raise RuntimeError("attn_impl='flash' but flash-attn is not importable; use attn_implementation='sdpa'")
        self.impl = "sdpa" if impl == "auto" else impl
        self.attn_mscale = yarn_mscale(cfg.rope_scaling)
        self.attn_output_gate = cfg.attn_output_gate
        self.q_proj = nn.Linear(cfg.d_model, q_proj_out_features(cfg), bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.d_model, bias=False)
        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps, cfg.rms_norm_in_fp32)
            self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps, cfg.rms_norm_in_fp32)

    def forward(self, x, cos, sin, attention_mask=None, cache=None):
        B, T, _ = x.shape
        q, gate = split_q_gate(self.q_proj(x), self.n_heads, self.head_dim, self.attn_output_gate)
        q = q.transpose(1, 2)                                                          # [B, H, T, D]
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, Hkv, T, D]
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rotary(q, k, cos, sin)
        if self.attn_mscale != 1.0:
            q = q * (self.attn_mscale * self.attn_mscale)
        drop = self.attn_dropout if self.training else 0.0
        past = None
        if cache is not None:
            past = cache.kv.get(self.layer_idx)
            if past is not None:
                k = torch.cat([past[0], k], dim=2)
                v = torch.cat([past[1], v], dim=2)
            cache.kv[self.layer_idx] = (k, v)
        if past is not None:   # decoding: the T new queries sit at the end of the S cached positions
            S = k.shape[2]
            gqa = self.n_rep > 1
            if attention_mask is None and T == 1:
                out = F.scaled_dot_product_attention(q, k, v, enable_gqa=gqa)
            else:
                pos_q = torch.arange(S - T, S, device=q.device)
                keep = (torch.arange(S, device=q.device)[None, :] <= pos_q[:, None])[None, None]
                if attention_mask is not None:
                    keep = keep & attention_mask.bool()[:, None, None, -S:]
                bias = torch.zeros(keep.shape, dtype=q.dtype, device=q.device)
                bias.masked_fill_(~keep, torch.finfo(q.dtype).min)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, enable_gqa=gqa)
            out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        elif self.impl == "flash" and attention_mask is None:
            out = _FLASH_FN(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=drop, causal=True)
            if isinstance(out, tuple):
                out = out[0]
            out = out.reshape(B, T, self.n_heads * self.head_dim)
        else:
            gqa = self.n_rep > 1
            if attention_mask is None:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop, enable_gqa=gqa)
            else:
                bias = self._build_additive_mask(attention_mask, T, q.dtype, q.device)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=drop, enable_gqa=gqa)
            out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(apply_output_gate(out, gate))

    @staticmethod
    def _build_additive_mask(attention_mask, T, dtype, device):
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        keep = attention_mask.bool()[:, None, None, :] & causal[None, None]
        bias = torch.zeros(keep.shape, dtype=dtype, device=device)
        bias.masked_fill_(~keep, torch.finfo(dtype).min)
        return bias


# ------------------------------------------------------------------------------------------------------- Mamba2
class Mamba2Mixer(nn.Module):
    """mamba_ssm.Mamba2 on its fused kernel path (conv1d + SSD scan); RoPE inputs are accepted and ignored."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        try:
            from mamba_ssm import Mamba2
            import causal_conv1d  # noqa: F401  (the fused kernel path needs it)
        except ImportError as e:
            raise ImportError("this checkpoint's Mamba2 layers need `pip install mamba-ssm causal-conv1d`") from e
        headdim = int(cfg.mamba2_headdim) if getattr(cfg, "mamba2_headdim", None) else cfg.head_dim
        self.mamba = Mamba2(
            d_model=cfg.d_model,
            headdim=headdim,
            d_state=int(getattr(cfg, "mamba2_d_state", 128)),
            expand=int(getattr(cfg, "mamba2_expand", 2)),
            ngroups=int(getattr(cfg, "mamba2_ngroups", 1)),
            chunk_size=int(getattr(cfg, "mamba2_chunk_size", 256)),
            layer_idx=layer_idx,
        )

    def forward(self, x, cos=None, sin=None, attention_mask=None, cache=None):
        return self.mamba(x, inference_params=cache.mamba_params if cache is not None else None)


# ------------------------------------------------------------------------------------------------ GatedDeltaNet
class GatedDeltaNetMixer(nn.Module):
    """fla's GatedDeltaNet (chunk mode, gate, short convolution); RoPE inputs are accepted and ignored."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        try:
            from fla.layers import GatedDeltaNet
        except ImportError as e:
            raise ImportError("this checkpoint's GatedDeltaNet layers need `pip install flash-linear-attention`") from e
        head_dim = cfg.gdn_head_dim or cfg.head_dim
        num_heads = cfg.gdn_num_heads or max(1, cfg.d_model // head_dim)
        extra = {}
        if getattr(cfg, "gdn_num_v_heads", None):   # omitted when unset: passing None is not equivalent in older fla
            extra["num_v_heads"] = cfg.gdn_num_v_heads
        self.gdn = GatedDeltaNet(hidden_size=cfg.d_model, head_dim=head_dim, num_heads=num_heads, **extra,
                                 mode="chunk", use_gate=True, use_short_conv=True, layer_idx=layer_idx)

    def forward(self, x, cos=None, sin=None, attention_mask=None, cache=None):
        out = self.gdn(x) if cache is None else self.gdn(x, past_key_values=cache.fla, use_cache=True)
        return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------------------------------------- KDA
class KDAMixer(nn.Module):
    """fla's Kimi Delta Attention (chunk mode, short convolution, per-channel decay); RoPE inputs are accepted and
    ignored. head_dim / num_heads fall back to the gdn_* fields, as in training."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        try:
            from fla.layers.kda import KimiDeltaAttention
        except ImportError as e:
            raise ImportError("this checkpoint's KDA layers need `pip install flash-linear-attention`") from e
        head_dim = cfg.kda_head_dim or cfg.gdn_head_dim or cfg.head_dim
        num_heads = cfg.kda_num_heads or cfg.gdn_num_heads or max(1, cfg.d_model // head_dim)
        extra = {}
        if getattr(cfg, "kda_num_v_heads", None):
            extra["num_v_heads"] = int(cfg.kda_num_v_heads)
        if getattr(cfg, "kda_lower_bound", None) is not None:
            extra["lower_bound"] = float(cfg.kda_lower_bound)
        self.kda = KimiDeltaAttention(hidden_size=cfg.d_model, head_dim=head_dim, num_heads=num_heads,
                                      expand_v=float(getattr(cfg, "kda_expand_v", 1.0)), mode="chunk",
                                      use_short_conv=True, conv_size=int(getattr(cfg, "kda_conv_size", 4)),
                                      allow_neg_eigval=bool(getattr(cfg, "kda_allow_neg_eigval", False)),
                                      safe_gate=bool(getattr(cfg, "kda_safe_gate", False)), layer_idx=layer_idx,
                                      **extra)

    def forward(self, x, cos=None, sin=None, attention_mask=None, cache=None):
        out = self.kda(x) if cache is None else self.kda(x, past_key_values=cache.fla, use_cache=True)
        return out[0] if isinstance(out, tuple) else out


# ---------------------------------------------------------------------------------------------------------- MLA
def _rope_leading(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rd: int) -> torch.Tensor:
    """Rotate the leading `rd` channels of x [B, h, T, D] (h = 1 or H) with cos / sin [B, T, rd]; the tail passes."""
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    xr = x[..., :rd].float()
    out = (xr * c + rotate_half(xr) * s).to(x.dtype)
    return out if rd == x.shape[-1] else torch.cat([out, x[..., rd:]], dim=-1)


class MLAAttention(nn.Module):
    """Multi-head latent attention at one head dim for q, k and v (the bf16 core the study's evaluation ran).

    q = q_proj(x) per head [rope | nope]; [c ; k_r] = kv_a_proj(x); c = kv_a_norm(c); k_nope = k_proj(c);
    v = v_proj(c); four sub-vector norms (qk_norm); k_r rotated once and shared by every head; MHA on the wire."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.rotary_dim = cfg.rotary_dim
        self.nope_dim = cfg.head_dim - cfg.rotary_dim
        self.kv_lora_rank = int(cfg.kv_lora_rank)
        self.attn_output_gate = cfg.attn_output_gate
        self.qk_norm = cfg.qk_norm
        assert cfg.n_kv_heads == cfg.n_heads, "MLA is MHA on the wire: n_kv_heads must equal n_heads"
        assert 0 < self.rotary_dim < self.head_dim, "MLA needs 0 < rotary_dim < head_dim"
        self.attn_mscale = yarn_mscale(cfg.rope_scaling)
        d, H, D, rd, dn, dc = cfg.d_model, self.n_heads, self.head_dim, self.rotary_dim, self.nope_dim, self.kv_lora_rank
        self.q_proj = nn.Linear(d, q_proj_out_features(cfg), bias=False)
        self.kv_a_proj = nn.Linear(d, dc + rd, bias=False)
        self.kv_a_norm = RMSNorm(dc, cfg.norm_eps, cfg.rms_norm_in_fp32)
        self.k_proj = nn.Linear(dc, H * dn, bias=False)
        self.v_proj = nn.Linear(dc, H * D, bias=False)
        self.o_proj = nn.Linear(H * D, d, bias=False)
        if self.qk_norm:
            self.q_rope_norm = RMSNorm(rd, cfg.norm_eps, cfg.rms_norm_in_fp32)
            self.q_nope_norm = RMSNorm(dn, cfg.norm_eps, cfg.rms_norm_in_fp32)
            self.k_rope_norm = RMSNorm(rd, cfg.norm_eps, cfg.rms_norm_in_fp32)
            self.k_nope_norm = RMSNorm(dn, cfg.norm_eps, cfg.rms_norm_in_fp32)
        impl = "flash" if (cfg.attn_impl == "auto" and _FLASH_FN is not None) else cfg.attn_impl
        if impl == "flash" and _FLASH_FN is None:
            raise RuntimeError("attn_impl='flash' but flash-attn is not importable; use attn_implementation='sdpa'")
        self.impl = impl if impl == "flash" else "sdpa"

    def _qkv(self, x, cos, sin):
        B, T, _ = x.shape
        H, D, rd, dn, dc = self.n_heads, self.head_dim, self.rotary_dim, self.nope_dim, self.kv_lora_rank
        q, gate = split_q_gate(self.q_proj(x), H, D, self.attn_output_gate)
        q = q.transpose(1, 2)                                             # [B, H, T, D]
        ckr = self.kv_a_proj(x)
        c, k_r = ckr[..., :dc], ckr[..., dc:]                              # [B, T, dc], [B, T, rd]
        c = self.kv_a_norm(c)
        k_n = self.k_proj(c).view(B, T, H, dn).transpose(1, 2)            # [B, H, T, dn]
        v = self.v_proj(c).view(B, T, H, D).transpose(1, 2)               # [B, H, T, D]
        k_r = k_r.unsqueeze(1)                                            # [B, 1, T, rd]
        if self.qk_norm:
            q = torch.cat([self.q_rope_norm(q[..., :rd]), self.q_nope_norm(q[..., rd:])], dim=-1)
            k_r = self.k_rope_norm(k_r)
            k_n = self.k_nope_norm(k_n)
        q = _rope_leading(q, cos, sin, rd)
        if self.attn_mscale != 1.0:
            q = q * (self.attn_mscale * self.attn_mscale)
        k_r = _rope_leading(k_r, cos, sin, rd)
        k = torch.cat([k_r.expand(B, H, T, rd), k_n], dim=-1)
        return q, k, v, gate

    def forward(self, x, cos, sin, attention_mask=None, cache=None):
        B, T, _ = x.shape
        q, k, v, gate = self._qkv(x, cos, sin)
        past = None
        if cache is not None:
            past = cache.kv.get(self.layer_idx)
            if past is not None:
                k = torch.cat([past[0], k], dim=2)
                v = torch.cat([past[1], v], dim=2)
            cache.kv[self.layer_idx] = (k, v)
        S = k.shape[2]
        if attention_mask is None and past is None and self.impl == "flash":
            out = _FLASH_FN(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True)
            if isinstance(out, tuple):
                out = out[0]
            out = out.reshape(B, T, self.n_heads * self.head_dim)
        else:
            if attention_mask is None and past is None:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            elif attention_mask is None and T == 1:
                out = F.scaled_dot_product_attention(q, k, v)
            else:
                pos_q = torch.arange(S - T, S, device=q.device)
                keep = (torch.arange(S, device=q.device)[None, :] <= pos_q[:, None])[None, None]
                if attention_mask is not None:
                    keep = keep & attention_mask.bool()[:, None, None, -S:]
                bias = torch.zeros(keep.shape, dtype=q.dtype, device=q.device)
                bias.masked_fill_(~keep, torch.finfo(q.dtype).min)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(apply_output_gate(out, gate))


MIXER_REGISTRY = {"gqa": GQAAttention, "mla": MLAAttention, "mamba2": Mamba2Mixer, "gated_deltanet": GatedDeltaNetMixer,
                  "kda": KDAMixer}


def build_mixer(name: str, cfg: ModelConfig, layer_idx: int) -> nn.Module:
    if name not in MIXER_REGISTRY:
        raise ValueError(f"mixer {name!r} is not shipped with this checkpoint's code (have {sorted(MIXER_REGISTRY)})")
    return MIXER_REGISTRY[name](cfg, layer_idx)


# ------------------------------------------------------------------------------------------------------ backbone
class DecoderBlock(nn.Module):
    """Pre-norm block: x += mixer(norm(x)); x += mlp(norm(x))."""

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps, cfg.rms_norm_in_fp32)
        self.mixer = build_mixer(cfg.mixer_for_layer(layer_idx), cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps, cfg.rms_norm_in_fp32)
        self.mlp = SwiGLUMLP(cfg.d_model, cfg.intermediate_size, cfg.resid_dropout)
        self.resid_drop = nn.Dropout(cfg.resid_dropout) if cfg.resid_dropout > 0 else nn.Identity()

    def forward(self, x, cos, sin, attention_mask=None, cache=None):
        x = x + self.resid_drop(self.mixer(self.input_layernorm(x), cos, sin, attention_mask, cache))
        x = x + self.resid_drop(self.mlp(self.post_attention_layernorm(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([DecoderBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps, cfg.rms_norm_in_fp32)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta,
                                      rope_scaling=cfg.rope_scaling, rotary_dim=cfg.rotary_dim)

    def forward(self, input_ids, position_ids, attention_mask=None, cache=None):
        if attention_mask is not None and bool(attention_mask.all()):
            attention_mask = None   # no padding: take the maskless kernels, the same path as a plain forward
        h = self.embed_tokens(input_ids)
        cos, sin = self.rotary(position_ids)
        if getattr(self.cfg, "nope", False):
            cos, sin = torch.ones_like(cos), torch.zeros_like(sin)
        cos, sin = cos.to(h.dtype), sin.to(h.dtype)
        for layer in self.layers:
            h = layer(h, cos, sin, attention_mask, cache)
        if cache is not None:
            cache.advance(input_ids.shape[1])
        return self.norm(h)


# ------------------------------------------------------------------------------------------------ HF causal LM
_ATTN_MAP = {"flash_attention_2": "flash", "sdpa": "sdpa", "eager": "sdpa"}


@dataclass
class DMLMCausalLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    cache_params: DMLMCache | None = None


class DMLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = DMLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False   # no layer implements it: gradient_checkpointing_enable() refuses clearly
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_attention_backend = True
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _is_stateful = True   # the Mamba2 state cannot be rewound, so no assisted generation

    @classmethod
    def _supports_default_dynamic_cache(cls) -> bool:
        return False      # the model keeps its own state in `cache_params` (DMLMCache)

    def __init__(self, config: DMLMConfig):
        super().__init__(config)
        mc = config.to_model_config()
        hf_impl = getattr(config, "_attn_implementation", None)
        if hf_impl in _ATTN_MAP:
            mc.attn_impl = _ATTN_MAP[hf_impl]
        self._mc = mc
        self.model = Transformer(mc)
        self.lm_head = nn.Linear(mc.d_model, mc.vocab_size, bias=False)
        if mc.tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None,
                cache_params=None, use_cache=None, past_key_values=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, **kwargs):
        B, T = input_ids.shape
        if use_cache and cache_params is None:
            cache_params = DMLMCache(B, self._mc.max_seq_len)
        past = cache_params.seen if cache_params is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past, past + T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        hidden = self.model(input_ids, position_ids, attention_mask, cache=cache_params)
        # `logits_to_keep` (the transformers convention): an int k > 0 computes the logits of the last k positions only;
        # 0 or None computes all of them; a 1-D tensor keeps those sequence indices. Read from **kwargs, so generate()
        # (which builds its inputs in prepare_inputs_for_generation) calls this model exactly as before.
        keep = kwargs.get("logits_to_keep")
        if isinstance(keep, torch.Tensor):
            sel = keep
        elif keep:
            if int(keep) < 0:
                raise ValueError(f"logits_to_keep must be >= 0, got {keep}")
            sel = slice(-int(keep), None)
        else:
            sel = None
        if sel is not None and labels is not None:
            raise ValueError("labels need the logits of every position: omit logits_to_keep (or pass 0) with labels")
        logits = self.lm_head(hidden if sel is None else hidden[:, sel, :])
        loss = None
        if labels is not None:
            sl = logits[:, :-1, :].contiguous().float()
            lb = labels[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lb.view(-1), ignore_index=-100)
        return DMLMCausalLMOutput(loss=loss, logits=logits, cache_params=cache_params)

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, cache_params=None, use_cache=None,
                                      **kwargs):
        # with a cache only the newest token is fed; use_cache=False recomputes the whole prefix every step
        if cache_params is not None and cache_params.seen > 0:
            input_ids = input_ids[:, -1:]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "cache_params": cache_params,
                "use_cache": use_cache}
