"""Multi-head Latent Attention (the DeepSeek-V2/V3 mechanism) at ONE head dim for q, k and v.

Built for the 3:1 GatedDeltaNet : MLA hybrid at head_dim 128. Per head the query and key are `[rope (rotary_dim) | nope
(head_dim - rotary_dim)]` with the rope lanes LEADING (the repository's partial-rotary convention), the value is head_dim
wide, so the Delta-Matching kernels see dense per-head bf16 q, k, v of one head dim and nothing under src/kernel changes:

    q          = q_proj(x)                      per head [rope | nope] (+ the output gate lanes when attn_output_gate)
    [c ; k_r]  = kv_a_proj(x)                   the joint KV latent (kv_lora_rank) and the shared rope key (rotary_dim)
    c          = kv_a_norm(c)                   RMSNorm over the latent (the inference-cache quantity)
    k_nope     = k_proj(c)                      W_UK: latent -> H x (head_dim - rotary_dim)
    v          = v_proj(c)                      W_UV: latent -> H x head_dim
    q = [q_rope_norm(q_r) | q_nope_norm(q_n)];  k_r = k_rope_norm(k_r) rotated ONCE at [B, 1, T, rd] and broadcast
    k = [k_r (shared across heads) | k_nope_norm(k_nope)]
    O = attention(q, k, v) with H == Hk (MLA is MHA on the wire);  out = o_proj(gate * O)

The four per-sub-vector RMSNorms replace the GQA mixer's q_norm / k_norm (qk_norm); a single full-head norm would
divide every head's copy of k_r by a head-dependent RMS. The score is ONE gemm over the concatenated [rope | nope]
head: never S_nope + S_rope in two gemms (an LSE merge is exact over keys, not over channels).

Registered mixers: 'mla' (bf16 core: sdpa | flash; the clean arm and every scored eval) and 'mla_dm' (the
Delta-Matching package's API wrapper `dm_attn_func_bhnd`: all seven attention matmuls in fp8 with the delta chosen by
dm_mode, stale -> stale, matched -> matched). The fp8 mixer has no fused-producer stage (the key is a concatenation of
two projections, so the producer cannot be fused into one GEMM); the projections themselves run in fp8 under
--fp8_scope ffn_qkvo (q_proj, k_proj, v_proj, o_proj and kv_a_proj by name).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import (apply_output_gate, q_proj_out_features, RMSNorm, rotate_half, split_q_gate)
from .attention import resolve_attn_impl, _sdpa_backend_ctx, _FLASH_FN


def _rope_leading(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rd: int) -> torch.Tensor:
    """Rotate the leading `rd` channels of x [B, h, T, D] (h = 1 or H) with cos/sin [B, T, rd]; the tail passes through."""
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    xr = x[..., :rd].float()
    out = (xr * c + rotate_half(xr) * s).to(x.dtype)
    return out if rd == x.shape[-1] else torch.cat([out, x[..., rd:]], dim=-1)


def _is_mla(cfg: ModelConfig) -> bool:
    names = set(cfg.layer_mixers or []) | {cfg.mixer}
    return bool(names & {"mla", "mla_dm"})


class MLAAttention(nn.Module):
    """Multi-head Latent Attention with a bf16 attention core (the clean arm; the eval of every MLA arm)."""
    _announced = False
    _fp8_core = False      # MLADMAttention replaces the bf16 core with the Delta-Matching one

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_heads          # MHA on the wire
        self.head_dim = cfg.head_dim
        self.rotary_dim = cfg.rotary_dim
        self.nope_dim = cfg.head_dim - cfg.rotary_dim
        self.kv_lora_rank = int(cfg.kv_lora_rank)
        self.scale = self.head_dim ** -0.5
        self.attn_output_gate = cfg.attn_output_gate
        self.qk_norm = cfg.qk_norm
        assert cfg.n_kv_heads == cfg.n_heads, "MLA is MHA on the wire: set n_kv_heads == n_heads"
        assert 0 < self.rotary_dim < self.head_dim, "MLA needs 0 < rotary_dim < head_dim (rope lanes + nope lanes)"
        assert self.kv_lora_rank % 16 == 0, "kv_lora_rank must be a multiple of 16 (fp8 GEMM)"
        # YaRN (the context-extension runs): the rope lanes' rotation comes from the shared RotaryEmbedding (its
        # inverse frequencies are laid out over rotary_dim and NTK-by-parts interpolated under rope_scaling); the
        # attention temperature is applied here exactly as GQAAttention / DMAttention do it: the post-RoPE q times
        # mscale^2 (logits x mscale^2), on the whole head (rope and nope lanes), for both the bf16 core and the
        # Delta-Matching core (MLADMAttention shares _qkv).
        from .layers import yarn_mscale
        self.attn_mscale = yarn_mscale(cfg.rope_scaling)
        d, H, D, rd, dn, dc = cfg.d_model, self.n_heads, self.head_dim, self.rotary_dim, self.nope_dim, self.kv_lora_rank
        # registration ORDER is load-bearing for the seed-matched init across the mla / mla_dm arms
        self.q_proj = nn.Linear(d, q_proj_out_features(cfg), bias=False)
        self.kv_a_proj = nn.Linear(d, dc + rd, bias=False)
        self.kv_a_norm = RMSNorm(dc, cfg.norm_eps, cfg.rms_norm_in_fp32, fused=cfg.fused_rmsnorm)
        self.k_proj = nn.Linear(dc, H * dn, bias=False)      # W_UK
        self.v_proj = nn.Linear(dc, H * D, bias=False)       # W_UV
        self.o_proj = nn.Linear(H * D, d, bias=False)
        if self.qk_norm:
            self.q_rope_norm = RMSNorm(rd, cfg.norm_eps, cfg.rms_norm_in_fp32, fused=cfg.fused_rmsnorm)
            self.q_nope_norm = RMSNorm(dn, cfg.norm_eps, cfg.rms_norm_in_fp32, fused=cfg.fused_rmsnorm)
            self.k_rope_norm = RMSNorm(rd, cfg.norm_eps, cfg.rms_norm_in_fp32, fused=cfg.fused_rmsnorm)
            self.k_nope_norm = RMSNorm(dn, cfg.norm_eps, cfg.rms_norm_in_fp32, fused=cfg.fused_rmsnorm)
        if self._fp8_core:     # mla_dm: the Delta-Matching core replaces the bf16 one, so attn_impl is not read there
            self.impl = resolve_attn_impl(cfg.attn_impl) if cfg.attn_impl in ("auto", "flash", "sdpa") else "sdpa"
        else:                  # the bf16 core: a closed set as in the GQA mixer, never a silent sdpa fallback
            self.impl = resolve_attn_impl(cfg.attn_impl)
            if self.impl not in ("flash", "sdpa"):
                raise ValueError(f"the mla mixer runs a bf16 core (attn_impl auto | flash | sdpa), got attn_impl="
                                 f"{cfg.attn_impl!r}; the Delta-Matching fp8 MLA core is the mixer 'mla_dm'")
        if not MLAAttention._announced:
            MLAAttention._announced = True
            print(f"[mla] {type(self).__name__}: H {H} (MHA on the wire), head_dim {D} = rope {rd} + nope {dn}, "
                  f"v {D}, latent {dc}, qk_norm={self.qk_norm} (four sub-vector norms), gate={self.attn_output_gate}, "
                  f"core={self._core_name()}, yarn_mscale={self.attn_mscale:.4f}", flush=True)

    def _core_name(self) -> str:
        return self.impl

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
        if self.attn_mscale != 1.0:                                       # YaRN temperature: logits x mscale^2 (q only)
            q = q * (self.attn_mscale * self.attn_mscale)
        k_r = _rope_leading(k_r, cos, sin, rd)                            # once, shared across heads
        k = torch.cat([k_r.expand(B, H, T, rd), k_n], dim=-1)             # rope lanes lead, as in q
        return q, k, v, gate

    def forward(self, x, cos, sin, attention_mask=None):
        assert attention_mask is None, "MLA uses packed windows (no padding mask)"
        B, T, _ = x.shape
        q, k, v, gate = self._qkv(x, cos, sin)
        if self.impl == "flash":   # the flash-attn function the GQA mixer resolved at import (DM_FLASH_IMPL honoured)
            out = _FLASH_FN(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True)
            if isinstance(out, tuple):
                out = out[0]
            out = out.reshape(B, T, self.n_heads * self.head_dim)
        else:
            with _sdpa_backend_ctx():
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(apply_output_gate(out, gate))


class MLADMAttention(MLAAttention):
    """MLA with the Delta-Matching fp8 core (the kernel package's API wrapper dm_attn_func_bhnd; delta from dm_mode)."""
    _fp8_announced = False
    _fp8_core = True

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        self._delta = {"stale": "stale", "matched": "matched"}.get(getattr(cfg, "dm_mode", ""), None)
        if self._delta is None:
            raise ValueError(f"mla_dm needs dm_mode stale (stale delta) or matched (matched delta), got {getattr(cfg, 'dm_mode', None)!r}")
        if cfg.head_dim not in (128, 256):
            raise ValueError("mla_dm runs on the Delta-Matching kernels: head_dim 128 or 256")
        super().__init__(cfg, layer_idx)
        try:
            import delta_matching  # noqa: F401
        except ImportError as e:
            raise ImportError("mla_dm needs the Delta-Matching kernel package: install the delta-matching kernel package "
                              "(see docs/SETUP.md section 3: git clone --recursive, then pip install --no-build-isolation .)") from e
        from delta_matching import dm_attn_func_bhnd, kernel_version
        from delta_matching.attn import _ops_gt
        from delta_matching.core import _check_hd
        self._attn = dm_attn_func_bhnd
        ops = _ops_gt()               # load now (and refuse a build without the global-memory twins)
        _check_hd(self.head_dim, ops)
        if not MLADMAttention._fp8_announced:
            MLADMAttention._fp8_announced = True
            print(f"[mla_dm] Delta-Matching core: delta={self._delta} (dm_mode {cfg.dm_mode}), kernel {kernel_version()}, "
                  f"build_hd128={int(ops.build_hd128())} build_gt={int(ops.build_gt())}", flush=True)

    def _core_name(self) -> str:
        return f"dm:{self._delta}"

    def forward(self, x, cos, sin, attention_mask=None):
        assert attention_mask is None, "mla_dm uses packed windows (no padding mask)"
        B, T, _ = x.shape
        q, k, v, gate = self._qkv(x, cos, sin)
        out = self._attn(q.contiguous(), k.contiguous(), v.contiguous(), causal=True, delta_mode=self._delta)  # (B, T, H, D)
        out = out.reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(apply_output_gate(out, gate))
