"""Per-slot state pool for the dmlm continuous-batching engine.

Sized from the *actual* mixer modules of a loaded dmlm model (a per-layer hybrid of
GQA / MLA / Mamba2 / GatedDeltaNet / KDA), mirroring `cache.py`'s slot-indexed idiom:

  - GQA / MLA layer     -> k_pool[i]/v_pool[i]   (S, max_len, n_kv_heads, head_dim)  growing KV
  - Mamba2 layer        -> ssm_pool[i]/conv_pool[i]                                   O(1) recurrent
  - GatedDeltaNet / KDA -> recur_pool[i]/gconv_pool[i]                                O(1) recurrent

A sequence's state never moves once admitted: the decode kernels gather/scatter by slot id
(`flash_attn_with_kvcache(cache_batch_idx=)`, `selective_state_update(state_batch_indices=)`,
fla recurrent with an index-gathered initial state). Free-list allocator, one slot per
concurrent sequence.
"""
from __future__ import annotations

import torch


def mixer_kind(mixer) -> str:
    """Map a dmlm mixer module to its engine kind by class name (robust to fla/mamba_ssm
    lazy imports: no isinstance against the optional backends)."""
    n = type(mixer).__name__
    if n in ("GQAAttention", "EngineGQA"):
        return "gqa"
    if n in ("Mamba2Mixer", "EngineMamba2"):
        return "mamba2"
    if n in ("GatedDeltaNetMixer", "EngineGDN"):
        return "gdn"
    if n in ("KDAMixer", "EngineKDA"):   # Kimi Delta Attention: the GDN pools, sized from mx.kda
        return "kda"
    # MLA is MHA on the wire (MLAAttention asserts n_kv_heads == n_heads) and _qkv assembles K to
    # head_dim wide (rope lanes + nope lanes), so it pools exactly like GQA. MLADMAttention is
    # deliberately NOT listed: rebinding it to EngineMLA would silently swap its Delta-Matching core for
    # the bf16 one. Eval exports are patched to attn_impl=sdpa (eval_arm.sh asserts it), so every
    # MLA arm evaluates as plain MLAAttention anyway. The training-only `dm` mixer is likewise not
    # listed: exports are patched to `gqa` before evaluation.
    if n in ("MLAAttention", "EngineMLA"):
        return "mla"
    raise ValueError(f"unknown dmlm mixer for engine pool: {n}")


class DMStatePool:
    def __init__(self, layers, max_num_seqs, max_len, device,
                 kv_dtype: torch.dtype = torch.bfloat16, state_dtype: torch.dtype | None = None):
        self.S = int(max_num_seqs)
        self.max_len = int(max_len)
        self.device = device
        self.kv_dtype = kv_dtype
        self.state_dtype = state_dtype if state_dtype is not None else kv_dtype

        self.k_pool: dict[int, torch.Tensor] = {}
        self.v_pool: dict[int, torch.Tensor] = {}
        self.ssm_pool: dict[int, torch.Tensor] = {}
        self.conv_pool: dict[int, torch.Tensor] = {}
        self.recur_pool: dict[int, torch.Tensor] = {}   # GDN recurrent state (fp32)
        self.gconv_q: dict[int, torch.Tensor] = {}      # GDN q/k/v short-conv states
        self.gconv_k: dict[int, torch.Tensor] = {}
        self.gconv_v: dict[int, torch.Tensor] = {}

        S, ml = self.S, self.max_len
        for i, blk in enumerate(layers):
            mx = blk.mixer
            kind = mixer_kind(mx)
            if kind == "gqa":
                n_kv, hd = mx.n_kv_heads, mx.head_dim
                self.k_pool[i] = torch.zeros(S, ml, n_kv, hd, device=device, dtype=kv_dtype)
                self.v_pool[i] = torch.zeros(S, ml, n_kv, hd, device=device, dtype=kv_dtype)
            elif kind == "mla":
                # K is [B, T, H, head_dim] (rope lanes + nope lanes), V is head_dim wide: GQA's pool
                # with n_heads. The compressed latent would be smaller, but the assembled form is what
                # flash_attn_with_kvcache can append to in place, and eval batch sizes make it moot.
                n_h, hd = mx.n_heads, mx.head_dim
                self.k_pool[i] = torch.zeros(S, ml, n_h, hd, device=device, dtype=kv_dtype)
                self.v_pool[i] = torch.zeros(S, ml, n_h, hd, device=device, dtype=kv_dtype)
            elif kind == "mamba2":
                m = mx.mamba  # mamba_ssm.Mamba2
                conv_dim = m.conv1d.weight.shape[0]          # d_inner + 2*ngroups*d_state
                self.conv_pool[i] = torch.zeros(S, conv_dim, m.d_conv, device=device, dtype=kv_dtype)
                self.ssm_pool[i] = torch.zeros(S, m.nheads, m.headdim, m.d_state,
                                               device=device, dtype=self.state_dtype)
            elif kind == "gdn":
                g = mx.gdn  # fla.layers.GatedDeltaNet
                # recurrent delta-rule state is fp32 (chunk/fused_recurrent return fp32 state)
                # state is per V head: GVA (num_v_heads > num_heads) returns [B, num_v_heads, K, V];
                # equal-heads models are unchanged.
                n_state_heads = getattr(g, "num_v_heads", None) or g.num_heads
                self.recur_pool[i] = torch.zeros(S, n_state_heads, g.head_k_dim, g.head_v_dim,
                                                 device=device, dtype=torch.float32)
                if g.use_short_conv:  # 3 separate q/k/v short-conv states
                    cs = g.conv_size
                    self.gconv_q[i] = torch.zeros(S, g.q_conv1d.weight.shape[0], cs, device=device, dtype=kv_dtype)
                    self.gconv_k[i] = torch.zeros(S, g.k_conv1d.weight.shape[0], cs, device=device, dtype=kv_dtype)
                    self.gconv_v[i] = torch.zeros(S, g.v_conv1d.weight.shape[0], cs, device=device, dtype=kv_dtype)
            elif kind == "kda":
                g = mx.kda  # fla.layers.kda.KimiDeltaAttention: the same state layout as GatedDeltaNet
                n_state_heads = getattr(g, "num_v_heads", None) or g.num_heads
                self.recur_pool[i] = torch.zeros(S, n_state_heads, g.head_k_dim, g.head_v_dim,
                                                 device=device, dtype=torch.float32)
                if g.use_short_conv:
                    cs = g.conv_size
                    self.gconv_q[i] = torch.zeros(S, g.q_conv1d.weight.shape[0], cs, device=device, dtype=kv_dtype)
                    self.gconv_k[i] = torch.zeros(S, g.k_conv1d.weight.shape[0], cs, device=device, dtype=kv_dtype)
                    self.gconv_v[i] = torch.zeros(S, g.v_conv1d.weight.shape[0], cs, device=device, dtype=kv_dtype)

        # Per-slot current length (= KV write offset / RoPE position for the next decode token).
        self.seqlens = torch.zeros(S, dtype=torch.int32, device=device)
        self._free: list[int] = list(range(S))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def alloc(self) -> int:
        return self._free.pop()

    def free(self, slot: int) -> None:
        self.seqlens[slot] = 0
        self._free.append(slot)
