"""Architecture configuration for the pretraining model.

Llama-style decoder: RMSNorm pre-norm, RoPE, a swappable token *mixer* (gqa, the recurrent gated_deltanet /
mamba2 / kda, mla, and the fp8 attention mixers dm / mla_dm), SwiGLU MLP, tied embeddings. The SwiGLU
intermediate-size heuristic (8/3 * d_model, rounded up to `ffn_multiple_of`) gives, e.g.,
d_model 768 -> d_ff 2048 and d_model 1024 -> d_ff 2816.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    # --- core dims ---
    vocab_size: int = 32768
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4              # GQA: n_heads % n_kv_heads == 0; == n_heads is MHA
    head_dim: int | None = None     # default d_model // n_heads

    # --- FFN (SwiGLU) ---
    intermediate_size: int | None = None  # explicit overrides the heuristic below
    ffn_mult: float = 8.0 / 3.0
    ffn_multiple_of: int = 256

    # --- positional / norm ---
    rope_theta: float = 10000.0
    # YaRN RoPE context-extension (DeepSeek-V3 long-context recipe). None -> vanilla theta RoPE
    # (bit-identical to no-scaling). When set, e.g.
    #   {"type":"yarn","factor":8.0,"original_max_position_embeddings":8192,
    #    "beta_fast":32,"beta_slow":1}
    # RotaryEmbedding NTK-by-parts-interpolates inv_freq (high-freq/local bands untouched,
    # low-freq/long-range bands scaled by 1/factor) and the attention temperature mscale is
    # folded into q in the mixers (rope_theta is NOT changed). See layers.py::yarn_inv_freq.
    rope_scaling: dict | None = None
    # PARTIAL RoPE (Qwen3.5 idiom, 2026-08-04). Rotate only the first
    # `int(head_dim * partial_rotary_factor)` channels of each head and pass the rest through
    # unrotated; 1.0 == every channel, i.e. the classic behaviour, bit-identical to before.
    # Qwen3.5-2B-Base uses 0.25 with head_dim 256, so 64 rotated / 192 untouched. The unrotated
    # tail carries position-independent content, which is why the factor can be this small.
    # NOTE: incompatible with the fused Liger RoPE kernel (it assumes a full rotation), so
    # `fused_rope` is force-disabled below when the factor is < 1.
    partial_rotary_factor: float = 1.0
    # NoPE (2026-09-16): no positional encoding in attention. Realized as an IDENTITY rotation (cos = 1, sin = 0 on
    # the rotated slice), which is exact in bf16 and leaves every kernel path untouched (sdpa/cuDNN clean, the fused
    # fp8 RoPE+quantize producers, the fused RoPE backward, the HF export and its eval); rope_theta and the partial
    # factor are ignored when set. Exported to the HF config so the eval model is NoPE too.
    nope: bool = False
    norm_eps: float = 1e-5
    max_seq_len: int = 2048

    # --- mixer selection (all registered in modeling/mixers.py::MIXER_REGISTRY) ---
    # gqa (bf16 attention), gated_deltanet / mamba2 / kda (recurrent), mla (bf16 MLA), dm (GQA with the Delta-Matching
    # fp8 attention core and the fused fp8 producers), mla_dm (MLA on the Delta-Matching fp8 core).
    mixer: Literal["gqa", "gated_deltanet", "mamba2", "kda", "mla", "dm", "mla_dm"] = "gqa"
    # dm: the plain Delta-Matching fp8 kernel on the GQA mixer (no fused producers; delta from dm_mode).
    # te_fp8: the core from Transformer Engine's cuDNN fp8 fused attention (DelayedScaling), head_dim 128 only on sm90,
    # a Python environment with TE; modeling/te_attn.py. DISTINCT NUMERICAL BACKENDS: never mix or resume across them.
    attn_impl: Literal["auto", "flash", "sdpa", "dm", "te_fp8"] = "auto"
    qk_norm: bool = False             # per-head RMSNorm on q,k before RoPE (attention stability)
    # OUTPUT-GATED ATTENTION (Qwen3-Next / Qwen3.5 idiom, 2026-08-04). q_proj emits 2*q_dim;
    # the second half is a gate applied as `attn_out * sigmoid(gate)` before o_proj. Costs one
    # extra d_model x q_dim matrix per attention layer. False == the classic ungated path.
    attn_output_gate: bool = False
    # Per-layer override: a pattern repeated over the layers (layer i uses layer_mixers[i % len]), e.g. the 3:1
    # hybrid ["gated_deltanet", "gated_deltanet", "gated_deltanet", "gqa"]. None = `mixer` on every layer.
    layer_mixers: list[str] | None = None
    # GDN head geometry, DECOUPLED from the attention head geometry (Qwen3.5 idiom, 2026-08-04):
    # Qwen3.5-2B-Base runs its linear-attention layers at 16 heads x 128 while its full-attention
    # layers run 8 heads x 256, so the two cannot share `head_dim`. None keeps the legacy
    # behaviour (GDN inherits cfg.head_dim and derives num_heads = d_model // head_dim).
    gdn_head_dim: int | None = None
    gdn_num_heads: int | None = None
    # GDN V-head count, decoupled from the QK-head count (scale-ladder large cell, 2026-08-18).
    # fla's GatedDeltaNet defaults num_v_heads = num_heads with head_v_dim = head_dim * expand_v
    # (expand_v=2, never overridden), i.e. V is 2x the QK width. Setting this LARGER than
    # gdn_num_heads widens value capacity beyond 2*d_model (the 5.289B cell runs QK 16 / V 32 =
    # value_dim 2.67*d_model). None keeps fla's default -- every pre-existing config builds
    # bit-identically.
    gdn_num_v_heads: int | None = None
    # MLA (2026-09-18, modeling/mla.py): the joint KV latent width; rotary_dim (= head_dim * partial_rotary_factor)
    # is the shared rope key width, head_dim - rotary_dim the per-head nope width, head_dim the value width.
    kv_lora_rank: int | None = None
    # Mamba2 (2026-09-18, modeling/mamba2.py): mamba_ssm.Mamba2's geometry, exposed so the 1.665 B Mamba2:GQA
    # hybrid can be parameter-matched; None = mamba_ssm's defaults (headdim None = this config's head_dim, the
    # 360M study's convention).
    mamba2_headdim: int | None = None
    mamba2_d_state: int = 128
    mamba2_expand: int = 2
    mamba2_ngroups: int = 1
    mamba2_chunk_size: int = 256
    # KDA (2026-09-22, modeling/kda.py wrapping fla.layers.kda.KimiDeltaAttention): head_dim / num_heads fall back to the
    # gdn_* fields (a KDA:GQA hybrid reuses the 16 x 128 linear geometry); expand_v 1.0 = fla's KDA default (value width =
    # key width; fla's GDN uses 2.0); num_v_heads None = num_heads; lower_bound None = fla's default gate range.
    kda_head_dim: int | None = None
    kda_num_heads: int | None = None
    kda_num_v_heads: int | None = None
    kda_expand_v: float = 1.0
    kda_conv_size: int = 4
    kda_allow_neg_eigval: bool = False
    kda_safe_gate: bool = False
    kda_lower_bound: float | None = None

    # --- training-time numerics ---
    tie_embeddings: bool = True
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    initializer_range: float = 0.02
    z_loss_weight: float = 1e-4       # logit-normalizer regularizer
    rms_norm_in_fp32: bool = True     # compute RMSNorm reduction in fp32 (stability)

    # --- fused (Liger Triton) kernels for the eager primitives (opt-in; default = eager) ---
    # Each replaces a memory-bound eager op (run TWICE under full activation checkpointing) with a
    # fused Liger kernel; parity-checked within fp tolerance (src/pretrain/scripts/test_fused_layers.py, run as
    # `python -m pretrain.scripts.test_fused_layers`). Guarded
    # like fused_ce: if liger_kernel is unavailable the eager path is used. Do NOT affect the fp8
    # attention kernel, YaRN, or qk-norm math.
    fused_rmsnorm: bool = False        # RMSNorm -> LigerRMSNormFunction (llama casting_mode == in_fp32)
    fused_rope: bool = False           # apply_rotary -> liger_rotary_pos_emb
    fused_swiglu: bool = False         # SwiGLU silu(gate)*up -> LigerSiLUMulFunction

    # --- the fp8 attention arms (modeling/dm_attention.py, modeling/mla.py::MLADMAttention, attn_impl 'dm') ---
    # dm_mode is the backward's delta realization: "stale" (delta = rowsum(dO * O) from the forward's O, the naive
    # fp8 backward), "matched" (delta from the backward's own realized fp8 P / dP pair: Delta-Matching) or "bf16_dp"
    # (the stale delta with the dP matmul kept in bf16, the paper's bf16-dP arm; needs the delta_matching kernel built with
    # DM_BF16_DP=1, checked at model build; the dm mixer only, head_dim 128 only).
    dm_mode: str = "matched"
    # The dm mixer's fused fp8 producers (the only realization of that mixer; kept for the launch contract and the
    # configs of the study, which set it explicitly). The plain kernel route is the gqa mixer with attn_impl 'dm'.
    dm_fused_producers: bool = True

    # filled in __post_init__
    _resolved: bool = field(default=False, repr=False)

    def __post_init__(self):
        if self.head_dim is None:
            assert self.d_model % self.n_heads == 0, "d_model must divide by n_heads when head_dim is None"
            self.head_dim = self.d_model // self.n_heads
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.intermediate_size is None:
            raw = self.ffn_mult * self.d_model
            m = self.ffn_multiple_of
            self.intermediate_size = int(((int(raw) + m - 1) // m) * m)
        assert 0.0 < self.partial_rotary_factor <= 1.0, (
            f"partial_rotary_factor must be in (0, 1], got {self.partial_rotary_factor}")
        assert self.dm_mode in ("stale", "matched", "bf16_dp"), f"dm_mode must be stale|matched|bf16_dp, got {self.dm_mode!r}"
        assert self.rotary_dim % 2 == 0 and self.rotary_dim > 0, (
            f"rotary_dim = head_dim({self.head_dim}) * partial_rotary_factor"
            f"({self.partial_rotary_factor}) = {self.rotary_dim}, which must be a positive even number")
        names = set(self.layer_mixers or []) | {self.mixer}
        if names & {"mla", "mla_dm"}:
            assert self.kv_lora_rank and self.kv_lora_rank % 16 == 0, "MLA needs kv_lora_rank (a multiple of 16)"
            assert self.n_kv_heads == self.n_heads, "MLA is MHA on the wire: n_kv_heads must equal n_heads"
            assert 0 < self.rotary_dim < self.head_dim, "MLA needs 0 < rotary_dim < head_dim"
            assert self.head_dim in (128, 256), "MLA on the Delta-Matching kernels: head_dim 128 or 256"
        if self.rotary_dim != self.head_dim and self.fused_rope:
            # Liger's fused RoPE rotates the whole head; silently letting it run on a partial
            # config would rotate the pass-through tail and corrupt every attention logit.
            self.fused_rope = False
        self._resolved = True

    @property
    def rotary_dim(self) -> int:
        """Number of leading head_dim channels that receive RoPE (== head_dim when full)."""
        return int(self.head_dim * self.partial_rotary_factor)

    # --- derived helpers ---
    @property
    def n_rep(self) -> int:
        """How many query heads share each KV head (GQA group size)."""
        return self.n_heads // self.n_kv_heads

    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def mixer_for_layer(self, layer_idx: int) -> str:
        if self.layer_mixers is not None:
            return self.layer_mixers[layer_idx % len(self.layer_mixers)]
        return self.mixer

    def num_params_analytic(self) -> int | None:
        """Closed-form parameter count (tied embeddings), layer by layer. Verified against the live
        module in the smoke test. None when a layer is recurrent (gated_deltanet / mamba2 / kda):
        those wrap fla / mamba_ssm modules whose parameter layout is the library's, so there is no
        closed form here.

          embed         = V * d                         (tied: lm_head shares it)
          per attn      = d*q_dim + 2*d*kv_dim + q_dim*d  (gqa / dm; mla / mla_dm below)
          per mlp       = 3 * d * d_ff
          per norms     = 2 * d                         (pre-attn + pre-mlp RMSNorm)
          final norm    = d
        """
        d, V = self.d_model, self.vocab_size
        q_out = self.q_dim * (2 if self.attn_output_gate else 1)   # gated q_proj emits q + gate
        mlp = 3 * d * self.intermediate_size
        total = V * d + d
        for i in range(self.n_layers):
            kind = self.mixer_for_layer(i)
            if kind in ("mla", "mla_dm"):
                # MLA (modeling/mla.py): q_proj + kv_a_proj (latent + rope key) + latent norm + W_UK + W_UV + o_proj
                dc, rd, dn = int(self.kv_lora_rank), self.rotary_dim, self.head_dim - self.rotary_dim
                attn = d * q_out + d * (dc + rd) + dc + dc * (self.n_heads * dn) + dc * self.q_dim + self.q_dim * d
                if self.qk_norm:
                    attn += 2 * (rd + dn)   # four sub-vector norms = 2 * head_dim
            elif kind in ("gqa", "dm"):
                attn = d * q_out + 2 * d * self.kv_dim + self.q_dim * d
                if self.qk_norm:
                    attn += 2 * self.head_dim   # q_norm + k_norm RMSNorm weights (per-head, over head_dim)
            else:
                return None                 # a recurrent (or user-registered) mixer: no closed form
            total += attn + mlp + 2 * d
        return total
