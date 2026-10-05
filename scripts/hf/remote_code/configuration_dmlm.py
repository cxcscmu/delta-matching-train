"""HF `PretrainedConfig` for DMLM, the hybrid decoder of the Delta-Matching study.

Self-contained copy of delta-matching-train's `src/pretrain/hf/configuration_dmlm.py`, with the backbone's
`ModelConfig` (`src/pretrain/modeling/config.py`) vendored below so the checkpoint loads through
`trust_remote_code` without the training source tree. Field names, defaults and `to_model_config` are unchanged, so a
`config.json` written by the exporter round-trips exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from transformers import PretrainedConfig


@dataclass
class ModelConfig:
    """Backbone architecture (the training repository's `ModelConfig`). Only the fields the shipped mixers read matter here; the rest
    are carried so the exporter's configs keep their exact meaning."""
    # core dims
    vocab_size: int = 32768
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4              # GQA: n_heads % n_kv_heads == 0; == n_heads is MHA
    head_dim: int | None = None      # default d_model // n_heads
    # FFN (SwiGLU)
    intermediate_size: int | None = None
    ffn_mult: float = 8.0 / 3.0
    ffn_multiple_of: int = 256
    # positional / norm
    rope_theta: float = 10000.0
    rope_scaling: dict | None = None          # {"type": "yarn", ...} for YaRN context extension, else None
    partial_rotary_factor: float = 1.0        # rotate the leading head_dim * factor channels only
    nope: bool = False                        # identity rotation (no positional encoding in attention)
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    # mixers
    mixer: str = "gqa"
    attn_impl: str = "auto"
    qk_norm: bool = False                     # per-head RMSNorm on q, k before RoPE
    attn_output_gate: bool = False            # q_proj emits 2 * q_dim; out * sigmoid(gate) before o_proj
    layer_mixers: list[str] | None = None     # per-layer pattern, repeated over n_layers
    gdn_head_dim: int | None = None
    gdn_num_heads: int | None = None
    gdn_num_v_heads: int | None = None
    kv_lora_rank: int | None = None
    mamba2_headdim: int | None = None
    mamba2_d_state: int = 128
    mamba2_expand: int = 2
    mamba2_ngroups: int = 1
    mamba2_chunk_size: int = 256
    kda_head_dim: int | None = None
    kda_num_heads: int | None = None
    kda_num_v_heads: int | None = None
    kda_expand_v: float = 1.0
    kda_conv_size: int = 4
    kda_allow_neg_eigval: bool = False
    kda_safe_gate: bool = False
    kda_lower_bound: float | None = None
    # numerics
    tie_embeddings: bool = True
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    initializer_range: float = 0.02
    z_loss_weight: float = 1e-4
    rms_norm_in_fp32: bool = True
    fused_rmsnorm: bool = False
    fused_rope: bool = False
    fused_swiglu: bool = False
    # the fp8 training arms' knobs, carried for config fidelity only (not read by the shipped mixers)
    dm_mode: str = "matched"
    dm_fused_producers: bool = True
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
        assert self.rotary_dim % 2 == 0 and self.rotary_dim > 0, (
            f"rotary_dim = head_dim({self.head_dim}) * partial_rotary_factor"
            f"({self.partial_rotary_factor}) = {self.rotary_dim}, which must be a positive even number")
        if self.rotary_dim != self.head_dim and self.fused_rope:
            self.fused_rope = False
        self._resolved = True

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def n_rep(self) -> int:
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


class DMLMConfig(PretrainedConfig):
    model_type = "dmlm"

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        n_kv_heads: int = 4,
        head_dim: int | None = None,
        intermediate_size: int | None = None,
        ffn_mult: float = 8.0 / 3.0,
        ffn_multiple_of: int = 256,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        norm_eps: float = 1e-5,
        rms_norm_in_fp32: bool = True,
        qk_norm: bool = False,
        attn_output_gate: bool = False,
        partial_rotary_factor: float = 1.0,
        nope: bool = False,
        gdn_head_dim: int | None = None,
        gdn_num_heads: int | None = None,
        gdn_num_v_heads: int | None = None,
        kv_lora_rank: int | None = None,
        mamba2_headdim: int | None = None,
        mamba2_d_state: int = 128,
        mamba2_expand: int = 2,
        mamba2_ngroups: int = 1,
        mamba2_chunk_size: int = 256,
        kda_head_dim: int | None = None,
        kda_num_heads: int | None = None,
        kda_num_v_heads: int | None = None,
        kda_expand_v: float = 1.0,
        kda_conv_size: int = 4,
        kda_allow_neg_eigval: bool = False,
        kda_safe_gate: bool = False,
        kda_lower_bound: float | None = None,
        max_seq_len: int = 2048,
        mixer: str = "gqa",
        attn_impl: str = "auto",
        layer_mixers: list[str] | None = None,
        tie_embeddings: bool = True,
        z_loss_weight: float = 0.0,
        dm_mode: str = "matched",       # the fp8 arms' delta mode; a provenance field, not read by the shipped mixers
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.ffn_mult = ffn_mult
        self.ffn_multiple_of = ffn_multiple_of
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.norm_eps = norm_eps
        self.rms_norm_in_fp32 = rms_norm_in_fp32
        self.qk_norm = qk_norm
        self.attn_output_gate = attn_output_gate
        self.partial_rotary_factor = partial_rotary_factor
        self.nope = bool(nope)
        self.gdn_head_dim = gdn_head_dim
        self.gdn_num_heads = gdn_num_heads
        self.gdn_num_v_heads = gdn_num_v_heads
        self.kv_lora_rank = kv_lora_rank
        self.mamba2_headdim = mamba2_headdim
        self.mamba2_d_state = mamba2_d_state
        self.mamba2_expand = mamba2_expand
        self.mamba2_ngroups = mamba2_ngroups
        self.mamba2_chunk_size = mamba2_chunk_size
        self.kda_head_dim = kda_head_dim
        self.kda_num_heads = kda_num_heads
        self.kda_num_v_heads = kda_num_v_heads
        self.kda_expand_v = kda_expand_v
        self.kda_conv_size = kda_conv_size
        self.kda_allow_neg_eigval = kda_allow_neg_eigval
        self.kda_safe_gate = kda_safe_gate
        self.kda_lower_bound = kda_lower_bound
        self.max_seq_len = max_seq_len
        # set before super().__init__: transformers' rope validation reads max_position_embeddings during init
        self.max_position_embeddings = max_seq_len
        self.mixer = mixer
        self.attn_impl = attn_impl
        self.layer_mixers = layer_mixers
        self.tie_embeddings = tie_embeddings
        self.z_loss_weight = z_loss_weight
        self.dm_mode = dm_mode
        kwargs.setdefault("max_position_embeddings", max_seq_len)
        kwargs.setdefault("hidden_size", d_model)
        kwargs.setdefault("num_hidden_layers", n_layers)
        kwargs.setdefault("num_attention_heads", n_heads)
        kwargs.setdefault("tie_word_embeddings", tie_embeddings)
        super().__init__(**kwargs)

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size, d_model=self.d_model, n_layers=self.n_layers,
            n_heads=self.n_heads, n_kv_heads=self.n_kv_heads, head_dim=self.head_dim,
            intermediate_size=self.intermediate_size, ffn_mult=self.ffn_mult,
            ffn_multiple_of=self.ffn_multiple_of, rope_theta=self.rope_theta,
            rope_scaling=self.rope_scaling,
            norm_eps=self.norm_eps, rms_norm_in_fp32=self.rms_norm_in_fp32, qk_norm=self.qk_norm,
            attn_output_gate=self.attn_output_gate,
            partial_rotary_factor=self.partial_rotary_factor,
            nope=bool(getattr(self, "nope", False)),
            gdn_head_dim=self.gdn_head_dim, gdn_num_heads=self.gdn_num_heads,
            gdn_num_v_heads=self.gdn_num_v_heads,
            kv_lora_rank=getattr(self, "kv_lora_rank", None),
            mamba2_headdim=getattr(self, "mamba2_headdim", None), mamba2_d_state=getattr(self, "mamba2_d_state", 128),
            mamba2_expand=getattr(self, "mamba2_expand", 2), mamba2_ngroups=getattr(self, "mamba2_ngroups", 1),
            mamba2_chunk_size=getattr(self, "mamba2_chunk_size", 256),
            kda_head_dim=getattr(self, "kda_head_dim", None), kda_num_heads=getattr(self, "kda_num_heads", None),
            kda_num_v_heads=getattr(self, "kda_num_v_heads", None), kda_expand_v=getattr(self, "kda_expand_v", 1.0),
            kda_conv_size=getattr(self, "kda_conv_size", 4), kda_allow_neg_eigval=getattr(self, "kda_allow_neg_eigval", False),
            kda_safe_gate=getattr(self, "kda_safe_gate", False), kda_lower_bound=getattr(self, "kda_lower_bound", None),
            max_seq_len=self.max_seq_len, mixer=self.mixer,
            attn_impl=self.attn_impl, layer_mixers=self.layer_mixers,
            tie_embeddings=self.tie_embeddings, z_loss_weight=self.z_loss_weight,
            dm_mode=getattr(self, "dm_mode", "matched"),
        )
