"""HF `PretrainedConfig` for the delta-matching-train decoder (`model_type="dmlm"`), so the model trained by
`src/pretrain` can be loaded with `transformers.AutoModelForCausalLM` (and thus by the evaluation engine,
src/evaluation). Carries the `ModelConfig` fields verbatim.
"""
from __future__ import annotations

from transformers import PretrainedConfig

from ..modeling.config import ModelConfig


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
        # Qwen3.5 architecture fields. These MUST round-trip: an export that drops them reloads as an ungated,
        # fully-rotated model with the wrong GDN geometry, which is a silent weight-shape mismatch on q_proj rather
        # than a loud failure.
        attn_output_gate: bool = False,
        partial_rotary_factor: float = 1.0,
        nope: bool = False,
        gdn_head_dim: int | None = None,
        gdn_num_heads: int | None = None,
        gdn_num_v_heads: int | None = None,
        kv_lora_rank: int | None = None,        # MLA
        mamba2_headdim: int | None = None,      # Mamba2 geometry
        mamba2_d_state: int = 128,
        mamba2_expand: int = 2,
        mamba2_ngroups: int = 1,
        mamba2_chunk_size: int = 256,
        kda_head_dim: int | None = None,        # KDA geometry
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
        z_loss_weight: float = 0.0,        # eval: no z-loss in the reported loss
        # The fp8 attention arms' delta mode (stale | matched | bf16_dp). Carried so an export records WHICH numerics
        # produced the weights; the launcher's clean-kernel patch drops it from config.json and keeps it in
        # config.train.json.
        dm_mode: str = "matched",
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
        # set BEFORE super().__init__: transformers' rope validation (standardize_rope_params, triggered when
        # rope_scaling/YaRN is present) reads self.max_position_embeddings during super init, so it must exist first.
        self.max_position_embeddings = max_seq_len
        self.mixer = mixer
        self.attn_impl = attn_impl
        self.layer_mixers = layer_mixers
        self.tie_embeddings = tie_embeddings
        self.z_loss_weight = z_loss_weight
        self.dm_mode = dm_mode
        # HF expects these names; mirror for generation utilities / tooling.
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
            kv_lora_rank=getattr(self, 'kv_lora_rank', None),
            mamba2_headdim=getattr(self, 'mamba2_headdim', None), mamba2_d_state=getattr(self, 'mamba2_d_state', 128),
            mamba2_expand=getattr(self, 'mamba2_expand', 2), mamba2_ngroups=getattr(self, 'mamba2_ngroups', 1),
            mamba2_chunk_size=getattr(self, 'mamba2_chunk_size', 256),
            kda_head_dim=getattr(self, 'kda_head_dim', None), kda_num_heads=getattr(self, 'kda_num_heads', None),
            kda_num_v_heads=getattr(self, 'kda_num_v_heads', None), kda_expand_v=getattr(self, 'kda_expand_v', 1.0),
            kda_conv_size=getattr(self, 'kda_conv_size', 4), kda_allow_neg_eigval=getattr(self, 'kda_allow_neg_eigval', False),
            kda_safe_gate=getattr(self, 'kda_safe_gate', False), kda_lower_bound=getattr(self, 'kda_lower_bound', None),
            max_seq_len=self.max_seq_len, mixer=self.mixer,
            attn_impl=self.attn_impl, layer_mixers=self.layer_mixers,
            tie_embeddings=self.tie_embeddings, z_loss_weight=self.z_loss_weight,
            dm_mode=getattr(self, "dm_mode", "matched"),
        )

    @classmethod
    def from_model_config(cls, mc: ModelConfig, **kwargs) -> "DMLMConfig":
        return cls(
            vocab_size=mc.vocab_size, d_model=mc.d_model, n_layers=mc.n_layers,
            n_heads=mc.n_heads, n_kv_heads=mc.n_kv_heads, head_dim=mc.head_dim,
            intermediate_size=mc.intermediate_size, ffn_mult=mc.ffn_mult,
            ffn_multiple_of=mc.ffn_multiple_of, rope_theta=mc.rope_theta,
            rope_scaling=mc.rope_scaling,
            norm_eps=mc.norm_eps, rms_norm_in_fp32=mc.rms_norm_in_fp32, qk_norm=mc.qk_norm,
            attn_output_gate=mc.attn_output_gate,
            partial_rotary_factor=mc.partial_rotary_factor,
            nope=bool(getattr(mc, "nope", False)),
            gdn_head_dim=mc.gdn_head_dim, gdn_num_heads=mc.gdn_num_heads,
            gdn_num_v_heads=mc.gdn_num_v_heads,
            kv_lora_rank=getattr(mc, 'kv_lora_rank', None),
            mamba2_headdim=getattr(mc, 'mamba2_headdim', None), mamba2_d_state=getattr(mc, 'mamba2_d_state', 128),
            mamba2_expand=getattr(mc, 'mamba2_expand', 2), mamba2_ngroups=getattr(mc, 'mamba2_ngroups', 1),
            mamba2_chunk_size=getattr(mc, 'mamba2_chunk_size', 256),
            kda_head_dim=getattr(mc, 'kda_head_dim', None), kda_num_heads=getattr(mc, 'kda_num_heads', None),
            kda_num_v_heads=getattr(mc, 'kda_num_v_heads', None), kda_expand_v=getattr(mc, 'kda_expand_v', 1.0),
            kda_conv_size=getattr(mc, 'kda_conv_size', 4), kda_allow_neg_eigval=getattr(mc, 'kda_allow_neg_eigval', False),
            kda_safe_gate=getattr(mc, 'kda_safe_gate', False), kda_lower_bound=getattr(mc, 'kda_lower_bound', None),
            max_seq_len=mc.max_seq_len, mixer=mc.mixer,
            attn_impl=mc.attn_impl, layer_mixers=mc.layer_mixers,
            tie_embeddings=mc.tie_embeddings, z_loss_weight=0.0,
            dm_mode=getattr(mc, "dm_mode", "matched"), **kwargs,
        )
