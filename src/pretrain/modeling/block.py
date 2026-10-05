"""Pre-norm decoder block: x += mixer(norm(x)); x += mlp(norm(x)).

This is the unit that FSDP2 shards (one `fully_shard` per block) and that
activation-checkpointing wraps, so it is a self-contained nn.Module taking the
shared rotary tables as forward args (no per-block RoPE state).
"""
from __future__ import annotations

import torch.nn as nn

from .config import ModelConfig
from .layers import RMSNorm, SwiGLUMLP  # noqa: F401 (SwiGLUMLP re-exported for callers)
from .mixers import build_mixer
from .mlp_fp8glue import build_mlp


class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps, cfg.rms_norm_in_fp32,
                                        fused=cfg.fused_rmsnorm)
        self.mixer = build_mixer(cfg.mixer_for_layer(layer_idx), cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg.d_model, cfg.norm_eps, cfg.rms_norm_in_fp32,
                                                fused=cfg.fused_rmsnorm)
        # SwiGLUMLP, or (env DM_FP8GLUE=1, the default) its fp8-glue subclass with identical params/FQNs
        self.mlp = build_mlp(cfg)
        self.resid_drop = nn.Dropout(cfg.resid_dropout) if cfg.resid_dropout > 0 else nn.Identity()
        self._fp8glue = None   # GlueCtl after install_fp8glue (delayed-scaling state), else None

    def _fp8glue_enter(self):
        """Once per microbatch, push the previous microbatch's amaxes into the delayed-scaling
        history (no-op in dynamic mode / when the port is off). Idempotent under AC recompute."""
        ctl = getattr(self, "_fp8glue", None)
        if ctl is not None:
            ctl.state.enter_microbatch()

    def forward(self, x, cos, sin, attention_mask=None):
        self._fp8glue_enter()
        x = x + self.resid_drop(self.mixer(self.input_layernorm(x), cos, sin, attention_mask))
        x = x + self.resid_drop(self.mlp(self.post_attention_layernorm(x)))
        return x
