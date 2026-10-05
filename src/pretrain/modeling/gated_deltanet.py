"""Gated DeltaNet mixer (wraps flash-linear-attention `fla.layers.GatedDeltaNet`).

A linear-attention token mixer: O(L) time / O(1) state (vs attention's O(L²) compute),
so it is the recurrent baseline for the hybrid comparison. RoPE does not apply
(it carries its own gating/decay), so `cos`/`sin` are accepted and ignored: keeping
the drop-in `forward(x, cos, sin, attention_mask)` mixer signature. `fla` is imported
lazily so the registry import stays cheap and GQA-only runs need no `fla`.
"""
from __future__ import annotations

import torch.nn as nn

from .config import ModelConfig


class GatedDeltaNetMixer(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        from fla.layers import GatedDeltaNet  # lazy
        # GDN's head geometry is independent of the attention layers' (Qwen3.5 runs 16x128 linear
        # against 8x256 full). Falling back to cfg.head_dim reproduces the pre-2026-08-04 default.
        head_dim = cfg.gdn_head_dim or cfg.head_dim
        num_heads = cfg.gdn_num_heads or max(1, cfg.d_model // head_dim)
        # num_v_heads only when explicitly set: passing None is NOT equivalent for older fla
        # versions, and omitting the kwarg keeps every pre-2026-08-18 run byte-identical.
        extra = {}
        if getattr(cfg, "gdn_num_v_heads", None):
            extra["num_v_heads"] = cfg.gdn_num_v_heads
        self.gdn = GatedDeltaNet(
            hidden_size=cfg.d_model,
            head_dim=head_dim,
            num_heads=num_heads,
            **extra,
            mode="chunk",
            use_gate=True,
            use_short_conv=True,
            layer_idx=layer_idx,
        )

    def forward(self, x, cos=None, sin=None, attention_mask=None):
        out = self.gdn(x)
        return out[0] if isinstance(out, tuple) else out
