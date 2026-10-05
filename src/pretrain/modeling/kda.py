"""Kimi Delta Attention mixer (wraps flash-linear-attention `fla.layers.kda.KimiDeltaAttention`; 2026-09-22).

The third recurrent baseline of the hybrid comparison, after GatedDeltaNet (gated_deltanet.py) and Mamba2 (mamba2.py):
the delta rule of GDN with a FINE-GRAINED (per key channel) decay gate, from Kimi Linear (Moonshot, 2025). O(L) time,
O(1) state; no RoPE (the decay carries the positional dynamics), so `cos`/`sin` are accepted and ignored, keeping the
drop-in `forward(x, cos, sin, attention_mask)` mixer signature. `fla` is imported lazily. Geometry from ModelConfig
(kda_* fields; head_dim / num_heads fall back to the gdn_* fields so a KDA:GQA hybrid can reuse a GDN:GQA yaml's
16 x 128 linear geometry); `kda_expand_v` (fla default 1.0: value width = key width; fla's GatedDeltaNet uses 2.0, which
is why a KDA layer at expand_v 1 carries fewer parameters and the hybrid yaml re-matches the count through the FFN).
The inner module is `self.kda`, and the fp8 scope filter (parallel/wrap.py, scope ffn_qkvo) excludes it by that nesting
exactly as it excludes `.gdn.`: the recurrent projections stay bf16 in every arm.
"""
from __future__ import annotations

import torch.nn as nn

from .config import ModelConfig


class KDAMixer(nn.Module):
    _announced = False

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        from fla.layers.kda import KimiDeltaAttention  # lazy
        head_dim = cfg.kda_head_dim or cfg.gdn_head_dim or cfg.head_dim
        num_heads = cfg.kda_num_heads or cfg.gdn_num_heads or max(1, cfg.d_model // head_dim)
        extra = {}
        if getattr(cfg, "kda_num_v_heads", None):
            extra["num_v_heads"] = int(cfg.kda_num_v_heads)
        if getattr(cfg, "kda_lower_bound", None) is not None:
            extra["lower_bound"] = float(cfg.kda_lower_bound)
        self.kda = KimiDeltaAttention(
            hidden_size=cfg.d_model,
            head_dim=head_dim,
            num_heads=num_heads,
            expand_v=float(getattr(cfg, "kda_expand_v", 1.0)),
            mode="chunk",
            use_short_conv=True,
            conv_size=int(getattr(cfg, "kda_conv_size", 4)),
            allow_neg_eigval=bool(getattr(cfg, "kda_allow_neg_eigval", False)),
            safe_gate=bool(getattr(cfg, "kda_safe_gate", False)),
            layer_idx=layer_idx,
            **extra,
        )
        if not KDAMixer._announced:
            KDAMixer._announced = True
            import fla
            k = self.kda
            print(f"[kda] fla {fla.__version__} KimiDeltaAttention: heads {k.num_heads} (v heads {k.num_v_heads}) x head_dim "
                  f"{k.head_k_dim} (v {k.head_v_dim}, expand_v {k.expand_v}), short conv {k.conv_size}, "
                  f"allow_neg_eigval {k.allow_neg_eigval}, safe_gate {k.safe_gate}, lower_bound {k.lower_bound}, mode {k.mode}",
                  flush=True)

    def forward(self, x, cos=None, sin=None, attention_mask=None):
        out = self.kda(x)
        return out[0] if isinstance(out, tuple) else out
