"""Mamba2 mixer (wraps `mamba_ssm.Mamba2`).

A selective state-space token mixer: O(L) time, fixed recurrent state, the second recurrent baseline of the hybrid
comparison. No RoPE (SSM positional dynamics are intrinsic), so `cos`/`sin` are accepted and ignored. `mamba_ssm` is
imported lazily. 2026-09-18: the geometry (headdim, d_state, expand, ngroups, chunk_size) comes from ModelConfig so the
1.665 B Mamba2:GQA hybrid can be parameter-matched; the module runs mamba_ssm's DEDICATED fused kernels
(`use_mem_eff_path=True`: mamba_split_conv1d_scan_combined over causal_conv1d + the SSD Triton scan), asserted at build.
"""
from __future__ import annotations

import torch.nn as nn

from .config import ModelConfig


class Mamba2Mixer(nn.Module):
    _announced = False

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        from mamba_ssm import Mamba2  # lazy
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
        # the dedicated kernel path: mamba_ssm's fused conv + scan (needs causal_conv1d); refuse the slow fallback
        assert getattr(self.mamba, "use_mem_eff_path", False), "Mamba2: use_mem_eff_path is off (the fused kernel path)"
        try:
            import causal_conv1d  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"Mamba2 needs causal_conv1d for its fused kernel path: {e}")
        if not Mamba2Mixer._announced:
            Mamba2Mixer._announced = True
            m = self.mamba
            print(f"[mamba2] mamba_ssm.Mamba2: d_inner {m.d_inner} = expand {getattr(m, 'expand', '?')} x d_model, "
                  f"headdim {m.headdim}, nheads {m.nheads}, d_state {m.d_state}, ngroups {m.ngroups}, chunk {m.chunk_size}, "
                  f"fused kernel path (use_mem_eff_path={m.use_mem_eff_path})", flush=True)

    def forward(self, x, cos=None, sin=None, attention_mask=None):
        return self.mamba(x)
