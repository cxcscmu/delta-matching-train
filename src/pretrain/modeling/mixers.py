"""Token-mixer registry: the extension point for adding a mixer. Create its module exposing
`nn.Module(cfg, layer_idx)` with the same `forward(x, cos, sin, attention_mask)` signature and call `register_mixer`.
"""
from __future__ import annotations

import torch.nn as nn

from .config import ModelConfig
from .attention import GQAAttention
from .gated_deltanet import GatedDeltaNetMixer
from .mamba2 import Mamba2Mixer
from .kda import KDAMixer
from .dm_attention import DMAttention
from .mla import MLAAttention, MLADMAttention

# GDN / Mamba2 / KDA wrappers import their heavy backends (fla / mamba_ssm) lazily inside __init__, and the dm mixers
# import the delta_matching kernel package lazily, so registering the classes here is cheap and a GQA-only run needs
# none of them.
MIXER_REGISTRY: dict[str, type[nn.Module]] = {
    "gqa": GQAAttention,
    "gated_deltanet": GatedDeltaNetMixer,
    "mamba2": Mamba2Mixer,
    "kda": KDAMixer,                # Kimi Delta Attention, fla.layers.kda
    "dm": DMAttention,              # GQA with the Delta-Matching fp8 attention core + fused fp8 producers (dm_mode)
    "mla": MLAAttention,            # Multi-head Latent Attention, bf16 core
    "mla_dm": MLADMAttention,       # MLA on the Delta-Matching fp8 core (delta from dm_mode)
}


def register_mixer(name: str, cls: type[nn.Module]) -> None:
    MIXER_REGISTRY[name] = cls


def build_mixer(name: str, cfg: ModelConfig, layer_idx: int) -> nn.Module:
    if name not in MIXER_REGISTRY:
        raise ValueError(f"Unknown mixer '{name}'. Registered: {sorted(MIXER_REGISTRY)}.")
    return MIXER_REGISTRY[name](cfg, layer_idx)
