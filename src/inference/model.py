"""Explicit Nemotron-H forward loop for the engine (the dmlm analogue is dm_model.py).

Reuses the loaded HF module weights verbatim; the only behavioural change is that the
Mamba2 and attention mixers are rebound (`__class__` swap) to the engine subclasses
so they read the `EngineContext` instead of the HF cache. MoE/MLP mixers, the per-block
RMSNorm, embeddings, final norm and lm_head are used unchanged. No causal mask / position
ids are built (NoPE attention; FlashAttention handles causality; Mamba decode needs no mask;
Mamba prefill runs on unpadded sequences).
"""

from __future__ import annotations

import torch

from .context import EngineContext
from .layers import EngineAttention, EngineMamba2Mixer


class EngineNemotronH:
    def __init__(self, hf_model):
        self.hf = hf_model
        self.config = hf_model.config
        backbone = hf_model.model
        self.embeddings = backbone.embeddings
        self.layers = backbone.layers
        self.norm_f = backbone.norm_f
        self.lm_head = hf_model.lm_head

        # Rebind the stateful mixers onto the engine subclasses (weights shared, zero-copy).
        for blk in self.layers:
            if blk.block_type == "mamba" and not isinstance(blk.mixer, EngineMamba2Mixer):
                blk.mixer.__class__ = EngineMamba2Mixer
            elif blk.block_type == "attention" and not isinstance(blk.mixer, EngineAttention):
                blk.mixer.__class__ = EngineAttention

    @property
    def device(self) -> torch.device:
        return self.embeddings.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embeddings.weight.dtype

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, ctx: EngineContext, logits_last_only: bool = False) -> torch.Tensor:
        """input_ids: (B, T). Returns fp32 logits (B, T_out, vocab): T_out=1 if logits_last_only."""
        h = self.embeddings(input_ids)
        for blk in self.layers:
            residual = h
            hn = blk.norm(h.to(blk.norm.weight.dtype))
            if blk.block_type == "mamba":
                hn = blk.mixer(hn, ctx)
            elif blk.block_type == "attention":
                hn, _ = blk.mixer(hn, ctx)
            else:  # moe / mlp: unchanged
                hn = blk.mixer(hn)
            h = residual + hn
        h = self.norm_f(h)
        if logits_last_only:
            h = h[:, -1:, :]
        return self.lm_head(h).float()  # HF casts logits to fp32 before sampling
