"""Explicit dmlm forward loop for the engine (analogue of model.py::EngineNemotronH).

Reuses the loaded HF dmlm weights verbatim; the only behavioural change is rebinding each
block's mixer (`__class__` swap) to its engine subclass so it reads the `EngineContext`
(per-slot state pool) instead of recomputing. Embeddings / per-block RMSNorms / SwiGLU MLP /
final norm / lm_head are used unchanged. RoPE cos/sin are computed per forward at the right
positions (prefill 0..L-1; decode = each running slot's current length) and threaded via the
context for the attention mixers (Mamba2/GDN/KDA ignore them). A NoPE model (`nope` in its
config) gets the identity rotation (cos = 1, sin = 0), exactly as the training forward
(`pretrain.modeling.model.Transformer.forward`) builds it.
"""
from __future__ import annotations

import torch

from .dm_cache import mixer_kind
from .dm_layers import EngineGQA, EngineMamba2, EngineGDN, EngineMLA, EngineKDA
from .context import PHASE_DECODE

_ENGINE_CLS = {"gqa": EngineGQA, "mamba2": EngineMamba2, "gdn": EngineGDN, "mla": EngineMLA, "kda": EngineKDA}
_ROPE_KINDS = ("gqa", "mla")     # attention mixers that need RoPE cos/sin (recurrent ones don't)


class EngineDMLM:
    def __init__(self, hf_model):
        self.hf = hf_model
        self.config = hf_model.config
        backbone = hf_model.model              # pretrain.modeling.model.Transformer
        self.embed_tokens = backbone.embed_tokens
        self.layers = backbone.layers
        self.norm = backbone.norm
        self.rotary = backbone.rotary
        # NoPE: read from the backbone's ModelConfig, the flag the training forward reads
        self._nope = bool(getattr(getattr(backbone, "cfg", None), "nope", False))
        self.lm_head = hf_model.lm_head

        self._needs_rope = False
        for i, blk in enumerate(self.layers):
            kind = mixer_kind(blk.mixer)
            if kind in _ROPE_KINDS:
                self._needs_rope = True
            cls = _ENGINE_CLS[kind]
            if not isinstance(blk.mixer, cls):
                blk.mixer.__class__ = cls      # weights/attrs shared, zero-copy
            blk.mixer._eidx = i                # block index -> pool key (Mamba2/GDN don't store layer_idx)

    @property
    def device(self) -> torch.device:
        return self.embed_tokens.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.embed_tokens.weight.dtype

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, ctx, logits_last_only: bool = False) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        if self._needs_rope:
            if ctx.phase == PHASE_DECODE:
                pos = ctx.pool.seqlens.index_select(0, ctx.slot_ids).long()[:, None]   # (N,1) per-slot position
            else:
                pos = torch.arange(input_ids.shape[1], device=h.device)[None]          # (1,T) prefill positions
            cos, sin = self.rotary(pos)
            if self._nope:      # identity rotation (cos = 1, sin = 0), as the training forward applies it
                cos, sin = torch.ones_like(cos), torch.zeros_like(sin)
            ctx.cos, ctx.sin = cos.to(h.dtype), sin.to(h.dtype)
        for blk in self.layers:
            h = h + blk.mixer(blk.input_layernorm(h), ctx)
            h = h + blk.mlp(blk.post_attention_layernorm(h))
        h = self.norm(h)
        if logits_last_only:
            h = h[:, -1:, :]
        return self.lm_head(h).float()
