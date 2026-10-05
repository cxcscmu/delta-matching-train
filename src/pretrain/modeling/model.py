"""The pretraining causal LM.

Structure (HF-style, so the FSDP2 sharder can walk `model.model.layers`):
    PretrainLM
      .model : Transformer            # backbone
        .embed_tokens : Embedding
        .layers : ModuleList[DecoderBlock]
        .norm : RMSNorm
        .rotary : RotaryEmbedding
      .lm_head : Linear               # weight tied to embed_tokens.weight

forward(input_ids, position_ids?, labels?, attention_mask?) -> {"loss"?, "logits"}.
Loss = token-shifted cross-entropy (ignore_index=-100) + z_loss_weight * E[logsumexp(logits)^2].
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import RMSNorm, RotaryEmbedding
from .block import DecoderBlock


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([DecoderBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps, cfg.rms_norm_in_fp32, fused=cfg.fused_rmsnorm)
        self.rotary = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta,
                                      rope_scaling=cfg.rope_scaling, rotary_dim=cfg.rotary_dim)

    def forward(self, input_ids, position_ids, attention_mask=None):
        h = self.embed_tokens(input_ids)
        cos, sin = self.rotary(position_ids)
        if getattr(self.cfg, "nope", False):          # NoPE: identity rotation, exact (config.py::nope)
            cos, sin = torch.ones_like(cos), torch.zeros_like(sin)
            if not getattr(self, "_nope_banner", False):
                self._nope_banner = True
                print(f"[model] NoPE active: identity rotary (cos=1, sin=0) on {cos.shape[-1]} of {self.cfg.head_dim} channels", flush=True)
        cos, sin = cos.to(h.dtype), sin.to(h.dtype)
        for layer in self.layers:
            h = layer(h, cos, sin, attention_mask)
        return self.norm(h)


class PretrainLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.config = cfg  # HF-style: the trainer reads .config
        self.model = Transformer(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)
        # scaled init for residual-output projections (GPT-2/NeoX): 1/sqrt(2*L)
        scale = 1.0 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                with torch.no_grad():
                    p.mul_(scale)
        # MLA (2026-09-18): W_UV has fan-in kv_lora_rank instead of d_model, so its init std is raised by
        # sqrt(d_model / kv_lora_rank) to give the value path the same init RMS as GQA's v_proj (a declared deviation
        # from the one initializer_range std of every other Linear); W_UK stays at 0.02 (its output is normalized, so
        # its scale is inert).
        from .mla import MLAAttention as _MLA
        for m in self.modules():
            if isinstance(m, _MLA):
                with torch.no_grad():
                    nn.init.normal_(m.v_proj.weight, mean=0.0,
                                    std=cfg.initializer_range * math.sqrt(cfg.d_model / m.kv_lora_rank))
        if cfg.tie_embeddings:  # tie AFTER init so head shares the embed tensor
            self.lm_head.weight = self.model.embed_tokens.weight
        self.fused_ce = False   # set via enable_fused_ce(); training-only loss optimization
        self._flce = None

    def enable_fused_ce(self, enabled: bool = True) -> bool:
        """Use a fused linear cross-entropy for the TRAINING loss: compute CE (+z-loss)
        directly from hidden states + the tied head weight WITHOUT materializing the
        [B,T,V] logits (the memory wall, docs/train_engine.md §14.3). Liger's `lse_square_scale`
        reproduces our z-loss `zw·E[logsumexp²]` exactly. Returns whether it was enabled
        (falls back to the unfused path, with a warning, if liger_kernel is unavailable)."""
        if not enabled:
            self.fused_ce = False
            return False
        try:
            from liger_kernel.transformers.fused_linear_cross_entropy import (
                LigerFusedLinearCrossEntropyLoss,
            )
        except Exception as e:  # keep training runnable without the optional dep
            import warnings
            warnings.warn(f"fused_ce requested but liger_kernel unavailable ({e}); using unfused CE.")
            self.fused_ce = False
            return False
        self._flce = LigerFusedLinearCrossEntropyLoss(
            lse_square_scale=float(self.cfg.z_loss_weight or 0.0),
            ignore_index=-100, reduction="mean")
        self.fused_ce = True
        return True

    def _init_weights(self, m: nn.Module):
        std = self.cfg.initializer_range
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)

    def forward(self, input_ids, position_ids=None, labels=None, attention_mask=None):
        B, T = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        hidden = self.model(input_ids, position_ids, attention_mask)
        if labels is not None and self.fused_ce:
            # fused linear-CE: no [B,T,V] logits materialized (training path only)
            return {"loss": self._fused_loss(hidden, labels), "logits": None}
        logits = self.lm_head(hidden)
        out = {"logits": logits}
        if labels is not None:
            out["loss"] = self._loss(logits, labels)
        return out

    def _fused_loss(self, hidden, labels):
        # shift (position t predicts t+1), flatten to [N, d] / [N], same as _loss
        hidden = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
        labels = labels[:, 1:].reshape(-1)
        # match the head weight to the hidden dtype so the kernel sees one dtype
        # (DDP keeps fp32 master weights; autocast would otherwise bf16-cast the matmul)
        weight = self.lm_head.weight.to(hidden.dtype)
        # liger returns a bare loss tensor by default (z-loss already folded in via
        # lse_square_scale); a CrossEntropyOutput only when extra-return flags are set
        out = self._flce(weight, hidden, labels)
        return out.loss if hasattr(out, "loss") else out

    def _loss(self, logits, labels):
        # shift: position t predicts token t+1
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        V = logits.size(-1)
        logits_f = logits.float()  # CE/z-loss in fp32 for numerical stability under bf16 autocast
        ce = F.cross_entropy(logits_f.view(-1, V), labels.view(-1), ignore_index=-100)
        zw = self.cfg.z_loss_weight
        if zw and zw > 0:
            lse = torch.logsumexp(logits_f, dim=-1)            # [B, T-1]
            mask = labels != -100
            z = (lse[mask] ** 2).mean() if mask.any() else lse.sum() * 0.0
            return ce + zw * z
        return ce

    # -- config (de)serialization, used by the trainer's HF-style save --
    def save_config(self, save_dir: str | os.PathLike):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "model_config.json"), "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)

    @classmethod
    def from_config_file(cls, path: str | os.PathLike) -> "PretrainLM":
        with open(path) as f:
            d = json.load(f)
        d.pop("_resolved", None)
        return cls(ModelConfig(**d))


def build_model(cfg: ModelConfig) -> PretrainLM:
    return PretrainLM(cfg)
