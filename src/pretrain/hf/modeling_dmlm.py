"""HF `PreTrainedModel` wrapper for the delta-matching-train decoder.

Reuses `src/pretrain`'s `Transformer` backbone + tied head verbatim, so parameter names match the training checkpoint
(`model.embed_tokens.*`, `model.layers.N.*`, `model.norm.*`, `lm_head.weight`) and the consolidated `model.safetensors`
(hf/export.py) loads directly. The mixers carry their own backends: GQA -> flash-attn/SDPA, GDN / KDA -> fla, Mamba2 ->
mamba_ssm. `.forward` returns logits (used by the evaluation engine's PPL / loglikelihood paths; `logits_to_keep`
restricts them to the last positions, see forward); `.generate` works via HF
GenerationMixin without a KV cache (recompute; fine for short eval generations, the continuous-batching engine in
src/inference is the cached path).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..modeling.model import Transformer
from .configuration_dmlm import DMLMConfig

_ATTN_MAP = {"flash_attention_2": "flash", "sdpa": "sdpa", "eager": "sdpa"}


class DMLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = DMLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False   # no layer implements it: gradient_checkpointing_enable() refuses clearly
    _supports_flash_attn = True          # transformers 5.x name
    _supports_flash_attn_2 = True        # back-compat
    _supports_sdpa = True
    _supports_attention_backend = True
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: DMLMConfig):
        super().__init__(config)
        mc = config.to_model_config()
        # respect the eval's attn_implementation choice (flash_attention_2/sdpa/eager)
        hf_impl = getattr(config, "_attn_implementation", None)
        if hf_impl in _ATTN_MAP:
            mc.attn_impl = _ATTN_MAP[hf_impl]
        self._mc = mc
        self.model = Transformer(mc)
        self.lm_head = nn.Linear(mc.d_model, mc.vocab_size, bias=False)
        if mc.tie_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    # --- HF embedding plumbing (needed by tie_weights / resize / generate) ---
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, labels=None,
                past_key_values=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, **kwargs):
        B, T = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        hidden = self.model(input_ids, position_ids, attention_mask)
        # `logits_to_keep` (the transformers convention): an int k > 0 computes the logits of the last k positions only
        # (the PG19 growing-window scorer passes stride + 1, which also keeps the [B, T, V] transient small at long
        # context); 0 or None computes all of them; a 1-D tensor keeps those sequence indices. Read from **kwargs, not
        # named in the signature, so transformers' generate() calls this cache-free wrapper exactly as before.
        keep = kwargs.get("logits_to_keep")
        if isinstance(keep, torch.Tensor):
            sel = keep
        elif keep:
            if int(keep) < 0:
                raise ValueError(f"logits_to_keep must be >= 0, got {keep}")
            sel = slice(-int(keep), None)
        else:
            sel = None
        if sel is not None and labels is not None:
            raise ValueError("labels need the logits of every position: omit logits_to_keep (or pass 0) with labels")
        logits = self.lm_head(hidden if sel is None else hidden[:, sel, :])
        loss = None
        if labels is not None:
            sl = logits[:, :-1, :].contiguous().float()
            lb = labels[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.size(-1)), lb.view(-1), ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    # cache-free generation (recompute each step). Enough for short eval gens; the
    # continuous-batching engine (src/inference) is the cached path.
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def register_auto() -> None:
    """Register DMLM with the HF Auto classes so AutoModelForCausalLM.from_pretrained
    loads a `model_type: dmlm` checkpoint without trust_remote_code. Idempotent."""
    from transformers import AutoConfig, AutoModelForCausalLM
    try:
        AutoConfig.register("dmlm", DMLMConfig)
        AutoModelForCausalLM.register(DMLMConfig, DMLMForCausalLM)
    except (ValueError, KeyError):
        pass  # already registered
