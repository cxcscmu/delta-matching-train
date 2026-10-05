"""Export a trained `src/pretrain` model to a HuggingFace checkpoint directory
(config.json + model.safetensors [+ tokenizer]) loadable by AutoModelForCausalLM:
the format the evaluation engine (src/evaluation) consumes.
"""
from __future__ import annotations

import os

import torch

from ..modeling.config import ModelConfig
from .configuration_dmlm import DMLMConfig


def save_hf_checkpoint(state_dict: dict, model_cfg: ModelConfig, out_dir: str,
                       tokenizer=None, dtype: torch.dtype = torch.bfloat16) -> str:
    """Write a HF checkpoint. `state_dict` is the consolidated (unsharded) model state
    dict with `src/pretrain` naming (model.* / lm_head.weight); names match
    `DMLMForCausalLM`. Tied `lm_head.weight` is dropped (re-tied on load) so
    safetensors doesn't choke on shared storage."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    cfg = DMLMConfig.from_model_config(model_cfg)
    cfg.save_pretrained(out_dir)  # config.json with model_type=dmlm + tie_word_embeddings

    sd = {k: v for k, v in state_dict.items()}
    if model_cfg.tie_embeddings:
        sd.pop("lm_head.weight", None)  # tied -> re-created on load (_tied_weights_keys)
    # clone+contiguous in the target dtype; clone severs any shared storage safetensors rejects
    sd = {k: v.detach().to(dtype).contiguous().clone() for k, v in sd.items()}
    save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})

    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    return out_dir
