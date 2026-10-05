"""Model loader for PG19 PPL eval (and the dmlm registration memmap_ppl uses).

Mirrors `evaluation/ruler/load_model.py` but adds a `max_pe_override` knob so the
bare baseline (`Qwen3-{1.7B,4B,8B}-Base`, native max_position_embeddings=32768) can be
forwarded with sequences up to 131072. Above its native max_pe the baseline produces
extrapolated RoPE: that is the OOD-collapse signal we want to measure.
"""

import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _maybe_register_pretrain_models():
    """Register the src/pretrain decoder (`model_type=dmlm`) with the HF Auto classes
    (GQA flash-attn / GDN fla / Mamba2 / KDA / MLA). Silent no-op when the `pretrain` package
    is absent; any other failure is printed, since a dmlm export then fails to load with a
    misleading "unknown model type" error."""
    try:
        from pretrain.hf import register_auto
        register_auto()
    except Exception as e:
        if isinstance(e, ModuleNotFoundError) and e.name == "pretrain":
            return
        if os.environ.get("RANK", "0") == "0":
            print(f"[load] model_type dmlm not registered: {type(e).__name__}: {e}", flush=True)


def load_model_for_ppl(model_path: str, max_pe_override: int | None = None):
    _maybe_register_pretrain_models()
    # See evaluation/ruler/load_model.py: prefer the built-in class
    # (trust_remote_code=False) so FA2/SDPA are available; the remote Nemotron-H
    # code is eager-only. Override with EVAL_TRUST_REMOTE_CODE / EVAL_ATTN_IMPL.
    trust_remote_code = _env_flag("EVAL_TRUST_REMOTE_CODE", False)
    attn_impl = os.environ.get("EVAL_ATTN_IMPL", "flash_attention_2")
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    if max_pe_override is not None and getattr(cfg, "max_position_embeddings", 0) < max_pe_override:
        cfg.max_position_embeddings = max_pe_override

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=cfg,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    )
    model.cuda().eval()
    return model, tokenizer
