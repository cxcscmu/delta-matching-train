import os
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from modeling import HFModel
from utils import master_print


# dmlm mixers with a recurrent state over every input token. They ignore `attention_mask`, so on the plain HF
# generation path the left padding of a batch runs through their state ahead of the prompt.
_RECURRENT_DMLM_MIXERS = frozenset({"Mamba2Mixer", "GatedDeltaNetMixer", "KDAMixer"})


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _maybe_register_pretrain_models():
    """Register the src/pretrain decoder (`model_type=dmlm`) with the HF Auto classes so a
    pretrain checkpoint loads natively (flash-attn GQA / fla Gated-DeltaNet and KDA / mamba_ssm
    Mamba2 / MLA); requires `PYTHONPATH` to include this repository's src/. Silent no-op when
    the `pretrain` package is absent (e.g. Nemotron-H runs); any other failure is printed,
    since a dmlm export then fails to load with a misleading "unknown model type" error."""
    try:
        from pretrain.hf import register_auto
        register_auto()
    except ModuleNotFoundError as e:
        if e.name != "pretrain":
            master_print(f"[load_model] model_type dmlm not registered: {type(e).__name__}: {e}")
    except Exception as e:
        master_print(f"[load_model] model_type dmlm not registered: {type(e).__name__}: {e}")


def padded_generation_error(model, batch_size: int) -> Optional[str]:
    """Why batched generation through `model` would be wrong, or None when it is fine.

    `modeling.HFModel.generate` (the plain HF path) left-pads a batch to its longest prompt and
    hides the padding with `attention_mask`. The recurrent dmlm mixers (Mamba2, Gated DeltaNet,
    KDA) ignore that mask, so with batch_size > 1 their generations would depend on the other
    prompts of the batch. The continuous-batching engine (EVAL_ENGINE=mini_eval_dev) keeps one
    state per sequence and is not affected; neither are models without recurrent mixers.
    """
    if batch_size <= 1 or not isinstance(model, HFModel):
        return None
    hf_model = model.model
    if getattr(getattr(hf_model, "config", None), "model_type", None) != "dmlm":
        return None
    layers = getattr(getattr(hf_model, "model", None), "layers", None) or []
    kinds = sorted({type(getattr(blk, "mixer", None)).__name__ for blk in layers} & _RECURRENT_DMLM_MIXERS)
    if not kinds:
        return None
    return (f"batch size {batch_size} on the plain HF generation path is not supported for this dmlm model: its "
            f"{' / '.join(kinds)} layers ignore the padding mask, so the left padding of a batch would change the "
            f"generations. Use the continuous-batching engine (EVAL_ENGINE=mini_eval_dev) or --eval_batch_size 1.")


def load_model_for_eval(model_name_or_path, device_map=None):
    _maybe_register_pretrain_models()
    # transformers >= 5.3 ships Qwen3 / Qwen3.5 / Nemotron-H natively. Prefer the
    # built-in class (trust_remote_code=False) so memory-efficient attention
    # (flash_attention_2 / sdpa) is available: the HF repos' remote Nemotron-H code
    # is eager-only (its model class never sets _supports_flash_attn_2 / _supports_sdpa,
    # so the dispatch refuses anything but eager). Override per run with
    # EVAL_TRUST_REMOTE_CODE=1 and/or EVAL_ATTN_IMPL=sdpa|eager for other models.
    #
    # device_map (e.g. "auto"): shard a model too large for one GPU across this
    # process's visible GPUs via HF/accelerate pipeline parallelism. NemotronH has
    # no tensor-parallel plan in transformers 5.3 (Mamba2/MoE unshardable), so
    # device_map PP is the supported way to fit its 262k context. When device_map is
    # set, from_pretrained streams shards straight onto the GPUs and we must NOT call
    # .cuda() (which would collapse every shard back onto one device -> OOM).
    trust_remote_code = _env_flag("EVAL_TRUST_REMOTE_CODE", False)
    attn_impl = os.environ.get("EVAL_ATTN_IMPL", "flash_attention_2")
    # MoE experts kernel. transformers 5.3 defaults Nemotron-H's MoE to the optimized
    # "grouped_mm" path (torch._grouped_mm), which hits a CUDA illegal-memory-access at
    # very long sequences (S=262144). EVAL_EXPERTS_IMPL=eager forces the model's own
    # reference looped expert forward: correct and fast enough at our token counts.
    # Valid: eager | grouped_mm | batched_mm. Unset -> HF default (only valid for MoE).
    experts_impl = os.environ.get("EVAL_EXPERTS_IMPL") or None
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    master_print(f"[LOADING MODEL] {model_name_or_path} "
                 f"(trust_remote_code={trust_remote_code}, attn_implementation={attn_impl}, "
                 f"experts_implementation={experts_impl}, device_map={device_map})")
    from_kwargs = dict(
        config=cfg,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_impl,
        dtype=torch.bfloat16,
    )
    if device_map:
        from_kwargs["device_map"] = device_map
    if experts_impl:
        from_kwargs["experts_implementation"] = experts_impl
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **from_kwargs)
    if not device_map:
        model.cuda()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
