"""inference: a continuous-batching generation engine for this repository's dmlm models and for Nemotron-H.

`EngineModel` is a drop-in replacement for `modeling.wrapper.HFModel.generate`. It reuses the loaded HF model's weight
modules and replaces only the cache interaction: each stateful mixer is rebound to an engine subclass that keeps its
state in a pool pre-allocated per slot (one slot per concurrent sequence; per layer, a contiguous buffer indexed by slot
id: K and V of shape (slots, max_len, kv_heads, head_dim) for attention, the O(1) recurrent state for Mamba2,
GatedDeltaNet and KDA). A finished sequence frees its slot for the next prompt, so the decode batch stays full
(continuous batching).

Two model paths, picked by `config.model_type`:
  * dmlm (this repository's GQA / MLA / Mamba2 / GatedDeltaNet / KDA hybrids): dm_model.py, dm_layers.py, dm_cache.py;
  * Nemotron-H (transformers' NemotronH: Mamba2, MLP / MoE and NoPE attention layers): model.py, layers.py, cache.py.
The Nemotron-H path imports transformers' nemotron_h module, mamba_ssm and causal_conv1d when a Nemotron-H model is
wrapped; the dmlm path imports mamba_ssm and causal_conv1d only for Mamba2 layers and fla only for GatedDeltaNet / KDA
layers.
"""

__all__ = ["EngineModel"]


def __getattr__(name):  # lazy so partial-build imports (cache/layers/model) work standalone
    if name == "EngineModel":
        from .engine import EngineModel

        return EngineModel
    raise AttributeError(name)
