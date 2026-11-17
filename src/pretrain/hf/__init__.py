"""The HF side of src/pretrain: DMLMConfig / DMLMForCausalLM (AutoModelForCausalLM-loadable, register_auto), the export
(save_hf_checkpoint) and the whole-export check (check_export, `python -m pretrain.hf.check_export <dir>`).

The names are imported lazily (PEP 562), so that `python -m pretrain.hf.check_export`, which the launchers run after
every training, imports neither torch nor transformers. `from pretrain.hf import register_auto` etc. work as before."""
import importlib

__all__ = ["DMLMConfig", "DMLMForCausalLM", "register_auto", "save_hf_checkpoint"]
_LAZY = {"DMLMConfig": ".configuration_dmlm", "DMLMForCausalLM": ".modeling_dmlm", "register_auto": ".modeling_dmlm",
         "save_hf_checkpoint": ".export"}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(mod, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
