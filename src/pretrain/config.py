"""Configuration + a dependency-free YAML/CLI loader (draccus is not installed).

Nested dataclasses (TrainConfig owns ModelConfig / OptimConfig / DataConfig), a
__post_init__ that derives gradient_accumulation_steps from the global batch, and
`load_config(argv)` that merges: dataclass defaults < `--config a.yaml`
< `--section.key=value` CLI overrides (values parsed with yaml.safe_load for clean
type inference). Resolves exactly like draccus for our needs without the dependency.
A key that is not a dataclass field (a typo in a YAML key or a CLI flag) raises
ValueError naming it, instead of silently training the default.
"""
from __future__ import annotations

import difflib
import os
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path
from typing import Any, Literal

import yaml

from .modeling.config import ModelConfig


@dataclass
class OptimConfig:
    lr: float = 3e-3
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    lr_scheduler_type: Literal["wsd", "constant_with_warmup"] = "wsd"
    warmup_ratio: float = 0.02
    decay_ratio: float = 0.2                      # WSD: 20% decay tail
    decay_type: Literal["cosine", "linear"] = "cosine"
    no_wd_regex: tuple[str, ...] = (r"norm", r"embed_tokens", r"\.bias$")
    name: Literal["adamw", "muon"] = "adamw"     # 2026-09-26: "muon" = Muon on the 2-D hidden weights + AdamW on the rest
                                                 # (embeddings, norms / gains, conv, SSM scalars); parallel/muon.py. lr,
                                                 # weight_decay and the schedule are shared (Moonlight's 0.2*sqrt(max(m,n)) scale)
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr: str = "match_rms_adamw"      # "original" = Keller Jordan's sqrt(max(1, m/n)) (then lr needs retuning)
    muon_exclude_regex: tuple[str, ...] = (r"embed_tokens", r"lm_head")


@dataclass
class DataConfig:
    paths: Any = "data/tokens"     # dir or list of .bin/.npy token files
    dtype: Literal["uint16", "uint32"] = "uint16"
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    seed: int = 42
    pad_token_id: int | None = None  # None => packed, no padding mask
    doc_lengths: str | None = None   # path to int64 doc-length sidecar -> document-aware loader (ProLong)
    position_mode: str = "packed"    # "packed" (continuous positions) | "reset" (per-document)
    chunk_subset: Any = None         # [start_frac,end_frac] -> sample a disjoint slice of the seed-fixed
                                     # chunk permutation (e.g. separate slices for successive training stages)
    prepend_bos: bool = False        # BOS-anchored windows (2026-09-24): every chunk becomes [bos_token_id] + chunk[:-1],
                                     # so the windows, their order and their length are unchanged and BOS predicts the
                                     # chunk's first token (the chunk's last token is dropped). Packed positions only.
    bos_token_id: int = 1            # Llama-2 BOS

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    run_name: str | None = None
    backend: Literal["ddp", "fsdp2"] = "fsdp2"
    seed: int = 42
    VERSION_STRING: str = "pt-v0.1"

    # I/O
    ckpt_root_dir: str = "runs/ckpt"
    log_root_dir: str = "runs/logs"

    # batch / steps
    num_devices: int = 1                 # overwritten with the real world size in train.py
    per_device_batch_size: int = 8
    global_batch_size: int = 64
    num_steps: int = 1000
    # Stop the loop early WITHOUT shortening the LR schedule. num_steps alone drives both the
    # wsd horizon and the stop point, so a phased run that passes a shorter num_steps to phase A
    # silently gives that phase its OWN schedule: it decays to lr=0 by the phase-A end and then
    # RESTARTS at peak in phase B, which confounds the result. Set stop_at_step to the phase
    # boundary and keep num_steps at the FULL horizon in every phase so the schedule stays
    # continuous across the resume. 0 = disabled (stop at num_steps, the original behaviour).
    stop_at_step: int = 0

    # numerics
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = False
    ac_mode: str = "full"     # activation-checkpoint granularity: "full" (recompute block) | "selective" (save matmul/attn, recompute cheap ops)
    compile: bool = False     # torch.compile each decoder block (regional/per-block compilation)
    compile_mode: str = "default"   # "default" | "max-autotune-no-cudagraphs" | "max-autotune" | ...
    fused_ce: bool = False    # fused linear cross-entropy (liger): skip the [B,T,V] logits in training
    fp8: bool = False         # FP8 training (torchao float8) on the block Linears: past the bf16 MFU ceiling
    fp8_recipe: str = "tensorwise"  # "tensorwise" (fastest) | "rowwise" (more accurate)
    # WHICH Linears fp8 converts (2026-08-06). "layers" = every nn.Linear under model.layers.*
    # (attention q/k/v/o AND GDN in/out AND the MLP) -- the original behaviour. "ffn" = the
    # SwiGLU MLP only (gate/up/down_proj). The distinction matters for the qwen35 campaign:
    # the arms that quantize ATTENTION do so via the dm mixer's fp8 kernel, not torchao, so a
    # "clean attention + fp8 FFN" arm must NOT also convert the attention projections or the
    # two mechanisms are confounded in the same run.
    fp8_scope: str = "layers"   # layers | ffn
    # torch.compile ONLY the MLP submodules. Required to make fp8 FFN a win at all: eager fp8 is
    # 3x SLOWER than bf16, compiled fp8 is 1.30x faster (measured, 2048->6144, 16384 tokens).
    # Whole-model compile is unusable because the fla GDN kernels graph-break.
    compile_mlp: bool = False

    # checkpoint / resume
    init_from_hf: str | None = None  # warm-start: load weights from this HF export (a finished run) into a FRESH
                                     # continued-pretrain (fresh optimizer + schedule, step 0). Overridden by an
                                     # existing resume ckpt in ckpt_root_dir. Used for the context-extension runs.
    resume: bool = True
    save_interval: int = 200
    save_total_limit: int | None = 3
    logging_steps: int = 10
    force_resume_check: bool = True
    hf_save_dir: str | None = None   # if set, also export a HF checkpoint here at end of training (for eval)
    tokenizer_dir: str | None = None  # if set, bake this tokenizer into the HF export so meta_eval (recall/commonsense/longbench) can load it
    hf_export_steps: list[int] = field(default_factory=list)  # extra step indices at which to export a {hf_save_dir}/step{N} HF ckpt (the M-tier token-budget ladder)
    result_root_dir: str | None = None  # curated per-run result dump root (default: <dirname(log_root_dir)>/result)

    # wandb (optional)
    wandb: bool = False
    wandb_entity: str | None = None    # None = the wandb account's default entity
    wandb_project: str = "delta-matching"
    wandb_job_type: str | None = None   # free-form wandb grouping label (the campaign launcher sets one per study round)

    # nested
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        # env world size is correct for torchrun (set before python starts); for a
        # bare process it defaults to 1 and train.py re-derives after dist_init.
        self.derive(int(os.environ.get("WORLD_SIZE", self.num_devices)))

    def derive(self, world_size: int) -> "TrainConfig":
        self.num_devices = int(world_size)
        single_iter = self.per_device_batch_size * self.num_devices
        assert self.global_batch_size % single_iter == 0, (
            f"global_batch_size ({self.global_batch_size}) must be divisible by "
            f"per_device_batch_size*num_devices ({single_iter})")
        self.gradient_accumulation_steps = self.global_batch_size // single_iter
        self.total_size = self.global_batch_size * self.num_steps
        self.chunk_size = self.model.max_seq_len  # data chunk length == model context
        if self.run_name is None:
            self.run_name = (f"{self.backend}_{self.model.mixer}_L{self.model.n_layers}"
                             f"_d{self.model.d_model}_gbs{self.global_batch_size}_s{self.seed}")
        self.ckpt_dir = Path(self.ckpt_root_dir) / self.run_name
        self.log_dir = Path(self.log_root_dir) / self.run_name
        self.log_path = self.log_dir / "train.log"
        self.data_cfg = self.data.as_dict()
        self.opt_cfg = self.optim  # base_trainer reads cfg.opt_cfg.*
        return self


# ------------------------------------------------------------------ CLI / YAML
def _unknown_keys(cls, d: dict, path: str = "") -> list[str]:
    """The keys of `d` that are not fields of dataclass `cls`, dotted (`model.dm_mdoe`) and recursing into
    nested dataclass sections, each with the closest field name as a hint."""
    by_name = {f.name: f for f in fields(cls)}
    bad = []
    for k, val in d.items():
        f = by_name.get(k)
        if f is None:
            close = difflib.get_close_matches(str(k), list(by_name), n=1)
            bad.append(f"{path}{k}" + (f" (did you mean {path}{close[0]}?)" if close else ""))
            continue
        default = f.default_factory() if f.default_factory is not None and callable(f.default_factory) else None
        if is_dataclass(default) and isinstance(val, dict):
            bad += _unknown_keys(type(default), val, f"{path}{k}.")
    return bad


def _coerce(cls, d: dict, _checked: bool = False):
    """Construct dataclass `cls` from dict `d`, recursing into nested dataclass fields.
    Keys that are not fields raise ValueError (all of them at once): a typo such as
    `--model.dm_mdoe stale` must fail loudly, not train the default arm."""
    if not _checked:
        bad = _unknown_keys(cls, d)
        if bad:
            raise ValueError(f"unknown config key(s): {', '.join(bad)}. Every YAML key and --section.key override "
                             f"must name a field of {cls.__name__} or of its model / optim / data sections.")
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]
        # nested dataclass?
        default = f.default_factory() if f.default_factory is not None and callable(f.default_factory) else None
        if is_dataclass(default) and isinstance(val, dict):
            kwargs[f.name] = _coerce(type(default), val, _checked=True)
        else:
            kwargs[f.name] = val
    return cls(**kwargs)


def _coerce_scalar(s: Any) -> Any:
    """Parse a CLI/YAML scalar string into the right Python type.

    Fixes PyYAML's quirk where "3e-3"/"1e-4" load as *strings* (YAML 1.1 floats need
    a dot). Ladder: bool/null -> int -> float (handles sci-notation) -> yaml (lists,
    plain strings). Non-str values pass through unchanged."""
    if not isinstance(s, str):
        return s
    low = s.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)  # handles "3e-3", "1e-4", ".5"
    except ValueError:
        pass
    return yaml.safe_load(s)  # lists/dicts/strings ("sdpa", "[gqa,gqa]")


def _normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _normalize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize(v) for v in obj]
    return _coerce_scalar(obj) if isinstance(obj, str) else obj


def _set_dotted(tree: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = tree
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(argv: list[str] | None = None) -> TrainConfig:
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)

    base: dict = {}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == "--config":
            with open(argv[i + 1]) as fh:
                base = _normalize(yaml.safe_load(fh) or {})
            i += 2
            continue
        if tok.startswith("--"):
            body = tok[2:]
            if "=" in body:
                k, v = body.split("=", 1)
            else:
                k, v = body, argv[i + 1]
                i += 1
            _set_dotted(base, k, _coerce_scalar(v))
        i += 1

    return _coerce(TrainConfig, base)
