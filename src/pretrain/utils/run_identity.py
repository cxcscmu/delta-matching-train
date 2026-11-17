"""The identity of a training run: <ckpt_dir>/run_identity.json.

A run that resumes is a continuation only when it is the same experiment. Every 4-hour job of a chain rebuilds the
configuration from the launcher's knobs, the yaml and the environment, and the resume restores weights, optimizer and
scheduler state but rebuilds the LR schedule from the current num_steps / warmup / decay. A resubmission with another
NUM_STEPS, DM_LR, EXTRA_ARGS or S2_* knob, or an edited yaml, would therefore silently continue a different experiment
under the same run name. So the master records the run's identity at its first start (or at the first resume of a run
that has none: one started before this file existed) and checks it at every resume:

  identity  the values that define the trajectory. A difference refuses the resume, listing every differing key with
            its recorded and current value, unless DM_ALLOW_RUN_CHANGE=1 (then a warning, and the new values are
            recorded, with the change, so the following jobs continue without the knob).
  env       the software it runs on: torch / triton / fla / torchao / transformer_engine versions (from the loaded
            module, else the installed distribution; None when absent) and DM_GIT_REV (the launcher's code revision).
            A difference is a warning (and recorded): the chains run the live tree and the shared environment, and an
            upgrade mid-run is worth knowing about, but not worth stopping a chain for.

What is in the identity (run_identity; choose with care: a false refusal stops a chain, a missing key lets a changed
experiment through):
  * the schedule and its horizon: num_steps; every optimizer field (optim.*: name, lr, betas, eps, weight decay and its
    exclusions, max_grad_norm, the scheduler type, warmup / decay ratios and decay type, the Muon settings);
  * the batch and the sequence: global_batch_size, model.max_seq_len (= the data chunk);
  * the data: data.paths, data.dtype, data.doc_lengths, data.seed, data.chunk_subset, data.position_mode,
    data.pad_token_id, data.prepend_bos, data.bos_token_id. The two paths (PATH_KEYS) are recorded as the file system
    resolves them (os.path.realpath: symlinks, "..", a relative path against the working directory) and compared that
    way, on the recorded side too (records written before kept the spelling). So `$DM_REPO/workspace/...` and the
    absolute dm-workspace path are the same data, while a symlink retargeted to other files is a change;
  * the model: every ModelConfig field as resolved (attn_impl after "auto" was resolved, the derived head_dim and
    intermediate_size, the fused kernels, dm_mode, ...);
  * the precision and the kernels that set the round-off: bf16, fp16, fp8, fp8_recipe, fp8_scope, fused_ce, compile,
    compile_mode, compile_mlp; backend (DDP keeps fp32 parameters, FSDP2 bf16 ones);
  * seed (model init);
  * the arm's numerics environment that scripts/campaign/train_arm.sh and arms.sh manage (NUMERICS_ENV): an unset
    variable reads as its code default, so "unset" and "set to the default" are the same configuration.
NOT in the identity (they may differ between the jobs of one run): stop_at_step (phases share the schedule),
per_device_batch_size and the world size (the gradient accumulation follows; arms.sh picks the micro-batch from the GPU
memory), save_interval / save_total_limit / logging_steps / hf_export_steps / hf_save_dir / tokenizer_dir (output
settings), init_from_hf (stage 2 passes it only before the first checkpoint), resume / force_resume_check,
gradient_checkpointing / ac_mode (memory knobs: the recompute is the same computation), the data loader's workers /
pinning / prefetch, the wandb settings, run_name and every path that only says where things are written,
VERSION_STRING (a checkpoint of another format version is not resumed at all).
Keys only one side has (a field the code added or dropped since the record) are a warning, never a refusal.

When the record is written: at a first start at once; at a resume only once the checkpoint has loaded on every rank
(plan_run_identity decides, the trainer calls record_run_identity after the load). A resume whose load fails (an
architecture change DCP rejects, a broken checkpoint) therefore leaves the record as it was: reverting the change needs
no knob, and a run without a record is not given the values of a configuration that could not even load.
DM_ALLOW_RUN_CHANGE=1 while nothing differs is a warning: a chain hands its environment to every later job, so a knob
set for one resubmission stays set, and while it is set a later accidental change is accepted with a warning only.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .ckpt import read_json_file, write_json_atomic

IDENTITY_FILE = "run_identity.json"
ALLOW_ENV = "DM_ALLOW_RUN_CHANGE"

TRAIN_KEYS = ("backend", "seed", "num_steps", "global_batch_size", "bf16", "fp16", "fp8", "fp8_recipe", "fp8_scope",
              "fused_ce", "compile", "compile_mode", "compile_mlp")
DATA_KEYS = ("paths", "dtype", "doc_lengths", "seed", "chunk_subset", "position_mode", "pad_token_id", "prepend_bos",
             "bos_token_id")
PATH_KEYS = ("data.paths", "data.doc_lengths")   # recorded and compared as resolved paths (module docstring)
MODEL_SKIP = ("_resolved",)
# The numerics switches of the arms (train_arm.sh exports or unsets each; arms.sh sets the arm's own), with the default
# the code applies when the variable is unset.
NUMERICS_ENV = {
    "DM_FP8_FSDP_AG": "1", "DM_FP8_PRECOMPUTE": "1", "DM_FP8GLUE": "1", "DM_FP8GLUE_SCALING": "dynamic",
    "DM_FP8GLUE_SAVE_HT": "0", "DM_STALE_DOQ2S": "0", "DM_SHARED_X8": "1", "DM_FLASH_IMPL": "",
    "DM_SAC_SAVE_FA3WHEEL": "0", "DM_SDPA_BACKEND": "", "DM_DELTA_Q8": "0",
    "DM_AUX_CFG": "128,8,3",   # dm_mode aux_out launch configuration (kernel/dm_port/aux_out.py; speed only)
    "DM_AUX_NORM": "0",        # dm_mode aux_out: 1 = O' rows divided by the sum of their bf16-cast weights (switch test)
}
# (the name in the record, the module, the distributions that may provide it)
PACKAGES = (("torch", "torch", ("torch",)), ("triton", "triton", ("triton",)),
            ("fla", "fla", ("fla-core", "flash-linear-attention")), ("torchao", "torchao", ("torchao",)),
            ("transformer_engine", "transformer_engine", ("transformer_engine", "transformer-engine")))
ABSENT = "<absent>"


def _jsonable(v: Any) -> Any:
    """v as it reads back from JSON (tuples become lists, paths strings), so recorded and current values compare."""
    return json.loads(json.dumps(v, default=str))


def resolved_path(v: Any) -> Any:
    """A data path as the file system resolves it (os.path.realpath; a list or tuple element by element); None, an
    empty string and anything that is not a path unchanged. Never raises (a path that does not exist resolves as far
    as it can)."""
    if isinstance(v, (list, tuple)):
        return [resolved_path(x) for x in v]
    if isinstance(v, (str, os.PathLike)) and os.fspath(v):
        try:
            return os.path.realpath(v)
        except (OSError, ValueError):
            return os.fspath(v)
    return v


def _comparable(key: str, v: Any) -> Any:
    return resolved_path(v) if key in PATH_KEYS else v


def run_identity(cfg) -> dict:
    """The flat {key: value} identity of a TrainConfig (module docstring)."""
    out: dict[str, Any] = {k: getattr(cfg, k, None) for k in TRAIN_KEYS}
    for f in dataclasses.fields(cfg.optim):
        out[f"optim.{f.name}"] = getattr(cfg.optim, f.name)
    for f in dataclasses.fields(cfg.model):
        if f.name not in MODEL_SKIP:
            out[f"model.{f.name}"] = getattr(cfg.model, f.name)
    for k in DATA_KEYS:
        out[f"data.{k}"] = _comparable(f"data.{k}", getattr(cfg.data, k, None))
    for k, default in NUMERICS_ENV.items():
        out[k] = os.environ.get(k, default).strip()
    return _jsonable(out)


def _pkg_version(module: str, dists: tuple) -> Optional[str]:
    m = sys.modules.get(module)
    v = getattr(m, "__version__", None) if m is not None else None
    if v:
        return str(v)
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        return None
    for d in dists:
        try:
            return metadata.version(d)
        except Exception:  # noqa: BLE001 (not installed, or broken metadata: unknown)
            continue
    return None


def env_identity() -> dict:
    """The software of this process (module docstring): warn-only."""
    out = {name: _pkg_version(module, dists) for name, module, dists in PACKAGES}
    out["DM_GIT_REV"] = os.environ.get("DM_GIT_REV") or None
    return out


def diff(old: dict, new: dict) -> tuple[list, list]:
    """(changed, other): changed = [(key, old, new)] for the keys both have with different values (PATH_KEYS compared
    as resolved paths); other = the keys only one side has, with ABSENT for the missing side. Sorted by key."""
    changed = [(k, old[k], new[k]) for k in sorted(old.keys() & new.keys())
               if _comparable(k, old[k]) != _comparable(k, new[k])]
    other = [(k, old.get(k, ABSENT), new.get(k, ABSENT)) for k in sorted(old.keys() ^ new.keys())]
    return changed, other


def describe(rows: list) -> str:
    def show(v):
        return v if v == ABSENT else json.dumps(v)
    return "\n".join(f"    {k}: {show(o)} -> {show(n)}" for k, o, n in rows)


OK, REFUSED, UNREADABLE = "ok", "refused", "unreadable"


def plan_run_identity(ckpt_dir, cfg, resumed_step: Optional[int], allow: bool) -> tuple[str, list, Optional[dict]]:
    """Check <ckpt_dir>/run_identity.json without writing it (master only). resumed_step None = the run starts from
    scratch: the current identity is to be recorded (replacing any record of an earlier start that never completed a
    checkpoint). Otherwise the record is checked (and one is to be made when the run has none). Returns (status,
    [(level, message), ...], the record to write or None): status "ok"; "refused" = the identity differs, or the
    record does not parse (a retry cannot change that: the trainer exits 2); "unreadable" = an I/O error reading the
    record (refused too, but it may pass: exit 1). The last message says why. Write the record with
    record_run_identity: at once for a first start, after the load for a resume (module docstring)."""
    path = Path(ckpt_dir) / IDENTITY_FILE
    ident, env = run_identity(cfg), env_identity()
    stamp = {"time": round(time.time(), 3), "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "step": 0 if resumed_step is None else int(resumed_step), "job": os.environ.get("SLURM_JOB_ID") or None}
    msgs: list = []
    run = getattr(cfg, "run_name", None)

    def fresh(note: str) -> dict:
        return {"run_name": run, "recorded": dict(stamp, note=note), "identity": ident, "env": env, "changes": []}

    if resumed_step is None:
        msgs.append(("info", f"[identity] the run identity of {run} ({len(ident)} values) is in {path}"))
        return OK, msgs, fresh("first start")
    try:
        doc = read_json_file(path)
        if not isinstance(doc, dict) or not isinstance(doc.get("identity"), dict):
            raise ValueError("not a run identity record")
    except (FileNotFoundError, NotADirectoryError):
        msgs.append(("warning", f"[identity] {path} did not exist (the run started before run identities were "
                                f"recorded): this resume's values are recorded and later resumes are checked against them"))
        return OK, msgs, fresh(f"first resume (step {resumed_step}) of a run started before run identities were recorded")
    except (OSError, ValueError) as e:
        why = f"{type(e).__name__}: {e}"
        if not allow:
            status = UNREADABLE if isinstance(e, OSError) else REFUSED
            return status, [("critical", f"refusing to resume {run} from step {resumed_step}: {path} cannot be read "
                                        f"({why}), so the run's identity cannot be checked. Repair it, remove it (the "
                                        f"next resume records the current values), or set {ALLOW_ENV}=1 to record the "
                                        f"current values and continue")], None
        msgs.append(("warning", f"[identity] {path} could not be read ({why}); {ALLOW_ENV}=1: the current values are "
                                f"recorded"))
        return OK, msgs, fresh(f"re-recorded at step {resumed_step} ({ALLOW_ENV}=1): the record could not be read ({why})")

    changed, other = diff(doc["identity"], ident)
    env_changed, env_other = diff(doc["env"] if isinstance(doc.get("env"), dict) else {}, env)
    if changed and not allow:
        return REFUSED, [("critical", (
            f"refusing to resume {run} from step {resumed_step}: values that define its trajectory differ from the ones "
            f"recorded in {path} (recorded -> now):\n{describe(changed)}\n"
            f"Resuming would continue a different experiment under the same run name. Restore them (the launcher's "
            f"knobs NUM_STEPS, DM_LR, EXTRA_ARGS, S2_*, the arm's environment, the yaml), or set {ALLOW_ENV}=1 to "
            f"continue with the new values (the change is recorded in {path.name})"))], None
    if changed:
        msgs.append(("warning", f"[identity] {ALLOW_ENV}=1: {run} continues from step {resumed_step} with values that "
                                f"differ from its record (recorded -> now):\n{describe(changed)}"))
    elif allow:
        msgs.append(("warning", f"[identity] {ALLOW_ENV}=1 is set, but {run}'s identity matches its record: the knob is "
                                f"not needed. Unset it (a chain of jobs passes it on to every later job): while it is set, "
                                f"a later accidental change of the run's values is accepted with a warning only"))
    if other:
        msgs.append(("warning", f"[identity] keys present on one side only (a field the code added or removed since the "
                                f"record; not a refusal):\n{describe(other)}"))
    if env_changed or env_other:
        msgs.append(("warning", f"[identity] the software differs from the record (warning only; recorded -> now):\n"
                                f"{describe(env_changed + env_other)}"))
    if not (changed or other or env_changed or env_other):
        return OK, msgs, None
    new = dict(doc, identity=ident, env=env)
    new["changes"] = list(doc.get("changes") or []) + [dict(
        stamp, allowed=bool(changed), identity=[list(r) for r in changed + other],
        env=[list(r) for r in env_changed + env_other])]
    return OK, msgs, new


def record_run_identity(ckpt_dir, doc: dict) -> Optional[str]:
    """Write `doc` (from plan_run_identity) to <ckpt_dir>/run_identity.json atomically. None, or the warning when it
    could not be written (never raises: the next resume records it again)."""
    path = Path(ckpt_dir) / IDENTITY_FILE
    try:
        write_json_atomic(path, doc)
    except OSError as e:
        return f"[identity] could not write {path} ({type(e).__name__}: {e}); the next resume records it again"
    return None


def check_run_identity(ckpt_dir, cfg, resumed_step: Optional[int], allow: bool) -> tuple[str, list]:
    """plan_run_identity, then the record written at once (when there is one to write; a failure is a warning, added
    to the messages). Returns (status, [(level, message), ...]). The trainer plans and records separately (the record
    of a resume is written after the load)."""
    status, msgs, doc = plan_run_identity(ckpt_dir, cfg, resumed_step, allow)
    if doc is not None:
        warning = record_run_identity(ckpt_dir, doc)
        if warning:
            msgs.append(("warning", warning))
    return status, msgs
