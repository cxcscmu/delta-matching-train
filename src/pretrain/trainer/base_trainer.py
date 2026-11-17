"""Shared training engine. Backends (DDP / FSDP2) subclass this and implement a
handful of hooks; everything else (the loop, gradient accumulation, clipping,
logging, checkpoint orchestration, exact resume) lives here.

Gradient-accumulation math (token-weighted, exact for the packed-pretraining case):
  A rank's per-step batch has `local_bs` sequences split into `A` micro-batches.
  Let v_i = #valid (non -100) target tokens in micro i, V = sum_i v_i. The model
  returns per-micro CE_i = (1/v_i) * sum_{tok in micro i} loss_tok. We scale micro i
  by r_i = v_i / V before backward, so the accumulated gradient corresponds to
  L_rank = sum_i r_i * CE_i = (1/V) * sum_tok loss_tok : the rank token-mean.
  DDP/FSDP then mean-reduce across ranks. With equal V per rank (packed,
  fixed-length) this is exactly the global token-mean; with padding it is the
  mean-of-rank-means (document, switch to all-reduced token weights if needed).
  CE_i is the model's full training loss: loss_tok is the token's cross-entropy plus,
  when model.z_loss_weight > 0, the z-loss z_loss_weight * lse_tok^2 (lse = the log of
  the softmax normalizer; with 1e-4 and a 32k vocabulary about +0.011 at init). L_rank
  is logged as `train/ce_loss`, z-loss included.

Checkpoints and resume (a run is a chain of 4-hour jobs; docs/SETUP.md):
  * A checkpoint is complete when its `.complete` marker is there (utils/ckpt.py). Every file the resume needs is
    fsynced before the marker: the DCP shards (DCP itself), next_train_data_<rank>.pt (each rank), pytorch_model.bin,
    model_config.json, kernel_version.json and the sidecar trainer_state.json (the master; DDP: each rank's
    training_state_<rank>.pt). The sidecar holds what the DCP TrainState does not (a new TrainState key would make every
    older checkpoint unloadable): the skipped-update count, the world size, the last logged metrics, the training wall
    time. It is read with defaults, so checkpoints written before it existed still resume.
  * Right after a marker the master appends "<epoch> <SLURM_JOB_ID or 0> ckpt <run_name> <step>" to $DM_PROGRESS_FILE
    when it is set (scripts/slurm/job.sh's progress gate; never fails the save).
  * Rotation keeps the newest save_total_limit complete checkpoints and removes a victim by renaming it to
    `.trash-<name>.<ns>` first (never matched by `checkpoint-*`), then deleting it; a directory whose marker cannot be
    read is kept. The master removes leftover `.trash-*` directories at startup.
  * The resume is decided on the master and broadcast: it finds the newest complete checkpoint (an unreadable marker
    raises instead of falling back to an older one; the error says how to set such a checkpoint aside), reads
    kernel_version.json and the sidecar, checks the run identity (utils/run_identity.py: <ckpt_dir>/run_identity.json;
    DM_ALLOW_RUN_CHANGE=1 turns a refusal into a warning; the record is written only once the state has loaded, so a
    load that fails changes nothing) and every rank acts on that one decision; an error on the master is raised on
    every rank. After the state load the ranks' loaded steps are compared (all-gather). A per-rank failure to read
    next_train_data_<rank>.pt only skips the data-stream sanity check (the data position comes from global_step); so
    does a world-size change (the global sample order does not depend on it; DCP reshards, and a DDP rank without a
    training_state file of its own loads rank 0's: the replicas are identical).
  * A stop that is pending before the first step (a signal during startup, or DM_STOP_BEFORE already past) returns
    "stopped" without training or saving: a resumed run's newest checkpoint already holds this state, a fresh run has
    nothing to save. A run resumed AT max_steps (a finished run) still exports.
  * W&B: the trainer logs to `wandb_run` (pretrain.train sets it once the trainer is built) but never closes it.
    pretrain.train does that after the process group is closed (finish_wandb_bounded: DM_WANDB_FINISH_TIMEOUT seconds,
    default 60, for run.finish(); on a stop the run is marked preempting first), so a slow upload holds only the
    master.
  * Env: DM_STOP_BEFORE (epoch seconds: the time budget), DM_PROGRESS_FILE, DM_ALLOW_RUN_CHANGE, DM_ALLOW_KERNEL_CHANGE,
    DM_WANDB_FINISH_TIMEOUT, SLURM_JOB_ID (the progress lines' job id).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
import traceback
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from ..parallel.dist import (
    is_master, dist_barrier, get_rank, get_world_size, sync_tensor, broadcast_object, all_gather_objects,
)
from ..parallel.dist import is_dist as _is_dist
from ..parallel.wrap import build_optimizer, build_lr_scheduler, fp8_fsdp_all_gather_enabled
from ..utils.misc import Peekable, get_amp_dtype
from ..utils.ckpt import (
    CKPT_NAME_FORMAT, CKPT_GLOB, TRAINER_STATE_FILE, COMPLETE, ABSENT, MISMATCH, UNREADABLE,
    CheckpointMarkerUnreadable, checkpoint_step, find_last_complete_checkpoint, create_complete_marker_atomic_swap,
    marker_state_why, read_text_retry, read_trainer_state, torch_save_durable, write_text_atomic, write_json_atomic,
    fsync_dir, trash_checkpoint, cleanup_trash, unreadable_marker_message,
)
from ..utils.run_identity import (
    plan_run_identity, record_run_identity, ALLOW_ENV, OK as IDENTITY_OK, REFUSED as IDENTITY_REFUSED,
)
from ..utils.logging import RED, CYAN, RESET


class ResumeRefused(RuntimeError):
    """A resume refused for a reason a retry cannot change: the run identity differs (utils/run_identity.py) or the
    checkpoint's delta_matching build does (or cannot be told). Raised on every rank; pretrain.train prints its
    "RESUME REFUSED" line and exits 2, the configuration-error status a chain of jobs does not retry. (A record that
    cannot be READ is a plain RuntimeError, exit 1: an I/O error may pass.)"""


def finish_wandb_bounded(run, log, preempting: bool = False) -> bool:
    """run.finish(), bounded: it has no timeout of its own (it waits for the upload), so it runs in a daemon thread
    joined for DM_WANDB_FINISH_TIMEOUT seconds (default 60). On a stop (preempting) the run is first marked preempting
    when it has mark_preempting(): W&B then shows it as preempted (to be resumed), not finished. Returns False when
    finish() did not return in time (the run's data stays in WANDB_DIR; `wandb sync` uploads it later), else True.
    Never raises; `log(msg, level)` reports problems. pretrain.train calls it after the process group is closed (the
    other ranks must not wait for the master's upload); `run` None (the other ranks, W&B off) is a no-op."""
    if run is None:
        return True
    if preempting:
        mark = getattr(run, "mark_preempting", None)
        if callable(mark):
            try:
                mark()
            except Exception as e:  # noqa: BLE001
                log(f"[wandb] mark_preempting failed: {e}", "warning")
    try:
        timeout = float(os.environ.get("DM_WANDB_FINISH_TIMEOUT", "60") or 60)
    except ValueError:
        timeout = 60.0
    errors: list = []

    def finish():
        try:
            run.finish()
        except BaseException as e:  # noqa: BLE001 (reported by the caller's thread)
            errors.append(e)

    t = threading.Thread(target=finish, name="wandb-finish", daemon=True)
    t.start()
    t.join(max(timeout, 0.0))
    if t.is_alive():
        log(f"{RED}[wandb] run.finish() did not return within {timeout:g} s (DM_WANDB_FINISH_TIMEOUT): not waiting any "
            f"longer; the run's data stays in WANDB_DIR (upload it with `wandb sync`){RESET}", "warning")
        return False
    if errors:
        log(f"[wandb] finish failed: {errors[0]}", "warning")
    return True


def append_progress(kind: str, run: Any, detail: Any = "") -> None:
    """Append "<epoch> <SLURM_JOB_ID or 0> <kind> <run> [detail]" to $DM_PROGRESS_FILE when it is set: the durable-
    progress record scripts/slurm/job.sh reads to tell a link that made progress from one that did not. One write of
    the whole line with O_APPEND (concurrent writers do not interleave within a line); any failure is ignored: a
    progress line never fails a save."""
    path = os.environ.get("DM_PROGRESS_FILE")
    if not path:
        return
    try:
        line = f"{int(time.time())} {os.environ.get('SLURM_JOB_ID') or 0} {kind} {run}"
        line += f" {detail}\n" if str(detail) != "" else "\n"
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)
    except Exception:  # noqa: BLE001 (best effort by contract)
        pass


def last_logged_step(metrics_path) -> Optional[int]:
    """The highest step of a "metrics" record in a <run>.metrics.jsonl (what a W&B run that resumes the same id has
    already seen); None when there is none or the file cannot be read. Never raises."""
    if metrics_path is None:
        return None
    best = None
    try:
        with open(metrics_path, "r", errors="replace") as fh:
            for line in fh:
                if '"metrics"' not in line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and rec.get("event") == "metrics" and isinstance(rec.get("step"), int):
                    best = rec["step"] if best is None else max(best, rec["step"])
    except OSError:
        return None
    return best


def export_key(key: str) -> str:
    """A training state-dict key -> the plain module name the HF export uses: drops the activation-checkpoint
    wrapper and the torch.compile (OptimizedModule) levels."""
    return key.replace("._checkpoint_wrapped_module", "").replace("._orig_mod", "")


KERNEL_FILE = "kernel_version.json"   # in every checkpoint: the delta_matching build the run trained with


def kernel_py_sha(pkg_dir) -> Optional[str]:
    """The Python half of a delta_matching package (the fp8 producers, the backward split, ...), which code_sha (csrc/
    only) does not cover: sha256 over the sorted names and bytes of its *.py files except the build stamp _build_info.py,
    first 12 hex digits (kernels/PROVENANCE.json py_sha). None when the directory cannot be read."""
    try:
        h = hashlib.sha256()
        for f in sorted(os.listdir(pkg_dir)):
            if f.endswith(".py") and f != "_build_info.py":
                h.update(f.encode())
                h.update(Path(pkg_dir, f).read_bytes())
        return h.hexdigest()[:12]
    except OSError:
        return None


def kernel_identity() -> Optional[dict]:
    """The delta_matching kernel build this process trains with: kernel_version()'s code_sha (the hash of the csrc/ the
    package was built from), its bf16_dp flag and library path; py_sha, the hash of the package's Python files
    (kernel_py_sha); and dm_so, the library DM_SO names instead of the package's own build (resolved path; None without
    DM_SO): code_sha is the installed package's stamp, so the path is all that identifies such a library. None when no
    layer imported the kernel package (the bf16-attention arms). Builds of different kernel sources (the bundled
    KDA-evolved kernel, the original release) differ in their head_dim-128 numerics at round-off level, so checkpoints
    record it and a resume compares it."""
    mod = sys.modules.get("delta_matching")
    if mod is None:
        return None
    so = os.environ.get("DM_SO") or None
    so = os.path.realpath(so) if so else None
    f = getattr(mod, "__file__", None)
    py = kernel_py_sha(os.path.dirname(os.path.abspath(f))) if f else None
    try:
        kv = mod.kernel_version()
    except Exception as e:  # noqa: BLE001 (a provenance stamp must never stop a run)
        return {"code_sha": None, "py_sha": py, "dm_so": so, "error": f"{type(e).__name__}: {e}"}
    return {"code_sha": kv.get("code_sha"), "bf16_dp": kv.get("bf16_dp"), "py_sha": py, "dm_so": so,
            "library": kv.get("library")}


def check_kernel_identity(saved: Optional[dict], now: Optional[dict], ckpt, allow: bool) -> Optional[str]:
    """None when a checkpoint's kernel (`saved`, its KERNEL_FILE) and this process's (`now`, kernel_identity()) are the
    same build (code_sha, bf16_dp, py_sha and dm_so; a missing field reads as None; the package's library path may
    differ).
    Otherwise the mismatch message: raised as a RuntimeError, or returned for a warning when `allow`
    (DM_ALLOW_KERNEL_CHANGE=1)."""
    def key(d):
        return None if d is None else (d.get("code_sha"), bool(d.get("bf16_dp")), d.get("py_sha"), d.get("dm_so"))
    if key(saved) == key(now):
        return None
    msg = (f"{ckpt} was trained with the delta_matching kernel {saved}, but this process runs {now}. Resuming would "
           "continue the run on another kernel build, whose head_dim-128 numerics are not bitwise the same; install "
           "the checkpoint's kernel (scripts/install_kernel.sh, DM_KERNEL_SRC; kernels/README.md) or set "
           "DM_ALLOW_KERNEL_CHANGE=1 to continue anyway")
    if not allow:
        raise RuntimeError(msg)
    return msg


def _kernel_verdict(rec: dict, now: Optional[dict], ckpt, allow: bool) -> tuple[Optional[str], bool, bool]:
    """(message, refuse, deterministic) for a checkpoint's KERNEL_FILE as the master read it (`rec`: {"text": ...},
    {"missing": True} or {"error": "..."}) against this process's kernel_identity() `now`: (None, False, _) = the same
    build or no kernel to compare; (msg, False, _) = a difference allowed by DM_ALLOW_KERNEL_CHANGE=1 (a warning);
    (msg, True, det) = refuse; det False only when the record could not be read (an I/O error may pass)."""
    deterministic = True
    if "text" in rec:
        try:
            saved = json.loads(rec["text"])
            if saved is not None and not isinstance(saved, dict):
                raise ValueError(f"not a kernel record: {saved!r}")
        except ValueError as e:
            problem = f"it is unreadable ({type(e).__name__}: {e})"
        else:
            try:
                return check_kernel_identity(saved, now, ckpt, allow), False, True
            except RuntimeError as e:
                return str(e), True, True
    elif rec.get("missing"):
        problem = "there is none (written before kernel builds were recorded)"
    else:
        problem = f"it is unreadable ({rec.get('error')})"
        deterministic = False
    if now is None:   # no kernel in this process: nothing to compare
        return None, False, True
    msg = (f"cannot tell which delta_matching build {ckpt} was trained with: {KERNEL_FILE} {problem}; this process "
           f"runs {now}. Install the kernel the run was trained with (scripts/install_kernel.sh, DM_KERNEL_SRC; "
           "kernels/README.md) or set DM_ALLOW_KERNEL_CHANGE=1 to resume on this build")
    return msg, not allow, deterministic


def _resolve_forward_ctx(model: nn.Module):
    """The forward context published by the model's mixers (`mixer_forward_ctx`), nullcontext
    if none. All publishers must agree (one region for the whole forward)."""
    fns = {}
    for m in model.modules():
        f = getattr(m, "mixer_forward_ctx", None)
        if f is not None:
            fns[id(f)] = f
    if not fns:
        return nullcontext
    if len(fns) > 1:
        raise RuntimeError(f"conflicting mixer_forward_ctx publishers in the model: {list(fns.values())}")
    return next(iter(fns.values()))


class BaseTrainer:
    backend: str = "base"

    def __init__(self, model, cfg, train_loader, device, main_LOG, wandb_run=None):
        self.model = model
        self.cfg = cfg
        self.train_loader = train_loader
        self.device = device
        self.main_log = main_LOG
        self.wandb_run = wandb_run

        self.pad_token_id = getattr(train_loader, "pad_token_id", None)
        # lazily resolved in _optimizer_step (None = unresolved; False = off).
        self._fp8_precompute_fn = None
        # default-inert: a mixer may publish `mixer_forward_ctx` (a callable
        # returning a context manager) that must wrap the WHOLE model forward of every microbatch.
        # attn_impl='te_fp8' uses it for its single Transformer Engine autocast region (TE's
        # delayed-scaling amax/scale update runs once at the exit of the outermost region, so it
        # must not be entered per layer; modeling/te_attn.py). nullcontext for every other arm.
        self._fwd_ctx = _resolve_forward_ctx(self.unwrapped_model)
        self.global_batch_size = cfg.global_batch_size
        self.gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self.num_devices = cfg.num_devices
        self.resume = cfg.resume
        # stop point vs LR-schedule horizon: the scheduler is always built from cfg.num_steps (see
        # build_lr_scheduler below), so stop_at_step ends the loop early while leaving the WSD curve
        # unchanged (a run split into phases keeps one continuous schedule). Guarded: a stop beyond
        # the horizon is meaningless, so fall back to num_steps.
        stop_at = int(getattr(cfg, "stop_at_step", 0) or 0)
        self.max_steps = stop_at if 0 < stop_at <= cfg.num_steps else cfg.num_steps
        self.chunk_size = cfg.chunk_size

        self.global_step = 0
        self._last_metrics: dict[str, Any] = {}
        self._nonfinite_steps = 0          # skipped updates of the whole run (seeded from the resumed sidecar)
        self._trained_steps = 0            # optimizer steps this process ran (0: it only resumed and exported, or stopped)
        self._resume_state: dict = {}      # the resumed checkpoint's sidecar (trainer_state.json), {} when none
        self._train_t0: Optional[float] = None
        self._identity_record = None       # master: the run_identity.json to write once the resume has loaded
        self._hf_export_steps = set(cfg.hf_export_steps or [])  # token-budget ladder: export {hf_save_dir}/step{N}
        self.epoch = 0
        self.train_loader.sampler.set_epoch(self.epoch)
        if len(self.train_loader) == 0:   # an epoch without a single batch would roll over forever
            raise ValueError(
                f"the training data yields no batch per epoch: {self.train_loader.sampler.num_samples} samples per rank "
                f"< the per-rank batch of {self.train_loader.batch_size} (more data, a wider data.chunk_subset, or a "
                f"smaller global_batch_size)")
        self.train_iter = Peekable(self.train_loader)
        self.next_train_data_for_check = None
        self.resume_sanity_check_done = True

        self._setup_amp(cfg.bf16, cfg.fp16)
        self._setup_dirs()
        self._setup_training()

        # preemption-safe stops and a local metrics log (docs/SETUP.md section 5).
        # A SIGTERM (or SIGUSR1) on any rank, or the master's time budget DM_STOP_BEFORE (epoch seconds), makes
        # every rank agree at the next step boundary (one 1-int all-reduce per step), save a complete checkpoint and
        # return "stopped"; pretrain.train then exits 75 so a launcher resumes instead of exporting.
        self._stop_requested: Optional[str] = None
        self._deadline = float(os.environ.get("DM_STOP_BEFORE", "0") or 0)
        self._metrics_path = (Path(str(cfg.log_root_dir)) / f"{cfg.run_name}.metrics.jsonl") if is_master() else None
        try:
            signal.signal(signal.SIGTERM, self._on_stop_signal)
            signal.signal(signal.SIGUSR1, self._on_stop_signal)
        except (ValueError, OSError):  # not the main thread / unsupported: periodic checkpoints still apply
            pass

        dist_barrier()
        self.resume_routine()
        dist_barrier()

    # ----------------------------------------------------------------- setup
    @property
    def unwrapped_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    def _setup_amp(self, bf16: bool, fp16: bool) -> None:
        self.amp_dtype = get_amp_dtype(bf16, fp16)
        if self.amp_dtype == torch.float16:
            raise NotImplementedError(
                "fp16 loss-scaling is not wired (bf16 is the supported mixed-precision path "
                "for H100/Blackwell pretraining). Set bf16=True.")
        self.enable_amp = self.amp_dtype != torch.float32

    def _setup_dirs(self) -> None:
        self.checkpoint_dir = Path(self.cfg.ckpt_dir)
        self.log_dir = Path(self.cfg.log_dir)
        if is_master():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            # what an interrupted rotation left (it renames a victim to .trash-<name>.<ns> before deleting it)
            removed = cleanup_trash(self.checkpoint_dir)
            if removed:
                self.main_log(f"[ckpt] removed {len(removed)} leftover directories of an interrupted rotation under "
                              f"{self.checkpoint_dir}: {', '.join(removed[:4])}{' ...' if len(removed) > 4 else ''}")
        dist_barrier()

    def _setup_training(self) -> None:
        self.optimizer = build_optimizer(self.model, self.cfg.opt_cfg)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.cfg.opt_cfg, self.cfg.num_steps)
        self._build_states()  # backend hook (FSDP2 sets up DCP Stateful payload)
        self.model.train()

    # ----------------------------------------------------------------- data
    def _start_next_epoch(self) -> None:
        """Roll the data stream over to the next epoch (the sampler reshuffles with seed + epoch)."""
        self.epoch += 1
        self.train_loader.sampler.set_epoch(self.epoch)
        self.train_iter = Peekable(self.train_loader)

    def next_train_data(self) -> dict[str, Any]:
        try:
            return next(self.train_iter)
        except StopIteration:   # end of the epoch; every epoch has at least one batch (checked in __init__)
            self._start_next_epoch()
            return next(self.train_iter)

    def peek_next_train_data(self) -> dict[str, Any]:
        """The batch next_train_data returns next, without consuming it. At the end of an epoch the stream rolls over
        here, exactly as next_train_data would do. A throwaway look-ahead iterator must not be used: with persistent
        loader workers the DataLoader hands out ONE iterator object and resets it on every iter(), so a look-ahead
        would reset the live iterator and drop the next epoch's first batch."""
        try:
            return self.train_iter.peek()
        except StopIteration:
            self._start_next_epoch()
            return self.train_iter.peek()

    def get_train_batch(self) -> dict[str, Any]:
        feed = self.next_train_data()
        input_ids = feed["input_ids"].to(self.device, non_blocking=True)
        if "labels" in feed:
            labels = feed["labels"].to(self.device, non_blocking=True)
        else:
            labels = input_ids.clone()
            if self.pad_token_id is not None:
                labels[input_ids == self.pad_token_id] = -100
        if self.pad_token_id is None:
            attention_mask = None
        else:
            attention_mask = (input_ids != self.pad_token_id).to(torch.long).to(self.device, non_blocking=True)
        # document-aware loader supplies per-token position_ids (per-doc RoPE positions)
        position_ids = feed.get("position_ids")
        if position_ids is not None:
            position_ids = position_ids.to(self.device, non_blocking=True)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask,
                "position_ids": position_ids}

    # ----------------------------------------------------------------- step
    def _micro_step(self, input_ids, labels, loss_ratio, attention_mask=None, position_ids=None) -> dict[str, Any]:
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.enable_amp), \
                self._fwd_ctx():   # mixer-published forward context (nullcontext unless te_fp8)
            out = self.model(input_ids, position_ids=position_ids, labels=labels, attention_mask=attention_mask)
            loss = out["loss"] * loss_ratio
        loss.backward()
        return {"ce_loss": out["loss"].detach()}

    def train_step(self, grad_clip: Optional[float]) -> dict[str, Any]:
        self.optimizer.zero_grad(set_to_none=True)
        feed = self.get_train_batch()
        if self.resume and not self.resume_sanity_check_done:
            self._resume_sanity_check(feed)

        input_ids, labels, attn = feed["input_ids"], feed["labels"], feed["attention_mask"]
        pos = feed.get("position_ids")
        local_bs = input_ids.size(0)
        A = self.gradient_accumulation_steps
        assert local_bs % A == 0, f"per-rank batch {local_bs} not divisible by grad_accum {A}"

        ids_chunks = torch.chunk(input_ids, A)
        lab_chunks = torch.chunk(labels, A)
        attn_chunks = torch.chunk(attn, A) if attn is not None else [None] * A
        pos_chunks = torch.chunk(pos, A) if pos is not None else [None] * A

        # token-weighted accumulation weights r_i = v_i / V (see module docstring)
        valid_counts = [(lc[:, 1:] != -100).sum() for lc in lab_chunks]
        total_valid = torch.stack(valid_counts).sum().clamp(min=1)

        step_ntokens = input_ids.numel()
        t0 = time.perf_counter()
        logging_metrics: dict[str, Any] = defaultdict(float)

        for i in range(A):
            self._set_grad_sync(i == A - 1)
            loss_ratio = (valid_counts[i].float() / total_valid.float())
            micro = self._micro_step(ids_chunks[i], lab_chunks[i], loss_ratio, attn_chunks[i], pos_chunks[i])
            # log the token-weighted loss: sum_i r_i * CE_i = the rank token-mean (z-loss included, see the module docstring)
            logging_metrics["train/ce_loss"] += (loss_ratio * micro["ce_loss"]).detach()

        for k in list(logging_metrics):
            logging_metrics[k] = sync_tensor(logging_metrics[k], reduce="mean").item()

        grad_norm = self._optimizer_step(grad_clip)
        self.lr_scheduler.step()

        dt = time.perf_counter() - t0
        logging_metrics["train/grad_norm"] = grad_norm
        # Cumulative skipped updates. A run that quietly skips steps is a different experiment from
        # one that does not, so this rides in the metrics rather than only in the text log.
        logging_metrics["train/skipped_updates"] = getattr(self, "_nonfinite_steps", 0)
        logging_metrics["train/step_time"] = dt
        logging_metrics["train/tokens_per_sec"] = step_ntokens * get_world_size() / dt
        logging_metrics["train/learning_rate"] = self.optimizer.param_groups[0]["lr"]
        logging_metrics["train/valid_tokens"] = int(total_valid.item())
        if self.device.type == "cuda":  # running peak (not reset): "does it fit" headroom
            logging_metrics["train/peak_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
        return logging_metrics

    def _frozen_params(self):
        """OPT-IN: DM_FREEZE_GRAD_REGEX=<regex> freezes every parameter whose name matches
        (their gradients are zeroed before clipping and their values restored after every optimizer step, so
        neither Adam nor decay can move them). Default unset: inert."""
        if getattr(self, "_freeze_cache", None) is None:
            import re
            pat = os.environ.get("DM_FREEZE_GRAD_REGEX", "")
            frz = []
            if pat:
                rx = re.compile(pat)
                for name, p in self.model.named_parameters():
                    if rx.search(name):
                        frz.append((name, p, p.detach().clone()))
                self.main_log(f"[freeze] DM_FREEZE_GRAD_REGEX={pat!r}: {len(frz)} parameters frozen: "
                              + ", ".join(n for n, _, _ in frz[:8]) + (" ..." if len(frz) > 8 else ""))
            self._freeze_cache = frz
        return self._freeze_cache

    def _log_gains(self) -> None:
        """OPT-IN (default OFF): DM_GAIN_LOG_EVERY=<n> appends, every n optimizer steps, the qk-norm gain vectors
        (q_norm / k_norm weights) of every attention mixer to <log_root>/<run>.gains.jsonl (rank 0 writes; every rank
        takes part in the all-gather of the sharded 128-element vectors).  Works on every arm, the bf16 ones included
        (DM_DIAG, the kernel-side instrument, only sees the fp8 attention)."""
        every = getattr(self, "_gain_log_every", None)
        if every is None:
            every = int(os.environ.get("DM_GAIN_LOG_EVERY", "0") or 0)
            self._gain_log_every = every
            self._gain_log_path = (Path(str(self.cfg.log_root_dir)) / f"{self.cfg.run_name}.gains.jsonl") if (every and is_master()) else None
        if not every or self.global_step % every:
            return
        rec = {}
        for name, mod in self.model.named_modules():
            if name.endswith("mixer") and hasattr(mod, "q_norm") and hasattr(mod, "k_norm") and "layers." in name:
                L = int(name.split("layers.")[1].split(".")[0])
                g = []
                for w in (mod.q_norm.weight, mod.k_norm.weight):
                    w = w.full_tensor() if hasattr(w, "full_tensor") else w
                    g.append([round(float(x), 5) for x in w.detach().float().cpu()])
                rec[str(L)] = {"g_q": g[0], "g_k": g[1]}
        if self._gain_log_path is not None and rec:
            with open(self._gain_log_path, "a") as fh:
                fh.write(json.dumps({"step": self.global_step, "layers": rec}) + "\n")

    def _optimizer_step(self, grad_clip: Optional[float]) -> float:
        grad_norm = 0.0
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        frozen = self._frozen_params()
        for _, p, _ in frozen:
            if p.grad is not None:
                p.grad.zero_()
        if grad_clip is not None and grad_clip > 0:
            gn = torch.nn.utils.clip_grad_norm_(params, grad_clip)
            if hasattr(gn, "full_tensor"):   # FSDP2 DTensor global norm
                gn = gn.full_tensor()
            grad_norm = float(gn.item()) if torch.is_tensor(gn) else float(gn)
        else:                                # no clipping: still need a finiteness verdict
            with torch.no_grad():
                sq = sum((p.grad.detach().float() ** 2).sum() for p in params if p.grad is not None)
                if hasattr(sq, "full_tensor"):
                    sq = sq.full_tensor()
                grad_norm = float(sq.sqrt().item())

        # SKIP THE UPDATE ON A NON-FINITE GRADIENT. Without this a single bad step is terminal:
        # optimizer.step() writes NaN into the weights, every subsequent forward is NaN, and the
        # run keeps burning GPU on garbage while still logging steps; the trainer, not the kernel,
        # is what turns one bad gradient into a dead run, and no trainer should be one bad step
        # away from unrecoverable.
        if not math.isfinite(grad_norm):
            self._nonfinite_steps = getattr(self, "_nonfinite_steps", 0) + 1
            if self._nonfinite_steps <= 20 or self._nonfinite_steps % 50 == 0:
                self.main_log(f"[skip] non-finite grad_norm={grad_norm} at step "
                              f"{self.global_step}; update SKIPPED "
                              f"(total skipped: {self._nonfinite_steps})")
            self.optimizer.zero_grad(set_to_none=True)
            return grad_norm

        self.optimizer.step()
        if frozen:
            with torch.no_grad():
                for _, p, p0 in frozen:
                    p.copy_(p0)
        # DM_FP8_PRECOMPUTE=1 (the default; 0 opts out), effective when the model has torchao Float8Linear weights
        # wrapped for the FSDP2 fp8 all-gather (fp8 FSDP2 runs with the tensorwise recipe and DM_FP8_FSDP_AG=1, see
        # parallel/wrap.py::apply_fp8): precompute the tensorwise weight scales ONCE per optimizer step (one batched
        # foreach-amax + one all-reduce) instead of torchao all-reducing an amax inside every fsdp_pre_all_gather
        # hook. Weights are static within a step, so this is the correct dynamic-scaling semantics, not delayed
        # scaling. Every other run (bf16, DDP, another recipe, DM_FP8_FSDP_AG=0) has no wrapped weights, and the
        # switch is inert there: nothing is called and torchao is not imported.
        if self._fp8_precompute_fn is None:
            self._fp8_precompute_fn = False
            if (os.environ.get("DM_FP8_PRECOMPUTE", "1") == "1" and self.backend == "fsdp2"
                    and getattr(self.cfg, "fp8", False)
                    and fp8_fsdp_all_gather_enabled(getattr(self.cfg, "fp8_recipe", "tensorwise"))):
                from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
                self._fp8_precompute_fn = precompute_float8_dynamic_scale_for_fsdp
        if self._fp8_precompute_fn:
            self._fp8_precompute_fn(self.model)
        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm

    # ------------------------------------------------------- backend hooks
    def _build_states(self) -> None:
        """Optional: prepare the resume-state payload (FSDP2 DCP)."""

    def _set_grad_sync(self, is_last_micro: bool) -> None:
        """Gate gradient all-reduce/reduce-scatter to the last micro-batch."""

    def _save_resume_state(self, save_dir: Path) -> None:
        raise NotImplementedError

    def _load_resume_state(self, load_dir: Path) -> int:
        raise NotImplementedError

    def _gather_full_state_dict(self) -> Optional[dict]:
        """Return a CPU full (unsharded) state dict on master (None elsewhere), for
        the HF-loadable pytorch_model.bin artifact."""
        raise NotImplementedError

    # ----------------------------------------------------------- checkpoint
    def save_ckpt(self, save_dir: Optional[Path] = None) -> Path:
        base = self.checkpoint_dir if save_dir is None else Path(save_dir)
        save_dir = base / CKPT_NAME_FORMAT.format(step=self.global_step)
        if is_master():
            save_dir.mkdir(parents=True, exist_ok=True)
        dist_barrier()

        self.main_log(f"Saving checkpoint -> {save_dir}")
        t0 = time.time()
        self._save_resume_state(save_dir)            # backend: DCP (fsyncs its files) / torch.save (fsynced)
        self._save_full_model(save_dir)              # HF-loadable bin + config + kernel record (master, fsynced)
        self._save_dataloader_state(save_dir)        # per-rank next batch (resume sanity check, fsynced)
        if is_master():
            self._write_trainer_state(save_dir)      # the sidecar (atomic, fsynced): before the barrier of the marker
        dist_barrier()
        if is_master():
            fsync_dir(save_dir)                      # the names of the files above (best effort)
            create_complete_marker_atomic_swap(save_dir, version_str=self.cfg.VERSION_STRING)
            # durable progress for the chain (scripts/slurm/job.sh): right after the marker, never raises
            append_progress("ckpt", getattr(self.cfg, "run_name", None), self.global_step)
            # rotate only once the new checkpoint is complete, so an interrupted save never costs an old one
            self._rotate_checkpoints(base)
        dist_barrier()
        self._metrics_event("checkpoint", path=str(save_dir), seconds=round(time.time() - t0, 1))
        return save_dir

    def _trainer_state(self) -> dict:
        """The sidecar of a checkpoint (TRAINER_STATE_FILE): what the resume needs beyond the DCP payload. Every field is
        read with a default (read_trainer_state), so a checkpoint without it (an older one) still resumes."""
        prev = getattr(self, "_resume_state", None) or {}
        t0 = getattr(self, "_train_t0", None)
        try:
            seconds = float(prev.get("train_seconds", 0) or 0) + (time.time() - t0 if t0 else 0.0)
        except (TypeError, ValueError):
            seconds = 0.0
        return {
            "format": 1,
            "run_name": getattr(self.cfg, "run_name", None),
            "global_step": int(self.global_step),
            "world_size": get_world_size(),                        # a resume on another world size skips the per-rank
            "skipped_updates": int(getattr(self, "_nonfinite_steps", 0) or 0),   # next-batch check
            "last_metrics": dict(getattr(self, "_last_metrics", None) or {}),
            "train_seconds": round(seconds, 1),                    # wall time in train() over every job of the run
            "time": round(time.time(), 3),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def _write_trainer_state(self, save_dir: Path) -> None:
        """Master: write the sidecar atomically (temporary file + fsync + os.replace). A failure fails the save (it is
        written before the marker, like every other file of the checkpoint)."""
        write_json_atomic(Path(save_dir) / TRAINER_STATE_FILE, self._trainer_state())

    def _rotate_checkpoints(self, base: Path) -> None:
        """Keep the newest `save_total_limit` COMPLETE checkpoints under `base`, the one just saved included
        (None or 0 keeps all). Runs on the master after the new checkpoint's marker is written. A partial dir (no
        valid marker: a save that was killed, or another format version's marker) older than the newest complete one
        is removed; a partial dir never displaces a complete one. Dirs whose name carries no step number are left
        alone, and so is a dir whose marker exists but cannot be read (marker_state "unreadable": it may be the
        newest complete checkpoint). A victim is renamed to `.trash-<name>.<ns>` before it is deleted
        (utils/ckpt.py::trash_checkpoint), so an interrupted deletion never leaves a complete-looking remnant. Never
        raises: the other ranks wait in a barrier meanwhile, and a failed rotation only keeps old checkpoints longer."""
        try:
            limit = self.cfg.save_total_limit
            version = self.cfg.VERSION_STRING
            dirs = sorted((p for p in Path(base).glob(CKPT_GLOB) if p.is_dir() and checkpoint_step(p) >= 0),
                          key=checkpoint_step)
            states = {}
            for p in dirs:
                states[p], why = marker_state_why(p, version)
                if states[p] == UNREADABLE:
                    self.main_log(f"{RED}[ckpt] keeping {p}: its marker exists but cannot be read ({why}){RESET}",
                                  "warning")
            complete = [p for p in dirs if states[p] == COMPLETE]
            newest = checkpoint_step(complete[-1]) if complete else -1
            for p in dirs:
                if states[p] in (ABSENT, MISMATCH) and checkpoint_step(p) < newest:
                    self.main_log(f"Removing partial checkpoint {p} (older than the newest complete one)", "warning")
                    self._remove_checkpoint(p)
            if limit:
                # DM_CKPT_KEEP_STEPS="6000,..." (opt-in, default empty): checkpoints of these steps are never rotated
                # away (an analysis snapshot inside a run that otherwise keeps only the newest save_total_limit)
                keep = {int(x) for x in os.environ.get("DM_CKPT_KEEP_STEPS", "").replace(" ", "").split(",") if x}
                for p in complete[:-limit]:
                    if checkpoint_step(p) in keep:
                        continue
                    self._remove_checkpoint(p)
        except Exception as e:  # noqa: BLE001
            self.main_log(f"{RED}[ckpt] checkpoint rotation under {base} failed ({type(e).__name__}: {e}); the old "
                          f"checkpoints stay until a later save rotates them{RESET}", "warning")

    def _remove_checkpoint(self, p: Path) -> None:
        msg = trash_checkpoint(p)
        if msg:
            self.main_log(f"{RED}[ckpt] {msg}{RESET}", "warning")

    def _save_full_model(self, save_dir: Path) -> None:
        full_sd = self._gather_full_state_dict()
        if is_master():
            torch_save_durable(full_sd, save_dir / "pytorch_model.bin")
            self.unwrapped_model.save_config(save_dir)              # model_config.json (fsynced by the model)
            # the kernel build, compared on resume
            write_text_atomic(save_dir / KERNEL_FILE, json.dumps(kernel_identity()))
        dist_barrier()

    def _on_master(self, fn, what: str):
        """fn() run on the master only; its result (plain, picklable data) is broadcast and returned on every rank. An
        exception on the master is raised on every rank (the master's own, the others a RuntimeError with its message),
        so no rank is left waiting in the next collective while the master is gone. Without a process group: fn()."""
        result, err, exc = None, None, None
        if is_master():
            try:
                result = fn()
            except Exception as e:  # noqa: BLE001 (handed to every rank below)
                exc = e
                err = f"{what} failed on rank 0: {type(e).__name__}: {e}"
                self.main_log(f"{RED}{err}; every rank stops{RESET}", "critical")
        result, err = broadcast_object((result, err))
        if exc is not None:
            raise exc
        if err is not None:
            raise RuntimeError(err)
        return result

    def _check_resume_kernel(self, ckpt: Path) -> None:
        """Refuse to resume `ckpt` on another delta_matching build than the one it was trained with
        (check_kernel_identity), and, when this process runs the kernel, from a checkpoint whose build cannot be read: no
        KERNEL_FILE (written before builds were recorded, most likely on the original release) or an unreadable one.
        DM_ALLOW_KERNEL_CHANGE=1 turns a refusal into a warning (and is itself a warning when nothing differs: a chain
        passes it on to every later job, where it would accept a later accidental change; it is needed until the run
        has saved a checkpoint on the new build). The master reads KERNEL_FILE and broadcasts what it read (or why it
        could not), so every rank compares the same record; the ranks' verdicts are then gathered and a refusal on any
        rank is raised on every rank, before any collective of the state load."""
        allow = os.environ.get("DM_ALLOW_KERNEL_CHANGE", "0") == "1"
        now = kernel_identity()
        f = Path(ckpt) / KERNEL_FILE

        def read() -> dict:
            try:
                return {"text": read_text_retry(f)}
            except (FileNotFoundError, NotADirectoryError):
                return {"missing": True}
            except OSError as e:
                return {"error": f"{type(e).__name__}: {e}"}

        rec = self._on_master(read, f"reading {f}")
        msg, fatal, deterministic = _kernel_verdict(rec, now, ckpt, allow)
        verdicts = all_gather_objects((msg, fatal, deterministic))
        refusals = [(m, det) for m, bad, det in verdicts if bad]
        if refusals:
            m, det = refusals[0]
            raise (ResumeRefused if det else RuntimeError)(m)
        if msg:
            self.main_log(f"{RED}[kernel] {msg} (continuing: DM_ALLOW_KERNEL_CHANGE=1){RESET}", "warning")
        elif allow:
            self.main_log(f"{RED}[kernel] DM_ALLOW_KERNEL_CHANGE=1 is set, but {ckpt} needs no kernel change (the same "
                          "build, or no kernel to compare): the knob is not needed. Unset it once the run has saved a "
                          "checkpoint on the new build (a chain of jobs passes it on to every later job): while it is "
                          f"set, a later change of the kernel build is accepted with a warning only{RESET}", "warning")

    def _save_dataloader_state(self, save_dir: Path) -> None:
        # per-rank next batch for the resume sanity check (the data position itself is re-derived from global_step)
        nxt = self.peek_next_train_data()
        torch_save_durable({"input_ids": nxt["input_ids"]}, save_dir / f"next_train_data_{get_rank()}.pt")
        dist_barrier()

    def _resume_decision(self, load_dir: Path, global_step: int = -1) -> Optional[Path]:
        """The checkpoint to resume, decided on the master and broadcast (every rank loads the same one): the newest
        complete `checkpoint-*` under `load_dir`, or `checkpoint-<global_step>` when it is complete. None when there
        is none. A marker that exists but cannot be read raises on every rank (never a silent fall back to an older
        checkpoint; a per-rank decision let one rank's transient read error resume it elsewhere than the others)."""
        version = self.cfg.VERSION_STRING

        def decide() -> Optional[str]:
            if global_step == -1:
                ckpt = find_last_complete_checkpoint(load_dir, version_str=version)
                return None if ckpt is None else str(ckpt)
            ckpt = Path(load_dir) / CKPT_NAME_FORMAT.format(step=global_step)
            state, why = marker_state_why(ckpt, version)
            if state == UNREADABLE:
                raise CheckpointMarkerUnreadable(unreadable_marker_message(ckpt, why))
            return str(ckpt) if state == COMPLETE else None

        path = self._on_master(decide, f"choosing the checkpoint to resume under {load_dir}")
        return None if path is None else Path(path)

    def load(self, load_dir: Path, global_step: int = -1) -> Optional[Path]:
        """Resume from the newest complete checkpoint under `load_dir` (or checkpoint-<global_step>): None when there is
        none, else the checkpoint loaded (_resume_decision, then _load_checkpoint)."""
        ckpt = self._resume_decision(load_dir, global_step)
        if ckpt is None:
            return None
        return self._load_checkpoint(ckpt)

    def _load_checkpoint(self, ckpt: Path) -> Path:
        """Load `ckpt` (the master's decision) on every rank: the kernel check, the sidecar (master, broadcast), the
        backend state, then the ranks' loaded steps are compared (all-gather) and a failure on any rank is raised on
        every rank. Then the data position, the replay warning, the sidecar's skipped-update count and world size, and
        this rank's next_train_data for the sanity check (a file that cannot be read only skips the check)."""
        self._check_resume_kernel(ckpt)
        state, note = self._on_master(lambda: read_trainer_state(ckpt), f"reading {Path(ckpt) / TRAINER_STATE_FILE}")
        if note:
            self.main_log(f"{RED}[resume] {note}{RESET}", "warning")

        loaded, err = None, None
        try:
            loaded = int(self._load_resume_state(ckpt))
        except Exception as e:  # noqa: BLE001 (gathered below and raised on every rank)
            err = f"rank {get_rank()}: {type(e).__name__}: {e}"
            tb = traceback.format_exc()
            if is_master():
                self.main_log(f"{RED}Failed to load resume state from {ckpt}: {e}{RESET}\n{tb}", "critical")
            else:
                print(f"[rank {get_rank()}] Failed to load resume state from {ckpt}: {e}\n{tb}", file=sys.stderr,
                      flush=True)
        results = all_gather_objects((loaded, err))
        errors = [e for _, e in results if e]
        if errors:
            raise RuntimeError(f"Failed to load resume state from {ckpt}: " + "; ".join(errors))
        steps = [s for s, _ in results]
        if len(set(steps)) != 1:
            raise RuntimeError(f"the ranks loaded different steps from {ckpt} (in rank order: {steps}); refusing to "
                               "train on a mix of states")
        loaded_step = steps[0]

        # global_step is the COMPLETED-step count; the checkpoint saved after step
        # `loaded_step` completed, so we continue *at* loaded_step (its batch is the
        # one stored in next_train_data_{rank}.pt for the sanity check).
        self.jump_to_step(loaded_step)
        self._replay_warning(loaded_step)

        if state and state.get("global_step") not in (None, loaded_step):
            self.main_log(f"{RED}[resume] {TRAINER_STATE_FILE} of {ckpt} is for step {state.get('global_step')}, not "
                          f"{loaded_step}: ignored{RESET}", "warning")
            state = {}
        self._resume_state = state or {}
        try:
            self._nonfinite_steps = int(self._resume_state.get("skipped_updates", 0) or 0)
        except (TypeError, ValueError):
            self._nonfinite_steps = 0
        if self._nonfinite_steps:
            self.main_log(f"[resume] {self._nonfinite_steps} updates were skipped (non-finite gradients) before step "
                          f"{loaded_step}; train/skipped_updates continues from there")

        self.next_train_data_for_check, self._resume_check_note = None, None
        saved_ws, now_ws = self._resume_state.get("world_size"), get_world_size()
        if isinstance(saved_ws, int) and saved_ws != now_ws:
            # the global sample order does not depend on the world size (rank r takes every W-th sample of one
            # order) and DCP reshards the model and optimizer state: only the per-rank comparison cannot be made
            self.main_log(f"{RED}[resume] {ckpt} was saved by {saved_ws} ranks and this run has {now_ws}: the per-rank "
                          f"next-batch sanity check is skipped (the data order is the same at any world size){RESET}",
                          "warning")
            self.resume_sanity_check_done = True
        else:
            nxt_path = Path(ckpt) / f"next_train_data_{get_rank()}.pt"
            try:
                self.next_train_data_for_check = torch.load(nxt_path, map_location="cpu", weights_only=True)
            except FileNotFoundError:
                self._resume_check_note = f"{nxt_path.name} is missing"
            except Exception as e:  # noqa: BLE001 (EOFError, UnpicklingError, OSError, RuntimeError, ...): skip the check
                self._resume_check_note = f"{nxt_path.name} cannot be read ({type(e).__name__}: {e})"
            self.resume_sanity_check_done = False
        dist_barrier()
        return Path(ckpt)

    def _replay_warning(self, step: int) -> None:
        """Master: a resume at a step that <run>.metrics.jsonl has already logged (a crash resumed from an older
        periodic checkpoint) replays steps step..L. A W&B run that resumes the same id drops every step below the one it
        has seen, so its curve keeps the abandoned segment's values for them; metrics.jsonl keeps both segments (the
        records after the last resume event are the canonical ones)."""
        if not is_master():
            return
        last = last_logged_step(getattr(self, "_metrics_path", None))
        if last is not None and step <= last:
            self.main_log(f"{RED}[resume] resuming at step {step}, but steps up to {last} were already logged: steps "
                          f"{step}..{last} are replayed. A W&B run resumed under the same id drops the replayed steps "
                          f"(its curve keeps the earlier values for them); {getattr(self.cfg, 'run_name', 'the run')}"
                          f".metrics.jsonl keeps both segments (the records after the last resume event count){RESET}",
                          "warning")

    def jump_to_step(self, step: int) -> None:
        # batches/epoch is iteration-independent (sampler.num_samples is the full per-rank
        # count); the sampler's offset is in PER-RANK SAMPLES, so multiply batches by
        # the per-rank batch_size: one optimizer step == batch_size samples per rank.
        bs = self.train_loader.batch_size
        n_batches = max(1, self.train_loader.sampler.num_samples // bs)
        epoch = step // n_batches
        iter_in_epoch = step - epoch * n_batches
        self.global_step = step
        self.epoch = epoch
        self.train_loader.sampler.set_epoch(epoch, iter_in_epoch * bs)
        self.train_iter = Peekable(self.train_loader)
        self.main_log(f"{CYAN}Resumed: global_step={step}, epoch={epoch}, iter_in_epoch={iter_in_epoch} "
                      f"(of {n_batches} batches/epoch){RESET}", "critical")
        self._metrics_event("resume", epoch=epoch, iter_in_epoch=iter_in_epoch)

    def resume_routine(self) -> None:
        """Resume the newest complete checkpoint (the master's decision), after checking the run identity against it
        (before any state is read). The identity record is written once the state has loaded on every rank, so a load
        that fails leaves it as it was; a run that starts from scratch records its identity at once."""
        if not self.resume:
            self.main_log("resume disabled: training from scratch", "info")
            self._check_run_identity(None)
            self._record_run_identity()
            return
        ckpt = self._resume_decision(self.checkpoint_dir)
        self._check_run_identity(None if ckpt is None else checkpoint_step(ckpt))
        if ckpt is None:
            self._record_run_identity()
            self.main_log(f"{RED}No complete checkpoint under {self.checkpoint_dir}; training from scratch{RESET}")
            return
        found = self._load_checkpoint(ckpt)
        self._record_run_identity()
        self.main_log(f"{CYAN}Resumed from {found} at step {self.global_step}{RESET}", "critical")

    def _check_run_identity(self, resumed_step: Optional[int]) -> None:
        """Check <ckpt_dir>/run_identity.json on the master for a resume, or prepare the record of a start from scratch
        (resumed_step None) (utils/run_identity.py: plan_run_identity). The record to write, if any, stays on the master
        for _record_run_identity. A refusal is raised on every rank: ResumeRefused when the identity differs
        (pretrain.train exits 2), a RuntimeError when the record cannot be read. An unexpected error of the check itself
        is a warning, not a stop: the guard must not become a way to break every chain."""
        allow = os.environ.get(ALLOW_ENV, "0") == "1"
        self._identity_record = None

        def check():
            try:
                status, msgs, doc = plan_run_identity(self.checkpoint_dir, self.cfg, resumed_step, allow)
            except Exception as e:  # noqa: BLE001
                return IDENTITY_OK, [("warning", f"[identity] the run identity could not be checked "
                                                 f"({type(e).__name__}: {e}); continuing without the check")]
            self._identity_record = doc          # master only; never broadcast
            return status, msgs

        status, msgs = self._on_master(check, "checking the run identity")
        for level, msg in msgs:
            self.main_log(f"{RED}{msg}{RESET}" if level != "info" else msg, level)
        if status != IDENTITY_OK:
            self._identity_record = None
            raise (ResumeRefused if status == IDENTITY_REFUSED else RuntimeError)(msgs[-1][1])

    def _record_run_identity(self) -> None:
        """Master: write the run_identity.json that _check_run_identity prepared (none when nothing changed). A failure
        to write it is a warning only; never raises."""
        doc, self._identity_record = getattr(self, "_identity_record", None), None
        if doc is None or not is_master():
            return
        try:
            warning = record_run_identity(self.checkpoint_dir, doc)
        except Exception as e:  # noqa: BLE001 (a record that cannot be written is recorded by the next resume)
            warning = f"[identity] could not write the run identity ({type(e).__name__}: {e})"
        if warning:
            self.main_log(f"{RED}{warning}{RESET}", "warning")

    def _resume_sanity_check(self, feed: dict) -> None:
        """The first batch after a resume against the one the checkpoint stored for this rank. Every rank's verdict is
        gathered, so all ranks pass, warn or raise alike (one rank raising alone would leave the others waiting in the
        step's collectives). A rank whose next_train_data_<rank>.pt is missing or could not be read skips the
        comparison with a warning: the data position comes from global_step, the file only feeds this check."""
        ref = self.next_train_data_for_check
        note = getattr(self, "_resume_check_note", None)
        if note is not None or ref is None:
            mine = ("skipped", note or "no saved next batch")
        elif torch.equal(feed["input_ids"].cpu(), ref["input_ids"].cpu()):
            mine = ("ok", "")
        else:
            mine = ("failed", "the next batch differs from the one saved with the checkpoint")
        verdicts = all_gather_objects(mine)
        failed = [f"rank {r}: {why}" for r, (kind, why) in enumerate(verdicts) if kind == "failed"]
        skipped = [f"rank {r}: {why}" for r, (kind, why) in enumerate(verdicts) if kind == "skipped"]
        self.resume_sanity_check_done = True
        self.next_train_data_for_check = None
        if failed:
            msg = "Resume sanity check FAILED (data stream mismatch after resume): " + "; ".join(failed)
            if getattr(self.cfg, "force_resume_check", False):
                raise RuntimeError(msg)
            self.main_log(f"{RED}{msg}{RESET}", "warning")
        elif skipped:
            ok = len(verdicts) - len(skipped)
            self.main_log(f"{RED}Resume sanity check SKIPPED on {len(skipped)} rank(s) (" + "; ".join(skipped[:4])
                          + f"); the data position comes from global_step. {ok} rank(s) PASSED{RESET}", "warning")
        else:
            self.main_log("Resume sanity check PASSED (next batch matches)", "info")

    # ----------------------------------------------------------- logging
    def log_metrics(self, metrics: dict, step: int) -> None:
        if not is_master():
            return
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=step)
        self.main_log(f"[step {step}] " + pformat({k: round(v, 5) if isinstance(v, float) else v
                                                   for k, v in metrics.items()}, width=120))
        self._metrics_event("metrics", step=step, **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()})

    def _metrics_event(self, event: str, **fields) -> None:
        """Append one JSON record to <log_root_dir>/<run_name>.metrics.jsonl (master only; never raises). Events:
        start, metrics (every logging_steps), checkpoint, resume, stop, export, complete. A resume event marks a
        replayed segment: for the canonical trajectory keep, per step, the record after the LAST resume."""
        if self._metrics_path is None:
            return
        rec = {"event": event, "step": self.global_step, "time": round(time.time(), 3),
               "iso": time.strftime("%Y-%m-%dT%H:%M:%S")}
        rec.update(fields)
        try:
            with open(self._metrics_path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except OSError as e:
            self.main_log(f"[metrics] append failed: {e}", "warning")

    def _on_stop_signal(self, signum, frame) -> None:
        if self._stop_requested is None:
            self._stop_requested = f"signal {signum}"

    def _stop_now(self) -> Optional[str]:
        """The stop decision, agreed across ranks: a signal on any rank, or the master's deadline."""
        reason = self._stop_requested
        if reason is None and self._deadline and is_master() and time.time() >= self._deadline:
            reason = "deadline"
        if _is_dist():
            t = torch.tensor([0 if reason is None else 1], device=self.device, dtype=torch.int32)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            if int(t.item()) and reason is None:
                reason = "peer"
        return reason

    # ----------------------------------------------------------- main loop
    def train(self) -> str:
        """Returns "complete" (final checkpoint + export written) or "stopped" (a preemption-safe early stop: the
        newest complete checkpoint holds the state at the returned step; the caller exits 75 so a relaunch resumes).
        A stop already pending before the first step returns "stopped" at once, without training or saving (the
        checkpoint just resumed holds this state; a fresh run has nothing to save); a run resumed at max_steps skips
        that check and exports. The W&B run stays open: the caller closes it after the process group (pretrain.train's
        _close_and_exit; marked preempting when "stopped")."""
        total_tokens = self.max_steps * self.global_batch_size * self.chunk_size
        self.main_log(
            f"Start training: backend={self.backend} step={self.global_step}/{self.max_steps} "
            f"global_batch={self.global_batch_size} grad_accum={self.gradient_accumulation_steps} "
            f"chunk={self.chunk_size} total_tokens={total_tokens / 1e9:.3f}B", "critical")
        self._metrics_event("start", max_steps=self.max_steps, world=self.num_devices, global_batch=self.global_batch_size,
                            chunk=self.chunk_size, deadline=(self._deadline or None), resumed=self.global_step > 0,
                            kernel=(kernel_identity() or {}).get("code_sha"), code=os.environ.get("DM_GIT_REV"))
        self._train_t0 = time.time()
        if self.global_step < self.max_steps:
            # a SIGTERM during the startup or the checkpoint load, or a time budget already spent (a run started late
            # in a job): one agreed check before the first step, whose compile + autotune alone can take minutes
            reason = self._stop_now()
            if reason is not None:
                held = (f"the checkpoint it resumed from (step {self.global_step}) holds this state"
                        if self.global_step > 0 else "nothing was trained, so there is nothing to save")
                self.main_log(f"{RED}Stopping early ({reason}) at step {self.global_step}, before the first step: "
                              f"{held}; relaunch to resume{RESET}", "critical")
                self._metrics_event("stop", reason=reason, before_first_step=True)
                return "stopped"
        with tqdm(total=self.max_steps, initial=self.global_step, desc=f"train[{self.backend}]",
                  disable=not is_master()) as bar:
            while self.global_step < self.max_steps:
                metrics = self.train_step(grad_clip=self.cfg.opt_cfg.max_grad_norm)
                self._trained_steps = getattr(self, "_trained_steps", 0) + 1
                self._log_gains()
                self._last_metrics = metrics
                if self.global_step % self.cfg.logging_steps == 0:
                    self.log_metrics(metrics, self.global_step)
                self.global_step += 1
                self._set_train_state_step(self.global_step)
                if (self.cfg.hf_save_dir and self.global_step in self._hf_export_steps
                        and self.global_step < self.max_steps):
                    # token-budget ladder snapshot (collective; all ranks gather) -> {hf_save_dir}/step{N}
                    self.save_hf(os.path.join(str(self.cfg.hf_save_dir), f"step{self.global_step}"))
                saved = False
                if self.global_step % self.cfg.save_interval == 0 and self.global_step < self.max_steps:
                    self.save_ckpt()
                    saved = True
                reason = self._stop_now()
                if reason is not None and self.global_step < self.max_steps:
                    if not saved:
                        self.save_ckpt()
                    self.main_log(f"{RED}Stopping early ({reason}) at step {self.global_step}: complete checkpoint saved; "
                                  f"relaunch to resume{RESET}", "critical")
                    self._metrics_event("stop", reason=reason)
                    return "stopped"
                dist_barrier()
                bar.update()
        final_dir = self.checkpoint_dir / CKPT_NAME_FORMAT.format(step=self.global_step)
        # decided on the master: ranks that disagreed here would split between save_ckpt's and save_hf's collectives
        state = self._on_master(lambda: marker_state_why(final_dir, self.cfg.VERSION_STRING)[0],
                                f"checking the final checkpoint {final_dir}")
        if state == COMPLETE:
            self.main_log(f"Final checkpoint {final_dir} already complete (resumed into a finished run); not re-saving")
        else:
            if state == UNREADABLE:
                self.main_log(f"{RED}the marker of the final checkpoint {final_dir} cannot be read: saving it again{RESET}",
                              "warning")
            self.save_ckpt()  # final
        if self.cfg.hf_save_dir:
            self.save_hf(self.cfg.hf_save_dir)
            self._metrics_event("export", path=str(self.cfg.hf_save_dir))
        self.main_log(f"Training complete at step {self.global_step}", "critical")
        self._metrics_event("complete")
        self._dump_result_summary()
        return "complete"

    def _dump_result_summary(self) -> None:
        """Durable per-run record (master-only, never crashes the run): write a curated
        train_summary.json under <result_root_dir or dirname(log_root_dir)/result>/{run_name}/ (atomically).
        final_metrics are the last logged step's; a process that trained no step (it resumed into a finished run and
        only exported) keeps the run's previous ones: the resumed checkpoint's sidecar, else the existing summary's."""
        if not is_master():
            return
        try:
            result_root = self.cfg.result_root_dir or os.path.join(
                os.path.dirname(str(self.cfg.log_root_dir)), "result")
            result_dir = os.path.join(result_root, self.cfg.run_name)
            os.makedirs(result_dir, exist_ok=True)
            out_path = os.path.join(result_dir, "train_summary.json")
            final_metrics = self._last_metrics
            if not getattr(self, "_trained_steps", 0):
                final_metrics = (getattr(self, "_resume_state", None) or {}).get("last_metrics") or {}
                if not final_metrics:
                    try:
                        with open(out_path) as fh:
                            final_metrics = (json.load(fh) or {}).get("final_metrics") or {}
                    except (OSError, ValueError, AttributeError):
                        final_metrics = {}
            wandb_run_id = str(self.wandb_run.id) if self.wandb_run is not None else None
            summary = {
                "run_name": self.cfg.run_name,
                "wandb_run_id": wandb_run_id,
                "global_step": self.global_step,
                "num_steps": self.cfg.num_steps,
                "final_metrics": final_metrics,
                "skipped_updates": int(getattr(self, "_nonfinite_steps", 0) or 0),
                "hf_save_dir": self.cfg.hf_save_dir,
                "ckpt_dir": str(self.cfg.ckpt_dir),
                "config": dataclasses.asdict(self.cfg),
            }
            write_text_atomic(out_path, json.dumps(summary, indent=2, default=str))
            self.main_log(f"Wrote result summary -> {out_path}", "critical")
        except Exception as e:  # never crash the run on a result-dump failure
            self.main_log(f"[result] could not write train_summary.json: {e}", "warning")

    def save_hf(self, out_dir, tokenizer=None) -> None:
        """Export a HuggingFace checkpoint (config + safetensors) for the eval pipeline.
        Collective: every rank participates in the DTensor gather; master writes."""
        full_sd = self._gather_full_state_dict()  # CPU full state dict on master, None elsewhere
        if is_master():
            from ..hf.export import save_hf_checkpoint
            if tokenizer is None and self.cfg.tokenizer_dir:
                # bake the tokenizer into the export so meta_eval (recall/commonsense/longbench) can load it
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer_dir, local_files_only=True)
                except Exception as e:
                    self.main_log(f"[hf] could not load tokenizer_dir={self.cfg.tokenizer_dir}: {e}", "warning")
            save_hf_checkpoint(full_sd, self.cfg.model, str(out_dir), tokenizer=tokenizer)
            self.main_log(f"Exported HF checkpoint -> {out_dir}", "critical")
            if self.cfg.wandb and self.wandb_run is not None:
                # let the eval link to the SAME wandb run
                try:
                    with open(Path(out_dir) / "wandb_run_id.txt", "w") as fh:
                        fh.write(str(self.wandb_run.id))
                except Exception as e:
                    self.main_log(f"[wandb] could not write wandb_run_id.txt: {e}", "warning")
        dist_barrier()

    def _set_train_state_step(self, step: int) -> None:
        """FSDP2 mirrors global_step into the DCP TrainState; DDP is a no-op."""
