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
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import shutil
import signal
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
    is_master, dist_barrier, get_rank, get_world_size, sync_tensor,
)
from ..parallel.dist import is_dist as _is_dist
from ..parallel.wrap import build_optimizer, build_lr_scheduler, fp8_fsdp_all_gather_enabled
from ..utils.misc import Peekable, get_amp_dtype
from ..utils.ckpt import (
    CKPT_NAME_FORMAT, checkpoint_step,
    find_last_complete_checkpoint, check_complete_marker, create_complete_marker_atomic_swap,
)
from ..utils.logging import RED, CYAN, RESET


def export_key(key: str) -> str:
    """A training state-dict key -> the plain module name the HF export uses: drops the activation-checkpoint
    wrapper and the torch.compile (OptimizedModule) levels."""
    return key.replace("._checkpoint_wrapped_module", "").replace("._orig_mod", "")


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
        self._save_resume_state(save_dir)            # backend: DCP / torch.save
        self._save_full_model(save_dir)              # HF-loadable bin + config (master)
        self._save_dataloader_state(save_dir)        # per-rank next batch (resume sanity check)
        dist_barrier()
        if is_master():
            create_complete_marker_atomic_swap(save_dir, version_str=self.cfg.VERSION_STRING)
            # rotate only once the new checkpoint is complete, so an interrupted save never costs an old one
            self._rotate_checkpoints(base)
        dist_barrier()
        self._metrics_event("checkpoint", path=str(save_dir), seconds=round(time.time() - t0, 1))
        return save_dir

    def _rotate_checkpoints(self, base: Path) -> None:
        """Keep the newest `save_total_limit` COMPLETE checkpoints under `base`, the one just saved included
        (None or 0 keeps all). Runs on the master after the new checkpoint's marker is written. A partial dir (no
        valid marker: a save that was killed) older than the newest complete one is removed; a partial dir never
        displaces a complete one. Dirs whose name carries no step number are left alone."""
        limit = self.cfg.save_total_limit
        dirs = sorted((p for p in Path(base).glob("checkpoint-*") if p.is_dir() and checkpoint_step(p) >= 0),
                      key=checkpoint_step)
        complete = [p for p in dirs if check_complete_marker(p, self.cfg.VERSION_STRING)]
        newest = checkpoint_step(complete[-1]) if complete else -1
        for p in dirs:
            if p not in complete and checkpoint_step(p) < newest:
                self.main_log(f"Removing partial checkpoint {p} (older than the newest complete one)", "warning")
                shutil.rmtree(p, ignore_errors=True)
        if limit:
            for p in complete[:-limit]:
                shutil.rmtree(p, ignore_errors=True)

    def _save_full_model(self, save_dir: Path) -> None:
        full_sd = self._gather_full_state_dict()
        if is_master():
            torch.save(full_sd, save_dir / "pytorch_model.bin")
            self.unwrapped_model.save_config(save_dir)
        dist_barrier()

    def _save_dataloader_state(self, save_dir: Path) -> None:
        # per-rank next batch for the resume sanity check (the data position itself is re-derived from global_step)
        nxt = self.peek_next_train_data()
        torch.save({"input_ids": nxt["input_ids"]}, save_dir / f"next_train_data_{get_rank()}.pt")
        dist_barrier()

    def load(self, load_dir: Path, global_step: int = -1) -> Optional[Path]:
        if global_step == -1:
            ckpt = find_last_complete_checkpoint(load_dir, version_str=self.cfg.VERSION_STRING)
        else:
            ckpt = Path(load_dir) / CKPT_NAME_FORMAT.format(step=global_step)
            if not check_complete_marker(ckpt, self.cfg.VERSION_STRING):
                ckpt = None
        if ckpt is None:
            return None

        try:
            loaded_step = self._load_resume_state(ckpt)
        except BaseException as e:  # noqa
            self.main_log(f"{RED}Failed to load resume state from {ckpt}: {e}{RESET}\n{traceback.format_exc()}",
                          "critical")
            raise

        # global_step is the COMPLETED-step count; the checkpoint saved after step
        # `loaded_step` completed, so we continue *at* loaded_step (its batch is the
        # one stored in next_train_data_{rank}.pt for the sanity check).
        self.jump_to_step(loaded_step)

        nxt_path = ckpt / f"next_train_data_{get_rank()}.pt"
        if nxt_path.exists():
            self.next_train_data_for_check = torch.load(nxt_path, map_location="cpu", weights_only=True)
        self.resume_sanity_check_done = False
        dist_barrier()
        return ckpt

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
        if not self.resume:
            self.main_log("resume disabled: training from scratch", "info")
            return
        found = self.load(self.checkpoint_dir)
        if found is None:
            self.main_log(f"{RED}No complete checkpoint under {self.checkpoint_dir}; training from scratch{RESET}")
        else:
            self.main_log(f"{CYAN}Resumed from {found} at step {self.global_step}{RESET}", "critical")

    def _resume_sanity_check(self, feed: dict) -> None:
        ref = self.next_train_data_for_check
        ok = ref is not None and torch.equal(feed["input_ids"].cpu(), ref["input_ids"].cpu())
        if ok:
            self.main_log("Resume sanity check PASSED (next batch matches)", "info")
        else:
            msg = "Resume sanity check FAILED (data stream mismatch after resume)"
            if getattr(self.cfg, "force_resume_check", False):
                raise RuntimeError(msg)
            self.main_log(f"{RED}{msg}{RESET}", "warning")
        self.resume_sanity_check_done = True
        self.next_train_data_for_check = None

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
        """Returns "complete" (final checkpoint + export written) or "stopped" (a preemption-safe early stop:
        a complete checkpoint was saved at the returned step; the caller exits 75 so a relaunch resumes)."""
        total_tokens = self.max_steps * self.global_batch_size * self.chunk_size
        self.main_log(
            f"Start training: backend={self.backend} step={self.global_step}/{self.max_steps} "
            f"global_batch={self.global_batch_size} grad_accum={self.gradient_accumulation_steps} "
            f"chunk={self.chunk_size} total_tokens={total_tokens / 1e9:.3f}B", "critical")
        self._metrics_event("start", max_steps=self.max_steps, world=self.num_devices, global_batch=self.global_batch_size,
                            chunk=self.chunk_size, deadline=(self._deadline or None), resumed=self.global_step > 0)
        with tqdm(total=self.max_steps, initial=self.global_step, desc=f"train[{self.backend}]",
                  disable=not is_master()) as bar:
            while self.global_step < self.max_steps:
                metrics = self.train_step(grad_clip=self.cfg.opt_cfg.max_grad_norm)
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
                    self._finish_wandb()
                    return "stopped"
                dist_barrier()
                bar.update()
        final_dir = self.checkpoint_dir / CKPT_NAME_FORMAT.format(step=self.global_step)
        if check_complete_marker(final_dir, self.cfg.VERSION_STRING):
            self.main_log(f"Final checkpoint {final_dir} already complete (resumed into a finished run); not re-saving")
        else:
            self.save_ckpt()  # final
        if self.cfg.hf_save_dir:
            self.save_hf(self.cfg.hf_save_dir)
            self._metrics_event("export", path=str(self.cfg.hf_save_dir))
        self.main_log(f"Training complete at step {self.global_step}", "critical")
        self._metrics_event("complete")
        self._dump_result_summary()
        self._finish_wandb()
        return "complete"

    def _finish_wandb(self) -> None:
        if is_master() and self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception as e:  # never crash on a wandb teardown
                self.main_log(f"[wandb] finish failed: {e}", "warning")

    def _dump_result_summary(self) -> None:
        """Durable per-run record (master-only, never crashes the run): write a curated
        train_summary.json under <result_root_dir or dirname(log_root_dir)/result>/{run_name}/."""
        if not is_master():
            return
        try:
            result_root = self.cfg.result_root_dir or os.path.join(
                os.path.dirname(str(self.cfg.log_root_dir)), "result")
            result_dir = os.path.join(result_root, self.cfg.run_name)
            os.makedirs(result_dir, exist_ok=True)
            wandb_run_id = str(self.wandb_run.id) if self.wandb_run is not None else None
            summary = {
                "run_name": self.cfg.run_name,
                "wandb_run_id": wandb_run_id,
                "global_step": self.global_step,
                "num_steps": self.cfg.num_steps,
                "final_metrics": self._last_metrics,
                "hf_save_dir": self.cfg.hf_save_dir,
                "ckpt_dir": str(self.cfg.ckpt_dir),
                "config": dataclasses.asdict(self.cfg),
            }
            out_path = os.path.join(result_dir, "train_summary.json")
            with open(out_path, "w") as fh:
                json.dump(summary, fh, indent=2, default=str)
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
