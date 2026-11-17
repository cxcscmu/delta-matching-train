"""DDP backend. Parameters/optimizer state are fully replicated, so each rank saves
and loads its own `training_state_{rank}.pt`: correct (replicas are identical) and
safe on node-local /tmp (no shared FS needed). For large models prefer FSDP2 (sharded)
to avoid the N-replica checkpoint cost. Grad sync is gated to the last micro-batch by
toggling DDP's require_backward_grad_sync.
A resume on another world size: on fewer ranks the extra files are not read; on more ranks, a rank without a file of
its own loads training_state_0.pt (the replicas are identical: the same all-reduced gradients, the same deterministic
optimizer step, Muon included, which under DDP runs on whole matrices on every rank).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
from torch.nn.parallel import DistributedDataParallel

from .base_trainer import BaseTrainer, export_key
from ..parallel.dist import is_master, get_rank, dist_barrier
from ..utils.ckpt import torch_save_durable


class DDPTrainer(BaseTrainer):
    backend = "ddp"

    def _set_grad_sync(self, is_last_micro: bool) -> None:
        if isinstance(self.model, DistributedDataParallel):
            # only all-reduce grads on the last micro-batch of the accumulation window
            self.model.require_backward_grad_sync = is_last_micro

    def _save_resume_state(self, save_dir: Path) -> None:
        payload = {
            "model": self.unwrapped_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
        }
        # fsynced: the checkpoint's .complete marker is written only once every file of it is on disk
        torch_save_durable(payload, save_dir / f"training_state_{get_rank()}.pt")
        dist_barrier()

    def _load_resume_state(self, load_dir: Path) -> int:
        load_dir = Path(load_dir)
        path = load_dir / f"training_state_{get_rank()}.pt"
        if get_rank() > 0 and not path.exists():
            # saved by fewer ranks than this run has (a world-size increase): every replica is the same, so rank 0's
            # file holds this rank's state too (no file at all still fails below, on the path this rank wanted)
            alt = load_dir / "training_state_0.pt"
            if alt.exists():
                print(f"[rank {get_rank()}] {path.name} is not in {load_dir} (saved by fewer ranks): loading {alt.name}, "
                      "the replicas are identical", file=sys.stderr, flush=True)
                path = alt
        # tensors, numbers and strings only (torchao registers its fp8 weight wrapper as a safe global itself)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        self.unwrapped_model.load_state_dict(payload["model"])
        # load_state_dict moves the optimizer state onto each parameter's device itself, and keeps Adam's `step`
        # counters on the CPU as a fresh run does (a CUDA `step` would cost a host sync per parameter per step)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.lr_scheduler.load_state_dict(payload["lr_scheduler"])
        return int(payload["global_step"])

    def _gather_full_state_dict(self) -> Optional[dict]:
        if not is_master():
            return None
        # plain module names: `compile: true` wraps each block in an OptimizedModule (keys carry `._orig_mod`)
        return {export_key(k): v.detach().to(self.amp_dtype).cpu() for k, v in self.unwrapped_model.state_dict().items()}
