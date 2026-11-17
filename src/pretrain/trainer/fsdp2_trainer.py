"""FSDP2 backend. Sharded resume state via torch Distributed Checkpoint (DCP);
the HF-loadable pytorch_model.bin is produced by gathering each DTensor with
.full_tensor() on the master rank. Grad sync is gated to the last micro-batch via
FSDP2's set_requires_gradient_sync.

DCP's own coordination (the save/load plans: gather_object / scatter_object / all_reduce of small objects) runs over a
CPU (gloo) process group, created once on every rank at the first save or load (both are collective). Over NCCL those
object collectives allocate device buffers, and on a nearly full GPU (the 5.3 B model peaks at ~74 of 80 GB) NCCL's own
cudaMalloc fails: "NCCL Error 1: unhandled cuda error" in gather_object, rank 0 dies and the others hang. The cached
blocks of the training step are released before the save for the same reason. The tensors themselves are written by
each rank from its own shards, as before.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor

from .base_trainer import BaseTrainer, export_key
from .stateful import TrainState, ModelWrapper, OptimizerWrapper
from ..parallel.dist import is_master, dist_barrier
from ..utils.misc import str_to_dir_format


class FSDP2Trainer(BaseTrainer):
    backend = "fsdp2"

    def _build_states(self) -> None:
        self.train_state = TrainState(global_step=0)
        self.states = {
            "train_state": self.train_state,
            "model": ModelWrapper(self.model),
            "optimizer": OptimizerWrapper(self.model, self.optimizer),
            "lr_scheduler": self.lr_scheduler,
        }

    def _set_grad_sync(self, is_last_micro: bool) -> None:
        self.model.set_requires_gradient_sync(is_last_micro)

    def _set_train_state_step(self, step: int) -> None:
        self.train_state.global_step = step

    def _dcp_group(self):
        """The gloo group of DCP's coordination (module docstring); None (the default group) without a process group."""
        g = getattr(self, "_dcp_pg", None)
        if g is None and torch.distributed.is_available() and torch.distributed.is_initialized():
            g = self._dcp_pg = torch.distributed.new_group(backend="gloo")
        return g

    def _save_resume_state(self, save_dir: Path) -> None:
        dist_barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()   # the training step's cached blocks: room for the save's own buffers
        dcp.save(self.states, checkpoint_id=str_to_dir_format(save_dir), process_group=self._dcp_group())

    def _load_resume_state(self, load_dir: Path) -> int:
        dcp.load(self.states, checkpoint_id=str_to_dir_format(load_dir), process_group=self._dcp_group())
        return int(self.train_state.global_step)

    def _gather_full_state_dict(self) -> Optional[dict]:
        sd = self.model.state_dict()
        full: dict = {}
        for k, v in sd.items():
            t = v.full_tensor() if isinstance(v, DTensor) else v   # collective: all ranks call
            # On a 1-rank mesh full_tensor() hands back torchao's FSDP2 fp8
            # all-gather wrapper (WeightWithDynamicFloat8CastTensor, DM_FP8_FSDP_AG=1) instead of a
            # plain tensor, and safetensors cannot take its storage pointer ("invalid python
            # storage" at export). Unwrap it; at world > 1 the real all-gather already yields a
            # plain tensor, so this is inert there.
            if type(t).__name__ == "WeightWithDynamicFloat8CastTensor" and getattr(t, "_tensor", None) is not None:
                t = t._tensor
            if is_master():
                full[export_key(k)] = t.detach().to(self.amp_dtype).cpu()
            del t
        return full if is_master() else None
