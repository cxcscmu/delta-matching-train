"""torch Distributed-Checkpoint `Stateful` wrappers.

Used by the FSDP2 trainer to dcp.save/load a single payload of
{train_state, model, optimizer, lr_scheduler} where model/optimizer are sharded
DTensors. The model load is non-strict (tolerates tied-weight / buffer key drift);
the optimizer state is flattened so it keys by parameter FQN (matches the sharded
params on resume).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict, set_model_state_dict,
    get_optimizer_state_dict, set_optimizer_state_dict,
    StateDictOptions,
)


@dataclass
class TrainState(Stateful):
    global_step: int = 0

    def state_dict(self) -> dict:
        return {"global_step": self.global_step}

    def load_state_dict(self, state_dict: dict) -> None:
        self.global_step = int(state_dict["global_step"])


class ModelWrapper(Stateful):
    def __init__(self, model: nn.Module):
        self.model = model

    def state_dict(self) -> dict:
        return get_model_state_dict(self.model)

    def load_state_dict(self, state_dict: dict) -> None:
        set_model_state_dict(self.model, state_dict, options=StateDictOptions(strict=False))


class OptimizerWrapper(Stateful):
    def __init__(self, model: nn.Module, optim):
        self.model = model
        self.optim = optim

    def state_dict(self) -> dict:
        return get_optimizer_state_dict(
            self.model, self.optim, options=StateDictOptions(flatten_optimizer_state_dict=True))

    def load_state_dict(self, state_dict: dict) -> None:
        set_optimizer_state_dict(
            self.model, self.optim, optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True))
