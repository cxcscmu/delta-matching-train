# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# Modified by the delta-matching-train authors, 2026.

import os
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed

__all__ = [
    "dist_init",
    "dist_close",
    "dist_barrier",
    "get_dist_rank",
    "get_dist_size",
    "is_master",
    "dist_print",
]


def _get_dist_local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def dist_close() -> None:
    dist_barrier()
    torch.distributed.destroy_process_group()


def dist_init(gpu: Optional[str] = None, cudnn_benchmark: bool = False, timeout: int = 3600,
              device_index: Optional[int] = None) -> None:
    """Initialize torch.distributed (NCCL) and bind this rank's CUDA device.

    `cudnn_benchmark` sets three cuDNN switches at once: torch.backends.cudnn.benchmark = cudnn_benchmark,
    torch.backends.cudnn.enabled = cudnn_benchmark and torch.backends.cudnn.deterministic = not cudnn_benchmark.
    With the default False, which every evaluation entry point uses (meta_eval, memmap_ppl, pg19_ppl), cuDNN is
    switched off for the whole process. The published evaluation numbers were produced this way: keep it.

    `gpu` (optional) overrides CUDA_VISIBLE_DEVICES; `timeout` is the NCCL collective timeout in seconds;
    `device_index` is the device_map (sharded model) path, see below.
    """
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.enabled = cudnn_benchmark
    torch.backends.cudnn.deterministic = not cudnn_benchmark

    if device_index is not None:
        # Sharded (device_map) path: each rank's CUDA_VISIBLE_DEVICES is already pinned to
        # its own GPU slice by the launch wrapper, so bind the current device to the FIRST
        # visible GPU and pass it as device_id BEFORE init so NCCL does not guess
        # device-id == global-rank (wrong under per-rank CUDA_VISIBLE_DEVICES; can hang).
        # device_index (not LOCAL_RANK) is used because LOCAL_RANK can exceed this rank's
        # per-replica GPU count once the model spans several GPUs.
        torch.cuda.set_device(device_index)
        torch.distributed.init_process_group(
            backend="nccl", timeout=timedelta(seconds=timeout),
            device_id=torch.device("cuda", device_index))
    else:
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout))
        torch.cuda.set_device(_get_dist_local_rank())
    assert torch.distributed.is_initialized()


def get_dist_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_dist_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_master() -> bool:
    return get_dist_rank() == 0


def dist_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def dist_print(*args, rank: int = 0) -> None:
    """print(*args) on one rank (default rank 0). `rank` is keyword-only, so a second positional argument is
    printed, never taken as the rank."""
    if get_dist_rank() == rank:
        print(*args)

