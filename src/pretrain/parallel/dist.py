"""Distributed bootstrap + collectives.

Backend = nccl on CUDA (gloo CPU fallback). Rank/world/master are read from the
environment by `init_process_group` (the torchrun/`env://` convention). For a bare
`python train.py` (no torchrun) we synthesize a 1-process world so single-GPU runs
and unit tests work without a launcher.
"""
from __future__ import annotations

import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist


def _free_port() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return str(port)


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_master() -> bool:
    return get_rank() == 0


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def dist_init(timeout_s: int = 1800) -> torch.device:
    """Init the default process group and bind this rank's CUDA device.

    Synthesizes single-process env vars when launched without torchrun so the same
    code path works for `python ...` and `torchrun ...`. Returns the local device.
    """
    if "RANK" not in os.environ:  # bare process -> single-rank world
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", _free_port())

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_s))
    if use_cuda:
        torch.cuda.set_device(get_local_rank())
        device = torch.device("cuda", get_local_rank())
    else:
        device = torch.device("cpu")
    return device


def dist_barrier() -> None:
    if is_dist():
        dist.barrier()


def dist_close() -> None:
    if is_dist():
        dist_barrier()
        dist.destroy_process_group()


def sync_tensor(value, reduce: str = "mean", dim: int = 0):
    """All-gather a (scalar or tensor) `value` across ranks and reduce.

    reduce in {"mean","sum","cat","stack","root","list"}. Used for logging-metric
    reduction. Scalars are wrapped to a CUDA/CPU tensor first. With world_size==1
    this is a no-op passthrough.
    """
    if not is_dist() or get_world_size() == 1:
        return value
    device = torch.device("cuda", get_local_rank()) if torch.cuda.is_available() else torch.device("cpu")
    t = value if torch.is_tensor(value) else torch.tensor(value, device=device)
    t = t.to(device)
    gathered = [torch.zeros_like(t) for _ in range(get_world_size())]
    dist.all_gather(gathered, t)
    if reduce == "mean":
        return torch.stack(gathered, dim=0).float().mean(dim=0)
    if reduce == "sum":
        return torch.stack(gathered, dim=0).sum(dim=0)
    if reduce == "cat":
        return torch.cat(gathered, dim=dim)
    if reduce == "stack":
        return torch.stack(gathered, dim=dim)
    if reduce == "root":
        return gathered[0]
    return gathered
