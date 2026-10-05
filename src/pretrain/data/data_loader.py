"""build_train_dataloader: data_cfg -> DistDataLoader.

The returned loader yields, per __next__, one *optimizer step's* worth of data for
this rank: `input_ids` of shape [global_batch_size // world_size, chunk_size]. The
trainer micro-chunks that into gradient-accumulation steps, and relies on
loader.batch_size (= global // world), loader.sampler and loader.pad_token_id.
"""
from __future__ import annotations

import atexit
import functools
import multiprocessing
import os
import signal

import torch
from torch.utils.data import DataLoader

from ..parallel.dist import get_world_size, get_rank
from .memmap_dataset import MemmapTokenDataset
from .doc_dataset import DocAwareMemmapDataset
from .dist_sampler import ResumableDistSampler


def _doc_collate(batch):
    """Stack the fixed-shape fields; keep ragged per-sample `cu_seqlens` as a list
    (the trainer consumes input_ids/position_ids; cu_seqlens rides along for varlen)."""
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "position_ids": torch.stack([b["position_ids"] for b in batch]),
        "index": torch.tensor([b["index"] for b in batch]),
    }
    if "cu_seqlens" in batch[0]:
        out["cu_seqlens"] = [b["cu_seqlens"] for b in batch]
    return out


# Loader workers and a scheduler's SIGTERM (tests/test_dataloader_sigterm.py).
# A job scheduler may signal EVERY process of a job, the loader workers included. A worker that died of it made
# PyTorch's SIGCHLD handler raise in the trainer ("DataLoader worker (pid N) is killed by signal: Terminated") about 3 s
# later, before the step boundary where the trainer saves and exits 75, so the stop never saved. So a worker ignores
# SIGTERM while the trainer runs, and keeps feeding it through the save. It cannot ignore it for good: at interpreter
# exit multiprocessing terminate()s the daemonic workers with SIGTERM and then joins them, so a worker that ignored it
# would hang every exit, a normal completion included. An atexit hook registered here runs before multiprocessing's
# own (atexit is LIFO) and raises a shared flag; from then on SIGTERM ends the worker as before. A trainer killed
# outright never runs the hook, and its orphaned workers exit through PyTorch's parent watchdog within 5 s.
# Sampling runs in the main process, so none of this changes the data order.
def _mark_exiting(exiting) -> None:
    exiting.value = 1


def _worker_sigterm(exiting, signum, frame) -> None:
    if exiting.value:
        os._exit(0)


def _worker_init(exiting, _worker_id: int) -> None:
    signal.signal(signal.SIGTERM, functools.partial(_worker_sigterm, exiting))


class DistDataLoader(DataLoader):
    """A DataLoader that also carries `pad_token_id`. Resume state is not kept here: the trainer re-derives the
    sampler position from global_step (BaseTrainer.jump_to_step)."""

    def __init__(self, *args, pad_token_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_token_id = pad_token_id


def build_train_dataloader(
    data_cfg: dict,
    max_seq_len: int,
    global_batch_size: int,
    total_size: int | None = None,   # unused (the sampler covers the whole dataset); kept for existing callers
    shuffle: bool = True,
    seed: int | None = None,
) -> DistDataLoader:
    world = get_world_size()
    rank = get_rank()
    assert global_batch_size % world == 0, (
        f"global_batch_size ({global_batch_size}) must be divisible by world_size ({world})")
    per_rank_bs = global_batch_size // world

    doc_lengths = data_cfg.get("doc_lengths", None)
    bos_kw = dict(prepend_bos=bool(data_cfg.get("prepend_bos", False)), bos_token_id=int(data_cfg.get("bos_token_id", 1)))
    if bos_kw["prepend_bos"] and rank == 0:
        print(f"[data] prepend_bos ON: every window = [{bos_kw['bos_token_id']}] + chunk[:-1] "
              f"(same chunks and order, {max_seq_len} positions)", flush=True)
    if doc_lengths:  # document-boundary-aware path: one flat .bin + its doc_lengths sidecar
        path = data_cfg["paths"]
        if isinstance(path, (list, tuple)):
            if len(path) != 1:
                raise ValueError(f"data.doc_lengths describes one token file, but data.paths lists {len(path)}: {path}")
            path = path[0]
        dataset = DocAwareMemmapDataset(
            path=path, chunk_size=max_seq_len, doc_lengths=doc_lengths,
            dtype=data_cfg.get("dtype", "uint16"),
            position_mode=data_cfg.get("position_mode", "packed"),
            **bos_kw,
        )
    else:
        dataset = MemmapTokenDataset(
            paths=data_cfg["paths"],
            chunk_size=max_seq_len,
            dtype=data_cfg.get("dtype", "uint16"),
            **bos_kw,
        )
    sampler = ResumableDistSampler(
        dataset_len=len(dataset),
        num_replicas=world,
        rank=rank,
        seed=seed if seed is not None else data_cfg.get("seed", 42),
        shuffle=shuffle,
        drop_last=True,
        chunk_subset=data_cfg.get("chunk_subset", None),   # [start_frac, end_frac]: a slice of the seed-fixed permutation
    )
    sampler.set_epoch(0, 0)

    num_workers = int(data_cfg.get("num_workers", 2))
    loader_kwargs = dict(
        batch_size=per_rank_bs,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        drop_last=True,
        pad_token_id=data_cfg.get("pad_token_id", None),
    )
    if doc_lengths:
        loader_kwargs["collate_fn"] = _doc_collate
    if num_workers > 0:
        exiting = multiprocessing.RawValue("b", 0)        # shared with the forked workers
        atexit.register(_mark_exiting, exiting)
        loader_kwargs["worker_init_fn"] = functools.partial(_worker_init, exiting)
        loader_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 4))
    return DistDataLoader(dataset, **loader_kwargs)
