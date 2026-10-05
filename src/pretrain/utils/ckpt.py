"""Crash-safe checkpoint discovery.

A checkpoint dir is "complete & compatible" iff it contains a `.complete` marker
file whose *content* equals the run's VERSION_STRING. The marker is written LAST,
via an atomic rename, so a half-written checkpoint never looks complete; the
version-string content means an incompatible-format checkpoint is treated as
garbage rather than silently mis-loaded.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

MARKER_NAME = ".complete"
CKPT_NAME_FORMAT = "checkpoint-{step}"
CKPT_GLOB = "checkpoint-*"


def _marker_path(ckpt_dir: str | os.PathLike) -> Path:
    return Path(ckpt_dir) / MARKER_NAME


def create_complete_marker_atomic_swap(ckpt_dir: str | os.PathLike, version_str: str) -> Path:
    """Write `version_str` into `<ckpt_dir>/.complete` atomically (tmpfile + fsync +
    os.replace). Returns the marker path. MUST be called only on the master rank,
    and only after every other checkpoint artifact has been flushed."""
    ckpt_dir = Path(ckpt_dir)
    marker = _marker_path(ckpt_dir)
    fd, tmp_name = tempfile.mkstemp(dir=str(ckpt_dir), prefix=".complete.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(str(version_str))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, marker)  # atomic on POSIX
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    return marker


def check_complete_marker(ckpt_dir: str | os.PathLike, version_str: str) -> bool:
    """True iff the marker exists and its content matches `version_str`."""
    marker = _marker_path(ckpt_dir)
    if not marker.is_file():
        return False
    try:
        return marker.read_text().strip() == str(version_str).strip()
    except OSError:
        return False


def checkpoint_step(ckpt_dir: str | os.PathLike) -> int:
    """The step of a `checkpoint-<step>` dir; -1 for any other name (e.g. a hand-made `checkpoint-final`)."""
    try:
        return int(Path(ckpt_dir).name.split("-")[-1])
    except (ValueError, IndexError):
        return -1


def find_last_complete_checkpoint(output_dir: str | os.PathLike, version_str: str) -> Path | None:
    """Newest `checkpoint-<step>` dir under `output_dir` with a valid version marker,
    or None. Scans high-step-first and returns the first complete one."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in output_dir.glob(CKPT_GLOB) if p.is_dir()),
        key=checkpoint_step,
        reverse=True,
    )
    for p in candidates:
        if check_complete_marker(p, version_str):
            return p
    return None
