"""Rank-aware logging: `main_LOG(s, level)` is a single callable that only emits on
the master rank, so call sites stay clean.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Callable, Literal

# ANSI colors
RED = "\033[31m"
GREEN = "\033[32m"
ORANGE = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
RESET = "\033[0m"

LogFunc = Callable[..., None]


def setup_logging(log_path: str | os.PathLike | None, *, rank: int, log_to_file: bool = True) -> logging.Logger:
    logger = logging.getLogger(f"pretrain.rank{rank}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter(f"[%(asctime)s][r{rank}][%(levelname)s] %(message)s", "%H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_to_file and log_path is not None:
        os.makedirs(os.path.dirname(str(log_path)) or ".", exist_ok=True)
        fh = logging.FileHandler(str(log_path))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


class _Tee:
    """Mirror a text stream (sys.stdout/sys.stderr) to a file as well. Writes to the
    real stream AND the file, flushing both, so the full terminal output of a run is
    also captured on disk. Used master-rank-only."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        self._stream.write(data)
        try:
            self._fh.write(data)
        except Exception:
            pass
        return len(data)

    def flush(self):
        self._stream.flush()
        try:
            self._fh.flush()
        except Exception:
            pass

    def __getattr__(self, name):  # isatty/fileno/encoding/... -> defer to real stream
        return getattr(self._stream, name)


def tee_stdio(log_file_path: str | os.PathLike) -> None:
    """Tee sys.stdout/sys.stderr into `log_file_path` (append). Call on master rank
    only, early, so all of the run's terminal output is also saved to the file."""
    os.makedirs(os.path.dirname(str(log_file_path)) or ".", exist_ok=True)
    fh = open(str(log_file_path), "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)


def make_main_log(logger: logging.Logger | None, is_master: bool) -> LogFunc:
    """Return a `main_LOG(s, level="info")` that no-ops off the master rank."""

    def main_log(s: object, level: Literal["info", "critical", "warning", "error"] = "info") -> None:
        if not is_master or logger is None:
            return
        getattr(logger, level)(str(s))

    return main_log
