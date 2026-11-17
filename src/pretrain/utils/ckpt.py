"""Crash-safe checkpoint discovery, durable writes and removal.

A checkpoint dir is "complete & compatible" iff it contains a `.complete` marker
file whose *content* equals the run's VERSION_STRING. The marker is written LAST,
via an atomic rename, so a half-written checkpoint never looks complete; the
version-string content means an incompatible-format checkpoint is treated as
garbage rather than silently mis-loaded.

Marker states (marker_state): "complete" (our version), "absent" (no marker: a save that was cut short), "mismatch"
(a marker of another format version, or not a marker file) and "unreadable" (the marker exists but every read failed
with an I/O error: an OSError other than "does not exist" is retried MARKER_TRIES times, MARKER_RETRY_DELAY s apart).
An unreadable marker is never taken for "absent": find_last_complete_checkpoint raises instead of silently falling back
to an older checkpoint (which would resume an older state, and on a multi-rank job with the decision taken per rank,
mix the shards of two histories), and the rotation keeps such a directory. The error (unreadable_marker_message) says
how to get past a marker that stays unreadable: set the directory aside by hand as `.bad-<name>` (a name nothing reads
or removes), and the next launch resumes from the newest older complete checkpoint.

Removal (trash_checkpoint): a directory is first renamed to `.trash-<name>.<ns>` (a name `checkpoint-*` never matches,
so neither the trainer nor the launchers' dm_ckpt_step can pick a half-deleted directory), then removed;
cleanup_trash removes what an interrupted removal left (the master calls it at startup).

The sidecar TRAINER_STATE_FILE (trainer_state.json, written atomically by the master before the marker) holds what the
DCP TrainState does not: the skipped-update count, the world size, the last logged metrics. It is read with defaults:
checkpoints written before it existed have none (a new key in the DCP TrainState would make them unloadable).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

MARKER_NAME = ".complete"
CKPT_NAME_FORMAT = "checkpoint-{step}"
CKPT_GLOB = "checkpoint-*"
TRASH_PREFIX = ".trash-"
TRAINER_STATE_FILE = "trainer_state.json"

# An OSError reading a marker (EIO / ESTALE during a Lustre client recovery, ...) is retried: MARKER_TRIES reads in all
# (the first and three retries), MARKER_RETRY_DELAY seconds apart. "Does not exist" is an answer, not an error.
MARKER_TRIES = 4
MARKER_RETRY_DELAY = 1.0

COMPLETE, ABSENT, MISMATCH, UNREADABLE = "complete", "absent", "mismatch", "unreadable"


class CheckpointMarkerUnreadable(RuntimeError):
    """A checkpoint's .complete marker exists but cannot be read (after the retries)."""


BAD_PREFIX = ".bad-"   # a checkpoint set aside by hand (unreadable_marker_message): never read, rotated or removed


def unreadable_marker_message(ckpt_dir: str | os.PathLike, why: Optional[str]) -> str:
    """The error for a checkpoint whose marker exists but cannot be read: why the run does not fall back on its own,
    and how to get past a marker that stays unreadable."""
    p = Path(ckpt_dir)
    return (f"{p / MARKER_NAME} exists but cannot be read after {MARKER_TRIES} tries ({why}): refusing to fall back "
            f"to an older checkpoint. Check the file system; once the marker reads again a relaunch resumes from "
            f"{p.name}. If it stays unreadable (a lost file system object), set the checkpoint aside by hand: "
            f"mv {p} {p.parent / (BAD_PREFIX + p.name)} (a name neither the trainer nor the launchers read or remove), "
            f"then relaunch: the run resumes from its newest older complete checkpoint (remove the set-aside directory "
            f"once it is no longer needed)")


def _marker_path(ckpt_dir: str | os.PathLike) -> Path:
    return Path(ckpt_dir) / MARKER_NAME


def read_text_retry(path: str | os.PathLike, tries: Optional[int] = None, delay: Optional[float] = None) -> str:
    """The text of `path` (UTF-8, undecodable bytes replaced). FileNotFoundError / NotADirectoryError are raised at once
    (the file is not there); IsADirectoryError too (it is not a file); any other OSError is retried `tries` times in all
    (default MARKER_TRIES), `delay` seconds apart (default MARKER_RETRY_DELAY), and the last one is raised."""
    tries = MARKER_TRIES if tries is None else max(1, int(tries))
    delay = MARKER_RETRY_DELAY if delay is None else float(delay)
    last: Optional[OSError] = None
    for i in range(tries):
        try:
            with open(path, "rb") as f:
                return f.read().decode("utf-8", errors="replace")
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
            raise
        except OSError as e:
            last = e
            if i + 1 < tries:
                time.sleep(delay)
    assert last is not None
    raise last


def marker_state_why(ckpt_dir: str | os.PathLike, version_str: str, tries: Optional[int] = None,
                     delay: Optional[float] = None) -> tuple[str, Optional[str]]:
    """(marker_state, the error behind "unreadable" or None)."""
    try:
        text = read_text_retry(_marker_path(ckpt_dir), tries, delay)
    except (FileNotFoundError, NotADirectoryError):
        return ABSENT, None
    except IsADirectoryError:
        return MISMATCH, None            # something named .complete that is not a marker file
    except OSError as e:
        return UNREADABLE, f"{type(e).__name__}: {e}"
    return (COMPLETE if text.strip() == str(version_str).strip() else MISMATCH), None


def marker_state(ckpt_dir: str | os.PathLike, version_str: str, tries: Optional[int] = None,
                 delay: Optional[float] = None) -> str:
    """"complete" | "absent" | "mismatch" | "unreadable" (module docstring). Never raises."""
    return marker_state_why(ckpt_dir, version_str, tries, delay)[0]


def create_complete_marker_atomic_swap(ckpt_dir: str | os.PathLike, version_str: str) -> Path:
    """Write `version_str` into `<ckpt_dir>/.complete` atomically (tmpfile + fsync +
    os.replace, then a best-effort fsync of the directory). Returns the marker path. MUST be called only on the master
    rank, and only after every other checkpoint artifact has been flushed."""
    ckpt_dir = Path(ckpt_dir)
    marker = _marker_path(ckpt_dir)
    fd, tmp_name = tempfile.mkstemp(dir=str(ckpt_dir), prefix=".complete.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            try:
                os.fchmod(f.fileno(), _file_mode())   # readable like the checkpoint's other files (mkstemp: 0o600)
            except OSError:
                pass
            f.write(str(version_str))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, marker)  # atomic on POSIX
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    fsync_dir(ckpt_dir)
    return marker


def check_complete_marker(ckpt_dir: str | os.PathLike, version_str: str) -> bool:
    """True iff the marker exists and its content matches `version_str` (marker_state == "complete"; an unreadable
    marker is False here: callers that must tell it from "absent" use marker_state)."""
    return marker_state(ckpt_dir, version_str) == COMPLETE


def checkpoint_step(ckpt_dir: str | os.PathLike) -> int:
    """The step of a `checkpoint-<step>` dir; -1 for any other name (e.g. a hand-made `checkpoint-final`)."""
    try:
        return int(Path(ckpt_dir).name.split("-")[-1])
    except (ValueError, IndexError):
        return -1


def find_last_complete_checkpoint(output_dir: str | os.PathLike, version_str: str) -> Path | None:
    """Newest `checkpoint-<step>` dir under `output_dir` with a valid version marker,
    or None. Scans high-step-first and returns the first complete one; directories without a marker (a save cut short)
    or with another version's marker are passed over. A marker that exists but cannot be read raises
    CheckpointMarkerUnreadable: falling back to an older checkpoint would silently resume an older state. An I/O error
    listing the directory propagates as well."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in output_dir.glob(CKPT_GLOB) if p.is_dir()),
        key=checkpoint_step,
        reverse=True,
    )
    for p in candidates:
        state, why = marker_state_why(p, version_str)
        if state == COMPLETE:
            return p
        if state == UNREADABLE:
            raise CheckpointMarkerUnreadable(unreadable_marker_message(p, why))
    return None


# ---------------------------------------------------------------- durable writes
def fsync_path(path: str | os.PathLike) -> None:
    """fsync a file that is already written (any open descriptor flushes the file's data). Raises on failure: a file
    the resume needs that may not be on disk must fail the save before its marker is written."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: str | os.PathLike) -> bool:
    """Best-effort fsync of a directory (its entries: the names of the files just written). False when the file system
    does not support it or it fails; never raises."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return False
    try:
        os.fsync(fd)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def torch_save_durable(obj: Any, path: str | os.PathLike) -> None:
    """torch.save(obj, path), then flush and fsync it (the same file format as torch.save to a path)."""
    import torch
    with open(path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())


def _file_mode() -> int:
    """The mode open(path, "w") would create a file with (0o666 minus the umask, read race-free from /proc), instead
    of mkstemp's 0o600."""
    mask = 0o022
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Umask:"):
                    mask = int(line.split()[1], 8)
                    break
    except (OSError, ValueError, IndexError):
        pass
    return 0o666 & ~mask


def write_text_atomic(path: str | os.PathLike, text: str) -> None:
    """Write `text` to `path` atomically and durably: a temporary file in the same directory, fsync, os.replace. A
    reader sees the old content or the new one, never a part. Raises on failure (the temporary file is removed)."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            try:
                os.fchmod(f.fileno(), _file_mode())
            except OSError:
                pass
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_json_atomic(path: str | os.PathLike, obj: Any) -> None:
    """write_text_atomic of obj as JSON (indented; values json cannot encode are written as their str())."""
    write_text_atomic(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def read_json_file(path: str | os.PathLike, tries: Optional[int] = None) -> Any:
    """The JSON content of `path` (read_text_retry: FileNotFoundError when absent, OSError after the retries,
    ValueError when it does not parse)."""
    return json.loads(read_text_retry(path, tries))


def read_trainer_state(ckpt_dir: str | os.PathLike) -> tuple[dict, Optional[str]]:
    """(the checkpoint's TRAINER_STATE_FILE as a dict, None) when present and readable; ({}, None) when absent (a
    checkpoint written before the sidecar existed); ({}, why) when it exists but cannot be read or parsed. Never
    raises: every field has a default on resume."""
    p = Path(ckpt_dir) / TRAINER_STATE_FILE
    try:
        d = read_json_file(p)
    except (FileNotFoundError, NotADirectoryError):
        return {}, None
    except (OSError, ValueError) as e:
        return {}, f"{p} cannot be read ({type(e).__name__}: {e}); resuming with its defaults"
    if not isinstance(d, dict):
        return {}, f"{p} is not a JSON object; resuming with its defaults"
    return d, None


# ---------------------------------------------------------------- removal
def trash_checkpoint(path: str | os.PathLike) -> Optional[str]:
    """Remove a checkpoint directory without ever leaving a complete-looking remnant: rename it to
    `.trash-<name>.<ns>` (which `checkpoint-*` never matches), then rmtree. If the rename fails, the marker is unlinked
    first and the directory removed in place; if even the marker cannot be unlinked the directory is kept. Returns None
    when the directory is gone (or its remnant is a .trash- directory), else a message saying what happened. Never
    raises."""
    path = Path(path)
    trash = path.parent / f"{TRASH_PREFIX}{path.name}.{time.time_ns()}"
    try:
        os.rename(path, trash)
    except FileNotFoundError:
        return None
    except OSError as e:
        try:
            os.unlink(path / MARKER_NAME)
        except FileNotFoundError:
            pass
        except OSError as e2:
            return f"kept {path}: it could not be renamed ({e}) and its marker could not be removed ({e2})"
        shutil.rmtree(path, ignore_errors=True)
        return f"{path} could not be renamed ({e}): its marker was removed first, then the directory in place"
    shutil.rmtree(trash, ignore_errors=True)
    return None


def cleanup_trash(base: str | os.PathLike) -> list[str]:
    """Remove the `.trash-*` directories an interrupted removal left under `base`. Returns their names. Never
    raises."""
    removed = []
    try:
        entries = list(os.scandir(str(base)))
    except OSError:
        return removed
    for e in entries:
        if e.name.startswith(TRASH_PREFIX):
            try:
                if e.is_dir(follow_symlinks=False):
                    shutil.rmtree(e.path, ignore_errors=True)
                else:
                    os.unlink(e.path)
                removed.append(e.name)
            except OSError:
                pass
    return removed
