import hashlib
import json
import os
import shutil
import tempfile
import time
from typing import Any, Mapping, Optional

import portalocker
import wandb

from .dist import is_master, dist_barrier

__all__ = [
    "atomic_write_json",
    "locked_atomic_write_json",
    "locked_read_json",
    "log_and_flush",
    "MARKER_NAME",
    "get_complete_marker_path",
    "create_complete_marker",
    "all_check_complete_marker",
    "check_complete_marker",
]


def _fsync_dir(dirpath: str) -> None:
    """Durability: ensure the directory entry for the replaced file is flushed."""
    try:
        fd = os.open(dirpath, os.O_DIRECTORY)
    except (AttributeError, FileNotFoundError, NotADirectoryError):
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_json(path: str, obj: Any, *, indent: Optional[int] = None) -> None:
    """
    Write JSON atomically (temp file -> fsync -> replace) so readers never see partial content.
    """
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=d, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", closefd=True) as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())  # ensure file bytes hit disk
        os.replace(tmp, path)  # atomic on POSIX & Windows
        _fsync_dir(d)  # ensure directory entry is durable (POSIX)
    finally:
        # If replace succeeded, tmp is gone; if not, try to clean it up.
        try:
            os.remove(tmp)
        except FileNotFoundError:
            pass


def locked_atomic_write_json(
    path: str, obj: Any, *, timeout: float = 10.0, indent: Optional[int] = None
) -> None:
    """
    Take an exclusive lock on a dedicated lock file, then perform an atomic write.
    """
    lock_path = path + ".lock"
    # 'a+' ensures the lock file exists on first use; EX gives mutual exclusion.
    with portalocker.Lock(
        lock_path, mode="a+", flags=portalocker.LOCK_EX, timeout=timeout
    ):
        atomic_write_json(path, obj, indent=indent)


def locked_read_json(
    path: str, default: Optional[Mapping[str, Any]] = None, *, timeout: float = 20.0
) -> Mapping[str, Any]:
    """
    Take a shared lock on the lock file and read JSON safely.
    Returns `default` (or {}) if file is missing/invalid.
    """
    lock_path = path + ".lock"
    try:
        with portalocker.Lock(
            lock_path, mode="a+", flags=portalocker.LOCK_SH, timeout=timeout
        ):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
    except FileNotFoundError:
        return {} if default is None else default
    except json.JSONDecodeError:
        # If some external writer isn’t using our atomic+lock discipline, fall back gracefully.
        return {} if default is None else default

    # Optional: enforce dict output
    if isinstance(data, dict):
        return data
    raise ValueError("json dict failure load, a sync bug happened")


def log_and_flush(wandb_run, length, log_dict):
    # 1) Log metrics (async)
    wandb_run.log(log_dict, step=length, commit=True)

    # 2) Barrier marker artifact: best-effort progress tracking only. Its .wait() is
    #    not supported in offline/disabled wandb (raises ArtifactNotLoggedError), so skip
    #    it there, and never let this non-essential step crash the eval.
    if os.environ.get("WANDB_MODE", "").lower() in {"offline", "disabled"}:
        return
    try:
        marker_meta = {
            "run_id": wandb_run.id,
            "step": length,
            "payload_hash": hashlib.sha1(
                json.dumps(log_dict, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "ts": time.time(),
        }
        art = wandb.Artifact(
            name=f"marker-run-{wandb_run.id}",
            type="marker",
            metadata={str(length): marker_meta},
        )
        # reuse the same name each step, so you just get new versions
        wandb_run.log_artifact(art, aliases=["latest", f"upto-{length}"]).wait()
    except Exception:
        pass


MARKER_NAME = ".complete"  # suffix of the completion marker, written last


def get_complete_marker_path(path: str) -> str:
    """
    Returns the path to the completion marker.
    If path is a directory, returns path/MARKER_NAME.
    If path is a file, returns path with extension replaced by MARKER_NAME.
    """
    if os.path.isdir(path):
        return os.path.join(path, MARKER_NAME)
    else:
        # It is a file: replace extension
        base, _ = os.path.splitext(path)
        return base + MARKER_NAME


def create_complete_marker(path: str, version_str: int) -> str:
    complete_path = get_complete_marker_path(path=path)

    # Ensure directory exists
    os.makedirs(os.path.dirname(complete_path), exist_ok=True)

    # Create/overwrite the marker file with version id
    with open(complete_path, "w") as f:
        f.write(str(version_str))

    return complete_path


def all_check_complete_marker(complete_path, version_str) -> bool:
    if os.path.exists(complete_path):
        with open(complete_path, "r") as f:
            existing = f.read().strip()
        if existing == str(version_str):
            return True  # matches, nothing to do
        else:
            # Version mismatch detected
            return False
    else:
        return False


def check_complete_marker(path: str, version_str: str, clean_dir: bool = False) -> bool:
    """
    Return True if the completion marker of `path` (get_complete_marker_path) exists and holds `version_str`,
    else False. Collective: every rank must call it (it runs two barriers).

    clean_dir=False (the default) never deletes anything. clean_dir=True: when the check fails (marker missing
    or holding another version), rank 0 deletes the directory that holds the marker and recreates it empty; for
    a file `path` (a results JSON) that is the directory of the file, with everything else in it.
    """
    complete_path = get_complete_marker_path(path=path)
    dir_to_clean = os.path.dirname(complete_path)

    check_passed = all_check_complete_marker(
        complete_path=complete_path, version_str=version_str
    )

    dist_barrier()
    if not check_passed and clean_dir and is_master():
        if os.path.isdir(dir_to_clean):
            # Remove the entire directory tree and recreate the empty folder
            shutil.rmtree(dir_to_clean)
            os.mkdir(dir_to_clean)
    dist_barrier()
    return check_passed

