"""python -m pretrain.hf.check_export <export dir>: is an HF export whole?

Exit 0 (and "export whole: ..." on stdout) iff
  * config.json parses as a JSON object, and
  * model.safetensors exists, its header parses (8-byte little-endian header length N, then N bytes of a JSON object
    whose tensor entries carry [begin, end] data_offsets), and the file is exactly 8 + N + the largest data_offsets end
    bytes long: a file cut short (a killed or failed write, a full quota) or with trailing bytes is not whole.
Otherwise the reason on stdout and exit 1 (2: usage). Reads the header only (about 100 KB), never the tensors, and
imports neither torch nor transformers (the launchers run it after every training: scripts/campaign/train_arm.sh,
stage2_arm.sh). The exporter (hf/export.py) writes config.json last, so a whole export's config.json is its last file.
In Python: export_is_whole(dir) -> (bool, reason).
"""
from __future__ import annotations

import json
import os
import struct
import sys

MAX_HEADER = 100 * 1024 * 1024   # safetensors' own limit on the header


def _safetensors_whole(path: str) -> tuple[bool, str]:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(8)
            if len(head) < 8:
                return False, f"{path} is {size} bytes: shorter than the 8-byte header length"
            (n,) = struct.unpack("<Q", head)
            if n > MAX_HEADER or 8 + n > size:
                return False, f"{path} is {size} bytes, its header length says {n} bytes of header (cut short?)"
            raw = f.read(n)
    except OSError as e:
        return False, f"{path} cannot be read ({type(e).__name__}: {e})"
    if len(raw) < n:
        return False, f"{path}: the header is cut short ({len(raw)} of {n} bytes)"
    try:
        hdr = json.loads(raw)
    except ValueError as e:
        return False, f"{path}: the header does not parse ({type(e).__name__}: {e})"
    if not isinstance(hdr, dict):
        return False, f"{path}: the header is not a JSON object"
    end, count = 0, 0
    for name, entry in hdr.items():
        if name == "__metadata__":
            continue
        offs = entry.get("data_offsets") if isinstance(entry, dict) else None
        if (not isinstance(offs, list) or len(offs) != 2 or not all(isinstance(x, int) for x in offs)
                or not 0 <= offs[0] <= offs[1]):
            return False, f"{path}: tensor {name!r} has no valid data_offsets ({offs!r})"
        end, count = max(end, offs[1]), count + 1
    if count == 0:
        return False, f"{path}: the header lists no tensor"
    want = 8 + n + end
    if size != want:
        return False, f"{path} is {size} bytes, its header says {want}" + (" (cut short)" if size < want else "")
    return True, f"{count} tensors, {size} bytes"


def export_is_whole(export_dir: str | os.PathLike) -> tuple[bool, str]:
    """(True, a summary) when the export in `export_dir` is whole (module docstring), else (False, the reason)."""
    d = os.fspath(export_dir)
    cfg = os.path.join(d, "config.json")
    try:
        with open(cfg, "rb") as f:
            c = json.loads(f.read())
    except FileNotFoundError:
        return False, f"{cfg} is missing (no export, or one that did not finish: config.json is written last)"
    except OSError as e:
        return False, f"{cfg} cannot be read ({type(e).__name__}: {e})"
    except ValueError as e:
        return False, f"{cfg} does not parse ({type(e).__name__}: {e})"
    if not isinstance(c, dict):
        return False, f"{cfg} is not a JSON object"
    model = os.path.join(d, "model.safetensors")
    if not os.path.isfile(model):
        return False, f"{model} is missing"
    ok, why = _safetensors_whole(model)
    return ok, why


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0] in ("-h", "--help"):
        print("usage: python -m pretrain.hf.check_export <export dir>", file=sys.stderr)
        return 2
    ok, why = export_is_whole(argv[0])
    print(f"export whole: {argv[0]} ({why})" if ok else f"export not whole: {why}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
