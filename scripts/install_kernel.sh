#!/usr/bin/env bash
# scripts/install_kernel.sh: build the fp8 attention kernel package bundled with this repository (kernels/delta-matching,
# import name `delta_matching`; kernels/README.md) and install it into the Python environment of DM_PY (default: the
# repo's .venv, else `python` on PATH, as scripts/campaign/env.sh resolves it; with a conda env, activate it first).
#
#   scripts/install_kernel.sh                                         # the bundled kernel (the KDA-evolved build)
#   DM_KERNEL_SRC=/path/to/delta-matching scripts/install_kernel.sh   # another checkout, e.g. the original release
#
# Needs a torch built for CUDA 13 (cu130) in that environment, nvcc 13 (CUDA_HOME, else nvcc on PATH), and git with
# network access the first time: the kernels compile against the hopper/ headers of flash-attention and its CUTLASS at
# the commits the package pins (delta_matching/__init__.py), fetched the first time (into DM_FA_SRC when that directory
# does not exist yet). No GPU is needed to build (a few minutes with MAX_JOBS=16).
#   DM_FA_SRC   a flash-attention checkout with hopper/ and csrc/cutlass/include at the pinned commits (default
#               $DM_WORKSPACE/build/flash-attention; fetched there when the directory does not exist)
#   DM_FA_ALLOW_UNPINNED=1   build against a DM_FA_SRC at other commits anyway (otherwise refused)
#   DM_KERNEL_BUILD_DIR      where the build copy goes (default $TMPDIR, else /tmp; a few hundred MB, removed after)
#   MAX_JOBS    parallel nvcc jobs (default 16)
#   DM_BF16_DP=1 / DM_DBG=1   the variant builds of kernels/delta-matching/README.md (each one is a different binary)
# The build runs in a fresh copy of the source tree (no object of an earlier build is reused; nothing is written into the
# tree). At the end the installed package is imported from outside the tree: its kernel_version()["code_sha"] must equal
# the hash of the source tree's csrc/ (kernels/PROVENANCE.json records the bundled one), its compiled library must be
# the installed package's, and its Python files must be the source tree's: the environment now runs this source.
# Exit status: 0 installed and checked, 2 a pre-flight check failed, otherwise the failing build or check's status.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/campaign/env.sh"
fatal () { echo "FATAL: $*" >&2; exit 2; }

SRC_IN=${DM_KERNEL_SRC:-$REPO/kernels/delta-matching}
[ -f "$SRC_IN/setup.py" ] && [ -d "$SRC_IN/csrc" ] && [ -f "$SRC_IN/delta_matching/__init__.py" ] \
  || fatal "$SRC_IN is not a delta-matching checkout (setup.py, csrc/, delta_matching/)"
SRC=$(cd "$SRC_IN" && pwd)
[ -n "${DM_PY:-}" ] && [ -x "$DM_PY" ] || fatal "no Python interpreter (DM_PY='${DM_PY:-}'); activate the environment or set DM_PY"
pin () { sed -n "s/^$1 = \"\([0-9a-f]*\)\".*/\1/p" "$SRC/delta_matching/__init__.py"; }
FA_COMMIT=$(pin FA_SRC_COMMIT); CUTLASS_COMMIT=$(pin CUTLASS_COMMIT)
[ -n "$FA_COMMIT" ] && [ -n "$CUTLASS_COMMIT" ] || fatal "$SRC/delta_matching/__init__.py names no FA_SRC_COMMIT / CUTLASS_COMMIT"

# torch: a CUDA 13 build in the target environment
"$DM_PY" - <<'PY' || exit 2
import sys
try:
    import torch
except ImportError:
    sys.exit("FATAL: torch is not installed in this environment (docs/SETUP.md section 2)")
cu = torch.version.cuda or ""
if not cu.startswith("13."):
    sys.exit(f"FATAL: torch {torch.__version__} is built for CUDA {cu or 'none'}; the kernel needs a CUDA 13 (cu130) torch")
print(f"[install_kernel] python {sys.version.split()[0]} at {sys.prefix}; torch {torch.__version__} (cuda {cu})")
PY

# nvcc 13
NVCC=""
if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then NVCC=$CUDA_HOME/bin/nvcc; else NVCC=$(command -v nvcc || true); fi
[ -n "$NVCC" ] || fatal "nvcc not found: set CUDA_HOME to a CUDA 13 toolkit"
NVCC_REL=$("$NVCC" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p')
case "$NVCC_REL" in 13.*) ;; *) fatal "$NVCC is release '${NVCC_REL:-?}'; the kernel needs nvcc 13 (set CUDA_HOME)" ;; esac
CUDA_HOME=$(cd "$(dirname "$NVCC")/.." && pwd); export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
echo "[install_kernel] nvcc $NVCC_REL at $CUDA_HOME"

CLEAN=()   # temporary directories, removed on exit
cleanup () { if [ ${#CLEAN[@]} -gt 0 ]; then rm -rf "${CLEAN[@]}"; fi; }
trap cleanup EXIT

# flash-attention + CUTLASS headers at the pinned commits (fetched into DM_FA_SRC when it does not exist)
FA=${DM_FA_SRC:-$DM_WORKSPACE/build/flash-attention}
if [ ! -d "$FA/hopper" ] || [ ! -d "$FA/csrc/cutlass/include" ]; then
  [ -e "$FA" ] && fatal "$FA exists but has no hopper/ or csrc/cutlass/include (an interrupted fetch or another checkout): remove it, or set DM_FA_SRC to a complete checkout"
  echo "[install_kernel] fetching flash-attention $FA_COMMIT and CUTLASS $CUTLASS_COMMIT into $FA"
  TMPFA="$FA.partial.$$"; CLEAN+=("$TMPFA")
  rm -rf "$TMPFA"; mkdir -p "$(dirname "$FA")"
  git init -q "$TMPFA"
  git -C "$TMPFA" remote add origin https://github.com/Dao-AILab/flash-attention.git
  git -C "$TMPFA" fetch -q --depth 1 origin "$FA_COMMIT"
  git -C "$TMPFA" checkout -q FETCH_HEAD
  git -C "$TMPFA" submodule init csrc/cutlass
  git -C "$TMPFA" submodule update --depth 1 csrc/cutlass
  [ -e "$FA" ] && fatal "$FA appeared while fetching (another install running?); remove it or set DM_FA_SRC"
  mv "$TMPFA" "$FA"
fi
FA=$(cd "$FA" && pwd)
HAVE_FA=$(git -C "$FA" rev-parse HEAD 2>/dev/null || echo unknown)
HAVE_CUT=$(git -C "$FA/csrc/cutlass" rev-parse HEAD 2>/dev/null || echo unknown)
if [ "$HAVE_FA" != "$FA_COMMIT" ] || [ "$HAVE_CUT" != "$CUTLASS_COMMIT" ]; then
  MSG="$FA is at flash-attention $HAVE_FA / CUTLASS $HAVE_CUT; the kernel pins $FA_COMMIT / $CUTLASS_COMMIT"
  [ "${DM_FA_ALLOW_UNPINNED:-0}" = 1 ] || fatal "$MSG (point DM_FA_SRC at a checkout of the pins, or set DM_FA_ALLOW_UNPINNED=1)"
  echo "WARN: $MSG; building anyway (DM_FA_ALLOW_UNPINNED=1)" >&2
fi
echo "[install_kernel] flash-attention headers: $FA ($HAVE_FA, cutlass $HAVE_CUT)"

# build + install (no build isolation: the extension compiles against the installed torch; no deps: never touch torch).
# pip when the environment has it, else uv (a `uv venv` has no pip); uv without its cache, which would otherwise reuse a
# wheel built before an edit of csrc/ (its cache key of a source tree is the packaging files, not the sources)
# a fresh copy of the source tree: ninja decides by mtime, and a tree re-vendored with `git archive` keeps its commit's
# mtimes, so the objects of an earlier in-tree build could be relinked under the new stamp
BD=${DM_KERNEL_BUILD_DIR:-${TMPDIR:-/tmp}}; mkdir -p "$BD"
BUILD_SRC=$(mktemp -d "$BD/dm-kernel-build.XXXXXX"); CLEAN+=("$BUILD_SRC")
tar -C "$SRC" --exclude=./build --exclude='./*.egg-info' --exclude=./delta_matching/_build_info.py --exclude=__pycache__ \
    -cf - . | tar -C "$BUILD_SRC" -xf -
echo "[install_kernel] building $SRC (a copy in $BUILD_SRC) with MAX_JOBS=${MAX_JOBS:-16}${DM_BF16_DP:+ DM_BF16_DP=$DM_BF16_DP}${DM_DBG:+ DM_DBG=$DM_DBG}"
if "$DM_PY" -m pip --version > /dev/null 2>&1; then
  INSTALL=("$DM_PY" -m pip install --no-build-isolation --no-deps --force-reinstall .)
elif command -v uv > /dev/null 2>&1; then
  INSTALL=(uv pip install --python "$DM_PY" --no-build-isolation --no-deps --reinstall --no-cache .)
else
  fatal "neither pip (in $DM_PY's environment) nor uv is available"
fi
( cd "$BUILD_SRC" && DM_FA_SRC="$FA" MAX_JOBS="${MAX_JOBS:-16}" "${INSTALL[@]}" )

# check: the installed package (imported from outside the source tree) is this tree: its csrc/ hash, its own compiled
# library, its Python files
( cd / && env -u DM_SO PYTHONDONTWRITEBYTECODE=1 "$DM_PY" - "$SRC" "$BUILD_SRC" "$HAVE_FA" "$HAVE_CUT" <<'PY'
import filecmp, importlib.util, os, sys
src, copy = os.path.realpath(sys.argv[1]), os.path.realpath(sys.argv[2])
spec = importlib.util.spec_from_file_location("_dm_src_buildcfg", os.path.join(src, "delta_matching", "_buildcfg.py"))
cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
want = cfg.csrc_sha()
import delta_matching
pkg = os.path.realpath(os.path.dirname(delta_matching.__file__))
for tree in (src, copy):
    if pkg == tree or pkg.startswith(tree + os.sep):
        sys.exit(f"FATAL: delta_matching was imported from the source tree {pkg}, not from the installed package "
                 "(is the tree on PYTHONPATH?)")
kv = delta_matching.kernel_version()
if kv.get("code_sha") != want:
    sys.exit(f"FATAL: the installed delta_matching reports code_sha {kv.get('code_sha')}, the source tree {src} hashes to {want}")
lib = kv.get("library")
if not lib or not os.path.realpath(lib).startswith(pkg + os.sep):
    sys.exit(f"FATAL: the installed delta_matching did not load its own compiled library (library={lib!r}, package {pkg})")
srcpkg = os.path.join(src, "delta_matching")
py = sorted(f for f in os.listdir(srcpkg) if f.endswith(".py") and f != "_build_info.py")
bad = [f for f in py if not os.path.isfile(os.path.join(pkg, f))
       or not filecmp.cmp(os.path.join(srcpkg, f), os.path.join(pkg, f), shallow=False)]
if bad:
    sys.exit(f"FATAL: the installed package's Python files differ from the source tree's: {bad}")
print(f"[install_kernel] OK: delta_matching {delta_matching.__version__} installed at {pkg}; code_sha {want} "
      f"(bf16_dp={kv.get('bf16_dp')}, dbg={kv.get('dbg')}), its library and {len(py)} Python files built from {src} "
      f"against flash-attention {sys.argv[3][:12]} / CUTLASS {sys.argv[4][:12]}")
PY
)
