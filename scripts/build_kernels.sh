#!/usr/bin/env bash
# Build a delta-matching-train development environment (Hopper / sm90, CUDA 13.0) with uv.
#
# The certified environment of the study is the lock file of docs/SETUP.md section 2 (torch 2.12.0+cu130); this script
# is the `uv` recipe for a fresh development environment at a newer torch. Why a script instead of a plain `uv sync`:
#   * mamba-ssm 2.3.1 still compiles its host .cpp with -std=c++17, but torch>=2.14
#     headers (ATen) require C++20 -> "#error C++20 or later compatible compiler".
#     There is no env var for it, so the sdist's setup.py is patched in place.
#   * mamba-ssm / causal-conv1d have no CUDA-arch env var and would otherwise emit
#     sm75..sm121. We pin sm90 (Hopper) to cut build time.
#   * flash-attn is built from the pinned flash-attention commit (see pyproject
#     [tool.uv.sources]) with FLASH_ATTN_CUDA_ARCHS=90.
#   * the fp8 attention kernel is the separate delta-matching package, built from its checkout (step 8).
#
# NOTE: mamba-ssm must stay <= 2.3.1 with this torch: 2.3.2.post1 pins apache-tvm-ffi<=0.1.9.
# NOTE: requires the CUDA forward-compat driver (libcuda.so.1 -> 580.x) for CUDA 13
#       on a 570.x kernel module.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The build tree: a fresh private directory by default. BUILD_DIR reuses one across runs (the clone and the sdists in it
# are trusted and built as they are, so it must be your own directory, never a shared path another user could create).
if [ -n "${BUILD_DIR:-}" ]; then
    BUILD="$BUILD_DIR"; mkdir -p "$BUILD"
    [ -O "$BUILD" ] || { echo "FATAL: BUILD_DIR=$BUILD is not owned by $(id -un)" >&2; exit 1; }
else
    BUILD="$(mktemp -d "${TMPDIR:-/tmp}/dm-build.XXXXXX")"
fi
FA_REV="1bda8f9290cd48d030f1516f0e680cd464ef3554"
CC1D_V="1.7.0"
MAMBA_V="2.3.1"
DM_KERNEL_SRC="${DM_KERNEL_SRC:-}"          # a checkout of the delta-matching repository (the fp8 attention kernel)

export VIRTUAL_ENV="$REPO/.venv"
export TMPDIR="$BUILD"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-64}"
export NVCC_THREADS="${NVCC_THREADS:-4}"
export UV_LINK_MODE=copy
mkdir -p "$BUILD/sd"

echo "==> [1/8] venv (python 3.12)"
[ -d "$REPO/.venv" ] || uv venv --python 3.12 --prompt dm "$REPO/.venv"

echo "==> [2/8] torch + build deps (torch>=2.14 on PyPI is already a cu130 build)"
uv pip install "torch>=2.12" numpy einops PyYAML packaging ninja setuptools wheel psutil

echo "==> [3/8] wheel-only deps"
# tilelang: fla's GDN backward on Hopper refuses to run on Triton >= 3.4 without its TileLang backend (pinned as in the lock)
uv pip install "transformers>=5.9" datasets portalocker liger_kernel torchao \
    "flash-linear-attention==0.5.2" "fla-core==0.5.2" "tilelang==0.1.8" wandb

echo "==> [3b/8] optional extras (eval harness + analysis); drop if not needed"
uv pip install lm_eval nltk pandas matplotlib

echo "==> [4/8] flash-attn (FA2 CUDA ext, sm90 only)"
if [ ! -d "$BUILD/flash-attention" ]; then
    git clone https://github.com/Dao-AILab/flash-attention.git "$BUILD/flash-attention"
fi
git -C "$BUILD/flash-attention" checkout -q "$FA_REV"
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTN_CUDA_ARCHS=90 \
    uv pip install --no-build-isolation --no-deps "$BUILD/flash-attention"

# fetch + patch an sdist: $1=pypi name  $2=version  $3=extracted dir name
fetch_patch() {
    local name="$1" ver="$2" dir="$3"
    cd "$BUILD/sd"
    if [ ! -d "$dir" ]; then
        "$VIRTUAL_ENV/bin/python" - "$name" "$ver" <<'PY'
import sys, json, urllib.request as u
name, ver = sys.argv[1], sys.argv[2]
d = json.load(u.urlopen(f"https://pypi.org/pypi/{name}/{ver}/json"))
url = [f["url"] for f in d["urls"] if f["filename"].endswith(".tar.gz")][0]
u.urlretrieve(url, url.rsplit("/", 1)[-1]); print("fetched", url)
PY
        tar xzf "${dir}.tar.gz"
    fi
    "$VIRTUAL_ENV/bin/python" "$REPO/scripts/_patch_kernel_sdist.py" "$dir/setup.py"
}

echo "==> [5/8] causal_conv1d + mamba_ssm (sm90, C++20)"
fetch_patch causal-conv1d "$CC1D_V" "causal_conv1d-$CC1D_V"
CAUSAL_CONV1D_FORCE_BUILD=TRUE uv pip install --no-build-isolation "$BUILD/sd/causal_conv1d-$CC1D_V"

fetch_patch mamba-ssm "$MAMBA_V" "mamba_ssm-$MAMBA_V"
MAMBA_FORCE_BUILD=TRUE uv pip install --no-build-isolation "$BUILD/sd/mamba_ssm-$MAMBA_V"

echo "==> [6/8] flash_attn_3 (FA3 hopper, sm90a): the bf16 baseline of the head_dim-256 arms (DM_FLASH_IMPL=fa3)"
# Arch is hardcoded to compute_90a/sm_90a in hopper/setup.py, so no arch var needed.
# DISABLE_SM80 drops 140 of 451 instantiations we can never run on a Hopper GPU.
( cd "$BUILD/flash-attention/hopper" \
  && FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_DISABLE_SM80=TRUE \
     MAX_JOBS="${MAX_JOBS}" NVCC_THREADS=2 \
     uv pip install --no-build-isolation --no-deps . )

echo "==> [7/8] transformer_engine -- optional: attn_impl: te_fp8 (the cuDNN fp8 attention arm); a failed build only warns"
# NVTE_WITH_NCCL_EP=0: TE's NCCL expert-parallel path needs NCCL>=2.30 headers and
#   auto-enables for arch>=90; a system nccl.h at 2.27.7 (ncclWindow_vidmem missing)
#   makes ep.cpp fail to compile. This repo uses TE only for fused fp8 attention.
# CPLUS_INCLUDE_PATH: cudnn.h comes from the pip nvidia-cudnn-cu13 package, which is
#   not on the default include path.
if ! CUDNN_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn" \
    CPLUS_INCLUDE_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/include:${CPLUS_INCLUDE_PATH:-}" \
    LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:${LIBRARY_PATH:-}" \
    NVTE_FRAMEWORK=pytorch NVTE_CUDA_ARCHS=90 NVTE_WITH_NCCL_EP=0 MAX_JOBS="${MAX_JOBS}" \
    uv pip install --no-build-isolation "transformer-engine[pytorch,core-cu13]"; then
    echo "WARN: the Transformer Engine build failed; it is optional (only the te_fp8 arm needs it), continuing"
fi

echo "==> [8/8] this repository (editable) + the delta-matching kernel package"
uv pip install --no-build-isolation --no-deps -e "$REPO"
if [ -n "$DM_KERNEL_SRC" ] && [ -f "$DM_KERNEL_SRC/setup.py" ]; then
    # the fp8 attention kernel (Delta-Matching-128 / -256, sm_90a; a few minutes with MAX_JOBS=16); DM_BF16_DP=1 in the
    # environment builds the bf16_dp variant instead (one build = one binary; see the package's README)
    ( cd "$DM_KERNEL_SRC" && MAX_JOBS="${MAX_JOBS}" uv pip install --no-build-isolation . )
else
    echo "NOTE: the delta-matching kernel package was not installed (set DM_KERNEL_SRC=/path/to/delta-matching checkout"
    echo "      and rerun, or install it by hand: cd <checkout> && MAX_JOBS=16 pip install --no-build-isolation .;"
    echo "      docs/SETUP.md section 3). The fp8 arms need it; the bf16 arms and the evaluation do not."
fi

echo
echo "Done. Verify with:  $REPO/.venv/bin/python scripts/verify_env.py"
echo "The build tree is $BUILD (delete it when you no longer need it)."
echo
echo "IMPORTANT -- transformer_engine + NCCL load order:"
echo "  TE links a NCCL that resolves to the system libnccl.so.2 when that one is older (2.27.7 lacks"
echo "  ncclCommResume). If transformer_engine is imported BEFORE torch, torch then fails with:"
echo "  libtorch_cuda.so: undefined symbol: ncclCommResume. The repo imports TE lazily (after torch)"
echo "  so normal runs are fine, but to make the order irrelevant, export this in your launcher:"
echo "    export LD_LIBRARY_PATH=\$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nccl/lib:\$LD_LIBRARY_PATH"
