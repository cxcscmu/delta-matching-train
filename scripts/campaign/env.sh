#!/bin/bash
# scripts/campaign/env.sh: the environment contract shared by train_arm.sh, stage2_arm.sh, eval_arm.sh, eval_arm_64k.sh
# and the experiment drivers (scripts/experiments/). Source it; every variable has a default relative to the repository
# and can be overridden from the shell. See docs/SETUP.md section 5.
DM_REPO=${DM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
DM_WORKSPACE=${DM_WORKSPACE:-$DM_REPO/workspace}                  # fast local scratch (a symlink or directory in the repo root)
DM_DATA_ROOT=${DM_DATA_ROOT:-$DM_WORKSPACE/data}                     # the data root built by data/ (data/README.md)
# The interpreter: an explicit DM_PY wins, then the repo's own uv environment (docs/SETUP.md section 2), then PATH.
# The venv default matters on a batch node, where `python3` may be a system interpreter without torch.
if [ -z "${DM_PY:-}" ]; then
  if [ -x "$DM_REPO/.venv/bin/python" ]; then DM_PY=$DM_REPO/.venv/bin/python
  else DM_PY=$(command -v python || command -v python3 || true); fi
fi
DM_TORCHRUN=${DM_TORCHRUN:-${DM_PY:+$(dirname "$DM_PY")/torchrun}}
DM_NPROC=${DM_NPROC:-8}                                            # GPUs per node used by torchrun
DM_TOKENIZER_DIR=${DM_TOKENIZER_DIR:-$DM_DATA_ROOT/tokenizer/llama2_32k}
HF_HOME=${HF_HOME:-$DM_DATA_ROOT/hf_cache}                          # the evaluation datasets (data/prepare_eval.py fills it)
export HF_HOME
export PYTHONPATH=$DM_REPO/src${PYTHONPATH:+:$PYTHONPATH}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$DM_WORKSPACE/triton_cache} TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-$DM_WORKSPACE/tilelang_cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# The fp8 attention kernel comes from the installed `delta_matching` package (the delta-matching repository; docs/SETUP.md
# section 3); arms.sh checks its build flags at launch. No binary paths are pinned here.
# wandb: the entity stays unset (the account's default) unless WANDB_ENTITY is exported; project delta-matching; DM_WANDB=0 disables
export WANDB_PROJECT=${WANDB_PROJECT:-delta-matching} WANDB_DIR=${WANDB_DIR:-$DM_WORKSPACE}
export DM_REPO DM_WORKSPACE DM_DATA_ROOT DM_PY DM_TORCHRUN DM_NPROC DM_TOKENIZER_DIR
