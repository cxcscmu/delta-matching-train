#!/bin/bash
# scripts/experiments/stage1_xl.sh [arm ...]: the 9.275 B GatedDeltaNet : GQA 3:1 cell (qwen35_xl.yaml: the Qwen3.5-9B-Base
# block at 36 layers; head_dim 256; 30.0 B tokens), three arms, seed 11785:
#   xl_clean_hd256       Standard BF16/FP32 Mix-precision (FlashAttention-3 bf16 attention, fp8 FFN)
#   xl_dm_stale_hd256    Naive FP8 (with Stale Delta)
#   xl_dm_matched_hd256  Delta-Matching
# The recipe of stage1_hd256_scale.sh with activation checkpointing on (selective: the 9.3 B cell does not fit 80 GB
# without it, qwen35_xl.yaml's header). A 9.3 B export takes longer to evaluate than a 5.3 B one, hence the larger default
# evaluation budget (DM_EVAL_BUDGET, _run.sh). Driver: _run.sh.
# One chain per arm (scripts/slurm/README.md), e.g. on 32 GPUs (estimated 41-46 h per run):
#   scripts/slurm/submit.sh -p batch -N 8 --gpus-per-node=4 --cpus-per-task=64 --mem=800G -t 4:00:00 -- \
#     scripts/experiments/stage1_xl.sh xl_clean_hd256
# LR: 2.4e-3, the ladder's, untested at this width; a clean-arm hedge at 1.2e-3 is a run (and a chain) of its own:
#   TAG=_lr1p2e3 DM_LR=1.2e-3 scripts/slurm/submit.sh ... -- scripts/experiments/stage1_xl.sh xl_clean_hd256
EXP_NAME=stage1_xl; EXP_STAGE=1; EXP_SEEDS="11785"
EXP_ARMS=(xl_clean_hd256 xl_dm_stale_hd256 xl_dm_matched_hd256)
DM_EVAL_BUDGET=${DM_EVAL_BUDGET:-8400}
source "$(dirname "${BASH_SOURCE[0]}")/_run.sh" "$@"
