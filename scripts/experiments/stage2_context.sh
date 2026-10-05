#!/bin/bash
# scripts/experiments/stage2_context.sh [arm ...]: stage 2, the context extension of the head_dim-128 GatedDeltaNet : GQA
# cell from 8K to 64K (YaRN, 28,000 steps of 16 x 65,536 tokens on ProLong-64K; the recipe: scripts/campaign/stage2_arm.sh),
# four arms, seeds 11785 and 2. Every arm warm-starts from clean_hd128's stage-1 export of the same seed, so run
# stage1_hd128.sh (at least clean_hd128) first.
#   clean_hd128 (Standard BF16/FP32 Mix-precision), dm_matched_hd128 (Delta-Matching),
#   dm_stale_hd128 (Naive FP8 (with Stale Delta)), te_fp8 (cuDNN / Transformer Engine FP8). Driver: _run.sh.
EXP_NAME=stage2_context; EXP_STAGE=2; EXP_SEEDS="11785 2"
EXP_ARMS=(clean_hd128 dm_matched_hd128 dm_stale_hd128 te_fp8)
source "$(dirname "${BASH_SOURCE[0]}")/_run.sh" "$@"
