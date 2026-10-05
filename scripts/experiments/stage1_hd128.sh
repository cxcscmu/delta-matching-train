#!/bin/bash
# scripts/experiments/stage1_hd128.sh [arm ...]: the head_dim-128 cell (1.665 B GatedDeltaNet : GQA 3:1, 28,610 steps of
# 128 x 8,192 tokens = 30.0 B tokens), five arms, seeds 11785 and 2. The paper's main comparison.
#   clean_hd128          Standard BF16/FP32 Mix-precision
#   dm_stale_hd128       Naive FP8 (with Stale Delta)
#   dm_matched_hd128     Delta-Matching
#   dm_consistent_hd128  Naive FP8 (Consistent dO)
#   te_fp8               cuDNN / Transformer Engine FP8 (needs the Transformer Engine environment: DM_TE_PY, DM_TE_TORCHRUN)
# The stage-2 experiment (stage2_context.sh) warm-starts from clean_hd128's exports. Driver and knobs: _run.sh.
EXP_NAME=stage1_hd128; EXP_STAGE=1; EXP_SEEDS="11785 2"
EXP_ARMS=(clean_hd128 dm_stale_hd128 dm_matched_hd128 dm_consistent_hd128 te_fp8)
source "$(dirname "${BASH_SOURCE[0]}")/_run.sh" "$@"
