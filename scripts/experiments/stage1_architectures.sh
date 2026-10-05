#!/bin/bash
# scripts/experiments/stage1_architectures.sh [arm ...]: the fp8 attention in other architectures (head_dim 128, 30.0 B
# tokens each), three arms per architecture: <family>_clean_hd128 (Standard BF16/FP32 Mix-precision),
# <family>_dm_stale_hd128 (Naive FP8 (with Stale Delta)), <family>_dm_matched_hd128 (Delta-Matching).
#   mla_*     GatedDeltaNet : MLA 3:1, 1.665 B          kda_*   KDA : GQA 3:1, 1.665 B
#   mamba2_*  Mamba2 : GQA 3:1, 1.666 B                 gqa_*   pure GQA (the Qwen3-1.7B geometry), 1.475 B
# Seeds 11785 and 2; the pure-GQA arms default to 11785 and 1, the seeds of the paper's pure-GQA cell. Driver: _run.sh.
EXP_NAME=stage1_architectures; EXP_STAGE=1; EXP_SEEDS="11785 2"
EXP_ARMS=(mla_clean_hd128 mla_dm_stale_hd128 mla_dm_matched_hd128 kda_clean_hd128 kda_dm_stale_hd128 kda_dm_matched_hd128
          mamba2_clean_hd128 mamba2_dm_stale_hd128 mamba2_dm_matched_hd128 gqa_clean_hd128 gqa_dm_stale_hd128 gqa_dm_matched_hd128)
declare -A EXP_ARM_SEEDS=([gqa_clean_hd128]="11785 1" [gqa_dm_stale_hd128]="11785 1" [gqa_dm_matched_hd128]="11785 1")
source "$(dirname "${BASH_SOURCE[0]}")/_run.sh" "$@"
