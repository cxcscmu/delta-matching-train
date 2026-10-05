#!/bin/bash
# scripts/experiments/stage1_hd256_scale.sh [arm ...]: head_dim 256 at three sizes (GatedDeltaNet : GQA 3:1, 30.0 B tokens
# each), three arms per size, seeds 11785 and 2: 0.569 B (xs_*), 1.665 B (no prefix), 5.289 B (l_*).
#   *clean_hd256       Standard BF16/FP32 Mix-precision (FlashAttention-3 bf16 attention, fp8 FFN)
#   *dm_stale_hd256    Naive FP8 (with Stale Delta)
#   *dm_matched_hd256  Delta-Matching
# Every size runs the full fp8 pipeline (fp8 FFN and q/k/v/o, the Delta-Matching kernel at head_dim 256). Driver: _run.sh.
EXP_NAME=stage1_hd256_scale; EXP_STAGE=1; EXP_SEEDS="11785 2"
EXP_ARMS=(xs_clean_hd256 xs_dm_stale_hd256 xs_dm_matched_hd256 clean_hd256 dm_stale_hd256 dm_matched_hd256
          l_clean_hd256 l_dm_stale_hd256 l_dm_matched_hd256)
source "$(dirname "${BASH_SOURCE[0]}")/_run.sh" "$@"
