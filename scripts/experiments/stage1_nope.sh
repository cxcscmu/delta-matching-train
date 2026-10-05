#!/bin/bash
# scripts/experiments/stage1_nope.sh [arm ...]: the head_dim-128 cell without positional encoding (--model.nope true: the
# rotary embedding replaced by the identity in every attention layer), three arms, seeds 11785 and 2:
#   nope_clean_hd128 (Standard BF16/FP32 Mix-precision), nope_dm_stale_hd128 (Naive FP8 (with Stale Delta)),
#   nope_dm_matched_hd128 (Delta-Matching). Driver: _run.sh.
EXP_NAME=stage1_nope; EXP_STAGE=1; EXP_SEEDS="11785 2"
EXP_ARMS=(nope_clean_hd128 nope_dm_stale_hd128 nope_dm_matched_hd128)
source "$(dirname "${BASH_SOURCE[0]}")/_run.sh" "$@"
