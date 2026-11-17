#!/bin/bash
# scripts/experiments/stage1_hd128_auxout.sh [arm ...]: the reviewer's Attn-QAT comparison on the head_dim-128 cell
# (1.665 B GatedDeltaNet : GQA 3:1, 28,610 steps of 128 x 8,192 tokens = 30.0 B tokens, every setting as in
# stage1_hd128.sh), seed 11785:
#   dm_auxout_hd128   Auxiliary-output delta (Attn-QAT's correction on the Delta-Matching fp8 kernels: delta =
#                     rowsum(dequant(dO8) * O'), O' = bf16(P) @ V-hat saved by the forward; no pass 0)
#   dm_matched_hd128  Delta-Matching, rerun beside it on the same kernel build and node count, so that the two differ
#                     only in how delta is formed.
# Driver and knobs: _run.sh.
EXP_NAME=stage1_hd128_auxout; EXP_STAGE=1; EXP_SEEDS="11785"
EXP_ARMS=(dm_auxout_hd128 dm_matched_hd128)
source "$(dirname "${BASH_SOURCE[0]}")/_run.sh" "$@"
