#!/bin/bash
# scripts/experiments/stage1_l_kda.sh [arm ...]: the 5.293 B KDA : GQA 3:1 cell (qwen35_l_kda.yaml: the KDA twin of the
# 5.289 B GatedDeltaNet : GQA l_ cell of stage1_hd256_scale.sh; head_dim 256, 30.0 B tokens), three arms, seed 11785:
#   l_kda_clean_hd256       Standard BF16/FP32 Mix-precision (FlashAttention-3 bf16 attention, fp8 FFN)
#   l_kda_dm_stale_hd256    Naive FP8 (with Stale Delta)
#   l_kda_dm_matched_hd256  Delta-Matching
# The recipe of stage1_hd256_scale.sh: the full fp8 pipeline (fp8 FFN and q/k/v/o, the Delta-Matching kernel at head_dim
# 256) for the fp8 arms; the KDA projections stay bf16 in every arm. A 5.3 B export takes longer to evaluate than a
# 1.665 B one, hence the larger default evaluation budget (DM_EVAL_BUDGET, _run.sh). Driver: _run.sh.
# One chain per arm (scripts/slurm/README.md), e.g. on 16 GPUs:
#   scripts/slurm/submit.sh -p batch -N 4 --gpus-per-node=4 --cpus-per-task=64 --mem=800G -t 4:00:00 -- \
#     scripts/experiments/stage1_l_kda.sh l_kda_clean_hd256
# LR: 2.4e-3, the GDN 5.3 B cell's; the yaml's header records why a clean-arm hedge at 1.2e-3 runs through the warmup:
#   TAG=_lr1p2e3 DM_LR=1.2e-3 scripts/slurm/submit.sh ... -- scripts/experiments/stage1_l_kda.sh l_kda_clean_hd256
EXP_NAME=stage1_l_kda; EXP_STAGE=1; EXP_SEEDS="11785"
EXP_ARMS=(l_kda_clean_hd256 l_kda_dm_stale_hd256 l_kda_dm_matched_hd256)
DM_EVAL_BUDGET=${DM_EVAL_BUDGET:-4800}
source "$(dirname "${BASH_SOURCE[0]}")/_run.sh" "$@"
