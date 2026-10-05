#!/bin/bash
# scripts/experiments/stage1_muon.sh [arm ...]: the head_dim-128 cell trained with Muon (--optim.name muon: Muon on every
# 2-D hidden weight matrix, AdamW on the rest, src/pretrain/parallel/muon.py), three arms, seed 11785 (the paper's one
# seed; SEEDS="11785 2" adds a second):
#   muon_clean_hd128 (Standard BF16/FP32 Mix-precision), muon_dm_stale_hd128 (Naive FP8 (with Stale Delta)),
#   muon_dm_matched_hd128 (Delta-Matching). The AdamW references are stage1_hd128.sh's runs. Driver: _run.sh.
EXP_NAME=stage1_muon; EXP_STAGE=1; EXP_SEEDS="11785"
EXP_ARMS=(muon_clean_hd128 muon_dm_stale_hd128 muon_dm_matched_hd128)
source "$(dirname "${BASH_SOURCE[0]}")/_run.sh" "$@"
