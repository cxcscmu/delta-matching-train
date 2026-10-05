#!/bin/bash
# scripts/campaign/stage2_arm.sh <arm> [seed] [init_run_name]
# Stage 2, the context extension: continue a stage-1 model of the head_dim-128 GatedDeltaNet : GQA cell at 65,536 tokens
# with YaRN (factor 8, theta unchanged, mscale^2 temperature) on the ProLong-64K corpus (data/pretrain_stage2.md), through
# train_arm.sh with the same per-arm environment and kernels as stage 1. Arms: clean_hd128 (Standard BF16/FP32
# Mix-precision), dm_matched_hd128 (Delta-Matching), dm_stale_hd128 (Naive FP8 (with Stale Delta)), te_fp8 (cuDNN /
# Transformer Engine FP8). Every arm warm-starts from the Standard arm's stage-1 export of the same seed
# (Q35M_clean_sdpa_ffn8_hd128_<seed>), so each arm's result is a paired contrast against the Standard arm's stage-2 run;
# a third argument names another export under $DM_WORKSPACE/qwen35_m/hf/. Run name = <stage-1 run name>_s2_64k.
# The recipe (the paper's): global batch 16 x 64k = 1,048,576 tokens per step, full gradient checkpointing, 28,000 steps
# (29.4 B tokens) with a fresh AdamW: a 100-step linear re-warm to 8e-4, constant to step 26,000, then linear decay to 0 at
# 28,000; data.chunk_subset [0, 0.85] (481,906 of the 566,949 training windows: no window repeats in 448,000 draws); an HF
# export every 2,000 steps (<export dir>/step<N>) and the final one.
# env: S2_LR (8e-4), S2_STEPS (28000), S2_WARM_STEPS (100), S2_DECAY_STEPS (2000), S2_CHUNKS ("[0.0,0.85]"),
#      S2_EXPORT_EVERY (2000; 0 = none), S2_PDBS (per-device batch, 1: fits 80 GB GPUs; 2 where memory allows gives the
#      same gradient with half the accumulation steps), S2_DATA (the corpus dir), S2_DRY=1 (print the composed arguments and
#      exit 0 without training); SMOKE / STOP_AT_STEP / TAG / DM_SAVE_INTERVAL (1000) as train_arm.sh.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/env.sh"; source "$HERE/arms.sh"
set -uo pipefail
ARM=${1:?arm}; SEED=${2:-11785}
case $ARM in
  clean_hd128|dm_matched_hd128|dm_stale_hd128|te_fp8) ;;
  *) echo "FATAL: stage 2 runs the head_dim-128 GatedDeltaNet : GQA arms clean_hd128, dm_matched_hd128, dm_stale_hd128 and te_fp8, not $ARM" >&2; exit 2 ;;
esac
INIT=${3:-$(TAG='' NAME_ONLY=1 dm_run_name clean_hd128 "$SEED")}   # TAG-free: the stage-1 export name never carries this run's tag
INIT_DIR=$DM_WORKSPACE/qwen35_m/hf/$INIT
[ -f "$INIT_DIR/model.safetensors" ] || { echo "FATAL: warm-start export missing: $INIT_DIR/model.safetensors (train clean_hd128 seed $SEED first)" >&2; exit 2; }
S2_DATA=${S2_DATA:-$DM_DATA_ROOT/prolong_64k_llama2}
[ -f "$S2_DATA/prolong64k_train.bin" ] && [ -f "$S2_DATA/prolong64k_train.doc_lengths.npy" ] || { echo "FATAL: ProLong-64K corpus missing under $S2_DATA (data/pretrain_stage2.md)" >&2; exit 2; }
S2_LR=${S2_LR:-8e-4}; S2_STEPS=${S2_STEPS:-28000}; S2_CHUNKS=${S2_CHUNKS:-"[0.0,0.85]"}
S2_WARM_STEPS=${S2_WARM_STEPS:-100}; S2_DECAY_STEPS=${S2_DECAY_STEPS:-2000}; S2_EXPORT_EVERY=${S2_EXPORT_EVERY:-2000}; S2_PDBS=${S2_PDBS:-1}
# The trainer takes the warmup and the decay as ratios of num_steps and truncates (int(ratio * num_steps)); the ratios
# are composed half a step up so the step counts come out exact (a ratio rounded down would lose a warmup step).
S2_SCHED=$("$DM_PY" -c "
w, d, n, e = int($S2_WARM_STEPS), int($S2_DECAY_STEPS), int($S2_STEPS), int($S2_EXPORT_EVERY)
assert 0 <= w and 0 <= d and w + d <= n, 'S2_WARM_STEPS + S2_DECAY_STEPS must not exceed S2_STEPS'
wr, dr = (w + 0.5) / n if w else 0.0, (d + 0.5) / n if d else 0.0
assert int(wr * n) == w and int(dr * n) == d
print(repr(wr), repr(dr), '[' + ','.join(str(k) for k in range(e, n, e)) + ']' if e > 0 else '[]')") \
  || { echo "FATAL: could not compose the schedule with DM_PY=$DM_PY (S2_WARM_STEPS=$S2_WARM_STEPS S2_DECAY_STEPS=$S2_DECAY_STEPS S2_STEPS=$S2_STEPS S2_EXPORT_EVERY=$S2_EXPORT_EVERY)" >&2; exit 2; }
read -r S2_WARM_RATIO S2_DECAY_RATIO S2_EXPORTS <<< "$S2_SCHED"
export TAG="_s2_64k${TAG:-}" NUM_STEPS=$S2_STEPS DM_SAVE_INTERVAL=${DM_SAVE_INTERVAL:-1000} WANDB_JT=${WANDB_JT:-stage2}
export DM_LR=$S2_LR   # train_arm.sh passes its own --optim.lr AFTER the extra arguments, so the rate goes through this knob
export EXTRA_ARGS="--init_from_hf $INIT_DIR --model.max_seq_len 65536 \
--model.rope_scaling {\"type\":\"yarn\",\"factor\":8.0,\"original_max_position_embeddings\":8192,\"beta_fast\":32,\"beta_slow\":1} \
--global_batch_size 16 --per_device_batch_size $S2_PDBS --gradient_checkpointing true \
--data.paths $S2_DATA/prolong64k_train.bin --data.doc_lengths $S2_DATA/prolong64k_train.doc_lengths.npy \
--data.chunk_subset $S2_CHUNKS --optim.lr $S2_LR \
--optim.warmup_ratio $S2_WARM_RATIO --optim.decay_ratio $S2_DECAY_RATIO --optim.decay_type linear --hf_export_steps $S2_EXPORTS ${EXTRA_ARGS:-}"
echo "[$(hostname -s) $(date '+%F %T')] STAGE2 $ARM seed $SEED from $INIT: lr $S2_LR steps $S2_STEPS (re-warm $S2_WARM_STEPS, linear decay over the last $S2_DECAY_STEPS) pdbs $S2_PDBS exports $S2_EXPORTS chunks $S2_CHUNKS data $S2_DATA"
if [ "${S2_DRY:-0}" = 1 ]; then echo "S2_DRY: TAG=$TAG NUM_STEPS=$NUM_STEPS DM_LR=$DM_LR EXTRA_ARGS=$EXTRA_ARGS"; exit 0; fi
exec bash "$HERE/train_arm.sh" "$ARM" "$SEED"
