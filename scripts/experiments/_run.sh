#!/bin/bash
# scripts/experiments/_run.sh: the shared driver of the experiment scripts in this directory (sourced, not run).
# An experiment script sets EXP_NAME, EXP_STAGE (1 | 2), EXP_ARMS (array), EXP_SEEDS (the default seeds) and optionally
# EXP_ARM_SEEDS (an associative array: per-arm default seeds), then sources this file with its own arguments.
#
#   <experiment script> [arm ...]     every arm of the experiment by default, or the arms named (they must belong to it)
#
# For each seed, for each arm, in order: train (scripts/campaign/train_arm.sh, or stage2_arm.sh for stage 2), then
# evaluate (eval_arm.sh, or eval_arm_64k.sh for stage 2). Both steps resume and skip what is already complete, so the
# same command continues an interrupted experiment. One node runs one run at a time (DM_NPROC GPUs); to use several
# nodes, give each node its own arms or seeds (scripts/experiments/README.md).
# env: SEEDS="11785 2" (override the seeds of every arm); DRY=1 (print the plan: arm, seed, run name, commands; run
#      nothing); SMOKE=1 (a 3-step training smoke of every arm, no evaluation; train_arm.sh SMOKE semantics);
#      EVAL=0 (train only); every knob of train_arm.sh / stage2_arm.sh / eval_arm*.sh passes through.
# Exit status: 0 when every requested run is trained and evaluated (or smoke-tested); 75 when a training run stopped
# early with a checkpoint (relaunch to resume); 1 otherwise, with a summary of what is missing.
HERE_EXP=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAMPAIGN=$(cd "$HERE_EXP/../campaign" && pwd)
source "$CAMPAIGN/env.sh"; source "$CAMPAIGN/arms.sh"
set -uo pipefail
: "${EXP_NAME:?} ${EXP_STAGE:?}"
[ ${#EXP_ARMS[@]} -gt 0 ] || { echo "FATAL: $EXP_NAME lists no arms" >&2; exit 2; }
if [ $# -gt 0 ]; then
  for a in "$@"; do
    case " ${EXP_ARMS[*]} " in *" $a "*) ;; *) echo "FATAL: $a is not an arm of $EXP_NAME (its arms: ${EXP_ARMS[*]})" >&2; exit 2 ;; esac
  done
  RUN_ARMS=("$@")
else
  RUN_ARMS=("${EXP_ARMS[@]}")
fi
if [ "$EXP_STAGE" = 2 ]; then TRAIN=$CAMPAIGN/stage2_arm.sh; EVAL_SH=$CAMPAIGN/eval_arm_64k.sh; STAG=_s2_64k
else TRAIN=$CAMPAIGN/train_arm.sh; EVAL_SH=$CAMPAIGN/eval_arm.sh; STAG=""; fi
seeds_of () {   # the seeds of one arm: SEEDS, else the experiment's per-arm default, else EXP_SEEDS
  if [ -n "${SEEDS:-}" ]; then echo "$SEEDS"; return; fi
  if declare -p EXP_ARM_SEEDS >/dev/null 2>&1 && [ -n "${EXP_ARM_SEEDS[$1]:-}" ]; then echo "${EXP_ARM_SEEDS[$1]}"; return; fi
  echo "$EXP_SEEDS"
}
say () { echo "[$(hostname -s) $(date '+%F %T')] $*"; }
PLAN=()   # "arm seed run"
for arm in "${RUN_ARMS[@]}"; do
  for seed in $(seeds_of "$arm"); do
    rn=$(TAG="$STAG${TAG:-}" dm_run_name "$arm" "$seed") || true
    [ -n "$rn" ] || { echo "FATAL: cannot name $arm seed $seed" >&2; exit 2; }
    PLAN+=("$arm $seed $rn")
  done
done
say "$EXP_NAME (stage $EXP_STAGE): ${#PLAN[@]} runs"
if [ "${DRY:-0}" = 1 ]; then
  for p in "${PLAN[@]}"; do
    read -r arm seed rn <<< "$p"
    if [ "${SMOKE:-0}" != 0 ]; then echo "  $arm seed $seed: SMOKE=${SMOKE} bash $TRAIN $arm $seed"
    else echo "  $arm seed $seed -> $rn: bash $TRAIN $arm $seed; bash $EVAL_SH $rn"; fi
  done
  exit 0
fi
FAILED=()
for p in "${PLAN[@]}"; do
  read -r arm seed rn <<< "$p"
  say "== $arm seed $seed ($rn): train"
  bash "$TRAIN" "$arm" "$seed"; rc=$?
  if [ "$rc" = 75 ]; then say "$rn stopped early with a checkpoint: relaunch this command to resume"; exit 75; fi
  if [ "$rc" != 0 ]; then say "$rn training FAILED (exit $rc)"; FAILED+=("$rn:train"); continue; fi
  [ "${SMOKE:-0}" != 0 ] && continue
  [ "${EVAL:-1}" = 0 ] && continue
  say "== $rn: evaluate"
  bash "$EVAL_SH" "$rn"; rc=$?
  [ "$rc" = 0 ] || { say "$rn evaluation incomplete (exit $rc)"; FAILED+=("$rn:eval"); }
done
if [ ${#FAILED[@]} -eq 0 ]; then say "$EXP_NAME: all ${#PLAN[@]} runs done"; exit 0; fi
say "$EXP_NAME: incomplete: ${FAILED[*]}"; exit 1
