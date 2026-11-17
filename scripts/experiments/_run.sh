#!/bin/bash
# scripts/experiments/_run.sh: the shared driver of the experiment scripts in this directory (sourced, not run).
# An experiment script sets EXP_NAME, EXP_STAGE (1 | 2), EXP_ARMS (array), EXP_SEEDS (the default seeds) and optionally
# EXP_ARM_SEEDS (an associative array: per-arm default seeds), then sources this file with its own arguments.
#
#   <experiment script> [arm ...]     every arm of the experiment by default, or the arms named (they must belong to it)
#
# The plan: for each arm, for each of its seeds, one run (train with scripts/campaign/train_arm.sh, or stage2_arm.sh for
# stage 2; evaluate with eval_arm.sh, or eval_arm_64k.sh). Every step resumes and skips what is already complete, so the
# same command continues an interrupted experiment; a scripts/slurm chain runs it once per job (a 4-hour link). One run
# trains at a time, on DM_NPROC GPUs per node and, in a Slurm job of several nodes, on every node of the job
# (scripts/campaign/train_arm.sh); to train several runs at once, give each node or chain its own arms or seeds
# (scripts/experiments/README.md). Each invocation works in two phases:
#   A. the evaluations still owed, first, while the job has all its time: every planned run whose export carries
#      TRAIN_COMPLETE but has no evaluation marker (EVAL_COMPLETE; stage 2: EVAL64K_COMPLETE) is evaluated when the
#      evaluation fits in the job, else it stays deferred;
#   B. in plan order: a run whose export carries TRAIN_COMPLETE is skipped without calling its launcher; any other run
#      is trained (no training starts less than DM_MIN_TRAIN_TIME before the trainer's deadline), then evaluated when
#      that fits in what is left of the job; an evaluation that does not fit is deferred to the next job's phase A and
#      the time left goes to the next run's training.
# env: SEEDS="11785 2" (override the seeds of every arm); DRY=1 (print the plan: arm, seed, run name, commands; run
#      nothing); SMOKE=1 (a 3-step training smoke of every arm, no evaluation; train_arm.sh SMOKE semantics);
#      EVAL=0 (train only); DM_EVAL_BUDGET (seconds; see below); DM_MIN_TRAIN_TIME (600 s); DM_RUN_MAX_FAILS (2; see
#      below); every knob of train_arm.sh / stage2_arm.sh / eval_arm*.sh passes through.
# Deadline (scripts/slurm/job.sh): DM_STOP_BEFORE is the trainer's deadline, DM_JOB_END the job's end (default
# DM_STOP_BEFORE + DM_STOP_MARGIN, 600 s). An evaluation starts only when DM_JOB_END leaves DM_EVAL_BUDGET seconds
# (default 2400 for stage 1, 11700 for stage 2: the ~25 min / ~3 h they take plus slack; a job that started less than
# 10 minutes ago evaluates anyway: the next job would have no more time). A SIGTERM (the time limit, scancel) lets the
# running step shut down (the trainer saves a checkpoint), then this script stops with 75.
# Failures in a chain (DM_CHAIN_DIR set; scripts/slurm/job.sh sets it only when a next job will run): a configuration
# error (exit 2: a refused resume, missing data, a bad knob; the same in every job) gives that training (or evaluation)
# up at once; any other failure (an exit status but 0, 75, 2 and 3) is counted in $DM_CHAIN_DIR/runfails/<run>.train or
# <run>.eval ("<metric> <count> <job>"; metric = the run's newest complete checkpoint step, or the number of completion
# markers under its eval directory; job = <SLURM_JOB_ID>.<SLURM_RESTART_COUNT>): the count restarts at 1 when the
# metric grew since the previous failure (progress), stays as it is when the previous failure was in this same job (its
# in-place retry by job.sh: a job counts once, and the next one runs elsewhere), else it grows by one; at
# DM_RUN_MAX_FAILS jobs in a row that training (or evaluation) is given up for the rest of the chain (a new
# scripts/slurm/submit.sh clears runfails/). Success removes the counter. Outside a chain a failure is reported and the
# others go on. Exit 3 from a launcher (another live job holds the run's lock) lists the run as owned by another job.
# Exit status: 0 when every requested run is trained and evaluated (or smoke-tested); 75 when work remains (a stop or an
# early stop with a checkpoint, a deferred evaluation, a training not started for lack of time, or, in a chain, a
# failure that is retried while another step of this job succeeded): relaunch the same command (a scripts/slurm chain
# does); 1 in a chain when a failure is retried and no step of this job succeeded (an exit 0 of a launcher): a failed
# job, so that job.sh retries it in place and keeps the next job off the node(s) it failed on; otherwise, when only
# given-up runs, runs
# owned by another job or (outside a chain) failures are left, 4 in a chain and 1 outside one. A summary lists what is
# done, deferred, retried, given up and owned.
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
for v in DM_STOP_BEFORE DM_JOB_END DM_STOP_MARGIN DM_EVAL_BUDGET DM_MIN_TRAIN_TIME DM_RUN_MAX_FAILS SMOKE; do
  case "${!v-}" in *[!0-9]*) echo "FATAL: $v='${!v}' is not a whole number" >&2; exit 2 ;; esac
  [ -n "${!v-}" ] && printf -v "$v" '%s' "$((10#${!v}))"   # base 10: a leading 0 must not read as octal
done
SMOKE=${SMOKE:-0}; MAXF=${DM_RUN_MAX_FAILS:-2}
[ "$MAXF" -ge 1 ] || { echo "FATAL: DM_RUN_MAX_FAILS=$MAXF: at least 1" >&2; exit 2; }
if [ "$EXP_STAGE" = 2 ]; then TRAIN=$CAMPAIGN/stage2_arm.sh; EVAL_SH=$CAMPAIGN/eval_arm_64k.sh; STAG=_s2_64k; EVAL_MARK=EVAL64K_COMPLETE
else TRAIN=$CAMPAIGN/train_arm.sh; EVAL_SH=$CAMPAIGN/eval_arm.sh; STAG=""; EVAL_MARK=EVAL_COMPLETE; fi
SWD=$DM_WORKSPACE/qwen35_m
CHAIN=${DM_CHAIN_DIR:-}   # in a scripts/slurm chain: the per-run failure counters live under it
# the name train_arm.sh trains a run under (a smoke adds _smoke<steps>): its checkpoints and its failure counter
SMOKE_SFX=""; [ "$SMOKE" != 0 ] && SMOKE_SFX="_smoke$([ "$SMOKE" = 1 ] && echo 3 || echo "$SMOKE")"
seeds_of () {   # the seeds of one arm: SEEDS, else the experiment's per-arm default, else EXP_SEEDS
  if [ -n "${SEEDS:-}" ]; then echo "$SEEDS"; return; fi
  if declare -p EXP_ARM_SEEDS >/dev/null 2>&1 && [ -n "${EXP_ARM_SEEDS[$1]:-}" ]; then echo "${EXP_ARM_SEEDS[$1]}"; return; fi
  echo "$EXP_SEEDS"
}
say () { echo "[$(hostname -s) $(date '+%F %T')] $*"; }
trained () { [ -f "$SWD/hf/$1/TRAIN_COMPLETE" ]; }
evaluated () { [ -f "$SWD/eval/$1/$EVAL_MARK" ]; }
job_end () {   # the job's end (epoch seconds): DM_JOB_END, else DM_STOP_BEFORE + DM_STOP_MARGIN; nothing without a deadline
  if [ -n "${DM_JOB_END:-}" ]; then echo "$DM_JOB_END"
  elif [ -n "${DM_STOP_BEFORE:-}" ]; then echo $((DM_STOP_BEFORE + ${DM_STOP_MARGIN:-600})); fi
}
eval_fits () {   # whether an evaluation fits in what is left of this job (always without a deadline)
  local end t budget
  end=$(job_end); [ -n "$end" ] || return 0
  printf -v t '%(%s)T' -1
  budget=${DM_EVAL_BUDGET:-$([ "$EXP_STAGE" = 2 ] && echo 11700 || echo 2400)}
  [ $((end - t)) -ge "$budget" ] && return 0
  [[ ${SLURM_JOB_START_TIME:-} =~ ^[0-9]+$ ]] && [ $((t - SLURM_JOB_START_TIME)) -lt 600 ] && return 0   # the next job has no more time
  return 1
}
deadline_near () {   # whether the trainer's deadline is too close to start a training (it would save after one step)
  [ -n "${DM_STOP_BEFORE:-}" ] || return 1
  local t; printf -v t '%(%s)T' -1
  [ $((DM_STOP_BEFORE - t)) -lt "${DM_MIN_TRAIN_TIME:-600}" ]
}
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
    if [ "$SMOKE" != 0 ]; then echo "  $arm seed $seed: SMOKE=${SMOKE} bash $TRAIN $arm $seed"
    else echo "  $arm seed $seed -> $rn: bash $TRAIN $arm $seed; bash $EVAL_SH $rn"; fi
    if [ "$EXP_STAGE" = 2 ] && ! trained "$rn" && [ "$(dm_ckpt_step "$SWD/ckpts/$rn$SMOKE_SFX")" -lt 0 ]; then
      # a stage-2 run starts from the stage-1 clean_hd128 export of its seed, which must be finished (stage2_arm.sh)
      init=$(TAG='' NAME_ONLY=1 dm_run_name clean_hd128 "$seed")
      [ -f "$SWD/hf/$init/TRAIN_COMPLETE" ] || echo "    (cannot start yet: its warm-start export $init in $SWD/hf is not finished (no TRAIN_COMPLETE))"
    fi
  done
  exit 0
fi
# ---- the per-run failure counters of a chain: "<metric> <count> <job>" in $CHAIN/runfails/<key>; I/O is best effort
# JOB_INST: this job (its instance: a requeue is another one); job.sh's in-place retry runs this script again in it
JOB_INST=""
if [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
  JOB_INST="$SLURM_JOB_ID.0"; [[ ${SLURM_RESTART_COUNT:-} =~ ^[0-9]+$ ]] && JOB_INST="$SLURM_JOB_ID.$((10#$SLURM_RESTART_COUNT))"
fi
FM=-1; FC=0; FJ=""
fails_read () {   # fails_read <key>: FM, FC, FJ = the metric, the count and the job stored for <key> (-1, 0, "")
  local m="" c="" j=""
  FM=-1; FC=0; FJ=""
  [ -n "$CHAIN" ] || return 0
  { read -r m c j < "$CHAIN/runfails/$1"; } 2>/dev/null
  [[ $m =~ ^-?[0-9]+$ ]] && FM=$m
  [[ $c =~ ^[0-9]+$ ]] && FC=$c
  [[ $j =~ ^[0-9]+\.[0-9]+$ ]] && FJ=$j
  return 0
}
capped () {   # capped <key>: whether <key> has been given up for this chain
  [ -n "$CHAIN" ] || return 1
  fails_read "$1"; [ "$FC" -ge "$MAXF" ]
}
fails_write () {   # fails_write <key> <metric> <count>: store the counter (atomic; best effort)
  local tmp=$CHAIN/runfails/$1.tmp.$$
  { mkdir -p "$CHAIN/runfails" && printf '%s %s %s\n' "$2" "$3" "${JOB_INST:--}" > "$tmp" && mv -f "$tmp" "$CHAIN/runfails/$1"; } 2>/dev/null \
    || { rm -f "$tmp" 2>/dev/null; say "WARN: could not record the failure in $CHAIN/runfails/$1"; }
}
fails_note () {   # fails_note <key> <metric now>: count the failure (FC); status 0 while it is still retried
  fails_read "$1"
  if [ "$2" -gt "$FM" ]; then FC=1                                   # progress since the previous failure
  elif [ -n "$JOB_INST" ] && [ "$JOB_INST" = "$FJ" ]; then [ "$FC" -ge 1 ] || FC=1   # again in this job: counted once
  else FC=$((FC + 1)); fi
  fails_write "$1" "$2" "$FC"
  [ "$FC" -lt "$MAXF" ]
}
fails_clear () { [ -n "$CHAIN" ] && rm -f "$CHAIN/runfails/$1" 2>/dev/null; return 0; }
# ---- the outcome of each run in this job: done | deferred | notstarted | stopped (an early stop or a SIGTERM) |
# "retry <step>" | "givenup <step>" | "owned <step>" | "failed <step>" (step = train | eval); a run not reached has none.
# WORKED=1 once a launcher of this job exited 0 (a training or an evaluation finished here).
declare -A ST=()
WORKED=0
failed () {   # failed <run> <train|eval> <metric> <exit status> <message>
  local rn=$1 step=$2 key=$1.$2
  [ "$step" = train ] && key=$1$SMOKE_SFX.train
  if [ -z "$CHAIN" ]; then say "$5 (exit $4)"; ST[$rn]="failed $step"; return 0; fi
  if [ "$4" = 2 ]; then   # a configuration error (a refused resume, missing data, a bad knob): the same in every job
    FC=$MAXF; fails_write "$key" "$3" "$FC"
    say "$5 (exit 2: a configuration error, e.g. a refused resume): given up for the rest of this chain at once (fix the cause, then resubmit)"
    ST[$rn]="givenup $step"; return 0
  fi
  if fails_note "$key" "$3"; then
    say "$5 (exit $4): failure $FC of $MAXF without progress (DM_RUN_MAX_FAILS; a job counts once): it is retried"; ST[$rn]="retry $step"
  else
    say "$5 (exit $4): $FC failures without progress (DM_RUN_MAX_FAILS=$MAXF): given up for the rest of this chain (a new submission resets it)"
    ST[$rn]="givenup $step"
  fi
}
summary () {   # the outcome of every planned run, by kind; FINAL = the exit status it implies
  local p arm seed rn s ndone=0 remain=0
  local -A L=()
  for p in "${PLAN[@]}"; do
    read -r arm seed rn <<< "$p"
    s=${ST[$rn]:-}
    case $s in
      done) ndone=$((ndone + 1)) ;;
      deferred|notstarted|stopped) L[$s]+=" $rn"; remain=1 ;;
      "retry "*|"givenup "*|"owned "*|"failed "*) L[${s%% *}]+=" $rn:${s#* }"; [ "${s%% *}" = retry ] && remain=1 ;;
      *) L[later]+=" $rn"; remain=1 ;;
    esac
  done
  say "$EXP_NAME: $ndone of ${#PLAN[@]} runs done"
  [ -n "${L[deferred]:-}" ] && say "  evaluation deferred to the next job:${L[deferred]}"
  [ -n "${L[notstarted]:-}" ] && say "  training not started (too close to the deadline):${L[notstarted]}"
  [ -n "${L[stopped]:-}" ] && say "  stopped (an early stop or a SIGTERM; it resumes in the next job):${L[stopped]}"
  [ -n "${L[later]:-}" ] && say "  not reached in this job:${L[later]}"
  [ -n "${L[retry]:-}" ] && say "  failed, retried (in place by scripts/slurm/job.sh, or by the next job):${L[retry]}"
  [ -n "${L[givenup]:-}" ] && say "  given up for this chain (DM_RUN_MAX_FAILS):${L[givenup]}"
  [ -n "${L[owned]:-}" ] && say "  owned by another job (its lock was busy):${L[owned]}"
  [ -n "${L[failed]:-}" ] && say "  failed:${L[failed]}"
  if [ "$ndone" -eq ${#PLAN[@]} ]; then FINAL=0
  elif [ "$remain" = 1 ]; then
    # nothing of this job succeeded and a failure is retried: a failed job (job.sh retries it in place once, then keeps
    # the next job off the node(s) it failed on); otherwise the chain simply goes on
    if [ -n "$CHAIN" ] && [ "$WORKED" = 0 ] && [ -n "${L[retry]:-}" ]; then FINAL=1; else FINAL=75; fi
  elif [ -n "$CHAIN" ]; then FINAL=4
  else FINAL=1; fi
}
stop_now () { say "$1: relaunch this command to resume"; summary; exit 75; }   # the stop paths: exit 75 at once
STOPPED=""
trap 'STOPPED=1' TERM   # the training / evaluation gets the SIGTERM itself; this script stops after it has shut down
run_eval () {   # run_eval <run>: evaluate it now; status 1 when a SIGTERM stopped the evaluation (stop with 75)
  local rn=$1 rc
  [ -n "$STOPPED" ] && { say "$rn: SIGTERM received: its evaluation is left to the next job"; ST[$rn]=stopped; return 1; }
  say "== $rn: evaluate"
  bash "$EVAL_SH" "$rn"; rc=$?
  if [ -n "$STOPPED" ]; then say "$rn evaluation stopped by SIGTERM (exit $rc): relaunch this command to finish it"; ST[$rn]=stopped; return 1; fi
  case $rc in
    0) fails_clear "$rn.eval"; ST[$rn]=done; WORKED=1 ;;
    3) say "$rn: its evaluation is running in another job (the lock is busy): skipped"; ST[$rn]="owned eval" ;;
    *) failed "$rn" eval "$(dm_eval_markers "$SWD/eval/$rn")" "$rc" "$rn evaluation incomplete" ;;
  esac
  return 0
}
maybe_eval () {   # maybe_eval <run>: evaluate a trained run now when that fits in this job, else defer it
  local rn=$1
  if capped "$rn.eval"; then say "$rn: its evaluation is given up for this chain (DM_RUN_MAX_FAILS)"; ST[$rn]="givenup eval"; return 0; fi
  if ! eval_fits; then
    say "$rn: no time left in this job for its evaluation (DM_EVAL_BUDGET): the next job evaluates it first"; ST[$rn]=deferred; return 0
  fi
  run_eval "$rn"
}
# A. the evaluations owed by earlier jobs, while this job has all its time
if [ "$SMOKE" = 0 ] && [ "${EVAL:-1}" != 0 ]; then
  for p in "${PLAN[@]}"; do
    read -r arm seed rn <<< "$p"
    trained "$rn" && ! evaluated "$rn" || continue
    [ -n "$STOPPED" ] && stop_now "stopped by SIGTERM"
    maybe_eval "$rn" || { summary; exit 75; }   # a SIGTERM stopped the evaluation
  done
fi
# B. train each run (an early stop ends this job); evaluate what this job finished when it fits
for p in "${PLAN[@]}"; do
  read -r arm seed rn <<< "$p"
  [ -n "$STOPPED" ] && stop_now "stopped by SIGTERM"
  if [ "$SMOKE" = 0 ] && trained "$rn"; then          # finished: its launcher is not called at all
    if [ "${EVAL:-1}" = 0 ] || evaluated "$rn"; then ST[$rn]=done; continue; fi
    [ -n "${ST[$rn]:-}" ] && continue                 # phase A dealt with its evaluation
    # finished since phase A looked (by another job): evaluated like a run this job trained
  else
    if capped "$rn$SMOKE_SFX.train"; then say "$rn: its training is given up for this chain (DM_RUN_MAX_FAILS)"; ST[$rn]="givenup train"; continue; fi
    if deadline_near; then
      say "$rn: the trainer deadline is less than ${DM_MIN_TRAIN_TIME:-600} s away: its training is left to the next job"; ST[$rn]=notstarted; continue
    fi
    [ -n "$STOPPED" ] && stop_now "stopped by SIGTERM"
    say "== $arm seed $seed ($rn): train"
    bash "$TRAIN" "$arm" "$seed"; rc=$?
    if [ "$rc" = 75 ]; then say "$rn stopped early with a checkpoint: relaunch this command to resume"; ST[$rn]=stopped; summary; exit 75; fi
    if [ -n "$STOPPED" ]; then say "$rn training stopped by SIGTERM (exit $rc): relaunch this command to resume"; ST[$rn]=stopped; summary; exit 75; fi
    case $rc in
      0) fails_clear "$rn$SMOKE_SFX.train"; WORKED=1 ;;
      3) say "$rn: its training is running in another job (the run lock is busy): skipped"; ST[$rn]="owned train"; continue ;;
      *) failed "$rn" train "$(dm_ckpt_step "$SWD/ckpts/$rn$SMOKE_SFX")" "$rc" "$rn training FAILED"; continue ;;
    esac
    if [ "$SMOKE" != 0 ] || [ "${EVAL:-1}" = 0 ]; then ST[$rn]=done; continue; fi
  fi
  maybe_eval "$rn" || { summary; exit 75; }     # a SIGTERM stopped the evaluation
done
summary
[ "$FINAL" = 0 ] && say "$EXP_NAME: all ${#PLAN[@]} runs done"
if [ "$FINAL" = 1 ]; then
  if [ -n "$CHAIN" ]; then say "$EXP_NAME: nothing of this job succeeded and a failure is retried: exit 1, a failed job (scripts/slurm/job.sh retries it in place, then keeps the next job off the node(s) it failed on)"
  else say "$EXP_NAME: incomplete: nothing left that this command can do (see above)"; fi
fi
[ "$FINAL" = 4 ] && say "$EXP_NAME: nothing left that this chain can do (see above): it stops; resubmit once the cause is fixed"
[ "$FINAL" = 75 ] && say "$EXP_NAME: work remains: relaunch this command to continue (a scripts/slurm chain does)"
exit "$FINAL"
