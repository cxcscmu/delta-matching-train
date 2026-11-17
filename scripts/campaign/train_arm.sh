#!/bin/bash
# scripts/campaign/train_arm.sh <arm> [seed]
# One arm on one node (DM_NPROC GPUs, 8 by default) or on every node of a Slurm job (below): the run name, per-arm
# environment and trainer arguments come from scripts/campaign/arms.sh (the arm list is at its top), every path from
# scripts/campaign/env.sh. The experiment drivers in scripts/experiments/ call this script once per arm and seed.
#   te_fp8 needs a Python environment with Transformer Engine (docs/SETUP.md section 2): DM_TE_PY / DM_TE_TORCHRUN.
# env: SMOKE=1 (3-step smoke under <run>_smoke3, wandb off, dirs removed) | SMOKE=<n> (n-step probe);
#      STOP_AT_STEP=<n> (stop at n, WSD horizon unchanged; the run ends there and TRAIN_COMPLETE records steps=<n>);
#      NUM_STEPS (28610); EXTRA_ARGS="--flag value ..."; SAVE_TOTAL_LIMIT (2);
#      DM_SAVE_INTERVAL (the arm's SAVE_IV from arms.sh: about an hour of steps, 1000 for the 1.665 B arms);
#      WANDB_JT (stage1); DM_WANDB=0 (no wandb); TAG=<suffix> appended to the run name;
#      DM_LAUNCH=torchrun|env (env = the ranks are started by this script, so a SIGTERM reaches the trainer directly and
#      it saves a checkpoint before exiting 75); DM_STOP_BEFORE=<epoch seconds> (the trainer's time budget: it saves and
#      exits 75 before that time);
#      DM_RANK_GRACE (60): env launch, the seconds the other ranks get after one rank failed (an exit other than 0 and
#      75) before they are SIGKILLed: they are most likely blocked in a collective that only the 30-minute
#      process-group timeout would end;
#      DM_TERM_GRACE (90): env launch under Slurm (SLURM_JOB_ID set), the seconds the ranks get after a forwarded SIGTERM
#      before they are SIGKILLed, below Slurm's KillWait (120 s) so that this script, the driver and job.sh still exit
#      on their own (a save cut short leaves no .complete marker: the previous checkpoint holds);
#      DM_EXIT_GRACE (600): env launch, the seconds the other ranks get after the first rank exited cleanly (0 or 75)
#      before they are SIGKILLed: by then every collective is over (the final export and a checkpoint end with a
#      barrier), so a rank still running is stuck in its teardown, which would otherwise hold the job to its time
#      limit (keep it above DM_WANDB_FINISH_TIMEOUT: the master closes W&B after the others have left);
#      DM_LOCK_WAIT (600): the seconds to wait for the run lock when its holder is ending, has ended or is this job
#      (env.sh dm_lock);
#      DM_PROGRESS_FILE (scripts/slurm/job.sh, in a chain): a "train_complete <run>" line when the run completes;
#      DM_ALLOW_RUN_CHANGE=1 / DM_ALLOW_KERNEL_CHANGE=1 (the trainer: accept a changed run identity / kernel build at the
#      resume). A chain hands them to every later job, where they would let any later change through, so in a chain
#      (DM_CHAIN_DIR) each is dropped once this submission (the last "submit" record of $DM_CHAIN_DIR/links) has recorded
#      the change in this run: DM_ALLOW_RUN_CHANGE when the run's run_identity.json (rewritten after a load that accepted
#      the change) is newer than the submission, DM_ALLOW_KERNEL_CHANGE when its newest complete checkpoint is (saved on
#      the new build);
#      DM_VALCE_PORT (29797): the rendezvous port of the end-of-run val CE;
#      DM_NNODES: the number of nodes to train on (default: the job's, SLURM_JOB_NUM_NODES, inside a Slurm job; else 1);
#      DM_SRUN_EXTRA (words, appended to the srun options of a multi-node launch: a site's needs);
#      DM_SRUN_GRACE (DM_TERM_GRACE + 60): multi-node, the seconds srun gets after a stop signal before its step is killed.
# Several nodes (NNODES > 1: needs DM_LAUNCH=env, which scripts/slurm/job.sh sets, a Slurm job and srun; a bad node
# count or a missing one of these: exit 2). One srun step with one task per node runs scripts/campaign/node_ranks.sh on
# every node, which starts that node's DM_NPROC ranks (global RANK = SLURM_NODEID * DM_NPROC + r, MASTER_ADDR = the
# job's first node, the batch host; MASTER_PORT = DM_MASTER_PORT, else 29517) with the collector of a one-node launch
# (scripts/campaign/ranks.sh). srun runs in a session of its own (setsid -w): a SIGTERM to srun SIGKILLs its step (seen on
# this cluster), so job.sh's `kill -TERM 0` must not reach it, and a stop is never sent to srun: this script passes it
# to the node launcher on its own host (whose ranks stop every node at the same step: the trainer MAX-all-reduces its
# stop flag), or, before that launcher is up, leaves a stop file the launchers look at before they start their ranks.
# The stop signals of a multi-node launch, here and in every node launcher, are job.sh's: TERM, INT, HUP, USR1 and USR2
# (`scancel --signal=<sig> <job>` reaches the srun step's tasks, i.e. the node launchers, not their ranks: Slurm signals
# a step's tasks only, unless the signal is SIGKILL; with --full this script too. A launcher killed by one would leave
# no report, and its ranks no stop); the one-node launch keeps TERM and INT.
# srun still running DM_SRUN_GRACE s after a stop, or DM_EXIT_GRACE s after a node's ranks ended: its step is killed.
# Each node launcher reports "<status> <host> <first failure> <failed ranks>" in $DM_WORKSPACE/logs/nodes/<run>/
# <job id>.<launch>/node<N> (the newest 10 launches are kept); the launch's status: 75 when any node's ranks saved and
# exited 75, else the failure of the node that failed first (else any node's failure), else 0 when every node reported
# 0, else 143 when this script killed the step, else srun's status (1 when 0). Every node's line is printed. The
# launcher of a node exits 0 when its ranks exited 0 or 75 (a node that is done must not make srun's kill-on-bad-exit
# cut the others short while they still save or export), its ranks' failure otherwise.
# In a chain (DM_CHAIN_DIR) a failure (not a stop, not 2) appends "<epoch> <job id> <host>" to $DM_CHAIN_DIR/badnodes
# for each node it is blamed on: the node that failed first, with every node whose first failure came within 2 s of it
# (a tie: each launcher sees a failure at its next once-a-second poll, or, for a rank that failed just before a stop
# reached its launcher, at the stop; srun's kill-on-bad-exit stops the other nodes about a second after the first
# failing node's launcher exited), else the nodes that never reported (a launcher killed, a lost node), else every
# failing node; one node: this host. job.sh keeps the chain's next job off those nodes after a failure without progress.
# Resume: relaunching the same arm and seed resumes from $DM_WORKSPACE/qwen35_m/ckpts/<run>; a finished run (the
# TRAIN_COMPLETE marker in its export) exits 0 at once, before any other check (data, kernel build, imports).
# One trainer per run: an flock on $DM_WORKSPACE/qwen35_m/ckpts/<run>.lock (env.sh dm_lock), taken before any rank starts
# and held by this script and, under the env launch, by every rank (they inherit it); torchrun's workers do not inherit
# it (torch closes the descriptor in them), so under torchrun it is held by this script and torchrun, and under a
# multi-node launch by this script and srun (Slurm starts the node launchers and their ranks).
# <run>.lock.owner names the holder. Another live job holding it: exit 3.
# The rank logs ($DM_WORKSPACE/logs/<run>.rank<r>.log, r = the global rank) are appended to, one header line per launch;
# when the ranks fail, the last 30 lines each failed rank wrote in this launch are printed (by its node's launcher).
# The end of a run: the run is complete when its HF export is whole (python -m pretrain.hf.check_export: config.json,
# which the trainer writes last, parses, and model.safetensors is as long as its header says), whatever the trainer's
# exit status, provided this launch wrote it or the trainer exited 0 (a crash at teardown after the export must not
# re-train the run in every job; an export left by an earlier launch never completes a run that failed now). Then
# config.json is patched for the clean-kernel evaluation (atomically; config.train.json keeps the trained config, and a
# second pass changes nothing), TRAIN_COMPLETE is written atomically ("<date> steps=<trained steps> num_steps=<N>"),
# the progress line appended, and only then the informational 128-window val CE runs (a failure or a SIGTERM there
# changes nothing). SMOKE runs check the same tail, write no marker and remove their directories.
# Exit status: 0 complete (or already complete); 75 stopped early with a complete checkpoint (resume later); 2 a
# configuration error, never retried: before any rank started (data, tokenizer, kernel build, import, unknown arm, a bad
# knob), or the trainer refused the resume ("FATAL: RESUME REFUSED" in the log: the run identity or the kernel build
# differ from the run's record; DM_ALLOW_RUN_CHANGE=1 / DM_ALLOW_KERNEL_CHANGE=1 accept the change); 3 the run is being
# trained by another live job; 1 the trainer exited 0 but its export is incomplete (relaunch: it resumes the
# finished run and exports again), or the run lock was not released within DM_LOCK_WAIT by a holder that is ending,
# has ended or is this job (relaunch, preferably on another node), or the status directory of a multi-node launch
# could not be created; otherwise the trainer's (or torchrun's) failure status: the first rank's failure (never the 137
# of a rank this script killed), 143 when ranks had to be killed after a stop signal or after the others had exited
# (DM_TERM_GRACE, DM_EXIT_GRACE) and no rank saved; several nodes: the launch's status (above).
# The code revision (DM_GIT_REV, unless set): `git describe --always --dirty`, and for a tree with changes under src/
# or scripts/, uncommitted or untracked, "-dirty+<8 hex>" of the sha1 of those changes (the trainer's start event and
# the run identity's env section record it).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/env.sh"; source "$HERE/arms.sh"
ARM=${1:?arm}; SEED=${2:-11785}; SMOKE=${SMOKE:-0}
REPO=$DM_REPO; W=$DM_WORKSPACE; SW=$W/qwen35_m; PY=$DM_PY; TR=$DM_TORCHRUN
for v in SMOKE NUM_STEPS STOP_AT_STEP DM_NPROC; do
  case "${!v-}" in *[!0-9]*) echo "FATAL: $v='${!v}' is not a whole number" >&2; exit 2 ;; esac
  [ -n "${!v-}" ] && printf -v "$v" '%s' "$((10#${!v}))"   # base 10: a leading 0 must not read as octal
done
[ "$DM_NPROC" -ge 1 ] || { echo "FATAL: DM_NPROC=$DM_NPROC: at least one rank" >&2; exit 2; }
# the node count (header): DM_NNODES, else the job's inside a Slurm job, else 1
if [ -n "${DM_NNODES:-}" ]; then NNODES=$DM_NNODES NN_FROM=DM_NNODES
elif [ -n "${SLURM_JOB_ID:-}" ]; then NNODES=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}} NN_FROM=SLURM_JOB_NUM_NODES
else NNODES=1 NN_FROM=default; fi
case $NNODES in ''|*[!0-9]*) echo "FATAL: $NN_FROM='$NNODES' is not a whole number (the node count)" >&2; exit 2 ;; esac
NNODES=$((10#$NNODES))
[ "$NNODES" -ge 1 ] || { echo "FATAL: $NN_FROM=$NNODES: at least one node" >&2; exit 2; }
# A finished run exits at once: its name from the arm table alone (NAME_ONLY: no kernel or import check) and its marker
if [ "$SMOKE" = 0 ]; then
  RN=$(dm_run_name "$ARM" "$SEED")
  if [ -n "$RN" ] && [ -f "$SW/hf/$RN/TRAIN_COMPLETE" ]; then
    echo "[$(hostname -s) $(date +%T)] $RN already complete ($SW/hf/$RN/TRAIN_COMPLETE); nothing to do"; exit 0
  fi
fi
# several nodes: one srun step inside a Slurm job, ranks started by this script's own launchers
if [ "$NNODES" -gt 1 ]; then
  [ "${DM_LAUNCH:-torchrun}" = env ] || { echo "FATAL: multi-node needs DM_LAUNCH=env (NNODES=$NNODES from $NN_FROM, DM_LAUNCH=${DM_LAUNCH:-torchrun}; scripts/slurm/job.sh sets it; DM_NNODES=1 trains on this node only)" >&2; exit 2; }
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "FATAL: multi-node (NNODES=$NNODES) runs inside a Slurm job (no SLURM_JOB_ID)" >&2; exit 2; }
  command -v srun >/dev/null 2>&1 || { echo "FATAL: multi-node (NNODES=$NNODES) needs srun on PATH" >&2; exit 2; }
  command -v setsid >/dev/null 2>&1 || { echo "FATAL: multi-node (NNODES=$NNODES) needs setsid (util-linux) on PATH" >&2; exit 2; }
  if [[ ${SLURM_JOB_NUM_NODES:-} =~ ^[0-9]+$ ]] && [ "$NNODES" -gt "$SLURM_JOB_NUM_NODES" ]; then
    echo "FATAL: DM_NNODES=$NNODES but job $SLURM_JOB_ID has $SLURM_JOB_NUM_NODES node(s)" >&2; exit 2
  fi
fi
DATA=$DM_DATA_ROOT/nemotron_cc_v2d1_hq_dqa
[ -f "$DATA/nemotron_rs_train.bin" ] || { echo "FATAL: training data missing at $DATA (docs/SETUP.md section 4)" >&2; exit 2; }
[ -f "$DM_TOKENIZER_DIR/tokenizer.model" ] || { echo "FATAL: tokenizer missing at $DM_TOKENIZER_DIR" >&2; exit 2; }
DATA_ARGS=(--data.paths "$DATA/nemotron_rs_train.bin" --data.doc_lengths "$DATA/nemotron_rs_train.doc_lengths.npy")
if [ "${DM_WANDB:-1}" = 1 ]; then export WANDB_MODE=online; else export WANDB_MODE=disabled; fi   # DM_WANDB=0: no wandb at all
# The fp8 recipe's switches are the code defaults (parallel/wrap.py, trainer/base_trainer.py, modeling/mlp_fp8glue.py);
# exported explicitly as well so a launch log shows the configuration every arm runs.
export DM_FP8_FSDP_AG=1 DM_FP8_PRECOMPUTE=1                                     # fp8 weight all-gather + one scale precompute per step, always both
export DM_FP8GLUE=1 DM_FP8GLUE_SCALING=dynamic DM_FP8GLUE_SAVE_HT=0             # the fp8 cast glue, on every arm
unset DM_STALE_DOQ2S DM_SHARED_X8 DM_FLASH_IMPL DM_SAC_SAVE_FA3WHEEL DM_SDPA_BACKEND DM_DELTA_Q8   # arms.sh sets the arm's own
cd "$REPO" || exit 2; mkdir -p "$SW/ckpts" "$SW/hf" "$W/logs"
dm_arm_setup "$ARM" "$SEED" || exit 2
if [ "$ARM" = te_fp8 ]; then
  if [ -z "${DM_TE_TORCHRUN:-}" ] || [ -z "${DM_TE_PY:-}" ]; then
    echo "FATAL: te_fp8 needs the Transformer Engine environment: DM_TE_PY and DM_TE_TORCHRUN (docs/SETUP.md section 2)" >&2; exit 2
  fi
  TR=$DM_TE_TORCHRUN; PY=$DM_TE_PY
fi
"$PY" -c "import pretrain.train" || { echo "FATAL: tree not importable with $PY (PYTHONPATH=$PYTHONPATH)" >&2; exit 2; }
NS=${NUM_STEPS:-28610}
STOPARG=(); [ -n "${STOP_AT_STEP:-}" ] && STOPARG=(--stop_at_step "$STOP_AT_STEP")
if [ "$SMOKE" != 0 ]; then
  SN=$([ "$SMOKE" = 1 ] && echo 3 || echo "$SMOKE")
  if [ -n "${STOP_AT_STEP:-}" ]; then STOPARG=(--stop_at_step "$SN"); else NS=$SN; fi
  RN="${RN}_smoke${SN}"; export WANDB_MODE=disabled
fi
HFD=$SW/hf/$RN; DONE_MARK=$HFD/TRAIN_COMPLETE
already_complete () { [ "$SMOKE" = 0 ] && [ -f "$DONE_MARK" ] && echo "[$(hostname -s) $(date +%T)] $RN already complete ($DONE_MARK)$1; nothing to do"; }
already_complete "" && exit 0
# one trainer per run: the lock is held from here on by this script and every process it starts itself (fd 9: the env
# launch's ranks, the val CE; not torchrun's workers, see the header)
dm_lock 9 "$SW/ckpts/$RN.lock" "$RN is being trained"; rc=$?; [ "$rc" = 0 ] || exit "$rc"
already_complete " (by the job that held its lock)" && exit 0
LAUNCH=${DM_LAUNCH:-torchrun}
# the allow knobs are one-shot in a chain (header): dropped once this submission recorded the change in this run
if [ -n "${DM_CHAIN_DIR:-}" ] && [ "$SMOKE" = 0 ] && { [ "${DM_ALLOW_RUN_CHANGE:-0}" = 1 ] || [ "${DM_ALLOW_KERNEL_CHANGE:-0}" = 1 ]; }; then
  TSUB=$(awk '$3 == "submit" && $1 ~ /^[0-9]+$/ { t = $1 } END { if (t != "") print t }' "$DM_CHAIN_DIR/links" 2>/dev/null)
  if [[ ${TSUB:-} =~ ^[0-9]+$ ]]; then
    since_submit () { local m; m=$(stat -c %Y "$1" 2>/dev/null) && [ "$m" -ge "$TSUB" ]; }
    if [ "${DM_ALLOW_RUN_CHANGE:-0}" = 1 ] && since_submit "$SW/ckpts/$RN/run_identity.json"; then
      echo "[$(hostname -s) $(date +%T)] DM_ALLOW_RUN_CHANGE dropped: $RN recorded its identity since this chain's submission ($(date -d "@$TSUB" '+%F %T')), so a later change is refused again"
      unset DM_ALLOW_RUN_CHANGE
    fi
    CK_S=$(dm_ckpt_step "$SW/ckpts/$RN")
    if [ "${DM_ALLOW_KERNEL_CHANGE:-0}" = 1 ] && [ "$CK_S" -ge 0 ] && since_submit "$SW/ckpts/$RN/checkpoint-$CK_S/.complete"; then
      echo "[$(hostname -s) $(date +%T)] DM_ALLOW_KERNEL_CHANGE dropped: $RN saved checkpoint-$CK_S since this chain's submission ($(date -d "@$TSUB" '+%F %T')), on this build"
      unset DM_ALLOW_KERNEL_CHANGE
    fi
  fi
fi
# the code revision of this launch (in the LAUNCH line and the trainer's start event): "-dirty" = changes under src/ or
# scripts/ (uncommitted, or files git does not track yet: new modules count too), then "+<8 hex>" = the sha1 of `git
# diff HEAD -- src scripts` followed by the sha1sum of every untracked file there, so two different trees never look
# the same
if [ -z "${DM_GIT_REV:-}" ]; then
  DM_GIT_REV=$(GIT_OPTIONAL_LOCKS=0 git -C "$DM_REPO" describe --always --dirty 2>/dev/null) || DM_GIT_REV=""
  if [ -z "$DM_GIT_REV" ]; then DM_GIT_REV=none
  else
    UNTRACKED=$(cd "$DM_REPO" && GIT_OPTIONAL_LOCKS=0 git ls-files -o --exclude-standard -- src scripts 2>/dev/null)
    if [[ $DM_GIT_REV == *-dirty ]] || [ -n "$UNTRACKED" ]; then
      [[ $DM_GIT_REV == *-dirty ]] || DM_GIT_REV="$DM_GIT_REV-dirty"
      H=$(cd "$DM_REPO" && { GIT_OPTIONAL_LOCKS=0 git diff --no-ext-diff HEAD -- src scripts
                             GIT_OPTIONAL_LOCKS=0 git ls-files -z -o --exclude-standard -- src scripts | xargs -0 -r sha1sum --; } 2>/dev/null | sha1sum) \
        && DM_GIT_REV="$DM_GIT_REV+${H:0:8}"
    fi
  fi
fi
export DM_GIT_REV
SAVE_EVERY=${DM_SAVE_INTERVAL:-$SAVE_IV}
echo "[$(hostname -s) $(date '+%F %T')] LAUNCH $RN smoke=$SMOKE stop_at=${STOP_AT_STEP:-none} extra=${EXTRA_ARGS:-none} launch=$LAUNCH nodes=$NNODES world=$((NNODES * DM_NPROC)) save_interval=$SAVE_EVERY deadline=${DM_STOP_BEFORE:-none} py=$PY data=$DATA git=$DM_GIT_REV args=${ARGS[*]}"
# The trainer takes the last occurrence of a flag. --hf_export_steps comes before EXTRA_ARGS so that EXTRA_ARGS can set it
# (stage2_arm.sh's periodic exports); the flags after EXTRA_ARGS have their own knobs (NUM_STEPS, DM_LR, ...).
# shellcheck disable=SC2206   # EXTRA_ARGS is split into words on purpose
TRAIN_ARGS=(-m pretrain.train --config "${DM_TRAIN_CONFIG:-$CFG}"   # CFG from arms.sh (qwen35_m / _mla / _mamba2 / _kda / qwen3_m); DM_TRAIN_CONFIG: tests drive this launcher with tests/configs/tiny_hybrid.yaml
  --tokenizer_dir "$DM_TOKENIZER_DIR" "${DATA_ARGS[@]}" "${ARGS[@]}" --hf_export_steps "[]" ${EXTRA_ARGS:-} --num_steps "$NS" "${STOPARG[@]}"
  --save_interval "$SAVE_EVERY" --save_total_limit "${SAVE_TOTAL_LIMIT:-2}"
  --hf_save_dir "$HFD" --ckpt_root_dir "$SW/ckpts" --log_root_dir "$W/logs"
  --optim.lr "${DM_LR:-2.4e-3}" --seed "$SEED" --run_name "$RN" --wandb "$([ "$WANDB_MODE" = disabled ] && echo false || echo true)" --wandb_job_type "${WANDB_JT:-stage1}")
# the export's identity before this launch (inode, mtime, size of config.json): tells an export this launch wrote from
# one an earlier launch left
EXPORT0=$(stat -c '%i %Y %s' "$HFD/config.json" 2>/dev/null)
# the ranks of a node, the collector and the signal forwarding (start_ranks, failed_tails, forward_term, wait_ranks,
# wait_torchrun): shared with node_ranks.sh, the launcher of one node of a multi-node launch
# shellcheck source=scripts/campaign/ranks.sh
source "$HERE/ranks.sh"
ME_HOST=${SLURMD_NODENAME:-$DM_HOST}   # this host as Slurm names it (the node lists, the node launchers, badnodes)
CULPRITS=()                            # multi-node: the host(s) a failure is blamed on (badnodes)
STOP_SIGS=(TERM INT)                   # the signals that stop the launch (header); multi-node: job.sh's five
# ------------------------------------------------------------------------------------------- multi-node (NNODES > 1)
# One srun step, one node launcher (node_ranks.sh) per node; srun in a session of its own. STATUS = this launch's status
# directory; LPID = the node launcher on this host (from its pid file); STEP_ID = the srun step; FWD = 1 once the stop
# went to LPID; KILLED = 1 once this script killed the step (2: srun SIGKILLed too); SR_RC = srun's status.
STATUS="" LPID="" STEP_ID="" FWD=0 KILLED=0 SR_PID="" SRUN_PID="" SR_RC=""
# forward_nodes: the multi-node stop handler (STOP_SIGS; builtins only, as forward_term): the stop goes once, as
# SIGTERM, to the node launcher on this host, never to srun (it would SIGKILL the step). Its ranks raise the trainer's
# stop flag, which is MAX-all-reduced at the next step boundary: every rank of every node saves the same checkpoint and
# exits 75. A stop before that launcher's pid is known: the wait loop leaves the stop file and passes the stop on once
# it is.
forward_nodes () {
  NSIG=$((NSIG + 1)); [ "$NSIG" -gt 1 ] && return 0
  printf -v T_FWD '%(%s)T' -1
  if [ -n "$LPID" ]; then
    kill -TERM "$LPID" 2>/dev/null; FWD=1
    printf '[%s %(%T)T] SIGTERM -> the node launcher on this host (pid %s): the trainer stops the ranks of every node at the same step\n' "$ME_HOST" -1 "$LPID"
  else
    printf '[%s %(%T)T] SIGTERM before the node launcher on this host reported: the launchers are told by a stop file\n' "$ME_HOST" -1
  fi
}
# find_lpid: LPID = the node launcher on this host and STEP_ID = the step, from the launchers' pid files
# ("<pid> <host> <step id>"); status 1 while there is none for this host
find_lpid () {
  local n p h s
  for ((n = 0; n < NNODES; n++)); do
    p="" h="" s=""
    { read -r p h s < "$STATUS/node$n.pid"; } 2>/dev/null || continue
    [[ $s =~ ^[0-9]+$ ]] && STEP_ID=$s
    if [ "$h" = "$ME_HOST" ] && [[ $p =~ ^[0-9]+$ ]]; then LPID=$p; return 0; fi
  done
  return 1
}
# kill_step <why>: kill every process of the srun step. srun answers a SIGTERM by SIGKILLing every task of its step
# (seen on this cluster); srun is SR_PID, or SR_PID's child when setsid had to fork (it leads a process group). The step
# is also cancelled by id when a launcher reported it (scancel --signal=KILL <job>.<step>), which needs no srun at all.
kill_step () {
  local c="" k=()
  echo "[$ME_HOST $(date +%T)] srun still running $1: killing its step"
  SRUN_PID=$SR_PID
  { read -r c < "/proc/$SR_PID/comm"; } 2>/dev/null
  if [ "$c" = setsid ]; then
    { read -r -a k < "/proc/$SR_PID/task/$SR_PID/children"; } 2>/dev/null
    SRUN_PID=${k[0]:-}
  fi
  if [[ $STEP_ID =~ ^[0-9]+$ ]]; then
    timeout 30 scancel --signal=KILL "$SLURM_JOB_ID.$STEP_ID" \
      || echo "[$ME_HOST $(date +%T)] WARN: scancel --signal=KILL $SLURM_JOB_ID.$STEP_ID failed"
  fi
  [ -n "$SRUN_PID" ] && kill -TERM "$SRUN_PID" 2>/dev/null
  KILLED=1
}
# wait_srun: srun's exit status -> SR_RC, polling once a second like wait_ranks (a trapped signal interrupts the pause at
# once). Meanwhile it learns LPID, passes on a stop that came before LPID was known (the stop file until then), and
# kills the step when srun still runs DM_SRUN_GRACE s (default DM_TERM_GRACE + 60) after a stop, or DM_EXIT_GRACE s
# after a node's launcher reported that its ranks ended (0 or 75: the other nodes' ranks are past the run's last
# collective, so ranks still running then are stuck; each launcher's own DM_EXIT_GRACE cannot see across nodes).
# 30 s after that kill srun is SIGKILLed, 60 s after it srun is left behind (SR_RC 137).
wait_srun () {
  local now tk n s why t_kill="" t_exit="" en="" es=""
  local g_term=${DM_TERM_GRACE:-90} g_exit=${DM_EXIT_GRACE:-600} g_srun=${DM_SRUN_GRACE:-}
  case $g_term in ''|*[!0-9]*) g_term=90 ;; esac
  case $g_exit in ''|*[!0-9]*) g_exit=600 ;; esac
  case $g_srun in ''|*[!0-9]*) g_srun=$((g_term + 60)) ;; esac
  while :; do
    if ! kill -0 "$SR_PID" 2>/dev/null; then wait "$SR_PID"; SR_RC=$?; break; fi
    [ -z "$LPID" ] && find_lpid
    if [ "$NSIG" -gt 0 ] && [ "$FWD" = 0 ]; then
      if [ -n "$LPID" ]; then
        kill -TERM "$LPID" 2>/dev/null; FWD=1
        echo "[$ME_HOST $(date +%T)] SIGTERM -> the node launcher on this host (pid $LPID), which reported after the stop"
      elif [ ! -e "$STATUS/stop" ]; then
        { : > "$STATUS/stop"; } 2>/dev/null || echo "[$ME_HOST $(date +%T)] WARN: cannot create $STATUS/stop"
      fi
    fi
    printf -v now '%(%s)T' -1
    if [ -z "$t_exit" ]; then
      for ((n = 0; n < NNODES; n++)); do
        s=""; { read -r s _ < "$STATUS/node$n"; } 2>/dev/null
        case $s in 0|75) t_exit=$((now + g_exit)) en=$n es=$s; break ;; esac
      done
    fi
    if [ -z "$t_kill" ]; then
      why=""
      if [ "$NSIG" -gt 0 ] && [ "$now" -ge $(( ${T_FWD:-$now} + g_srun )) ]; then why="$g_srun s after the stop signal (DM_SRUN_GRACE)"
      elif [ -n "$t_exit" ] && [ "$now" -ge "$t_exit" ]; then why="$g_exit s after node $en reported that its ranks ended (status $es; DM_EXIT_GRACE)"; fi
      if [ -n "$why" ]; then kill_step "$why"; t_kill=$now; fi
    elif [ "$KILLED" = 1 ] && [ "$now" -ge $((t_kill + 30)) ]; then
      echo "[$ME_HOST $(date +%T)] srun still running 30 s after its step was killed: SIGKILL"
      [ -n "$SRUN_PID" ] && kill -KILL "$SRUN_PID" 2>/dev/null
      kill -KILL "$SR_PID" 2>/dev/null
      KILLED=2
    elif [ "$now" -ge $((t_kill + 60)) ]; then
      echo "[$ME_HOST $(date +%T)] srun still running 60 s after its step was killed: not waiting for it"
      SR_RC=137; break
    fi
    sleep 1 & tk=$!
    wait "$tk"; [ $? -gt 128 ] && kill "$tk" 2>/dev/null
  done
}
# aggregate_nodes: RC from the node launchers' reports (the header's rule), one line per node, and CULPRITS: the node
# that failed first and every node whose first failure came within TIE_S = 2 s of it (a tie, header), else the nodes
# that never reported, else every failing node. A node that failed only after a stop reached it (srun's
# kill-on-bad-exit, the time limit) has no failure time: it is not blamed while another node has one.
TIE_S=2
aggregate_nodes () {
  local n rc h t r p hp sp tmin="" any75=0 line
  local -a timed=() untimed=() unrep=() rcs=() hosts=() times=()
  for ((n = 0; n < NNODES; n++)); do
    rc="" h="" t="" r=""
    { read -r rc h t r < "$STATUS/node$n"; } 2>/dev/null
    if [[ $rc =~ ^[0-9]+$ ]]; then
      rcs[n]=$rc hosts[n]=${h:-${HOSTS[n]:-?}} times[n]=$t
      line="node $n (${hosts[n]}): status $rc"
      [[ $t =~ ^[0-9]+$ ]] && line+=", first failure at $(date -d "@$t" +%T)"
      [ -n "$r" ] && line+=", failed global rank(s) $r"
      case $rc in
        0) ;;
        75) any75=1 ;;
        *) if [[ $t =~ ^[0-9]+$ ]]; then
             timed+=("$n")
             if [ -z "$tmin" ] || [ "$t" -lt "$tmin" ]; then tmin=$t; fi
           else untimed+=("$n"); fi ;;
      esac
    else
      p="" hp="" sp=""; { read -r p hp sp < "$STATUS/node$n.pid"; } 2>/dev/null
      hosts[n]=${hp:-${HOSTS[n]:-?}}
      unrep+=("$n")
      line="node $n (${hosts[n]}): no report (its launcher was killed, or the node was lost)"
    fi
    echo "[$ME_HOST $(date +%T)] $line"
  done
  CULPRITS=()
  if [ ${#timed[@]} -gt 0 ]; then
    for n in "${timed[@]}"; do [ "${times[n]}" -le $((tmin + TIE_S)) ] && CULPRITS+=("${hosts[n]}"); done
  elif [ ${#unrep[@]} -gt 0 ]; then
    for n in "${unrep[@]}"; do CULPRITS+=("${hosts[n]}"); done
  else
    for n in ${untimed[@]+"${untimed[@]}"}; do CULPRITS+=("${hosts[n]}"); done
  fi
  if [ "$any75" = 1 ]; then RC=75
  elif [ ${#timed[@]} -gt 0 ]; then
    for n in "${timed[@]}"; do [ "${times[n]}" = "$tmin" ] && { RC=${rcs[n]}; break; }; done
  elif [ ${#untimed[@]} -gt 0 ]; then RC=${rcs[${untimed[0]}]}
  elif [ ${#unrep[@]} -eq 0 ]; then RC=0
  elif [ "$KILLED" != 0 ]; then RC=143
  elif [ "${SR_RC:-0}" -ne 0 ]; then RC=$SR_RC
  else RC=1; fi
  echo "[$ME_HOST $(date +%T)] $NNODES nodes done: status $RC (srun exit ${SR_RC:-none})"
}
# prune_status: keep the newest 10 status directories of the run (one per multi-node launch)
prune_status () {
  local d i=0
  while IFS= read -r d; do
    [[ $d =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || continue
    i=$((i + 1)); [ "$i" -le 10 ] && continue
    rm -rf "${NSD:?}/$d" 2>/dev/null
  done < <(ls -1t "$NSD" 2>/dev/null)
}
if [ "$LAUNCH" = env ] && [ "$NNODES" -eq 1 ]; then
  export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} MASTER_PORT=${DM_MASTER_PORT:-29517} WORLD_SIZE=$DM_NPROC LOCAL_WORLD_SIZE=$DM_NPROC OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
  start_ranks 0 "$PY" "${TRAIN_ARGS[@]}"
  trap forward_term TERM INT
  wait_ranks
  echo "--- rank 0 tail ---"; tail -n 5 "$W/logs/${RN}.rank0.log"
  if [ "$RC" -ne 0 ] && [ "$RC" -ne 75 ]; then failed_tails 0; fi
elif [ "$LAUNCH" = env ]; then
  HOSTS=()
  [ -n "${SLURM_JOB_NODELIST:-}" ] && mapfile -t HOSTS < <(timeout 30 scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null)
  if [ ${#HOSTS[@]} -eq 0 ]; then
    echo "[$ME_HOST $(date +%T)] WARN: cannot list the job's nodes (scontrol show hostnames '${SLURM_JOB_NODELIST:-}'): MASTER_ADDR=$ME_HOST"
    HOSTS=("$ME_HOST")
  elif [ "${HOSTS[0]}" != "$ME_HOST" ]; then
    echo "[$ME_HOST $(date +%T)] WARN: the job's first node is ${HOSTS[0]}, not this host ($ME_HOST): global rank 0 runs there (SLURM_NODEID 0), MASTER_ADDR=${HOSTS[0]}"
  fi
  export MASTER_ADDR=${HOSTS[0]} MASTER_PORT=${DM_MASTER_PORT:-29517} WORLD_SIZE=$((NNODES * DM_NPROC)) LOCAL_WORLD_SIZE=$DM_NPROC OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
  NSD=$W/logs/nodes/$RN
  printf -v T_LAUNCH '%(%s)T' -1
  STATUS=$NSD/$SLURM_JOB_ID.$T_LAUNCH.$$
  mkdir -p "$STATUS" || { echo "[$ME_HOST $(date +%T)] FATAL: cannot create the launch's status directory $STATUS" >&2; exit 1; }
  export DM_NODE_STATUS=$STATUS DM_BATCH_HOST=$ME_HOST
  # the srun options, in one place: one task per node, each with the whole node (its GPUs only when the job has GPUs:
  # the same path runs on CPU nodes); kill-on-bad-exit: a node whose ranks failed ends the others' tasks at once
  SRUN=(srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 --mpi=none --cpu-bind=none)
  [[ ${SLURM_CPUS_ON_NODE:-} =~ ^[0-9]+$ ]] && SRUN+=(--cpus-per-task="$SLURM_CPUS_ON_NODE")
  [[ ${SLURM_GPUS_ON_NODE:-} =~ ^[0-9]+$ ]] && [ "$SLURM_GPUS_ON_NODE" -gt 0 ] && SRUN+=(--gpus-per-node="$DM_NPROC")
  SRUN+=(--mem=0 --export=ALL)
  [ ${#HOSTS[@]} -gt "$NNODES" ] && SRUN+=(--nodelist="$(IFS=,; echo "${HOSTS[*]:0:NNODES}")")   # DM_NNODES below the job's
  read -r -a SRUN_EXTRA <<< "${DM_SRUN_EXTRA:-}"
  SRUN+=(${SRUN_EXTRA[@]+"${SRUN_EXTRA[@]}"})
  STOP_SIGS=(TERM INT HUP USR1 USR2)   # job.sh's stop signals (header); the tail below keeps them
  trap forward_nodes "${STOP_SIGS[@]}"
  echo "[$ME_HOST $(date '+%F %T')] $NNODES nodes (${HOSTS[*]:0:NNODES}), world $WORLD_SIZE, master $MASTER_ADDR:$MASTER_PORT, status $STATUS"
  if [ "$NSIG" -gt 0 ]; then
    echo "[$ME_HOST $(date +%T)] a stop signal came before srun was started: not starting it"
    RC=143
  else
    echo "[$ME_HOST $(date +%T)] ${SRUN[*]} bash $HERE/node_ranks.sh $RN $PY <trainer args>"
    # its own session: job.sh's `kill -TERM 0` must never reach srun (header)
    setsid -w "${SRUN[@]}" bash "$HERE/node_ranks.sh" "$RN" "$PY" "${TRAIN_ARGS[@]}" & SR_PID=$!
    wait_srun
    aggregate_nodes
  fi
  echo "--- rank 0 tail ---"; tail -n 5 "$W/logs/${RN}.rank0.log" 2>/dev/null
  prune_status
else
  # torchrun reports a rank's exit 75 as its own failure (1): the stop is read from the trainer log, which every attempt
  # appends to, so only the part written by this launch counts, and only when this launch wrote no export (config.json
  # as it was before the launch: a whole export an earlier launch left must not turn a clean stop into a failure). grep
  # reads the log through a process substitution: in a `tail | grep -q` pipeline under pipefail, tail dies of SIGPIPE
  # when more than a pipe buffer follows the match
  TLOG=$W/logs/${RN}.log; LOG0=0; [ -f "$TLOG" ] && LOG0=$(wc -c < "$TLOG")
  "$TR" --standalone --nproc_per_node="$DM_NPROC" "${TRAIN_ARGS[@]}" & TR_PID=$!
  trap forward_term TERM INT
  wait_torchrun
  if [ "$RC" -gt 128 ] && [ "$NSIG" -eq 0 ]; then
    # torchrun itself was killed (the OOM killer, kill -9): its workers run in sessions of their own and survive it,
    # without the run lock (they never held it)
    echo "[$(hostname -s) $(date +%T)] WARN: torchrun was killed by signal $((RC - 128)): its workers may still be running (ps -u $(id -un) -f | grep $RN), without the run lock: make sure they are gone before $RN is launched again"
  fi
  if [ $RC -ne 0 ] && grep -aq "Stopping early" <(tail -c +$((LOG0 + 1)) "$TLOG" 2>/dev/null) \
     && [ "$(stat -c '%i %Y %s' "$HFD/config.json" 2>/dev/null)" = "$EXPORT0" ]; then RC=75; fi
  # stopped by the signal before a rank saved: report a death by signal (128 + 15), not torchrun's 1, so a caller can tell
  # a stop from a configuration failure
  if [ $RC -ne 0 ] && [ $RC -ne 75 ] && [ "$NSIG" -gt 0 ]; then RC=143; fi
  # a refused resume: every rank exits 2, which torchrun reports as its own 1 (a rank may even show -15: torchrun's
  # SIGTERM races the interpreter's exit), so the trainer's line in this launch's part of the log decides
  if [ $RC -ne 0 ] && [ $RC -ne 75 ] && grep -aq "FATAL: RESUME REFUSED" <(tail -c +$((LOG0 + 1)) "$TLOG" 2>/dev/null); then RC=2; fi
fi
# The tail. A stop signal from here on (the ranks are gone) is recorded and passed to the val CE; it never kills this
# script half-way through the patch or the marker. Builtins only in the handler. TSIG starts at the number of stop
# signals the ranks' run already got: after a stop the informational val CE is not started at all. Every stop signal of
# the launch (STOP_SIGS) comes here now: none may reach forward_nodes and a launcher pid that is gone.
TSIG=$NSIG; VPID=""
tail_signal () { TSIG=$((TSIG + 1)); [ "$TSIG" -gt 1 ] && return 0; [ -n "$VPID" ] && kill -TERM "$VPID" 2>/dev/null; return 0; }
trap tail_signal "${STOP_SIGS[@]}"
echo "[$(hostname -s) $(date +%T)] $RN done (exit $RC)"
if [ "$RC" -eq 75 ]; then
  echo "[$(hostname -s) $(date +%T)] $RN stopped early with a complete checkpoint: relaunch to resume"
  exit 75
fi
signal_rc () { if [ "$RC" -ne 0 ]; then echo "$RC"; else echo 143; fi; }   # the exit status after a stop in the tail
WHOLE=0; CHK=$("$PY" -m pretrain.hf.check_export "$HFD" 2>&1) && WHOLE=1
EXPORT1=$(stat -c '%i %Y %s' "$HFD/config.json" 2>/dev/null)
if [ "$WHOLE" = 1 ] && { [ "$RC" -eq 0 ] || [ "$EXPORT1" != "$EXPORT0" ]; }; then
  [ "$RC" -ne 0 ] && echo "[$(hostname -s) $(date +%T)] the trainer exited $RC after writing a whole export of $RN: the run is complete (the failure came after the export)"
  # The patch: every arm is scored through the common clean kernel. Atomic (a temporary file, then os.replace) and
  # idempotent: config.train.json gets the trainer's config.json when it does not exist yet or config.json is still
  # unpatched, never a patched one.
  if ! "$PY" - "$HFD" <<'PYEOF'
import json, os, sys
d = sys.argv[1]; p = os.path.join(d, "config.json"); t = os.path.join(d, "config.train.json")
raw = open(p).read(); c = json.loads(raw)
n = dict(c)
n["attn_impl"] = "sdpa"                                   # every arm is scored through the common clean kernel
if n.get("mixer") == "dm": n["mixer"] = "gqa"
if n.get("layer_mixers"): n["layer_mixers"] = [{"dm": "gqa", "mla_dm": "mla"}.get(m, m) for m in n["layer_mixers"]]
if n.get("mixer") == "mla_dm": n["mixer"] = "mla"
for k in ("dm_mode", "dm_fused_producers"): n.pop(k, None)   # the fp8-only keys; config.train.json keeps them
def write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
unpatched = n != c
if unpatched or not os.path.exists(t):
    write(t, raw); print("config.train.json <- the trained config")
if unpatched:
    write(p, json.dumps(n, indent=2, sort_keys=True)); print("export patched -> clean-kernel eval")
else:
    print("export already patched for the clean-kernel eval")
PYEOF
  then
    echo "[$(hostname -s) $(date +%T)] FATAL: could not patch $HFD/config.json for the clean-kernel evaluation" >&2
    [ "$TSIG" -gt 0 ] && exit "$(signal_rc)"
    exit 1
  fi
  if [ "$SMOKE" = 0 ]; then
    TRAINED=$NS; [ -n "${STOP_AT_STEP:-}" ] && [ "$STOP_AT_STEP" -lt "$NS" ] && TRAINED=$STOP_AT_STEP
    if ! { printf '%s steps=%s num_steps=%s\n' "$(date '+%F %T')" "$TRAINED" "$NS" > "$DONE_MARK.tmp" && mv -f "$DONE_MARK.tmp" "$DONE_MARK"; }; then
      rm -f "$DONE_MARK.tmp" 2>/dev/null
      echo "[$(hostname -s) $(date +%T)] FATAL: could not write $DONE_MARK" >&2
      [ "$TSIG" -gt 0 ] && exit "$(signal_rc)"
      exit 1
    fi
    echo "[$(hostname -s) $(date +%T)] TRAIN COMPLETE marker written: $DONE_MARK (steps=$TRAINED num_steps=$NS)"
    dm_progress train_complete "$RN"
  fi
  # the informational val CE: 128 windows of the held-out stream on one GPU
  VAL=$DATA/nemotron_rs_val.bin
  unset DM_FP8GLUE DM_FP8GLUE_SCALING DM_FP8GLUE_SAVE_HT DM_SHARED_X8 DM_STALE_DOQ2S DM_DELTA_Q8 DM_STOP_BEFORE
  VALCE_GPU=${CUDA_VISIBLE_DEVICES:-0}; VALCE_GPU=${VALCE_GPU%%,*}   # the first GPU this launcher was given
  # --wandb: eval/* go to this run's W&B page (memmap_ppl logs only when asked), exactly when the trainer ran with wandb
  VALCE_WANDB=(); [ "$WANDB_MODE" = disabled ] || VALCE_WANDB=(--wandb)
  if [ "$TSIG" -eq 0 ]; then
    RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=${DM_VALCE_PORT:-29797} CUDA_VISIBLE_DEVICES=$VALCE_GPU \
    "$PY" -m evaluation.ppl.memmap_ppl --model_name_or_path "$HFD" --data "$VAL" --dtype uint16 --seq_len 8192 --max_windows 128 --batch_size 8 "${VALCE_WANDB[@]}" > "$HFD/valce.txt" 2>&1 & VPID=$!
    while :; do n=$TSIG; wait "$VPID"; VRC=$?; [ "$TSIG" != "$n" ] && [ "$VRC" -gt 128 ] && continue; break; done
    VPID=""
    [ "$VRC" -eq 0 ] || echo "(val ce failed)"
    grep -a "memmap-ppl RESULT" "$HFD/valce.txt" | tail -1
  fi
  [ "$TSIG" -gt 0 ] && echo "[$(hostname -s) $(date +%T)] a stop signal arrived: the end-of-run val CE was skipped or cut short (informational only)"
  if [ "$SMOKE" != 0 ]; then echo "SMOKE $RN: export+valce tail OK; removing smoke dirs"; rm -rf "$HFD" "$SW/ckpts/$RN"; exit "$RC"; fi
  exit 0
fi
if [ "$TSIG" -gt 0 ]; then
  echo "[$(hostname -s) $(date +%T)] $RN: stopped by a signal before a whole export of this launch was found (exit $RC)"; exit "$(signal_rc)"
fi
if [ "$RC" -eq 0 ]; then
  echo "[$(hostname -s) $(date +%T)] FATAL: the trainer finished but the export is incomplete ($HFD: ${CHK:-no reason given}): relaunch to resume the finished run and export it again" >&2
  exit 1
fi
[ "$WHOLE" = 1 ] && echo "[$(hostname -s) $(date +%T)] $RN: the whole export in $HFD is from an earlier launch and the trainer failed (exit $RC): the run is not marked complete"
# badnodes (header): in a chain, "<epoch> <job id> <host>" for each host this failure is blamed on, which job.sh keeps
# the chain's next job off after a failure without progress (best effort). One node: this host. Never for a
# configuration error (2: the chain stops); a stop never gets here.
if [ -n "${DM_CHAIN_DIR:-}" ] && [ "$RC" -ne 2 ]; then
  [ "$NNODES" -eq 1 ] && CULPRITS=("$ME_HOST")
  printf -v T_BAD '%(%s)T' -1
  for h in ${CULPRITS[@]+"${CULPRITS[@]}"}; do
    [[ $h =~ ^[A-Za-z0-9._-]+$ ]] || continue   # a node name (job.sh passes it to scontrol)
    { printf '%s %s %s\n' "$T_BAD" "${SLURM_JOB_ID:-0}" "$h" >> "$DM_CHAIN_DIR/badnodes"; } 2>/dev/null
  done
  [ ${#CULPRITS[@]} -gt 0 ] && echo "[$(hostname -s) $(date +%T)] $RN failed on ${CULPRITS[*]}: noted in $DM_CHAIN_DIR/badnodes (after a failure without progress the chain's next job avoids it)"
fi
exit "$RC"
