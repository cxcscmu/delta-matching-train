#!/bin/bash
# scripts/slurm/job.sh: the body of every job (a "link") of a chain submitted by scripts/slurm/submit.sh
# (scripts/slurm/README.md); not run by hand.
#   job.sh <command> [args...]
# A chain is every job of the user with one Slurm job name (submit.sh names it; --dependency=singleton in every link's
# sbatch options makes Slurm run them one at a time). A link:
#  1. sources the site file (DM_SITE_ENV: conda env, workspace, caches) and refuses a command it cannot run from the
#     repository root (not on PATH, or not an executable file: exit 2, the chain stops); exports DM_LAUNCH=env (unless
#     set: torchrun would get Slurm's SIGTERM too and kill its ranks 30 s later), per-job rendezvous ports
#     DM_MASTER_PORT / DM_VALCE_PORT (unless set), DM_NPROC capped to the job's GPUs per node, DM_JOB_END = the job's end
#     (SLURM_JOB_END_TIME, else squeue) and DM_STOP_BEFORE = DM_JOB_END - DM_STOP_MARGIN (default 600 s: the trainer
#     saves a complete checkpoint at the next step boundary and exits 75). DM_DERIVED names the variables this script
#     set itself (ports, DM_NPROC, compile-cache dirs): a successor derives its own, and submit.sh run from inside a link
#     drops them. A time limit that leaves less than 10 minutes after the margin sets no deadline and makes the job a
#     single job (below); an unknown end sets no deadline.
#  2. waits while another job of the chain is RUNNING or COMPLETING (at most DM_PRED_WAIT seconds, default 600 =
#     KillWait 120 + UnkillableStepTimeout 480; 0: no wait): Slurm releases a singleton successor as soon as its
#     predecessor's state is final (TIMEOUT, CANCELLED, ...), while that predecessor's processes may still be saving and
#     holding the run's lock, and its job.sh still writing its records.
#  3. keeps the chain's state in DM_CHAIN_DIR (<workspace>/logs/slurm/chains/<name>/, from submit.sh): `links` (this
#     script's records: submit / judge / start / succ / run / attempt / end; every one that changes the failure count
#     carries fails=<n>), `progress` (DM_PROGRESS_FILE: one line per new complete checkpoint, finished training or
#     evaluation, written by the trainer and the launchers), `badnodes` ("<epoch> <job id> <host>": the hosts
#     scripts/campaign/train_arm.sh blamed a failed training on), `jit.tar` (the compile caches) and `submit.sh`
#     (submits the chain again). A state directory that cannot be written costs only the accounting, never the job.
#  4. judges a predecessor that left no `end` record (killed after KillWait, node failure, requeue): an attempt it
#     started (`run`) but never recorded (`attempt`), or a death before its command started, is one more failure unless
#     `progress` has a line of that job since; attempts it recorded are not counted twice. More than DM_MAX_FAILS
#     (default 2) failures in a row without progress: give up (exit 4). The first job of a submission (its `submit`
#     record) starts at 0, whatever older jobs of the chain recorded after the submission.
#  5. queues its successor at once (the same command and sbatch options, DM_RESUBMIT_COUNT+1, without this job's
#     DM_STOP_BEFORE / DM_JOB_END / DM_DERIVED variables; singleton holds it until this job has ended), or adopts a
#     pending job of the chain (a requeue's successor, a resubmission, or a timed-out sbatch that queued its job anyway).
#     So a link that is killed or hangs still hands over. Up to DM_MAX_RESUBMITS successors in a chain (default 400).
#  6. restores the chain's compile caches (DM_JIT_CACHE=1, default; Triton / TileLang / Inductor in
#     $DM_JIT_ROOT/<name>, default DM_JIT_ROOT=/tmp/dm-jit-<uid>: one fixed node-local path per chain, as Triton
#     records absolute paths; exported as DM_JIT_CACHE_DIR, where scripts/campaign/node_ranks.sh restores the archive on
#     the other nodes of a multi-node job), runs the command from the repository root and passes Slurm's signals on
#     (below).
#  7. after the command: a failure (an exit other than 0 / 75 / 2 / 3 / 4) with no signal and at least
#     DM_RETRY_MIN_LEFT seconds (default 1800) before the deadline is retried in place after DM_RETRY_PAUSE seconds
#     (default 30), up to DM_INPLACE_RETRIES times per job (default 1). An attempt without a progress line is a
#     failure without progress (unless Slurm requeued the job); one with progress resets the count.
#  8. at the end: exit 0 (everything requested is done): the chain is complete, its other jobs are cancelled. 2
#     (configuration error), 3 (the run belongs to another live job), 4 (gave up) or a stop by hand: the chain stops,
#     its other jobs are cancelled. Requeued by Slurm: the job runs again, nothing is changed. Otherwise (75 = work
#     remains, a failure, the time limit): more than DM_MAX_FAILS attempts in a row without progress gives up (exit 4:
#     resume with $DM_CHAIN_DIR/submit.sh once the cause is fixed); else the successor runs next: a pending one continues
#     the chain (after a failure without progress it is kept off the hosts train_arm.sh blamed the failure on in
#     `badnodes` during this job, else off every node of this job: scontrol update ExcNodeList), one that
#     Slurm already started (while this job completes) too; one that someone cancelled stops the chain; a missing one is
#     queued now (retried until 90 s before the end, at most 30 minutes). A link that ended by itself (no signal) with
#     >= 120 s left saves changed compile caches to jit.tar (at most DM_JIT_MAX_MB, default 8192).
# Signals: Slurm's SIGTERM (the time limit, scancel, a requeue) reaches only the batch script: TERM, INT, HUP, USR1 and
# USR2 are passed on once, as SIGTERM, to every process of the job (the handler is set before the site file); the script
# then waits for the command (the trainer's SIGTERM handler saves a checkpoint and exits 75). A multi-node training's
# srun runs in a session of its own (a SIGTERM to srun SIGKILLs its step): train_arm.sh passes the stop on to the ranks
# (scripts/campaign/node_ranks.sh); Slurm itself signals every srun task at the time limit. What the signal meant
# (squeue, asked while the command shuts down):
#  - a requeue (preemption, scontrol requeue): its restart count (squeue RestartCnt) is above this instance's
#    SLURM_RESTART_COUNT (the job stays COMPLETING meanwhile), or it is PENDING again: Slurm starts it again with the
#    same id;
#  - the time limit: it came at most 60 s before the job's start + time limit as squeue shows them now (`scontrol
#    update TimeLimit=` included; never squeue's end time, which a scancel sets to the moment of the cancel), or the
#    end is unknown;
#  - otherwise a stop by hand (scancel, a signal sent by hand).
# Cancelling the chain's other jobs: by id, the pending ones and one that Slurm already started while this job
# completes; never this job, a COMPLETING one, or the first job of a newer submission (a `submit` record: the chain was
# submitted again while it ran, and that submission continues it). squeue failing: scancel -u -n -t PENDING.
# DM_RESUBMIT=0: a single job: no successor, no in-place retry, nothing cancelled, no records, no compile caches, and
# the command is outside a chain (no DM_CHAIN / DM_CHAIN_DIR / DM_PROGRESS_FILE). sbatch retries: DM_RESUBMIT_BACKOFF
# (seconds before each retry) replaces the schedules (start of the link: 10 30; end: 10 30 60 120 120 ...). The job
# name must be a chain name ([A-Za-z0-9._-]) and DM_SBATCH_OPTS must hold --dependency=singleton (this submit.sh), else
# the job runs as a single job (a chain of an older submit.sh: resubmit it).
# Exit status: the command's; 4 = gave up (no progress); 76 = a successor was needed and none could be queued (the chain
# is broken: resume with $DM_CHAIN_DIR/submit.sh); 2 = a malformed knob, a failing site file, or a command that cannot
# be run. A link that hands over exits 75 (or the failure's status), which sacct shows as FAILED.
set -uo pipefail
# The repository: DM_SLURM_REPO from submit.sh. Slurm runs a COPY of this script from its spool directory, so the
# script's own location only names the repository when it is run by hand.
REPO="${DM_SLURM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[ -f "$REPO/scripts/slurm/job.sh" ] || { echo "FATAL: $REPO is not the delta-matching-train repository (DM_SLURM_REPO)" >&2; exit 2; }
[ $# -gt 0 ] || { echo "usage: job.sh <command> [args...] (submitted by scripts/slurm/submit.sh)" >&2; exit 2; }
for v in DM_STOP_MARGIN DM_MAX_RESUBMITS DM_RESUBMIT_COUNT DM_MAX_FAILS DM_INPLACE_RETRIES DM_RETRY_MIN_LEFT DM_RETRY_PAUSE \
         DM_JIT_MAX_MB DM_PRED_WAIT DM_STOP_BEFORE DM_MASTER_PORT DM_VALCE_PORT; do
  case "${!v-}" in *[!0-9]*) echo "FATAL: $v='${!v}' is not a whole number" >&2; exit 2 ;; esac
done
for v in DM_RESUBMIT DM_JIT_CACHE; do
  case "${!v-}" in ""|0|1) ;; *) echo "FATAL: $v='${!v}' is neither 0 nor 1" >&2; exit 2 ;; esac
done
case "${DM_RESUBMIT_BACKOFF-}" in *[!0-9\ ]*)
  echo "FATAL: DM_RESUBMIT_BACKOFF='$DM_RESUBMIT_BACKOFF' is not a list of whole numbers (seconds)" >&2; exit 2 ;; esac
# 10#: a leading zero is not octal
MARGIN=$((10#${DM_STOP_MARGIN:-600})) MAX_RESUB=$((10#${DM_MAX_RESUBMITS:-400})) K=$((10#${DM_RESUBMIT_COUNT:-0}))
MAX_FAILS=$((10#${DM_MAX_FAILS:-2})) INPLACE=$((10#${DM_INPLACE_RETRIES:-1})) MIN_LEFT=$((10#${DM_RETRY_MIN_LEFT:-1800}))
PAUSE=$((10#${DM_RETRY_PAUSE:-30})) JIT_MAX=$((10#${DM_JIT_MAX_MB:-8192})) PRED_WAIT=$((10#${DM_PRED_WAIT:-600}))
printf -v T_LINK '%(%s)T' -1   # this run of the job (a requeue runs it again with the same id): badnodes since then
say () { echo "[slurm $(hostname -s) $(date '+%F %T')] $*"; }

# Signals. Slurm's SIGTERM (the time limit, scancel, a requeue) reaches only this script, the batch shell, not the
# command. Pass the first signal of any kind on once, as SIGTERM, to every process of the job, i.e. this script's process
# group (the batch script leads its own, apart from slurmstepd), as train_arm.sh expects from a scheduler, and note which
# one and when (builtins only: printf -v forks nothing). The handler is set before anything else of the link, the site
# file included, so that a stop during the setup is a stop too; whatever runs long here runs in the background (run_bg),
# as bash runs a trap only after a foreground command has ended. Later signals are ignored.
SIG="" T_SIG=""
on_sig () { [ -n "$SIG" ] && return 0; SIG=$1; printf -v T_SIG '%(%s)T' -1; kill -TERM 0 2>/dev/null; }
sig_rc () { case "$1" in HUP) echo 129 ;; INT) echo 130 ;; USR1) echo 138 ;; USR2) echo 140 ;; *) echo 143 ;; esac; }
# run_bg <command...>: run it in the background and wait for it; a signal stops it, also one that came before the child
# existed (a timeout'ed command runs in its own process group, out of the reach of kill -TERM 0)
run_bg () {
  "$@" &
  local p=$! rc
  [ -n "$SIG" ] && kill -TERM "$p" 2>/dev/null
  wait "$p"; rc=$?
  while kill -0 "$p" 2>/dev/null; do
    [ -n "$SIG" ] && kill -TERM "$p" 2>/dev/null
    wait "$p"; rc=$?
  done
  return "$rc"
}
pause () { [ "${1:-0}" -gt 0 ] && run_bg sleep "$1"; return 0; }
# The compile-cache directory (step 6) and the site file's node-local scratch go when the job ends; nothing else.
JIT=""
cleanup () {
  [ -n "$JIT" ] && rm -rf "$JIT" 2>/dev/null
  if [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] && [ "${TMPDIR:-}" = "/tmp/dm-$SLURM_JOB_ID" ]; then rm -rf "$TMPDIR" 2>/dev/null; fi
  return 0
}
set_traps () { local s; for s in TERM INT HUP USR1 USR2; do trap "on_sig $s" "$s"; done; trap cleanup EXIT; }
set_traps
if [ -n "${DM_SITE_ENV:-}" ]; then
  # without set -u: site files are written for interactive shells (e.g. $LD_LIBRARY_PATH unset in a job)
  set +u
  # shellcheck disable=SC1090
  source "$DM_SITE_ENV"; rc=$?
  set -u
  set_traps   # again: a site file may have set its own
  if [ "$rc" -ne 0 ]; then
    # our own kill -TERM 0 may have stopped one of its commands: then the stop is what matters
    [ -n "$SIG" ] || { echo "FATAL: sourcing DM_SITE_ENV=$DM_SITE_ENV failed (status $rc)" >&2; exit 2; }
    say "WARN: sourcing DM_SITE_ENV=$DM_SITE_ENV failed (status $rc) after a SIG$SIG"
  fi
fi
# the command as it will run (from the repository root): one that cannot be run is a configuration error, not a failure
# to retry (bash would exit 126 / 127 in every link). Asked 3 times, 2 s apart: a file system's hiccup is not that.
for t in 1 2 3; do
  [ -z "$SIG" ] || break
  ( cd "$REPO" 2>/dev/null && command -v -- "$1" ) >/dev/null 2>&1 && break
  if [ "$t" = 3 ]; then
    echo "FATAL: the command '$1' cannot be run from $REPO (not on PATH, or not an executable file): the chain stops" >&2
    exit 2
  fi
  pause 2
done

# The stop reaches every process of this script's process group (Slurm signals only this script, which passes it on:
# above), torchrun included, and torchrun kills its ranks 30 s after a SIGTERM (before a large checkpoint can be
# saved): let train_arm.sh start the ranks itself, so they get the SIGTERM directly
export DM_LAUNCH=${DM_LAUNCH:-env}
# a job that does not take the whole node can share it with another: per-job rendezvous ports for the env launch and the
# end-of-run val CE (train_arm.sh). They lie outside this host's ephemeral port range (net.ipv4.ip_local_port_range;
# DM_EPHEMERAL_PORTS="<lo> <hi>" overrides it; unknown: 9000 65000, which keeps the ports below both Linux's default
# 32768-60999 and this cluster's 9000-65000), where any outgoing connection may hold a port (on a node shared with other
# jobs a rendezvous failed with EADDRINUSE): a band of 1300 ports just below the range (7700-8999 for 9000-65000), else
# just above it (from 1024 at the least), else 20000-31999. The port is the band's start + 2 x (job id mod its slots),
# the val CE's the next one, moved on by 2 while this host has a socket on either (/proc/net/tcp*: any address, any
# state) or a listener answers on 127.0.0.1. Derived again before an in-place retry (a port taken meanwhile, e.g. what
# made the attempt fail, is skipped). What is derived here is this job's own (DERIVED): the successor derives its own.
port_busy () {
  local h; printf -v h ':%04X' "$1"
  awk -v p="$h" 'FNR > 1 && substr($2, length($2) - 4) == p { f = 1; exit } END { exit !f }' /proc/net/tcp /proc/net/tcp6 \
    2>/dev/null && return 0
  timeout 2 bash -c ": </dev/tcp/127.0.0.1/$1" 2>/dev/null
}
# eph_range: EPH_LO EPH_HI = the ephemeral range (DM_EPHEMERAL_PORTS when it is two whole numbers 1 <= lo <= hi <= 65535)
valid_range () { [[ ${1:-} =~ ^[0-9]{1,5}$ && ${2:-} =~ ^[0-9]{1,5}$ && -z ${3:-} ]] && [ "$((10#$1))" -ge 1 ] \
                   && [ "$((10#$1))" -le "$((10#$2))" ] && [ "$((10#$2))" -le 65535 ]; }
eph_range () {
  local a="" b="" c=""
  if [ -n "${DM_EPHEMERAL_PORTS:-}" ]; then
    read -r a b c <<< "$DM_EPHEMERAL_PORTS" || true
    if valid_range "$a" "$b" "$c"; then EPH_LO=$((10#$a)) EPH_HI=$((10#$b)); return 0; fi
    say "WARN: DM_EPHEMERAL_PORTS='$DM_EPHEMERAL_PORTS' is not '<lo> <hi>' with 1 <= lo <= hi <= 65535: ignored"
  fi
  local f=${DM_PORT_RANGE_FILE:-/proc/sys/net/ipv4/ip_local_port_range}   # DM_PORT_RANGE_FILE: for the tests
  read -r a b c < "$f" 2>/dev/null || true
  if valid_range "$a" "$b" "$c"; then EPH_LO=$((10#$a)) EPH_HI=$((10#$b)); return 0; fi
  say "WARN: cannot read this host's ephemeral port range ($f): taking 9000 65000"
  EPH_LO=9000 EPH_HI=65000
}
derive_ports () {
  local p0 slots slot tried a
  eph_range
  if [ $((EPH_LO - 1300)) -ge 1024 ]; then p0=$((EPH_LO - 1300)) slots=650
  else
    a=$((EPH_HI + 1)); [ "$a" -lt 1024 ] && a=1024
    if [ $((a + 1299)) -le 65535 ]; then p0=$a slots=650; else p0=20000 slots=6000; fi
  fi
  slot=$((10#$SLURM_JOB_ID % slots)) tried=0
  while [ "$tried" -lt "$slots" ] && { port_busy $((p0 + 2 * slot)) || port_busy $((p0 + 2 * slot + 1)); }; do
    slot=$(((slot + 1) % slots)) tried=$((tried + 1))
  done
  [ "$tried" -ge "$slots" ] && say "WARN: every port pair of $p0-$((p0 + 2 * slots - 1)) is in use on this host: taking the job's own"
  export DM_MASTER_PORT=$((p0 + 2 * slot))
}
DERIVED=()
if [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
  if [ -z "${DM_MASTER_PORT:-}" ]; then derive_ports; DERIVED+=(DM_MASTER_PORT)
  else
    # one of the user's (or the site file's): kept; a port that cannot be one stops the chain (a configuration error)
    case "$DM_MASTER_PORT" in *[!0-9]*) echo "FATAL: DM_MASTER_PORT='$DM_MASTER_PORT' is not a whole number" >&2; exit 2 ;; esac
    if [ "$((10#$DM_MASTER_PORT))" -lt 1 ] || [ "$((10#$DM_MASTER_PORT))" -gt $([ -z "${DM_VALCE_PORT:-}" ] && echo 65534 || echo 65535) ]; then
      echo "FATAL: DM_MASTER_PORT=$DM_MASTER_PORT is not a usable port (1-65535, the val CE taking the next one unless DM_VALCE_PORT is set)" >&2; exit 2
    fi
    eph_range
    if [ "$((10#$DM_MASTER_PORT))" -ge "$EPH_LO" ] && [ "$((10#$DM_MASTER_PORT))" -le "$EPH_HI" ]; then
      say "WARN: DM_MASTER_PORT=$DM_MASTER_PORT lies in this host's ephemeral port range $EPH_LO-$EPH_HI: on a shared node another connection may hold it (EADDRINUSE at the rendezvous)"
    fi
  fi
  if [ -z "${DM_VALCE_PORT:-}" ]; then export DM_VALCE_PORT=$((10#$DM_MASTER_PORT + 1)); DERIVED+=(DM_VALCE_PORT); fi
fi
# the launchers' rank count (scripts/campaign/env.sh, default 8) follows the GPUs this job was given, and never exceeds
# them (a DM_NPROC=8 from the shell profile would start ranks on GPUs a partial-node job does not have)
if [[ ${SLURM_GPUS_ON_NODE:-} =~ ^[0-9]+$ ]]; then
  if [ -z "${DM_NPROC:-}" ]; then export DM_NPROC=$SLURM_GPUS_ON_NODE; DERIVED+=(DM_NPROC)
  elif [[ $DM_NPROC =~ ^[0-9]+$ ]] && [ "$DM_NPROC" -gt "$SLURM_GPUS_ON_NODE" ]; then
    say "WARN: DM_NPROC=$DM_NPROC but the job has $SLURM_GPUS_ON_NODE GPUs: running $SLURM_GPUS_ON_NODE ranks"
    export DM_NPROC=$SLURM_GPUS_ON_NODE; DERIVED+=(DM_NPROC)
  fi
fi
# squeue_end <job id>: the job's end (epoch seconds) from squeue's ISO end time; nothing (status 1) for anything else
# (squeue failing or printing nothing, "Unknown", another format): no deadline rather than a wrong one
squeue_end () {
  local e
  e=$(SLURM_TIME_FORMAT=standard timeout 10 squeue -h --jobs="$1" -o %e 2>/dev/null) || return 1
  [[ $e =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(:[0-9]{2})?$ ]] || return 1
  date -d "$e" +%s 2>/dev/null
}
# The job's end (epoch seconds): SLURM_JOB_END_TIME (Slurm >= 23.02), else squeue. DM_JOB_END is exported whenever it is
# known (the experiment driver budgets evaluations with it); never a stale one from the submitting shell.
END_S=""
unset DM_JOB_END
if [ -n "${SLURM_JOB_ID:-}" ]; then
  if [[ ${SLURM_JOB_END_TIME:-} =~ ^[0-9]+$ ]] && [ "$SLURM_JOB_END_TIME" -gt 0 ]; then
    END_S=$SLURM_JOB_END_TIME
  else
    END_S=$(squeue_end "$SLURM_JOB_ID") || END_S=""
  fi
  [ -n "$END_S" ] && export DM_JOB_END=$END_S
  if [ -z "${DM_STOP_BEFORE:-}" ]; then
    unset DM_STOP_BEFORE
    if [ -z "$END_S" ]; then
      say "WARN: no end time for job $SLURM_JOB_ID: no trainer deadline (the time limit's SIGTERM still makes the trainer save)"
    elif [ $((END_S - MARGIN)) -le $(( $(date +%s) + 600 )) ]; then
      say "WARN: job $SLURM_JOB_ID ends $(date -d "@$END_S" '+%F %T'), less than DM_STOP_MARGIN=$MARGIN s + 10 min from now: no trainer deadline and no resubmission (give a chain a longer time limit)"
      DM_RESUBMIT=0
    else
      export DM_STOP_BEFORE=$((END_S - MARGIN))
      say "job $SLURM_JOB_ID ends $(date -d "@$END_S" '+%F %T'); trainer deadline DM_STOP_BEFORE=$DM_STOP_BEFORE ($(date -d "@$DM_STOP_BEFORE" '+%F %T'))"
    fi
  fi
fi

# ------------------------------------------------------------------------------------------------------------ the chain
# CHAIN_OK: a job of a chain name with singleton (this submit.sh). CHAIN_MODE: and it chains (DM_RESUBMIT=1): state,
# successors, cancellations, in-place retries, compile caches. STATE_OK: its state directory is usable.
CHAIN=${SLURM_JOB_NAME:-} ME=${USER:-} NODE=${SLURMD_NODENAME:-$(hostname -s)}
CHAIN_OK=0 CHAIN_MODE=0 STATE_OK=0 FAILS=0 SUCC="" SUCC_HOW="" CDIR="" LINKS=/dev/null PF=/dev/null
[ -n "$ME" ] || ME=$(id -un 2>/dev/null) || ME=""
# the options every link replays (submit.sh: words without spaces); read -a splits them without glob expansion
read -r -a SB_OPTS <<< "${DM_SBATCH_OPTS:-}"
SINGLETON=0
for o in ${SB_OPTS[@]+"${SB_OPTS[@]}"}; do [ "$o" = --dependency=singleton ] && SINGLETON=1; done
if [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
  if [[ ! $CHAIN =~ ^[A-Za-z0-9._-]+$ ]] || [ "$CHAIN" = . ] || [ "$CHAIN" = .. ]; then
    say "WARN: the job name '$CHAIN' is not a chain name (letters, digits and ._- only): a single job, nothing is queued after it"
  elif [ "$SINGLETON" = 0 ]; then
    say "WARN: DM_SBATCH_OPTS has no --dependency=singleton, so this job was not submitted by this scripts/slurm/submit.sh (a chain of an older version?): a single job, nothing is queued after it; submit the chain again with scripts/slurm/submit.sh <options> -- $*"
  elif [ -z "$ME" ]; then
    say "WARN: no user name (USER): a single job, nothing is queued after it"
  else
    CHAIN_OK=1
  fi
fi
[ "$CHAIN_OK" = 1 ] && [ "${DM_RESUBMIT:-1}" = 1 ] && CHAIN_MODE=1
if [ "$CHAIN_MODE" = 1 ]; then
  D=${DM_CHAIN_DIR:-}
  if [ -z "$D" ] || [ "$(basename -- "$D")" != "$CHAIN" ]; then
    [ -n "$D" ] && say "WARN: DM_CHAIN_DIR=$D is not the state directory of the chain $CHAIN: using the workspace's"
    D=${DM_WORKSPACE:-$REPO/workspace}/logs/slurm/chains/$CHAIN
  fi
  case "$D" in /*) ;; *) D=${SLURM_SUBMIT_DIR:-$REPO}/$D ;; esac
  CDIR=$D LINKS=$D/links PF=$D/progress
  export DM_CHAIN=$CHAIN DM_CHAIN_DIR=$D DM_PROGRESS_FILE=$D/progress
  for t in 1 2 3; do
    if mkdir -p "$D" 2>/dev/null && [ -d "$D" ] && [ -w "$D" ]; then STATE_OK=1; break; fi
    [ "$t" -lt 3 ] && pause 2
  done
  [ "$STATE_OK" = 1 ] || say "WARN: the chain's state directory $D cannot be written: no failure accounting and no compile-cache archive in this job"
else
  # a single job (or no chain at all): the command runs outside a chain (no per-run failure counters, no progress lines,
  # a failure is final), whatever the submitting shell exported
  unset DM_CHAIN DM_CHAIN_DIR DM_PROGRESS_FILE
fi
# rec <event> [key=value...] / rec_at <epoch> <event> ...: append a record to `links` (best effort: never stops the job)
rec_at () {
  [ "$STATE_OK" = 1 ] || return 0
  { printf '%s %s %s\n' "$1" "${SLURM_JOB_ID:-0}" "${*:2}" >> "$LINKS"; } 2>/dev/null
  return 0
}
rec () { local t; printf -v t '%(%s)T' -1; rec_at "$t" "$@"; }
# progressed <job id> <epoch>: whether `progress` has a line of that job at or after that time
progressed () {
  [ "$STATE_OK" = 1 ] || return 1
  awk -v id="$1" -v t0="$2" '$2 == id && $1 >= t0 { f = 1; exit } END { exit !f }' "$PF" 2>/dev/null
}
# pending_member: the lowest id of the chain's PENDING jobs other than this one (none: prints nothing); status 1 when
# squeue itself failed
pending_member () {
  local out id best=""
  out=$(timeout 30 squeue -h -u "$ME" -n "$CHAIN" -t PENDING -o %i 2>/dev/null) || return 1
  while read -r id; do
    [[ $id =~ ^[0-9]+$ ]] && [ "$id" != "${SLURM_JOB_ID:-}" ] || continue
    if [ -z "$best" ] || [ "$id" -lt "$best" ]; then best=$id; fi
  done <<< "$out"
  [ -n "$best" ] && echo "$best"
  return 0
}
# cancel_pending: cancel the chain's other jobs by id: the pending ones and one that Slurm already started while this
# job completes (singleton releases a successor when this job's state is final); never this job, a COMPLETING one (it is
# ending), or the first job of a newer submission (an id of a `submit` record: the user submitted the chain again while
# it ran; that submission continues the chain). Best effort, logged; squeue failing: the pending jobs by name.
cancel_pending () {
  [ "$CHAIN_MODE" = 1 ] || return 0
  local out id st ids=() keep=" " rc
  if out=$(timeout 30 squeue -h -u "$ME" -n "$CHAIN" -o '%i %T' 2>/dev/null); then
    [ "$STATE_OK" = 1 ] && keep=" $(awk '$3 == "submit" { printf "%s ", $2 }' "$LINKS" 2>/dev/null) "
    while read -r id st; do
      [[ $id =~ ^[0-9]+$ ]] && [ "$id" != "${SLURM_JOB_ID:-}" ] || continue
      case "$st" in COMPLETING|STAGE_OUT|SIGNALING) continue ;; esac
      case "$keep" in *" $id "*) say "not cancelling job $id: a newer submission of the chain (it continues the chain)"; continue ;; esac
      ids+=("$id")
    done <<< "$out"
    if [ "${#ids[@]}" -eq 0 ]; then say "the chain has no other job to cancel"; return 0; fi
    timeout 30 scancel "${ids[@]}"; rc=$?
    if [ "$rc" -eq 0 ]; then say "cancelled the chain's other jobs: ${ids[*]}"
    else say "WARN: cancelling the chain's jobs failed (status $rc): cancel them by hand: scancel ${ids[*]}"; fi
  else
    timeout 30 scancel -u "$ME" -n "$CHAIN" -t PENDING; rc=$?
    if [ "$rc" -eq 0 ]; then say "cancelled the chain's pending jobs (squeue failed: scancel -u $ME -n $CHAIN -t PENDING)"
    else say "WARN: cancelling the chain's jobs failed (status $rc): cancel them by hand: scancel -u $ME -n $CHAIN"; fi
  fi
}
# submit_successor: sbatch the next link (prints sbatch's output). From the directory the chain was submitted from
# (relative paths of the options and the site file mean the same in every link), with this job's environment minus its
# own deadline, end time, chain variables and DERIVED ones, and the next link index; job.sh's own environment is left as
# it is. The chain's name is given last, whatever the recorded options say: a successor of another name would be
# outside the chain (no singleton with this job, never adopted or cancelled).
submit_successor () {
  (
    cd "${SLURM_SUBMIT_DIR:-$REPO}" 2>/dev/null || cd "$REPO" || exit 1
    unset DM_STOP_BEFORE DM_JOB_END DM_CHAIN DM_PROGRESS_FILE DM_DERIVED ${DERIVED[@]+"${DERIVED[@]}"}
    export DM_CHAIN_DIR=$CDIR DM_RESUBMIT_COUNT=$((K + 1))
    exec timeout 90 sbatch --parsable ${SB_OPTS[@]+"${SB_OPTS[@]}"} --job-name="$CHAIN" "$REPO/scripts/slurm/job.sh" "$@"
  )
}
# queue_successor <start|end> <command...>: SUCC = a pending job of the chain (adopted) or a new one. One attempt now,
# then one after each wait of the schedule; a pending job is looked for before every attempt, so an sbatch that timed
# out after queueing its job is not repeated. At the start the schedule is short (the command is waiting) and a signal
# ends it; at the end it runs until 90 s before the job's end, and for at most 30 minutes (a node held idle; 600 s when
# the end is unknown).
queue_successor () {
  local phase=$1 waits w try=0 j limit="" i now
  shift
  if [ -n "${DM_RESUBMIT_BACKOFF+x}" ]; then waits=$DM_RESUBMIT_BACKOFF
  elif [ "$phase" = start ]; then waits="10 30"
  else waits="10 30 60"; for ((i = 0; i < 300; i++)); do waits+=" 120"; done; fi
  if [ "$phase" = end ]; then
    printf -v now '%(%s)T' -1
    if [ -n "$END_S" ]; then limit=$((END_S - 90)); else limit=$((now + 600)); fi
    [ "$limit" -gt $((now + 1800)) ] && limit=$((now + 1800))
  fi
  for w in 0 $waits; do
    w=$((10#$w))
    if [ "$try" -gt 0 ]; then
      [ "$phase" = start ] && [ -n "$SIG" ] && break
      [ -n "$limit" ] && [ $(( $(date +%s) + w )) -gt "$limit" ] && break
      say "WARN: sbatch failed; retry $try in $w s"
      pause "$w"
    fi
    if j=$(pending_member) && [ -n "$j" ]; then SUCC=$j; SUCC_HOW=adopted; return 0; fi
    try=$((try + 1))
    if j=$(submit_successor "$@"); then
      j=${j##*$'\n'}; j=${j%%;*}   # --parsable: <id>[;<cluster>] (its last line)
      if [[ $j =~ ^[0-9]+$ ]]; then SUCC=$j; SUCC_HOW=queued; return 0; fi
      say "WARN: sbatch printed '$j', not a job id"
    fi
  done
  return 1
}
succ_said () {
  if [ "$SUCC_HOW" = adopted ]; then say "pending job $SUCC of the chain is the successor (chain step $((K + 1)); it starts when this job ends)"
  else say "queued successor $SUCC (chain step $((K + 1)); it starts when this job ends)"; fi
}
# succ_state <id>: prints the job's state (nothing: Slurm no longer knows it); status 1 when squeue failed
succ_state () {
  local out rc line
  out=$(timeout 10 squeue -h --jobs="$1" -o %T 2>&1); rc=$?
  if [ "$rc" -ne 0 ]; then
    case "$out" in *[Ii]nvalid\ job\ id*) return 0 ;; esac   # purged (MinJobAge after its end): gone
    return 1
  fi
  while read -r line; do [[ $line =~ ^[A-Z][A-Z_]*$ ]] && { echo "$line"; return 0; }; done <<< "$out"
  return 0
}
# exclude_node: keep the successor off the node(s) this link failed on (best effort; the job's own exclusions are kept:
# scontrol replaces the list). The hosts train_arm.sh blamed a failure on in $DM_CHAIN_DIR/badnodes during this job
# (its "<epoch> <job id> <host>" lines with this job's id, since this run of the job began); when there are none, every
# node of this job: one node, this node (SLURMD_NODENAME); several, scontrol show hostnames $SLURM_JOB_NODELIST.
exclude_node () {
  local old new h nn why hosts=() add=()
  [ -n "$SUCC" ] || return 0
  if [ "$STATE_OK" = 1 ] && [ -f "$CDIR/badnodes" ]; then
    while read -r h; do hosts+=("$h"); done < <(awk -v id="${SLURM_JOB_ID:-0}" -v t0="$T_LINK" \
      '$2 == id && $1 ~ /^[0-9]+$/ && $1 >= t0 && $3 ~ /^[A-Za-z0-9._-]+$/ && !seen[$3]++ { print $3 }' "$CDIR/badnodes" 2>/dev/null)
  fi
  if [ ${#hosts[@]} -gt 0 ]; then
    why="the command failed without progress; train_arm.sh blamed the failure on it in $CDIR/badnodes"
  else
    why="the command failed here without progress"
    nn=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}
    if [[ $nn =~ ^[0-9]+$ ]] && [ "$nn" -gt 1 ] && [ -n "${SLURM_JOB_NODELIST:-}" ]; then
      mapfile -t hosts < <(timeout 10 scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null)
      [ ${#hosts[@]} -gt 0 ] || say "WARN: cannot list the nodes of this job (scontrol show hostnames $SLURM_JOB_NODELIST): excluding this node only"
    fi
    if [ ${#hosts[@]} -eq 0 ] && [ -n "${SLURMD_NODENAME:-}" ]; then hosts=("$SLURMD_NODENAME"); fi
  fi
  [ ${#hosts[@]} -gt 0 ] || return 0
  if ! old=$(timeout 10 squeue -h --jobs="$SUCC" -o %x 2>/dev/null); then
    say "WARN: cannot read the excluded nodes of the successor $SUCC: not excluding ${hosts[*]}"; return 0
  fi
  old=${old//[[:space:]]/}
  [ "$old" = "(null)" ] && old=""
  for h in "${hosts[@]}"; do case ",$old," in *",$h,"*) ;; *) add+=("$h") ;; esac; done
  [ ${#add[@]} -gt 0 ] || return 0
  new=$(IFS=,; echo "${add[*]}")
  [ -n "$old" ] && new=$old,$new
  if timeout 30 scontrol update JobId="$SUCC" ExcNodeList="$new"; then
    if [ ${#add[@]} -eq 1 ]; then say "excluded node ${add[0]} from the successor $SUCC ($why)"
    else say "excluded nodes ${add[*]} from the successor $SUCC ($why)"; fi
  else say "WARN: excluding node(s) ${add[*]} from the successor $SUCC failed (scontrol update JobId=$SUCC ExcNodeList=$new)"; fi
}
# publish_jit: save changed compile caches to jit.tar (atomic rename; at most DM_JIT_MAX_MB; only when the link ended by
# itself, with >= 120 s left: after a signal Slurm kills the job KillWait later)
publish_jit () {
  [ -n "$JIT" ] && [ "$STATE_OK" = 1 ] && [ -f "$JIT/.stamp" ] || return 0
  local left mb to t tmp=$CDIR/jit.tar.tmp.${SLURM_JOB_ID:-0}
  if [ -n "$SIG" ]; then say "compile caches not saved: the job is ending (SIG$SIG)"; return 0; fi
  if [ -z "$END_S" ]; then say "compile caches not saved: the job's end is unknown"; return 0; fi
  left=$((END_S - $(date +%s)))
  if [ "$left" -lt 120 ]; then say "compile caches not saved: $left s left"; return 0; fi
  if [ -z "$(find "$JIT" -mindepth 1 -newer "$JIT/.stamp" -print -quit 2>/dev/null)" ]; then   # (not $JIT: .stamp is in it)
    say "compile caches unchanged: jit.tar kept"; return 0
  fi
  mb=$(du -sm "$JIT" 2>/dev/null | cut -f1) || mb=""
  [[ $mb =~ ^[0-9]+$ ]] || { say "WARN: cannot size the compile caches in $JIT: not saved"; return 0; }
  if [ "$mb" -gt "$JIT_MAX" ]; then say "compile caches not saved: $mb MB > DM_JIT_MAX_MB=$JIT_MAX"; return 0; fi
  to=$((left - 60)); [ "$to" -gt 300 ] && to=300
  t=$(date +%s)
  if run_bg timeout "$to" tar -cf "$tmp" --exclude=./.stamp -C "$JIT" . && mv -f "$tmp" "$CDIR/jit.tar"; then
    say "compile caches saved to $CDIR/jit.tar ($mb MB, $(( $(date +%s) - t )) s)"
  else
    rm -f "$tmp" 2>/dev/null
    say "WARN: saving the compile caches to $CDIR/jit.tar failed: the next link starts with the previous archive"
  fi
}
# wait_others: Slurm's singleton releases this job as soon as its predecessor's state is final (TIMEOUT, CANCELLED, ...),
# but the predecessor stays COMPLETING while its processes end (a checkpoint save, KillWait, an unkillable rank): they
# may hold the run's lock, and its job.sh may still write its records and act on its successor (this job). Wait until no
# other job of the chain is RUNNING or COMPLETING, at most DM_PRED_WAIT seconds; then start anyway (the run lock guards
# the run). Polls after 2, 5, 10, 10... s; a signal ends the wait.
wait_others () {
  local t0 now out id ids w n=0
  printf -v t0 '%(%s)T' -1
  while [ -z "$SIG" ]; do
    if ! out=$(timeout 30 squeue -h -u "$ME" -n "$CHAIN" -t RUNNING,COMPLETING -o %i 2>/dev/null); then
      say "WARN: cannot check whether another job of the chain is still running or completing (squeue failed): not waiting"
      return 0
    fi
    ids=""
    while read -r id; do [[ $id =~ ^[0-9]+$ ]] && [ "$id" != "${SLURM_JOB_ID:-}" ] && ids+=" $id"; done <<< "$out"
    printf -v now '%(%s)T' -1
    if [ -z "$ids" ]; then
      [ "$n" -gt 0 ] && say "the chain's other job has ended ($((now - t0)) s)"
      return 0
    fi
    if [ $((now - t0)) -ge "$PRED_WAIT" ]; then
      say "WARN: job(s)$ids of the chain still running or completing after $((now - t0)) s (DM_PRED_WAIT=$PRED_WAIT): starting anyway"
      return 0
    fi
    [ "$n" -eq 0 ] && say "waiting for job(s)$ids of the chain to end (Slurm started this job while it completes; at most DM_PRED_WAIT=$PRED_WAIT s)"
    case "$n" in 0) w=2 ;; 1) w=5 ;; *) w=10 ;; esac
    [ $((now - t0 + w)) -gt "$PRED_WAIT" ] && w=$((PRED_WAIT - (now - t0)))
    n=$((n + 1))
    pause "$w"
  done
  return 0
}
[ "$CHAIN_OK" = 1 ] && [ "$PRED_WAIT" -gt 0 ] && wait_others

# Judge a predecessor S (the last `start` record) that left no `end` record (SIGKILL after KillWait, node failure +
# requeue, an OOM-killed job.sh). Its attempts with an `attempt` record are counted already. When its last `run` has no
# `attempt` record, that attempt was cut off: one more failure unless `progress` has a line of S since that run began;
# when S has no `run` at all it died before its command: one more failure unless it progressed since its start. A
# requeued job judges its own previous run. FAILS = the failure count of the last record that has one. The first job of
# a submission (a `submit` record with its id) starts at 0: records of the chain's older jobs after it do not count.
if [ "$STATE_OK" = 1 ]; then
  read -r FAILS S_ID S_END S_LAST S_T0 FRESH < <(awk -v me="${SLURM_JOB_ID:-0}" '
    $3 == "submit" && $2 == me { fresh = 1; f = 0; si = ""; next }
    fresh && $2 != me { next }
    { for (i = 4; i <= NF; i++) if ($i ~ /^fails=[0-9]+$/) f = substr($i, 7) }
    $3 == "start" { si = $2; ended = 0; last = "none"; t = $1; next }
    si != "" && $2 == si && $3 == "run" { last = "run"; t = $1 }
    si != "" && $2 == si && $3 == "attempt" { last = "attempt" }
    si != "" && $2 == si && $3 == "end" { ended = 1 }
    END { if (si == "") print (f == "" ? 0 : f), "-", 1, "-", "-", fresh + 0
          else print (f == "" ? 0 : f), si, ended + 0, last, t, fresh + 0 }' "$LINKS" 2>/dev/null)
  [[ ${FAILS:-} =~ ^[0-9]+$ ]] && FAILS=$((10#$FAILS)) || FAILS=0
  [ "${FRESH:-0}" = 1 ] && say "the first job of a submission of the chain: failure count 0"
  if [[ ${S_ID:-} =~ ^[0-9]+$ ]] && [ "${S_END:-1}" = 0 ]; then
    if [ "${S_LAST:-}" = attempt ]; then
      say "the chain's previous link (job $S_ID) ended without a record of its end, after recording its attempts: $FAILS attempt(s) in a row without progress"
      rec "judge of=$S_ID progress=- fails=$FAILS"
    else
      [[ ${S_T0:-} =~ ^[0-9]+$ ]] || S_T0=0
      if [ "${S_LAST:-}" = run ]; then why="its last attempt was cut off"; else why="before its command ran"; fi
      if progressed "$S_ID" "$S_T0"; then P=1 FAILS=0; else P=0 FAILS=$((FAILS + 1)); fi
      say "the chain's previous link (job $S_ID) ended without a record (killed after the time limit, node failure, requeue; $why): progress=$P, $FAILS attempt(s) in a row without progress"
      rec "judge of=$S_ID progress=$P fails=$FAILS"
    fi
  fi
fi
if [ "$CHAIN_MODE" = 1 ] && [ "$FAILS" -gt "$MAX_FAILS" ]; then
  say "giving up: $FAILS links in a row without progress (DM_MAX_FAILS=$MAX_FAILS): resume with $CDIR/submit.sh once the cause is fixed"
  cancel_pending
  rec "end k=$K rc=4 action=giveup fails=$FAILS"
  exit 4
fi
[ "$CHAIN_MODE" = 1 ] && say "chain $CHAIN (state: $CDIR): link $K, $FAILS attempt(s) in a row without progress before it"
rec "start k=$K restart=${SLURM_RESTART_COUNT:-0} node=$NODE fails=$FAILS"
if [ "$CHAIN_MODE" = 1 ] && [ "$K" -lt "$MAX_RESUB" ] && [ -z "$SIG" ]; then
  if queue_successor start "$@"; then rec "succ id=$SUCC how=$SUCC_HOW"; succ_said
  else say "WARN: no successor could be queued at the start of this job: trying again at its end"; fi
fi

# The compile caches: one fixed node-local path per chain (Triton's cache records absolute paths), restored from the
# chain's archive; one link of a chain runs at a time (singleton, wait_others), so no two jobs share the path.
# DM_JIT_CACHE_DIR names it for the launchers of the other nodes of a multi-node job (scripts/campaign/node_ranks.sh
# restores the archive there); set here only, never inherited.
unset DM_JIT_CACHE_DIR
if [ "$CHAIN_MODE" = 1 ] && [ "${DM_JIT_CACHE:-1}" = 1 ]; then
  root=${DM_JIT_ROOT:-/tmp/dm-jit-$(id -u)}
  if [[ $root != /* ]]; then
    say "WARN: DM_JIT_ROOT=$root is not an absolute path: the compile caches are not kept"
  elif rm -rf "$root/$CHAIN" 2>/dev/null; mkdir -p "$root/$CHAIN" 2>/dev/null; then
    JIT=$root/$CHAIN
    [ "$STATE_OK" = 1 ] && rm -f "$CDIR"/jit.tar.tmp.* 2>/dev/null   # left by a killed link
    if [ "$STATE_OK" = 1 ] && [ -f "$CDIR/jit.tar" ] && [ -z "$SIG" ]; then
      t=$(date +%s)
      if run_bg timeout 300 tar -xf "$CDIR/jit.tar" -C "$JIT"; then
        say "compile caches restored from $CDIR/jit.tar into $JIT ($(( $(date +%s) - t )) s)"
      else
        say "WARN: restoring the compile caches from $CDIR/jit.tar failed: starting with empty caches"
        rm -rf "$JIT" 2>/dev/null; mkdir -p "$JIT" 2>/dev/null
      fi
    fi
    # what the command adds is newer than the stamp: the restored files keep the archive's (older) times and the
    # stamp is a second old, so that a file system with 1-second times still tells a file written right after it
    touch -d "@$(( $(date +%s) - 1 ))" "$JIT/.stamp" 2>/dev/null
    export TRITON_CACHE_DIR=$JIT/triton TILELANG_CACHE_DIR=$JIT/tilelang TORCHINDUCTOR_CACHE_DIR=$JIT/inductor DM_JIT_CACHE_DIR=$JIT
    DERIVED+=(TRITON_CACHE_DIR TILELANG_CACHE_DIR TORCHINDUCTOR_CACHE_DIR DM_JIT_CACHE_DIR)
  else
    say "WARN: cannot create $root/$CHAIN: the compile caches are not kept"
  fi
fi
# what this job set itself, for submit.sh run from inside it (a new chain derives its own)
if [ "${#DERIVED[@]}" -gt 0 ]; then export DM_DERIVED="${DERIVED[*]}"; else unset DM_DERIVED; fi

# ---------------------------------------------------------------------------------------------------------- the command
cd "$REPO" || exit 2
export DM_RESUBMIT_COUNT=$K
say "chain step $K: $*"
# limit_end: this job's planned end = its start + its time limit, as squeue shows them now (`scontrol update TimeLimit=`
# moves it); not squeue's end time, which a scancel or a requeue sets to the moment of the stop. Start: %S, else
# SLURM_JOB_START_TIME; limit %l: [days-]hours:minutes:seconds or minutes:seconds (UNLIMITED...: no end, status 1).
limit_end () {
  local out st lim
  out=$(SLURM_TIME_FORMAT=standard timeout 10 squeue -h --jobs="$SLURM_JOB_ID" -o '%S %l' 2>/dev/null) || return 1
  read -r st lim <<< "$out"
  if [[ $st =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(:[0-9]{2})?$ ]]; then st=$(date -d "$st" +%s 2>/dev/null) || return 1
  elif [[ ${SLURM_JOB_START_TIME:-} =~ ^[0-9]+$ ]]; then st=$SLURM_JOB_START_TIME
  else return 1; fi
  [[ ${lim:-} =~ ^(([0-9]+)-)?(([0-9]+):)?([0-9]+):([0-9]+)$ ]] || return 1
  echo $(( st + (((10#${BASH_REMATCH[2]:-0} * 24 + 10#${BASH_REMATCH[4]:-0}) * 60 + 10#${BASH_REMATCH[5]}) * 60) + 10#${BASH_REMATCH[6]} ))
}
# classify: what the signal meant (STOP): "requeue" (Slurm put this job back in the queue: it runs again with this id),
# "limit" (the time limit) or "user" (a stop by hand; END_NOW = the planned end). A shortened time limit also shortens
# END_S (the end of the link's own budget).
STOP="" END_NOW="" ATT=0 RETRIED=0 RC=0 LAST_P=0
classify () {
  local e="" out rs
  [[ $T_SIG =~ ^[0-9]+$ ]] || printf -v T_SIG '%(%s)T' -1
  if [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
    # a requeue: Slurm raises the job's restart count at once (squeue's RestartCnt) while the job stays COMPLETING until
    # this script exits (Slurm 24.05.5: never PENDING meanwhile), so the count is compared with this instance's own
    # SLURM_RESTART_COUNT; a job already PENDING / REQUEUED again counts too
    rs=${SLURM_RESTART_COUNT:-0}; [[ $rs =~ ^[0-9]+$ ]] || rs=0
    if { out=$(timeout 10 squeue -h --jobs="$SLURM_JOB_ID" -O RestartCnt 2>/dev/null) && out=${out//[[:space:]]/} \
         && [[ $out =~ ^[0-9]+$ ]] && [ "$((10#$out))" -gt "$((10#$rs))" ]; } \
       || { out=$(timeout 10 squeue -h --jobs="$SLURM_JOB_ID" -t PENDING,REQUEUED,REQUEUE_HOLD,REQUEUE_FED -o %i 2>/dev/null) \
            && [[ $'\n'$out$'\n' == *$'\n'"$SLURM_JOB_ID"$'\n'* ]]; }; then
      STOP=requeue
      say "SIG$SIG: Slurm requeued this job (preemption, scontrol requeue): it runs again as job $SLURM_JOB_ID"
      return 0
    fi
    e=$(limit_end) || e=""
  fi
  [ -n "$e" ] || e=$END_S
  if [ -n "$e" ] && { [ -z "$END_S" ] || [ "$e" -lt "$END_S" ]; }; then END_S=$e; fi
  if [ -z "$e" ] || [ "$T_SIG" -ge $((e - 60)) ]; then
    STOP=limit
    if [ -n "$e" ]; then say "SIG$SIG $((e - T_SIG)) s before the end of the job: the time limit"
    else say "SIG$SIG, the end of the job unknown: taken as the time limit"; fi
  else
    STOP=user END_NOW=$e
  fi
}
while :; do
  ATT=$((ATT + 1)) T0=$(date +%s)
  if [ -n "$SIG" ]; then
    RC=$(sig_rc "$SIG")
    say "SIG$SIG before the command started: not running it (status $RC)"
  else
    # The command runs in the background so that a signal is handled at once (bash runs a trap only after a foreground
    # command ends); wait again while it is still running (a signal interrupted the wait), and tell what the signal
    # meant meanwhile (the command's shutdown is part of KillWait too).
    if [ "$ATT" -gt 1 ] && [[ " ${DERIVED[*]} " == *" DM_MASTER_PORT "* ]]; then
      PREV_PORT=$DM_MASTER_PORT; derive_ports
      [[ " ${DERIVED[*]} " == *" DM_VALCE_PORT "* ]] && export DM_VALCE_PORT=$((DM_MASTER_PORT + 1))
      [ "$DM_MASTER_PORT" != "$PREV_PORT" ] && say "rendezvous ports for attempt $ATT: $DM_MASTER_PORT/${DM_VALCE_PORT:-} ($PREV_PORT or the next one is in use now)"
    fi
    rec_at "$T0" "run n=$ATT"
    "$@" &
    CPID=$!
    [ -n "$SIG" ] && kill -TERM "$CPID" 2>/dev/null   # a signal that came before the command started
    wait "$CPID"; RC=$?
    while kill -0 "$CPID" 2>/dev/null; do
      [ -n "$SIG" ] && [ -z "$STOP" ] && classify
      wait "$CPID"; RC=$?
    done
    say "exit $RC${SIG:+ after a SIG$SIG}: $*"
  fi
  [ -n "$SIG" ] && [ -z "$STOP" ] && classify
  case "$RC" in 0|2|3|4) break ;; esac
  [ "$STOP" = user ] && break
  if progressed "${SLURM_JOB_ID:-0}" "$T0"; then LAST_P=1 FAILS=0
  elif [ "$STOP" = requeue ]; then LAST_P=0   # Slurm took the job away: not the command's failure
  else LAST_P=0 FAILS=$((FAILS + 1)); fi
  rec "attempt n=$ATT rc=$RC progress=$LAST_P fails=$FAILS"
  # A failure retried in place: no queue wait, same deadline. The failed attempt's ranks are gone by now (train_arm.sh's
  # collector kills the survivors of a failed rank).
  if [ "$CHAIN_MODE" = 1 ] && [ "$RC" -ne 75 ] && [ -z "$SIG" ] && [ "$FAILS" -le "$MAX_FAILS" ] \
     && [ "$RETRIED" -lt "$INPLACE" ] && [ -n "${DM_STOP_BEFORE:-}" ] \
     && [ $((10#$DM_STOP_BEFORE - $(date +%s))) -ge "$MIN_LEFT" ]; then
    RETRIED=$((RETRIED + 1))
    say "exit $RC (progress: $LAST_P): retrying in place in $PAUSE s (retry $RETRIED of DM_INPLACE_RETRIES=$INPLACE, $((10#$DM_STOP_BEFORE - $(date +%s))) s before the deadline)"
    pause "$PAUSE"
    if [ -n "$SIG" ]; then classify; break; fi
    continue
  fi
  break
done

# ------------------------------------------------------------------------------------------------- the end of the link
end_rec () { rec "end k=$K rc=$1 action=$2 fails=$FAILS${3:+ $3}"; }
user_stopped () { say "stopped by SIG$SIG $((END_NOW - T_SIG)) s before the time limit${1:+: $1}"; }
if [ "$RC" -eq 0 ]; then
  end_rec 0 done
  [ "$CHAIN_MODE" = 1 ] && say "exit 0: everything is done; the chain is complete"
  cancel_pending
  exit 0
fi
if [ "$STOP" = user ] || [ "$RC" -eq 2 ] || [ "$RC" -eq 3 ] || [ "$RC" -eq 4 ]; then
  end_rec "$RC" stop
  if [ "$STOP" = user ]; then
    if [ "$CHAIN_MODE" = 1 ]; then user_stopped "the chain is stopped"; else user_stopped; fi
  elif [ "$CHAIN_MODE" = 1 ]; then
    case "$RC" in 2) why="a configuration error" ;; 3) why="the run belongs to another live job" ;; *) why="it gave up" ;; esac
    say "exit $RC ($why): the chain is stopped"
  fi
  cancel_pending
  exit "$RC"
fi
[ "$CHAIN_MODE" = 1 ] || exit "$RC"
if [ "$STOP" = requeue ]; then
  end_rec "$RC" requeued "succ=$SUCC"
  say "exit $RC: Slurm requeued this job: it runs again as job $SLURM_JOB_ID and continues the chain (nothing cancelled or queued)"
  exit "$RC"
fi
if [ "$FAILS" -gt "$MAX_FAILS" ]; then
  end_rec "$RC" giveup
  say "giving up after $FAILS attempts without progress (last exit $RC); resume with $CDIR/submit.sh once the cause is fixed"
  cancel_pending
  exit 4
fi
if [ "$K" -ge "$MAX_RESUB" ]; then
  say "exit $RC: the chain reached DM_MAX_RESUBMITS=$MAX_RESUB links: no successor"
  end_rec "$RC" max
  exit "$RC"
fi
EXCL=1   # the successor is pending: it can still be kept off the node(s) this link failed on
if [ -n "$SUCC" ]; then
  if st=$(succ_state "$SUCC"); then
    case "$st" in
      PENDING) ;;
      ""|CANCELLED)
        end_rec "$RC" stop "succ=$SUCC"
        say "successor $SUCC was cancelled (${st:-no longer known to Slurm}): chain stopped"
        cancel_pending
        exit "$RC" ;;
      *)   # singleton released it when this job's state became final (the time limit): it runs the chain on
        EXCL=0
        say "the successor $SUCC is $st already (Slurm started it while this job completes)" ;;
    esac
  else
    say "WARN: cannot check the successor $SUCC (squeue failed): assuming it is still queued"
  fi
else
  if ! queue_successor end "$@"; then
    end_rec "$RC" broken
    say "ERROR: no successor could be queued; the chain is broken: resume with $CDIR/submit.sh"
    exit 76
  fi
  rec "succ id=$SUCC how=$SUCC_HOW"
  succ_said
fi
[ "$EXCL" = 1 ] && [ "$RC" -ne 75 ] && [ "$LAST_P" = 0 ] && [ -z "$SIG" ] && exclude_node
# late_stop: a stop by hand that came while the end of the link was handled (a successor queued just now would outlive
# the user's scancel) is a stop too
late_stop () {
  [ -n "$SIG" ] && [ -z "$STOP" ] || return 0
  classify
  [ "$STOP" = user ] || return 0
  end_rec "$RC" stop "succ=$SUCC"
  user_stopped "the chain is stopped"
  cancel_pending
  exit "$RC"
}
late_stop
end_rec "$RC" continue "succ=$SUCC"
say "exit $RC: the successor $SUCC continues the chain"
publish_jit
late_stop
exit "$RC"
