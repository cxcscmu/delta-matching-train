#!/bin/bash
# scripts/campaign/node_ranks.sh <run> <python> <trainer args...>: the trainer ranks of one node of a multi-node launch.
# scripts/campaign/train_arm.sh starts it on every node of the job (one srun task per node, all in one step); it is not
# run by hand. NODE = SLURM_NODEID (else SLURM_PROCID). It starts DM_NPROC ranks, global RANK = NODE * DM_NPROC + r,
# LOCAL_RANK = r, with the per-rank compile caches and the appended rank logs of a one-node launch
# ($DM_WORKSPACE/logs/<run>.rank<RANK>.log, one header line per launch), and collects them with the same collector and
# signal forwarding (scripts/campaign/ranks.sh: DM_RANK_GRACE, DM_TERM_GRACE under Slurm, DM_EXIT_GRACE; a 75 wins);
# the last 30 lines each failed rank wrote in this launch are printed.
# Per-rank compile caches: <root>/r<local rank> (ranks.sh), with root = TRITON_CACHE_DIR / TILELANG_CACHE_DIR, as on one
# node; two ranks must never share one (a rank reads a metadata file another is still writing, dies, and its peers hang
# in a collective). A root that other nodes may see as well (a shared file system: env.sh's default in the workspace, a
# site's) gets a level per node on every node but the batch host, <root>/n<NODE>/r<local rank>. A node-local root keeps
# <root>/r<local rank>, the layout of the chain's jit.tar: one under /tmp or /dev/shm, or under the compile-cache
# directory this launcher created (below).
# From train_arm.sh, through srun's --export=ALL: MASTER_ADDR / MASTER_PORT / WORLD_SIZE / LOCAL_WORLD_SIZE,
# DM_NODE_STATUS (this launch's status directory, on the shared file system), DM_BATCH_HOST (the host train_arm.sh runs
# on), DM_WORKSPACE, DM_REPO, DM_NPROC, DM_GIT_REV; from scripts/slurm/job.sh: DM_JIT_CACHE_DIR, DM_CHAIN_DIR; from the
# site file: TMPDIR.
# The status directory:
#   node<NODE>.pid  "<pid> <host> <step id>", written (atomically) as soon as the signal handler is in place:
#                   train_arm.sh passes a stop to the launcher of its own host through this pid
#   stop            a stop train_arm.sh received before it knew that pid, or one a launcher received before it started
#                   its ranks: looked at before the ranks start (then none do)
#   node<NODE>      "<status> <host> <first failure, epoch seconds, or -> [<failed global ranks>...]", written
#                   (atomically) when the ranks are done; train_arm.sh aggregates these after srun has returned. The
#                   first failure is one that came before any stop (ranks.sh wait_ranks: also a rank already gone when
#                   the stop came)
# Signals: TERM, INT, HUP, USR1 and USR2 (scripts/slurm/job.sh's stop signals) are passed on to the ranks once, as
# SIGTERM (ranks.sh forward_term, builtins only). They come from train_arm.sh (on the batch host only: the trainer
# MAX-all-reduces its stop flag at the next step boundary, so every rank of every node saves the same checkpoint and
# exits 75), from srun (kill-on-bad-exit: another node's task failed) or from Slurm itself (the time limit and scancel
# reach every task of a step, i.e. this launcher and not its ranks: Slurm signals a step's tasks only, unless the signal
# is SIGKILL; so does `scancel --signal=<sig> <job>`, which a launcher must survive: one killed by it would leave no
# report, and its ranks no stop). A stop that came while the ranks were being started is passed to the ranks started
# after it too.
# Compile caches and scratch: DM_JIT_CACHE_DIR (the chain's node-local cache directory, which job.sh restored on the
# batch host) is created when it does not exist on this node, and $DM_CHAIN_DIR/jit.tar extracted into it (at most
# 300 s; a failure only warns, the ranks then start with empty caches); TMPDIR is created when it is missing. On a node
# other than the batch host this launcher removes at its exit the cache directory it created, and TMPDIR when it is
# exactly /tmp/dm-$SLURM_JOB_ID (the site file's); the batch host's are job.sh's.
# Exit status: 0 when the ranks exited 0 or 75 (the real status is in node<NODE>: a node that is done must not make srun
# cut the others short while they still save a checkpoint or export the model); 0 with status 143 recorded when a stop
# came before the ranks started (none started); 1 when it cannot tell its node number or has no status directory;
# otherwise the ranks' failure (srun then ends the other nodes' tasks at once: their ranks would wait in a collective for
# the dead ones until the 30-minute process-group timeout).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ $# -ge 2 ] || { echo "usage: node_ranks.sh <run> <python> <trainer args...> (started by scripts/campaign/train_arm.sh)" >&2; exit 1; }
RN=$1 PY=$2
shift 2
NODE=${SLURM_NODEID:-${SLURM_PROCID:-}} STATUS=${DM_NODE_STATUS:-} W=${DM_WORKSPACE:-}
# shellcheck source=scripts/campaign/ranks.sh
source "$HERE/ranks.sh"
HOST=${SLURMD_NODENAME:-$DM_HOST}
DM_HOST=$HOST   # the collector's messages name this node as Slurm does
trap forward_term TERM INT HUP USR1 USR2
say () { echo "[$HOST $(date +%T)] node ${NODE:-?}: $*"; }
case $NODE in ''|*[!0-9]*) say "FATAL: no node number (SLURM_NODEID / SLURM_PROCID): not started by srun?"; exit 1 ;; esac
[ -n "$STATUS" ] && [ -d "$STATUS" ] || { say "FATAL: the launch's status directory DM_NODE_STATUS='$STATUS' does not exist here"; exit 1; }
[ -n "$W" ] || { say "FATAL: DM_WORKSPACE is not set (the rank logs)"; exit 1; }
[[ ${DM_NPROC:-} =~ ^[0-9]+$ ]] && [ "$DM_NPROC" -ge 1 ] || { say "FATAL: DM_NPROC='${DM_NPROC:-}' is not a rank count"; exit 1; }
[[ ${WORLD_SIZE:-} =~ ^[0-9]+$ ]] || { say "FATAL: WORLD_SIZE='${WORLD_SIZE:-}' is not a rank count (set by train_arm.sh)"; exit 1; }
DM_GIT_REV=${DM_GIT_REV:-none}
[ -n "${DM_REPO:-}" ] && { cd "$DM_REPO" || say "WARN: cannot cd to $DM_REPO"; }
# put <file> <line>: write it atomically (a temporary file, then mv); status 1 when it could not be written
put () {
  { printf '%s\n' "$2" > "$1.tmp.$$" && mv -f "$1.tmp.$$" "$1"; } 2>/dev/null && return 0
  rm -f "$1.tmp.$$" 2>/dev/null
  say "WARN: cannot write $1"
  return 1
}
put "$STATUS/node$NODE.pid" "$$ $HOST ${SLURM_STEP_ID:-${SLURM_STEPID:--}}"
# the batch host (train_arm.sh's): its compile caches and scratch belong to job.sh
if [ -n "${DM_BATCH_HOST:-}" ]; then [ "$HOST" = "$DM_BATCH_HOST" ] && BATCH=1 || BATCH=0
else [ "$NODE" = 0 ] && BATCH=1 || BATCH=0; fi
JIT_MADE=""
cleanup () {
  [ "$BATCH" = 1 ] && return 0
  [ -n "$JIT_MADE" ] && rm -rf "$JIT_MADE" 2>/dev/null
  if [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] && [ "${TMPDIR:-}" = "/tmp/dm-$SLURM_JOB_ID" ]; then rm -rf "$TMPDIR" 2>/dev/null; fi
  return 0
}
trap cleanup EXIT
# a stop before the ranks start: none start, and the stop file tells the launchers that have not started theirs yet
# (a stop on the batch host's launcher once train_arm.sh knew its pid leaves no stop file otherwise); the others' ranks,
# if any started, are stopped by their own launchers or by train_arm.sh's kill of the step
stop_before_start () {
  { [ "$NSIG" -gt 0 ] || [ -e "$STATUS/stop" ]; } || return 0
  [ -e "$STATUS/stop" ] || { : > "$STATUS/stop"; } 2>/dev/null
  say "a stop came before the ranks started: none started (status 143)"
  put "$STATUS/node$NODE" "143 $HOST -"
  exit 0
}
stop_before_start
# bg_run <command...>: run it in the background and wait for it, polling, so that a stop signal is handled at once (and
# ends the command); its exit status
bg_run () {
  "$@" &
  local p=$! tk
  while kill -0 "$p" 2>/dev/null; do
    [ "$NSIG" -gt 0 ] && kill -TERM "$p" 2>/dev/null
    sleep 0.2 & tk=$!
    wait "$tk"; [ $? -gt 128 ] && kill "$tk" 2>/dev/null
  done
  wait "$p"
}
if [ -n "${TMPDIR:-}" ] && [ ! -d "$TMPDIR" ]; then
  mkdir -p "$TMPDIR" 2>/dev/null || say "WARN: cannot create TMPDIR=$TMPDIR"
fi
J=${DM_JIT_CACHE_DIR:-}
if [ -n "$J" ] && [[ $J == /* ]] && [ ! -e "$J" ]; then
  # mkdir (not -p) for the last level: the launcher that created it is the one that fills it and removes it
  if mkdir -p "$(dirname "$J")" 2>/dev/null && mkdir "$J" 2>/dev/null; then
    JIT_MADE=$J
    if [ -n "${DM_CHAIN_DIR:-}" ] && [ -f "$DM_CHAIN_DIR/jit.tar" ]; then
      t=$SECONDS
      if bg_run timeout 300 tar -xf "$DM_CHAIN_DIR/jit.tar" -C "$J"; then
        say "compile caches restored from $DM_CHAIN_DIR/jit.tar into $J ($((SECONDS - t)) s)"
      else
        if [ "$NSIG" -gt 0 ]; then say "restoring the compile caches from $DM_CHAIN_DIR/jit.tar was cut short by the stop"
        else say "WARN: restoring the compile caches from $DM_CHAIN_DIR/jit.tar into $J failed: the ranks start with empty caches"; fi
        rm -rf "$J" 2>/dev/null; mkdir -p "$J" 2>/dev/null
      fi
    fi
  else
    say "WARN: cannot create the compile-cache directory $J: the ranks start with empty caches"
  fi
fi
stop_before_start
# the per-rank compile caches (header): a level per node for a root other nodes may see, except on the batch host
node_local () { case $1 in /tmp|/tmp/*|/dev/shm|/dev/shm/*) return 0 ;; esac; [ -n "$JIT_MADE" ] && [[ $1 == "$JIT_MADE"/* ]]; }
if [ "$BATCH" = 0 ]; then
  TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$W/triton_cache} TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-$W/tilelang_cache}
  for v in TRITON_CACHE_DIR TILELANG_CACHE_DIR; do
    node_local "${!v}" && continue
    say "$v=${!v} may be shared with the other nodes: this node's ranks use ${!v}/n$NODE/r<local rank>"
    printf -v "$v" '%s/n%s' "${!v}" "$NODE"
  done
fi
RANK_BASE=$((NODE * DM_NPROC))   # the collector names global ranks
start_ranks "$NODE" "$PY" "$@"
# a stop that came while the ranks were being started reached only those started before it (NFWD of them)
if [ "$NSIG" -gt 0 ]; then for p in "${PIDS[@]:NFWD}"; do kill -TERM "$p" 2>/dev/null; done; fi
say "started global ranks $((NODE * DM_NPROC))-$((NODE * DM_NPROC + DM_NPROC - 1)) of $WORLD_SIZE (master ${MASTER_ADDR:-?}:${MASTER_PORT:-?})"
wait_ranks
if [ "$RC" -ne 0 ] && [ "$RC" -ne 75 ]; then failed_tails "$NODE"; fi
FAILED=()
for r in "${!RANK_RC[@]}"; do
  case ${RANK_RC[r]} in 0|75|killed|unkillable) ;; *) FAILED+=($((NODE * DM_NPROC + r))) ;; esac
done
put "$STATUS/node$NODE" "$RC $HOST ${T_FAIL:--}${FAILED[*]:+ ${FAILED[*]}}"
if [ -n "$T_FAIL" ]; then say "ranks done: status $RC (first failure at $(date -d "@$T_FAIL" +%T): global rank(s) ${FAILED[*]})"
else say "ranks done: status $RC"; fi
case $RC in 0|75) exit 0 ;; *) exit "$RC" ;; esac
