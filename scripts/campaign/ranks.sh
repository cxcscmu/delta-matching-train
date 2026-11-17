#!/bin/bash
# scripts/campaign/ranks.sh: the trainer ranks of one node, shared by the launchers that source it (not run by itself):
# scripts/campaign/train_arm.sh (a launch on one node: the env launch's ranks, or torchrun) and
# scripts/campaign/node_ranks.sh (one node of a multi-node launch, started there by srun). Sourcing it defines:
#   start_ranks <node> <python> <trainer args...>  the env launch: DM_NPROC ranks, global RANK = node * DM_NPROC + r,
#       LOCAL_RANK = r, per-rank Triton / TileLang caches (${TRITON_CACHE_DIR}/r<r>, ${TILELANG_CACHE_DIR}/r<r>), each
#       appending its output to $W/logs/<RN>.rank<RANK>.log after one header line per launch ("===== launch <date> job
#       <id> host <host> git <rev>: rank <RANK> of <WORLD_SIZE> of <RN> ====="). PIDS[r] = their pids, RLOG0[r] = where
#       this launch's part of rank r's log begins. Uses W, RN, DM_NPROC, WORLD_SIZE, DM_GIT_REV, DM_HOST.
#   failed_tails <node>  the last 30 lines each failed rank of this node (RANK_RC: not 0, 75 or killed) wrote in this launch.
#   the signal-forwarding block below (forward_term, wait_ranks, wait_torchrun; tests/test_launcher_signals.py runs it
#       on its own).
# SIGTERM handling: the trainer saves a complete checkpoint at the next step and exits 75. Under `env` launch the ranks
# are the launcher's children and get the signal directly; under torchrun the signal is forwarded to torchrun's children
# (the ranks), never to torchrun itself (it would kill the ranks after 30 s, before a 20 GB save can finish).

start_ranks () {   # start_ranks <node> <python> <trainer args...> (header)
  local node=$1 py=$2 r g log hdr
  shift 2
  RLOG0=(); PIDS=()
  hdr="$(date '+%F %T') job ${SLURM_JOB_ID:-none} host $DM_HOST git $DM_GIT_REV"
  for r in $(seq 0 $((DM_NPROC - 1))); do
    g=$((node * DM_NPROC + r)); log=$W/logs/${RN}.rank${g}.log
    RLOG0[r]=$(stat -c %s "$log" 2>/dev/null) || RLOG0[r]=0
    echo "===== launch $hdr: rank $g of $WORLD_SIZE of $RN =====" >> "$log"
    # Per-rank Triton/TileLang caches. With one shared cache on a network FS the 8 ranks autotune and
    # compile the same kernel concurrently and one rank reads a metadata file another is still writing
    # (triton/runtime/autotuner.py check_disk_cache -> FileNotFoundError); that rank dies and the other
    # seven deadlock in NCCL. Seen on the GDN kernel prepare_wy_repr_bwd_kernel.
    RANK=$g LOCAL_RANK=$r \
      TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$W/triton_cache}/r$r" \
      TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$W/tilelang_cache}/r$r" \
      "$py" "$@" >> "$log" 2>&1 & PIDS+=($!)
  done
}

failed_tails () {   # failed_tails <node> (header)
  local r g
  for r in "${!RANK_RC[@]}"; do
    case ${RANK_RC[r]} in 0|75|killed) continue ;; esac
    g=$(($1 * DM_NPROC + r))
    echo "--- rank $g failed (exit ${RANK_RC[r]}): the last 30 lines it wrote in this launch ($W/logs/${RN}.rank${g}.log) ---"
    tail -c +$(( ${RLOG0[r]:-0} + 1 )) "$W/logs/${RN}.rank${g}.log" 2>/dev/null | tail -n 30
  done
}

# >>> signal forwarding (tests/test_launcher_signals.py runs this block on its own, so keep it self-contained)
# The handler runs builtins only: no $(...) and no external command. A job scheduler may deliver SIGTERM twice within
# milliseconds (once to every process of the job, as job.sh passes Slurm's on, then again through a wrapper's own
# trap); under bash 5.2 a handler
# caught inside a command substitution by the second signal re-enters the parser and aborts the shell ("bad jump",
# exit 134), which would tear the ranks down with it.
# The ranks are signalled first and the message printed after; a repeated signal is only counted. T_FWD = when the
# first one was forwarded (printf's %(...)T is a builtin); NFWD = how many of PIDS it reached (a launcher that is still
# starting its ranks signals the later ones itself: node_ranks.sh); GONE[r] = 1 for a rank that had already exited when
# the signal came (`kill -0` fails once bash has reaped it, which it does as soon as the rank exits): wait_ranks
# collects ranks once a second, so a rank that failed just before the stop is often collected just after it.
# Under torchrun (TR_PID set) the ranks are torchrun's children, read from /proc when the signal arrives (builtins
# only), so ranks spawned after the launch are reached too. Only when torchrun has no child yet is torchrun itself
# signalled: it then stops before any rank has trained.
DM_HOST=$(hostname -s); PIDS=(); NSIG=0; TR_PID=""; T_FWD=""; RANK_RC=(); NFWD=0; T_FAIL=""; RANK_BASE=0; GONE=()
forward_term () {
  NSIG=$((NSIG + 1)); [ "$NSIG" -gt 1 ] && return 0
  printf -v T_FWD '%(%s)T' -1
  local p f k i who="trainer ranks"; local -a T=("${PIDS[@]}")
  NFWD=${#PIDS[@]}
  for i in "${!PIDS[@]}"; do kill -0 "${PIDS[i]}" 2>/dev/null || GONE[i]=1; done
  if [ -n "$TR_PID" ]; then
    T=()
    for f in /proc/"$TR_PID"/task/*/children; do k=(); [ -r "$f" ] && read -r -a k < "$f"; T+=(${k[@]+"${k[@]}"}); done
    [ ${#T[@]} -gt 0 ] || { T=("$TR_PID"); who="torchrun (no rank started yet)"; }
  fi
  [ ${#T[@]} -gt 0 ] || who="no rank (none started yet)"
  for p in "${T[@]}"; do kill -TERM "$p" 2>/dev/null; done
  printf '[%s %(%T)T] SIGTERM -> %s\n' "$DM_HOST" -1 "$who"
}
# wait_ranks: every rank's exit status, collected as it exits -> RC, and RANK_RC[r] = rank r's status ("killed" when
# this script SIGKILLed it, "unkillable" when it outlived that). RC: 75 when any rank exited 75 (the trainer saved a
# complete checkpoint), otherwise the first failure (a status other than 0 and 75; the 137 of a rank this script killed
# never counts), otherwise 143 when ranks had to be killed (after a stop signal, or after the others had exited).
# A failure "before the stop": one seen while no stop signal had been forwarded, or one of a rank that had already
# exited when the stop came (GONE), unless that rank died of a stop signal itself (128 + HUP, INT, USR1, USR2 or TERM:
# the stop can reach the ranks at the same moment, e.g. job.sh's `kill -TERM 0` reaches train_arm.sh's ranks on one
# node, so a rank without a handler yet can die of it before this launcher passes it on). T_FAIL = when the first
# failure before the stop happened (epoch seconds: when it was seen, or for a rank already gone, the stop's T_FWD at the
# latest; empty: none); node_ranks.sh reports it, and train_arm.sh blames the node that failed first. A failure after
# the stop gets no time (srun's kill-on-bad-exit and the time limit stop the ranks of healthy nodes too). The messages
# name rank RANK_BASE + r (node_ranks.sh: the global rank; one node: 0 + r).
#   * The first rank that fails before the stop: the others are most likely blocked in a collective that only the
#     process-group timeout (30 min) would end. They get DM_RANK_GRACE seconds (default 60) to exit on their own, then
#     SIGKILL.
#   * Under Slurm (SLURM_JOB_ID set) the ranks get DM_TERM_GRACE seconds (default 90, below Slurm's KillWait of 120 s)
#     after a forwarded stop signal to save and exit, then SIGKILL: this script, the driver and job.sh can then still
#     exit before Slurm kills the whole job (a save cut short leaves no .complete marker; the previous checkpoint holds).
#   * The first rank that exits cleanly (0 or 75): the others get DM_EXIT_GRACE seconds (default 600) to follow, then
#     SIGKILL. A clean exit comes after the run's last collective (the final export and every checkpoint end with a
#     barrier), so a rank still running that long after is stuck in its teardown (the process group's destruction).
#   * A rank still alive 30 s after its SIGKILL (stuck in the kernel) is left behind.
# Polling, once a second: `kill -0` fails once bash has reaped an exited rank, and `wait <pid>` then returns its stored
# status (the previous `while kill -0 "$p"; do wait "$p"; done` per rank never collected a rank that had exited before
# its turn). This works on every bash >= 4.4. `wait -n` is deliberately not used: bash moves a child that exits while
# the shell is not inside `wait -n` out of its job table, and the next `wait -n` then blocks on the live ranks (seen on
# bash 5.1.16), which is exactly the hang this collector must end. The pause is a background sleep waited for with
# `wait`, which a trapped signal interrupts at once, so the handler is never delayed.
wait_ranks () {
  local i p s now tk why before bnote t_fail="" t_term="" t_exit="" t_kill="" fr="" fs="" er="" es=""
  local g_rank=${DM_RANK_GRACE:-60} g_term=${DM_TERM_GRACE:-90} g_exit=${DM_EXIT_GRACE:-600}
  local -a live=(); local -A ours=()
  case $g_rank in ''|*[!0-9]*) g_rank=60 ;; esac
  case $g_term in ''|*[!0-9]*) g_term=90 ;; esac
  case $g_exit in ''|*[!0-9]*) g_exit=600 ;; esac
  RC=0; RANK_RC=(); T_FAIL=""
  for i in "${!PIDS[@]}"; do live[i]=${PIDS[i]}; done
  while :; do
    for i in "${!live[@]}"; do
      p=${live[i]}
      kill -0 "$p" 2>/dev/null && continue
      wait "$p"; s=$?; unset "live[i]"
      if [ -n "${ours[$p]:-}" ] && [ "$s" -eq 137 ]; then RANK_RC[i]=killed; continue; fi
      RANK_RC[i]=$s
      case $s in
        0|75) [ "$s" -eq 75 ] && RC=75
              if [ -z "$t_exit" ]; then printf -v now '%(%s)T' -1; t_exit=$((now + g_exit)); er=$((RANK_BASE + i)); es=$s; fi ;;
        *) [ "$RC" -eq 0 ] && RC=$s
           before=0 bnote=""   # a failure before the stop (above); bnote: said of a rank already gone when it came
           if [ "$NSIG" -eq 0 ]; then before=1
           elif [ -n "${GONE[i]:-}" ]; then
             case $s in 129|130|138|140|143) ;; *) before=1 bnote=" before the stop signal" ;; esac
           fi
           if [ -z "$t_fail" ] && [ "$before" = 1 ]; then
             printf -v now '%(%s)T' -1; T_FAIL=$now; [ "$NSIG" -gt 0 ] && T_FAIL=${T_FWD:-$now}
             t_fail=$((T_FAIL + g_rank)); fr=$((RANK_BASE + i)); fs=$s
             printf '[%s %(%T)T] rank %d failed (exit %d)%s: the other ranks get %d s (DM_RANK_GRACE) to exit\n' "$DM_HOST" -1 "$((RANK_BASE + i))" "$s" "$bnote" "$g_rank"
           fi ;;
      esac
    done
    [ ${#live[@]} -eq 0 ] && break
    printf -v now '%(%s)T' -1
    [ -z "$t_term" ] && [ "$NSIG" -gt 0 ] && [ -n "${SLURM_JOB_ID:-}" ] && t_term=$(( ${T_FWD:-$now} + g_term ))
    if [ -z "$t_kill" ]; then
      why=""
      if [ -n "$t_fail" ] && [ "$now" -ge "$t_fail" ]; then why="$g_rank s after rank $fr failed (exit $fs)"
      elif [ -n "$t_term" ] && [ "$now" -ge "$t_term" ]; then why="$g_term s after the forwarded SIGTERM"
      elif [ -n "$t_exit" ] && [ "$now" -ge "$t_exit" ]; then why="$g_exit s after rank $er exited $es (DM_EXIT_GRACE: stuck in its teardown?)"; fi
      if [ -n "$why" ]; then
        printf '[%s %(%T)T] %d rank(s) still running %s: SIGKILL\n' "$DM_HOST" -1 "${#live[@]}" "$why"
        for i in "${!live[@]}"; do ours[${live[i]}]=1; kill -KILL "${live[i]}" 2>/dev/null; done
        t_kill=$now
      fi
    elif [ "$now" -ge $((t_kill + 30)) ]; then
      printf '[%s %(%T)T] %d rank(s) still alive 30 s after their SIGKILL (stuck in the kernel?): not waiting for them\n' "$DM_HOST" -1 "${#live[@]}"
      for i in "${!live[@]}"; do RANK_RC[i]=unkillable; done
      break
    fi
    sleep 1 & tk=$!
    wait "$tk"; [ $? -gt 128 ] && kill "$tk" 2>/dev/null
  done
  [ "$RC" -eq 0 ] && [ ${#ours[@]} -gt 0 ] && RC=143
  return 0
}
# wait_torchrun: torchrun's exit status -> RC. The previous `sleep 5; while kill -0 ...; do wait ...; done` reported 0
# for a torchrun that had exited during the sleep, and a SIGTERM during the sleep killed this shell. `wait` is repeated
# only when a trapped signal interrupted it (it then returns 128+N early).
wait_torchrun () {
  local s n
  while :; do
    n=$NSIG; wait "$TR_PID"; s=$?
    [ "$NSIG" != "$n" ] && [ "$s" -gt 128 ] && continue
    break
  done
  RC=$s
}
# <<< signal forwarding
