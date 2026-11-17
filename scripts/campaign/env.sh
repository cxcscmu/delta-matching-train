#!/bin/bash
# scripts/campaign/env.sh: the environment contract shared by train_arm.sh, stage2_arm.sh, eval_arm.sh, eval_arm_64k.sh
# and the experiment drivers (scripts/experiments/). Source it; every variable has a default relative to the repository
# and can be overridden from the shell. See docs/SETUP.md section 5.
# It also defines the helpers of the launchers (the chain contract of scripts/slurm/: a chain is a series of jobs, each
# stopping with a checkpoint and handing over to the next):
#   dm_progress <kind> <run> [detail]  appends "<epoch> <SLURM_JOB_ID or 0> <kind> <run> [detail]" to $DM_PROGRESS_FILE
#                                      (set by scripts/slurm/job.sh in a chain; unset: nothing). Never fails.
#   dm_ckpt_step <ckpt dir>            the highest N with <ckpt dir>/checkpoint-N/.complete, else -1 (any marker: the
#                                      trainer also requires its content to be the checkpoint format's VERSION_STRING)
#   dm_eval_markers <eval dir>         the number of *.complete and *COMPLETE files under <eval dir> (0 when absent)
#   dm_lock <fd> <lock file> <what>    the run lock: opens <lock file> on fd <fd> and takes an flock on it. The processes
#                                      this shell starts itself inherit the fd (the env launch's ranks, the val CE), so
#                                      the lock is held while any of them runs; the kernel releases it on any exit,
#                                      SIGKILL included. torchrun's workers do NOT inherit it (torch starts them with
#                                      close_fds and in sessions of their own): under torchrun only this shell and
#                                      torchrun hold it.
#                                      When another process holds it, <lock file>.owner ("job=<id> host=<h> pid=<p>
#                                      time=<t>") names the holder, judged again every 10 s (squeue -j <id> -o %T):
#                                      - another live job (RUNNING, CONFIGURING, SUSPENDED, STOPPED, RESIZING), or a
#                                        launch outside Slurm that still runs (on this host: its pid lives; on another
#                                        host: assumed): "FATAL: <what> by <owner>", status 3;
#                                      - this job (requeued, or retried in place by scripts/slurm/job.sh), a job that is
#                                        ending (COMPLETING, SIGNALING, STAGE_OUT: Slurm starts a chain's next job while
#                                        the previous one's ranks may still be saving) or has ended, a launch of this
#                                        host whose pid is gone, or a holder nobody can tell about (squeue gives no
#                                        answer; no owner file yet): it waits for the lock, DM_LOCK_WAIT seconds in all
#                                        (default 600, more than Slurm's KillWait of 120 s). Still held then: status 1,
#                                        a failure to retry (a process stuck on its node, or a lock the file system has
#                                        not released yet), not a run owned by another job.
#                                      Status 0 when the lock is held (the owner file is then rewritten). Without a
#                                      usable flock (no command, a lock file that cannot be opened, a file system
#                                      without flock) it WARNs and returns 0: the run goes on without a lock. A signal
#                                      that kills the wait: 128 + its number. A caller that traps the stop signals
#                                      itself defines dm_lock_stop (true once a stop was requested): asked before every
#                                      look at the holder, it ends the wait within 10 s of the signal (status 143).
DM_REPO=${DM_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
DM_WORKSPACE=${DM_WORKSPACE:-$DM_REPO/workspace}                  # fast local scratch (a symlink or directory in the repo root)
DM_DATA_ROOT=${DM_DATA_ROOT:-$DM_WORKSPACE/data}                     # the data root built by data/ (data/README.md)
# The interpreter: an explicit DM_PY wins, then the repo's own uv environment (docs/SETUP.md section 2), then PATH.
# The venv default matters on a batch node, where `python3` may be a system interpreter without torch.
if [ -z "${DM_PY:-}" ]; then
  if [ -x "$DM_REPO/.venv/bin/python" ]; then DM_PY=$DM_REPO/.venv/bin/python
  else DM_PY=$(command -v python || command -v python3 || true); fi
fi
DM_TORCHRUN=${DM_TORCHRUN:-${DM_PY:+$(dirname "$DM_PY")/torchrun}}
DM_NPROC=${DM_NPROC:-8}                                            # GPUs per node used by torchrun
DM_TOKENIZER_DIR=${DM_TOKENIZER_DIR:-$DM_DATA_ROOT/tokenizer/llama2_32k}
HF_HOME=${HF_HOME:-$DM_DATA_ROOT/hf_cache}                          # the evaluation datasets (data/prepare_eval.py fills it)
export HF_HOME
export PYTHONPATH=$DM_REPO/src${PYTHONPATH:+:$PYTHONPATH}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$DM_WORKSPACE/triton_cache} TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-$DM_WORKSPACE/tilelang_cache}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# The fp8 attention kernel comes from the installed `delta_matching` package (the bundled kernels/delta-matching, built by
# scripts/install_kernel.sh; docs/SETUP.md section 3); arms.sh checks its build flags at launch. No binary paths are pinned here.
# wandb: the entity stays unset (the account's default) unless WANDB_ENTITY is exported; project delta-matching; DM_WANDB=0 disables
export WANDB_PROJECT=${WANDB_PROJECT:-delta-matching} WANDB_DIR=${WANDB_DIR:-$DM_WORKSPACE}
export DM_REPO DM_WORKSPACE DM_DATA_ROOT DM_PY DM_TORCHRUN DM_NPROC DM_TOKENIZER_DIR

dm_progress () {   # dm_progress <kind> <run> [detail]: one durable-progress line (O_APPEND); builtins only, never fails
  [ -n "${DM_PROGRESS_FILE:-}" ] || return 0
  local t IFS=' '; printf -v t '%(%s)T' -1
  { printf '%s %s %s\n' "$t" "${SLURM_JOB_ID:-0}" "$*" >> "$DM_PROGRESS_FILE"; } 2>/dev/null
  return 0
}
dm_ckpt_step () {   # dm_ckpt_step <ckpt dir>: the newest complete checkpoint's step, -1 when there is none
  local d n best=-1
  for d in "$1"/checkpoint-*; do
    n=${d##*/checkpoint-}
    case $n in ''|*[!0-9]*) continue ;; esac
    [ -e "$d/.complete" ] || continue
    [ $((10#$n)) -gt "$best" ] && best=$((10#$n))
  done
  echo "$best"
}
dm_eval_markers () {   # dm_eval_markers <eval dir>: how many completion markers the evaluations left there
  local n
  [ -d "$1" ] || { echo 0; return 0; }
  n=$(find "$1" -type f \( -name '*.complete' -o -name '*COMPLETE' \) 2>/dev/null | wc -l)
  echo $((n + 0))
}
_dm_lock_judge () {   # _dm_lock_judge <owner line>: the holder it names: alive | self | ending | ended | unknown (header)
  local job="" host="" pid="" out rc
  [[ $1 =~ (^|[[:space:]])job=([^[:space:]]*) ]] && job=${BASH_REMATCH[2]}
  [[ $1 =~ (^|[[:space:]])host=([^[:space:]]*) ]] && host=${BASH_REMATCH[2]}
  [[ $1 =~ (^|[[:space:]])pid=([^[:space:]]*) ]] && pid=${BASH_REMATCH[2]}
  if [[ $job =~ ^[0-9]+$ ]]; then
    [ "$job" = "${SLURM_JOB_ID:-}" ] && { echo self; return 0; }   # this job: an earlier instance (requeue) or attempt
    out=$(timeout 30 squeue -h -j "$job" -o %T 2>&1); rc=$?
    if [ "$rc" = 0 ]; then                                         # the job's state: alive while it may still train
      case " ${out//$'\n'/ } " in
        *" RUNNING "*|*" CONFIGURING "*|*" SUSPENDED "*|*" STOPPED "*|*" RESIZING "*) echo alive ;;
        *" COMPLETING "*|*" SIGNALING "*|*" STAGE_OUT "*) echo ending ;;   # its processes are being ended
        *) echo ended ;;                                           # COMPLETED, FAILED, TIMEOUT, PENDING (requeued), ...
      esac
      return 0
    fi
    case $out in *"Invalid job id"*) echo ended ;; *) echo unknown ;; esac   # purged / squeue gave no answer
    return 0
  fi
  if [ -n "$host" ] && [ "$host" = "$(hostname -s)" ] && [[ $pid =~ ^[0-9]+$ ]]; then   # a launch outside Slurm, here
    if kill -0 "$pid" 2>/dev/null; then echo alive; else echo ended; fi
    return 0
  fi
  if [ -n "$host" ]; then echo alive; else echo unknown; fi   # another host's launch: assumed alive; no owner file yet
}
dm_lock () {   # dm_lock <fd> <lock file> <what>: the run lock (see the header); 0 = held, 3 = another live job holds it,
               # 1 = not released within DM_LOCK_WAIT by a holder that is ending, has ended or is this job
  local fd=$1 lock=$2 what=$3 rc owner judged why said="" wt=${DM_LOCK_WAIT:-600} t0 now left tmp
  case $fd in ''|*[!0-9]*) echo "dm_lock: bad file descriptor '$fd'" >&2; return 2 ;; esac
  case $wt in ''|*[!0-9]*) wt=600 ;; esac
  wt=$((10#$wt))
  if ! command -v flock >/dev/null 2>&1; then
    echo "[$(hostname -s) $(date +%T)] WARN: no flock command: $what without a run lock (nothing stops a second job)" >&2; return 0
  fi
  if ! eval "exec $fd>>\"\$lock\"" 2>/dev/null; then
    echo "[$(hostname -s) $(date +%T)] WARN: cannot open the run lock $lock: $what without a run lock" >&2; return 0
  fi
  flock -n "$fd"; rc=$?
  if [ "$rc" = 1 ]; then                                           # held by another process: judge its holder every 10 s
    printf -v t0 '%(%s)T' -1
    while :; do
      if declare -F dm_lock_stop >/dev/null && dm_lock_stop; then   # the caller's own trap saw a stop signal
        echo "[$(hostname -s) $(date +%T)] FATAL: a stop signal ended the wait for the lock: $what (exit 143)" >&2; return 143
      fi
      owner=""; { read -r owner < "$lock.owner"; } 2>/dev/null
      owner=${owner:-an unknown holder (no $lock.owner yet)}
      judged=$(_dm_lock_judge "$owner")
      case $judged in
        alive) echo "[$(hostname -s) $(date +%T)] FATAL: $what by $owner: not starting a second one (exit 3)" >&2; return 3 ;;
        self) why="this job: an earlier instance or attempt of it is still ending" ;;
        ending) why="a job that is ending: its processes are being stopped" ;;
        ended) why="it has ended" ;;
        *) why="squeue cannot tell whether it still runs" ;;
      esac
      printf -v now '%(%s)T' -1; left=$((t0 + wt - now))
      if [ -z "$said" ]; then                                      # said once per holder and judgment
        echo "[$(hostname -s) $(date +%T)] $what: its lock is held by $owner ($why): waiting up to $wt s (DM_LOCK_WAIT) for it"
      elif [ "$said" != "$owner $judged" ]; then
        echo "[$(hostname -s) $(date +%T)] $what: its lock's holder now: $owner ($why): still waiting, $left s left (DM_LOCK_WAIT)"
      fi
      said="$owner $judged"
      if [ "$left" -le 0 ]; then
        echo "[$(hostname -s) $(date +%T)] FATAL: $what by $owner ($why): its lock was not released within $wt s (DM_LOCK_WAIT): exit 1, a failure to retry (a process stuck on its node, or a lock the file system has not released yet)" >&2
        return 1
      fi
      [ "$left" -gt 10 ] && left=10
      flock -w "$left" "$fd"; rc=$?
      [ "$rc" = 1 ] || break                                       # 0: taken; anything else: below
    done
  fi
  if [ "$rc" != 0 ]; then
    if [ "$rc" -gt 128 ]; then echo "[$(hostname -s) $(date +%T)] FATAL: a signal ended the wait for the lock: $what (exit $rc)" >&2; return "$rc"; fi
    echo "[$(hostname -s) $(date +%T)] WARN: flock failed (exit $rc) on $lock: $what without a run lock" >&2; return 0
  fi
  tmp=$lock.owner.tmp.$$
  { printf 'job=%s host=%s pid=%s time=%s\n' "${SLURM_JOB_ID:-none}" "$(hostname -s)" "$$" "$(date '+%FT%T')" > "$tmp" \
      && mv -f "$tmp" "$lock.owner"; } 2>/dev/null || { rm -f "$tmp" 2>/dev/null; echo "WARN: could not write $lock.owner" >&2; }
  return 0
}
