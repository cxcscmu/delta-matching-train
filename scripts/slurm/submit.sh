#!/bin/bash
# scripts/slurm/submit.sh: run a command of this repository as a chain of Slurm jobs ("links") that hand over to each
# other until the command is done (scripts/slurm/job.sh). One node per job unless -N/--nodes says more (every link
# replays it; scripts/campaign/train_arm.sh then trains on all of them).
#   scripts/slurm/submit.sh [sbatch options] -- <command> [args...]
# examples (scripts/slurm/README.md):
#   scripts/slurm/submit.sh -A <account> -p batch -t 4:00:00 -- scripts/campaign/train_arm.sh dm_matched_hd128 11785
#   scripts/slurm/submit.sh -p batch -t 4:00:00 -- scripts/experiments/stage1_hd128.sh dm_matched_hd128 dm_stale_hd128
#   SMOKE=1 DM_RESUBMIT=0 scripts/slurm/submit.sh -p batch_short -t 1:00:00 -- scripts/experiments/stage1_hd128.sh clean_hd128
# The command runs from the repository root: a path is taken relative to it (or absolute) and must be an executable
# file, a bare name must be on the job's PATH.
# The chain is every job of yours with its name: dm-<command>-<first argument>-<h8>, h8 hashing the command (as the job
# runs it: a path made physical and relative to the repository), its arguments, the workspace (physical path) and the
# knobs that define the work (NAME_KNOBS below): each arm / seed / knob setting is its own chain, the same command with
# the same knobs the same chain, however it is spelled. -J names it instead. Stop a whole chain with
# scancel -u $USER -n <name>. Jobs of the same command and first argument under another name (an older submit.sh's
# dm-<command>-<first argument>, other knobs or spellings) are named in a WARN.
# Defaults (an option given here, in any spelling sbatch accepts, or its SBATCH_* variable wins): one node; without a
# GPU option, 8 GPUs per node with the whole nodes' CPUs and memory (--exclusive --mem=0); the log at
# $DM_WORKSPACE/logs/slurm/%x-%j.log; DM_SLURM_ACCOUNT / DM_SLURM_PARTITION fill -A / -p. Every link replays these
# options plus --dependency=singleton (Slurm runs one job of a name at a time). -d/--dependency (merged with singleton),
# -b/--begin and -H/--hold apply to the first job only. Refused (exit 2): --array, --wait, --signal (job.sh handles the
# signals), --export other than ALL, --wrap, --test-only, an ambiguous abbreviation, a word that is not an option before
# --, option values with spaces, a chain (DM_RESUBMIT=1, the default) without a time limit (-t or SBATCH_TIMELIMIT; the
# partitions' default is 31 min), a command that cannot be run, and DRY=1 (the experiment driver's preview exits 0,
# which would complete the chain; DM_SUBMIT_DRY=1 is the dry run here).
# The job sources DM_SITE_ENV (default $DM_WORKSPACE/site_env.sh when it exists), stops the trainer DM_STOP_MARGIN seconds
# (default 600) before the time limit and queues the next link (up to DM_MAX_RESUBMITS, default 400; DM_RESUBMIT=0:
# one job, no chain state). This shell's environment goes with the job (--export=ALL): SMOKE, SEEDS, EXTRA_ARGS, TAG,
# STOP_AT_STEP...; a new chain starts at link 0 (DM_RESUBMIT_COUNT=0) without the deadline, chain or derived variables
# (DM_DERIVED) of a shell inside a link.
# State: $DM_WORKSPACE/logs/slurm/chains/<name>/ (DM_CHAIN_DIR): a submission clears its failure records (links,
# runfails/, badnodes) and writes submit.sh there, which submits the chain again. Jobs of the chain already queued: the
# new submission runs after them and continues the chain (they adopt it as their successor and never cancel it); stop
# them first to replace them.
# DM_SUBMIT_DRY=1 prints the sbatch command and the chain; no change.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The env knobs that define what a chain computes, hashed into its name (with every S2_* variable; DM_WORKSPACE as its
# physical path, always; DM_DATA_ROOT when it is not the workspace's data). One list: the chain's submit.sh records them.
NAME_KNOBS=(SEEDS TAG SMOKE STOP_AT_STEP NUM_STEPS EXTRA_ARGS EVAL DM_LR DM_TRAIN_CONFIG DM_WORKSPACE DM_DATA_ROOT)
# The other knobs the chain's submit.sh records (how the chain runs, not what it computes).
RECIPE_KNOBS=(DM_SITE_ENV DM_MAX_RESUBMITS DM_STOP_MARGIN DM_SAVE_INTERVAL DM_WANDB WANDB_JT DM_EVAL_BUDGET DM_RESUBMIT
              DM_SLURM_ACCOUNT DM_SLURM_PARTITION DM_MAX_FAILS DM_INPLACE_RETRIES DM_RETRY_MIN_LEFT DM_RETRY_PAUSE
              DM_JIT_CACHE DM_JIT_MAX_MB DM_PRED_WAIT DM_RUN_MAX_FAILS DM_MIN_TRAIN_TIME)
die () { echo "FATAL: $*" >&2; exit 2; }
UOPTS=()   # the options as given
while [ $# -gt 0 ] && [ "$1" != "--" ]; do UOPTS+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { awk 'NR > 1 && !/^#/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"; exit 2; }

# sbatch's options (Slurm 24.05.5: the table of its getopt_long, listed by `sbatch --<letter> --version`): a long one may
# be abbreviated to any prefix that names no other; SB_FLAG take no value (or an optional one, as --name=value only);
# SB_SHORT maps the short ones, SB_SHORT_VAL those that take a value ("-x v" or "-xv"; flags may be grouped: -Hs).
SB_LONG=" account acctg-freq array autocomplete batch begin bb bbf chdir cluster cluster-constraint clusters comment
  container container-id context contiguous constraint core-spec cores-per-socket cpu-freq cpus-per-gpu cpus-per-task
  deadline delay-boot dependency distribution error exclude exclusive export export-file extra extra-node-info
  get-user-env gid gpu-bind gpu-freq gpus gpus-per-node gpus-per-socket gpus-per-task gres gres-flags help hint hold
  ignore-pbs input job-name kill-on-invalid-dep licenses mail-type mail-user mcs-label mem mem-bind mem-per-cpu
  mem-per-gpu mincpus network nice no-kill no-requeue nodefile nodelist nodes ntasks ntasks-per-core ntasks-per-gpu
  ntasks-per-node ntasks-per-socket ntasks-per-tres open-mode output overcommit oversubscribe parsable partition prefer
  priority profile propagate qos quiet reboot requeue reservation resv-ports segment signal sockets-per-node spread-job
  stepmgr switches tasks-per-node test-only thread-spec threads-per-core time time-min tmp tres-bind tres-per-task uid
  use-min-nodes usage verbose version wait wait-all-nodes wckey wrap "
SB_FLAG=" contiguous exclusive get-user-env help hold ignore-pbs nice no-kill no-requeue overcommit oversubscribe parsable
  propagate quiet reboot requeue resv-ports spread-job stepmgr test-only usage use-min-nodes verbose version wait "
SB_LONG=$(echo $SB_LONG) SB_FLAG=$(echo $SB_FLAG); SB_LONG=" $SB_LONG " SB_FLAG=" $SB_FLAG "   # single spaces
declare -A SB_SHORT=([a]=array [A]=account [b]=begin [B]=extra-node-info [c]=cpus-per-task [C]=constraint
  [d]=dependency [D]=chdir [e]=error [F]=nodefile [G]=gpus [h]=help [H]=hold [i]=input [J]=job-name [k]=no-kill
  [L]=licenses [m]=distribution [M]=clusters [n]=ntasks [N]=nodes [o]=output [O]=overcommit [p]=partition [q]=qos
  [Q]=quiet [s]=oversubscribe [S]=core-spec [t]=time [v]=verbose [V]=version [w]=nodelist [W]=wait [x]=exclude)
SB_SHORT_VAL=aAbBcCdDeFGiJLmMnNopqStwx
# sbatch's input variables that stand for an option (they count as given; SBATCH_ARRAY_INX etc. are refused below)
declare -A SB_ENV=([account]=SBATCH_ACCOUNT [partition]=SBATCH_PARTITION [output]=SBATCH_OUTPUT [time]=SBATCH_TIMELIMIT
  [gpus]=SBATCH_GPUS [gpus-per-node]=SBATCH_GPUS_PER_NODE [gpus-per-task]=SBATCH_GPUS_PER_TASK [gres]=SBATCH_GRES
  [tres-per-task]=SBATCH_TRES_PER_TASK [mem]=SBATCH_MEM_PER_NODE [mem-per-cpu]=SBATCH_MEM_PER_CPU
  [mem-per-gpu]=SBATCH_MEM_PER_GPU [exclusive]=SBATCH_EXCLUSIVE)
# long_name <--word[=value]>: LN = the long option it names (exactly, or the only one it abbreviates); status 1 when it
# names none here (an option of another sbatch version: passed on as it is), 2 when it abbreviates several (LN: them)
long_name () {
  local w=${1#--} n hits=()
  w=${w%%=*}; LN=""
  [ -n "$w" ] || return 1
  case "$SB_LONG" in *" $w "*) LN=$w; return 0 ;; esac
  for n in $SB_LONG; do [ "${n#"$w"}" != "$n" ] && hits+=("$n"); done
  [ "${#hits[@]}" -eq 1 ] && { LN=${hits[0]}; return 0; }
  [ "${#hits[@]}" -eq 0 ] && return 1
  LN="${hits[*]}"; return 2
}
OPTS=()      # replayed by every link
REPLAY=()    # the given options minus the first-job-only ones (the chain's submit.sh repeats them)
GIVEN=()     # the canonical names of the given options
DEPS=() FIRST_ONLY=() FIRST_GIVEN=() JNAME=""
# opt <name> <value> <word>...: one option of the user's (its long name, its value, its words as given)
opt () {
  local n=$1 v=$2; shift 2
  GIVEN+=("$n")
  case "$n" in
    dependency) [ -n "$v" ] || die "$1 needs a value"
                DEPS+=("$v"); FIRST_GIVEN+=("$@"); return 0 ;;
    begin) FIRST_ONLY+=("--begin=$v"); FIRST_GIVEN+=("$@"); return 0 ;;
    hold) FIRST_ONLY+=(--hold); FIRST_GIVEN+=("$@"); return 0 ;;
    array) die "--array cannot be used: a chain replays its sbatch options in every job" ;;
    wait) die "--wait cannot be used: a chain replays its sbatch options in every job" ;;
    signal) die "--signal cannot be used: job.sh handles the signals (the trainer's deadline is DM_STOP_MARGIN before the time limit)" ;;
    export) [ "${v^^}" = ALL ] || die "--export=$v cannot be used: a chain passes its environment on (--export=ALL)" ;;
    wrap) die "--wrap cannot be used: the job's script is job.sh (the command goes after --)" ;;
    test-only) die "--test-only submits nothing (and no link could queue its successor): DM_SUBMIT_DRY=1 shows what would be submitted" ;;
    job-name) JNAME=$v ;;
  esac
  OPTS+=("$@"); REPLAY+=("$@")
}
i=0
while [ "$i" -lt "${#UOPTS[@]}" ]; do
  o=${UOPTS[$i]}; i=$((i + 1))
  case "$o" in
    --?*)
      rc=0; long_name "$o" || rc=$?
      [ "$rc" != 2 ] || die "sbatch option ${o%%=*} is ambiguous: it abbreviates $LN"
      if [ "$rc" = 1 ]; then GIVEN+=("?$o"); OPTS+=("$o"); REPLAY+=("$o"); continue; fi   # sbatch judges it
      if [[ $o == *=* ]]; then opt "$LN" "${o#*=}" "$o"
      elif [[ $SB_FLAG == *" $LN "* ]]; then opt "$LN" "" "$o"
      else
        [ "$i" -lt "${#UOPTS[@]}" ] || die "$o needs a value"
        v=${UOPTS[$i]}; i=$((i + 1)); opt "$LN" "$v" "$o" "$v"   # the next word, even one starting with '-'
      fi ;;
    -?*)
      j=1
      while [ "$j" -lt "${#o}" ]; do
        c=${o:$j:1}; j=$((j + 1)); n=${SB_SHORT[$c]:-}
        if [ -z "$n" ]; then w=-${o:$((j - 1))}; GIVEN+=("?$w"); OPTS+=("$w"); REPLAY+=("$w"); break; fi
        if [[ $SB_SHORT_VAL != *"$c"* ]]; then   # a flag (maybe grouped with others)
          if [ "${#o}" -eq 2 ]; then opt "$n" "" "$o"; else opt "$n" "" "-$c"; fi
          continue
        fi
        if [ "$j" -lt "${#o}" ]; then   # the value attached
          v=${o:$j}
          if [ "$j" -eq 2 ]; then opt "$n" "$v" "$o"; else opt "$n" "$v" "-$c$v"; fi
        else
          [ "$i" -lt "${#UOPTS[@]}" ] || die "-$c needs a value"
          v=${UOPTS[$i]}; i=$((i + 1))
          if [ "${#o}" -eq 2 ]; then opt "$n" "$v" "$o" "$v"; else opt "$n" "$v" "-$c" "$v"; fi
        fi
        break
      done ;;
    *) die "'$o' is not an sbatch option: give sbatch's options (an option's value with it) before --, the command and its arguments after it" ;;
  esac
done
# variables that would act as those options
[ -z "${SBATCH_ARRAY_INX:-}" ] || die "SBATCH_ARRAY_INX is set: a chain cannot be an array (unset it)"
[ -z "${SBATCH_WAIT:-}" ] || die "SBATCH_WAIT is set: a chain cannot wait (unset it)"
[ -z "${SBATCH_SIGNAL:-}" ] || die "SBATCH_SIGNAL is set: job.sh handles the signals (unset it)"
case "${SBATCH_EXPORT:-ALL}" in [Aa][Ll][Ll]) ;; *) die "SBATCH_EXPORT=$SBATCH_EXPORT: a chain passes its environment on (unset it)" ;; esac
case "${DRY:-0}" in ""|0) ;; *) die "DRY=$DRY is the experiment driver's plan preview: the job would print the plan and exit 0, which completes (and cancels) the chain; unset DRY (DM_SUBMIT_DRY=1 shows what would be submitted)" ;; esac
[ "${#DEPS[@]}" -le 1 ] || die "give one -d/--dependency (join its conditions with ','): ${DEPS[*]}"
DEP=""
if [ "${#DEPS[@]}" -eq 1 ]; then
  DEP=${DEPS[0]}
  case "$DEP" in *\?*) die "--dependency=$DEP: an OR dependency ('?') cannot be combined with the chain's singleton dependency" ;; esac
  case ",$DEP," in *,singleton,*) ;; *) DEP=$DEP,singleton ;; esac
fi
# knobs job.sh checks in every link: refused here already
for v in DM_STOP_MARGIN DM_MAX_RESUBMITS DM_MAX_FAILS DM_INPLACE_RETRIES DM_RETRY_MIN_LEFT DM_RETRY_PAUSE DM_JIT_MAX_MB \
         DM_PRED_WAIT; do
  case "${!v-}" in *[!0-9]*) die "$v='${!v}' is not a whole number" ;; esac
done
for v in DM_RESUBMIT DM_JIT_CACHE; do
  case "${!v-}" in ""|0|1) ;; *) die "$v='${!v}' is neither 0 nor 1" ;; esac
done
case "${DM_RESUBMIT_BACKOFF-}" in *[!0-9\ ]*) die "DM_RESUBMIT_BACKOFF='$DM_RESUBMIT_BACKOFF' is not a list of whole numbers (seconds)" ;; esac
# the command as job.sh will run it, from the repository root
case "$1" in
  */*) case "$1" in /*) CMDP=$1 ;; *) CMDP=$REPO/$1 ;; esac
       [ -f "$CMDP" ] && [ -x "$CMDP" ] || die "the command $1 is not an executable file as the job runs it, from the repository root: $CMDP (give a path relative to $REPO or an absolute one; a script without the x bit: bash <script>)" ;;
  *) command -v -- "$1" >/dev/null 2>&1 || echo "WARN: $1 is not on this shell's PATH (the job looks it up on the site file's PATH)" >&2 ;;
esac
# a new chain starts afresh, also from a shell inside a link: link 0, no deadline, end time or chain of that link, and
# none of the variables its job.sh derived for itself (rendezvous ports, DM_NPROC, compile-cache dirs: DM_DERIVED)
for v in ${DM_DERIVED:-}; do [[ $v =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] && unset "$v"; done
unset DM_STOP_BEFORE DM_JOB_END DM_CHAIN DM_PROGRESS_FILE DM_DERIVED
export DM_RESUBMIT_COUNT=0

W=${DM_WORKSPACE:-$REPO/workspace}
case "$W" in /*) ;; *) W=$PWD/$W ;; esac
if [ -z "${DM_SITE_ENV:-}" ] && [ -f "$W/site_env.sh" ]; then export DM_SITE_ENV=$W/site_env.sh; fi
case "${DM_SITE_ENV:-/}" in /*) ;; *) export DM_SITE_ENV=$PWD/$DM_SITE_ENV ;; esac
# The chain's name: -J, else dm-<command>-<first argument>-<h8>; h8 = sha1 of the command as the job runs it (canon_cmd)
# and its arguments (one per line), then K=V lines (sorted): the workspace's physical path, DM_DATA_ROOT's when it is not
# the workspace's data, and every other NAME_KNOBS / S2_* variable that is set and not empty. Physical paths: the repo's
# `workspace` link, a sourced site file or not, ./x or the absolute path name the same work.
WC=$(realpath -m -- "$W")
REPO_P=$(cd "$REPO" && pwd -P)
canon_cmd () {   # a path: physical, relative to the repository when inside it; a bare name (PATH): as it is
  local p
  case "$1" in
    */*) case "$1" in /*) p=$1 ;; *) p=$REPO/$1 ;; esac
         p=$(realpath -m -- "$p" 2>/dev/null) || p=$1
         case "$p" in "$REPO_P"/*) p=${p#"$REPO_P"/} ;; esac
         printf '%s\n' "$p" ;;
    *) printf '%s\n' "$1" ;;
  esac
}
name_knobs () {
  local k v
  {
    printf 'DM_WORKSPACE=%s\n' "$WC"
    if [ -n "${DM_DATA_ROOT:-}" ]; then
      case "$DM_DATA_ROOT" in /*) v=$DM_DATA_ROOT ;; *) v=$REPO/$DM_DATA_ROOT ;; esac   # env.sh's, from the repository root
      v=$(realpath -m -- "$v")
      [ "$v" = "$WC/data" ] || printf 'DM_DATA_ROOT=%s\n' "$v"
    fi
    for k in "${NAME_KNOBS[@]}" $(compgen -v S2_ || true); do
      case "$k" in DM_WORKSPACE|DM_DATA_ROOT) continue ;; esac
      if [ -n "${!k:-}" ]; then printf '%s=%s\n' "$k" "${!k}"; fi
    done
  } | LC_ALL=C sort
}
PREFIX=dm-$(basename -- "$1" .sh)${2:+-$2}
PREFIX=${PREFIX//[^A-Za-z0-9._-]/_}   # it names the log file (%x): no '/' or spaces
if [ -n "$JNAME" ]; then
  NAME=$JNAME
  [[ $NAME =~ ^[A-Za-z0-9._-]+$ ]] && [ "$NAME" != . ] && [ "$NAME" != .. ] \
    || die "job name '$NAME': a chain's name is made of letters, digits and ._- only (it names the log file and the chain's state directory, and squeue / scancel select the chain by it)"
else
  H8=$({ canon_cmd "$1"; [ $# -gt 1 ] && printf '%s\n' "${@:2}"; name_knobs; } | sha1sum | cut -c1-8)
  NAME=$PREFIX-$H8
fi
# given <name>...: whether any of these options was given (any spelling) or its SBATCH_* variable is set
given () {
  local n e
  for n in "$@"; do
    case " ${GIVEN[*]-} " in *" $n "*) return 0 ;; esac
    e=${SB_ENV[$n]:-}
    if [ -n "$e" ] && [ -n "${!e:-}" ]; then return 0; fi
  done
  return 1
}
given account || { [ -n "${DM_SLURM_ACCOUNT:-}" ] && OPTS+=("--account=$DM_SLURM_ACCOUNT"); }
given partition || { [ -n "${DM_SLURM_PARTITION:-}" ] && OPTS+=("--partition=$DM_SLURM_PARTITION"); }
given nodes || OPTS+=("--nodes=1")
given job-name || OPTS+=("--job-name=$NAME")
given output || OPTS+=("--output=$W/logs/slurm/%x-%j.log")
if ! given gpus gpus-per-node gpus-per-task gpus-per-socket gres tres-per-task; then   # the default: the whole node
  OPTS+=("--gpus-per-node=8")
  given mem mem-per-cpu mem-per-gpu || OPTS+=("--mem=0")
  given exclusive oversubscribe || OPTS+=("--exclusive")
fi
OPTS+=("--dependency=singleton")   # one job of the chain at a time, however it was queued
for o in "${OPTS[@]}"; do   # every option, the defaults included (DM_WORKSPACE, DM_SLURM_ACCOUNT, ...)
  case "$o" in *[[:space:]]*) die "sbatch option '$o' contains a space; pass it without one" ;; esac
done
if [ "${DM_RESUBMIT:-1}" = 1 ] && ! given time; then
  die "a chain needs a time limit (-t, e.g. -t 4:00:00, or SBATCH_TIMELIMIT): without one a job gets the partition's default (31 min here) and the chain hundreds of short links (DM_RESUBMIT=0 submits a single job)"
fi
# the first job: the chain's options, the user's dependency merged with singleton, -b / -H
FIRST=("${OPTS[@]}")
[ -n "$DEP" ] && FIRST[${#FIRST[@]}-1]=--dependency=$DEP
FIRST+=(${FIRST_ONLY[@]+"${FIRST_ONLY[@]}"})
DM_CHAIN_DIR=$W/logs/slurm/chains/$NAME
export DM_SBATCH_OPTS="${OPTS[*]}" DM_SLURM_REPO="$REPO" DM_CHAIN_DIR   # Slurm runs a copy of job.sh
if [ "${DM_SUBMIT_DRY:-0}" = 1 ]; then   # print what would be submitted (tests/test_slurm_wrappers.py); write nothing
  printf 'DRY sbatch --parsable'; printf ' %q' "${FIRST[@]}" "$REPO/scripts/slurm/job.sh" "$@"; printf '\n'
  echo "DRY DM_SITE_ENV=${DM_SITE_ENV:-} DM_SLURM_REPO=$DM_SLURM_REPO DM_SBATCH_OPTS=$DM_SBATCH_OPTS"
  echo "DRY chain=$NAME dir=$DM_CHAIN_DIR"
  exit 0
fi

mkdir -p "$W/logs/slurm" || { echo "FATAL: cannot create $W/logs/slurm" >&2; exit 1; }
ME=${USER:-$(id -un)}
# the user's queued jobs: this chain's (the new submission runs after them), and the same command and first argument
# under another name (a chain of an older submit.sh without the hash, other knobs, or another spelling of the command)
SAME=() OTHER=()
if QUEUED=$(timeout 30 squeue -h -u "$ME" -o '%i %j' 2>/dev/null); then
  while read -r id nm; do
    [[ ${id:-} =~ ^[0-9]+ ]] || continue
    if [ "$nm" = "$NAME" ]; then SAME+=("$id")
    elif [ "$nm" = "$PREFIX" ] || [[ ${nm#"$PREFIX"-} =~ ^[0-9a-f]{8}$ && $nm == "$PREFIX"-* ]]; then OTHER+=("$id ($nm)"); fi
  done <<< "$QUEUED"
fi
if [ "${#SAME[@]}" -gt 0 ]; then
  echo "WARN: the chain $NAME already has jobs in the queue (${SAME[*]}): the new submission runs after them (singleton) and continues the chain (they adopt it as their successor and never cancel it); to replace them instead, stop them first: scancel -u $ME -n $NAME" >&2
fi
if [ "${#OTHER[@]}" -gt 0 ]; then
  echo "WARN: jobs of $1 ${2:-} under another chain name are queued: ${OTHER[*]} (an older submit.sh, other knobs, or another spelling of the command): if they train the same runs, the run lock stops one of them (exit 3); if this submission replaces them, stop them first (scancel -u $ME -n <their name>)" >&2
fi
if [ "${DM_RESUBMIT:-1}" = 1 ]; then
  mkdir -p "$DM_CHAIN_DIR" || { echo "FATAL: cannot create $DM_CHAIN_DIR" >&2; exit 1; }
  # a new submission starts with fresh failure counts (the chain's `links`, the experiment driver's runfails/) and no
  # blamed nodes (train_arm.sh's badnodes); the progress lines and the compile caches stay
  rm -f "$DM_CHAIN_DIR/links" "$DM_CHAIN_DIR/badnodes"
  rm -rf "$DM_CHAIN_DIR/runfails"
fi
# the chain's submit.sh: this submission again (resume after a stop, a give-up or a broken chain)
write_recipe () {   # write_recipe <command> [args...]
  local f=$DM_CHAIN_DIR/submit.sh k
  {
    echo '#!/bin/bash'
    echo "# Written by $REPO/scripts/slurm/submit.sh on $(date '+%F %T') for the chain $NAME: run it to submit the chain"
    echo "# again with the same command, sbatch options and knobs (the runs resume from their newest checkpoints; failure"
    echo "# counts start afresh). Options given to it are added to the recorded ones."
    if [ "${#FIRST_GIVEN[@]}" -gt 0 ]; then
      printf '# Not repeated (they applied to the first job only):'; printf ' %q' "${FIRST_GIVEN[@]}"; printf '\n'
    fi
    printf 'cd %q || exit 2\n' "$PWD"
    printf 'unset %s $(compgen -v S2_)\n' "${NAME_KNOBS[*]} ${RECIPE_KNOBS[*]}"
    printf 'exec env'
    for k in "${NAME_KNOBS[@]}" $(compgen -v S2_ || true) "${RECIPE_KNOBS[@]}"; do
      if [ -n "${!k:-}" ]; then printf ' %s=%q' "$k" "${!k}"; fi
    done
    printf ' %q' "$REPO/scripts/slurm/submit.sh" ${REPLAY[@]+"${REPLAY[@]}"}
    printf ' "$@" --'
    printf ' %q' "$@"
    printf '\n'
  } > "$f.tmp.$$"
  chmod 755 "$f.tmp.$$"
  mv -f "$f.tmp.$$" "$f"
}
[ "${DM_RESUBMIT:-1}" = 1 ] && write_recipe "$@"
J=$(sbatch --parsable "${FIRST[@]}" "$REPO/scripts/slurm/job.sh" "$@") || {
  rc=$?
  echo "FATAL: sbatch failed (status $rc); if it timed out, check squeue -u $ME -n $NAME before submitting again (a second submission only runs after the first)" >&2
  exit "$rc"
}
J=${J##*$'\n'}; J=${J%%;*}   # --parsable: <id>[;<cluster>] (its last line)
[[ $J =~ ^[0-9]+$ ]] || { echo "FATAL: sbatch printed '$J', not a job id: check squeue -u $ME -n $NAME" >&2; exit 1; }
echo "submitted job $J: $*"
echo "  chain:  $NAME (stop the whole chain: scancel -u $ME -n $NAME)"
if [ "${DM_RESUBMIT:-1}" = 1 ]; then
  { printf '%s %s submit fails=0 k=0 by=%s\n' "$(date +%s)" "$J" "$ME" >> "$DM_CHAIN_DIR/links"; } 2>/dev/null || true
  echo "  state:  $DM_CHAIN_DIR (submit the chain again: $DM_CHAIN_DIR/submit.sh)"
else
  echo "  a single job (DM_RESUBMIT=0): no successor, no chain state"
fi
echo "  logs:   $W/logs/slurm/ (DM_SITE_ENV=${DM_SITE_ENV:-<none>})"
