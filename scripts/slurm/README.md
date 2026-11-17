# scripts/slurm: running the experiments on a Slurm cluster

The launchers of `scripts/campaign/` and the experiment scripts of `scripts/experiments/` run where they are started.
A stage-1 run takes about a day on 8 x H100 (about three days at 5.3 B), longer than a partition's time limit. These two
scripts run any of those commands as a chain of Slurm jobs ("links"): each link stops the trainer with a complete
checkpoint before its time limit and hands over to the next one, which resumes, until the command is done. A link can
use one node, part of one, or several nodes.

    scripts/slurm/submit.sh [sbatch options] -- <command> [args...]

`job.sh` is the body of every link; it is not run by hand. The headers of both scripts are the reference for every
detail; this file is the guide.

## How a chain works

- **A chain is every job of yours with one Slurm job name.** `submit.sh` names it `dm-<command>-<first argument>-<h8>`
  (characters other than letters, digits and `._-` become `_`). `h8` is the start of a sha1 over the command as the job
  runs it (a path made physical and relative to the repository), its arguments, the workspace's physical path and the
  knobs that define the work: `SEEDS TAG SMOKE STOP_AT_STEP NUM_STEPS EXTRA_ARGS EVAL DM_LR DM_TRAIN_CONFIG`, every
  `S2_*` variable (each when set and not empty), and `DM_DATA_ROOT` when it is not the workspace's `data`. So every arm,
  seed and knob setting is a chain of its own, and the same work is the same chain however it is spelled (`./x`, an
  absolute path, the repository's `workspace` link). `-J <name>` names a chain yourself (letters, digits and `._-`).
- **One link at a time.** Every link's sbatch options carry `--dependency=singleton`: Slurm runs one job of a name at a
  time. Slurm releases a successor as soon as its predecessor's state is final, while the predecessor may still be
  saving and holding the run's lock, so a link first waits while another job of the chain is RUNNING or COMPLETING (at
  most `DM_PRED_WAIT` seconds, 600).
- **The successor is queued at the start of a link,** with the same command and sbatch options, or a pending job of the
  chain is adopted as the successor (a resubmission, a requeued link's successor, an sbatch that timed out after
  queueing its job). Singleton holds it until the link has ended, so a link that is killed or hangs still hands over.
  A link whose sbatch failed at its start tries again at its end. A chain has at most `DM_MAX_RESUBMITS` (400)
  successors.
- **The deadline.** A link exports `DM_JOB_END` (the job's end: `SLURM_JOB_END_TIME`, else squeue) and `DM_STOP_BEFORE`
  = `DM_JOB_END` - `DM_STOP_MARGIN` (600 s): after that time the trainer saves a complete checkpoint at the next step
  boundary and exits 75. An unknown end sets no deadline; a time limit that leaves less than 10 minutes after the margin
  sets no deadline and makes the job a single job (no successor).
- **The command** runs from the repository root, with `DM_LAUNCH=env` (the launchers start the ranks themselves, so a
  SIGTERM reaches the trainer and not torchrun, which would kill its ranks 30 s later), rendezvous ports of its own
  (`DM_MASTER_PORT`, `DM_VALCE_PORT`, derived from the job id unless set) and `DM_NPROC` = the job's GPUs per node
  (lowered to them when set higher). Its exit status decides what the chain does next (below).
- **The chain's state** is `<workspace>/logs/slurm/chains/<name>/` (`DM_CHAIN_DIR`):

  | file | what |
  |---|---|
  | `links` | the records of submit.sh (`submit`) and job.sh (`judge`, `start`, `succ`, `run`, `attempt`, `end`); every one that changes the failure count carries `fails=<n>` |
  | `progress` | one line per durable progress, `<epoch> <job id> <kind> <run> [detail]`: `ckpt <run> <step>` (the trainer, right after a checkpoint's marker), `train_complete <run>` (train_arm.sh), `eval <run> <n>` and `eval_complete <run>` (eval_arm.sh, eval_arm_64k.sh) |
  | `runfails/` | the experiment driver's per-run failure counters (`scripts/experiments/README.md`) |
  | `badnodes` | `<epoch> <job id> <host>`: the hosts train_arm.sh blamed a failed training on |
  | `jit.tar` | the compile caches (below) |
  | `submit.sh` | submits the chain again: the same command, sbatch options and knobs (options given to it are added) |

  A new submission clears `links`, `runfails/` and `badnodes` (the failure counts start afresh) and keeps `progress`
  and `jit.tar`. A state directory that cannot be written costs only the accounting, never the job.
- **The logs**: one per link, `$DM_WORKSPACE/logs/slurm/<name>-<job id>.log` (lines `[slurm <host> <time>] ...` are
  job.sh's); the trainer, launcher and evaluation logs are under `$DM_WORKSPACE/logs/` as usual.

## Submitting

`submit.sh` passes your sbatch options on and adds what you leave out. An option counts as given in any spelling sbatch
accepts (`-p x`, `-px`, `--partition=x`, an unambiguous abbreviation such as `--part x`) and when its `SBATCH_*`
variable is set:

- one node (`--nodes=1`) unless `-N` / `--nodes` says more;
- without a GPU option (`--gpus*`, `--gres`, `--tres-per-task`): `--gpus-per-node=8 --mem=0 --exclusive`, whole nodes;
- `--job-name=<the chain's name>`, `--output=$DM_WORKSPACE/logs/slurm/%x-%j.log`, and `-A` / `-p` from
  `DM_SLURM_ACCOUNT` / `DM_SLURM_PARTITION` when they are set;
- `--dependency=singleton` in every link; `-d` / `--dependency` (merged with singleton), `-b` / `--begin` and `-H` /
  `--hold` apply to the first job only.

A chain needs a time limit (`-t`, or `SBATCH_TIMELIMIT`; the partitions' default is 31 minutes). `DM_RESUBMIT=0`
submits a single job instead: no successor, no in-place retry, no chain state, and the command runs outside a chain.
The shell's environment goes with the job (`--export=ALL`), so the launchers' knobs apply as usual (`SMOKE`, `SEEDS`,
`EXTRA_ARGS`, `TAG`, `STOP_AT_STEP`, `DM_SAVE_INTERVAL`, `DM_WANDB`, ...). A submission from a shell inside a link starts
a new chain at link 0, without that link's deadline, chain or derived variables. `DM_SUBMIT_DRY=1` prints the sbatch
command, the chain's name and its state directory, and changes nothing. On success `submit.sh` prints the job id, the
chain's name with the command that stops it, the state directory and the log directory.

    # one arm on one node (8 GPUs), chained 4-hour jobs until it is trained
    scripts/slurm/submit.sh -A <account> -p batch -t 4:00:00 -- scripts/campaign/train_arm.sh dm_matched_hd128 11785

    # the same on half a node: 4 GPUs (the same run, twice the gradient accumulation)
    scripts/slurm/submit.sh -A <account> -p batch -t 4:00:00 --gpus-per-node=4 --cpus-per-task=64 --mem=800G \
        -- scripts/campaign/train_arm.sh dm_matched_hd128 11785

    # an experiment's arms in one chain (each trained, then evaluated, in turn; it resumes where it stopped)
    scripts/slurm/submit.sh -A <account> -p batch -t 4:00:00 -- scripts/experiments/stage1_hd128.sh dm_matched_hd128 dm_stale_hd128

    # one run on two whole nodes (16 GPUs)
    scripts/slurm/submit.sh -A <account> -p batch -t 4:00:00 -N 2 -- scripts/experiments/stage1_l_kda.sh l_kda_dm_matched_hd256

    # the 5.3 B KDA experiment: one chain per arm on 4 x 4 GPUs, plus the clean arm's LR hedge (a run and a chain of its own)
    OPTS=(-A <account> -p batch -N 4 --gpus-per-node=4 --cpus-per-task=64 --mem=800G -t 4:00:00)
    for arm in l_kda_clean_hd256 l_kda_dm_stale_hd256 l_kda_dm_matched_hd256; do
      scripts/slurm/submit.sh "${OPTS[@]}" -- scripts/experiments/stage1_l_kda.sh $arm
    done
    TAG=_lr1p2e3 DM_LR=1.2e-3 scripts/slurm/submit.sh "${OPTS[@]}" -- scripts/experiments/stage1_l_kda.sh l_kda_clean_hd256

    # a 3-step smoke of two arms as a single job, no chain
    SMOKE=1 DM_RESUBMIT=0 scripts/slurm/submit.sh -A <account> -p batch_short -t 1:00:00 -- scripts/experiments/stage1_hd128.sh clean_hd128 dm_matched_hd128

    # the test suite on two GPUs of a shared node
    DM_RESUBMIT=0 scripts/slurm/submit.sh -A <account> -p batch -t 2:00:00 --gpus-per-node=2 --cpus-per-task=32 -- python -m pytest tests -rA

Two chains must never train the same run: the run lock stops the second one (exit 3, which stops its chain). An
evaluation submitted on its own (`eval_arm.sh`, `eval_arm_64k.sh`) is not ordered after the training: submit it once the
run's export carries `TRAIN_COMPLETE` (before that it finds no patched export and exits 2). Each of its links fills what
is missing (a benchmark interrupted by the time limit restarts from scratch), and a link that completes no benchmark is
a link without progress. An experiment script's chain avoids all this: it trains, then evaluates.

## Several nodes, or part of one

`-N <n>` (every link replays it) makes `scripts/campaign/train_arm.sh` train on every node of the job (`DM_NNODES`
trains on fewer). It starts one srun step with one task per node; each task runs `scripts/campaign/node_ranks.sh`,
which starts that node's `DM_NPROC` ranks (global rank = node x `DM_NPROC` + local rank; the master is the job's first
node, the batch host) with the collector and signal forwarding of a one-node launch (`scripts/campaign/ranks.sh`).
srun runs in a session of its own and a stop is never sent to it: a SIGTERM to srun SIGKILLs its step (seen on this
cluster), so train_arm.sh passes a stop to the node launcher on its own host, whose ranks raise the trainer's stop
flag; the trainer MAX-all-reduces that flag at the next step boundary, so every rank of every node saves the same
checkpoint and exits 75. A stop that comes before the ranks start (train_arm.sh does not know the launcher on its host
yet, or a launcher got the stop before it started its ranks) leaves a stop file in the launch's status directory, and a
launcher that has not started its ranks then starts none. The stop signals are job.sh's (TERM, INT, HUP, USR1, USR2),
in train_arm.sh and in every node launcher, which pass a stop on once, as SIGTERM (a one-node launch takes TERM and
INT). Slurm signals a step's tasks only, unless the signal is SIGKILL: the time limit, a scancel and
`scancel --signal=<sig> <job>` reach the node launchers, never their ranks, which get the stop from their launcher
(with `--full`, train_arm.sh gets it too). srun still running `DM_SRUN_GRACE` seconds after a stop (default
`DM_TERM_GRACE` + 60 = 150), or `DM_EXIT_GRACE` seconds after one node's ranks ended, gets its step killed.
`DM_SRUN_EXTRA` appends words to the srun options (a site's needs). A multi-node launch needs `srun` and `setsid` on
PATH and `DM_LAUNCH=env` (job.sh sets it); without them, or with a bad node count, train_arm.sh exits 2.

Each node launcher reports to `$DM_WORKSPACE/logs/nodes/<run>/<job id>.<epoch>.<pid>/` (the newest 10 launches of a run
are kept), and the link's log has one line per node. A node's failure time is when its launcher, which polls its ranks
once a second, saw the first rank failure before any stop reached it (a rank already gone when the stop came failed
before it, unless a stop signal killed it: the stop's time); ranks that failed only after a stop (srun's
kill-on-bad-exit SIGTERM once another node's launcher failed, the time limit) give none. The launch's status: 75
when any node's ranks saved, else the failure of the node that failed first, else 0 when every node reported 0 (the
remaining cases: train_arm.sh's header). A failure is blamed on the node that failed first and on every node whose
failure time is within 2 s of it (they failed together), else on the nodes that never reported, else on every
failing node: train_arm.sh writes those hosts to the chain's `badnodes`, and after a failure without progress the
next link avoids them (node exclusion, below). The evaluations and the end-of-run val CE run on the batch host's
GPUs only; the other nodes idle meanwhile.

Part of a node: give a GPU option (`--gpus-per-node=4`) with CPUs and memory of your own (`submit.sh` then adds neither
`--exclusive` nor `--mem=0`). `DM_NPROC` follows the job's GPUs per node, and each job has its own rendezvous ports, so
two jobs can share a node. Several partial nodes work the same way: `-N 4 --gpus-per-node=4 --cpus-per-task=64
--mem=800G` is a 16-GPU run. The derived ports lie outside the host's ephemeral port range (9000-65000 on this
cluster's compute nodes, so 7700-8999): inside it, any outgoing connection of another job on a shared node may hold the
port, and the rendezvous fails with EADDRINUSE (seen once on 2026-11-09 with the former 20000-31999 band).

The layout is not part of the run. The global batch, the data order (a function of the global step) and the schedule
stay; the gradient accumulation follows (`global_batch_size` / (`per_device_batch_size` x the world size) must be a whole
number); neither the world size nor the per-device batch is part of the run identity, and DCP reshards the state. So a
chain can be stopped and submitted again with other sbatch options and continues the same run. On this cluster
(2026-11-08) whole-node jobs waited hours in the queue while 4-GPU partial-node jobs started within seconds to minutes.

## Stopping a chain

- `scancel -u $USER -n <name>` stops the whole chain (`submit.sh` prints the command; `squeue -u $USER` lists the
  names), and so does `scancel <the running link's id>`: the link cancels the successor it queued, and nothing runs
  after it. The running command gets a SIGTERM: the trainer saves a checkpoint and exits 75, and the run resumes from it
  whenever it is submitted again.
- job.sh tells a stop by hand from the time limit: a signal at most 60 s before the job's start + time limit as squeue
  shows them when the signal comes (`scontrol update TimeLimit=` included), or with the end unknown, is the time limit,
  and the chain goes on. The trainer normally stops itself `DM_STOP_MARGIN` before the time limit; the SIGTERM at the
  limit is the fallback.
- A requeue (a preemption, `scontrol requeue`) keeps the chain: Slurm runs the same job id again, which resumes, and
  job.sh cancels and queues nothing; the attempt it cut short is not counted as a failure. job.sh tells a requeue by
  squeue's restart count (`RestartCnt`) being above the running instance's `SLURM_RESTART_COUNT`: Slurm 24.05.5 shows the
  job COMPLETING, never PENDING, while the batch script handles the SIGTERM.
- A submission while the chain still has jobs in the queue runs after them and continues the chain (they adopt it as
  their successor and never cancel it); `submit.sh` WARNs about it. To replace a chain (other sbatch options, say),
  stop it first.

Signals and wrapper scripts: Slurm's SIGTERM (the time limit, scancel, a requeue) reaches only the batch script of the
batch step (the tasks of a multi-node launch's srun step get Slurm's own at the time limit and on a scancel).
job.sh passes the first signal (TERM, INT, HUP, USR1, USR2) once, as SIGTERM, to every process of its process group and
waits for the command. The launchers and experiment scripts trap TERM and wait for their children. A wrapper script of
your own must do the same: a handler (not `trap '' TERM`: a signal ignored by the wrapper stays ignored in the bash
scripts it starts, so the launchers' own handlers never run: train_arm.sh's forwarding to the ranks and to the node
launchers, the driver's stop) and the command in the foreground (or in the background with a `wait` repeated until it
has exited: a trapped signal ends a `wait` at once), else the wrapper dies at once, the link ends, and Slurm kills the
trainer while it is still saving.

    #!/bin/bash
    trap 'echo "SIGTERM: waiting for the training to save"' TERM
    scripts/campaign/train_arm.sh dm_matched_hd128 11785   # bash runs the trap only once this has exited
    exit $?                                                # its status: 75 after a stop

## What a link does with the command's exit status

| status | meaning | the chain |
|---|---|---|
| 0 | everything requested is done | complete: its other jobs are cancelled |
| 75 | stopped with a complete checkpoint, or work remains | the successor continues |
| 2 | a configuration error: missing data, a bad knob, a refused resume, a command that cannot be run | stops (its other jobs are cancelled); never retried |
| 3 | the run belongs to another live job (its lock is held) | stops |
| 4 | gave up: nothing retryable is left (the experiment driver) | stops |
| anything else (1, 134, 137, 143, ...) | a failure | retried, below |

job.sh exits with the command's status, except 4 (it gave up, below), 76 (a successor was needed and none could be
queued: the chain is broken) and 2 (a malformed knob, a site file that fails, or a command that cannot be run, before
the command starts). sacct shows a link that handed over (75, or a failure's status) as FAILED: that is normal.

**Progress.** An attempt made progress when `progress` has a line of this job written since the attempt started: a new
complete checkpoint, a finished training, new evaluation markers. Exit statuses and log lines do not count.

- **In-place retry.** A failure with no signal, under a deadline with at least `DM_RETRY_MIN_LEFT` seconds (1800) left
  before it, runs again in the same job (no queue wait) after `DM_RETRY_PAUSE` seconds (30), at most
  `DM_INPLACE_RETRIES` times per job (1), unless the chain is about to give up.
- **Give-up.** Each attempt without progress (an exit 75 that saved nothing new included) adds one to the failure
  count, one with progress sets it back to 0, and an attempt a requeue cut short adds nothing. More than `DM_MAX_FAILS`
  (2) in a row: the link gives up (exit 4) and cancels the chain's other jobs. The first job of a new submission starts
  at 0.
- **Node exclusion.** After a failure without progress (and no signal) the pending successor is kept off the hosts
  train_arm.sh blamed the failure on in `badnodes` during this job, else off every node of this job (`scontrol update
  ExcNodeList`; the successor's own exclusions are kept).
- **A link killed without its records** (SIGKILL after KillWait, a node failure, an OOM-killed job.sh): the next link
  judges it. An attempt it started but never recorded, or a death before its command started, is one more failure
  unless `progress` has a line of it since; the attempts it recorded are not counted twice.
- **The run lock** (`scripts/campaign/env.sh` `dm_lock`): a holder that is ending, has ended or is this job is waited
  for (`DM_LOCK_WAIT` seconds, 600; still held then: exit 1, a failure to retry); another live job: exit 3.
- **The experiment driver** keeps counters per run (`runfails/`, `DM_RUN_MAX_FAILS`): a run that keeps failing is given
  up for the chain while the others go on (`scripts/experiments/README.md`).

## Compile caches

With `DM_JIT_CACHE=1` (the default) a chain keeps its Triton, TileLang and Inductor caches at one node-local path,
`$DM_JIT_ROOT/<name>` (default `DM_JIT_ROOT=/tmp/dm-jit-<uid>`; one fixed path, as Triton records absolute paths). A
link restores them from the chain's `jit.tar` before the command starts (at most 300 s; a failure starts with empty
caches), and node_ranks.sh does the same on the other nodes of a multi-node link (`DM_JIT_CACHE_DIR` names the
directory; a stop during that restore cuts it short, and that node starts no rank). A link that hands over to its
successor without a signal and with at least 120 s left saves changed caches back to `jit.tar` (at most
`DM_JIT_MAX_MB`, 8192). In a chain these directories replace the site file's `TRITON_CACHE_DIR`, `TILELANG_CACHE_DIR`
and `TORCHINDUCTOR_CACHE_DIR`; the env launch gives every rank its own Triton and TileLang subdirectory
(`r<local rank>`). On a node of a multi-node launch other than the batch host, a Triton or TileLang root that other
nodes may share (any root but the node-local ones below, e.g. env.sh's default in the workspace or a `DM_JIT_ROOT` on a
shared file system) gets a level per node, `<root>/n<node>/r<local rank>`, and node_ranks.sh logs that it "may be
shared"; a node-local root (under `/tmp` or `/dev/shm`, or the cache directory that node_ranks.sh created) keeps
`<root>/r<local rank>`, the layout of `jit.tar`.

## Knobs

| variable | default | meaning |
|---|---|---|
| `DM_STOP_MARGIN` | 600 | seconds between the trainer's deadline and the job's end |
| `DM_MAX_RESUBMITS` | 400 | the most successors a chain queues (a stage-1 run is about 7 four-hour links on one node) |
| `DM_MAX_FAILS` | 2 | attempts in a row without progress before the chain gives up |
| `DM_INPLACE_RETRIES` | 1 | in-place retries of a failure per job |
| `DM_RETRY_MIN_LEFT` | 1800 | seconds before the deadline an in-place retry needs |
| `DM_RETRY_PAUSE` | 30 | seconds before an in-place retry |
| `DM_PRED_WAIT` | 600 | seconds a link waits for a predecessor that is still running or completing (0: no wait) |
| `DM_RESUBMIT` | 1 | 0: a single job, no chain |
| `DM_RESUBMIT_BACKOFF` | unset | seconds before each sbatch retry; replaces the schedules (start of a link: 10 30; end: 10 30 60 120 120 ..., until 90 s before the job's end, at most 30 minutes) |
| `DM_JIT_CACHE`, `DM_JIT_ROOT`, `DM_JIT_MAX_MB` | 1, `/tmp/dm-jit-<uid>`, 8192 | the compile caches (above) |
| `DM_SITE_ENV` | `$DM_WORKSPACE/site_env.sh` when it exists | the site file (below) |
| `DM_SLURM_ACCOUNT`, `DM_SLURM_PARTITION` | unset | `-A` / `-p` when not given |
| `DM_SUBMIT_DRY` | 0 | 1: print what would be submitted, change nothing |
| `DM_LAUNCH` | `env` (set by job.sh unless set) | how train_arm.sh starts the ranks |
| `DM_MASTER_PORT`, `DM_VALCE_PORT` | per job, outside the batch host's ephemeral port range: the start of a band of 1300 ports just below that range (else just above it, from 1024 at the least, else 20000-31999) + 2 x (job id mod its slots), and the next port; a pair the host has a socket on (`/proc/net/tcp*`) or a listener on is skipped, and they are derived again before an in-place retry. A port of your own is kept (a WARN when it lies in the range; not 1-65535, or 65535 with no `DM_VALCE_PORT`: exit 2) | the rendezvous ports (job.sh) |
| `DM_EPHEMERAL_PORTS` | the host's `net.ipv4.ip_local_port_range` (unreadable: 9000 65000) | `<lo> <hi>` (1 <= lo <= hi <= 65535): the ephemeral port range the derived ports avoid |
| `DM_NPROC` | the job's GPUs per node | ranks per node |
| `DM_NNODES` | the job's node count | the nodes train_arm.sh trains on |
| `DM_SRUN_EXTRA` | none | words appended to the srun options of a multi-node launch |
| `DM_SRUN_GRACE` | `DM_TERM_GRACE` + 60 | multi-node: seconds srun gets after a stop before its step is killed |
| `DM_TERM_GRACE`, `DM_RANK_GRACE`, `DM_EXIT_GRACE` | 90, 60, 600 | the launchers' rank collector (`scripts/campaign/ranks.sh`, `docs/SETUP.md` section 8) |
| `DM_LOCK_WAIT` | 600 | seconds to wait for the run lock of a holder that is ending, has ended or is this job |
| `DM_EVAL_BUDGET`, `DM_MIN_TRAIN_TIME`, `DM_RUN_MAX_FAILS` | 2400 (stage 2: 11700; `stage1_l_kda.sh`: 4800), 600, 2 | the experiment driver (`scripts/experiments/README.md`) |

job.sh and submit.sh set the others themselves; do not set them: `DM_STOP_BEFORE`, `DM_JOB_END`, `DM_CHAIN`,
`DM_CHAIN_DIR`, `DM_PROGRESS_FILE`, `DM_RESUBMIT_COUNT` (the link's index), `DM_SBATCH_OPTS`, `DM_SLURM_REPO`, `DM_DERIVED`
(what a link derived for itself, never passed to the successor), `DM_JIT_CACHE_DIR`, and train_arm.sh's `DM_NODE_STATUS`
and `DM_BATCH_HOST`. The chain's `submit.sh` records the knobs of submit.sh's `NAME_KNOBS` (the work) and
`RECIPE_KNOBS` (how the chain runs: the site file, the save interval, W&B, the account and partition, job.sh's numeric
knobs with `DM_RESUBMIT` and `DM_JIT_CACHE`, and the experiment driver's `DM_EVAL_BUDGET`, `DM_MIN_TRAIN_TIME` and
`DM_RUN_MAX_FAILS`) and every `S2_*` variable, and the recorded values replace the shell's (a knob of these lists that
was not set at the submission is unset); every other variable comes from the shell that runs it
(`DM_ALLOW_RUN_CHANGE`, `DM_NNODES`, `DM_SRUN_EXTRA`, `DM_JIT_ROOT`, `DM_RESUBMIT_BACKOFF`, ...).

`submit.sh` refuses (exit 2): `--array`, `--wait`, `--signal` (job.sh handles the signals), `--export` other than `ALL`,
`--wrap`, `--test-only`, an ambiguous abbreviation, a word before `--` that is not an option, an option value with a
space, more than one `-d`, an OR dependency (`?`), a `-J` name with other characters than letters, digits and `._-`, a
chain without a time limit, a command that cannot be run (a path that is not an executable file as the job runs it,
from the repository root; a script without the x bit: `bash <script>`), `DRY=1` (the driver's preview would exit 0 and
complete the chain), `SBATCH_ARRAY_INX`, `SBATCH_WAIT` or `SBATCH_SIGNAL` set, `SBATCH_EXPORT` other than `ALL`, and a
malformed knob (`DM_STOP_MARGIN`, `DM_MAX_RESUBMITS`, `DM_MAX_FAILS`, `DM_INPLACE_RETRIES`, `DM_RETRY_MIN_LEFT`,
`DM_RETRY_PAUSE`, `DM_JIT_MAX_MB`, `DM_PRED_WAIT` not a whole number; `DM_RESUBMIT`, `DM_JIT_CACHE` neither 0 nor 1;
`DM_RESUBMIT_BACKOFF` not a list of whole numbers). It WARNs when the chain already has jobs in the queue (the new
submission continues the chain), when jobs of the same command and first argument are queued under another chain name
(an older submit.sh, other knobs, another spelling: two chains of one run would train it at once), and when a bare
command name is not on the shell's PATH (the job looks it up on the site file's PATH).

## Recovery

- **Exit 76, the chain is broken** (no successor could be queued, e.g. sbatch refused by a submit limit): run
  `<chain dir>/submit.sh`. The run resumes from its newest complete checkpoint.
- **Exit 4, the chain gave up** (attempts without progress, or the experiment driver has nothing retryable left): find
  the cause in the link logs and the run's logs (`$DM_WORKSPACE/logs/<run>.log`, `<run>.rank<r>.log`; `badnodes` names
  the hosts a failure was blamed on), fix it, then run `<chain dir>/submit.sh` (the failure counts start afresh).
- **Exit 2, a configuration error**: the link log's `FATAL:` line says what. A refused resume prints `FATAL: RESUME
  REFUSED` on every rank (the trainer's log, `$DM_WORKSPACE/logs/<run>.log`) with every value that differs from the run's
  record, `<ckpt dir>/run_identity.json` (recorded -> now), or the two kernel builds. Restore the values (`NUM_STEPS`,
  `DM_LR`, `EXTRA_ARGS`, `S2_*`, the arm's environment, the yaml) and submit again, or accept the change:
  `DM_ALLOW_RUN_CHANGE=1 <chain dir>/submit.sh` (the run identity), `DM_ALLOW_KERNEL_CHANGE=1 <chain dir>/submit.sh`
  (another kernel build). In a chain each knob is one-shot: train_arm.sh drops it in the later links once this
  submission has recorded the change in the run (`run_identity.json` rewritten, or a checkpoint saved on the new build),
  so a later accidental change is refused again. Under an experiment script a 2 gives that run up at once; the chain
  goes on with the other runs and stops with 4 when nothing else is left.
- **The newest checkpoint's marker cannot be read** (an I/O error on `<ckpt dir>/checkpoint-N/.complete` after 4 tries):
  the trainer refuses to fall back to an older checkpoint and fails (exit 1) at every resume, so the chain retries,
  then gives up. Once the file system reads the marker again, run `<chain dir>/submit.sh`. If it stays unreadable, set
  the checkpoint aside, `mv <ckpt dir>/checkpoint-N <ckpt dir>/.bad-checkpoint-N` (a name nothing reads or removes),
  then run `<chain dir>/submit.sh`: the run resumes from its newest older complete checkpoint. Remove the `.bad-`
  directory once it is no longer needed.
- **Exit 3, the run belongs to another job**: two chains train the same run (an older submit.sh's chain, or the same
  command with other knobs or another spelling: `submit.sh` WARNs about them). Stop one of them.
- **Another node layout**: stop the chain, wait until its links have ended, and submit the same command and knobs with
  the new sbatch options; the run continues where it stopped.

## Deploying this version

- Stop the chains of an older `submit.sh` first, `scancel -u $USER -n dm-<command>-<first argument>` (their names carry
  no hash), and wait until their links have ended (`squeue -u $USER`) before submitting the same work again. An old
  chain and a new one are not ordered by singleton, and an old link that started before the deploy runs the old
  train_arm.sh, which takes no run lock: both could train the same run. A link of an old chain that queues its successor
  after the deploy gives it the new job.sh, which runs it as a single job (its options carry no singleton; a WARN in its
  log says so), so the old chain ends after that link.
- Do not edit the scripts a running chain executes: every link reads `train_arm.sh`, `ranks.sh`, `node_ranks.sh`,
  `env.sh`, `arms.sh`, `stage2_arm.sh`, `eval_arm*.sh`, the experiment scripts and `_run.sh` from the repository, and bash
  reads a script while it runs it, so an edit in place can corrupt a running link. Only `job.sh` is copied by Slurm,
  when a link is queued; a pending successor keeps the copy it was queued with. Deploy between links, or write the new
  file beside the old one and rename it into place (a running bash keeps reading the old file).

## The site file

`DM_SITE_ENV` is a bash file of your cluster's settings, sourced (without `set -u`) at the start of every link, after
job.sh's signal handlers are in place (a stop during the site file is a stop); a site file that fails stops the chain
(exit 2). A relative path is made absolute by `submit.sh`. Its default is `site_env.sh` in the workspace of the
submitting shell (`$DM_WORKSPACE`, else `<repo>/workspace`), resolved by `submit.sh` before the file is ever read: a file
elsewhere needs `DM_SITE_ENV` (or `DM_WORKSPACE`) exported in the shell that runs `submit.sh` (its output names the file
it used). It must activate the Python environment (docs/SETUP.md section 2) and may set anything else the launchers
read. A minimal one:

    source /path/to/conda/etc/profile.d/conda.sh && conda activate <env>    # torch 2.12+cu130, the bundled kernel
    export DM_WORKSPACE=/path/to/workspace DM_DATA_ROOT=/path/to/workspace/data
    export HF_HOME=$DM_DATA_ROOT/hf_cache                                  # the evaluation datasets (data/prepare_eval.py)
    export TMPDIR=/tmp/dm-$SLURM_JOB_ID; mkdir -p $TMPDIR                  # node-local scratch
    export TRITON_CACHE_DIR=$TMPDIR/triton_cache TILELANG_CACHE_DIR=$TMPDIR/tilelang_cache   # compile caches of a single job

`TMPDIR=/tmp/dm-$SLURM_JOB_ID` is removed by job.sh when the job ends, and on the other nodes of a multi-node job by
node_ranks.sh (which also creates it there) when its launch ends; in a chain the compile-cache variables are replaced by the chain's own
(above). On nodes whose NVIDIA driver is older than 580 the CUDA 13 build needs NVIDIA's forward-compatibility libraries
(`cuda-compat-13-0`) on `LD_LIBRARY_PATH`; set that in the site file or in the environment's activation hook. Note that
a pre-set `HF_HOME` is kept by `scripts/campaign/env.sh`, so point it at the data root's `hf_cache` here.
