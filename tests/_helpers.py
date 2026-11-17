"""Shared helpers of the test suite: where things are, the environment of the subprocesses, the tiny arms.

Locations follow scripts/campaign/env.sh (DM_WORKSPACE, DM_DATA_ROOT, DM_TOKENIZER_DIR, HF_HOME), with one difference:
DM_PY defaults to the interpreter running pytest (env.sh: the repo's .venv, then PATH)."""
import importlib.util, os, pathlib, re, shutil, socket, subprocess, sys, time
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src"
NEMO = "nemotron_cc_v2d1_hq_dqa"
LN_VOCAB = 10.373491  # ln(32000): the cross-entropy of a uniform model over the 32k vocabulary


def workspace() -> pathlib.Path:
    return pathlib.Path(os.environ.get("DM_WORKSPACE", str(REPO / "workspace")))


def data_root() -> pathlib.Path:
    """DM_DATA_ROOT, else <workspace>/data (the data root data/ builds, data/README.md)."""
    r = os.environ.get("DM_DATA_ROOT")
    return pathlib.Path(r) if r else workspace() / "data"


def hf_home() -> pathlib.Path:
    """HF_HOME, else <data root>/hf_cache (the evaluation datasets data/prepare_eval.py downloads)."""
    return pathlib.Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else data_root() / "hf_cache"


def tokenizer_dir() -> pathlib.Path | None:
    """DM_TOKENIZER_DIR, else <data root>/tokenizer/llama2_32k; None when it holds no tokenizer.model."""
    d = pathlib.Path(os.environ.get("DM_TOKENIZER_DIR") or data_root() / "tokenizer" / "llama2_32k")
    return d if (d / "tokenizer.model").exists() else None


def require_kernel():
    """The fp8 arms need the delta_matching kernel package (docs/SETUP.md section 3); a missing package is a failure of
    the environment, not a reason to skip."""
    if importlib.util.find_spec("delta_matching") is None:
        pytest.fail("the delta_matching kernel package is not importable: install the delta-matching kernel package "
                    "(scripts/install_kernel.sh, docs/SETUP.md section 3)")


_BF16DP: dict = {}


def kernel_bf16dp() -> bool:
    """Whether the delta_matching library the tests load (DM_SO, else the package's own) is the bf16-dP variant, i.e. a
    DM_BF16_DP=1 build (build_bf16dp() == 1). Read once per session, in a subprocess with the tests' environment (the
    library the subprocess tests load); a missing package fails as in require_kernel."""
    if "flag" not in _BF16DP:
        require_kernel()
        p = subprocess.run([python(), "-c", "from delta_matching.core import _ops; print('FLAG', int(_ops().build_bf16dp()))"],
                           capture_output=True, text=True, env=base_env(), timeout=300)
        if p.returncode != 0 or "FLAG" not in p.stdout:
            pytest.fail(f"loading the delta_matching library failed:\n{p.stderr[-2000:]}")
        _BF16DP["flag"] = p.stdout.split("FLAG", 1)[1].split()[0] == "1"
    return _BF16DP["flag"]


def require_standard_kernel(what: str):
    """Skip a test of the stale / matched kernel paths when the loaded library is the bf16-dP variant (DM_BF16_DP=1):
    that build serves dm_mode bf16_dp only and refuses the other modes by design (the refusal itself is checked by
    test_kernels.py::test_dm_mixer_checks_the_bf16dp_build_flag). A missing package still fails (require_kernel)."""
    if kernel_bf16dp():
        pytest.skip(f"{what}: the loaded delta_matching library is the bf16-dP variant (a DM_BF16_DP=1 build, "
                    "build_bf16dp() == 1), which serves dm_mode bf16_dp only and refuses the stale / matched paths by "
                    "design; run this test against a DM_BF16_DP=0 build (unset DM_SO)")


def python() -> str:
    return os.environ.get("DM_PY", sys.executable)


def torchrun() -> str:
    return os.environ.get("DM_TORCHRUN", str(pathlib.Path(python()).parent / "torchrun"))


def free_port() -> int:
    """A TCP port nobody listens on right now (for a subprocess's torch.distributed rendezvous)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# The behaviour knobs of the launchers, the experiment drivers, the trainer, the kernels and the evaluators that a
# user's shell may carry (docs/SETUP.md, scripts/campaign/*.sh, scripts/experiments/*.sh). The subprocesses of the tests
# never inherit them: a test sets the ones it needs. Only the kernel library's location passes through (DM_SO names the library the
# delta_matching package loads, as PYTHONPATH names the code). SLURM_* goes too: under a Slurm job (a GPU node's test
# run) the launchers would otherwise take the job's identity (run-lock owners, the chain's grace periods, the progress
# lines' job id) and the driver its start time.
_SCRUB = ("EXTRA_ARGS", "TAG", "SMOKE", "STOP_AT_STEP", "NUM_STEPS", "SAVE_TOTAL_LIMIT", "FORCE", "NAME_ONLY", "WANDB_JT",
          "ARM", "SEED", "SEEDS", "DRY", "EVAL", "EVAL_ENGINE", "ENGINE_MAX_SEQS",
          "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
_SCRUB_PREFIXES = ("DM_", "S2_", "SLURM_")
_KEEP = ("DM_SO", "DM_JIT_DIR", "DM_FA_SRC")


def clean_env() -> dict:
    """The caller's environment minus the knobs above."""
    return {k: v for k, v in os.environ.items()
            if k in _KEEP or not (k in _SCRUB or k.startswith(_SCRUB_PREFIXES))}


def base_env(gpu: str | None = None) -> dict:
    """The environment for a subprocess that imports the repo (trainer, eval): the scrubbed environment, PYTHONPATH,
    caches, the test workspace, no wandb."""
    env = clean_env()
    env["PYTHONPATH"] = str(SRC) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["DM_WORKSPACE"] = str(workspace())
    env.setdefault("HF_HOME", str(hf_home()))
    env.setdefault("TRITON_CACHE_DIR", str(workspace() / "triton_cache"))
    env.setdefault("TILELANG_CACHE_DIR", str(workspace() / "tilelang_cache"))
    env["WANDB_MODE"] = "disabled"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def memmap_ppl_env(gpu) -> dict:
    """base_env for evaluation.ppl.memmap_ppl, which initializes a single-process torch.distributed group."""
    env = base_env(gpu)
    env.update({"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(free_port())})
    return env


def dry_env(tmp_path, **extra) -> dict:
    """The environment of the dry launcher checks (no GPU, no data): the scrubbed environment with every GPU hidden
    (CUDA_VISIBLE_DEVICES empty: the checks never need one, and a pre-flight that imports the model stack must not touch
    a busy GPU), and a workspace and a data root under tmp_path."""
    env = clean_env()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.update({"DM_WORKSPACE": str(tmp_path), "DM_DATA_ROOT": str(tmp_path / "data"), "DM_PY": python()})
    env.update({k: str(v) for k, v in extra.items()})
    return env


def fake_data_root(tmp_path) -> pathlib.Path:
    """<tmp_path>/data holding the files the launchers check for before they start anything: the stage-1 training
    stream and the tokenizer (empty files)."""
    d = tmp_path / "data"
    (d / NEMO).mkdir(parents=True, exist_ok=True)
    (d / NEMO / "nemotron_rs_train.bin").write_bytes(b"")
    (d / "tokenizer" / "llama2_32k").mkdir(parents=True, exist_ok=True)
    (d / "tokenizer" / "llama2_32k" / "tokenizer.model").write_bytes(b"")
    return d


# ---------------------------------------------------------------- stand-ins for the launcher tests (CPU, no trainer)
# scripts/campaign/train_arm.sh, stage2_arm.sh and eval_arm*.sh driven end to end with a stand-in interpreter (DM_PY),
# torchrun and squeue: what the launchers decide (locks, collection, the end-of-run tail, progress lines) without a GPU.
FAKE_PY = r"""#!/bin/bash
# a stand-in DM_PY: every call is logged to $FAKE_CALLS; `-m pretrain.train` runs a stand-in rank, `-m
# pretrain.hf.check_export` and `-m evaluation.ppl.memmap_ppl` stand-ins; `-c "import pretrain.train"` passes; anything
# else (the export patch, the eval config checks, the numpy checks, the kernel check) runs the real interpreter
echo "py ${1:-} ${2%%[[:space:]]*}" >> "$FAKE_CALLS"
case "${1:-}" in
  -c) case "${2:-}" in *"import pretrain.train"*) exit 0 ;; esac ;;
  -m) case "${2:-}" in
        pretrain.train) shift 2; exec bash "$FAKE_DIR/rank.sh" "$@" ;;
        pretrain.hf.check_export) shift 2; exec bash "$FAKE_DIR/check_export.sh" "$@" ;;
        evaluation.ppl.memmap_ppl) shift 2; exec bash "$FAKE_DIR/valce.sh" "$@" ;;
      esac ;;
esac
exec "$FAKE_REAL_PY" "$@"
"""
# A stand-in trainer rank (RANK from the environment). FAKE_MODE: complete (rank 0 writes a whole export: model, then
# config.json last, as the atomic export does), crash_after_export (the same, then rank 0 exits 1: a teardown crash),
# export_incomplete (config.json and a truncated model, exit 0), noexport (exit 0, the export untouched), stop (a
# complete checkpoint, exit 75), fail (every rank exits 1), peer_fail (rank 1 prints 40 lines and exits 1; the others
# hang and ignore SIGTERM, like ranks blocked in a collective), complete_on_term (wait for SIGTERM, then rank 0 writes a
# whole export and every rank exits 0: the stop came with the last step; "rank <r> ready" is logged once the handler is
# in place). FAKE_CONFIG: the exported config.json.
# The multi-node modes (tests/test_multinode.py): per_rank (FAKE_RANK_MODES holds one mode per global rank, the last one
# repeating) and allreduce_stop (every rank "stop"). ok: exit 0, rank 0 writes a whole export; fail: 40 lines, exit
# FAKE_FAIL_RC (1); failon:<flag>:<rc>: waits (ignoring SIGTERM) until the file $FAKE_DIR/<flag> exists, then 40 lines and
# exit <rc> (a test decides when it fails); hang: ignores SIGTERM (blocked in a collective); stop: the trainer's
# MAX-all-reduced stop flag (a SIGTERM or SIGUSR1, as for the trainer, to any rank of any node creates the shared file
# FAKE_STOP_FLAG, every rank polls it; rank 0 then writes a complete checkpoint and every rank exits 75). Each logs "rank
# <r> caches <TRITON_CACHE_DIR> <TILELANG_CACHE_DIR>", then "rank <r> mode <m> local <l> host <SLURMD_NODENAME> master
# <addr:port> world <W>/<local W> jit <0|1> tmpdir <0|1> ready" once its handler is in place (jit: $TRITON_CACHE_DIR/k.bin
# exists; tmpdir: $TMPDIR exists).
FAKE_RANK = r"""#!/bin/bash
HF=""; RUN=""; CK=""; ALL="$*"
while [ $# -gt 0 ]; do case "$1" in --hf_save_dir) HF=$2; shift ;; --run_name) RUN=$2; shift ;; --ckpt_root_dir) CK=$2; shift ;; esac; shift; done
echo "fake rank $RANK of $WORLD_SIZE: $RUN mode ${FAKE_MODE:-complete}"
echo "rank $RANK args: $ALL" >> "$FAKE_CALLS"
CFG=${FAKE_CONFIG:-}; [ -n "$CFG" ] || CFG='{"attn_impl": "sdpa", "layer_mixers": ["gated_deltanet", "gqa"]}'
export_whole () { mkdir -p "$HF"; printf WHOLE > "$HF/model.safetensors"; printf '%s\n' "$CFG" > "$HF/config.json.tmp" && mv -f "$HF/config.json.tmp" "$HF/config.json"; }
case "${FAKE_MODE:-complete}" in
  complete) [ "$RANK" = 0 ] && export_whole; exit 0 ;;
  crash_after_export) if [ "$RANK" = 0 ]; then export_whole; echo "RuntimeError: teardown"; exit 1; fi; exit 0 ;;
  export_incomplete) if [ "$RANK" = 0 ]; then mkdir -p "$HF"; printf '%s\n' "$CFG" > "$HF/config.json"; printf TRUNC > "$HF/model.safetensors"; fi; exit 0 ;;
  noexport) exit 0 ;;
  stop) if [ "$RANK" = 0 ]; then mkdir -p "$CK/$RUN/checkpoint-5"; : > "$CK/$RUN/checkpoint-5/.complete"; fi; echo "Stopping early (signal 15) at step 5"; exit 75 ;;
  fail) echo "rank $RANK: Traceback: RuntimeError: boom"; exit 1 ;;
  peer_fail)
    if [ "$RANK" = 1 ]; then for i in $(seq 1 40); do echo "rank 1 traceback line $i"; done; exit 1; fi
    trap 'echo "rank $RANK: SIGTERM pending (blocked in a collective)"' TERM
    while :; do sleep 0.1; done ;;
  complete_on_term)
    trap '[ "$RANK" = 0 ] && export_whole; exit 0' TERM; echo "rank $RANK ready" >> "$FAKE_CALLS"
    while :; do sleep 0.05; done ;;
  per_rank|allreduce_stop)
    F=${FAKE_STOP_FLAG:-$FAKE_DIR/stop.flag}
    if [ "$FAKE_MODE" = per_rank ]; then read -r -a MS <<< "${FAKE_RANK_MODES:-ok}"; M=${MS[RANK]:-${MS[-1]}}; else M=stop; fi
    case $M in
      hang|failon:*) trap 'echo "rank $RANK: SIGTERM pending (blocked in a collective)"' TERM ;;
      stop) trap ': > "$F"' TERM USR1 ;;
    esac
    echo "rank $RANK caches ${TRITON_CACHE_DIR:-} ${TILELANG_CACHE_DIR:-}" >> "$FAKE_CALLS"
    echo "rank $RANK mode $M local ${LOCAL_RANK:-} host ${SLURMD_NODENAME:-none} master ${MASTER_ADDR:-}:${MASTER_PORT:-} world $WORLD_SIZE/${LOCAL_WORLD_SIZE:-} jit $([ -e "${TRITON_CACHE_DIR:-/nonexistent}/k.bin" ] && echo 1 || echo 0) tmpdir $([ -d "${TMPDIR:-/nonexistent}" ] && echo 1 || echo 0) ready" >> "$FAKE_CALLS"
    case $M in
      ok) [ "$RANK" = 0 ] && export_whole; exit 0 ;;
      fail) for i in $(seq 1 40); do echo "rank $RANK traceback line $i"; done; exit "${FAKE_FAIL_RC:-1}" ;;
      failon:*) IFS=: read -r _ FL FRC <<< "$M"; while [ ! -e "$FAKE_DIR/$FL" ]; do sleep 0.05; done
                for i in $(seq 1 40); do echo "rank $RANK traceback line $i"; done; exit "${FRC:-1}" ;;
      hang) while :; do sleep 0.1; done ;;
      stop) while [ ! -e "$F" ]; do sleep 0.05; done
            if [ "$RANK" = 0 ]; then mkdir -p "$CK/$RUN/checkpoint-5"; : > "$CK/$RUN/checkpoint-5/.complete"; fi
            echo "Stopping early (signal 15) at step 5"; exit 75 ;;
    esac ;;
esac
"""
FAKE_CHECK_EXPORT = r"""#!/bin/bash
# stand-in for python -m pretrain.hf.check_export <dir>: exit 0 iff config.json parses and model.safetensors is "WHOLE"
d=${1:?dir}
"$FAKE_REAL_PY" -c 'import json, sys; json.load(open(sys.argv[1]))' "$d/config.json" 2>/dev/null || { echo "config.json missing or unreadable in $d"; exit 1; }
[ "$(cat "$d/model.safetensors" 2>/dev/null)" = WHOLE ] || { echo "model.safetensors in $d is not whole"; exit 1; }
echo "export whole: $d"
"""
# The val CE stand-in logs its port and what existed when it started (the marker, the progress line); with an output
# dir it writes the per-window artifact the eval scripts check. FAKE_VALCE_READY=<file>: touch it, then wait for a signal.
FAKE_VALCE = r"""#!/bin/bash
M=""; O=""; R=""; W=128
while [ $# -gt 0 ]; do case "$1" in --model_name_or_path) M=$2; shift ;; --output_dir) O=$2; shift ;; --run_name) R=$2; shift ;; --max_windows) W=$2; shift ;; esac; shift; done
P=0; [ -n "${DM_PROGRESS_FILE:-}" ] && P=$(grep -c " train_complete " "$DM_PROGRESS_FILE" 2>/dev/null)
echo "valce port=${MASTER_PORT:-} marker=$([ -f "$M/TRAIN_COMPLETE" ] && echo 1 || echo 0) progress=${P:-0}" >> "$FAKE_CALLS"
if [ -n "${FAKE_VALCE_READY:-}" ]; then trap 'echo "valce SIGTERM" >> "$FAKE_CALLS"; exit 143' TERM; : > "$FAKE_VALCE_READY"; sleep 60 & wait; exit 143; fi
[ -n "$O" ] && "$FAKE_REAL_PY" -c 'import numpy as np, sys; np.save(sys.argv[1], np.ones((2, int(sys.argv[2]))))' "$O/${R}_perwindow.npy" "$W"
echo "[memmap-ppl RESULT] nll/tok=2.5 windows=$W"
"""
# A stand-in torchrun for the eval scripts: writes <output dir>/<name>/<name>_results.complete for the config it is
# given (name = the yaml's stem); FAKE_FAIL_CFGS fail instead, FAKE_SLOW_CFG touches FAKE_READY and waits for a signal.
FAKE_TORCHRUN = r"""#!/bin/bash
Y=""; O=""
while [ $# -gt 0 ]; do case "$1" in --eval_config) Y=$2; shift ;; --output_dir) O=$2; shift ;; esac; shift; done
name=$(basename "$Y" .yaml)
echo "torchrun $name" >> "$FAKE_CALLS"
if [ "$name" = "${FAKE_SLOW_CFG:-}" ]; then trap 'exit 143' TERM; : > "$FAKE_READY"; sleep 60 & wait; exit 143; fi
case " ${FAKE_FAIL_CFGS:-} " in *" $name "*) echo "Traceback: $name failed"; exit 1 ;; esac
mkdir -p "$O/$name" && date > "$O/$name/${name}_results.complete"
"""
# A stand-in squeue: FAKE_SQUEUE = a state printed for any query, "invalid" (exit 1: Invalid job id specified, the job
# is gone) or "fail" (exit 1: another error, no answer); "file:<path>" reads one of those from <path> at every query (a
# test changes it while the launcher waits).
FAKE_SQUEUE = r"""#!/bin/bash
echo "squeue $*" >> "$FAKE_CALLS"
s=${FAKE_SQUEUE:-invalid}
case "$s" in file:*) s=$(cat "${s#file:}" 2>/dev/null) || s=invalid ;; esac
case "$s" in
  invalid) echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1 ;;
  fail) echo "slurm_load_jobs error: Socket timed out on send/recv operation" >&2; exit 1 ;;
  *) echo "$s" ;;
esac
"""


def fake_launch_env(tmp_path, **extra) -> dict:
    """dry_env plus the stand-ins above: DM_PY / DM_TORCHRUN / squeue (first on PATH) under <tmp_path>/fake, the calls
    log <tmp_path>/calls.log (read_calls), a data root with the training stream and the tokenizer, HF_HOME under
    tmp_path, env launch of 2 ranks, no wandb, a fixed DM_GIT_REV."""
    fake = tmp_path / "fake"
    (fake / "bin").mkdir(parents=True, exist_ok=True)
    for name, text in (("python", FAKE_PY), ("rank.sh", FAKE_RANK), ("check_export.sh", FAKE_CHECK_EXPORT),
                       ("valce.sh", FAKE_VALCE), ("torchrun", FAKE_TORCHRUN), ("bin/squeue", FAKE_SQUEUE)):
        (fake / name).write_text(text)
        (fake / name).chmod(0o755)
    fake_data_root(tmp_path)
    env = dry_env(tmp_path, DM_PY=fake / "python", DM_TORCHRUN=fake / "torchrun", DM_LAUNCH="env", DM_NPROC=2,
                  DM_WANDB=0, DM_GIT_REV="fake", FAKE_DIR=fake, FAKE_CALLS=tmp_path / "calls.log", FAKE_REAL_PY=sys.executable,
                  HF_HOME=tmp_path / "hf_home", DM_TOKENIZER_DIR=tmp_path / "data" / "tokenizer" / "llama2_32k")
    env["PATH"] = f"{fake / 'bin'}:{env.get('PATH', '/usr/bin:/bin')}"
    env.update({k: str(v) for k, v in extra.items()})
    return env


def read_calls(tmp_path) -> list:
    """The stand-ins' calls log (fake_launch_env), one entry per line."""
    p = tmp_path / "calls.log"
    return p.read_text().splitlines() if p.exists() else []


# ---------------------------------------------------------------- stand-ins for the multi-node launch (CPU, no Slurm)
# A stand-in srun (run by fake/bin/srun with the real interpreter): it runs its command once per task (--ntasks, else
# --nodes), locally, each task in a session of its own with SLURM_NODEID = SLURM_PROCID = k, SLURM_STEP_ID =
# FAKE_SRUN_STEP (7) and SLURMD_NODENAME = the k-th name of the comma-separated SLURM_JOB_NODELIST (fake-n0,fake-n1,..).
# What srun does on this cluster: a task that exits non-zero ends the step (kill-on-bad-exit): every task (the process
# srun started, not its children: Slurm signals a step's tasks only, unless the signal is SIGKILL; seen on this cluster)
# gets a catchable SIGTERM, and every process of every task SIGKILL FAKE_SRUN_KILL_WAIT seconds later (Slurm's KillWait,
# 120); a SIGTERM to srun itself SIGKILLs every task and srun exits 137 ("forcing job termination";
# FAKE_SRUN_IGNORE_TERM=1: a wedged srun that ignores it); a SIGUSR1 (the stand-in scancel of the step) SIGKILLs every
# task. Its status: the first failing task's (128 + n for a signal), else 0; whatever is left of the tasks is killed when
# the step ends. FAKE_SRUN_DELAY: seconds before the tasks start; FAKE_SRUN_KILL="<task>:<seconds>": SIGKILL that task's
# launcher alone (the OOM killer) then.
# FAKE_SRUN_REPORT_<k>="<status> <host> <first failure> [<ranks>]": task k runs no launcher; it writes that report to
# $DM_NODE_STATUS/node<k> and exits as node_ranks.sh would (0 for 0 and 75, else the status): train_arm.sh's
# aggregation of given reports.
# Node-local file systems: FAKE_SRUN_LOCAL_VARS names variables whose paths are node-local; for task k > 0 each gets the
# prefix FAKE_SRUN_LOCAL_ROOT/n<k> (task 0 runs on the "batch host": its paths are left as they are). Every call and
# event goes to $FAKE_CALLS ("srun <args>", "srun: ...", "srun exit <status> tasks <statuses>"); its pid to
# $FAKE_DIR/srun.pid.
FAKE_SRUN = r"""
import os, signal, subprocess, sys, time
ARGS = sys.argv[1:]


def log(msg):
    if os.environ.get("FAKE_CALLS"):
        with open(os.environ["FAKE_CALLS"], "a") as f:
            f.write(msg + "\n")


log("srun " + " ".join(ARGS))
nodes = ntasks = None
i = 0
while i < len(ARGS) and ARGS[i].startswith("-"):
    a = ARGS[i]
    i += 1
    if a.startswith("--nodes="):
        nodes = int(a.split("=", 1)[1])
    elif a.startswith("--ntasks="):
        ntasks = int(a.split("=", 1)[1])
cmd = ARGS[i:]
n = ntasks or nodes or 1
with open(os.path.join(os.environ.get("FAKE_DIR", "."), "srun.pid"), "w") as f:
    f.write(str(os.getpid()))
hosts = [h for h in os.environ.get("SLURM_JOB_NODELIST", "").split(",") if h]
hosts += ["fakenode%d" % k for k in range(len(hosts), n)]
kill_wait = float(os.environ.get("FAKE_SRUN_KILL_WAIT") or 120)
kt = os.environ.get("FAKE_SRUN_KILL", "")
ktask, kdelay = (int(kt.split(":")[0]), float(kt.split(":")[1])) if kt else (None, None)
root, local_vars = os.environ.get("FAKE_SRUN_LOCAL_ROOT", ""), os.environ.get("FAKE_SRUN_LOCAL_VARS", "").split()
step = os.environ.get("FAKE_SRUN_STEP", "7")
procs = []


def kill_all(sig):
    for p in procs:
        try:
            os.killpg(p.pid, sig)
        except OSError:
            pass


def term_tasks():
    for p in procs:
        try:
            os.kill(p.pid, signal.SIGTERM)
        except OSError:
            pass


def on_term(signum, frame):
    if os.environ.get("FAKE_SRUN_IGNORE_TERM") == "1":
        log("srun: SIGTERM ignored")
        return
    log("srun: SIGTERM: forcing job termination")
    kill_all(signal.SIGKILL)
    os._exit(137)


def on_usr1(signum, frame):
    log("srun: the step was cancelled: SIGKILL to every task")
    kill_all(signal.SIGKILL)


signal.signal(signal.SIGTERM, on_term)
signal.signal(signal.SIGUSR1, on_usr1)
time.sleep(float(os.environ.get("FAKE_SRUN_DELAY") or 0))
for k in range(n):
    env = dict(os.environ, SLURM_NODEID=str(k), SLURM_PROCID=str(k), SLURM_LOCALID="0", SLURMD_NODENAME=hosts[k],
               SLURM_STEP_ID=step, SLURM_STEPID=step, SLURM_NTASKS=str(n), SLURM_STEP_NUM_NODES=str(n))
    if k > 0 and root:
        for v in local_vars:
            if env.get(v):
                env[v] = "%s/n%d%s" % (root, k, env[v])
    task, rep = cmd, os.environ.get("FAKE_SRUN_REPORT_%d" % k)
    if rep is not None:
        st = (rep.split() or ["1"])[0]
        task = ["bash", "-c", 'printf "%s\\n" "$1" > "$DM_NODE_STATUS/node$SLURM_NODEID"; exit "$2"', "report", rep,
                "0" if st in ("0", "75") else st]
    procs.append(subprocess.Popen(task, env=env, start_new_session=True))
t0, first, term_at, rcs = time.time(), None, None, [None] * n
while any(r is None for r in rcs):
    for k, p in enumerate(procs):
        if rcs[k] is None and p.poll() is not None:
            rcs[k] = p.returncode if p.returncode >= 0 else 128 - p.returncode
            if rcs[k] != 0 and first is None:
                first = rcs[k]
                log("srun: task %d (%s) exited %d: terminating the step" % (k, hosts[k], rcs[k]))
                term_tasks()
                term_at = time.time() + kill_wait
    if ktask is not None and time.time() - t0 >= kdelay:
        log("srun: SIGKILL to the launcher of task %d alone" % ktask)
        try:
            os.kill(procs[ktask].pid, signal.SIGKILL)
        except OSError:
            pass
        ktask = None
    if term_at is not None and time.time() >= term_at:
        log("srun: KillWait is over: SIGKILL to every task")
        kill_all(signal.SIGKILL)
        term_at = None
    time.sleep(0.05)
kill_all(signal.SIGKILL)
log("srun exit %d tasks %s" % (first or 0, " ".join(map(str, rcs))))
sys.exit(first or 0)
"""
FAKE_SRUN_SH = r"""#!/bin/bash
exec "$FAKE_REAL_PY" -I "$FAKE_DIR/srun.py" "$@"
"""
# scontrol: `show hostnames [list]` prints the comma-separated list (else SLURM_JOB_NODELIST) one name per line; any
# other call is only logged
FAKE_SCONTROL = r"""#!/bin/bash
echo "scontrol $*" >> "$FAKE_CALLS"
if [ "${1:-}" = show ] && [ "${2:-}" = hostnames ]; then
  l=${3-${SLURM_JOB_NODELIST:-}}; [ -n "$l" ] || exit 1
  tr ',' '\n' <<< "$l"
fi
exit 0
"""
# scancel: logged; a <job>.<step> argument cancels the stand-in srun's step (SIGUSR1 to it: its tasks are SIGKILLed),
# unless FAKE_SCANCEL_NOOP=1 (a scancel that does nothing)
FAKE_SCANCEL = r"""#!/bin/bash
echo "scancel $*" >> "$FAKE_CALLS"
[ "${FAKE_SCANCEL_NOOP:-0}" = 1 ] && exit 0
for a in "$@"; do
  case $a in *[0-9].[0-9]*) [ -r "$FAKE_DIR/srun.pid" ] && kill -USR1 "$(cat "$FAKE_DIR/srun.pid")" 2>/dev/null ;; esac
done
exit 0
"""
# a setsid that forks (as util-linux setsid does when it leads a process group): the launcher's $! is then this script,
# with srun as its child
FAKE_SETSID_FORK = r"""#!/bin/bash
echo "setsid (forks) $*" >> "$FAKE_CALLS"
"$FAKE_REAL_SETSID" "$@" &
wait $!
"""


def fake_multinode_env(tmp_path, nnodes=2, job_id=4242, fork_setsid=False, **extra) -> dict:
    """fake_launch_env plus the multi-node stand-ins in <tmp_path>/fake/bin (srun, scontrol, scancel; a forking setsid
    with fork_setsid): job `job_id` of `nnodes` nodes fake-n0, fake-n1, ... with fake-n0 the batch host
    (SLURMD_NODENAME), 4 CPUs per node, no GPUs, the env launch, the shared stop flag <tmp_path>/stop.flag."""
    env = fake_launch_env(tmp_path)
    fake = tmp_path / "fake"
    files = [("srun.py", FAKE_SRUN), ("bin/srun", FAKE_SRUN_SH), ("bin/scontrol", FAKE_SCONTROL),
             ("bin/scancel", FAKE_SCANCEL)]
    if fork_setsid:
        files.append(("bin/setsid", FAKE_SETSID_FORK))
        env["FAKE_REAL_SETSID"] = shutil.which("setsid") or "/usr/bin/setsid"
    for name, text in files:
        (fake / name).write_text(text)
        (fake / name).chmod(0o755)
    env.update({"SLURM_JOB_ID": str(job_id), "SLURM_JOB_NUM_NODES": str(nnodes), "SLURMD_NODENAME": "fake-n0",
                "SLURM_JOB_NODELIST": ",".join(f"fake-n{k}" for k in range(nnodes)), "SLURM_CPUS_ON_NODE": "4",
                "DM_LAUNCH": "env", "FAKE_STOP_FLAG": str(tmp_path / "stop.flag")})
    env.update({k: str(v) for k, v in extra.items()})
    return env


def hold_lock(path, seconds: float):
    """Another process holding an flock on `path` for `seconds` (a run lock held by another job); returns the Popen."""
    code = ("import fcntl, sys, time; f = open(sys.argv[1], 'a'); fcntl.flock(f, fcntl.LOCK_EX); print('HELD', flush=True); "
            "time.sleep(float(sys.argv[2]))")
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen([sys.executable, "-c", code, str(path), str(seconds)], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "HELD"
    return p


def dm_env(stale: bool) -> dict:
    """The Delta-Matching runtime switches scripts/campaign/train_arm.sh and arms.sh export for the fp8 arms. Every one
    of them is also the code default; the launcher exports them so a run's environment is explicit in its log."""
    e = {"DM_SHARED_X8": "1", "DM_FP8_FSDP_AG": "1", "DM_FP8_PRECOMPUTE": "1",
         "DM_FP8GLUE": "1", "DM_FP8GLUE_SCALING": "dynamic", "DM_FP8GLUE_SAVE_HT": "0"}
    if stale:
        e["DM_STALE_DOQ2S"] = "1"
    return e


# ---------------------------------------------------------------- the GDN:GQA arms on the tiny config
# The trainer arguments and switches of scripts/campaign/arms.sh + train_arm.sh, minus the M-tier geometry (the tiny
# yaml keeps its own) and the data and output paths (tiny_cmd passes them).
CE_RE = re.compile(r"ce_loss'?\s*[:=]\s*([0-9.]+|nan|inf)")
GLUE = {"DM_FP8_FSDP_AG": "1", "DM_FP8_PRECOMPUTE": "1", "DM_FP8GLUE": "1", "DM_FP8GLUE_SCALING": "dynamic",
        "DM_FP8GLUE_SAVE_HT": "0"}
COMMON = ["--fp8", "true", "--compile_mlp", "true", "--model.fused_swiglu", "false", "--model.fused_rmsnorm", "true",
          "--compile", "false", "--logging_steps", "1"]
DM_LM = "[gated_deltanet, gated_deltanet, gated_deltanet, dm]"
DM_MODES = {"dm_stale_hd128": "stale", "dm_matched_hd128": "matched", "dm_consistent_hd128": "stale", "dm_auxout_hd128": "aux_out",
            "dm_stale_hd256": "stale", "dm_matched_hd256": "matched"}


def arm_args(arm: str) -> tuple[list, dict]:
    """(trainer arguments, environment) of a GDN:GQA arm on tests/configs/tiny_hybrid.yaml. The head_dim-256 arms add
    --model.head_dim 256 (the tiny yaml is head_dim 128); the consistent-dO arm is the stale arm plus DM_DELTA_Q8=1, as
    arms.sh does."""
    if arm == "clean_hd128":
        return COMMON + ["--model.attn_impl", "sdpa", "--model.fused_rope", "true", "--fp8_scope", "ffn"], \
            {**GLUE, "DM_SDPA_BACKEND": "cudnn"}
    mode = DM_MODES[arm]
    args = COMMON + ["--model.layer_mixers", DM_LM, "--model.dm_mode", mode, "--model.dm_fused_producers", "true",
                     "--fp8_scope", "ffn_qkvo"]
    if arm.endswith("_hd256"):
        args += ["--model.head_dim", "256"]
    env = {**GLUE, **dm_env(stale=(mode != "matched"))}
    if arm == "dm_consistent_hd128":
        env["DM_DELTA_Q8"] = "1"
    return args, env


def dm_route(log: str) -> dict:
    """What the attention banners of a run's log say: "dm" = the dm_mode of the `[dm] DMAttention` banner, "mla_dm" =
    the delta of the `[mla_dm]` line, "mla_core" = the core of the `[mla]` banner; a key is absent with its banner."""
    out = {}
    for key, pat in (("dm", r"\[dm\] DMAttention:.*?\bdm_mode=(\w+)"), ("mla_dm", r"\[mla_dm\] .*?\bdelta=(\w+)"),
                     ("mla_core", r"\[mla\] \w+:.*?\bcore=([\w:]+)")):
        m = re.search(pat, log)
        if m:
            out[key] = m.group(1)
    return out


def tiny_cmd(name, run_root, root, tok_dir, steps=3, extra_args=(), config=None):
    """The trainer command line for tests/configs/tiny_hybrid.yaml (single process, one GPU); `config` picks another
    tiny yaml of tests/configs (tiny_mla.yaml, tiny_mamba2.yaml, tiny_kda.yaml)."""
    val = root / NEMO / "nemotron_rs_val.bin"
    dl = root / NEMO / "nemotron_rs_val.doc_lengths.npy"
    hf = run_root / "hf" / name
    cmd = [python(), "-m", "pretrain.train", "--config", str(config or REPO / "tests" / "configs" / "tiny_hybrid.yaml"),
           "--run_name", name, "--num_steps", str(steps), "--wandb", "false",
           "--data.paths", str(val), "--data.doc_lengths", str(dl), "--tokenizer_dir", str(tok_dir),
           "--hf_save_dir", str(hf), "--ckpt_root_dir", str(run_root / "ckpts"), "--log_root_dir", str(run_root / "logs")]
    return cmd + list(extra_args), hf


def train_tiny(name, run_root, root, tok_dir, gpu, extra_args=(), extra_env=None, steps=3, config=None):
    """Run the trainer on tests/configs/tiny_hybrid.yaml (or `config`) for a few steps; returns (rc, log text, export dir, seconds)."""
    cmd, hf = tiny_cmd(name, run_root, root, tok_dir, steps, extra_args, config=config)
    env = base_env(gpu)
    env.update(extra_env or {})
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=1800)
    log = p.stdout + "\n" + p.stderr
    (run_root / f"{name}.console.log").write_text(log)
    return p.returncode, log, hf, time.time() - t0


def read_metrics(run_root, name):
    """The trainer's local metrics log as a list of dicts (empty if absent); a line still being written is skipped."""
    import json
    p = run_root / "logs" / f"{name}.metrics.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out
