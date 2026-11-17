"""scripts/experiments/_run.sh as a chain of jobs runs it, CPU only: stand-in launchers (a copy of scripts/ whose
train_arm.sh / stage2_arm.sh / eval_arm*.sh write the real markers, or fail, as each test asks) and one invocation per
"job" over the same workspace.
  * phase A: each job first evaluates the trained runs whose evaluation is still owed; a finished training
    (TRAIN_COMPLETE) is never handed to its launcher again;
  * an evaluation that does not fit is deferred and the job goes on with the next run's training; the next job
    evaluates it first;
  * no training starts within DM_MIN_TRAIN_TIME of the trainer's deadline;
  * in a chain (DM_CHAIN_DIR) a configuration error (exit 2) gives the step up at once; every other failure is counted
    per run and step; progress (a new checkpoint, new evaluation
    markers) restarts the count, DM_RUN_MAX_FAILS jobs in a row with a failure without progress give the step up for
    the chain (a failure again in the same job, job.sh's in-place retry, counts once; a requeue is another job);
    success clears it; outside a chain a failure is final for the invocation;
  * a launcher's exit 3 (another job holds the run's lock) lists the run as owned;
  * the exit status: 0 all done, 75 work remains, 1 in a chain when a failure is retried and nothing of the job
    succeeded (job.sh then retries in place and keeps the next job off the node(s) it failed on), otherwise 4 in a
    chain and 1 outside
    one."""
import shutil
import subprocess
import time

import pytest

from _helpers import REPO, dry_env

R1, R2, R3 = ("Q35M_clean_sdpa_ffn8_hd128_11785_muon", "Q35M_dm_stale_ffn8_qkvo8_hd128_11785_muon",
              "Q35M_dm_matched_ffn8_qkvo8_hd128_11785_muon")   # stage1_muon.sh's runs, in plan order
ARMS = {R1: "muon_clean_hd128", R2: "muon_dm_stale_hd128", R3: "muon_dm_matched_hd128"}

# the fate of <arm>:<seed> (STUB_FATES "<arm>:<seed>=<fate> ...", default complete): complete (TRAIN_COMPLETE), fail
# (exit 1), progress_fail (one more complete checkpoint, then exit 1), stop (75), owned (3), config (2)
STUB_TRAIN = r"""#!/bin/bash
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/env.sh"; source "$HERE/arms.sh"
ARM=$1; SEED=$2; T=""; [ "$(basename "$0")" = stage2_arm.sh ] && T=_s2_64k
RN=$(TAG="$T${TAG:-}" dm_run_name "$ARM" "$SEED"); SW=$DM_WORKSPACE/qwen35_m
echo "train $(basename "$0") $ARM $SEED" >> "$STUB_LOG"
fate=complete; for kv in ${STUB_FATES:-}; do [ "${kv%%=*}" = "$ARM:$SEED" ] && fate=${kv#*=}; done
case $fate in
  complete) mkdir -p "$SW/hf/$RN"; date > "$SW/hf/$RN/TRAIN_COMPLETE"; exit 0 ;;
  fail) exit 1 ;;
  progress_fail) n=$(dm_ckpt_step "$SW/ckpts/$RN"); n=$(( n < 0 ? 100 : n + 100 ))
                 mkdir -p "$SW/ckpts/$RN/checkpoint-$n"; : > "$SW/ckpts/$RN/checkpoint-$n/.complete"; exit 1 ;;
  stop) exit 75 ;;
  owned) exit 3 ;;
  config) exit 2 ;;
esac
"""
# the fate of a run's evaluation (STUB_EVAL_FATES "<run>=<fate> ...", default complete): complete (the stage's final
# marker), fail (exit 1), progress_fail (one more benchmark marker, then exit 1), owned (3)
STUB_EVAL = r"""#!/bin/bash
RN=$1; OUT=$DM_WORKSPACE/qwen35_m/eval/$RN; M=EVAL_COMPLETE; [ "$(basename "$0")" = eval_arm_64k.sh ] && M=EVAL64K_COMPLETE
echo "eval $(basename "$0") $RN" >> "$STUB_LOG"
fate=complete; for kv in ${STUB_EVAL_FATES:-}; do [ "${kv%%=*}" = "$RN" ] && fate=${kv#*=}; done
case $fate in
  complete) mkdir -p "$OUT"; date > "$OUT/$M"; exit 0 ;;
  fail) exit 1 ;;
  progress_fail) mkdir -p "$OUT/b"; : > "$OUT/b/b$(date +%s%N)_results.complete"; exit 1 ;;
  owned) exit 3 ;;
esac
"""


@pytest.fixture
def drive(tmp_path):
    """drive(*args, exp=..., **env) -> (exit status, output, the launcher calls of this "job")."""
    root = tmp_path / "repo"
    shutil.copytree(REPO / "scripts", root / "scripts")
    for name, text in (("train_arm.sh", STUB_TRAIN), ("stage2_arm.sh", STUB_TRAIN), ("eval_arm.sh", STUB_EVAL),
                       ("eval_arm_64k.sh", STUB_EVAL)):
        (root / "scripts" / "campaign" / name).write_text(text)
    log = tmp_path / "calls.log"

    def run(*args, exp="stage1_muon", **extra):
        log.unlink(missing_ok=True)
        p = subprocess.run(["bash", str(root / "scripts" / "experiments" / f"{exp}.sh"), *args], capture_output=True,
                           text=True, env=dry_env(tmp_path, STUB_LOG=log, **extra), timeout=300)
        return p.returncode, p.stdout + p.stderr, (log.read_text().splitlines() if log.exists() else [])
    return run


def _mark(tmp_path, rn, trained=True, evaluated=False, marker="EVAL_COMPLETE"):
    if trained:
        (tmp_path / "qwen35_m" / "hf" / rn).mkdir(parents=True, exist_ok=True)
        (tmp_path / "qwen35_m" / "hf" / rn / "TRAIN_COMPLETE").write_text("done\n")
    if evaluated:
        (tmp_path / "qwen35_m" / "eval" / rn).mkdir(parents=True, exist_ok=True)
        (tmp_path / "qwen35_m" / "eval" / rn / marker).write_text("done\n")


def _t(rn):
    return f"train train_arm.sh {ARMS[rn]} 11785"


def _e(rn):
    return f"eval eval_arm.sh {rn}"


def test_owed_evaluations_come_first_and_finished_trainings_are_skipped(tmp_path, drive):
    _mark(tmp_path, R2)
    _mark(tmp_path, R3)
    rc, out, calls = drive()
    assert rc == 0 and calls == [_e(R2), _e(R3), _t(R1), _e(R1)], (calls, out)
    rc, out, calls = drive()
    assert rc == 0 and calls == [] and "all 3 runs done" in out, (calls, out)   # nothing left: no launcher at all


def test_stage2_owes_the_64k_evaluation(tmp_path, drive):
    """Stage 2's evaluation marker is EVAL64K_COMPLETE: a stage-1 marker in the run's directory does not count."""
    rn = "Q35M_clean_sdpa_ffn8_hd128_11785_s2_64k"
    _mark(tmp_path, rn, evaluated=True, marker="EVAL_COMPLETE")
    rc, out, calls = drive("clean_hd128", exp="stage2_context", SEEDS="11785")
    assert rc == 0 and calls == [f"eval eval_arm_64k.sh {rn}"], (calls, out)


def test_a_deferred_evaluation_lets_the_job_train_the_next_runs(tmp_path, drive):
    """The evaluations do not fit (DM_EVAL_BUDGET): the job trains all three runs, exits 75, and the next job evaluates
    them first, without training anything."""
    now = int(time.time())
    rc, out, calls = drive(DM_STOP_BEFORE=now + 3600, DM_JOB_END=now + 4200, DM_EVAL_BUDGET=99999,
                           SLURM_JOB_START_TIME=now - 3600)
    assert rc == 75 and calls == [_t(R1), _t(R2), _t(R3)], (calls, out)
    assert out.count("no time left in this job for its evaluation") == 3 and f"evaluation deferred to the next job: {R1} {R2} {R3}" in out, out
    rc, out, calls = drive()
    assert rc == 0 and calls == [_e(R1), _e(R2), _e(R3)], (calls, out)


@pytest.mark.parametrize("extra,fits", [
    ({"DM_STOP_BEFORE": 2000}, True),                                  # the job's end: DM_STOP_BEFORE + 600 = 2600 s left
    ({"DM_STOP_BEFORE": 2000, "DM_STOP_MARGIN": 100}, False),          # 2100 s left
    ({"DM_STOP_BEFORE": 2000, "DM_JOB_END": 2100}, False),             # DM_JOB_END wins
    ({"DM_STOP_BEFORE": 2000, "DM_JOB_END": 2100, "SLURM_JOB_START_TIME": 0}, True),   # the job started just now
])
def test_an_evaluation_starts_when_it_fits_before_the_jobs_end(tmp_path, drive, extra, fits):
    """The budget (2400 s for stage 1) is measured to the job's end, DM_JOB_END, else DM_STOP_BEFORE + DM_STOP_MARGIN."""
    now = int(time.time())
    _mark(tmp_path, R1)
    env = {k: (v if k == "DM_STOP_MARGIN" else now + v) for k, v in extra.items()}   # epochs, relative to now
    env.setdefault("SLURM_JOB_START_TIME", now - 3600)
    rc, out, calls = drive("muon_clean_hd128", **env)
    assert (calls == [_e(R1)]) is fits and rc == (0 if fits else 75), (extra, calls, out)


def test_no_training_starts_close_to_the_deadline(tmp_path, drive):
    """Less than DM_MIN_TRAIN_TIME (600 s) before DM_STOP_BEFORE no training starts (it would pay the start-up and save
    after one step); the owed evaluation that fits still runs; exit 75."""
    now = int(time.time())
    _mark(tmp_path, R1)
    rc, out, calls = drive(DM_STOP_BEFORE=now + 300, DM_JOB_END=now + 6000)
    assert rc == 75 and calls == [_e(R1)], (calls, out)
    assert f"training not started (too close to the deadline): {R2} {R3}" in out, out
    rc, out, calls = drive(DM_STOP_BEFORE=now + 300, DM_JOB_END=now + 6000, DM_MIN_TRAIN_TIME=100)
    assert rc == 0 and calls == [_t(R2), _e(R2), _t(R3), _e(R3)], (calls, out)


def test_a_training_failure_in_a_chain_is_retried_then_given_up(tmp_path, drive):
    """The other runs of the job succeed: the failure is retried by the next job (75); failing again there without
    progress gives it up (DM_RUN_MAX_FAILS=2) and, nothing else being left, the chain stops (4)."""
    chain = tmp_path / "chain"
    fate = {"DM_CHAIN_DIR": chain, "STUB_FATES": "muon_dm_stale_hd128:11785=fail"}
    rc, out, calls = drive(**fate)
    assert rc == 75 and calls == [_t(R1), _e(R1), _t(R2), _t(R3), _e(R3)], (calls, out)
    assert f"failed, retried (in place by scripts/slurm/job.sh, or by the next job): {R2}:train" in out, out
    assert (chain / "runfails" / f"{R2}.train").read_text() == "-1 1 -\n"   # no checkpoint, 1 failure, no Slurm job
    rc, out, calls = drive(**fate)
    assert rc == 4 and calls == [_t(R2)] and f"given up for this chain (DM_RUN_MAX_FAILS): {R2}:train" in out, (calls, out)
    rc, out, calls = drive(**fate)
    assert rc == 4 and calls == [], (calls, out)   # given up: not even started again
    # a new submission clears runfails/ (scripts/slurm/submit.sh): the run is tried again, and a success clears its counter
    shutil.rmtree(chain / "runfails")
    rc, out, calls = drive(DM_CHAIN_DIR=chain)
    assert rc == 0 and calls == [_t(R2), _e(R2)] and not (chain / "runfails" / f"{R2}.train").exists(), (calls, out)


@pytest.mark.parametrize("chain", [True, False])
def test_a_job_where_nothing_succeeded_is_a_failed_job_in_a_chain(tmp_path, drive, chain):
    """R1 was done by an earlier job, R2 fails, R3 is owned by another job: nothing of this job succeeded, so in a chain
    it exits 1, a failure (scripts/slurm/job.sh retries it in place, then keeps the next job off the node(s) it failed
    on, which a 75
    never does); outside a chain 1 as before."""
    _mark(tmp_path, R1, evaluated=True)
    extra = {"DM_CHAIN_DIR": tmp_path / "chain"} if chain else {}
    rc, out, calls = drive(STUB_FATES="muon_dm_stale_hd128:11785=fail muon_dm_matched_hd128:11785=owned", **extra)
    assert rc == 1 and calls == [_t(R2), _t(R3)], (calls, out)
    if chain:
        assert "nothing of this job succeeded and a failure is retried: exit 1" in out, out
    else:
        assert "incomplete: nothing left that this command can do" in out, out


def test_progress_restarts_the_failure_count(tmp_path, drive):
    """Each failing attempt leaves a newer complete checkpoint: the run is never given up (each job fails, with progress:
    exit 1, job.sh's in-place retry resumes it at once); DM_RUN_MAX_FAILS=3 allows two failures without progress."""
    chain = tmp_path / "chain"
    for _ in range(3):
        rc, out, calls = drive("muon_dm_stale_hd128", DM_CHAIN_DIR=chain, STUB_FATES="muon_dm_stale_hd128:11785=progress_fail")
        assert rc == 1 and calls == [_t(R2)], (calls, out)
    assert (chain / "runfails" / f"{R2}.train").read_text() == "300 1 -\n"
    fate = {"DM_CHAIN_DIR": chain, "STUB_FATES": "muon_dm_stale_hd128:11785=fail", "DM_RUN_MAX_FAILS": 3}
    assert drive("muon_dm_stale_hd128", **fate)[0] == 1      # 2 of 3
    assert drive("muon_dm_stale_hd128", **fate)[0] == 4      # 3 of 3: given up


def test_a_failure_counts_once_per_job(tmp_path, drive):
    """job.sh retries a failed job in place (the same SLURM_JOB_ID): a failure again in that job is not counted again, so
    the run is not given up before the next job, which runs on another node, has tried it. A requeue (Slurm runs the
    job again, SLURM_RESTART_COUNT) and the next job each count."""
    chain = tmp_path / "chain"
    fate = {"DM_CHAIN_DIR": chain, "STUB_FATES": "muon_dm_stale_hd128:11785=fail"}
    counter = chain / "runfails" / f"{R2}.train"
    rc, out, calls = drive("muon_dm_stale_hd128", SLURM_JOB_ID=500, **fate)
    assert rc == 1 and calls == [_t(R2)] and counter.read_text() == "-1 1 500.0\n", (calls, out)
    rc, out, calls = drive("muon_dm_stale_hd128", SLURM_JOB_ID=500, **fate)   # the in-place retry
    assert rc == 1 and calls == [_t(R2)] and counter.read_text() == "-1 1 500.0\n", (calls, out)
    rc, out, calls = drive("muon_dm_stale_hd128", SLURM_JOB_ID=501, **fate)   # the next job: 2 of 2, given up
    assert rc == 4 and calls == [_t(R2)] and counter.read_text() == "-1 2 501.0\n", (calls, out)
    counter.unlink()
    assert drive("muon_dm_stale_hd128", SLURM_JOB_ID=600, **fate)[0] == 1
    rc, out, calls = drive("muon_dm_stale_hd128", SLURM_JOB_ID=600, SLURM_RESTART_COUNT=1, **fate)   # requeued: counts
    assert rc == 4 and counter.read_text() == "-1 2 600.1\n", (calls, out)


def test_an_evaluation_failure_in_a_chain_is_retried_then_given_up(tmp_path, drive):
    """Failed evaluations that add markers (benchmarks completed) keep the count at 1; the first one without a new
    marker after them is the second failure without progress: given up (DM_RUN_MAX_FAILS=2). Nothing else of these
    jobs succeeds: each failure is a failed job (1), the last one gives up (4)."""
    chain = tmp_path / "chain"
    _mark(tmp_path, R1)
    for fate, want in (("progress_fail", 1), ("progress_fail", 1), ("fail", 4)):
        rc, out, calls = drive("muon_clean_hd128", DM_CHAIN_DIR=chain, STUB_EVAL_FATES=f"{R1}={fate}")
        assert rc == want and calls == [_e(R1)], (fate, calls, out)
    rc, out, calls = drive("muon_clean_hd128", DM_CHAIN_DIR=chain)
    assert rc == 4 and calls == [] and f"{R1}:eval" in out, (calls, out)


def test_a_configuration_error_is_given_up_at_once(tmp_path, drive):
    """Exit 2 (a refused resume, missing data, a bad knob) is the same in every job: in a chain that training is given
    up at once (no in-place retry, no next job spent on it) while the other runs go on; the next job skips it; a new
    submission (runfails/ cleared) tries it again. Outside a chain it is a plain failure."""
    chain = tmp_path / "chain"
    fate = {"DM_CHAIN_DIR": chain, "STUB_FATES": "muon_dm_stale_hd128:11785=config"}
    rc, out, calls = drive(**fate)
    assert rc == 4 and calls == [_t(R1), _e(R1), _t(R2), _t(R3), _e(R3)], (calls, out)
    assert "exit 2: a configuration error" in out and f"given up for this chain (DM_RUN_MAX_FAILS): {R2}:train" in out, out
    rc, out, calls = drive(**fate)
    assert rc == 4 and calls == [], (calls, out)
    shutil.rmtree(chain / "runfails")
    rc, out, calls = drive("muon_dm_stale_hd128", **fate)
    assert rc == 4 and calls == [_t(R2)], (calls, out)
    rc, out, calls = drive("muon_dm_stale_hd128", STUB_FATES="muon_dm_stale_hd128:11785=config")
    assert rc == 1 and calls == [_t(R2)] and f"failed: {R2}:train" in out, (calls, out)


@pytest.mark.parametrize("chain", [True, False])
def test_a_run_owned_by_another_job_is_listed_and_skipped(tmp_path, drive, chain):
    extra = {"DM_CHAIN_DIR": tmp_path / "chain"} if chain else {}
    rc, out, calls = drive(STUB_FATES="muon_dm_stale_hd128:11785=owned", **extra)
    assert rc == (4 if chain else 1) and calls == [_t(R1), _e(R1), _t(R2), _t(R3), _e(R3)], (calls, out)
    assert f"owned by another job (its lock was busy): {R2}:train" in out and not (tmp_path / "chain" / "runfails").exists(), out


def test_outside_a_chain_a_failure_ends_with_1(tmp_path, drive):
    rc, out, calls = drive(STUB_FATES="muon_dm_stale_hd128:11785=fail")
    assert rc == 1 and calls == [_t(R1), _e(R1), _t(R2), _t(R3), _e(R3)] and f"failed: {R2}:train" in out, (calls, out)


def test_an_early_stop_ends_the_job_with_75(tmp_path, drive):
    rc, out, calls = drive(DM_CHAIN_DIR=tmp_path / "chain", STUB_FATES="muon_dm_stale_hd128:11785=stop")
    assert rc == 75 and calls == [_t(R1), _e(R1), _t(R2)], (calls, out)
    assert f"resumes in the next job): {R2}\n" in out and f"not reached in this job: {R3}\n" in out, out


def test_train_only_and_bad_knobs(tmp_path, drive):
    _mark(tmp_path, R1)
    rc, out, calls = drive(EVAL=0)
    assert rc == 0 and calls == [_t(R2), _t(R3)], (calls, out)
    for knob in ({"DM_STOP_BEFORE": "soon"}, {"DM_RUN_MAX_FAILS": 0}, {"DM_EVAL_BUDGET": "1h"}):
        rc, out, calls = drive(**knob)
        assert rc == 2 and "FATAL" in out and calls == [], (knob, out)
