"""scripts/campaign/train_arm.sh end to end on the CPU, with the stand-ins of tests/_helpers.py (fake_launch_env: a
stand-in interpreter running stand-in ranks, export check and val CE): what the launcher decides in a chain of jobs,
without a GPU or a trainer.
  * a finished run exits 0 before any check (data, kernel build, imports);
  * the end of a run: a whole export completes the run whatever the trainer's status, provided this launch wrote it;
    config.json is patched atomically (config.train.json keeps the trained config, a second pass changes nothing),
    TRAIN_COMPLETE records the trained steps, the progress line follows, and only then the informational val CE runs
    (a SIGTERM there changes nothing); an exit 0 without a whole export is a failure (1);
  * the rank logs are appended to, one header per launch, and the failed ranks' last 30 lines are printed; a failed
    rank gets its peers killed after DM_RANK_GRACE;
  * the run lock: another live job's lock -> exit 3; the lock of this job (requeued, retried in place), of a job that is
    ending (COMPLETING: Slurm starts a chain's next job while the previous one's ranks still save) or has ended, or of
    a holder squeue cannot tell about is waited for, its holder judged again while waiting; still held after
    DM_LOCK_WAIT -> exit 1 (a failure to retry, never the chain-stopping 3);
  * under torchrun, a clean early stop is a 75 even when an earlier launch left a whole export, and a refused resume
    (the trainer's "FATAL: RESUME REFUSED") is a 2;
  * in a chain the allow knobs (DM_ALLOW_RUN_CHANGE, DM_ALLOW_KERNEL_CHANGE) are dropped once the submission recorded
    the change in the run;
  * DM_GIT_REV tells two dirty trees apart, untracked files included; SAVE_IV gives every arm about an hour of steps
    between checkpoints."""
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time

import pytest

from _helpers import REPO, fake_launch_env, read_calls, hold_lock

RN = "Q35M_clean_sdpa_ffn8_hd128_11785"            # clean_hd128, seed 11785
DM_CFG = {"attn_impl": "sdpa", "layer_mixers": ["gated_deltanet", "gated_deltanet", "gated_deltanet", "dm"], "mixer": "gqa",
          "dm_mode": "matched", "dm_fused_producers": True}   # what a Delta-Matching arm exports (before the patch)
LAUNCHER = REPO / "scripts" / "campaign" / "train_arm.sh"


def _train(tmp_path, arm="clean_hd128", seed="11785", timeout=120, **extra):
    p = subprocess.run(["bash", str(LAUNCHER), arm, seed], cwd=str(REPO), env=fake_launch_env(tmp_path, **extra),
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def _hf(tmp_path, rn=RN):
    return tmp_path / "qwen35_m" / "hf" / rn


def _ranks_started(tmp_path):
    return [c for c in read_calls(tmp_path) if c.startswith("rank ")]


def _whole_export(d, cfg):
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.safetensors").write_text("WHOLE")
    (d / "config.json").write_text(json.dumps(cfg))


def test_a_finished_run_exits_before_any_check(tmp_path):
    """TRAIN_COMPLETE: exit 0 at once, before the data check, the kernel check of a Delta-Matching arm and the import
    check (the stand-in interpreter is never called)."""
    rn = "Q35M_dm_matched_ffn8_qkvo8_hd128_11785"
    _hf(tmp_path, rn).mkdir(parents=True)
    (_hf(tmp_path, rn) / "TRAIN_COMPLETE").write_text("done\n")
    env = fake_launch_env(tmp_path)
    shutil.rmtree(tmp_path / "data")   # the data check would fail
    p = subprocess.run(["bash", str(LAUNCHER), "dm_matched_hd128", "11785"], cwd=str(REPO), env=env, capture_output=True,
                       text=True, timeout=60)
    out = p.stdout + p.stderr
    assert p.returncode == 0 and "already complete" in out, out
    assert read_calls(tmp_path) == [], read_calls(tmp_path)


def test_a_complete_run_is_patched_marked_and_reported_before_the_val_ce(tmp_path):
    progress = tmp_path / "chain" / "progress"
    progress.parent.mkdir()
    rc, out = _train(tmp_path, FAKE_CONFIG=json.dumps(DM_CFG), NUM_STEPS=100, DM_PROGRESS_FILE=progress,
                     SLURM_JOB_ID=4242, DM_VALCE_PORT=31234)
    assert rc == 0, out
    hf = _hf(tmp_path)
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d steps=100 num_steps=100\n", (hf / "TRAIN_COMPLETE").read_text())
    c, t = json.loads((hf / "config.json").read_text()), json.loads((hf / "config.train.json").read_text())
    assert c["layer_mixers"][-1] == "gqa" and "dm_mode" not in c and "dm_fused_producers" not in c, c
    assert t == DM_CFG, t   # the trained config, as the trainer exported it
    lines = progress.read_text().splitlines()
    assert len(lines) == 1 and re.fullmatch(rf"\d+ 4242 train_complete {RN}", lines[0]), lines
    # the val CE ran last: the marker and the progress line existed when it started, on the job's port
    assert "valce port=31234 marker=1 progress=1" in read_calls(tmp_path), read_calls(tmp_path)
    assert "memmap-ppl RESULT" in out and len(_ranks_started(tmp_path)) == 2, out
    assert re.search(r"LAUNCH \S+ .*save_interval=1000 ", out), out   # SAVE_IV of the 1.665 B arms
    assert (tmp_path / "qwen35_m" / "ckpts" / f"{RN}.lock.owner").read_text().startswith("job=4242 host=")


def test_stop_at_step_records_the_trained_steps(tmp_path):
    rc, out = _train(tmp_path, NUM_STEPS=100, STOP_AT_STEP=40)
    assert rc == 0 and (_hf(tmp_path) / "TRAIN_COMPLETE").read_text().endswith(" steps=40 num_steps=100\n"), out


def test_a_crash_after_a_whole_export_completes_the_run(tmp_path):
    """The trainer wrote a whole export, then failed at teardown: the run is complete (it must not train again in every
    job)."""
    rc, out = _train(tmp_path, FAKE_MODE="crash_after_export")
    assert rc == 0 and (_hf(tmp_path) / "TRAIN_COMPLETE").exists(), out
    assert "the trainer exited 1 after writing a whole export" in out, out


def test_exit_0_without_a_whole_export_is_a_retryable_failure(tmp_path):
    rc, out = _train(tmp_path, FAKE_MODE="export_incomplete")
    assert rc == 1 and "FATAL: the trainer finished but the export is incomplete" in out, out
    assert not (_hf(tmp_path) / "TRAIN_COMPLETE").exists() and not (_hf(tmp_path) / "config.train.json").exists(), out


def test_an_export_left_by_an_earlier_launch_never_completes_a_failed_run(tmp_path):
    """A whole export from an earlier launch (an earlier STOP_AT_STEP, say) and a trainer that fails now: not complete."""
    _whole_export(_hf(tmp_path), {"attn_impl": "sdpa"})
    rc, out = _train(tmp_path, FAKE_MODE="fail")
    assert rc == 1 and not (_hf(tmp_path) / "TRAIN_COMPLETE").exists(), out
    assert "is from an earlier launch" in out, out


def test_the_end_of_run_tail_is_idempotent(tmp_path):
    """The tail runs again over an export it already patched (TRAIN_COMPLETE lost, the trainer exits 0 without
    re-exporting): config.train.json keeps the trained config, never the patched one."""
    rc, out = _train(tmp_path, FAKE_CONFIG=json.dumps(DM_CFG))
    assert rc == 0, out
    hf = _hf(tmp_path)
    patched = (hf / "config.json").read_text()
    (hf / "TRAIN_COMPLETE").unlink()
    rc, out = _train(tmp_path, FAKE_MODE="noexport")
    assert rc == 0 and (hf / "TRAIN_COMPLETE").exists() and "export already patched" in out, out
    assert json.loads((hf / "config.train.json").read_text()) == DM_CFG and (hf / "config.json").read_text() == patched


def test_rank_logs_are_appended_and_the_failed_ranks_tails_printed(tmp_path):
    rc, out = _train(tmp_path, FAKE_MODE="fail", SLURM_JOB_ID=77)
    assert rc == 1 and "Traceback: RuntimeError: boom" in out, out
    assert "--- rank 0 failed (exit 1)" in out and "--- rank 1 failed (exit 1)" in out, out
    rc, out = _train(tmp_path)
    assert rc == 0, out
    for r in (0, 1):
        log = (tmp_path / "logs" / f"{RN}.rank{r}.log").read_text()
        heads = re.findall(rf"^===== launch \S+ \S+ job (\S+) host \S+ git fake: rank {r} of 2 of {RN} =====$", log, re.M)
        assert heads == ["77", "none"] and "Traceback: RuntimeError: boom" in log and log.count("fake rank") == 2, log


def test_a_failed_rank_gets_its_peers_killed_and_only_its_own_tail_printed(tmp_path):
    """Rank 1 crashes, ranks 0 and 2 hang in a collective: they are killed DM_RANK_GRACE seconds later, the launcher
    exits with rank 1's status (not 137) and prints rank 1's last 30 lines, not the killed ranks'."""
    t0 = time.time()
    rc, out = _train(tmp_path, FAKE_MODE="peer_fail", DM_NPROC=3, DM_RANK_GRACE=1)
    assert rc == 1 and time.time() - t0 < 30, out
    assert "2 rank(s) still running 1 s after rank 1 failed (exit 1): SIGKILL" in out, out
    tail = out.split("--- rank 1 failed (exit 1)", 1)[1]
    assert "rank 1 traceback line 40" in tail and "rank 1 traceback line 11\n" in tail and "rank 1 traceback line 10\n" not in tail, tail
    assert "--- rank 0 failed" not in out and "--- rank 2 failed" not in out, out


def test_an_early_stop_exits_75(tmp_path):
    rc, out = _train(tmp_path, FAKE_MODE="stop")
    assert rc == 75 and "stopped early with a complete checkpoint" in out and not (_hf(tmp_path) / "TRAIN_COMPLETE").exists(), out


# a stand-in torchrun for the training launch: the trainer stopped early (a complete checkpoint, "Stopping early" in the
# trainer log) and torchrun reports the rank's 75 as its own 1
TORCHRUN_STOP = r"""#!/bin/bash
RUN=""; CK=""; LOGS=""
while [ $# -gt 0 ]; do case "$1" in --run_name) RUN=$2; shift ;; --ckpt_root_dir) CK=$2; shift ;; --log_root_dir) LOGS=$2; shift ;; esac; shift; done
mkdir -p "$CK/$RUN/checkpoint-5" && : > "$CK/$RUN/checkpoint-5/.complete"
echo "Stopping early (deadline) at step 5: complete checkpoint saved; relaunch to resume" >> "$LOGS/$RUN.log"
exit 1
"""


@pytest.mark.parametrize("earlier_export", [False, True])
def test_a_torchrun_early_stop_is_75_whatever_export_an_earlier_launch_left(tmp_path, earlier_export):
    """Under torchrun a rank's 75 arrives as torchrun's 1 and is read back from this launch's part of the trainer log.
    A whole export an earlier launch left (an earlier STOP_AT_STEP end whose marker was removed) is not this launch's
    export: the stop stays a 75, not a failure."""
    tr = tmp_path / "torchrun_stop.sh"
    tr.write_text(TORCHRUN_STOP)
    tr.chmod(0o755)
    if earlier_export:
        _whole_export(_hf(tmp_path), {"attn_impl": "sdpa"})
    rc, out = _train(tmp_path, DM_LAUNCH="torchrun", DM_TORCHRUN=tr)
    assert rc == 75 and "stopped early with a complete checkpoint" in out, out
    assert not (_hf(tmp_path) / "TRAIN_COMPLETE").exists(), out


# a stand-in torchrun: the trainer refused the resume (every rank exits 2; torchrun reports 1) when FAKE_REFUSE=1, else a
# plain failure (1); with FAKE_OLD_REFUSAL=1 an earlier launch's refusal is already in the trainer log
TORCHRUN_REFUSE = r"""#!/bin/bash
RUN=""; LOGS=""
while [ $# -gt 0 ]; do case "$1" in --run_name) RUN=$2; shift ;; --log_root_dir) LOGS=$2; shift ;; esac; shift; done
[ "${FAKE_REFUSE:-0}" = 1 ] && echo "[rank 0] FATAL: RESUME REFUSED (exit 2: a configuration error, not retried): num_steps differs" >> "$LOGS/$RUN.log"
echo "torchrun failed" >> "$LOGS/$RUN.log"
exit 1
"""


@pytest.mark.parametrize("refuse,old,want", [(True, False, 2), (False, False, 1), (False, True, 1)])
def test_a_refused_resume_under_torchrun_is_a_configuration_error(tmp_path, refuse, old, want):
    """torchrun reports the ranks' exit 2 (a refused resume) as its own 1: the trainer's "FATAL: RESUME REFUSED" line in
    this launch's part of the log makes it a 2 (never retried); a refusal an earlier launch logged does not."""
    tr = tmp_path / "torchrun_refuse.sh"
    tr.write_text(TORCHRUN_REFUSE)
    tr.chmod(0o755)
    if old:
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_path / "logs" / f"{RN}.log").write_text("[rank 0] FATAL: RESUME REFUSED (exit 2: ...): an earlier launch\n")
    rc, out = _train(tmp_path, DM_LAUNCH="torchrun", DM_TORCHRUN=tr, FAKE_REFUSE=int(refuse))
    assert rc == want, out


# a stand-in torchrun that logs the allow knobs it was given, then writes a whole export (the run completes)
TORCHRUN_KNOBS = r"""#!/bin/bash
HF=""
while [ $# -gt 0 ]; do case "$1" in --hf_save_dir) HF=$2; shift ;; esac; shift; done
echo "knobs run=${DM_ALLOW_RUN_CHANGE:-unset} kernel=${DM_ALLOW_KERNEL_CHANGE:-unset}" >> "$FAKE_CALLS"
mkdir -p "$HF" && printf WHOLE > "$HF/model.safetensors" && echo '{"attn_impl": "sdpa"}' > "$HF/config.json"
"""


@pytest.mark.parametrize("chain,ident_new,ckpt_new,want", [
    (True, False, False, "knobs run=1 kernel=1"),          # nothing recorded since the submission: both kept
    (True, True, False, "knobs run=unset kernel=1"),       # the identity was recorded since: DM_ALLOW_RUN_CHANGE dropped
    (True, False, True, "knobs run=1 kernel=unset"),       # a checkpoint was saved since: DM_ALLOW_KERNEL_CHANGE dropped
    (True, True, True, "knobs run=unset kernel=unset"),
    (False, True, True, "knobs run=1 kernel=1"),           # outside a chain: the knobs are the caller's
])
def test_the_allow_knobs_are_one_shot_in_a_chain(tmp_path, chain, ident_new, ckpt_new, want):
    """A chain hands DM_ALLOW_RUN_CHANGE / DM_ALLOW_KERNEL_CHANGE to every later job: each is dropped once this
    submission (the last submit record of links) recorded the change in the run (run_identity.json, resp. the newest
    complete checkpoint, newer than the submission)."""
    tr = tmp_path / "torchrun_knobs.sh"
    tr.write_text(TORCHRUN_KNOBS)
    tr.chmod(0o755)
    now = time.time()
    ck = tmp_path / "qwen35_m" / "ckpts" / RN
    (ck / "checkpoint-100").mkdir(parents=True)
    (ck / "checkpoint-100" / ".complete").write_text("x")
    (ck / "run_identity.json").write_text("{}")
    os.utime(ck / "run_identity.json", (now, now) if ident_new else (now - 7200, now - 7200))
    os.utime(ck / "checkpoint-100" / ".complete", (now, now) if ckpt_new else (now - 7200, now - 7200))
    extra = {}
    if chain:
        cd = tmp_path / "chain"
        cd.mkdir()
        (cd / "links").write_text(f"{int(now) - 9000} 11 submit k=0\n{int(now) - 3600} 4242 submit k=0\n"
                                  f"{int(now) - 3500} 4242 start k=0 fails=0\n")
        extra["DM_CHAIN_DIR"] = cd
    rc, out = _train(tmp_path, DM_LAUNCH="torchrun", DM_TORCHRUN=tr, DM_ALLOW_RUN_CHANGE=1, DM_ALLOW_KERNEL_CHANGE=1, **extra)
    assert rc == 0, out
    assert [c for c in read_calls(tmp_path) if c.startswith("knobs ")] == [want], out
    assert ("DM_ALLOW_RUN_CHANGE dropped" in out) == (chain and ident_new), out
    assert ("DM_ALLOW_KERNEL_CHANGE dropped" in out) == (chain and ckpt_new), out


@pytest.mark.parametrize("owner,squeue", [("job=777", "RUNNING"), ("job=777", "SUSPENDED"),
                                          ("job=none host=elsewhere pid=1", "invalid")])   # a launch outside Slurm elsewhere
def test_the_lock_of_a_live_job_exits_3(tmp_path, owner, squeue):
    """Another job holds the run's lock and squeue says it runs: exit 3 at once, no rank started."""
    lock = tmp_path / "qwen35_m" / "ckpts" / f"{RN}.lock"
    holder = hold_lock(lock, 60)
    try:
        (lock.parent / f"{RN}.lock.owner").write_text(f"{owner} time=2026-11-08T10:00:00\n")
        t0 = time.time()
        rc, out = _train(tmp_path, FAKE_SQUEUE=squeue, SLURM_JOB_ID=4242, DM_LOCK_WAIT=60)
    finally:
        holder.kill(); holder.wait()
    assert rc == 3 and f"FATAL: {RN} is being trained by {owner}" in out and time.time() - t0 < 30, out
    assert _ranks_started(tmp_path) == [] and "waiting" not in out, out
    assert ("squeue -h -j 777 -o %T" in read_calls(tmp_path)) == owner.startswith("job=777"), read_calls(tmp_path)


@pytest.mark.parametrize("owner,squeue", [("job=4242", "RUNNING"),          # this job, requeued: its old instance is dying
                                          ("job=777", "COMPLETING"),        # ending: a chain's previous job, saving
                                          ("job=777", "fail"),              # squeue gives no answer
                                          ("job=777", "invalid"),           # a job the controller no longer knows
                                          ("job=777", "COMPLETED"),         # a job that has ended
                                          ("job=none host=HOST pid=999999999", "fail"),    # a dead process of this host
                                          (None, "RUNNING")])               # no owner file (yet): nobody to ask
def test_the_lock_of_an_ending_or_ended_holder_is_waited_for(tmp_path, owner, squeue):
    lock = tmp_path / "qwen35_m" / "ckpts" / f"{RN}.lock"
    # held 6 s: building the stand-in environment on a slow file system took up to 1.9 s, and the launcher must still
    # find the lock held when it gets there
    holder = hold_lock(lock, 6)
    host = subprocess.run(["hostname", "-s"], capture_output=True, text=True).stdout.strip()
    if owner:
        (lock.parent / f"{RN}.lock.owner").write_text(owner.replace("HOST", host) + " time=t\n")
    rc, out = _train(tmp_path, FAKE_SQUEUE=squeue, SLURM_JOB_ID=4242, DM_LOCK_WAIT=60)
    holder.wait()
    assert rc == 0 and "waiting up to 60 s (DM_LOCK_WAIT)" in out and (_hf(tmp_path) / "TRAIN_COMPLETE").exists(), out
    assert (lock.parent / f"{RN}.lock.owner").read_text().startswith(f"job=4242 host={host} pid="), out


@pytest.mark.parametrize("owner,squeue", [("job=777", "invalid"), ("job=777", "COMPLETING"), ("job=777", "fail"),
                                          ("job=4242", "RUNNING")])   # this job: a rank its last attempt left behind
def test_a_lock_that_is_never_released_is_a_failure_to_retry(tmp_path, owner, squeue):
    """The holder is ending, has ended, cannot be told about, or is this job, and the lock stays held for DM_LOCK_WAIT:
    exit 1 (job.sh retries, the next job runs on another node), not the 3 that stops a chain."""
    lock = tmp_path / "qwen35_m" / "ckpts" / f"{RN}.lock"
    holder = hold_lock(lock, 60)
    try:
        (lock.parent / f"{RN}.lock.owner").write_text(f"{owner} host=x pid=1 time=t\n")
        rc, out = _train(tmp_path, FAKE_SQUEUE=squeue, SLURM_JOB_ID=4242, DM_LOCK_WAIT=1)
    finally:
        holder.kill(); holder.wait()
    assert rc == 1 and "was not released within 1 s (DM_LOCK_WAIT): exit 1" in out and _ranks_started(tmp_path) == [], out


def test_the_holder_is_judged_again_while_waiting(tmp_path):
    """The lock's holder was ending (COMPLETING) when the wait began; then the lock went to a job that runs (another
    chain started the run meanwhile): the next look (every 10 s) sees it and exits 3."""
    lock = tmp_path / "qwen35_m" / "ckpts" / f"{RN}.lock"
    state = tmp_path / "squeue_state"
    state.write_text("COMPLETING\n")
    holder = hold_lock(lock, 90)
    (lock.parent / f"{RN}.lock.owner").write_text("job=777 host=x pid=1 time=t\n")
    p = subprocess.Popen(["bash", str(LAUNCHER), "clean_hd128", "11785"], cwd=str(REPO), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         env=fake_launch_env(tmp_path, FAKE_SQUEUE=f"file:{state}", SLURM_JOB_ID=4242, DM_LOCK_WAIT=60))
    try:
        head = []
        while not head or "waiting up to 60 s" not in head[-1]:
            head.append(p.stdout.readline())
            assert head[-1], "".join(head)
        assert "a job that is ending" in head[-1], head[-1]
        (lock.parent / f"{RN}.lock.owner").write_text("job=888 host=y pid=2 time=t\n")
        state.write_text("RUNNING\n")
        out, _ = p.communicate(timeout=60)
    finally:
        holder.kill(); holder.wait()
        if p.poll() is None:
            p.kill(); p.wait()
    assert p.returncode == 3 and f"FATAL: {RN} is being trained by job=888 host=y" in out, "".join(head) + out
    assert _ranks_started(tmp_path) == [] and "squeue -h -j 888 -o %T" in read_calls(tmp_path), out


def test_a_sigterm_during_the_val_ce_leaves_a_complete_run(tmp_path):
    """The run is complete before the informational val CE starts: a SIGTERM to the whole job then stops the val CE and
    the launcher still exits 0."""
    ready = tmp_path / "valce_ready"
    p = subprocess.Popen(["bash", str(LAUNCHER), "clean_hd128", "11785"], cwd=str(REPO), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True,
                         env=fake_launch_env(tmp_path, FAKE_VALCE_READY=ready))
    t0 = time.time()
    while not ready.exists():
        assert p.poll() is None and time.time() - t0 < 60, "the val CE never started"
        time.sleep(0.05)
    os.killpg(p.pid, signal.SIGTERM)   # as job.sh does: every process of the job
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 0 and (_hf(tmp_path) / "TRAIN_COMPLETE").exists(), out
    assert "the end-of-run val CE was skipped or cut short" in out and "valce SIGTERM" in read_calls(tmp_path), out


def test_a_stop_during_training_skips_the_val_ce(tmp_path):
    """The SIGTERM came with the last step: the ranks still wrote a whole export and exited 0, so the run is complete;
    the informational val CE is not started after a stop (the job is about to end)."""
    p = subprocess.Popen(["bash", str(LAUNCHER), "clean_hd128", "11785"], cwd=str(REPO), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True,
                         env=fake_launch_env(tmp_path, FAKE_MODE="complete_on_term"))
    t0 = time.time()
    while sum(c.endswith(" ready") for c in read_calls(tmp_path)) < 2:
        assert p.poll() is None and time.time() - t0 < 60, "the ranks never started"
        time.sleep(0.05)
    os.kill(p.pid, signal.SIGTERM)   # the launcher alone: the ranks hear it from forward_term
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 0 and (_hf(tmp_path) / "TRAIN_COMPLETE").exists() and "SIGTERM -> trainer ranks" in out, out
    assert "val CE was skipped or cut short" in out and not [c for c in read_calls(tmp_path) if c.startswith("valce")], out


def test_a_smoke_checks_the_tail_and_removes_its_dirs(tmp_path):
    rc, out = _train(tmp_path, SMOKE=1)
    assert rc == 0 and "SMOKE Q35M_clean_sdpa_ffn8_hd128_11785_smoke3: export+valce tail OK" in out, out
    assert not _hf(tmp_path, RN + "_smoke3").exists() and not _hf(tmp_path).exists(), out
    assert "--num_steps 3" in _ranks_started(tmp_path)[0], _ranks_started(tmp_path)


@pytest.mark.parametrize("knob", ["NUM_STEPS=12x", "STOP_AT_STEP=-1", "SMOKE=yes", "DM_NPROC=0"])
def test_a_bad_knob_is_a_configuration_error(tmp_path, knob):
    k, v = knob.split("=")
    rc, out = _train(tmp_path, **{k: v})
    assert rc == 2 and "FATAL" in out and _ranks_started(tmp_path) == [], out


def test_save_interval_per_model_size():
    """SAVE_IV (arms.sh): about an hour of steps per model size."""
    code = ('source scripts/campaign/env.sh; source scripts/campaign/arms.sh; '
            'for a in $(dm_arms); do NAME_ONLY=1 dm_arm_setup "$a" 1 >/dev/null 2>&1 && echo "$a $SAVE_IV" || echo "$a FAILED"; done')
    p = subprocess.run(["bash", "-c", code], cwd=str(REPO), capture_output=True, text=True, timeout=60)
    iv = dict(l.split() for l in p.stdout.splitlines())
    assert len(iv) == 39 and "l_dm_matched_hd256" in iv and "l_kda_clean_hd256" in iv and "xs_clean_hd256" in iv, iv   # every arm of arms.sh
    assert "xl_clean_hd256" in iv, iv
    for arm, v in iv.items():
        assert v == ("400" if arm.startswith("l_") else "600" if arm.startswith("xl_") else "2500" if arm.startswith("xs_")
                     else "1000"), (arm, v)


def test_git_rev_tells_dirty_trees_apart(tmp_path):
    """DM_GIT_REV = git describe --always --dirty, plus +<8 hex of the sha1 of `git diff HEAD -- src scripts`, followed
    by the sha1sum of every untracked file there> when the tree has changes there; an untracked file alone (a new
    module not added yet) makes the tree dirty too, and an edit of it changes the revision."""
    repo = tmp_path / "repo"
    shutil.copytree(REPO / "scripts", repo / "scripts")
    (repo / "src").mkdir()
    (repo / "src" / "m.py").write_text("x = 1\n")
    (repo / ".gitignore").write_text("__pycache__/\n*.pyc\n")   # as the repository's own: bytecode is not code
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "base"]):
        subprocess.run(git + cmd, check=True, capture_output=True)
    head = subprocess.run(git + ["describe", "--always"], capture_output=True, text=True).stdout.strip()

    def rev():
        for d in ("qwen35_m", "logs"):   # every launch starts from the same state
            shutil.rmtree(tmp_path / d, ignore_errors=True)
        env = fake_launch_env(tmp_path, FAKE_MODE="noexport")
        del env["DM_GIT_REV"]
        p = subprocess.run(["bash", str(repo / "scripts" / "campaign" / "train_arm.sh"), "clean_hd128", "11785"],
                           cwd=str(repo), env=env, capture_output=True, text=True, timeout=120)
        return re.search(r" git=(\S+) ", p.stdout).group(1)

    assert rev() == head   # a clean tree
    (repo / "src" / "m.py").write_text("x = 2\n")
    diff = subprocess.run(git + ["diff", "--no-ext-diff", "HEAD", "--", "src", "scripts"], capture_output=True).stdout
    assert rev() == f"{head}-dirty+{hashlib.sha1(diff).hexdigest()[:8]}"
    (repo / "src" / "m.py").write_text("x = 1\n")   # clean again, then a new module git does not track yet
    (repo / "src" / "new.py").write_text("y = 1\n")
    one = rev()
    sums = subprocess.run(["sha1sum", "--", "src/new.py"], cwd=str(repo), capture_output=True).stdout
    assert one == f"{head}-dirty+{hashlib.sha1(sums).hexdigest()[:8]}", one
    (repo / "src" / "new.py").write_text("y = 2\n")
    assert rev() not in (one, head)
