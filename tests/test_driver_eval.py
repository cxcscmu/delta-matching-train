"""scripts/campaign/eval_arm.sh and eval_arm_64k.sh in a chain, on the CPU with the stand-ins of tests/_helpers.py (a
stand-in torchrun writes each benchmark's completion marker, the stand-in val CE its per-window artifact):
  * the run lock on <eval dir>/.lock: another live job's lock -> exit 3 before any benchmark starts;
  * the progress lines at exit (DM_PROGRESS_FILE): "eval <run> <n>" when n completion markers appeared during this
    invocation (each val CE this invocation scored writes one too: valce*_n<W>.complete), "eval_complete <run>" when it
    wrote the final marker, also after a SIGTERM;
  * the val CE's rendezvous port: DM_VALCE_PORT, else the fixed default."""
import json
import os
import re
import signal
import subprocess
import time

import pytest

from _helpers import REPO, NEMO, fake_launch_env, read_calls, hold_lock

RN = "Q35M_test_arm_11785"
CAMPAIGN = REPO / "scripts" / "campaign"
PATCHED = {"attn_impl": "sdpa", "layer_mixers": ["gated_deltanet"] * 3 + ["gqa"], "mixer": "gqa"}
YARN = {**PATCHED, "max_seq_len": 65536, "rope_scaling": {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 8192}}


def _setup(tmp_path, stage, **extra):
    """An export, the val streams, the RULER data dir and the MMLU cache dir the scripts look for; the stand-in env."""
    hf = tmp_path / "qwen35_m" / "hf" / RN
    hf.mkdir(parents=True)
    (hf / "config.json").write_text(json.dumps(YARN if stage == 2 else PATCHED))
    (hf / "model.safetensors").write_text("WHOLE")
    env = fake_launch_env(tmp_path, **extra)
    (tmp_path / "data" / NEMO / "nemotron_rs_val.bin").write_bytes(b"\x01\x00" * 64)
    (tmp_path / "data" / "prolong_64k_llama2").mkdir(parents=True)
    (tmp_path / "data" / "prolong_64k_llama2" / "prolong64k_val.bin").write_bytes(b"\x01\x00" * 64)
    (tmp_path / "data" / "ruler-500-llama2" / "8192" / "data").mkdir(parents=True)
    (tmp_path / "hf_home" / "datasets" / "cais___mmlu").mkdir(parents=True)
    return env


def _script(stage):
    return CAMPAIGN / ("eval_arm_64k.sh" if stage == 2 else "eval_arm.sh")


def _eval(env, stage, windows="4"):
    p = subprocess.run(["bash", str(_script(stage)), RN, windows], cwd=str(REPO), env=env, capture_output=True, text=True,
                       timeout=300)
    return p.returncode, p.stdout + p.stderr


def _progress(path):
    return [l.split(" ", 2)[2] for l in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize("stage,final,valces", [(1, "EVAL_COMPLETE", ["valce_n4"]),
                                                (2, "EVAL64K_COMPLETE", ["valce64k_n4", "valce_n2000"])])
def test_a_complete_evaluation_reports_its_progress_and_uses_the_jobs_port(tmp_path, stage, final, valces):
    progress = tmp_path / "progress"
    env = _setup(tmp_path, stage, DM_PROGRESS_FILE=progress, SLURM_JOB_ID=99, DM_VALCE_PORT=31234)
    rc, out = _eval(env, stage)
    out_dir = tmp_path / "qwen35_m" / "eval" / RN
    assert rc == 0 and (out_dir / final).exists(), out
    assert all((out_dir / f"{v}.complete").exists() for v in valces), sorted(p.name for p in out_dir.iterdir())
    # the val CE(s), six benchmark markers and the final one appeared in this invocation
    assert _progress(progress) == [f"eval {RN} {len(valces) + 7}", f"eval_complete {RN}"], (progress.read_text(), out)
    assert all(re.match(r"\d+ 99 ", l) for l in progress.read_text().splitlines()), progress.read_text()
    ports = [c for c in read_calls(tmp_path) if c.startswith("valce ")]
    assert ports and all(c.startswith("valce port=31234 ") for c in ports), ports
    assert (tmp_path / "qwen35_m" / "eval" / RN / ".lock.owner").read_text().startswith("job=99 host=")


@pytest.mark.parametrize("stage,port", [(1, "29800"), (2, "29820")])
def test_the_val_ce_port_defaults_outside_a_chain(tmp_path, stage, port):
    env = _setup(tmp_path, stage)
    rc, out = _eval(env, stage)
    ports = [c for c in read_calls(tmp_path) if c.startswith("valce ")]
    assert rc == 0 and ports and all(c.startswith(f"valce port={port} ") for c in ports), (ports, out)
    assert not (tmp_path / "progress").exists()


def test_an_incomplete_evaluation_reports_the_markers_it_added(tmp_path):
    """Two benchmarks fail: exit 1, and the progress line counts the markers this invocation did add (the val CE and
    three benchmarks); the next invocation counts only its own (the val CE is not scored again)."""
    progress = tmp_path / "progress"
    env = _setup(tmp_path, 1, DM_PROGRESS_FILE=progress, FAKE_FAIL_CFGS="mmlu triviaqa")
    out_dir = tmp_path / "qwen35_m" / "eval" / RN
    (out_dir / "commonsense").mkdir(parents=True)
    (out_dir / "commonsense" / "commonsense_results.complete").write_text("an earlier job\n")   # not this invocation's
    rc, out = _eval(env, 1)
    assert rc == 1 and "EVAL NOT COMPLETE" in out, out
    assert _progress(progress) == [f"eval {RN} 4"], progress.read_text() if progress.exists() else out
    del env["FAKE_FAIL_CFGS"]
    rc, out = _eval(env, 1)
    assert rc == 0 and "val CE already done" in out, out
    assert _progress(progress)[1:] == [f"eval {RN} 3", f"eval_complete {RN}"], progress.read_text()   # 2 benchmarks + final


@pytest.mark.parametrize("stage", [1, 2])
def test_a_sigterm_still_reports_the_progress_made(tmp_path, stage):
    """The time limit's SIGTERM stops the evaluation in its second benchmark: exit 143, and the markers of the val
    CE(s) and of the first benchmark are reported (the chain's progress gate counts them)."""
    progress, ready = tmp_path / "progress", tmp_path / "ready"
    env = _setup(tmp_path, stage, DM_PROGRESS_FILE=progress, FAKE_SLOW_CFG="commonsense", FAKE_READY=ready)
    p = subprocess.Popen(["bash", str(_script(stage)), RN, "4"], cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True)
    t0 = time.time()
    while not ready.exists():
        assert p.poll() is None and time.time() - t0 < 120, "the second benchmark never started"
        time.sleep(0.05)
    os.killpg(p.pid, signal.SIGTERM)
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 143 and "stopped by a signal" in out, out
    assert _progress(progress) == [f"eval {RN} {stage + 1}"], (progress.read_text() if progress.exists() else "", out)


@pytest.mark.parametrize("stage", [1, 2])
def test_the_lock_of_a_live_job_exits_3(tmp_path, stage):
    progress = tmp_path / "progress"
    env = _setup(tmp_path, stage, DM_PROGRESS_FILE=progress, FAKE_SQUEUE="RUNNING", SLURM_JOB_ID=99)
    lock = tmp_path / "qwen35_m" / "eval" / RN / ".lock"
    holder = hold_lock(lock, 60)
    try:
        (lock.parent / ".lock.owner").write_text("job=777 host=elsewhere pid=1 time=t\n")
        rc, out = _eval(env, stage)
    finally:
        holder.kill(); holder.wait()
    assert rc == 3 and f"FATAL: {RN} is being evaluated by job=777" in out, out
    assert not [c for c in read_calls(tmp_path) if c.startswith(("torchrun", "valce"))] and not progress.exists(), out


@pytest.mark.parametrize("stage", [1, 2])
def test_a_sigterm_ends_the_wait_for_the_lock(tmp_path, stage):
    """The script waits for a lock whose holder is ending (DM_LOCK_WAIT 60 s) and gets a SIGTERM of its own (its trap
    only counts it, and the flock it waits in never hears it): the wait ends within 10 s and the script exits 143, no
    benchmark started."""
    env = _setup(tmp_path, stage, SLURM_JOB_ID=99, DM_LOCK_WAIT=60, FAKE_SQUEUE="COMPLETING")
    out_dir = tmp_path / "qwen35_m" / "eval" / RN
    holder = hold_lock(out_dir / ".lock", 60)
    (out_dir / ".lock.owner").write_text("job=777 host=x pid=1 time=t\n")
    p = subprocess.Popen(["bash", str(_script(stage)), RN, "4"], cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    try:
        head = []
        while not head or "waiting up to 60 s" not in head[-1]:
            head.append(p.stdout.readline())
            assert head[-1], "".join(head)
        t0 = time.time()
        os.kill(p.pid, signal.SIGTERM)
        out, _ = p.communicate(timeout=30)
    finally:
        holder.kill(); holder.wait()
        if p.poll() is None:
            p.kill(); p.wait()
    assert p.returncode == 143 and time.time() - t0 < 15 and "a stop signal ended the wait for the lock" in out, "".join(head) + out
    assert not [c for c in read_calls(tmp_path) if c.startswith(("torchrun", "valce"))], read_calls(tmp_path)


@pytest.mark.parametrize("owner,squeue", [("job=99", "RUNNING"),         # this job, requeued
                                          ("job=777", "COMPLETING")])   # the chain's previous job, still ending
def test_a_finished_evaluation_after_the_wait_exits_0(tmp_path, owner, squeue):
    """The lock's holder finished the evaluation while this one waited: exit 0, nothing re-run."""
    env = _setup(tmp_path, 1, SLURM_JOB_ID=99, DM_LOCK_WAIT=60, FAKE_SQUEUE=squeue)
    out_dir = tmp_path / "qwen35_m" / "eval" / RN
    holder = hold_lock(out_dir / ".lock", 2)
    (out_dir / ".lock.owner").write_text(f"{owner} host=x pid=1 time=t\n")
    p = subprocess.Popen(["bash", str(_script(1)), RN, "4"], cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    head = []
    while not head or "waiting up to 60 s" not in head[-1]:   # the script is waiting for the lock: finish the evaluation
        head.append(p.stdout.readline())
        assert head[-1], "".join(head)
    (out_dir / "EVAL_COMPLETE").write_text("done\n")
    out, _ = p.communicate(timeout=60)
    holder.wait()
    assert p.returncode == 0 and "by the job that held its lock" in out, "".join(head) + out
    assert not [c for c in read_calls(tmp_path) if c.startswith(("torchrun", "valce"))], read_calls(tmp_path)
