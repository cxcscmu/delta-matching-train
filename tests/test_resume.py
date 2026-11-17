"""The resume pipeline a run that may be stopped relies on: periodic checkpoints and exact resume, a
SIGTERM that saves a checkpoint and exits 75, a time budget (DM_STOP_BEFORE) already spent at startup that stops
before the first step (exit 75, nothing trained or saved; a run resumed at max_steps still exports), the run identity
(a resume with another horizon is refused unless DM_ALLOW_RUN_CHANGE=1), the local metrics log with its events, and
the idempotent finish (relaunching a completed run only re-exports). The same paths on CPU, with injected faults:
tests/test_resume_cpu.py; the loss equality of a resumed run: tests/test_resume_exactness_cpu.py."""
import os, signal, subprocess, time
import pytest
from _helpers import REPO, base_env, tiny_cmd, train_tiny, read_metrics

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.data]
LOGGING = ["--logging_steps", "1"]


def _kept(saves, limit):
    """The complete checkpoints left after saving `saves` (in order) at save_total_limit `limit`: rotation runs once the
    new checkpoint's marker is written and keeps the newest `limit` complete ones (docs/SETUP.md section 8)."""
    return [(s, True) for s in saves[-limit:]]


def _ckpts(run_root, name):
    d = run_root / "ckpts" / name
    return sorted((int(p.name.split("-")[-1]), (p / ".complete").exists()) for p in d.glob("checkpoint-*")) if d.exists() else []


def test_periodic_checkpoint_and_exact_resume(run_root, root, tok_dir, test_gpu):
    """A first leg to step 6 of a 9-step horizon (stop_at_step keeps the schedule of the whole run), then the rest:
    the relaunch resumes from checkpoint-6, passes the data-order sanity check and continues. A relaunch with another
    horizon is then refused (the run identity), unless DM_ALLOW_RUN_CHANGE=1."""
    name = "tiny_resume"
    rc, log, hf, _ = train_tiny(name, run_root, root, tok_dir, test_gpu, steps=9,
                                extra_args=["--save_interval", "3", "--stop_at_step", "6"] + LOGGING)
    assert rc == 0, log[-3000:]
    assert _ckpts(run_root, name) == _kept([3, 6], 1), _ckpts(run_root, name)   # save_total_limit 1 (tiny config)
    m1 = read_metrics(run_root, name)
    assert [r["event"] for r in m1 if r["event"] != "metrics"] == ["start", "checkpoint", "checkpoint", "export", "complete"], [r["event"] for r in m1]
    assert sorted(r["step"] for r in m1 if r["event"] == "metrics") == list(range(6))
    # relaunch without the stop: it must resume from checkpoint-6, pass the data-order sanity check, and continue
    rc, log, hf, _ = train_tiny(name, run_root, root, tok_dir, test_gpu, steps=9, extra_args=["--save_interval", "3"] + LOGGING)
    assert rc == 0, log[-3000:]
    assert "checkpoint-6" in log and "Resumed from" in log, log[-3000:]
    assert "Resume sanity check PASSED" in log, "the data stream did not line up after the resume"
    m2 = read_metrics(run_root, name)
    ev = [r for r in m2 if r["event"] == "resume"]
    assert ev and ev[-1]["step"] == 6, ev
    steps_after = [r["step"] for r in m2[m2.index(ev[-1]):] if r["event"] == "metrics"]
    assert steps_after == [6, 7, 8], steps_after
    assert (hf / "config.json").exists()
    assert _ckpts(run_root, name) == _kept([3, 6, 9], 1), _ckpts(run_root, name)
    # another horizon would continue a different run: refused before any state is loaded ...
    rc, log, hf, _ = train_tiny(name, run_root, root, tok_dir, test_gpu, steps=12, extra_args=["--save_interval", "3"] + LOGGING)
    assert rc == 2 and "num_steps: 9 -> 12" in log and "DM_ALLOW_RUN_CHANGE" in log, log[-3000:]   # 2: not retried
    assert _ckpts(run_root, name) == _kept([3, 6, 9], 1)
    # ... unless allowed: then it continues from checkpoint-9 (the change is recorded in run_identity.json)
    rc, log, hf, _ = train_tiny(name, run_root, root, tok_dir, test_gpu, steps=12, extra_args=["--save_interval", "3"] + LOGGING,
                                extra_env={"DM_ALLOW_RUN_CHANGE": "1"})
    assert rc == 0 and "checkpoint-9" in log and "Training complete at step 12" in log, log[-3000:]


def test_finished_run_relaunch_is_idempotent(run_root, root, tok_dir, test_gpu):
    name = "tiny_idem"
    rc, log, hf, _ = train_tiny(name, run_root, root, tok_dir, test_gpu, steps=2, extra_args=LOGGING)
    assert rc == 0, log[-2000:]
    rc, log, hf, _ = train_tiny(name, run_root, root, tok_dir, test_gpu, steps=2, extra_args=LOGGING)
    assert rc == 0 and "already complete" in log and "Training complete at step 2" in log, log[-2000:]


def _launch_bg(name, run_root, root, tok_dir, gpu, steps, extra_args=(), extra_env=None):
    cmd, hf = tiny_cmd(name, run_root, root, tok_dir, steps, extra_args)
    env = base_env(gpu); env.update(extra_env or {})
    out = open(run_root / f"{name}.console.log", "w")
    return subprocess.Popen(cmd, cwd=str(REPO), env=env, stdout=out, stderr=subprocess.STDOUT), hf


def _wait_for_step(run_root, name, step, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if any(r["event"] == "metrics" and r["step"] >= step for r in read_metrics(run_root, name)):
            return True
        time.sleep(2)
    return False


def test_sigterm_saves_checkpoint_exits_75_and_resumes(run_root, root, tok_dir, test_gpu):
    name = "tiny_sigterm"
    p, hf = _launch_bg(name, run_root, root, tok_dir, test_gpu, steps=400, extra_args=["--save_interval", "1000"] + LOGGING)
    assert _wait_for_step(run_root, name, 3), (run_root / f"{name}.console.log").read_text()[-3000:]
    os.kill(p.pid, signal.SIGTERM)
    rc = p.wait(timeout=600)
    log = (run_root / f"{name}.console.log").read_text()
    assert rc == 75, f"exit {rc}; tail:\n{log[-3000:]}"
    ck = _ckpts(run_root, name)
    assert ck and ck[-1][1] and ck[-1][0] >= 3, ck
    stop = [r for r in read_metrics(run_root, name) if r["event"] == "stop"]
    assert stop and stop[-1]["reason"].startswith("signal") and stop[-1]["step"] == ck[-1][0], stop
    assert not (hf / "config.json").exists(), "an early stop must not export"
    # resume to a nearby stop point: exact resume from the saved step, normal completion, export
    stop_at = ck[-1][0] + 2
    rc, log, hf2, _ = train_tiny(name, run_root, root, tok_dir, test_gpu, steps=400, extra_args=["--stop_at_step", str(stop_at)] + LOGGING)
    assert rc == 0 and f"checkpoint-{ck[-1][0]}" in log and "Resume sanity check PASSED" in log, log[-3000:]
    assert (hf2 / "config.json").exists() and f"Training complete at step {stop_at}" in log


def test_spent_deadline_stops_before_the_first_step_and_exits_75(run_root, root, tok_dir, test_gpu):
    """A time budget that has already run out at startup (DM_STOP_BEFORE = now): the trainer stops before the first
    step (whose compile and autotune alone take minutes) and exits 75 without training, saving or exporting: a fresh
    run has nothing to save, a resumed run's newest checkpoint already holds the state. A run resumed AT max_steps (a
    finished run whose export is missing) still exports."""
    name = "tiny_deadline"
    late = lambda: {"DM_STOP_BEFORE": str(int(time.time()))}  # noqa: E731
    rc, log, hf, _ = train_tiny(name, run_root, root, tok_dir, test_gpu, steps=4,
                                extra_args=["--save_interval", "1000"] + LOGGING, extra_env=late())
    assert rc == 75, f"exit {rc}; tail:\n{log[-3000:]}"
    assert _ckpts(run_root, name) == [], _ckpts(run_root, name)
    m = read_metrics(run_root, name)
    stop = [r for r in m if r["event"] == "stop"]
    assert stop and stop[-1]["reason"] == "deadline" and stop[-1]["step"] == 0 and stop[-1].get("before_first_step"), stop
    assert "Stopping early (deadline)" in log and not [r for r in m if r["event"] == "metrics"], "trained with a stop pending"
    assert not (hf / "config.json").exists(), "an early stop must not export"
    rc, log, hf, _ = train_tiny(name, run_root, root, tok_dir, test_gpu, steps=4, extra_args=["--stop_at_step", "2"] + LOGGING)
    assert rc == 0 and _ckpts(run_root, name) == [(2, True)], log[-3000:]     # a first leg: checkpoint-2
    rc, log, hf, _ = train_tiny(name, run_root, root, tok_dir, test_gpu, steps=4, extra_args=LOGGING, extra_env=late())
    assert rc == 75 and "before the first step" in log and _ckpts(run_root, name) == [(2, True)], log[-3000:]
    rc, log, hf, _ = train_tiny(name, run_root, root, tok_dir, test_gpu, steps=4, extra_args=LOGGING)
    assert rc == 0 and (hf / "config.json").exists() and _ckpts(run_root, name) == [(4, True)], log[-3000:]
    (hf / "config.json").unlink()                                             # the export is lost ...
    rc, log, hf, _ = train_tiny(name, run_root, root, tok_dir, test_gpu, steps=4, extra_args=LOGGING, extra_env=late())
    assert rc == 0 and "already complete" in log and (hf / "config.json").exists(), log[-3000:]   # ... and redone
