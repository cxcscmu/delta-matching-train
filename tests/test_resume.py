"""The resume pipeline a run that may be stopped relies on: periodic checkpoints and exact resume, a
SIGTERM that saves a checkpoint and exits 75, a time budget (DM_STOP_BEFORE) that does the same, the local metrics log
with its events, and the idempotent finish (relaunching a completed run only re-exports)."""
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
    name = "tiny_resume"
    rc, log, hf, _ = train_tiny(name, run_root, root, tok_dir, test_gpu, steps=6, extra_args=["--save_interval", "3"] + LOGGING)
    assert rc == 0, log[-3000:]
    assert _ckpts(run_root, name) == _kept([3, 6], 1), _ckpts(run_root, name)   # save_total_limit 1 (tiny config)
    m1 = read_metrics(run_root, name)
    assert [r["event"] for r in m1 if r["event"] != "metrics"] == ["start", "checkpoint", "checkpoint", "export", "complete"], [r["event"] for r in m1]
    assert sorted(r["step"] for r in m1 if r["event"] == "metrics") == list(range(6))
    # relaunch with a longer horizon: it must resume from checkpoint-6, pass the data-order sanity check, and continue
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


def test_deadline_saves_checkpoint_and_exits_75(run_root, root, tok_dir, test_gpu):
    """A time budget that has already run out (DM_STOP_BEFORE = now): the trainer stops at the first step boundary,
    saves a complete checkpoint there and exits 75 without exporting, however fast the node is."""
    name = "tiny_deadline"
    rc, log, hf, _ = train_tiny(name, run_root, root, tok_dir, test_gpu, steps=400,
                                extra_args=["--save_interval", "1000"] + LOGGING,
                                extra_env={"DM_STOP_BEFORE": str(int(time.time()))})
    assert rc == 75, f"exit {rc}; tail:\n{log[-3000:]}"
    assert _ckpts(run_root, name) == [(1, True)], _ckpts(run_root, name)
    stop = [r for r in read_metrics(run_root, name) if r["event"] == "stop"]
    assert stop and stop[-1]["reason"] == "deadline" and stop[-1]["step"] == 1 and "Stopping early (deadline)" in log, stop
    assert not (hf / "config.json").exists(), "an early stop must not export"
