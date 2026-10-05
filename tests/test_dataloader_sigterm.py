"""The train loader's workers and a scheduler's SIGTERM (src/pretrain/data/data_loader.py `_worker_init`), CPU only.

A job scheduler may signal every process of a job, the DataLoader workers included. A worker that dies of it makes
PyTorch's SIGCHLD handler raise in the trainer ("DataLoader worker (pid N) is killed by signal: Terminated") about 3 s
later, before the step boundary where the trainer saves a checkpoint and exits 75, so the stop saves nothing.
A stand-in trainer (the real build_train_dataloader over a small token file, the trainer's flag-setting SIGTERM
handler) runs in its own process group, which is signalled as a job scheduler does. It must:
  * survive the SIGTERM (sent twice), finish its step, "save" and exit 75;
  * still exit promptly on every other path, a normal exit and an exception (workers that simply ignored SIGTERM
    would pass the first check and hang here, because multiprocessing's atexit terminate()s the daemonic workers
    with SIGTERM and then joins them);
  * a control with the workers' init removed must reproduce the crash, so the survival check cannot pass by the
    signal going astray;
  * and the data order is the same with and without the workers' init.
"""
import os
import select
import signal
import subprocess
import time

import numpy as np
import pytest

from _helpers import SRC, python

TRAINER = r"""
import os, signal, sys, time
os.environ.update(WORLD_SIZE="1", RANK="0")
from pretrain.data.data_loader import build_train_dataloader
mode, path = sys.argv[1], sys.argv[2]
loader = build_train_dataloader({"paths": [path], "num_workers": 2, "pin_memory": False, "prefetch_factor": 2},
                                16, 4, seed=0)
if mode == "control":
    loader.worker_init_fn = None                     # the loader without the workers' init: the failure mode
stop = []
signal.signal(signal.SIGTERM, lambda s, f: stop.append(s))   # the trainer's _on_stop_signal: set a flag, carry on
it = iter(loader)
next(it); next(it)                                   # one batch from each worker: both have run their init
print("READY", flush=True)
if mode == "exit":
    sys.exit(0)
if mode == "crash":
    raise RuntimeError("stand-in trainer failure")
t0 = time.time()
while not stop and time.time() - t0 < 30:
    next(it); time.sleep(0.02)
assert stop, "the signal never arrived"
for _ in range(4):
    next(it)                                         # the rest of the step (gradient accumulation) after the signal
time.sleep(1.0)                                      # the checkpoint save
print("STOPPED", flush=True)
sys.exit(75)
"""


@pytest.fixture
def tokens(tmp_path):
    p = tmp_path / "tokens.bin"
    (np.arange(1 << 16) % 32000).astype(np.uint16).tofile(p)     # 4096 chunks of 16: far more batches than needed
    return p


def _run(mode, path, preempt):
    env = dict(os.environ, PYTHONPATH=str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    p = subprocess.Popen([python(), "-c", TRAINER, mode, str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env, start_new_session=True)
    try:
        if not select.select([p.stdout], [], [], 120)[0]:     # a worker that cannot start must fail the test, not hang it
            os.killpg(p.pid, signal.SIGKILL)
            pytest.fail(f"{mode}: the stand-in trainer did not start within 120 s:\n{p.stderr.read()[-2000:]}")
        line = p.stdout.readline().strip()
        assert line == "READY", f"stand-in trainer did not start ({line!r}): {p.stderr.read()[-2000:]}"
        t0 = time.time()
        if preempt:
            os.killpg(p.pid, signal.SIGTERM)         # the scheduler: every process of the job, workers included
            time.sleep(0.005)
            os.killpg(p.pid, signal.SIGTERM)         # and again, as a wrapper's forward arrives
        try:
            out, err = p.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            pytest.fail(f"{mode}: the stand-in trainer did not exit within 60 s (a worker is blocking the exit)")
        return p.returncode, out, err, time.time() - t0
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_workers_survive_sigterm_so_the_trainer_saves(tokens):
    rc, out, err, _ = _run("fixed", tokens, preempt=True)
    assert "killed by signal" not in err, f"a loader worker died of the SIGTERM:\n{err[-2000:]}"
    assert rc == 75 and "STOPPED" in out, f"exit {rc}, expected the graceful 75:\n{out}\n{err[-2000:]}"


@pytest.mark.parametrize("mode,want", [("exit", 0), ("crash", 1)])
def test_every_other_exit_is_prompt(tokens, mode, want):
    rc, out, err, dt = _run(mode, tokens, preempt=False)
    assert rc == want, f"{mode}: exit {rc}, expected {want}:\n{err[-2000:]}"
    assert dt < 15, f"{mode}: took {dt:.1f} s to exit (the workers held it up)"


def test_control_reproduces_the_preemption_crash(tokens):
    rc, out, err, _ = _run("control", tokens, preempt=True)
    assert rc != 75 and "STOPPED" not in out, f"control did not fail (exit {rc}): the test would not catch the bug"
    assert "DataLoader worker" in err and "killed by signal" in err, err[-2000:]


def test_data_order_is_unchanged(tokens, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    from pretrain.data.data_loader import build_train_dataloader

    def first_batches(init):
        loader = build_train_dataloader({"paths": [str(tokens)], "num_workers": 2, "pin_memory": False,
                                         "persistent_workers": False}, 16, 4, seed=0)
        assert loader.worker_init_fn is not None
        if not init:
            loader.worker_init_fn = None
        it = iter(loader)
        return [next(it)["input_ids"] for _ in range(6)]

    for a, b in zip(first_batches(True), first_batches(False)):
        assert (a == b).all()
