"""The resume pipeline end to end on CPU: the real `python -m pretrain.train` (FSDP2 over gloo, fp32; DDP once) on a
tiny GQA model, 1 or 2 ranks, with faults injected where the chain of 4-hour jobs meets them:
  * a transient EIO reading a checkpoint marker on rank 1 only: the resume is decided on the master, so the ranks
    cannot resume different checkpoints (before: rank 1 silently resumed an older one, the ranks wrote shards of two
    histories into the newer checkpoints and the job hung at the end);
  * a marker the master cannot read at all: every rank raises at once (no fall back to an older checkpoint, no rank
    left waiting in a collective);
  * a world-size change (2 ranks -> 1): the per-rank next-batch check is skipped and the run continues exactly;
  * a stop pending before the first step (the time budget already spent): exit 75 without training or saving; a run
    resumed at max_steps still exports (whole, check_export) and keeps its final metrics; a checkpoint without the
    sidecar (an older one) still resumes;
  * a resume with a changed learning rate: refused with exit 2 (a configuration error a chain does not retry), listing
    the key (DM_ALLOW_RUN_CHANGE=1: a warning, and the change is recorded);
  * with W&B on (a stand-in module, 2 ranks): a refused resume never initializes W&B and prints the RESUME REFUSED line
    on every rank; at a stop the process group is closed before the master's W&B finish, so a slow upload holds no
    other rank in the teardown;
  * the DDP backend: save, resume on more ranks than the save (rank 1 loads rank 0's state), sanity check.
Each launch takes about 8-15 s (slow marker: it starts training subprocesses)."""
import json, os, re, signal, subprocess, sys

import numpy as np
import pytest

from _helpers import REPO, clean_env

pytestmark = pytest.mark.slow

CFG = """run_name: x
backend: fsdp2
global_batch_size: 4
per_device_batch_size: 2
num_steps: 16
hf_export_steps: []
fused_ce: false
bf16: false
save_interval: 4
save_total_limit: 2
logging_steps: 1
seed: 11785
model: {vocab_size: 256, d_model: 64, n_layers: 2, n_heads: 4, n_kv_heads: 2, max_seq_len: 32, mixer: gqa, attn_impl: sdpa,
        qk_norm: true, attn_output_gate: true, tie_embeddings: true, z_loss_weight: 1.0e-4}
optim: {lr: 3.0e-3, weight_decay: 0.1, max_grad_norm: 1.0, lr_scheduler_type: wsd, warmup_ratio: 0.25, decay_ratio: 0.3}
data: {paths: X, dtype: uint16, num_workers: 0, pin_memory: false, seed: 7}
"""

# pretrain.train with an EIO injected into the reads of checkpoint-<EIO_STEP>/.complete on rank EIO_RANK: the first
# EIO_COUNT reads fail (-1: every read), through every read path the trainer may use (open, Path.read_text /
# read_bytes); each injected failure prints "[eio] rank <r>".
EIO_WRAPPER = r'''
import builtins, errno, os, pathlib, runpy, sys
RANK, STEP, COUNT = os.environ.get("EIO_RANK"), os.environ.get("EIO_STEP"), int(os.environ.get("EIO_COUNT", "1"))
hits = [0]
def _hit(p):
    p = pathlib.Path(str(p))
    if os.environ.get("RANK", "0") != RANK or p.name != ".complete" or p.parent.name != f"checkpoint-{STEP}":
        return False
    if COUNT >= 0 and hits[0] >= COUNT:
        return False
    hits[0] += 1
    print(f"[eio] rank {RANK}: EIO reading {p.parent.name}/.complete", flush=True)
    return True
_open, _rt, _rb = builtins.open, pathlib.Path.read_text, pathlib.Path.read_bytes
def open_(file, *a, **k):
    if isinstance(file, (str, bytes, os.PathLike)) and _hit(os.fsdecode(file)):
        raise OSError(errno.EIO, "Input/output error", os.fsdecode(file))
    return _open(file, *a, **k)
def read_text(self, *a, **k):
    if _hit(self):
        raise OSError(errno.EIO, "Input/output error", str(self))
    return _rt(self, *a, **k)
def read_bytes(self, *a, **k):
    if _hit(self):
        raise OSError(errno.EIO, "Input/output error", str(self))
    return _rb(self, *a, **k)
builtins.open, pathlib.Path.read_text, pathlib.Path.read_bytes = open_, read_text, read_bytes
sys.argv = ["pretrain.train"] + sys.argv[1:]
runpy.run_module("pretrain.train", run_name="__main__", alter_sys=True)
'''

# pretrain.train with a stand-in `wandb` module (no network, nothing written): init, mark_preempting and finish print
# what they see ("[fakewandb] rank <r> ..."), finish() whether the process group is still open; FAKE_WANDB_FINISH_S
# makes finish() take that long (a slow upload).
WANDB_WRAPPER = r'''
import os, runpy, sys, time, types
import torch.distributed as dist
RANK = os.environ.get("RANK", "0")
class Run:
    id = "fakeid"
    def log(self, metrics, step=None):
        pass
    def mark_preempting(self):
        print(f"[fakewandb] rank {RANK} mark_preempting", flush=True)
    def finish(self):
        print(f"[fakewandb] rank {RANK} finish, process group {'open' if dist.is_initialized() else 'closed'}", flush=True)
        time.sleep(float(os.environ.get("FAKE_WANDB_FINISH_S", "0")))
        print(f"[fakewandb] rank {RANK} finished", flush=True)
def init(**kw):
    print(f"[fakewandb] rank {RANK} init lr={kw['config']['optim']['lr']}", flush=True)
    return Run()
sys.modules["wandb"] = types.SimpleNamespace(init=init)
sys.argv = ["pretrain.train"] + sys.argv[1:]
runpy.run_module("pretrain.train", run_name="__main__", alter_sys=True)
'''


@pytest.fixture
def ws(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.integers(2, 256, size=4096)
    np.concatenate([np.roll(base, k) for k in range(16)]).astype(np.uint16).tofile(tmp_path / "tok.bin")
    (tmp_path / "c.yaml").write_text(CFG)
    (tmp_path / "eio.py").write_text(EIO_WRAPPER)
    (tmp_path / "fakewandb.py").write_text(WANDB_WRAPPER)
    return tmp_path


def _rank_exits(out):
    """The ranks' exit statuses in the summary torchrun prints when one failed (it exits 1 itself), in its order."""
    return re.findall(r"^\s+exitcode\s+:\s*(-?\d+)", out, re.M)


def _run(ws, name, *args, nproc=1, env=None, entry=None, timeout=300):
    """Launch the trainer (`entry`: a wrapper script instead of -m pretrain.train); returns (rc, output)."""
    e = clean_env()
    e.update(PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled", OMP_NUM_THREADS="1",
             PYTHONDONTWRITEBYTECODE="1")
    e.update(env or {})
    target = [str(entry)] if entry else ["-m", "pretrain.train"]
    launch = [sys.executable, *target] if nproc == 1 else \
        [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", *target]
    cmd = launch + ["--config", str(ws / "c.yaml"), "--data.paths", str(ws / "tok.bin"), "--ckpt_root_dir", str(ws / "ck"),
                    "--log_root_dir", str(ws / "lg"), "--run_name", name, *args]
    log = ws / f"{name}.{len(list(ws.glob(name + '.*.log')))}.log"
    # its own session, so that a launch that hangs is killed with all of its ranks (torchrun's workers would otherwise
    # outlive it, waiting in a collective until the 30-minute process-group timeout)
    with open(log, "w") as fh:
        p = subprocess.Popen(cmd, cwd=str(ws), env=e, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            pytest.fail(f"{name}: the launch did not end within {timeout} s (ranks waiting on each other?)\n"
                        f"{log.read_text(errors='replace')[-3000:]}")
    return rc, log.read_text(errors="replace")


def _ckpts(ws, name):
    d = ws / "ck" / name
    return sorted(int(p.name.split("-")[-1]) for p in d.glob("checkpoint-*") if (p / ".complete").exists())


def _events(ws, name):
    p = ws / "lg" / f"{name}.metrics.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def _ce(ws, name):
    """step -> ce_loss, the record after the last resume winning."""
    return {e["step"]: e["train/ce_loss"] for e in _events(ws, name) if e["event"] == "metrics"}


def test_transient_marker_eio_on_rank1_does_not_split_the_resume(ws):
    """One EIO on rank 1's read of checkpoint-12/.complete. Only the master reads markers now: both ranks resume at 12,
    the loaded steps agree, training ends normally and every later checkpoint has both ranks' shards."""
    rc, out = _run(ws, "eio", "--stop_at_step", "12", nproc=2)
    assert rc == 0 and _ckpts(ws, "eio") == [8, 12], (_ckpts(ws, "eio"), out[-3000:])
    rc, out = _run(ws, "eio", nproc=2, entry=ws / "eio.py", env={"EIO_RANK": "1", "EIO_STEP": "12", "EIO_COUNT": "1"},
                   timeout=240)
    assert rc == 0, out[-4000:]
    assert "Resumed: global_step=12" in out and "different steps" not in out, out[-3000:]
    assert "[eio] rank 1" not in out, "rank 1 read a checkpoint marker: the resume is not decided on the master"
    assert _ckpts(ws, "eio") == [12, 16]
    shards = sorted(p.name for p in (ws / "ck" / "eio" / "checkpoint-16").glob("*.distcp"))
    assert shards == ["__0_0.distcp", "__1_0.distcp"], shards
    resumed = [e for e in _events(ws, "eio") if e["event"] == "resume"]
    assert resumed and resumed[-1]["step"] == 12, resumed


def test_unreadable_marker_on_the_master_stops_every_rank(ws):
    """checkpoint-12's marker cannot be read on the master (EIO on every try): the resume refuses to fall back to
    checkpoint-8 and the error reaches rank 1 too (its own traceback), well before any process-group timeout."""
    rc, out = _run(ws, "unread", "--stop_at_step", "12", nproc=2)
    assert rc == 0 and _ckpts(ws, "unread") == [8, 12], out[-3000:]
    rc, out = _run(ws, "unread", nproc=2, entry=ws / "eio.py", env={"EIO_RANK": "0", "EIO_STEP": "12", "EIO_COUNT": "-1"},
                   timeout=180)
    assert rc != 0, out[-3000:]
    assert out.count("[eio] rank 0") >= 4, "the marker read was not retried"
    assert "cannot be read" in out and "CheckpointMarkerUnreadable" in out, out[-3000:]
    assert "[rank1]" in out and "failed on rank 0" in out, f"rank 1 did not raise the master's error:\n{out[-3000:]}"
    assert "Resumed: global_step=8" not in out, "fell back to an older checkpoint"
    assert _ckpts(ws, "unread") == [8, 12]


def test_world_size_change_resumes_and_continues_exactly(ws):
    """A run saved on 2 ranks resumes on 1 rank: the sanity check is skipped (the sidecar records the world size), and
    the losses continue the uninterrupted 2-rank run's (the global sample order is world-size invariant, DCP reshards)."""
    rc, out = _run(ws, "ref", nproc=2)
    assert rc == 0, out[-3000:]
    rc, out = _run(ws, "ws", "--stop_at_step", "8", nproc=2)
    assert rc == 0 and _ckpts(ws, "ws") == [4, 8], out[-3000:]
    assert json.loads((ws / "ck" / "ws" / "checkpoint-8" / "trainer_state.json").read_text())["world_size"] == 2
    rc, out = _run(ws, "ws", nproc=1)
    assert rc == 0, out[-3000:]
    assert "saved by 2 ranks and this run has 1" in out and "sanity check FAILED" not in out, out[-3000:]
    ref, got = _ce(ws, "ref"), _ce(ws, "ws")
    assert sorted(got) == list(range(16))
    worst = max(abs(ref[s] - got[s]) for s in range(8, 16))
    assert worst <= 1e-4, f"the 1-rank continuation diverged from the 2-rank run: max |d ce| {worst}"


def test_pending_stop_before_the_first_step(ws):
    """DM_STOP_BEFORE already past: a fresh run exits 75 with nothing saved and no step trained; a resumed run exits 75
    without a new checkpoint; a run resumed at max_steps exports anyway (whole), with its final metrics kept."""
    late = {"DM_STOP_BEFORE": "1"}
    rc, out = _run(ws, "pre", "--stop_at_step", "8", env=late)
    assert rc == 75 and "Stopping early (deadline) at step 0, before the first step" in out, out[-3000:]
    ev = _events(ws, "pre")
    assert _ckpts(ws, "pre") == [] and not [e for e in ev if e["event"] == "metrics"], ev
    assert [e for e in ev if e["event"] == "stop"][-1]["step"] == 0
    rc, out = _run(ws, "pre", "--stop_at_step", "8")                       # trains 0..7, checkpoints 4 and 8
    assert rc == 0 and _ckpts(ws, "pre") == [4, 8], out[-3000:]
    n_metrics = len([e for e in _events(ws, "pre") if e["event"] == "metrics"])
    (ws / "ck" / "pre" / "checkpoint-8" / "trainer_state.json").unlink()   # an older checkpoint: no sidecar
    rc, out = _run(ws, "pre", env=late)                                     # resumed at 8 of 16, stop pending
    assert rc == 75 and "Stopping early (deadline) at step 8, before the first step" in out, out[-3000:]
    assert _ckpts(ws, "pre") == [4, 8] and len([e for e in _events(ws, "pre") if e["event"] == "metrics"]) == n_metrics
    hf = ws / "hf" / "pre"
    rc, out = _run(ws, "pre", "--hf_save_dir", str(hf))                     # trains 8..15, exports
    assert rc == 0 and (hf / "config.json").exists(), out[-3000:]
    summary = ws / "result" / "pre" / "train_summary.json"
    first = json.loads(summary.read_text())["final_metrics"]
    assert first and first["train/ce_loss"] > 0
    os.unlink(hf / "config.json")                                           # a lost export: the next job re-exports
    rc, out = _run(ws, "pre", "--hf_save_dir", str(hf), env=late)          # resumed AT max_steps, stop pending
    assert rc == 0 and "already complete" in out and "Training complete at step 16" in out, out[-3000:]
    chk = subprocess.run([sys.executable, "-m", "pretrain.hf.check_export", str(hf)], capture_output=True, text=True,
                         env=dict(clean_env(), PYTHONPATH=str(REPO / "src")))
    assert chk.returncode == 0, chk.stdout + chk.stderr
    assert json.loads(summary.read_text())["final_metrics"] == first, "a process that trained no step lost the metrics"


def test_changed_learning_rate_is_refused_unless_allowed(ws):
    rc, out = _run(ws, "ident", "--stop_at_step", "4")
    assert rc == 0, out[-3000:]
    rc, out = _run(ws, "ident", "--optim.lr", "0.001", "--stop_at_step", "8")
    assert rc == 2 and "optim.lr: 0.003 -> 0.001" in out and "DM_ALLOW_RUN_CHANGE" in out, out[-3000:]   # not retried
    assert _ckpts(ws, "ident") == [4], "a refused resume trained"
    rc, out = _run(ws, "ident", "--optim.lr", "0.001", "--stop_at_step", "8", env={"DM_ALLOW_RUN_CHANGE": "1"})
    assert rc == 0 and "optim.lr: 0.003 -> 0.001" in out and _ckpts(ws, "ident") == [4, 8], out[-3000:]
    rec = json.loads((ws / "ck" / "ident" / "run_identity.json").read_text())
    assert rec["identity"]["optim.lr"] == 0.001 and rec["changes"][-1]["allowed"], rec["changes"]
    rc, out = _run(ws, "ident", "--optim.lr", "0.001", "--per_device_batch_size", "1", "--stop_at_step", "12")
    assert rc == 0 and _ckpts(ws, "ident") == [8, 12], out[-3000:]   # the recorded change, another micro-batch: fine


def test_refused_resume_and_stop_with_wandb(ws):
    """W&B on (the stand-in module), 2 ranks:
      * a resume refused for another learning rate never initializes W&B (the run's W&B id is not resumed with a
        configuration that is not trained); every rank prints the RESUME REFUSED line and exits 2 (torchrun reports
        1, and may SIGTERM a rank that is still exiting: a launcher maps the line back to 2);
      * a stop: the process group is closed before the master's W&B finish (marked preempting), so an upload slower
        than the teardown bound (8 s > DM_DIST_CLOSE_TIMEOUT=3) holds no other rank: no teardown timeout or failure,
        both ranks exit 75. (Before, the other ranks waited for the upload in dist_close's barrier and timed out.)"""
    wrap = ws / "fakewandb.py"
    rc, out = _run(ws, "wb", "--stop_at_step", "4", "--wandb", "true", nproc=2, entry=wrap)
    assert rc == 0 and "[fakewandb] rank 0 init lr=0.003" in out, out[-3000:]
    assert "[fakewandb] rank 0 finish, process group closed" in out and "[fakewandb] rank 1" not in out, out[-3000:]
    rc, out = _run(ws, "wb", "--optim.lr", "0.001", "--wandb", "true", nproc=2, entry=wrap)
    assert rc != 0 and "optim.lr: 0.003 -> 0.001" in out, out[-3000:]
    assert "[fakewandb]" not in out, f"a refused resume touched W&B:\n{out[-3000:]}"
    for r in (0, 1):
        assert f"[rank {r}] FATAL: RESUME REFUSED (exit 2" in out, out[-3000:]
    exits = _rank_exits(out)
    assert "2" in exits and set(exits) <= {"2", "-15"}, exits
    rc, out = _run(ws, "wb", "--wandb", "true", nproc=2, entry=wrap,
                   env={"DM_STOP_BEFORE": "1", "FAKE_WANDB_FINISH_S": "8", "DM_DIST_CLOSE_TIMEOUT": "3"})
    assert "Stopping early (deadline) at step 4, before the first step" in out, out[-3000:]
    assert "process-group teardown" not in out, f"the teardown waited for the master's W&B upload:\n{out[-3000:]}"
    assert "[fakewandb] rank 0 mark_preempting" in out and "[fakewandb] rank 0 finish, process group closed" in out, \
        out[-3000:]
    assert "[fakewandb] rank 0 finished" in out, out[-3000:]
    assert _rank_exits(out) == ["75", "75"], out[-3000:]
    assert _ckpts(ws, "wb") == [4]


def test_ddp_backend_saves_and_resumes_on_more_ranks(ws):
    """DDP: a 1-rank leg, a resume on 2 ranks (rank 1 has no training_state_1.pt and loads rank 0's: the replicas are
    identical), then a 2-rank resume with the sanity check. The losses follow an uninterrupted 2-rank run."""
    rc, out = _run(ws, "ddpref", "--backend", "ddp", "--stop_at_step", "12", nproc=2)
    assert rc == 0, out[-3000:]
    rc, out = _run(ws, "ddp", "--backend", "ddp", "--stop_at_step", "4")
    assert rc == 0 and _ckpts(ws, "ddp") == [4], out[-3000:]
    rc, out = _run(ws, "ddp", "--backend", "ddp", "--stop_at_step", "8", nproc=2)
    assert rc == 0 and "loading training_state_0.pt" in out and _ckpts(ws, "ddp") == [4, 8], out[-3000:]
    assert "saved by 1 ranks and this run has 2" in out, out[-3000:]
    assert sorted(p.name for p in (ws / "ck" / "ddp" / "checkpoint-8").glob("training_state_*.pt")) == \
        ["training_state_0.pt", "training_state_1.pt"]
    rc, out = _run(ws, "ddp", "--backend", "ddp", "--stop_at_step", "12", nproc=2)
    assert rc == 0 and "Resume sanity check PASSED" in out and _ckpts(ws, "ddp") == [8, 12], out[-3000:]
    ref, got = _ce(ws, "ddpref"), _ce(ws, "ddp")
    assert sorted(got) == list(range(12))
    worst = max(abs(ref[s] - got[s]) for s in range(4, 12))
    assert worst <= 1e-4, f"the resumed DDP run diverged from the uninterrupted one: max |d ce| {worst}"
