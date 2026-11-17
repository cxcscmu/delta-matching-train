"""CPU unit checks of the trainer's resume decisions (src/pretrain/trainer/base_trainer.py, utils/run_identity.py,
train.py), with the model and the collectives stubbed out:
  * run identity: a resume whose trajectory-defining values changed is refused, listing every differing key
    (DM_ALLOW_RUN_CHANGE=1: a warning, and the new values are recorded); a run without a record gets one; the knobs that
    may differ between the jobs of one run (stop_at_step, per_device_batch_size, the world size, save / export / wandb
    settings, init_from_hf) never refuse; a software change only warns; data paths compare as resolved paths; the record
    of a resume is written only once the state has loaded; an allow knob that is not needed is a warning;
  * a stop pending before the first step returns "stopped" without training or saving; a run resumed at max_steps
    exports anyway;
  * the W&B teardown is bounded (a finish() that hangs returns within DM_WANDB_FINISH_TIMEOUT) and marks a stopped run
    preempting first; train() leaves it to pretrain.train, which runs it after the process group is closed;
  * pretrain.train exits 75 after a stop even when dist_close() fails or hangs;
  * DDP: a rank without a training_state file of its own (a resume on more ranks) loads rank 0's;
  * the replay warning (a resume below the last logged step), the result summary of a process that trained no step;
  * the master's decisions reach every rank: a failure or a different loaded step on another rank, and a failed data
    sanity check on another rank, raise here too."""
import json, os, subprocess, sys, threading, time, types
from pathlib import Path

import pytest
import torch

from _helpers import REPO, clean_env

VERSION = "pt-v0.1"


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    for k in ("DM_PROGRESS_FILE", "SLURM_JOB_ID", "DM_STOP_BEFORE", "DM_ALLOW_RUN_CHANGE", "DM_ALLOW_KERNEL_CHANGE",
              "DM_WANDB_FINISH_TIMEOUT", "DM_GIT_REV", "DM_FP8GLUE", "DM_FP8GLUE_SCALING"):
        monkeypatch.delenv(k, raising=False)
    from pretrain.utils import ckpt
    monkeypatch.setattr(ckpt, "MARKER_RETRY_DELAY", 0.0)


def _stub(**attrs):
    from pretrain.trainer.base_trainer import BaseTrainer
    tr = object.__new__(BaseTrainer)
    tr.logs, tr.events = [], []
    tr.main_log = lambda msg, *a, **k: tr.logs.append(str(msg))
    tr._metrics_event = lambda event, **f: tr.events.append(dict(f, event=event))
    for k, v in attrs.items():
        setattr(tr, k, v)
    return tr


# ---------------------------------------------------------------- run identity
def _cfg(*args):
    from pretrain.config import load_config
    return load_config(["--run_name", "r", *args])


def _check(tmp_path, cfg, resumed, allow=False):
    from pretrain.utils.run_identity import check_run_identity
    return check_run_identity(tmp_path, cfg, resumed, allow)


def test_run_identity_records_then_accepts_the_same_run(tmp_path):
    from pretrain.utils.run_identity import IDENTITY_FILE
    ok, msgs = _check(tmp_path, _cfg(), None)                   # a first start records
    rec = json.loads((tmp_path / IDENTITY_FILE).read_text())
    assert ok == "ok" and rec["identity"]["optim.lr"] == 3e-3 and rec["identity"]["num_steps"] == 1000
    assert rec["identity"]["DM_FP8GLUE"] == "1" and "torch" in rec["env"] and rec["changes"] == []
    for key in ("model.d_model", "data.paths", "data.seed", "global_batch_size", "optim.warmup_ratio", "seed",
                "model.max_seq_len", "fp8_scope", "DM_FP8_FSDP_AG"):
        assert key in rec["identity"], key
    for key in ("stop_at_step", "per_device_batch_size", "save_interval", "init_from_hf", "hf_save_dir", "wandb",
                "num_devices", "data.num_workers", "gradient_checkpointing", "run_name", "ckpt_root_dir"):
        assert key not in rec["identity"] and f"data.{key}" not in rec["identity"], key
    before = (tmp_path / IDENTITY_FILE).read_text()
    ok, msgs = _check(tmp_path, _cfg(), 400)                    # the same run resumes: no change, no rewrite
    assert ok == "ok" and not [m for lvl, m in msgs if lvl != "info"], msgs
    assert (tmp_path / IDENTITY_FILE).read_text() == before


def test_run_identity_refuses_a_changed_trajectory_listing_every_key(tmp_path):
    _check(tmp_path, _cfg(), None)
    ok, msgs = _check(tmp_path, _cfg("--optim.lr", "0.001", "--num_steps", "2000", "--data.chunk_subset", "[0.0,0.5]"), 400)
    assert ok == "refused"
    msg = msgs[-1][1]
    for line in ("optim.lr: 0.003 -> 0.001", "num_steps: 1000 -> 2000", "data.chunk_subset: null -> [0.0, 0.5]"):
        assert line in msg, msg
    assert "DM_ALLOW_RUN_CHANGE=1" in msg and "step 400" in msg
    ok, msgs = _check(tmp_path, _cfg("--model.d_model", "512"), 400)
    assert ok == "refused" and "model.d_model: 1024 -> 512" in msgs[-1][1]


def test_run_identity_change_allowed_is_recorded(tmp_path):
    from pretrain.utils.run_identity import IDENTITY_FILE
    _check(tmp_path, _cfg(), None)
    ok, msgs = _check(tmp_path, _cfg("--optim.lr", "0.001"), 400, allow=True)
    assert ok == "ok" and any(lvl == "warning" and "optim.lr: 0.003 -> 0.001" in m for lvl, m in msgs), msgs
    rec = json.loads((tmp_path / IDENTITY_FILE).read_text())
    assert rec["identity"]["optim.lr"] == 0.001
    assert rec["changes"][-1]["allowed"] and rec["changes"][-1]["identity"] == [["optim.lr", 0.003, 0.001]]
    assert rec["changes"][-1]["step"] == 400
    ok, _ = _check(tmp_path, _cfg("--optim.lr", "0.001"), 800)  # the next job continues without the knob
    assert ok == "ok"
    ok, _ = _check(tmp_path, _cfg(), 800)                       # ... and going back is a change again
    assert ok == "refused"


def test_run_identity_allows_what_differs_between_the_jobs_of_one_run(tmp_path, monkeypatch):
    _check(tmp_path, _cfg(), None)
    monkeypatch.setenv("WORLD_SIZE", "4")                       # another node size: gradient accumulation follows
    other = _cfg("--stop_at_step", "500", "--per_device_batch_size", "2", "--save_interval", "7", "--save_total_limit",
                 "5", "--logging_steps", "3", "--hf_save_dir", "/x/hf", "--hf_export_steps", "[100,200]",
                 "--init_from_hf", "/x/init", "--wandb", "true", "--wandb_job_type", "stage9", "--gradient_checkpointing",
                 "true", "--data.num_workers", "7", "--ckpt_root_dir", "/elsewhere", "--log_root_dir", "/logs2",
                 "--tokenizer_dir", "/tok", "--force_resume_check", "false")
    ok, msgs = _check(tmp_path, other, 400)
    assert ok == "ok" and not [m for lvl, m in msgs if lvl != "info"], msgs


def test_run_identity_numerics_environment(tmp_path, monkeypatch):
    _check(tmp_path, _cfg(), None)
    monkeypatch.setenv("DM_FP8GLUE", "1")                       # explicitly the code default: the same configuration
    assert _check(tmp_path, _cfg(), 400)[0] == "ok"
    monkeypatch.setenv("DM_FP8GLUE_SCALING", "delayed")         # another fp8 scaling: another experiment
    ok, msgs = _check(tmp_path, _cfg(), 400)
    assert ok == "refused" and 'DM_FP8GLUE_SCALING: "dynamic" -> "delayed"' in msgs[-1][1], msgs


def test_run_identity_software_change_and_code_keys_only_warn(tmp_path, monkeypatch):
    from pretrain.utils.run_identity import IDENTITY_FILE
    monkeypatch.setenv("DM_GIT_REV", "abc123")
    _check(tmp_path, _cfg(), None)
    rec = json.loads((tmp_path / IDENTITY_FILE).read_text())
    rec["identity"]["model.retired_knob"] = 1                   # a field the code has since removed
    del rec["identity"]["model.nope"]                           # ... and one it has since added
    rec["env"]["torch"] = "2.11.0"
    (tmp_path / IDENTITY_FILE).write_text(json.dumps(rec))
    monkeypatch.setenv("DM_GIT_REV", "def456-dirty+0123abcd")
    ok, msgs = _check(tmp_path, _cfg(), 400)
    text = "\n".join(m for _, m in msgs)
    assert ok == "ok" and "model.retired_knob: 1 -> <absent>" in text and "model.nope: <absent> -> false" in text, text
    assert 'DM_GIT_REV: "abc123" -> "def456-dirty+0123abcd"' in text and 'torch: "2.11.0"' in text, text
    rec = json.loads((tmp_path / IDENTITY_FILE).read_text())
    assert rec["env"]["DM_GIT_REV"] == "def456-dirty+0123abcd" and not rec["changes"][-1]["allowed"]
    assert _check(tmp_path, _cfg(), 800) == ("ok", [])          # recorded: the next job is quiet


def test_run_identity_without_or_with_a_broken_record(tmp_path):
    from pretrain.utils.run_identity import IDENTITY_FILE
    ok, msgs = _check(tmp_path, _cfg(), 400)                    # an older run resumed: no record yet, one is made
    assert ok == "ok" and (tmp_path / IDENTITY_FILE).exists() and "did not exist" in msgs[-1][1]
    assert _check(tmp_path, _cfg("--optim.lr", "0.001"), 800)[0] == "refused"
    (tmp_path / IDENTITY_FILE).write_text("{cut")                # garbled: no retry repairs it
    ok, msgs = _check(tmp_path, _cfg(), 800)
    assert ok == "refused" and "cannot be read" in msgs[-1][1]
    (tmp_path / IDENTITY_FILE).unlink()
    (tmp_path / IDENTITY_FILE).mkdir()                          # an I/O error (here: not a file): may pass, exit 1
    ok, msgs = _check(tmp_path, _cfg(), 800)
    assert ok == "unreadable" and "cannot be read" in msgs[-1][1]
    (tmp_path / IDENTITY_FILE).rmdir()
    (tmp_path / IDENTITY_FILE).write_text("{cut")
    ok, msgs = _check(tmp_path, _cfg(), 800, allow=True)
    assert ok == "ok" and json.loads((tmp_path / IDENTITY_FILE).read_text())["identity"]["optim.lr"] == 0.003


def test_trainer_raises_the_identity_refusal(tmp_path, monkeypatch):
    from pretrain.trainer.base_trainer import ResumeRefused
    from pretrain.utils.run_identity import IDENTITY_FILE
    tr = _stub(cfg=_cfg(), checkpoint_dir=tmp_path)
    tr._check_run_identity(None)
    assert not (tmp_path / IDENTITY_FILE).exists()                       # prepared; written by _record_run_identity
    tr._record_run_identity()
    tr.cfg = _cfg("--optim.lr", "0.001")
    with pytest.raises(ResumeRefused, match="optim.lr: 0.003 -> 0.001"):    # pretrain.train exits 2: not retried
        tr._check_run_identity(4)
    good = (tmp_path / IDENTITY_FILE).read_text()
    (tmp_path / IDENTITY_FILE).unlink()
    (tmp_path / IDENTITY_FILE).mkdir()
    with pytest.raises(RuntimeError, match="cannot be read") as e:          # an I/O error may pass: exit 1
        tr._check_run_identity(4)
    assert not isinstance(e.value, ResumeRefused)
    (tmp_path / IDENTITY_FILE).rmdir()
    (tmp_path / IDENTITY_FILE).write_text(good)
    monkeypatch.setenv("DM_ALLOW_RUN_CHANGE", "1")
    tr._check_run_identity(4)
    assert any("DM_ALLOW_RUN_CHANGE=1" in m for m in tr.logs), tr.logs
    from pretrain.utils import run_identity
    monkeypatch.setattr(run_identity, "run_identity", lambda cfg: 1 / 0)   # a bug in the check itself: never a stop
    tr._check_run_identity(4)
    assert any("could not be checked" in m for m in tr.logs)


def test_identity_record_of_a_resume_is_written_only_after_the_load(tmp_path, monkeypatch):
    """A resume whose state load fails leaves run_identity.json as it was: a run without a record is not given the
    values of a configuration that could not load, and an allowed change is not recorded, so reverting it needs no
    knob. A load that succeeds records."""
    from pretrain.utils.run_identity import IDENTITY_FILE
    rec = tmp_path / IDENTITY_FILE

    def resume(cfg, loads: bool):
        tr = _stub(cfg=cfg, checkpoint_dir=tmp_path, resume=True, global_step=0)
        tr._resume_decision = lambda d: tmp_path / "checkpoint-4"

        def load(ckpt):
            if not loads:                                      # e.g. DCP rejecting another architecture
                raise RuntimeError("Failed to load resume state: Missing key(s) in state_dict")
            tr.global_step = 4
            return ckpt
        tr._load_checkpoint = load
        tr.resume_routine()
        return tr

    with pytest.raises(RuntimeError, match="Missing key"):     # a run without a record, a wrong architecture
        resume(_cfg("--model.d_model", "512"), loads=False)
    assert not rec.exists(), "a configuration that could not load was recorded"
    resume(_cfg(), loads=True)                                  # the right one loads: its values are recorded
    assert json.loads(rec.read_text())["identity"]["model.d_model"] == 1024
    before = rec.read_text()
    monkeypatch.setenv("DM_ALLOW_RUN_CHANGE", "1")
    with pytest.raises(RuntimeError, match="Missing key"):     # an allowed change that does not load
        resume(_cfg("--model.d_model", "512"), loads=False)
    assert rec.read_text() == before
    monkeypatch.delenv("DM_ALLOW_RUN_CHANGE")
    tr = resume(_cfg(), loads=True)                             # reverted: no knob needed
    assert tr.global_step == 4 and rec.read_text() == before
    monkeypatch.setenv("DM_ALLOW_RUN_CHANGE", "1")
    resume(_cfg("--optim.lr", "0.001"), loads=True)             # an allowed change that loads: recorded
    assert json.loads(rec.read_text())["identity"]["optim.lr"] == 0.001


def test_run_identity_compares_data_paths_as_resolved(tmp_path):
    """The same files under another spelling (the repo's workspace symlink, or a path with ..) are the same data; a
    symlink retargeted to other files is a change; a record that kept the spelling (written before paths were
    resolved) still matches its files."""
    from pretrain.utils.run_identity import IDENTITY_FILE, run_identity
    (tmp_path / "dm-workspace" / "data").mkdir(parents=True)
    (tmp_path / "dm-workspace" / "data" / "tok.bin").write_bytes(b"\0" * 8)
    (tmp_path / "workspace").symlink_to(tmp_path / "dm-workspace")
    real = os.path.realpath(tmp_path / "dm-workspace" / "data" / "tok.bin")
    spelled = str(tmp_path / "workspace" / "data" / "tok.bin")
    ck = tmp_path / "ck"
    ck.mkdir()
    _check(ck, _cfg("--data.paths", spelled), None)
    rec = json.loads((ck / IDENTITY_FILE).read_text())
    assert rec["identity"]["data.paths"] == real                # recorded resolved
    assert run_identity(_cfg("--data.paths", f"[{spelled}]"))["data.paths"] == [real]
    for other in (real, str(tmp_path / "workspace" / "data" / ".." / "data" / "tok.bin")):
        assert _check(ck, _cfg("--data.paths", other), 400) == ("ok", []), other
    rec["identity"]["data.paths"] = spelled                    # a record of the earlier format
    (ck / IDENTITY_FILE).write_text(json.dumps(rec))
    assert _check(ck, _cfg("--data.paths", real), 400) == ("ok", [])
    _check(ck, _cfg("--data.paths", spelled), None)
    (tmp_path / "other" / "data").mkdir(parents=True)          # the symlink now points at other files
    (tmp_path / "other" / "data" / "tok.bin").write_bytes(b"\1" * 8)
    (tmp_path / "workspace").unlink()
    (tmp_path / "workspace").symlink_to(tmp_path / "other")
    ok, msgs = _check(ck, _cfg("--data.paths", spelled), 400)
    moved = os.path.realpath(tmp_path / "other" / "data" / "tok.bin")
    assert ok == "refused" and f'data.paths: "{real}" -> "{moved}"' in msgs[-1][1], msgs


def test_an_allow_knob_that_is_not_needed_is_a_warning(tmp_path, monkeypatch):
    """A chain passes its environment to every later job: DM_ALLOW_RUN_CHANGE / DM_ALLOW_KERNEL_CHANGE set for one
    resubmission stay set, and would accept a later accidental change with a warning only. Set while nothing differs,
    each is a warning that says so."""
    _check(tmp_path, _cfg(), None)
    ok, msgs = _check(tmp_path, _cfg(), 400, allow=True)
    assert ok == "ok" and [m for lvl, m in msgs if lvl == "warning" and "not needed" in m], msgs
    ok, msgs = _check(tmp_path, _cfg("--optim.lr", "0.001"), 400, allow=True)   # needed here: no such warning
    assert ok == "ok" and not [m for _, m in msgs if "not needed" in m], msgs
    from pretrain.trainer.base_trainer import kernel_identity
    ck = tmp_path / "checkpoint-4"
    ck.mkdir()
    (ck / "kernel_version.json").write_text(json.dumps(kernel_identity()))      # this process's build
    tr = _stub()
    tr._check_resume_kernel(ck)
    assert not tr.logs
    monkeypatch.setenv("DM_ALLOW_KERNEL_CHANGE", "1")
    tr._check_resume_kernel(ck)
    assert len(tr.logs) == 1 and "DM_ALLOW_KERNEL_CHANGE=1 is set" in tr.logs[0] and "not needed" in tr.logs[0]


def test_ddp_rank_without_a_state_file_of_its_own_loads_rank_0s(tmp_path, monkeypatch, capsys):
    """A DDP checkpoint saved by fewer ranks than the resume has (1 -> 2): rank 1 has no training_state_1.pt and loads
    training_state_0.pt (the replicas are identical); with no file at all it still fails, naming its own."""
    from pretrain.trainer.ddp_trainer import DDPTrainer

    def trainer(seed):
        torch.manual_seed(seed)
        tr = object.__new__(DDPTrainer)
        tr.model = torch.nn.Linear(4, 3)
        tr.optimizer = torch.optim.AdamW(tr.model.parameters(), lr=0.1)
        tr.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(tr.optimizer, lambda s: 1.0 / (1 + s))
        tr.model(torch.randn(2, 4)).sum().backward()
        tr.optimizer.step()
        tr.lr_scheduler.step()
        return tr
    saved = trainer(0)
    saved.global_step = 7
    saved._save_resume_state(tmp_path)                          # rank 0 of a 1-rank run
    assert sorted(p.name for p in tmp_path.iterdir()) == ["training_state_0.pt"]
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    tr = trainer(1)
    assert tr._load_resume_state(tmp_path) == 7
    assert "loading training_state_0.pt" in capsys.readouterr().err
    for a, b in zip(saved.model.parameters(), tr.model.parameters()):
        assert torch.equal(a, b)
    sa, sb = saved.optimizer.state_dict()["state"], tr.optimizer.state_dict()["state"]
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k]["exp_avg"], sb[k]["exp_avg"]) for k in sa)
    assert tr.lr_scheduler.state_dict()["last_epoch"] == saved.lr_scheduler.state_dict()["last_epoch"]
    (tmp_path / "training_state_0.pt").unlink()
    with pytest.raises(FileNotFoundError, match="training_state_1.pt"):
        trainer(1)._load_resume_state(tmp_path)


# ---------------------------------------------------------------- the stop before the first step
class FakeRun:
    def __init__(self, block=None, fail=False):
        self.calls, self.block, self.fail, self.id = [], block, fail, "fake"

    def mark_preempting(self):
        self.calls.append("mark_preempting")

    def log(self, metrics, step=None):
        pass

    def finish(self):
        self.calls.append("finish")
        if self.block is not None:
            self.block.wait(30)
        if self.fail:
            raise RuntimeError("upload failed")


def _looper(tmp_path, step, max_steps, stop=None, deadline=0.0):
    calls = []
    tr = _stub(global_step=step, max_steps=max_steps, global_batch_size=4, chunk_size=8, backend="fsdp2",
               num_devices=1, gradient_accumulation_steps=1, _deadline=deadline, _stop_requested=stop,
               checkpoint_dir=tmp_path, _hf_export_steps=set(), wandb_run=FakeRun(), device=torch.device("cpu"),
               cfg=types.SimpleNamespace(logging_steps=1, hf_save_dir=str(tmp_path / "hf"), save_interval=100,
                                         VERSION_STRING=VERSION, opt_cfg=types.SimpleNamespace(max_grad_norm=1.0)))
    tr.train_step = lambda grad_clip: calls.append("train_step") or {"train/ce_loss": 1.0}
    tr.save_ckpt = lambda: calls.append("save_ckpt")
    tr.save_hf = lambda d: calls.append("save_hf")
    tr._dump_result_summary = lambda: calls.append("summary")
    tr._log_gains = lambda: None
    tr._set_train_state_step = lambda s: None
    return tr, calls


@pytest.mark.parametrize("step", [0, 3])
def test_a_pending_stop_returns_before_the_first_step(tmp_path, step):
    tr, calls = _looper(tmp_path, step, 10, stop="signal 15")
    assert tr.train() == "stopped"
    assert calls == [], f"trained or saved with a stop pending: {calls}"
    stop = [e for e in tr.events if e["event"] == "stop"]
    assert stop == [{"event": "stop", "reason": "signal 15", "before_first_step": True}]
    assert tr.wandb_run.calls == [], "train() closed W&B: pretrain.train does, after the process group"
    assert any(f"Stopping early (signal 15) at step {step}, before the first step" in m for m in tr.logs), tr.logs


def test_a_spent_time_budget_stops_before_the_first_step(tmp_path):
    tr, calls = _looper(tmp_path, 3, 10, deadline=time.time() - 1)
    assert tr.train() == "stopped" and calls == []
    assert [e["reason"] for e in tr.events if e["event"] == "stop"] == ["deadline"]


def test_a_run_resumed_at_max_steps_exports_even_with_a_stop_pending(tmp_path):
    d = tmp_path / "checkpoint-10"
    d.mkdir()
    (d / ".complete").write_text(VERSION)
    tr, calls = _looper(tmp_path, 10, 10, stop="signal 15")
    assert tr.train() == "complete"
    assert calls == ["save_hf", "summary"], calls              # no step, no re-save, the export
    assert any("already complete" in m for m in tr.logs)
    (tmp_path / "fresh").mkdir()
    tr, calls = _looper(tmp_path / "fresh", 8, 10)             # no stop: trains, saves the final checkpoint, exports
    assert tr.train() == "complete" and calls == ["train_step", "train_step", "save_ckpt", "save_hf", "summary"], calls


# ---------------------------------------------------------------- bounded W&B teardown
def test_wandb_finish_is_bounded(monkeypatch):
    from pretrain.trainer.base_trainer import finish_wandb_bounded
    monkeypatch.setenv("DM_WANDB_FINISH_TIMEOUT", "0.5")
    logs = []

    def log(msg, *a, **k):
        logs.append(str(msg))
    gate = threading.Event()
    run = FakeRun(block=gate)
    t0 = time.time()
    assert finish_wandb_bounded(run, log, preempting=True) is False, "a hanging finish() reported as done"
    assert time.time() - t0 < 5, "a hanging finish() was waited for"
    assert run.calls == ["mark_preempting", "finish"]
    assert any("did not return within 0.5 s" in m and "wandb sync" in m for m in logs), logs
    gate.set()
    run = FakeRun(fail=True)
    assert finish_wandb_bounded(run, log) is True                # a failing finish: reported, not hung
    assert run.calls == ["finish"] and any("finish failed" in m for m in logs)
    assert finish_wandb_bounded(types.SimpleNamespace(finish=lambda: None), log, preempting=True)  # no mark_preempting
    assert finish_wandb_bounded(None, log, preempting=True)     # the other ranks, W&B off: nothing to do


# ---------------------------------------------------------------- the end: process group first, then W&B
@pytest.mark.parametrize("status, close, finish, rc, order, says", [
    ("stopped", "pass", "pass", 75, ["close", "mark_preempting", "finish"], None),
    ("complete", "pass", "pass", 0, ["close", "finish"], None),
    ("stopped", "pass", "time.sleep(60)", 75, ["close", "mark_preempting", "finish"], "did not return within 1 s"),
    ("stopped", "time.sleep(60)", "pass", 75, ["close", "mark_preempting", "finish"], "exiting 75 anyway"),
    ("complete", "raise RuntimeError('peer gone')", "pass", 1, ["close", "finish"], "RuntimeError: peer gone"),
])
def test_the_process_group_is_closed_before_the_master_finishes_wandb(status, close, finish, rc, order, says):
    """pretrain.train's end: dist_close first, then the master's W&B finish (marked preempting on a stop), so the other
    ranks never wait in the teardown's barrier for the master's upload (before, a 30-60 s upload made every rank's
    bounded teardown time out). A finish that hangs or a teardown that fails still ends the process with its status."""
    code = (f"import time\nimport pretrain.train as t\nev = []\n"
            f"def fake_close():\n    ev.append('close')\n    {close}\n"
            f"class Run:\n    id = 'fake'\n"
            f"    def mark_preempting(self):\n        ev.append('mark_preempting')\n"
            f"    def finish(self):\n        ev.append('finish')\n        print('ORDER', ev, flush=True)\n        {finish}\n"
            f"t.dist_close = fake_close\nt._close_and_exit({status!r}, wandb_run=Run())\nprint('END', flush=True)")
    env = dict(clean_env(), PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="", DM_DIST_CLOSE_TIMEOUT="1",
               DM_WANDB_FINISH_TIMEOUT="1", PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert p.returncode == rc, p.stdout[-2000:] + p.stderr[-3000:]
    assert f"ORDER {order}" in p.stdout, p.stdout[-2000:] + p.stderr[-2000:]
    assert ("END" in p.stdout) == (rc == 0), p.stdout[-2000:]
    if says:
        assert says in p.stderr, p.stderr[-2000:]
    assert time.time() - t0 < 120


# ---------------------------------------------------------------- exit 75 whatever dist_close does
@pytest.mark.parametrize("close, status, rc, says", [
    ("raise RuntimeError('Connection reset by peer')", "stopped", 75, "failed (RuntimeError: Connection reset by peer)"),
    ("time.sleep(60)", "stopped", 75, "did not return within 1 s"),
    ("pass", "stopped", 75, None),
    ("pass", "complete", 0, None),
])
def test_stopped_process_exits_75_whatever_dist_close_does(close, status, rc, says):
    code = ("import time\nimport pretrain.train as t\n"
            f"def fake():\n    {close}\n"
            f"t.dist_close = fake\nt._close_and_exit({status!r})\nprint('NOT REACHED' if {status!r} == 'stopped' else 'END')")
    env = dict(clean_env(), PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="", DM_DIST_CLOSE_TIMEOUT="1",
               PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert p.returncode == rc, p.stdout[-2000:] + p.stderr[-3000:]
    assert "NOT REACHED" not in p.stdout
    if says:
        assert says in p.stderr and "exiting 75 anyway" in p.stderr, p.stderr[-2000:]
    assert time.time() - t0 < 120


# ---------------------------------------------------------------- replay warning, result summary
def test_replay_warning(tmp_path):
    m = tmp_path / "r.metrics.jsonl"
    recs = [{"event": "start", "step": 0}] + [{"event": "metrics", "step": s, "train/ce_loss": 1.0} for s in range(21)]
    m.write_text("\n".join(json.dumps(r) for r in recs) + "\n{\"event\": \"metrics\", \"st")   # a line cut short
    tr = _stub(_metrics_path=m, cfg=types.SimpleNamespace(run_name="r"))
    tr._replay_warning(21)                                      # a deadline stop resumes past the last logged step
    assert not tr.logs
    tr._replay_warning(12)                                      # a crash resumed from an older checkpoint
    assert len(tr.logs) == 1 and "steps 12..20 are replayed" in tr.logs[0] and "W&B" in tr.logs[0], tr.logs
    tr = _stub(_metrics_path=tmp_path / "missing.jsonl", cfg=types.SimpleNamespace(run_name="r"))
    tr._replay_warning(0)
    assert not tr.logs


def test_summary_of_a_process_that_trained_no_step_keeps_the_final_metrics(tmp_path):
    cfg = _cfg("--log_root_dir", str(tmp_path / "logs"), "--result_root_dir", str(tmp_path / "result"))
    out = tmp_path / "result" / "r" / "train_summary.json"
    tr = _stub(cfg=cfg, global_step=10, wandb_run=None, _last_metrics={"train/ce_loss": 2.0}, _trained_steps=10)
    tr._dump_result_summary()
    assert json.loads(out.read_text())["final_metrics"] == {"train/ce_loss": 2.0}
    tr = _stub(cfg=cfg, global_step=10, wandb_run=None, _last_metrics={}, _trained_steps=0, _resume_state={})
    tr._dump_result_summary()                                   # no sidecar: the existing summary's metrics are kept
    assert json.loads(out.read_text())["final_metrics"] == {"train/ce_loss": 2.0}
    tr = _stub(cfg=cfg, global_step=10, wandb_run=None, _last_metrics={}, _trained_steps=0,
               _resume_state={"last_metrics": {"train/ce_loss": 1.5}}, _nonfinite_steps=4)
    tr._dump_result_summary()                                   # the resumed checkpoint's sidecar first
    s = json.loads(out.read_text())
    assert s["final_metrics"] == {"train/ce_loss": 1.5} and s["skipped_updates"] == 4


# ---------------------------------------------------------------- decisions agreed across ranks
def _ckpt(tmp_path, step=5):
    d = tmp_path / f"checkpoint-{step}"
    d.mkdir()
    (d / ".complete").write_text(VERSION)
    from pretrain.trainer.base_trainer import kernel_identity
    (d / "kernel_version.json").write_text(json.dumps(kernel_identity()))   # this process's build: the check passes
    torch.save({"input_ids": torch.arange(4)}, d / "next_train_data_0.pt")
    return d


def _resumer(tmp_path, step=5):
    tr = _stub(cfg=types.SimpleNamespace(VERSION_STRING=VERSION, force_resume_check=True), global_step=0)
    tr._load_resume_state = lambda d: step
    tr.jump_to_step = lambda s: setattr(tr, "global_step", s)
    return tr


def test_load_raises_what_another_rank_reports(tmp_path, monkeypatch):
    """The ranks' loaded steps and load errors are gathered: a failure or a different step on any rank raises on
    this one too (simulated by the gathered values of a second rank)."""
    from pretrain.trainer import base_trainer
    _ckpt(tmp_path)
    real = base_trainer.all_gather_objects
    peer = {"value": (3, None)}

    def gather(obj):
        if isinstance(obj, tuple) and len(obj) == 2 and (obj[0] is None or isinstance(obj[0], int)) and \
                not isinstance(obj[1], bool):
            return [obj, peer["value"]]                         # the loaded-step gather: rank 1 differs
        return real(obj) + real(obj)
    monkeypatch.setattr(base_trainer, "all_gather_objects", gather)
    with pytest.raises(RuntimeError, match=r"different steps .*\[5, 3\]"):
        _resumer(tmp_path).load(tmp_path)
    peer["value"] = (None, "rank 1: FileNotFoundError: __1_0.distcp")
    with pytest.raises(RuntimeError, match="Failed to load resume state.*rank 1: FileNotFoundError"):
        _resumer(tmp_path).load(tmp_path)
    peer["value"] = (5, None)
    assert _resumer(tmp_path).load(tmp_path) == tmp_path / "checkpoint-5"


def test_master_errors_reach_the_other_ranks(tmp_path, monkeypatch):
    from pretrain.trainer import base_trainer
    tr = _stub()
    with pytest.raises(ZeroDivisionError):                      # the master raises its own exception ...
        tr._on_master(lambda: 1 / 0, "dividing")
    assert any("dividing failed on rank 0" in m for m in tr.logs)
    monkeypatch.setenv("RANK", "1")                             # ... and the other ranks the message it broadcast
    monkeypatch.setattr(base_trainer, "broadcast_object",
                        lambda obj: (None, "choosing the checkpoint failed on rank 0: CheckpointMarkerUnreadable: x"))
    with pytest.raises(RuntimeError, match="failed on rank 0: CheckpointMarkerUnreadable"):
        _stub()._on_master(lambda: pytest.fail("ran on rank 1"), "choosing the checkpoint")


def test_an_unreadable_newest_marker_is_never_skipped_by_load(tmp_path, monkeypatch):
    from pretrain.utils import ckpt
    from pretrain.utils.ckpt import CheckpointMarkerUnreadable
    _ckpt(tmp_path, 5)
    _ckpt(tmp_path, 10)
    real = open

    def fake(path, *a, **k):
        if os.fspath(path).endswith("checkpoint-10/.complete"):
            raise OSError(5, "Input/output error")
        return real(path, *a, **k)
    monkeypatch.setattr(ckpt, "open", fake, raising=False)
    tr = _resumer(tmp_path)
    with pytest.raises(CheckpointMarkerUnreadable):
        tr.load(tmp_path)
    assert tr.global_step == 0, "fell back to checkpoint-5"
    with pytest.raises(CheckpointMarkerUnreadable):
        tr.load(tmp_path, global_step=10)
    assert tr.load(tmp_path, global_step=5) == tmp_path / "checkpoint-5" and tr.global_step == 5


def test_sanity_check_verdicts_are_agreed(tmp_path, monkeypatch):
    from pretrain.trainer import base_trainer
    _ckpt(tmp_path)
    tr = _resumer(tmp_path)
    tr.load(tmp_path)
    tr._resume_sanity_check({"input_ids": torch.arange(4)})
    assert any("Resume sanity check PASSED" in m for m in tr.logs)
    tr = _resumer(tmp_path)
    tr.load(tmp_path)
    with pytest.raises(RuntimeError, match="sanity check FAILED.*rank 0"):
        tr._resume_sanity_check({"input_ids": torch.arange(1, 5)})
    tr = _resumer(tmp_path)
    tr.load(tmp_path)
    monkeypatch.setattr(base_trainer, "all_gather_objects", lambda obj: [obj, ("failed", "the next batch differs")])
    with pytest.raises(RuntimeError, match="sanity check FAILED.*rank 1"):     # this rank passed, rank 1 did not
        tr._resume_sanity_check({"input_ids": torch.arange(4)})


def test_kernel_refusals_are_final_unless_the_record_could_not_be_read(monkeypatch):
    """A kernel-build refusal no retry can change (another build, no record, a garbled record) raises ResumeRefused
    (pretrain.train exits 2); a record the master could not read (an I/O error) a plain RuntimeError (exit 1)."""
    from pretrain.trainer.base_trainer import _kernel_verdict
    kv = {"code_sha": "9fe8da5f743a", "bf16_dp": False}
    assert _kernel_verdict({"text": json.dumps(kv)}, kv, "ck", allow=False) == (None, False, True)
    msg, refuse, det = _kernel_verdict({"text": json.dumps(dict(kv, code_sha="5905517248c5"))}, kv, "ck", allow=False)
    assert refuse and det and "was trained with the delta_matching kernel" in msg
    for rec in ({"missing": True}, {"text": "{garbled"}):
        msg, refuse, det = _kernel_verdict(rec, kv, "ck", allow=False)
        assert refuse and det and "cannot tell which delta_matching build" in msg, rec
    msg, refuse, det = _kernel_verdict({"error": "OSError: [Errno 5] Input/output error"}, kv, "ck", allow=False)
    assert refuse and not det and "Input/output error" in msg
    assert _kernel_verdict({"missing": True}, None, "ck", allow=False) == (None, False, True)   # no kernel: no check
    msg, refuse, _ = _kernel_verdict({"missing": True}, kv, "ck", allow=True)
    assert msg and not refuse                                                               # allowed: a warning
