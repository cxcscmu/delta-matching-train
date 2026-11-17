"""CPU unit checks of the checkpoint files a chain of 4-hour jobs relies on (src/pretrain/utils/ckpt.py, the trainer's
save path, src/pretrain/hf/export.py and check_export.py), with the model and the backend writes stubbed out:
  * marker states: absent / complete / mismatch / unreadable (an I/O error is retried, then reported, never taken for
    "absent"); the newest-checkpoint search raises on an unreadable marker instead of falling back to an older one, and
    its error says how to set such a checkpoint aside (`.bad-<name>`, which nothing reads or removes);
  * rotation keeps a directory whose marker cannot be read, removes victims through a `.trash-` rename (a half-deleted
    directory never matches checkpoint-*), and the startup cleanup removes what an interrupted removal left;
  * every file the resume needs is fsynced before the marker; the progress line follows the marker;
  * the sidecar trainer_state.json: written with the save, read with defaults (an older checkpoint has none);
  * the HF export is atomic with config.json last (a failing weights write leaves no config.json; a re-export into an
    existing export replaces only its own files), and check_export tells a whole export from a cut one without
    importing torch."""
import errno, json, os, struct, subprocess, sys, time, types
from pathlib import Path

import pytest
import torch

from _helpers import REPO, clean_env

VERSION = "pt-v0.1"


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    for k in ("DM_PROGRESS_FILE", "SLURM_JOB_ID", "DM_STOP_BEFORE", "DM_ALLOW_RUN_CHANGE", "DM_ALLOW_KERNEL_CHANGE"):
        monkeypatch.delenv(k, raising=False)
    from pretrain.utils import ckpt
    monkeypatch.setattr(ckpt, "MARKER_RETRY_DELAY", 0.0)


def _eio(monkeypatch, failing: dict) -> list:
    """Reads (through the ckpt module's open) of a path ending in a key of `failing` raise EIO: the next n times, or
    always for n = -1 (setting failing[key] = 0 later ends it). Returns the list of failed paths."""
    import builtins
    from pretrain.utils import ckpt
    real, calls = builtins.open, []

    def fake(path, *a, **k):
        p = os.fspath(path)
        for suffix, n in failing.items():
            if p.endswith(suffix) and n != 0:
                failing[suffix] = n - 1 if n > 0 else n
                calls.append(p)
                raise OSError(errno.EIO, "Input/output error", p)
        return real(path, *a, **k)

    monkeypatch.setattr(ckpt, "open", fake, raising=False)
    return calls


def _mk(root, step, marker=VERSION):
    d = Path(root) / f"checkpoint-{step}"
    d.mkdir(parents=True, exist_ok=True)
    if marker is not None:
        (d / ".complete").write_text(marker)
    return d


def _stub(**attrs):
    from pretrain.trainer.base_trainer import BaseTrainer
    tr = object.__new__(BaseTrainer)
    tr.logs = []
    tr.main_log = lambda msg, *a, **k: tr.logs.append(str(msg))
    tr._metrics_event = lambda *a, **k: None
    for k, v in attrs.items():
        setattr(tr, k, v)
    return tr


def _saver(root, limit, **cfg):
    tr = _stub(cfg=types.SimpleNamespace(save_total_limit=limit, VERSION_STRING=VERSION, **cfg),
               checkpoint_dir=Path(root), global_step=0)
    tr._save_resume_state = lambda d: (d / "state").write_text("x")
    tr._save_full_model = lambda d: None
    tr._save_dataloader_state = lambda d: None
    return tr


# ---------------------------------------------------------------- marker states
def test_marker_state_absent_complete_mismatch_unreadable(tmp_path, monkeypatch):
    from pretrain.utils.ckpt import marker_state, marker_state_why, check_complete_marker, MARKER_TRIES
    d = _mk(tmp_path, 3, marker=None)
    assert marker_state(d, VERSION) == "absent" and marker_state(tmp_path / "nope", VERSION) == "absent"
    (d / ".complete").write_text(VERSION + "\n")
    assert marker_state(d, VERSION) == "complete" and check_complete_marker(d, VERSION)
    for other in ("pt-v0.2", ""):
        (d / ".complete").write_text(other)
        assert marker_state(d, VERSION) == "mismatch"
    (d / ".complete").unlink()
    (d / ".complete").mkdir()                                   # not a marker file
    assert marker_state(d, VERSION) == "mismatch"
    (d / ".complete").rmdir()
    (d / ".complete").write_text(VERSION)
    calls = _eio(monkeypatch, {"checkpoint-3/.complete": 2})    # a transient error: retried, then read
    assert marker_state(d, VERSION) == "complete" and len(calls) == 2
    calls = _eio(monkeypatch, {"checkpoint-3/.complete": -1})   # a persistent one: reported, never "absent"
    state, why = marker_state_why(d, VERSION)
    assert state == "unreadable" and "Input/output error" in why and len(calls) == MARKER_TRIES
    assert not check_complete_marker(d, VERSION)                # the bool helper: not complete


def test_newest_checkpoint_search_never_falls_back_past_an_unreadable_marker(tmp_path, monkeypatch):
    from pretrain.utils.ckpt import find_last_complete_checkpoint, CheckpointMarkerUnreadable
    _mk(tmp_path, 3), _mk(tmp_path, 6), _mk(tmp_path, 9, marker=None), _mk(tmp_path, 12, marker="pt-v9")
    (tmp_path / ".trash-checkpoint-15.123").mkdir()
    (tmp_path / ".trash-checkpoint-15.123" / ".complete").write_text(VERSION)   # a half-removed one: never a candidate
    assert find_last_complete_checkpoint(tmp_path, VERSION) == tmp_path / "checkpoint-6"
    _eio(monkeypatch, {"checkpoint-6/.complete": -1})
    with pytest.raises(CheckpointMarkerUnreadable, match="checkpoint-6.*cannot be read") as e:
        find_last_complete_checkpoint(tmp_path, VERSION)
    assert find_last_complete_checkpoint(tmp_path / "missing", VERSION) is None
    # a marker that stays unreadable: the error says how to set the checkpoint aside; done, the search (and the
    # startup cleanup, which removes only .trash-*) leaves it alone and the run resumes from the older one
    bad = tmp_path / ".bad-checkpoint-6"
    assert f"mv {tmp_path / 'checkpoint-6'} {bad}" in str(e.value), str(e.value)
    os.rename(tmp_path / "checkpoint-6", bad)
    from pretrain.utils.ckpt import cleanup_trash
    cleanup_trash(tmp_path)
    assert bad.is_dir() and find_last_complete_checkpoint(tmp_path, VERSION) == tmp_path / "checkpoint-3"


# ---------------------------------------------------------------- rotation, trash, startup cleanup
def test_rotation_keeps_a_checkpoint_whose_marker_cannot_be_read(tmp_path, monkeypatch):
    tr = _saver(tmp_path, 1)
    tr.global_step = 3
    tr.save_ckpt()
    failing = {"checkpoint-3/.complete": -1}
    _eio(monkeypatch, failing)
    tr.global_step = 6
    tr.save_ckpt()                                              # limit 1, but checkpoint-3 may be the newest: kept
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == ["checkpoint-3", "checkpoint-6"]
    assert any("keeping" in m and "checkpoint-3" in m for m in tr.logs), tr.logs
    failing["checkpoint-3/.complete"] = 0                       # the file system recovered
    tr.global_step = 9
    tr.save_ckpt()                                              # readable again: rotated as usual
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == ["checkpoint-9"]
    assert not list(tmp_path.glob(".trash-*"))


def test_rotation_never_leaves_a_complete_looking_remnant(tmp_path, monkeypatch):
    """A removal cut short (here: the rmtree does nothing) leaves a `.trash-` directory, which neither the resume nor
    the launchers' checkpoint-* globs see; the master removes it at the next startup."""
    from pretrain.utils import ckpt
    from pretrain.utils.ckpt import find_last_complete_checkpoint, cleanup_trash, trash_checkpoint
    removed, real_rmtree = [], ckpt.shutil.rmtree
    monkeypatch.setattr(ckpt.shutil, "rmtree", lambda p, ignore_errors=False: removed.append(Path(p).name))
    tr = _saver(tmp_path, 1)
    for step in (3, 6):
        tr.global_step = step
        tr.save_ckpt()
    assert sorted(p.name for p in tmp_path.glob("checkpoint-*")) == ["checkpoint-6"]
    trash = list(tmp_path.glob(".trash-checkpoint-3.*"))
    assert len(trash) == 1 and (trash[0] / ".complete").exists() and removed == [trash[0].name], (trash, removed)
    assert find_last_complete_checkpoint(tmp_path, VERSION) == tmp_path / "checkpoint-6"
    monkeypatch.setattr(ckpt.shutil, "rmtree", real_rmtree)
    st = _stub(cfg=types.SimpleNamespace(ckpt_dir=str(tmp_path), log_dir=str(tmp_path / "logs")))
    st._setup_dirs()                                            # the master's startup cleanup
    assert not list(tmp_path.glob(".trash-*")) and (tmp_path / "checkpoint-6").exists()
    assert any("interrupted rotation" in m for m in st.logs), st.logs
    assert cleanup_trash(tmp_path / "missing") == []
    # the rename itself fails: the marker goes first, so even a removal cut short leaves nothing complete-looking
    d = _mk(tmp_path, 9)
    monkeypatch.setattr(ckpt.os, "rename", lambda a, b: (_ for _ in ()).throw(OSError(errno.EXDEV, "cross-device")))
    monkeypatch.setattr(ckpt.shutil, "rmtree", lambda p, ignore_errors=False: None)
    assert "marker was removed first" in trash_checkpoint(d) and not (d / ".complete").exists()
    d = _mk(tmp_path, 12)                                       # ... and if even the marker cannot go: kept as it is
    monkeypatch.setattr(ckpt.os, "unlink", lambda p: (_ for _ in ()).throw(OSError(errno.EIO, "Input/output error")))
    assert "kept" in trash_checkpoint(d) and (d / ".complete").exists()


def test_rotation_failure_never_raises(tmp_path, monkeypatch):
    from pretrain.utils import ckpt
    tr = _saver(tmp_path, 1)
    tr.global_step = 3
    tr.save_ckpt()
    monkeypatch.setattr(ckpt, "marker_state_why", lambda *a, **k: (_ for _ in ()).throw(OSError(errno.EIO, "eio")))
    from pretrain.trainer import base_trainer
    monkeypatch.setattr(base_trainer, "marker_state_why", ckpt.marker_state_why)
    tr.global_step = 6
    tr.save_ckpt()                                              # rotation fails: the save still completes
    assert (tmp_path / "checkpoint-6" / ".complete").exists() and (tmp_path / "checkpoint-3").exists()
    assert any("rotation" in m and "failed" in m for m in tr.logs), tr.logs


# ---------------------------------------------------------------- durability, progress line, sidecar
def test_every_file_the_resume_needs_is_fsynced_before_the_marker(tmp_path, monkeypatch):
    from pretrain.modeling.config import ModelConfig
    from pretrain.modeling.model import PretrainLM
    synced = []
    real_fsync = os.fsync

    def fsync(fd):
        synced.append(os.path.basename(os.readlink(f"/proc/self/fd/{fd}")))
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    mc = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16)
    model = types.SimpleNamespace(save_config=lambda d: PretrainLM.save_config(types.SimpleNamespace(cfg=mc), d))
    tr = _stub(cfg=types.SimpleNamespace(save_total_limit=None, VERSION_STRING=VERSION, run_name="r"),
               checkpoint_dir=tmp_path, global_step=4, model=model, _nonfinite_steps=2, _last_metrics={"train/ce_loss": 3.5})
    tr._save_resume_state = lambda d: None
    tr._gather_full_state_dict = lambda: {"w": torch.zeros(2)}
    tr.peek_next_train_data = lambda: {"input_ids": torch.arange(4)}
    tr.save_ckpt()
    marker = next(i for i, n in enumerate(synced) if n.startswith(".complete."))
    before = synced[:marker]
    for need in ("pytorch_model.bin", "model_config.json", "next_train_data_0.pt"):
        assert need in before, (need, synced)
    for tmp_prefix in (".kernel_version.json.", ".trainer_state.json."):    # atomic writes: fsynced under their tmp name
        assert any(n.startswith(tmp_prefix) for n in before), (tmp_prefix, synced)
    st = json.loads((tmp_path / "checkpoint-4" / "trainer_state.json").read_text())
    assert (st["global_step"], st["world_size"], st["skipped_updates"]) == (4, 1, 2)
    assert st["last_metrics"] == {"train/ce_loss": 3.5}
    mode = (tmp_path / "checkpoint-4" / "trainer_state.json").stat().st_mode & 0o777
    assert mode & 0o044, f"the sidecar is not readable like the other files ({oct(mode)})"


def test_progress_line_follows_the_marker(tmp_path, monkeypatch):
    from pretrain.trainer import base_trainer
    prog = tmp_path / "chain" / "progress"
    prog.parent.mkdir()
    monkeypatch.setenv("DM_PROGRESS_FILE", str(prog))
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    real = base_trainer.append_progress
    seen = []

    def spy(kind, run, detail=""):
        seen.append((tmp_path / "ck" / f"checkpoint-{detail}" / ".complete").exists())
        real(kind, run, detail)

    monkeypatch.setattr(base_trainer, "append_progress", spy)
    tr = _saver(tmp_path / "ck", 2, run_name="runA")
    t0 = int(time.time())
    for step in (3, 6):
        tr.global_step = step
        tr.save_ckpt()
    assert seen == [True, True], "the progress line was appended before the marker"
    lines = [line.split() for line in prog.read_text().splitlines()]
    assert [ln[1:] for ln in lines] == [["777", "ckpt", "runA", "3"], ["777", "ckpt", "runA", "6"]], lines
    assert all(t0 - 5 <= int(ln[0]) <= time.time() + 5 for ln in lines)
    monkeypatch.delenv("SLURM_JOB_ID")
    real("ckpt", "runA", 9)
    assert prog.read_text().splitlines()[-1].split()[1:] == ["0", "ckpt", "runA", "9"]
    monkeypatch.setenv("DM_PROGRESS_FILE", str(tmp_path / "no" / "such" / "dir" / "progress"))
    tr.global_step = 9
    tr.save_ckpt()                                              # an unwritable progress file never fails a save
    assert (tmp_path / "ck" / "checkpoint-9" / ".complete").exists()
    monkeypatch.delenv("DM_PROGRESS_FILE")
    tr.global_step = 12
    tr.save_ckpt()
    assert len(prog.read_text().splitlines()) == 3


def _loader(tr, step):
    tr._load_resume_state = lambda d: step
    tr.jump_to_step = lambda s: setattr(tr, "global_step", s)
    return tr


def test_sidecar_round_trip_and_defaults(tmp_path, monkeypatch):
    """skipped_updates and the last metrics come back on resume; a checkpoint without the sidecar (an older one) or
    with an unreadable or mismatched one resumes with the defaults; a world-size change skips the per-rank check."""
    tr = _saver(tmp_path, None, run_name="r")
    tr.global_step, tr._nonfinite_steps, tr._last_metrics = 5, 3, {"train/ce_loss": 2.25}
    tr.save_ckpt()
    torch.save({"input_ids": torch.arange(4)}, tmp_path / "checkpoint-5" / "next_train_data_0.pt")
    side = tmp_path / "checkpoint-5" / "trainer_state.json"

    def resume():
        t = _loader(_saver(tmp_path, None, run_name="r"), 5)
        assert t.load(tmp_path) == tmp_path / "checkpoint-5" and t.global_step == 5
        return t

    t = resume()
    assert t._nonfinite_steps == 3 and t._resume_state["last_metrics"] == {"train/ce_loss": 2.25}
    assert t.resume_sanity_check_done is False and torch.equal(t.next_train_data_for_check["input_ids"], torch.arange(4))
    rec = json.loads(side.read_text())
    side.write_text(json.dumps(dict(rec, world_size=2)))        # saved by 2 ranks, resumed by 1
    t = resume()
    assert t.resume_sanity_check_done is True and any("saved by 2 ranks" in m for m in t.logs), t.logs
    side.write_text(json.dumps(dict(rec, global_step=4)))       # not this checkpoint's: ignored
    t = resume()
    assert t._nonfinite_steps == 0 and any("ignored" in m for m in t.logs)
    side.write_text("{cut sh")                                   # unreadable: defaults, with a warning
    t = resume()
    assert t._nonfinite_steps == 0 and t._resume_state == {} and any("cannot be read" in m for m in t.logs)
    side.unlink()                                                # an older checkpoint: no sidecar, no warning
    t = resume()
    assert t._nonfinite_steps == 0 and t.resume_sanity_check_done is False and not any("[resume]" in m for m in t.logs)


def test_unreadable_next_batch_file_only_skips_the_sanity_check(tmp_path):
    tr = _saver(tmp_path, None, force_resume_check=True)
    tr.global_step = 5
    tr.save_ckpt()
    (tmp_path / "checkpoint-5" / "next_train_data_0.pt").write_bytes(b"")   # empty after a crash: EOFError
    t = _loader(_saver(tmp_path, None, force_resume_check=True), 5)
    assert t.load(tmp_path) == tmp_path / "checkpoint-5"
    assert "cannot be read" in t._resume_check_note
    t._resume_sanity_check({"input_ids": torch.arange(4)})       # force_resume_check: still no raise
    assert t.resume_sanity_check_done and any("SKIPPED" in m for m in t.logs), t.logs


# ---------------------------------------------------------------- the HF export and its check
def _sd(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"model.embed_tokens.weight": torch.randn(64, 32, generator=g), "model.norm.weight": torch.ones(32),
            "lm_head.weight": torch.randn(64, 32, generator=g)}


def _mc():
    from pretrain.modeling.config import ModelConfig
    return ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1, max_seq_len=16)


class _Tok:
    def save_pretrained(self, d):
        Path(d, "tokenizer_config.json").write_text("{}")
        Path(d, "tokenizer.model").write_bytes(b"tok")


def test_export_is_atomic_with_config_json_last(tmp_path, monkeypatch):
    import safetensors.torch as st
    from pretrain.hf.export import save_hf_checkpoint
    from pretrain.hf.check_export import export_is_whole
    out = tmp_path / "hf"
    real = st.save_file

    def full_disk(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(st, "save_file", full_disk)
    with pytest.raises(OSError):
        save_hf_checkpoint(_sd(), _mc(), str(out), tokenizer=_Tok())
    assert not (out / "config.json").exists() and sorted(os.listdir(out)) == [], os.listdir(out)
    monkeypatch.setattr(st, "save_file", real)
    order = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda a, b: (order.append(os.path.basename(b)), real_replace(a, b))[1])
    save_hf_checkpoint(_sd(), _mc(), str(out), tokenizer=_Tok())
    assert order[-1] == "config.json" and set(order) == {"model.safetensors", "tokenizer_config.json",
                                                          "tokenizer.model", "config.json"}, order
    assert export_is_whole(out)[0] and not [n for n in os.listdir(out) if n.startswith(".export-tmp.")]
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_type"] == "dmlm" and cfg["d_model"] == 32
    # a re-export into the finished export (the next job of a run resumed at max_steps): only its own files change
    for extra, text in (("TRAIN_COMPLETE", "done"), ("valce.txt", "x"), ("config.train.json", "{}")):
        (out / extra).write_text(text)
    (out / "step2000").mkdir()
    (out / "step2000" / "config.json").write_text("{}")
    (out / ".export-tmp.dead").mkdir()                          # the staging dir of an export that was killed
    monkeypatch.setattr(st, "save_file", full_disk)
    with pytest.raises(OSError):
        save_hf_checkpoint(_sd(1), _mc(), str(out))
    assert not (out / "config.json").exists() and not export_is_whole(out)[0], "a failed re-export still looks whole"
    monkeypatch.setattr(st, "save_file", real)
    save_hf_checkpoint(_sd(1), _mc(), str(out))
    assert export_is_whole(out)[0]
    w = st.load_file(str(out / "model.safetensors"))
    assert torch.equal(w["model.embed_tokens.weight"], _sd(1)["model.embed_tokens.weight"].to(torch.bfloat16))
    assert "lm_head.weight" not in w                            # tied: dropped
    for extra in ("TRAIN_COMPLETE", "valce.txt", "config.train.json", "step2000/config.json", "tokenizer.model"):
        assert (out / extra).exists(), extra
    assert not [n for n in os.listdir(out) if n.startswith(".export-tmp.")]


def _safetensors(path, tensors=2, cut=0, extra=b""):
    hdr, off = {"__metadata__": {"format": "pt"}}, 0
    for i in range(tensors):
        hdr[f"t{i}"] = {"dtype": "F32", "shape": [4], "data_offsets": [off, off + 16]}
        off += 16
    h = json.dumps(hdr).encode()
    blob = struct.pack("<Q", len(h)) + h + b"\0" * off + extra
    Path(path).write_bytes(blob[:len(blob) - cut] if cut else blob)


@pytest.mark.parametrize("case, whole, why", [
    ("whole", True, "2 tensors"),
    ("truncated", False, "cut short"),
    ("trailing", False, "its header says"),
    ("no_config", False, "config.json is missing"),
    ("bad_config", False, "does not parse"),
    ("list_config", False, "not a JSON object"),
    ("no_model", False, "model.safetensors is missing"),
    ("tiny_model", False, "shorter than the 8-byte"),
    ("huge_header", False, "header length says"),
    ("bad_header", False, "header does not parse"),
    ("no_tensors", False, "no tensor"),
    ("bad_offsets", False, "no valid data_offsets"),
])
def test_check_export(tmp_path, case, whole, why):
    from pretrain.hf.check_export import export_is_whole, main
    d = tmp_path / "exp"
    d.mkdir()
    m = d / "model.safetensors"
    (d / "config.json").write_text('{"model_type": "dmlm"}')
    _safetensors(m)
    if case == "truncated":
        _safetensors(m, cut=10)
    elif case == "trailing":
        _safetensors(m, extra=b"junk")
    elif case == "no_config":
        (d / "config.json").unlink()
    elif case == "bad_config":
        (d / "config.json").write_text('{"model_type": ')
    elif case == "list_config":
        (d / "config.json").write_text("[1, 2]")
    elif case == "no_model":
        m.unlink()
    elif case == "tiny_model":
        m.write_bytes(b"abc")
    elif case == "huge_header":
        m.write_bytes(struct.pack("<Q", 10 ** 6) + b"{}")
    elif case == "bad_header":
        m.write_bytes(struct.pack("<Q", 4) + b"{no}")
    elif case == "no_tensors":
        _safetensors(m, tensors=0)
    elif case == "bad_offsets":
        h = json.dumps({"t": {"dtype": "F32", "shape": [1], "data_offsets": [8, 4]}}).encode()
        m.write_bytes(struct.pack("<Q", len(h)) + h + b"\0" * 8)
    ok, reason = export_is_whole(d)
    assert ok is whole and why in reason, reason
    assert main([str(d)]) == (0 if whole else 1)


def test_check_export_cli_imports_no_torch(tmp_path):
    d = tmp_path / "exp"
    d.mkdir()
    (d / "config.json").write_text("{}")
    _safetensors(d / "model.safetensors")
    env = dict(clean_env(), PYTHONPATH=str(REPO / "src"))
    code = ("import runpy, sys\nsys.argv = ['check_export', sys.argv[1]]\ntry:\n"
            "    runpy.run_module('pretrain.hf.check_export', run_name='__main__', alter_sys=True)\n"
            "except SystemExit as e:\n    rc = e.code\n"
            "print('RC', rc, 'TORCH', 'torch' in sys.modules, 'TRANSFORMERS', 'transformers' in sys.modules)")
    p = subprocess.run([sys.executable, "-c", code, str(d)], capture_output=True, text=True, env=env, timeout=120)
    assert "export whole" in p.stdout and "RC 0 TORCH False TRANSFORMERS False" in p.stdout, p.stdout + p.stderr
    (d / "config.json").unlink()
    p = subprocess.run([sys.executable, "-m", "pretrain.hf.check_export", str(d)], capture_output=True, text=True,
                       env=env, timeout=120)
    assert p.returncode == 1 and "config.json is missing" in p.stdout, p.stdout + p.stderr
    p = subprocess.run([sys.executable, "-m", "pretrain.hf.check_export"], capture_output=True, text=True, env=env,
                       timeout=120)
    assert p.returncode == 2 and "usage" in p.stderr


def test_hf_package_names_still_import():
    import pretrain.hf as hf
    from pretrain.hf import register_auto, save_hf_checkpoint, DMLMConfig, DMLMForCausalLM  # noqa: F401
    assert callable(register_auto) and callable(save_hf_checkpoint)
    with pytest.raises(AttributeError):
        hf.no_such_name  # noqa: B018
