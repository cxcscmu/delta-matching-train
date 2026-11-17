"""CPU unit checks of the trainer's checkpoint and data-stream logic (src/pretrain/trainer/base_trainer.py,
src/pretrain/data), with the model and the backend writes stubbed out: checkpoint rotation keeps exactly
save_total_limit complete checkpoints (the new one included) and never touches a hand-made directory; the data stream
is the same with and without a save and after a resume, across epoch boundaries and with persistent loader workers; an
epoch without a batch is refused; the document-aware loader refuses a stream that does not match its doc-length
sidecar. The GPU tests (test_resume.py, test_launcher.py) run the same paths end to end."""
import types
from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn

T = 4   # tokens per sample in the stream tests


@pytest.fixture(autouse=True)
def _single_rank(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")
    # a save appends a progress line to $DM_PROGRESS_FILE: a test run inside a chain job must never fake its progress
    monkeypatch.delenv("DM_PROGRESS_FILE", raising=False)


def _stub_trainer(**attrs):
    from pretrain.trainer.base_trainer import BaseTrainer
    tr = object.__new__(BaseTrainer)
    tr.main_log = lambda *a, **k: None
    tr._metrics_event = lambda *a, **k: None
    for k, v in attrs.items():
        setattr(tr, k, v)
    return tr


def _saver(root, limit):
    tr = _stub_trainer(cfg=types.SimpleNamespace(save_total_limit=limit, VERSION_STRING="pt-v0.1"), checkpoint_dir=Path(root),
                       global_step=0)
    tr._save_resume_state = lambda d: (d / "state").write_text("x")
    tr._save_full_model = lambda d: None
    tr._save_dataloader_state = lambda d: None
    return tr


def _listing(root):
    return sorted(p.name for p in Path(root).glob("checkpoint-*"))


@pytest.mark.parametrize("limit,kept", [(1, [9]), (2, [6, 9]), (None, [3, 6, 9])])
def test_checkpoint_rotation_keeps_exactly_the_limit(tmp_path, limit, kept):
    tr = _saver(tmp_path, limit)
    for step in (3, 6, 9):
        tr.global_step = step
        tr.save_ckpt()
    assert _listing(tmp_path) == sorted(f"checkpoint-{s}" for s in kept)
    assert all((tmp_path / f"checkpoint-{s}" / ".complete").exists() for s in kept)


def test_checkpoint_rotation_removes_stale_partials_and_skips_foreign_dirs(tmp_path):
    (tmp_path / "checkpoint-4").mkdir()          # a save killed before its marker
    (tmp_path / "checkpoint-final").mkdir()      # a hand-made directory without a step number
    tr = _saver(tmp_path, 2)
    for step in (3, 6):
        tr.global_step = step
        tr.save_ckpt()
    assert _listing(tmp_path) == ["checkpoint-3", "checkpoint-6", "checkpoint-final"], _listing(tmp_path)


def _tokens(tmp_path, n_samples):
    p = tmp_path / f"tok{n_samples}.bin"
    np.arange(n_samples * T, dtype=np.uint16).tofile(p)
    return p


def _stream_trainer(path, workers, batch):
    from pretrain.data.data_loader import build_train_dataloader
    from pretrain.utils.misc import Peekable
    loader = build_train_dataloader({"paths": [str(path)], "num_workers": workers, "pin_memory": False,
                                     "persistent_workers": True, "prefetch_factor": 2}, T, batch, seed=7)
    loader.sampler.set_epoch(0)
    return _stub_trainer(train_loader=loader, epoch=0, global_step=0, train_iter=Peekable(loader))


@pytest.mark.parametrize("workers", [0, 2])
def test_data_stream_survives_saves_and_resumes(tmp_path, workers):
    """8 samples, batch 2: 4 steps per epoch, 12 steps = 3 epochs. A save (the trainer's peek at the next batch) at a
    step inside an epoch or on an epoch boundary must not change the stream, the peek must equal the next batch, and a
    resume from the save must continue the uninterrupted stream exactly."""
    path = _tokens(tmp_path, 8)

    def run(save_at=None):
        tr = _stream_trainer(path, workers, 2)
        out, peek = [], None
        for s in range(12):
            out.append(tr.next_train_data()["index"].tolist())
            if s + 1 == save_at:
                peek = tr.peek_next_train_data()["index"].tolist()
        return out, peek

    ref, _ = run()
    assert len({tuple(x) for x in ref[:4]}) == 4, ref           # four distinct batches in the first epoch
    for save in (3, 4, 8):
        got, peek = run(save)
        assert got == ref, f"a save at step {save} changed the stream: {got} vs {ref}"
        assert peek == ref[save], f"the peek at step {save} is not the next batch: {peek} vs {ref[save]}"
        tr = _stream_trainer(path, workers, 2)
        tr.jump_to_step(save)
        resumed = [tr.next_train_data()["index"].tolist() for _ in range(12 - save)]
        assert resumed == ref[save:], f"the resume from step {save} diverged: {resumed} vs {ref[save:]}"


def test_an_epoch_without_a_batch_is_refused(tmp_path):
    from pretrain.data.data_loader import build_train_dataloader
    from pretrain.trainer.base_trainer import BaseTrainer
    cfg = types.SimpleNamespace(global_batch_size=4, gradient_accumulation_steps=1, num_devices=1, resume=False,
                                stop_at_step=0, num_steps=10, chunk_size=T, hf_export_steps=[])
    loader = build_train_dataloader({"paths": [str(_tokens(tmp_path, 3))], "num_workers": 0, "pin_memory": False}, T, 4, seed=7)
    with pytest.raises(ValueError, match="no batch per epoch"):
        BaseTrainer(nn.Linear(2, 2), cfg, loader, None, lambda *a, **k: None)


def test_doc_aware_loader_checks_the_stream_against_its_sidecar(tmp_path):
    from pretrain.data.data_loader import build_train_dataloader
    from pretrain.data.doc_dataset import DocAwareMemmapDataset
    dl = np.array([7, 20, 9, 11, 19, 14], dtype=np.int64)       # 80 tokens
    side = tmp_path / "s.doc_lengths.npy"
    np.save(side, dl)
    ids = np.arange(80) % 500 + 2
    good = tmp_path / "good.bin"; ids.astype(np.uint16).tofile(good)
    assert len(DocAwareMemmapDataset(str(good), 16, str(side))) == 5
    bad = {"uint32.bin": ids.astype(np.uint32), "short.bin": ids[:64].astype(np.uint16)}
    for name, arr in bad.items():
        arr.tofile(tmp_path / name)
    np.save(tmp_path / "tok.npy", ids.astype(np.uint16))
    for name in ("uint32.bin", "short.bin", "tok.npy"):          # read as uint16 / sidecar longer than the stream / a .npy
        with pytest.raises(ValueError):
            DocAwareMemmapDataset(str(tmp_path / name), 16, str(side))
    with pytest.raises(ValueError):                             # one doc_lengths sidecar cannot describe two streams
        build_train_dataloader({"paths": [str(good), str(good)], "doc_lengths": str(side), "num_workers": 0}, 16, 2, seed=0)


# ---------------------------------------------------------------- the kernel build a run trains with
# Every checkpoint records the delta_matching build (kernel_version.json: code_sha, bf16_dp, py_sha, dm_so, library) and a resume
# onto another build, or onto the kernel from a checkpoint whose build cannot be read, is refused (DM_ALLOW_KERNEL_CHANGE=1:
# a warning), since the bundled KDA-evolved kernel and the original release differ in their head_dim-128 numerics
# (kernels/README.md).
KV_KDA = {"code_sha": "9fe8da5f743a", "bf16_dp": False, "library": "/a/_C.so"}
KV_ORIG = {"code_sha": "5905517248c5", "bf16_dp": False, "library": "/b/_C.so"}


def _fake_kernel(monkeypatch, kv):
    import sys
    monkeypatch.setitem(sys.modules, "delta_matching", types.SimpleNamespace(kernel_version=lambda: dict(kv, version="0.1.0")))


def test_kernel_identity_reads_the_loaded_kernel_package(monkeypatch, tmp_path):
    import sys
    from pretrain.trainer.base_trainer import kernel_identity
    monkeypatch.delenv("DM_SO", raising=False)
    monkeypatch.delitem(sys.modules, "delta_matching", raising=False)
    assert kernel_identity() is None                       # no layer imported the kernel: a bf16-attention arm
    _fake_kernel(monkeypatch, KV_KDA)
    assert kernel_identity() == dict(KV_KDA, py_sha=None, dm_so=None)   # a stand-in module: no files to hash
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("x = 1\n")
    (pkg / "_build_info.py").write_text("BUILD_INFO = {}\n")
    sys.modules["delta_matching"].__file__ = str(pkg / "__init__.py")
    py = kernel_identity()["py_sha"]
    (pkg / "_build_info.py").write_text("BUILD_INFO = {'other': 1}\n")      # the build stamp is not part of it
    assert py is not None and kernel_identity()["py_sha"] == py
    (pkg / "__init__.py").write_text("x = 2\n")                             # an edit of the Python side is
    assert kernel_identity()["py_sha"] != py
    del sys.modules["delta_matching"].__file__
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib" / "variant.so").write_bytes(b"")
    (tmp_path / "link.so").symlink_to(tmp_path / "lib" / "variant.so")
    monkeypatch.setenv("DM_SO", str(tmp_path / "link.so"))  # a library named by DM_SO: identified by its resolved path
    assert kernel_identity() == dict(KV_KDA, py_sha=None, dm_so=str((tmp_path / "lib" / "variant.so").resolve()))

    def broken():
        raise RuntimeError("no library")
    monkeypatch.setitem(sys.modules, "delta_matching", types.SimpleNamespace(kernel_version=broken))
    assert kernel_identity()["code_sha"] is None           # a provenance stamp never stops a run


@pytest.mark.parametrize("saved, now, same", [
    (KV_KDA, KV_KDA, True), (KV_KDA, dict(KV_KDA, library="/elsewhere/_C.so"), True), (None, None, True),
    (KV_KDA, dict(KV_KDA, dm_so=None, py_sha=None), True),   # a missing field reads as None
    (dict(KV_KDA, py_sha="438885e62436"), dict(KV_KDA, py_sha="958769f516d4"), False),   # another Python side
    (KV_KDA, KV_ORIG, False), (KV_ORIG, KV_KDA, False), (KV_KDA, None, False), (None, KV_KDA, False),
    (KV_KDA, dict(KV_KDA, bf16_dp=True), False), (KV_KDA, dict(KV_KDA, dm_so="/x/_C.so"), False),
    (dict(KV_KDA, dm_so="/x/_C.so"), dict(KV_KDA, dm_so="/y/_C.so"), False)])
def test_check_kernel_identity(saved, now, same):
    from pretrain.trainer.base_trainer import check_kernel_identity
    if same:
        assert check_kernel_identity(saved, now, "ckpt", allow=False) is None
    else:
        with pytest.raises(RuntimeError, match="DM_ALLOW_KERNEL_CHANGE"):
            check_kernel_identity(saved, now, "ckpt", allow=False)
        assert "DM_ALLOW_KERNEL_CHANGE" in check_kernel_identity(saved, now, "ckpt", allow=True)


def test_checkpoint_records_the_kernel_and_resume_checks_it(tmp_path, monkeypatch):
    import json, sys
    from pretrain.trainer.base_trainer import KERNEL_FILE
    monkeypatch.delenv("DM_SO", raising=False)
    monkeypatch.delenv("DM_ALLOW_KERNEL_CHANGE", raising=False)
    _fake_kernel(monkeypatch, KV_KDA)
    logs = []
    tr = _stub_trainer(model=types.SimpleNamespace(save_config=lambda d: None))
    tr.main_log = lambda msg, *a, **k: logs.append(msg)
    tr._gather_full_state_dict = lambda: {}
    ckpt = tmp_path / "checkpoint-3"
    ckpt.mkdir()
    tr._save_full_model(ckpt)
    assert json.loads((ckpt / KERNEL_FILE).read_text()) == dict(KV_KDA, py_sha=None, dm_so=None)
    tr._check_resume_kernel(ckpt)                          # the same build: resumes
    _fake_kernel(monkeypatch, KV_ORIG)
    with pytest.raises(RuntimeError, match="was trained with the delta_matching kernel"):
        tr._check_resume_kernel(ckpt)                      # another build: refused
    monkeypatch.setenv("DM_ALLOW_KERNEL_CHANGE", "1")
    tr._check_resume_kernel(ckpt)                          # ... unless allowed, with a warning
    assert any("DM_ALLOW_KERNEL_CHANGE=1" in m for m in logs), logs
    monkeypatch.delenv("DM_ALLOW_KERNEL_CHANGE")
    for record, why in ((None, "there is none"), ('"9fe8da5f743a"', "unreadable"), ("{not json", "unreadable")):
        if record is None:
            (ckpt / KERNEL_FILE).unlink()                  # a checkpoint written before builds were recorded
        else:
            (ckpt / KERNEL_FILE).write_text(record)        # a hand-edited or truncated record
        with pytest.raises(RuntimeError, match=f"cannot tell which delta_matching build.*{why}"):
            tr._check_resume_kernel(ckpt)                  # the build cannot be read: refused for a kernel run
        monkeypatch.setenv("DM_ALLOW_KERNEL_CHANGE", "1")
        n = len(logs)
        tr._check_resume_kernel(ckpt)                      # ... unless allowed, with a warning
        assert len(logs) == n + 1 and "cannot tell" in logs[-1]
        monkeypatch.delenv("DM_ALLOW_KERNEL_CHANGE")
    monkeypatch.delitem(sys.modules, "delta_matching")
    n = len(logs)
    tr._check_resume_kernel(ckpt)                          # a bf16-attention arm: no kernel, nothing to check
    assert len(logs) == n


def test_load_checks_the_kernel_before_loading_any_state(tmp_path, monkeypatch):
    """BaseTrainer.load runs the kernel check before _load_resume_state (whose dcp.load is collective): a refused resume
    loads nothing, on every rank alike."""
    import json
    from pretrain.trainer.base_trainer import KERNEL_FILE
    monkeypatch.delenv("DM_SO", raising=False)
    monkeypatch.delenv("DM_ALLOW_KERNEL_CHANGE", raising=False)
    tr = _saver(tmp_path, None)
    tr.global_step = 3
    tr.save_ckpt()                                         # checkpoint-3 with its complete marker
    ckpt = tmp_path / "checkpoint-3"
    (ckpt / KERNEL_FILE).write_text(json.dumps(KV_ORIG))
    loaded = []
    tr._load_resume_state = lambda d: loaded.append(Path(d)) or 3
    tr.jump_to_step = lambda step: setattr(tr, "global_step", step)
    _fake_kernel(monkeypatch, KV_KDA)
    with pytest.raises(RuntimeError, match="was trained with the delta_matching kernel"):
        tr.load(tmp_path)
    assert loaded == []                                    # refused before any state was read
    _fake_kernel(monkeypatch, KV_ORIG)
    assert tr.load(tmp_path) == ckpt and loaded == [ckpt] and tr.global_step == 3
