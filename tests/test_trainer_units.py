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
