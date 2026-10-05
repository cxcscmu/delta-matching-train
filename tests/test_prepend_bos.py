"""prepend_bos: BOS-anchored windows = [bos] + chunk[:-1] in both packed loaders; the flag off leaves every window
unchanged. CPU only, synthetic token files."""
import numpy as np
import pytest
import torch

from pretrain.config import load_config
from pretrain.data.doc_dataset import DocAwareMemmapDataset
from pretrain.data.memmap_dataset import MemmapTokenDataset

T = 16


@pytest.fixture
def stream(tmp_path):
    tok = (np.arange(5 * T) % 500 + 2).astype(np.uint16)          # never 1, so the BOS anchor is unambiguous
    p = tmp_path / "s.bin"
    tok.tofile(p)
    dl = np.array([7, 20, 9, 11, 19, 14], dtype=np.int64)          # document starts 7, 27, 36, 47, 66; 47 = chunk 2's last token
    q = tmp_path / "s.doc_lengths.npy"
    np.save(q, dl)
    return str(p), str(q), tok


def test_memmap_prepend_bos(stream):
    p, _, tok = stream
    off, on = MemmapTokenDataset(p, T), MemmapTokenDataset(p, T, prepend_bos=True, bos_token_id=1)
    assert len(off) == len(on) == 5
    for i in range(5):
        a, b = off[i]["input_ids"], on[i]["input_ids"]
        assert torch.equal(a, torch.from_numpy(tok[i * T:(i + 1) * T].astype(np.int64)))   # flag off: the old window
        assert b.shape == a.shape and b[0].item() == 1 and torch.equal(b[1:], a[:-1])
        assert on[i]["index"] == off[i]["index"] == i


def test_docaware_prepend_bos(stream):
    p, q, _ = stream
    off, on = DocAwareMemmapDataset(p, T, q), DocAwareMemmapDataset(p, T, q, prepend_bos=True, bos_token_id=1)
    for i in range(5):
        a, b = off[i], on[i]
        assert b["input_ids"][0].item() == 1 and torch.equal(b["input_ids"][1:], a["input_ids"][:-1])
        assert torch.equal(b["position_ids"], torch.arange(T)) and torch.equal(a["position_ids"], torch.arange(T))
        # every in-chunk document start moves right by one; one on the dropped last token disappears
        exp = [0] + [c + 1 for c in a["cu_seqlens"][1:-1].tolist() if c + 1 < T] + [T]
        assert b["cu_seqlens"].tolist() == exp
    assert off[1]["cu_seqlens"].tolist() == [0, 11, T] and on[1]["cu_seqlens"].tolist() == [0, 12, T]
    assert off[2]["cu_seqlens"].tolist() == [0, 4, 15, T] and on[2]["cu_seqlens"].tolist() == [0, 5, T]


def test_docaware_reset_rejects_bos(stream):
    p, q, _ = stream
    with pytest.raises(NotImplementedError):
        DocAwareMemmapDataset(p, T, q, position_mode="reset", prepend_bos=True)


def test_cli_flags():
    assert load_config([]).data_cfg["prepend_bos"] is False
    cfg = load_config(["--data.prepend_bos", "true", "--data.bos_token_id", "1"])
    assert cfg.data_cfg["prepend_bos"] is True and cfg.data_cfg["bos_token_id"] == 1
