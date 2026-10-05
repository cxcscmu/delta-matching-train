#!/usr/bin/env python
"""Tests for the document-boundary-aware loader (data/doc_dataset.py).

CPU-only, no distributed. Builds a synthetic flat .bin + doc_lengths (200 documents of 50..949 tokens), then checks: chunk count, per-doc position-id reset, cu_seqlens
boundaries, packed-vs-reset modes, the doc collate, and a PretrainLM forward that actually
consumes the per-document position_ids.

Run: PYTHONPATH=src python -m pretrain.scripts.test_doc_loader
"""
from __future__ import annotations

import tempfile

import numpy as np
import torch

from pretrain.data.doc_dataset import DocAwareMemmapDataset
from pretrain.data.data_loader import _doc_collate
from pretrain.modeling.config import ModelConfig
from pretrain.modeling.model import build_model


def main():
    with tempfile.TemporaryDirectory() as tmp:   # removed on exit, pass or fail (_run raises SystemExit)
        _run(tmp)


def _run(tmp: str):
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    out = f"{tmp}/synth"
    docs = [(np.arange(50 + (i * 37) % 900) + i) % 1000 for i in range(200)]   # deterministic lengths 50..949
    np.concatenate(docs).astype(np.uint16).tofile(out + ".bin")
    np.save(out + ".doc_lengths.npy", np.asarray([len(d) for d in docs], dtype=np.int64))
    doc_lengths = np.load(out + ".doc_lengths.npy")
    doc_starts = np.concatenate([[0], np.cumsum(doc_lengths)])
    total = int(doc_starts[-1])
    T = 256

    ds_reset = DocAwareMemmapDataset(out + ".bin", chunk_size=T, doc_lengths=out + ".doc_lengths.npy",
                                     dtype="uint16", position_mode="reset")
    ds_pack = DocAwareMemmapDataset(out + ".bin", chunk_size=T, doc_lengths=out + ".doc_lengths.npy",
                                    dtype="uint16", position_mode="packed")
    check("chunk count == total//T", len(ds_reset) == total // T, f"{len(ds_reset)} vs {total // T}")

    # ground-truth per-token position (reset) for every chunk, via doc_starts
    ok_pos, ok_cu, ok_pack = True, True, True
    for idx in range(len(ds_reset)):
        s = idx * T
        g = np.arange(s, s + T)
        di = np.searchsorted(doc_starts, g, side="right") - 1
        want_pos = g - doc_starts[di]
        got = ds_reset[idx]
        if not torch.equal(got["position_ids"], torch.from_numpy(want_pos.astype(np.int64))):
            ok_pos = False
        # cu_seqlens: local doc-boundary offsets + endpoints, monotonic, spanning [0,T]
        cu = got["cu_seqlens"].numpy()
        want_change = np.nonzero(np.diff(di))[0] + 1
        want_cu = np.concatenate([[0], want_change, [T]]).astype(np.int32)
        if not np.array_equal(cu, want_cu):
            ok_cu = False
        # internal doc starts (boundaries strictly inside the chunk) reset position to 0;
        # the first segment may continue a document that began in an earlier chunk (>0).
        for bnd in cu[1:-1]:
            if got["position_ids"][int(bnd)].item() != 0:
                ok_pos = False
        for a, b in zip(cu[:-1], cu[1:]):  # positions increment by 1 within each document
            seg = got["position_ids"][int(a):int(b)]
            if not torch.equal(seg, torch.arange(int(seg[0]), int(seg[0]) + int(b - a))):
                ok_pos = False
        if not torch.equal(ds_pack[idx]["position_ids"], torch.arange(T)):
            ok_pack = False
    check("reset position_ids restart per document", ok_pos)
    check("cu_seqlens match document boundaries", ok_cu)
    check("packed position_ids are continuous 0..T-1", ok_pack)

    # collate: fixed fields stack, cu_seqlens kept as a ragged list
    batch = _doc_collate([ds_reset[0], ds_reset[1], ds_reset[2]])
    check("collate stacks input_ids/position_ids", batch["input_ids"].shape == (3, T)
          and batch["position_ids"].shape == (3, T) and len(batch["cu_seqlens"]) == 3)

    # model forward actually consumes the per-document position_ids (finite, right shape)
    cfg = ModelConfig(vocab_size=1000, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                      head_dim=32, max_seq_len=T, mixer="gqa", attn_impl="sdpa")
    model = build_model(cfg).eval()
    with torch.no_grad():
        ids = batch["input_ids"].clamp_max(999)
        out_default = model(ids)["logits"]
        out_docpos = model(ids, position_ids=batch["position_ids"])["logits"]
    check("forward with doc position_ids is finite", torch.isfinite(out_docpos).all().item()
          and out_docpos.shape == (3, T, 1000))
    check("doc position_ids change the logits vs default arange",
          not torch.allclose(out_default, out_docpos),
          "(per-doc RoPE positions differ from continuous arange)")

    n = sum(results)
    print(f"\n==== {n}/{len(results)} checks PASS ====")
    raise SystemExit(0 if n == len(results) else 1)


if __name__ == "__main__":
    main()
