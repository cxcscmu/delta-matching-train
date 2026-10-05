#!/usr/bin/env python
"""Stage 1, step 2: tokenize Nemotron-CC-v2.1 (raw-text parquet) into one flat uint16 token stream + a document-length
sidecar (data/pretrain_stage1.md). Called by data/build_stage1.py; runnable on its own:

    python data/stage1_tokenize.py --raw-dir <raw> --configs High-Quality High-Quality-DQA \
        --tokenizer <root>/tokenizer/llama2_32k --out <root>/nemotron_cc_v2d1_hq_dqa/nemotron_hqdqa

Nemotron-CC ships ONE document per parquet row in a `text` column (plus a `metadata` struct + `uuid`). Each document is
tokenized with the Llama-2 tokenizer (vocab 32000, fits uint16) and laid down as

    [BOS] + llama2_ids(text) + [EOS]

so the model learns a real document terminator (EOS); `doc_lengths[i] = len(ids)+2`, hence sum(doc_lengths) == #tokens
in the `.bin` (the invariant the document-aware loader relies on). Documents are concatenated with no separator beyond
the per-document BOS/EOS; boundaries live in the sidecar (no intra-document masking at train time: naive packing).
A document whose text tokenizes to nothing is dropped.

Parallel: one worker process per parquet part (each part -> a `shard_NNNN.bin` + `shard_NNNN.lengths.npy`),
concatenated in a fixed order (configs in --configs order, parts in file-name order), so the output does not depend on
--workers or --threads-per. Per-worker tokenizer threads = --threads-per (RAYON_NUM_THREADS).
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import time
import multiprocessing as mp

import numpy as np

_TOK = None
_BOS = None
_EOS = None
_BATCH = 8192
_MAXB = None


def _init(tok_dir: str, threads_per: int, batch: int, max_batches):
    """Per-worker init: pin tokenizer threads, load the (fast) tokenizer once."""
    global _TOK, _BOS, _EOS, _BATCH, _MAXB
    os.environ["RAYON_NUM_THREADS"] = str(threads_per)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tok_dir)
    _BOS, _EOS = _TOK.bos_token_id, _TOK.eos_token_id
    assert _BOS is not None and _EOS is not None, "tokenizer lacks bos/eos"
    _BATCH = int(batch)
    _MAXB = max_batches


def _tokenize_part(task):
    """Tokenize one parquet part -> shard .bin + .lengths.npy. Returns stats."""
    import pyarrow.parquet as pq
    idx, path, out_dir = task
    binp = os.path.join(out_dir, f"shard_{idx:04d}.bin")
    lenp = os.path.join(out_dir, f"shard_{idx:04d}.lengths.npy")
    pf = pq.ParquetFile(path)
    chunks_len = []
    ndocs = ntok = maxid = 0
    with open(binp, "wb") as f:
        for bi, batch in enumerate(pf.iter_batches(batch_size=_BATCH, columns=["text"])):
            if _MAXB is not None and bi >= _MAXB:
                break
            texts = batch.column("text").to_pylist()
            enc = _TOK(texts, add_special_tokens=False)["input_ids"]
            enc = [ids for ids in enc if ids]
            if not enc:
                continue
            lens = np.fromiter((len(ids) for ids in enc), dtype=np.int64, count=len(enc))
            content = np.concatenate([np.asarray(ids, dtype=np.uint16) for ids in enc])
            outlen = lens + 2
            total = int(outlen.sum())
            out = np.empty(total, dtype=np.uint16)
            ooff = np.empty(len(enc) + 1, dtype=np.int64)
            ooff[0] = 0
            np.cumsum(outlen, out=ooff[1:])
            out[ooff[:-1]] = _BOS            # BOS at each doc start
            out[ooff[1:] - 1] = _EOS         # EOS at each doc end
            mask = np.ones(total, dtype=bool)
            mask[ooff[:-1]] = False
            mask[ooff[1:] - 1] = False
            out[mask] = content              # content fills the remaining slots, in order
            out.tofile(f)
            chunks_len.append(outlen)
            ntok += total
            ndocs += len(enc)
            if content.size:
                m = int(content.max())
                if m > maxid:
                    maxid = m
    L = np.concatenate(chunks_len) if chunks_len else np.zeros(0, dtype=np.int64)
    np.save(lenp, L)
    return dict(idx=idx, path=path, bin=binp, lenp=lenp,
                ndocs=int(ndocs), ntok=int(ntok), maxid=int(maxid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="dir with <config>/*.parquet")
    ap.add_argument("--configs", nargs="+", default=["High-Quality", "High-Quality-DQA"])
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True, help="output prefix (.bin + .doc_lengths.npy)")
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--threads-per", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--max-batches-per-part", type=int, default=None, help="smoke test")
    ap.add_argument("--keep-shards", action="store_true")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    vocab = len(AutoTokenizer.from_pretrained(a.tokenizer))   # every id is < len(tokenizer), added tokens included
    if vocab > 65536:
        raise SystemExit(f"--tokenizer {a.tokenizer} has {vocab} ids: they do not fit the uint16 output")

    files = []
    gidx = 0
    for cfg in a.configs:
        cfg_files = sorted(glob.glob(os.path.join(a.raw_dir, cfg, "*.parquet")))
        if not cfg_files:
            raise FileNotFoundError(f"no parquet under {os.path.join(a.raw_dir, cfg)}")
        for p in cfg_files:
            files.append((gidx, p, None))
            gidx += 1
    out_dir = a.out + "_shards"
    os.makedirs(out_dir, exist_ok=True)
    tasks = [(idx, p, out_dir) for idx, p, _ in files]
    print(f"[ingest] {len(tasks)} parts -> {a.out}.bin  (workers={a.workers} "
          f"threads/worker={a.threads_per} batch={a.batch_size}"
          f"{' SMOKE='+str(a.max_batches_per_part) if a.max_batches_per_part else ''})", flush=True)

    t0 = time.time()
    results = []
    with mp.get_context("spawn").Pool(
        a.workers, initializer=_init,
        initargs=(a.tokenizer, a.threads_per, a.batch_size, a.max_batches_per_part)) as pool:
        for r in pool.imap_unordered(_tokenize_part, tasks):
            results.append(r)
            print(f"[part {r['idx']:>3}] {os.path.basename(r['path'])}: "
                  f"docs={r['ndocs']:,} tok={r['ntok']:,} maxid={r['maxid']} "
                  f"({len(results)}/{len(tasks)} done, {time.time()-t0:.0f}s)", flush=True)
    results.sort(key=lambda r: r["idx"])

    gmax = max(r["maxid"] for r in results)
    assert gmax < vocab, f"token id {gmax} >= {vocab} (vocab overflow)"

    binout, lenout = a.out + ".bin", a.out + ".doc_lengths.npy"
    with open(binout, "wb") as fo:
        for r in results:
            with open(r["bin"], "rb") as fi:
                shutil.copyfileobj(fi, fo, length=1 << 24)
    L = np.concatenate([np.load(r["lenp"]) for r in results])
    np.save(lenout, L)

    nbin = os.path.getsize(binout) // 2
    assert nbin == int(L.sum()), f"sum-check FAIL: bin={nbin} sum(doc_lengths)={int(L.sum())}"
    if not a.keep_shards:
        shutil.rmtree(out_dir, ignore_errors=True)
    print(f"[ingest] DONE {binout}: docs={len(L):,} tokens={nbin:,} "
          f"({nbin*2/1e9:.1f} GB) maxid={gmax}  sum-check OK  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[ingest] doc_lengths -> {lenout}  (mean {L.mean():.1f} tok/doc, "
          f"p50 {np.percentile(L,50):.0f}, max {L.max():,})", flush=True)


if __name__ == "__main__":
    main()
