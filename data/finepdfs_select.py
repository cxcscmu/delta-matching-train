#!/usr/bin/env python3
"""FinePDFs-Edu English, steps 3 and 4: select and write (data/pretrain_finepdfs_edu.md). Called by
data/build_finepdfs_edu.py; runnable on its own, after data/finepdfs_tokenize.py:

    python data/finepdfs_select.py --work <work> --out <out> --step select --step write \
        --train-tokens 60e9 --val-tokens 8e8 --seed 11785

select  reads <work>/shards/task_*.docs.parquet. The pool is every counted document with edu >= --edu-min (the
        tokenize step's), after an exact-id dedup across files (of the documents sharing an id, the one with the most
        tokens stays; ties go to the lower (file, row)). Documents with the same norm_url form a group, and a group
        always goes to one split. The pool is sorted by (key, norm_url, id, file, row), key =
        BLAKE2b-64(b'finepdfs-edu-en|<seed>|' + norm_url) (data/finepdfs_tokenize.py), which orders the groups at
        random and keeps each group together; the validation set takes whole groups in that order until it holds
        >= --val-tokens tokens, the training set the following whole groups until it holds >= --train-tokens tokens,
        and the rest of the pool is unused. So --train-tokens <= train < --train-tokens + the tokens of its last group,
        a smaller target selects a prefix of the same order (nested subsets for the same --val-tokens), and a pool too
        small for both targets stops the build. Writes <out>/selection.parquet (split, file, row, id, norm_url, key,
        n_tokens, edu: the training documents in source order, then the validation documents in key order),
        <work>/select/write_plan.npz and <work>/select_summary.json (pool, dedup, URL groups, exact pool sizes at
        each counted edu threshold, targets and what was reached, length shares, the validation set's coverage of the
        first 2,000 windows of the val-CE protocol).
write   copies the selected documents' tokens out of the task shards: <out>/finepdfs_edu_en_train.bin (source order:
        file, then row, as the stage-1 training stream) and <out>/finepdfs_edu_en_val.bin (key order, so already
        shuffled), uint16, documents back to back as [BOS] + ids + [EOS], each with its .doc_lengths.npy (int64, one
        length per document); then checks the invariants of both streams (bytes == 2 * sum(lengths), every id <
        vocab, BOS exactly at the document starts and EOS exactly at the document ends, no unk id, the stream starts
        with BOS) and writes <work>/write_summary.json with the sha256 of every output. The train file is written by
        --workers processes at precomputed offsets; the outputs do not depend on --workers.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify import sha256  # noqa: E402

PREFIX = "finepdfs_edu_en"
STEPS = ("select", "write")
SEQ_LEN, VAL_WINDOWS = 8192, 2000          # the val-CE protocol (docs/benchmark.md): first 2,000 windows of 8,192
LENGTH_CUTS = (8192, 100_000, 1_000_000)


def stream_names(split: str) -> tuple[str, str]:
    return f"{PREFIX}_{split}.bin", f"{PREFIX}_{split}.doc_lengths.npy"


# ---- select --------------------------------------------------------------------------------------------------------
def string_ranks(col) -> np.ndarray:
    """The rank of each string of an Arrow (chunked) string column among its distinct values, in code point order."""
    import pyarrow as pa
    import pyarrow.compute as pc
    arr = col.cast(pa.large_string())
    arr = arr.combine_chunks() if hasattr(arr, "combine_chunks") else arr
    enc = pc.dictionary_encode(arr)
    order = pc.sort_indices(enc.dictionary).to_numpy()
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order), dtype=np.int64)
    return rank[enc.indices.to_numpy(zero_copy_only=False)]


def dedup_ids(id_rank, n_tokens, file, row) -> np.ndarray:
    """Positions (into the given arrays) of the documents that stay: one per id, the one with the most tokens, ties to
    the lower (file, row)."""
    if len(id_rank) == 0:
        return np.zeros(0, dtype=np.int64)
    o = np.lexsort((row, file, -np.asarray(n_tokens, dtype=np.int64), id_rank))
    first = np.ones(len(o), dtype=bool)
    first[1:] = id_rank[o][1:] != id_rank[o][:-1]
    return np.sort(o[first])


def select_groups(key, url_rank, id_rank, file, row, n_tokens, train_target: int, val_target: int):
    """The sampling rule on a deduplicated pool. Returns (order, n_val, n_train, info): `order` is the pool in key order
    (key, norm_url, id, file, row); its first n_val positions are the validation documents, the next n_train the
    training documents. Raises ValueError when the pool cannot reach both targets."""
    n_tokens = np.asarray(n_tokens, dtype=np.int64)
    order = np.lexsort((row, file, id_rank, url_rank, key))
    u = url_rank[order]
    starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]]) if len(order) else np.zeros(0, dtype=np.int64)
    gtok = np.add.reduceat(n_tokens[order], starts) if len(order) else np.zeros(0, dtype=np.int64)
    cum = np.cumsum(gtok)
    total = int(cum[-1]) if len(cum) else 0
    gv = int(np.searchsorted(cum, val_target, side="left"))
    if val_target <= 0 or gv >= len(cum):
        raise ValueError(f"the pool ({total:,} tokens in {len(gtok):,} URL groups) cannot reach the validation target "
                         f"of {val_target:,} tokens")
    base = int(cum[gv])
    gt = int(np.searchsorted(cum, base + train_target, side="left"))
    if train_target <= 0 or gt >= len(cum):
        raise ValueError(f"the pool ({total:,} tokens in {len(gtok):,} URL groups) cannot reach the training target "
                         f"of {train_target:,} tokens after {base:,} validation tokens: it needs about "
                         f"{train_target + val_target:,} tokens plus one URL group")
    bounds = np.r_[starts, len(order)]
    n_val = int(bounds[gv + 1])
    n_train = int(bounds[gt + 1]) - n_val
    val_tok, train_tok = base, int(cum[gt]) - base
    info = {
        "groups": int(len(gtok)), "val_groups": gv + 1, "train_groups": gt - gv, "unused_groups": int(len(gtok) - gt - 1),
        "val_tokens": val_tok, "train_tokens": train_tok, "pool_tokens": total,
        "val_last_group_tokens": int(gtok[gv]), "train_last_group_tokens": int(gtok[gt]),
        "val_overshoot": val_tok - int(val_target), "train_overshoot": train_tok - int(train_target),
        "largest_group_tokens": int(gtok.max()),
    }
    assert info["train_overshoot"] >= 0 and info["train_overshoot"] < info["train_last_group_tokens"]
    assert info["val_overshoot"] >= 0 and info["val_overshoot"] < info["val_last_group_tokens"]
    return order, n_val, n_train, info


def length_shares(n_tokens) -> dict:
    L = np.asarray(n_tokens, dtype=np.int64)
    tot = max(int(L.sum()), 1)
    out = {"docs": int(len(L)), "tokens": int(L.sum()),
           "mean": round(float(L.mean()), 1) if len(L) else 0, "median": float(np.median(L)) if len(L) else 0,
           "max": int(L.max()) if len(L) else 0}
    for c in LENGTH_CUTS:
        out[f"token_share_docs_over_{c}"] = round(float(L[L > c].sum()) / tot, 5)
    return out


def val_coverage(val_lengths, seq_len: int = SEQ_LEN, windows: int = VAL_WINDOWS) -> dict:
    """The documents the first `windows` windows of the val-CE protocol score (evaluation/ppl/memmap_ppl.py: window w
    reads tokens [w*L, (w+1)*L] and scores tokens w*L+1 .. (w+1)*L) and the largest document's share of them."""
    L = np.asarray(val_lengths, dtype=np.int64)
    starts = np.r_[0, np.cumsum(L)]
    total = int(starts[-1])
    w = min(windows, max((total - 1) // seq_len, 0))
    lo, hi = 1, w * seq_len + 1                       # scored token positions [lo, hi)
    if w == 0:
        return {"windows": 0, "protocol_windows": windows, "docs": 0, "largest_doc_share": None}
    d0 = int(np.searchsorted(starts, lo, side="right") - 1)
    d1 = int(np.searchsorted(starts, hi - 1, side="right") - 1)
    seg_lo = np.maximum(starts[d0:d1 + 1], lo)
    seg_hi = np.minimum(starts[d0 + 1:d1 + 2], hi)
    share = (seg_hi - seg_lo) / (hi - lo)
    return {"windows": w, "protocol_windows": windows, "scored_tokens": hi - lo, "docs": d1 - d0 + 1,
            "largest_doc_share": round(float(share.max()), 4), "short_of_protocol": w < windows}


def _load_docs(work: str):
    import pyarrow as pa
    import pyarrow.parquet as pq
    with open(os.path.join(work, "tokenize_summary.json")) as f:
        tsum = json.load(f)
    tables = []
    for t in tsum["task_ids"]:
        tb = pq.read_table(os.path.join(work, "shards", f"task_{t:04d}.docs.parquet"))
        tables.append(tb.append_column("task", pa.array(np.full(tb.num_rows, t, dtype=np.int32))))
    return tsum, pa.concat_tables(tables)


def select(work: str, out: str, train_tokens: int, val_tokens: int, seed: int) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    t0 = time.time()
    tsum, T = _load_docs(work)
    P = tsum["params"]
    edu_min = P["edu_min"]
    n = T.num_rows
    col = {c: T.column(c).to_numpy() for c in ("file", "row", "n_tokens", "edu", "in_shard", "tok_off", "task",
                                               "special_literal")}
    if not np.array_equal(col["in_shard"], col["edu"] >= edu_min):
        raise SystemExit("[finepdfs-select] in_shard disagrees with edu >= edu_min: the shards are inconsistent")
    if seed == P["seed"]:
        key = T.column("key").to_numpy()
    else:                                        # another seed than the tokenize step's: recompute the keys
        from finepdfs_tokenize import url_key
        print(f"[finepdfs-select] seed {seed} != tokenize seed {P['seed']}: recomputing {n:,} keys", flush=True)
        key = np.fromiter((url_key(u, seed) for u in T.column("norm_url").to_pylist()), dtype=np.uint64, count=n)
    id_rank = string_ranks(T.column("id"))
    url_rank = string_ranks(T.column("norm_url"))
    t1 = time.time()
    print(f"[finepdfs-select] {n:,} counted documents loaded and ranked ({t1 - t0:.0f}s)", flush=True)

    def pool_at(mask):
        idx = np.flatnonzero(mask)
        keep = idx[dedup_ids(id_rank[idx], col["n_tokens"][idx], col["file"][idx], col["row"][idx])]
        return idx, keep

    thresholds = {}
    for t in P["edu_thresholds"]:
        _, keep_t = pool_at(col["edu"] >= t)
        thresholds[f"{t:g}"] = {"docs": int(len(keep_t)), "tokens": int(col["n_tokens"][keep_t].sum())}
    idx, keep = pool_at(col["in_shard"])
    pool_before = {"docs": int(len(idx)), "tokens": int(col["n_tokens"][idx].sum())}
    dedup = {"docs_dropped": int(len(idx) - len(keep)),
             "tokens_dropped": int(pool_before["tokens"] - col["n_tokens"][keep].sum()),
             "rule": "exact id across all files: the document with the most tokens stays, ties to the lower (file, row)"}
    order, n_val, n_train, info = select_groups(key[keep], url_rank[keep], id_rank[keep], col["file"][keep],
                                                col["row"][keep], col["n_tokens"][keep], int(train_tokens),
                                                int(val_tokens))
    S = keep[order]                              # the pool in key order
    val_idx = S[:n_val]
    train_idx = S[n_val:n_val + n_train]
    unused_idx = S[n_val + n_train:]
    train_idx = train_idx[np.lexsort((col["row"][train_idx], col["file"][train_idx]))]    # source order
    # the two splits share no document and no URL
    assert len(np.intersect1d(id_rank[val_idx], id_rank[train_idx])) == 0, "train and val share an id"
    assert len(np.intersect1d(url_rank[val_idx], url_rank[train_idx])) == 0, "train and val share a norm_url"
    # URL groups of the pool
    ur = url_rank[keep]
    _, ginv, gcnt = np.unique(ur, return_inverse=True, return_counts=True)
    gtok = np.bincount(ginv, weights=col["n_tokens"][keep]).astype(np.int64)
    big_t, big_d = int(np.argmax(gtok)), int(np.argmax(gcnt))
    norm_url = T.column("norm_url")
    rep_t = keep[int(np.flatnonzero(ginv == big_t)[0])]
    rep_d = keep[int(np.flatnonzero(ginv == big_d)[0])]
    groups = {
        "groups": int(len(gcnt)), "docs_sharing_a_url": int(gcnt[gcnt > 1].sum()),
        "share_docs_sharing_a_url": round(float(gcnt[gcnt > 1].sum()) / max(len(keep), 1), 5),
        "largest_by_tokens": {"norm_url": norm_url[int(rep_t)].as_py(), "docs": int(gcnt[big_t]), "tokens": int(gtok[big_t])},
        "largest_by_docs": {"norm_url": norm_url[int(rep_d)].as_py(), "docs": int(gcnt[big_d]), "tokens": int(gtok[big_d])},
    }

    def part(ix):
        d = length_shares(col["n_tokens"][ix])
        w = col["n_tokens"][ix].astype(np.float64)
        d["token_weighted_edu"] = round(float((col["edu"][ix] * w).sum() / max(w.sum(), 1)), 4)
        d["special_literal_docs"] = int(col["special_literal"][ix].sum())
        g = pa.table({"e": T.column("extractor").take(ix), "n": col["n_tokens"][ix]}).group_by("e").aggregate(
            [("n", "sum")]).to_pydict()
        d["extractor_tokens"] = {str(k): int(v) for k, v in sorted(zip(g["e"], g["n_sum"]), key=lambda kv: str(kv[0]))}
        return d

    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(work, "select"), exist_ok=True)
    np.savez(os.path.join(work, "select", "write_plan.tmp.npz"),
             train_task=col["task"][train_idx], train_off=col["tok_off"][train_idx], train_len=col["n_tokens"][train_idx],
             val_task=col["task"][val_idx], val_off=col["tok_off"][val_idx], val_len=col["n_tokens"][val_idx])
    os.replace(os.path.join(work, "select", "write_plan.tmp.npz"), os.path.join(work, "select", "write_plan.npz"))
    rows = np.r_[train_idx, val_idx]
    sel = T.select(["file", "row", "id", "norm_url", "n_tokens", "edu"]).take(rows)
    split = pa.DictionaryArray.from_arrays(pa.array(np.r_[np.zeros(len(train_idx), np.int8), np.ones(len(val_idx), np.int8)]),
                                           pa.array(["train", "val"]))
    sel = sel.add_column(0, "split", split)
    sel = sel.add_column(sel.schema.get_field_index("norm_url") + 1, "key", pa.array(key[rows], type=pa.uint64()))
    pq.write_table(sel, os.path.join(out, "selection.tmp.parquet"), compression="zstd")
    os.replace(os.path.join(out, "selection.tmp.parquet"), os.path.join(out, "selection.parquet"))
    summary = {
        "edu_min": edu_min, "seed": seed, "tokenize_seed": P["seed"],
        "targets": {"train_tokens": int(train_tokens), "val_tokens": int(val_tokens)},
        "counted_docs": n, "pool_before_dedup": pool_before, "dedup": dedup, "url_groups": groups,
        "selection": info, "pool_by_edu_threshold": thresholds,
        "headroom": round(info["pool_tokens"] / (int(train_tokens) + int(val_tokens)), 4),
        "pool": part(keep), "train": part(train_idx), "val": part(val_idx),
        "unused": {"docs": int(len(unused_idx)), "tokens": int(col["n_tokens"][unused_idx].sum())},
        "val_coverage": val_coverage(col["n_tokens"][val_idx]),
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(work, "select_summary.json.tmp"), "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(os.path.join(work, "select_summary.json.tmp"), os.path.join(work, "select_summary.json"))
    print(f"[finepdfs-select] pool {len(keep):,} documents / {info['pool_tokens']:,} tokens after the id dedup "
          f"(-{dedup['docs_dropped']:,} documents) in {groups['groups']:,} URL groups; val {len(val_idx):,} documents / "
          f"{info['val_tokens']:,} tokens (+{info['val_overshoot']:,}); train {len(train_idx):,} documents / "
          f"{info['train_tokens']:,} tokens (+{info['train_overshoot']:,}); unused {summary['unused']['tokens']:,} "
          f"tokens; headroom {summary['headroom']:.2f}x ({summary['seconds']:.0f}s)", flush=True)
    return summary


# ---- write ---------------------------------------------------------------------------------------------------------
def pwrite_all(fd: int, data, offset: int, chunk: int = 1 << 30):
    mv = memoryview(data).cast("B")
    done = 0
    while done < len(mv):
        k = os.pwrite(fd, mv[done:done + chunk], offset + done)
        if k <= 0:
            raise OSError(f"pwrite wrote {k} bytes")
        done += k


def _copy_task(args):
    """The selected documents of one task shard, gathered in order and written at their offset of the output."""
    src, dst, out_tok, offs, lens = args
    total = int(lens.sum())
    if total == 0:
        return 0
    mm = np.memmap(src, dtype=np.uint16, mode="r")
    buf = np.empty(total, dtype=np.uint16)
    brk = np.flatnonzero(offs[1:] != offs[:-1] + lens[:-1]) + 1     # runs of documents adjacent in the shard
    p = 0
    for s, e in zip(np.r_[0, brk], np.r_[brk, len(offs)]):
        a, b = int(offs[s]), int(offs[e - 1] + lens[e - 1])
        buf[p:p + b - a] = mm[a:b]
        p += b - a
    assert p == total
    fd = os.open(dst, os.O_WRONLY)
    try:
        pwrite_all(fd, buf, 2 * int(out_tok))
    finally:
        os.close(fd)
    return total


def _scan_chunk(args):
    path, tok_lo, tok_hi, starts_rel, vocab, bos, eos, unk = args
    a = np.memmap(path, dtype=np.uint16, mode="r", offset=2 * tok_lo, shape=(tok_hi - tok_lo,))
    ends_rel = np.r_[starts_rel[1:], tok_hi - tok_lo] - 1
    return {"lo": tok_lo, "maxid": int(a.max()), "bos_ok": bool(np.array_equal(np.flatnonzero(a == bos), starts_rel)),
            "eos_ok": bool(np.array_equal(np.flatnonzero(a == eos), ends_rel)),
            "unk": int(np.count_nonzero(a == unk)) if unk is not None else 0}


def check_stream(path: str, lengths: np.ndarray, vocab: int, bos: int, eos: int, unk, pool=None,
                 chunk_tokens: int = 1 << 27) -> dict:
    """The invariants of a [BOS] + ids + [EOS] stream and its lengths; raises AssertionError naming the first failure."""
    L = np.asarray(lengths)
    assert L.dtype == np.int64 and L.ndim == 1 and len(L) > 0 and (L > 2).all(), f"{path}: bad doc_lengths"
    starts = np.r_[0, np.cumsum(L)]
    total = int(starts[-1])
    assert os.path.getsize(path) == 2 * total, f"{path}: {os.path.getsize(path)} bytes != 2 * sum(lengths) = {2 * total}"
    cut = np.unique(np.r_[np.searchsorted(starts, np.arange(0, total, chunk_tokens), side="left"), len(L)])
    jobs = []
    for d0, d1 in zip(cut[:-1], cut[1:]):
        lo, hi = int(starts[d0]), int(starts[d1])
        jobs.append((path, lo, hi, (starts[d0:d1] - lo).astype(np.int64), vocab, bos, eos, unk))
    res = pool.map(_scan_chunk, jobs) if pool is not None else [_scan_chunk(j) for j in jobs]
    maxid = max(r["maxid"] for r in res)
    first = int(np.memmap(path, dtype=np.uint16, mode="r", shape=(1,))[0])
    out = {"tokens": total, "docs": int(len(L)), "maxid": maxid, "first_token": first,
           "bos_exactly_at_starts": all(r["bos_ok"] for r in res), "eos_exactly_at_ends": all(r["eos_ok"] for r in res),
           "unk_count": sum(r["unk"] for r in res)}
    assert maxid < vocab, f"{path}: token id {maxid} >= vocab {vocab}"
    assert first == bos, f"{path}: the stream starts with {first}, not BOS {bos}"
    assert out["bos_exactly_at_starts"], f"{path}: BOS ids are not exactly at the document starts"
    assert out["eos_exactly_at_ends"], f"{path}: EOS ids are not exactly at the document ends"
    assert out["unk_count"] == 0, f"{path}: {out['unk_count']} unk ids"
    return out


def write(work: str, out: str, workers: int) -> dict:
    t0 = time.time()
    with open(os.path.join(work, "tokenize_summary.json")) as f:
        P = json.load(f)["params"]
    plan = np.load(os.path.join(work, "select", "write_plan.npz"))
    shard = lambda t: os.path.join(work, "shards", f"task_{int(t):04d}.bin")  # noqa: E731
    os.makedirs(out, exist_ok=True)
    tb, tl = stream_names("train")
    vb, vl = stream_names("val")
    # train: source order = task order, each task's documents at a precomputed offset
    task, off, ln = plan["train_task"], plan["train_off"], plan["train_len"].astype(np.int64)
    total = int(ln.sum())
    tmp = os.path.join(out, tb + ".tmp")
    with open(tmp, "wb") as f:
        f.truncate(2 * total)
    assert (np.diff(task) >= 0).all(), "the training documents are not in task order"
    cut = np.flatnonzero(np.r_[True, task[1:] != task[:-1], True])
    out_off = np.r_[0, np.cumsum(ln)]
    jobs = [(shard(task[a]), tmp, int(out_off[a]), off[a:b].astype(np.int64), ln[a:b]) for a, b in zip(cut[:-1], cut[1:])]
    ctx = mp.get_context("spawn")
    with ctx.Pool(max(1, workers)) as pool:
        wrote = sum(pool.imap_unordered(_copy_task, jobs))
        assert wrote == total
        os.replace(tmp, os.path.join(out, tb))
        np.save(os.path.join(out, tl), ln)
        t1 = time.time()
        print(f"[finepdfs-write] {tb}: {len(ln):,} documents, {total:,} tokens ({2 * total / 1e9:.1f} GB) in "
              f"{t1 - t0:.0f}s ({2 * total / 1e6 / max(t1 - t0, 1e-9):.0f} MB/s)", flush=True)
        # val: key order, one document at a time
        vtask, voff, vln = plan["val_task"], plan["val_off"], plan["val_len"].astype(np.int64)
        mms = {}
        with open(os.path.join(out, vb + ".tmp"), "wb", buffering=1 << 24) as f:
            for t, o, n in zip(vtask.tolist(), voff.tolist(), vln.tolist()):
                mm = mms.get(t)
                if mm is None:
                    mm = mms[t] = np.memmap(shard(t), dtype=np.uint16, mode="r")
                f.write(mm[o:o + n].tobytes())
        del mms
        os.replace(os.path.join(out, vb + ".tmp"), os.path.join(out, vb))
        np.save(os.path.join(out, vl), vln)
        t2 = time.time()
        print(f"[finepdfs-write] {vb}: {len(vln):,} documents, {int(vln.sum()):,} tokens in {t2 - t1:.0f}s", flush=True)
        # invariants (parallel scan) and sha256 (threads) of the outputs
        names = [tb, tl, vb, vl, "selection.parquet"]
        with ThreadPoolExecutor(len(names)) as ex:
            hashes = {nm: ex.submit(sha256, os.path.join(out, nm)) for nm in names}
            inv = {split: check_stream(os.path.join(out, b), np.load(os.path.join(out, l_)), P["vocab"], P["bos"],
                                       P["eos"], P["unk"], pool=pool)
                   for split, b, l_ in (("train", tb, tl), ("val", vb, vl))}
            outputs = {nm: {"bytes": os.path.getsize(os.path.join(out, nm)), "sha256": h.result()}
                       for nm, h in hashes.items()}
    t3 = time.time()
    summary = {"outputs": outputs, "invariants": inv, "workers": workers,
               "seconds": {"train": round(t1 - t0, 1), "val": round(t2 - t1, 1), "check_and_sha256": round(t3 - t2, 1)}}
    with open(os.path.join(work, "write_summary.json.tmp"), "w") as f:
        json.dump(summary, f, indent=1)
    os.replace(os.path.join(work, "write_summary.json.tmp"), os.path.join(work, "write_summary.json"))
    print(f"[finepdfs-write] invariants OK (train {inv['train']['docs']:,} docs / {inv['train']['tokens']:,} tokens, "
          f"val {inv['val']['docs']:,} docs / {inv['val']['tokens']:,} tokens, max id {max(inv['train']['maxid'], inv['val']['maxid'])})"
          f"; sha256 of {len(outputs)} files ({t3 - t2:.0f}s)", flush=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="the work dir of data/finepdfs_tokenize.py")
    ap.add_argument("--out", required=True, help="where the streams and selection.parquet go")
    ap.add_argument("--step", choices=STEPS, action="append", help="run only this step (repeatable; default both)")
    ap.add_argument("--train-tokens", type=float, default=60e9)
    ap.add_argument("--val-tokens", type=float, default=8e8)
    ap.add_argument("--seed", type=int, default=11785)
    ap.add_argument("--workers", type=int, default=16, help="processes of the write step (the output does not depend "
                                                            "on it)")
    a = ap.parse_args(argv)
    steps = a.step or list(STEPS)
    if "select" in steps:
        try:
            select(a.work, a.out, int(a.train_tokens), int(a.val_tokens), a.seed)
        except ValueError as e:
            raise SystemExit(f"[finepdfs-select] FAILED: {e}")
    if "write" in steps:
        write(a.work, a.out, a.workers)


if __name__ == "__main__":
    sys.exit(main())
