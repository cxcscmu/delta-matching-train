#!/usr/bin/env python
"""Stage 2, step 2: re-tokenize ProLong's 512K release (Llama-3 ids) with the Llama-2 tokenizer, pack it into
65,536-token windows and flatten it into one uint16 token stream + a document-length sidecar (data/pretrain_stage2.md).
Called by data/build_stage2.py; runnable on its own:

    python data/stage2_repack.py --src <raw>/prolong-data-512K --llama3 <root>/tokenizer/llama3/tokenizer.model \
        --llama2 <root>/tokenizer/llama2_32k/tokenizer.model --out <root>/prolong_64k_llama2/prolong64k_train

ProLong ships token ids, not text, so a tokenizer change is decode -> re-encode -> re-pack. The unit of work is one
`procNN-64` directory of the release: a self-contained packed stream (its documents never continue in another one),
768 of them over 12 source directories. Per unit:

  1. reconstruct  split the unit's Llama-3 stream into whole source documents: a document starts at a Llama-3 BOS
                  (128000) or right after a Llama-3 EOS (128001), and a document cut by the release's 524,288-token
                  windows is rejoined. It ended genuinely iff its last token is the EOS; ids >= 128000 (BOS, EOS, the
                  chat markers of tuluv2) are dropped, and a document with no other token is skipped.
  2. re-encode    decode the whole document with the Llama-3 tokenizer (bytes of each id, UTF-8 with replacement:
                  tiktoken's decode) and encode the text with the Llama-2 SentencePiece model, no special tokens;
                  lay it down as [BOS=1] + ids + [EOS=2 iff the source document ended genuinely]. The length-chunked
                  domains (books, code) carry no EOS in the release, so their documents get none.
  3. pack         datatools' SingleBuffer: concatenate documents into 65,536-token windows, carry the remainder of a
                  document into the next window, drop the unit's trailing partial window.
  4. flatten      write the windows back to back and recover the document lengths from each window's document spans:
                  a new document starts at every span but a window's first, and at a window's first span only when
                  the previous window ended on EOS (this rejoins documents split by the packing, including the EOS-less
                  book and code documents).

The units are concatenated in a fixed order (the ten output domains in the order of DOMAINS below, the units of a domain
in name order), so the output does not depend on --workers; token c * 65,536 starts window c of the stream.

Depends on numpy and sentencepiece only: the release's MDS shards (uncompressed, version 2) are read directly."""
from __future__ import annotations

import argparse
import base64
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np

PACK = 65536
L3_BOS, L3_EOS, L3_SPECIAL = 128000, 128001, 128000   # Llama-3: ids >= 128000 are special
L2_BOS, L2_EOS, L2_VOCAB = 1, 2, 32000
# output domain -> the 512K source directories it is made from (the long domains take both length bins), in the
# concatenation order of the output
DOMAINS = (
    ("thestackv1_concat_by_repo-65536", ("thestackv1_concat_by_repo-65536", "thestackv1_concat_by_repo-524288")),
    ("book-65536", ("book-65536", "book-524288")),
    ("fineweb-edu", ("fineweb-edu",)),
    ("fineweb-2023-50", ("fineweb-2023-50",)),
    ("stackexchange", ("stackexchange",)),
    ("dolmawiki", ("dolmawiki",)),
    ("tuluv2", ("tuluv2",)),
    ("arxiv", ("arxiv",)),
    ("openwebmath", ("openwebmath",)),
    ("textbooks", ("textbooks",)),
)
SOURCE_DIRS = tuple(s for _, srcs in DOMAINS for s in srcs)
PROCS_PER_DIR = 64


# ---- reading the release ------------------------------------------------------------------------------------------
def mds_column(dirname: str, column: str):
    """Yield one ndarray column of every sample of an MDS dataset directory (index.json + shard files), in order: the
    uncompressed version-2 format of mosaicml-streaming, which is what the ProLong release uses."""
    with open(os.path.join(dirname, "index.json")) as f:
        shards = json.load(f)["shards"]
    for sh in shards:
        if sh.get("version") != 2 or sh.get("format") != "mds" or sh.get("compression") or sh.get("zip_data"):
            raise ValueError(f"{dirname}: unsupported MDS shard {sh.get('raw_data')}")
        names, sizes = sh["column_names"], sh["column_sizes"]
        col = names.index(column)
        enc = sh["column_encodings"][col].split(":")    # ndarray[:dtype[:shape]]
        if enc[0] != "ndarray":
            raise ValueError(f"{dirname}: column {column} is {sh['column_encodings'][col]}, not an ndarray")
        with open(os.path.join(dirname, sh["raw_data"]["basename"]), "rb") as f:
            data = f.read()
        n = int(np.frombuffer(data, np.uint32, 1, 0)[0])
        if n != sh["samples"]:
            raise ValueError(f"{dirname}/{sh['raw_data']['basename']}: {n} samples, index.json says {sh['samples']}")
        offs = np.frombuffer(data, np.uint32, n + 1, 4)   # absolute byte offsets of the samples
        for i in range(n):
            pos = int(offs[i])
            lens = []
            for size in sizes:                              # each variable-size column leads with a uint32 size
                if size:
                    lens.append(int(size))
                else:
                    lens.append(int(np.frombuffer(data, np.uint32, 1, pos)[0]))
                    pos += 4
            pos += sum(lens[:col])
            end = pos + lens[col]
            if len(enc) > 1:
                dtype = np.dtype(enc[1])
            else:                                           # dynamic dtype: one code byte
                dtype = np.dtype(_MDS_DTYPES[data[pos]])
                pos += 1
            if len(enc) > 2:
                shape = tuple(int(x) for x in enc[2].split(","))
            else:                                           # dynamic shape: (ndim << 2) | shape dtype, then the shape
                ndim, sdt = data[pos] >> 2, np.dtype(("u1", "u2", "u4", "u8")[data[pos] & 3])
                shape = tuple(int(x) for x in np.frombuffer(data, sdt, ndim, pos + 1))
                pos += 1 + ndim * sdt.itemsize
            yield np.frombuffer(data, dtype, (end - pos) // dtype.itemsize, pos).reshape(shape)


_MDS_DTYPES = {8: "uint8", 9: "int8", 16: "uint16", 17: "int16", 18: "float16", 32: "uint32", 33: "int32",
               34: "float32", 64: "uint64", 65: "int64", 66: "float64"}


def source_documents(windows):
    """(content ids, ended genuinely) for every source document of a unit (step 1 of the module docstring)."""
    cur, cur_last = [], None

    def finish(parts):
        if not parts:
            return None
        seg = np.concatenate(parts)
        if len(seg) == 0:
            return None
        return seg[seg < L3_SPECIAL], bool(seg[-1] == L3_EOS)

    for ii in windows:
        L = len(ii)
        if L == 0:
            continue
        if cur and cur_last == L3_EOS:                  # the open document ended exactly at the window seam
            r = finish(cur)
            if r is not None:
                yield r
            cur, cur_last = [], None
        start = ii == L3_BOS
        start[np.where(ii[:-1] == L3_EOS)[0] + 1] = True
        cuts = np.where(start)[0]
        if len(cuts) == 0:
            cur.append(ii)
            cur_last = int(ii[-1])
            continue
        if cuts[0] > 0:
            cur.append(ii[:cuts[0]])
        r = finish(cur)
        if r is not None:
            yield r
        bounds = list(cuts) + [L]
        for k in range(len(cuts) - 1):
            r = finish([ii[bounds[k]:bounds[k + 1]]])
            if r is not None:
                yield r
        cur, cur_last = [ii[cuts[-1]:]], int(ii[-1])
    r = finish(cur)
    if r is not None:
        yield r


class Llama3Decoder:
    """Llama-3 ids -> text exactly as tiktoken decodes them: the bytes of each id, joined, decoded as UTF-8 with
    replacement. The model file is Meta's tiktoken BPE file (one `base64(token bytes) rank` line per id)."""

    def __init__(self, path: str):
        table = []
        with open(path, "rb") as f:
            for line in f.read().splitlines():
                if line:
                    tok, rank = line.split()
                    if int(rank) != len(table):
                        raise ValueError(f"{path}: rank {int(rank)} out of order")
                    table.append(base64.b64decode(tok))
        if len(table) != L3_SPECIAL:
            raise ValueError(f"{path}: {len(table)} ids, expected {L3_SPECIAL}")
        self._get = table.__getitem__

    def decode(self, ids: np.ndarray) -> str:
        return b"".join(map(self._get, ids.tolist())).decode("utf-8", "replace")


# ---- one unit -------------------------------------------------------------------------------------------------------
_DEC = _ENC = None


def _init(llama3: str, llama2: str):
    global _DEC, _ENC
    import sentencepiece
    _DEC = Llama3Decoder(llama3)
    _ENC = sentencepiece.SentencePieceProcessor(model_file=llama2)
    if _ENC.vocab_size() != L2_VOCAB or _ENC.bos_id() != L2_BOS or _ENC.eos_id() != L2_EOS:
        raise ValueError(f"{llama2} is not the Llama-2 tokenizer")


def repack_unit(task):
    """Steps 1-4 for one unit -> <prefix>.bin (its windows) + <prefix>.lengths.npy (its document lengths)."""
    proc_dir, prefix = task
    if os.path.exists(prefix + ".lengths.npy"):         # finished by an earlier invocation
        L = np.load(prefix + ".lengths.npy")
        return prefix, int(L.sum()), len(L), -1
    lens, off, cur_start, prev_last, maxid = [], 0, 0, None, 0
    buf, n = [], 0
    with open(prefix + ".bin", "wb") as out:
        for content, genuine in source_documents(mds_column(proc_dir, "input_ids")):
            if len(content) == 0:
                continue
            ids = _ENC.encode(_DEC.decode(content))
            doc = np.asarray([L2_BOS] + ids + ([L2_EOS] if genuine else []), dtype=np.uint16)
            buf.append(doc)
            n += len(doc)
            while n >= PACK:
                cat = np.concatenate(buf)
                window, spans, s = cat[:PACK], [], 0
                for seq in buf:                          # the window's document spans [s, min(s + len, PACK))
                    spans.append(s)
                    s += len(seq)
                    if s >= PACK:
                        break
                for k, a in enumerate(spans):            # step 4: recover document starts
                    g = off + a
                    if g == 0 or (k == 0 and prev_last != L2_EOS):
                        continue
                    lens.append(g - cur_start)
                    cur_start = g
                window.tofile(out)
                maxid = max(maxid, int(window.max()))
                prev_last = int(window[-1])
                off += PACK
                rem = cat[PACK:]
                buf, n = ([rem], len(rem)) if len(rem) else ([], 0)
    if off:
        lens.append(off - cur_start)
    np.save(prefix + ".lengths.tmp.npy", np.asarray(lens, dtype=np.int64))
    os.replace(prefix + ".lengths.tmp.npy", prefix + ".lengths.npy")
    return prefix, off, len(lens), maxid


def units(src: str):
    """[(output domain, unit name, unit directory)] in the concatenation order."""
    out = []
    for dom, srcs in DOMAINS:
        found = []
        for s in srcs:
            d = os.path.join(src, s)
            procs = sorted(p for p in os.listdir(d) if p.startswith("proc") and
                           os.path.isfile(os.path.join(d, p, "index.json"))) if os.path.isdir(d) else []
            if len(procs) != PROCS_PER_DIR:
                sys.exit(f"[stage2] {d}: {len(procs)} proc*/index.json, expected {PROCS_PER_DIR} (incomplete download?)")
            found += [(f"{s}__{p}", os.path.join(d, p)) for p in procs]
        out += [(dom, name, path) for name, path in sorted(found)]
    return out


def unit_bytes(path: str) -> int:
    with open(os.path.join(path, "index.json")) as f:
        return sum(sh["raw_data"]["bytes"] for sh in json.load(f)["shards"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the downloaded prolong-data-512K directory")
    ap.add_argument("--llama3", required=True, help="Meta's Llama-3 tokenizer.model (tiktoken BPE file)")
    ap.add_argument("--llama2", required=True, help="the Llama-2 tokenizer.model (SentencePiece)")
    ap.add_argument("--out", required=True, help="output prefix: <out>.bin + <out>.doc_lengths.npy")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--keep-shards", action="store_true", help="keep the per-unit files (<out>_shards/)")
    a = ap.parse_args()

    todo = units(a.src)
    shard_dir = a.out + "_shards"
    os.makedirs(shard_dir, exist_ok=True)
    tasks = [(path, os.path.join(shard_dir, f"{i:03d}_{name}")) for i, (_, name, path) in enumerate(todo)]
    by_size = sorted(tasks, key=lambda t: -unit_bytes(t[0]))    # largest first: the pool finishes together
    print(f"[stage2] {len(tasks)} units over {len(DOMAINS)} domains -> {a.out}.bin (workers={a.workers})", flush=True)
    t0, done, res = time.time(), 0, {}
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.llama3, a.llama2)) as pool:
        for prefix, ntok, ndocs, maxid in pool.imap_unordered(repack_unit, by_size):
            res[prefix] = (ntok, ndocs)
            done += 1
            if maxid >= L2_VOCAB:
                sys.exit(f"[stage2] {prefix}: token id {maxid} >= {L2_VOCAB}")
            if done % 50 == 0 or done == len(tasks):
                print(f"  [{done:>3}/{len(tasks)}] {os.path.basename(prefix)}: {ntok // PACK} windows "
                      f"({time.time() - t0:.0f}s)", flush=True)

    lengths = [np.load(prefix + ".lengths.npy") for _, prefix in tasks]
    per_dom = {}
    with open(a.out + ".bin", "wb") as fo:
        for (dom, _, _), (_, prefix) in zip(todo, tasks):
            with open(prefix + ".bin", "rb") as fi:
                shutil.copyfileobj(fi, fo, length=1 << 24)
            if not a.keep_shards:                           # a unit's files go together: a rerun redoes only those
                os.remove(prefix + ".lengths.npy")
                os.remove(prefix + ".bin")
            per_dom[dom] = per_dom.get(dom, 0) + res[prefix][0] // PACK
    L = np.concatenate(lengths)
    np.save(a.out + ".doc_lengths.npy", L)
    ntok = os.path.getsize(a.out + ".bin") // 2
    if ntok != int(L.sum()) or ntok % PACK:
        sys.exit(f"[stage2] FAILED: {ntok} tokens, document lengths sum to {int(L.sum())}")
    if not a.keep_shards:
        shutil.rmtree(shard_dir)
    first = 0
    for dom, _ in DOMAINS:
        print(f"  {dom:32s} windows [{first:>7,}, {first + per_dom[dom]:>7,})  {per_dom[dom]:>7,}")
        first += per_dom[dom]
    print(f"[stage2] DONE {a.out}.bin: {ntok:,} tokens = {ntok // PACK:,} windows, {len(L):,} documents "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
