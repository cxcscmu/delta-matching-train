#!/usr/bin/env python3
"""FinePDFs-Edu English, step 2: filter and tokenize (data/pretrain_finepdfs_edu.md). Called by
data/build_finepdfs_edu.py; runnable on its own:

    python data/finepdfs_tokenize.py --raw <raw> --tokenizer <root>/tokenizer/llama2_32k --work <work> [--files 0-3]

Reads the eng_Latn parquet files of HuggingFaceFW/finepdfs-edu (one document per row; pyarrow only, nothing of the
dataset is executed) and checks every document against these rules, in this order. A dropped document is counted
under the first rule it fails (the cascade) and, separately, under every rule it fails (stand-alone):

  L1  full_doc_lid == eng_Latn                                  \\ together: the dataset card's keep_document() for
  L2  #pages labeled eng_Latn >= #pages // 2                     / English (per_page_languages)
  E   edu = max(fw_edu_scores) >= --edu-min (1.5; the -edu set starts at 1.32 for English)
  L3  the non-English pages hold <= 10 % of the attributed characters. Pages are cut from the text at page_ends
      clipped to len(text); a page is non-English only if its label is not eng_Latn, not 'unknown', not zxx_* and its
      text is not English-like (>= 20 ASCII words [A-Za-z]+, >= 8 % of them, lowercased, in FW28)
  G1  the characters of repeated lines (stripped, non-empty: sum len(line) * (count - 1)) are <= 50 % of the characters
  G2  ASCII letters are >= 30 % of the characters
  G3  function-word rate >= 2 %: #matches of (?<!\\w)(the|of|and|to|in|is|that|for|with|as)(?!\\w) in text.lower(),
      over max(len(text.split()), 1)
  G4  >= 50 Llama-2 content tokens (checked after tokenization)

No text is rewritten. A document that passes every rule is tokenized exactly as

    AutoTokenizer.from_pretrained(<tokenizer>)(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]

(the stage-1 call, data/stage1_tokenize.py, plus split_special_tokens: a literal "<s>", "</s>" or "<unk>" in the text is
encoded as text, never as a special id) and laid down as [BOS] + ids + [EOS]; an id 0 (unk), 1 (BOS) or 2 (EOS) inside
a document stops the build. Documents with --edu-min > edu >= min(--edu-thresholds) that pass every other rule are
tokenized too but only counted (the exact pool size at each threshold of --edu-thresholds, for the manifest).

Work: one task per (file, quarter of its row groups), in a fixed partition, so the outputs depend neither on --workers
nor on --threads-per. Each task writes to <work>/shards/:
  task_NNNN.bin          the kept documents' tokens (uint16, [BOS] + ids + [EOS], in row order)
  task_NNNN.lengths.npy  their lengths (int64, len(ids) + 2)
  task_NNNN.docs.parquet one row per counted document: file, row, id, norm_url, key, n_tokens, edu, in_shard, tok_off,
                         extractor, special_literal (in_shard: edu >= --edu-min, its tokens are in task_NNNN.bin)
  task_NNNN.stats.json   per-rule drops (documents, characters, the dataset's token_count, Llama-2 tokens once known),
                         timings; written last: a task with its stats.json is done and is skipped on a rerun with the
                         same parameters
and <work>/tokenize_summary.json sums them up.

norm_url = url.strip() without http:// or https:// (any case), the host lowercased and a leading 'www.' removed (path
and query unchanged); key = BLAKE2b-64(b'finepdfs-edu-en|<seed>|' + norm_url), big-endian: the sampling key of
data/finepdfs_select.py (documents of one norm_url share their key, hence their split).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np

LANG = "eng_Latn"
FW28 = frozenset(("the", "of", "and", "to", "in", "is", "that", "for", "with", "as", "are", "be", "this", "by", "on",
                  "or", "an", "it", "from", "at", "which", "was", "were", "not", "can", "will", "have", "has"))
FW10_RE = re.compile(r"(?<!\w)(the|of|and|to|in|is|that|for|with|as)(?!\w)")
ASCII_WORD_RE = re.compile(r"[A-Za-z]+")
SCHEME_RE = re.compile(r"(?i)https?://")
LETTERS = bytes(range(65, 91)) + bytes(range(97, 123))
SPECIAL_LITERALS = ("<s>", "</s>", "<unk>")
NAMESPACE = b"finepdfs-edu-en"

EDU_MIN = 1.5
EDU_THRESHOLDS = (1.4, 1.45, 1.5, 1.55, 1.6, 1.65)
NONEN_MAX = 0.10            # L3
ENGLISH_LIKE_WORDS = 20     # L3: an English-like page has >= 20 ASCII words ...
ENGLISH_LIKE_FW = 0.08      # ... and >= 8 % of them in FW28
DUP_MAX = 0.50              # G1
LETTERS_MIN = 0.30          # G2
FW_RATE_MIN = 0.02          # G3
MIN_CONTENT_TOKENS = 50     # G4
RULES = ("L1", "L2", "E", "L3", "G1", "G2", "G3", "G4")
TEXT_RULES = ("L1", "L2", "E", "L3", "G1", "G2", "G3")      # decided before tokenization
TASKS_PER_FILE = 4
COLUMNS = ["text", "id", "url", "token_count", "full_doc_lid", "per_page_languages", "page_ends", "fw_edu_scores",
           "extractor"]
RULE_TEXT = {
    "L1": "full_doc_lid == eng_Latn",
    "L2": "count(per_page_languages == eng_Latn) >= len(per_page_languages) // 2 (with L1: the card's keep_document)",
    "E": "edu = max(fw_edu_scores) >= edu_min",
    "L3": ("chars of non-English pages <= 10 % of the attributed chars; pages = text cut at page_ends clipped to "
           "len(text) (prev = max(prev, min(end, n))); non-English = label not eng_Latn, not 'unknown', not zxx_* and "
           "not English-like (>= 20 ASCII words [A-Za-z]+, >= 8 % of them lowercased in FW28)"),
    "G1": "sum(len(line) * (count - 1)) over repeated stripped non-empty lines <= 50 % of len(text)",
    "G2": "ASCII letters [A-Za-z] >= 30 % of len(text)",
    "G3": ("len(re.findall(r'(?<!\\w)(the|of|and|to|in|is|that|for|with|as)(?!\\w)', text.lower())) / "
           "max(len(text.split()), 1) >= 0.02"),
    "G4": "len(Llama-2 ids) >= 50 (after tokenization)",
}


# ---- the per-document definitions (pure functions; tests/test_finepdfs_edu.py) -------------------------------------
def norm_url(url: str | None) -> str:
    """url.strip(), without its http:// or https:// (any case), the host (up to the first '/', '?' or '#') lowercased
    and a leading 'www.' removed; the path and query unchanged."""
    u = (url or "").strip()
    m = SCHEME_RE.match(u)
    if m:
        u = u[m.end():]
    cut = len(u)
    for ch in "/?#":
        i = u.find(ch)
        if 0 <= i < cut:
            cut = i
    host = u[:cut].lower()
    if host.startswith("www."):
        host = host[4:]
    return host + u[cut:]


def url_key(nu: str, seed: int) -> int:
    """The sampling key: BLAKE2b-64(b'finepdfs-edu-en|<seed>|' + norm_url as UTF-8), read as a big-endian uint64."""
    h = hashlib.blake2b(NAMESPACE + b"|" + str(int(seed)).encode() + b"|" + nu.encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "big")


def edu_score(scores) -> float:
    return max(scores) if scores else math.nan


def english_like(seg: str) -> bool:
    words = ASCII_WORD_RE.findall(seg)
    if len(words) < ENGLISH_LIKE_WORDS:
        return False
    return sum(1 for w in words if w.lower() in FW28) >= ENGLISH_LIKE_FW * len(words)


def nonenglish_chars(text: str, page_ends, labels) -> tuple[int, int]:
    """(characters on non-English pages, characters attributed to any page) of rule L3."""
    n = len(text)
    prev = attributed = nonen = 0
    for k, end in enumerate(page_ends):
        e = min(end, n)
        if e > prev:
            lab = labels[k] if k < len(labels) else None
            attributed += e - prev
            if (lab and lab != LANG and lab != "unknown" and not lab.startswith("zxx_")
                    and not english_like(text[prev:e])):
                nonen += e - prev
        prev = max(prev, e)
    return nonen, attributed


def dup_line_chars(text: str) -> int:
    cnt = collections.Counter(s for s in (ln.strip() for ln in text.split("\n")) if s)
    return sum(len(s) * (c - 1) for s, c in cnt.items() if c > 1)


def ascii_letters(text: str) -> int:
    b = text.encode("utf-8")
    return len(b) - len(b.translate(None, LETTERS))


def fw_rate(text: str) -> float:
    return len(FW10_RE.findall(text.lower())) / max(len(text.split()), 1)


def text_rules(text: str, full_doc_lid, labels, page_ends, scores, edu_min: float) -> tuple[dict, float]:
    """{rule: passed} for the rules decided before tokenization, and the document's edu score."""
    n = len(text)
    edu = edu_score(scores)
    nonen, attributed = nonenglish_chars(text, page_ends, labels)
    ok = {
        "L1": full_doc_lid == LANG,
        "L2": sum(1 for x in labels if x == LANG) >= len(labels) // 2,
        "E": edu >= edu_min,                           # False for a document without scores (nan)
        "L3": nonen <= NONEN_MAX * attributed,
        "G1": dup_line_chars(text) / max(n, 1) <= DUP_MAX,
        "G2": ascii_letters(text) / max(n, 1) >= LETTERS_MIN,
        "G3": fw_rate(text) >= FW_RATE_MIN,
    }
    return ok, edu


# ---- the worker ----------------------------------------------------------------------------------------------------
_TOK = None
_SPEC = None     # (bos, eos, bad-id lookup table, vocab)


def _init(tok_dir: str, threads_per: int):
    """Per-worker init: pin the tokenizer threads, load the fast tokenizer once."""
    global _TOK, _SPEC
    os.environ["RAYON_NUM_THREADS"] = str(threads_per)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(tok_dir)
    bos, eos, unk = _TOK.bos_token_id, _TOK.eos_token_id, _TOK.unk_token_id
    assert bos is not None and eos is not None, "the tokenizer has no BOS / EOS"
    vocab = len(_TOK)
    bad = np.zeros(max(vocab, 65536), dtype=bool)
    for t in (bos, eos, unk):
        if t is not None:
            bad[t] = True
    _SPEC = (int(bos), int(eos), bad, int(vocab))


def encode(texts: list[str]) -> list[list[int]]:
    """The tokenizer call of the recipe (see the module docstring)."""
    return _TOK(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]


def _counter():
    return {"docs": 0, "chars": 0, "token_count": 0}


def _add(c: dict, nc: int, tc: int, l2: int | None = None):
    c["docs"] += 1
    c["chars"] += nc
    c["token_count"] += tc
    if l2 is not None:
        c["l2_tokens"] = c.get("l2_tokens", 0) + l2


def run_task(task: dict) -> dict:
    """Filter and tokenize the row groups [rg_lo, rg_hi) of one file (see the module docstring for the outputs)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t_start = time.time()
    shards = task["shards"]
    stem = os.path.join(shards, f"task_{task['task']:04d}")
    stats_path = stem + ".stats.json"
    if os.path.exists(stats_path) and all(os.path.exists(stem + x) for x in (".bin", ".lengths.npy", ".docs.parquet")):
        with open(stats_path) as f:
            old = json.load(f)
        if old.get("params") == task["params"]:
            old["skipped"] = True
            return old
    bos, eos, bad, vocab = _SPEC
    edu_min, count_min, seed = task["params"]["edu_min"], task["params"]["count_min"], task["params"]["seed"]
    thresholds = task["params"]["edu_thresholds"]
    batch_chars = task["batch_chars"]
    pf = pq.ParquetFile(task["path"])
    md = pf.metadata
    row0 = sum(md.row_group(g).num_rows for g in range(task["rg_lo"]))
    totals = _counter()
    cascade = {r: _counter() for r in RULES}
    standalone = {r: _counter() for r in TEXT_RULES}
    pool = dict(_counter(), tokens=0)
    thr = {f"{t:g}": {"docs": 0, "tokens": 0} for t in thresholds}
    extractor = collections.defaultdict(lambda: {"docs": 0, "tokens": 0})
    D = collections.defaultdict(list)         # the docs table
    lengths = []
    special_docs = maxid = tok_docs = tok_tokens = 0
    t_read = t_stats = t_tok = t_write = 0.0
    tok_off = 0
    rg_row0 = row0                          # the file row of the current row group's first row
    with open(stem + ".bin.tmp", "wb") as fbin:
        for rg in range(task["rg_lo"], task["rg_hi"]):
            if rg > task["rg_lo"]:
                rg_row0 += md.row_group(rg - 1).num_rows
            t0 = time.time()
            tb = pf.read_row_group(rg, columns=COLUMNS)
            cols = {c: tb.column(c).to_pylist() for c in COLUMNS}
            del tb
            n = len(cols["text"])
            t1 = time.time()
            t_read += t1 - t0
            cand = []                        # (row in rg, edu, in the pool) of the documents to tokenize
            for i in range(n):
                text = cols["text"][i] or ""
                nc, tc = len(text), int(cols["token_count"][i] or 0)
                ok, edu = text_rules(text, cols["full_doc_lid"][i], cols["per_page_languages"][i] or [],
                                     cols["page_ends"][i] or [], cols["fw_edu_scores"][i] or [], edu_min)
                _add(totals, nc, tc)
                for r in TEXT_RULES:
                    if not ok[r]:
                        _add(standalone[r], nc, tc)
                first = next((r for r in TEXT_RULES if not ok[r]), None)
                if first is not None:
                    _add(cascade[first], nc, tc)
                if first is None or (first == "E" and edu >= count_min
                                     and all(ok[r] for r in TEXT_RULES if r != "E")):
                    cand.append((i, edu, first is None))
            t2 = time.time()
            t_stats += t2 - t1
            # tokenize the candidates, in batches of at most batch_chars characters (a longer document alone)
            ids_of = {}
            batch, size = [], 0
            for i, _, _ in cand + [(None, None, None)]:
                if i is not None and (not batch or size + len(cols["text"][i]) <= batch_chars):
                    batch.append(i)
                    size += len(cols["text"][i])
                    continue
                if batch:
                    for j, ids in zip(batch, encode([cols["text"][j] for j in batch])):
                        ids_of[j] = ids
                if i is not None:
                    batch, size = [i], len(cols["text"][i])
            t3 = time.time()
            t_tok += t3 - t2
            keep_ids, keep_rows = [], []
            for i, edu, in_pool in cand:
                ids = ids_of.pop(i)
                text = cols["text"][i]
                nc, tc, nt = len(text), int(cols["token_count"][i] or 0), len(ids) + 2
                tok_docs += 1
                tok_tokens += nt
                if len(ids) < MIN_CONTENT_TOKENS:
                    if in_pool:
                        _add(cascade["G4"], nc, tc, nt)
                    continue
                for t in thresholds:
                    if edu >= t:
                        thr[f"{t:g}"]["docs"] += 1
                        thr[f"{t:g}"]["tokens"] += nt
                nu = norm_url(cols["url"][i])
                lit = any(s in text for s in SPECIAL_LITERALS)
                D["file"].append(task["file_idx"])
                D["row"].append(rg_row0 + i)
                D["id"].append(cols["id"][i])
                D["norm_url"].append(nu)
                D["key"].append(url_key(nu, seed))
                D["n_tokens"].append(nt)
                D["edu"].append(edu)
                D["in_shard"].append(in_pool)
                D["tok_off"].append(tok_off if in_pool else -1)
                D["extractor"].append(cols["extractor"][i])
                D["special_literal"].append(lit)
                if not in_pool:
                    continue
                keep_ids.append(ids)
                keep_rows.append(rg_row0 + i)
                tok_off += nt
                _add(pool, nc, tc)
                pool["tokens"] += nt
                special_docs += lit
                ex = extractor[str(cols["extractor"][i])]
                ex["docs"] += 1
                ex["tokens"] += nt
            if keep_ids:                     # [BOS] + ids + [EOS] per document, back to back (data/stage1_tokenize.py)
                lens = np.fromiter((len(x) for x in keep_ids), dtype=np.int64, count=len(keep_ids))
                content = np.concatenate([np.asarray(x, dtype=np.int64) for x in keep_ids])
                m, lo = int(content.max()), int(content.min())
                if m >= vocab or lo < 0:
                    raise RuntimeError(f"{task['rel']}: token id {m if m >= vocab else lo} outside 0..{vocab - 1}")
                if bad[content].any():
                    k = int(np.searchsorted(np.cumsum(lens), int(np.flatnonzero(bad[content])[0]), side="right"))
                    raise RuntimeError(f"{task['rel']} row {keep_rows[k]}: a special id (unk / BOS / EOS) inside a "
                                       "document: the tokenizer call no longer encodes special strings as text")
                maxid = max(maxid, m)
                outlen = lens + 2
                ooff = np.zeros(len(lens) + 1, dtype=np.int64)
                np.cumsum(outlen, out=ooff[1:])
                out = np.empty(int(ooff[-1]), dtype=np.uint16)
                out[ooff[:-1]] = bos
                out[ooff[1:] - 1] = eos
                mask = np.ones(len(out), dtype=bool)
                mask[ooff[:-1]] = False
                mask[ooff[1:] - 1] = False
                out[mask] = content
                out.tofile(fbin)
                lengths.append(outlen)
            del cols, ids_of, keep_ids, keep_rows
            t_write += time.time() - t3
    L = np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int64)
    assert int(L.sum()) == pool["tokens"] == tok_off
    assert os.path.getsize(stem + ".bin.tmp") == 2 * int(L.sum())
    np.save(stem + ".lengths.tmp.npy", L)
    schema = pa.schema([("file", pa.int16()), ("row", pa.int64()), ("id", pa.string()), ("norm_url", pa.string()),
                        ("key", pa.uint64()), ("n_tokens", pa.int64()), ("edu", pa.float64()), ("in_shard", pa.bool_()),
                        ("tok_off", pa.int64()), ("extractor", pa.string()), ("special_literal", pa.bool_())])
    pq.write_table(pa.table({f.name: D.get(f.name, []) for f in schema}, schema=schema), stem + ".docs.tmp.parquet",
                   compression="zstd")
    os.replace(stem + ".bin.tmp", stem + ".bin")
    os.replace(stem + ".lengths.tmp.npy", stem + ".lengths.npy")
    os.replace(stem + ".docs.tmp.parquet", stem + ".docs.parquet")
    stats = {
        "params": task["params"], "task": task["task"], "file_idx": task["file_idx"], "rel": task["rel"],
        "rg_lo": task["rg_lo"], "rg_hi": task["rg_hi"], "row0": row0,
        "rows": sum(md.row_group(g).num_rows for g in range(task["rg_lo"], task["rg_hi"])),
        "totals": totals, "cascade": cascade, "standalone": standalone, "pool": pool,
        "thresholds": thr, "special_literal_docs": special_docs, "extractor": dict(extractor), "maxid": maxid,
        "tokenized": {"docs": tok_docs, "tokens": tok_tokens},
        "timing": {"read_s": round(t_read, 3), "stats_s": round(t_stats, 3), "tokenize_s": round(t_tok, 3),
                   "write_s": round(t_write, 3), "wall_s": round(time.time() - t_start, 3)},
        "host": os.uname().nodename,
    }
    with open(stats_path + ".tmp", "w") as f:
        json.dump(stats, f, indent=1)
    os.replace(stats_path + ".tmp", stats_path)
    return stats


# ---- the driver ----------------------------------------------------------------------------------------------------
def parse_files(spec: str | None, n: int) -> list[int]:
    """'0-3,7' (indices into the listing), or None for all."""
    if not spec:
        return list(range(n))
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    bad = [i for i in out if not 0 <= i < n]
    if bad:
        raise SystemExit(f"--files: indices {sorted(bad)} outside the listing's 0..{n - 1}")
    return sorted(out)


def load_listing(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def make_tasks(raw: str, listing: dict, files: list[int], shards: str, params: dict, batch_chars: int) -> list[dict]:
    import pyarrow.parquet as pq
    tasks = []
    for fi in files:
        ent = listing["files"][fi]
        path = os.path.join(raw, ent["path"])
        if not os.path.isfile(path):
            raise SystemExit(f"[finepdfs-tokenize] missing {path}: run the download step first")
        if os.path.getsize(path) != ent["bytes"]:
            raise SystemExit(f"[finepdfs-tokenize] {path}: {os.path.getsize(path)} bytes, the listing says {ent['bytes']}")
        md = pq.ParquetFile(path).metadata
        if ("rows" in ent and md.num_rows != ent["rows"]) or ("row_groups" in ent and md.num_row_groups != ent["row_groups"]):
            raise SystemExit(f"[finepdfs-tokenize] {path}: {md.num_rows} rows / {md.num_row_groups} row groups, the "
                             f"listing says {ent.get('rows')} / {ent.get('row_groups')}")
        R = md.num_row_groups
        bounds = [R * q // TASKS_PER_FILE for q in range(TASKS_PER_FILE + 1)]
        for q in range(TASKS_PER_FILE):
            if bounds[q] == bounds[q + 1]:
                continue
            p = dict(params, file=ent["path"], file_sha256=ent.get("sha256"), rg=[bounds[q], bounds[q + 1]])
            tasks.append({"task": fi * TASKS_PER_FILE + q, "file_idx": fi, "rel": ent["path"], "path": path,
                          "rg_lo": bounds[q], "rg_hi": bounds[q + 1], "shards": shards, "params": p,
                          "batch_chars": batch_chars})
    return tasks


def tokenizer_digest(tok_dir: str) -> dict:
    out = {}
    for name in sorted(os.listdir(tok_dir)):
        p = os.path.join(tok_dir, name)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                out[name] = hashlib.sha256(f.read()).hexdigest()
    return out


def summarize(results: list[dict], wall: float) -> dict:
    def tot(get, keys):
        out = {}
        for k in keys:
            acc = {}
            for r in results:
                for kk, v in get(r)[k].items():
                    acc[kk] = acc.get(kk, 0) + v
            out[k] = acc
        return out
    s = {
        "tasks": len(results), "files": sorted({r["file_idx"] for r in results}),
        "totals": {k: sum(r["totals"][k] for r in results) for k in ("docs", "chars", "token_count")},
        "cascade": tot(lambda r: r["cascade"], RULES),
        "standalone": tot(lambda r: r["standalone"], TEXT_RULES),
        "pool_before_dedup": {k: sum(r["pool"][k] for r in results) for k in ("docs", "chars", "token_count", "tokens")},
        "thresholds_before_dedup": tot(lambda r: r["thresholds"], list(results[0]["thresholds"]) if results else []),
        "special_literal_docs": sum(r["special_literal_docs"] for r in results),
        "maxid": max((r["maxid"] for r in results), default=0),
        "tokenized": {k: sum(r["tokenized"][k] for r in results) for k in ("docs", "tokens")},
    }
    ex = {}
    for r in results:
        for k, v in r["extractor"].items():
            e = ex.setdefault(k, {"docs": 0, "tokens": 0})
            e["docs"] += v["docs"]
            e["tokens"] += v["tokens"]
    s["extractor"] = ex
    tim = {k: round(sum(r["timing"][k] for r in results), 1) for k in ("read_s", "stats_s", "tokenize_s", "write_s",
                                                                        "wall_s")}
    fresh = [r for r in results if not r.get("skipped")]
    s["timing"] = {"wall_s": round(wall, 1), "task_sums": tim, "tasks_run": len(fresh),
                   "tokens_per_s": round(sum(r["tokenized"]["tokens"] for r in fresh) / max(wall, 1e-9)),
                   "chars_per_s": round(sum(r["totals"]["chars"] for r in fresh) / max(wall, 1e-9))}
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="the download dir (<raw>/data/eng_Latn/train/*.parquet)")
    ap.add_argument("--files-json", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "finepdfs_edu_files.json"), help="the pinned file listing")
    ap.add_argument("--files", default=None, help="indices into the listing, e.g. '0-3,7' (default: every file)")
    ap.add_argument("--tokenizer", required=True, help="the Llama-2 tokenizer dir (<root>/tokenizer/llama2_32k)")
    ap.add_argument("--work", required=True, help="work dir: the shards go to <work>/shards")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threads-per", type=int, default=4, help="tokenizer threads per worker (RAYON_NUM_THREADS)")
    ap.add_argument("--edu-min", type=float, default=EDU_MIN)
    ap.add_argument("--edu-thresholds", default=",".join(f"{t:g}" for t in EDU_THRESHOLDS),
                    help="edu thresholds at which the exact pool size is counted (documents down to the lowest are "
                         "tokenized)")
    ap.add_argument("--seed", type=int, default=11785, help="the sampling seed (part of the key)")
    ap.add_argument("--batch-chars", type=int, default=16_000_000, help="characters per tokenizer call")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    vocab = len(tok)
    if vocab > 65536:
        raise SystemExit(f"--tokenizer {a.tokenizer} has {vocab} ids: they do not fit the uint16 output")
    thresholds = sorted({round(float(x), 6) for x in a.edu_thresholds.split(",") if x.strip()})
    count_min = min(thresholds + [a.edu_min])
    listing = load_listing(a.files_json)
    files = parse_files(a.files, len(listing["files"]))
    shards = os.path.join(a.work, "shards")
    os.makedirs(shards, exist_ok=True)
    params = {"edu_min": a.edu_min, "count_min": count_min, "edu_thresholds": thresholds, "seed": a.seed,
              "tokenizer_sha256": tokenizer_digest(a.tokenizer), "vocab": vocab,
              "bos": tok.bos_token_id, "eos": tok.eos_token_id, "unk": tok.unk_token_id,
              "rules": RULE_TEXT, "version": 1}
    tasks = make_tasks(a.raw, listing, files, shards, params, a.batch_chars)
    print(f"[finepdfs-tokenize] {len(files)} files, {len(tasks)} tasks -> {shards} (workers={a.workers} "
          f"threads/worker={a.threads_per} edu_min={a.edu_min} counted down to {count_min})", flush=True)
    t0 = time.time()
    results = []
    done_tok = 0
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(a.tokenizer, a.threads_per)) as pool:
        for r in pool.imap_unordered(run_task, tasks):
            results.append(r)
            el = time.time() - t0
            if not r.get("skipped"):
                done_tok += r["tokenized"]["tokens"]
            t = r["timing"]
            print(f"[task {r['task']:>4}] {os.path.basename(r['rel'])} rg {r['rg_lo']}-{r['rg_hi']}: "
                  f"docs={r['totals']['docs']:,} pool={r['pool']['docs']:,} docs / {r['pool']['tokens']:,} tok"
                  f"{' (done before)' if r.get('skipped') else ''} | read {t['read_s']:.0f}s stats {t['stats_s']:.0f}s "
                  f"tok {t['tokenize_s']:.0f}s write {t['write_s']:.0f}s | {len(results)}/{len(tasks)} tasks, "
                  f"{el:.0f}s, {done_tok / max(el, 1e-9) / 1e6:.2f} M tok/s", flush=True)
    results.sort(key=lambda r: r["task"])
    wall = time.time() - t0
    s = summarize(results, wall)
    s["params"] = {k: v for k, v in params.items()}
    s["workers"], s["threads_per"] = a.workers, a.threads_per
    s["task_ids"] = [r["task"] for r in results]
    with open(os.path.join(a.work, "tokenize_summary.json.tmp"), "w") as f:
        json.dump(s, f, indent=1)
    os.replace(os.path.join(a.work, "tokenize_summary.json.tmp"), os.path.join(a.work, "tokenize_summary.json"))
    p = s["pool_before_dedup"]
    print(f"[finepdfs-tokenize] DONE {s['totals']['docs']:,} documents -> pool {p['docs']:,} documents, "
          f"{p['tokens']:,} tokens (before id dedup); tokenized {s['tokenized']['tokens']:,} tokens in {wall:.0f}s "
          f"({s['timing']['tokens_per_s'] / 1e6:.2f} M tok/s); maxid {s['maxid']}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
