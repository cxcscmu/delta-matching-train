#!/usr/bin/env python3
"""Build the FinePDFs-Edu English pretraining corpus, a 60 B-token quality-filtered random document sample of
HuggingFaceFW/finepdfs-edu in the stage-1 format (data/pretrain_finepdfs_edu.md):

    python data/build_finepdfs_edu.py --root $DM_DATA_ROOT                     # every step
    python data/build_finepdfs_edu.py --root $DM_DATA_ROOT --step tokenize     # one step (repeatable)

  1. download  the 100 eng_Latn parquet files of the public dataset HuggingFaceFW/finepdfs-edu at a pinned revision
               (data/finepdfs_edu_files.json: 298.7 GB, 23.0 M documents) into --raw, --dl-concurrency at a time; each
               file is checked against its LFS sha256 (never skipped, also with --revision) and a verified file is not
               fetched again on a rerun (its <file>.sha256ok marker)
  2. tokenize  per-document English and quality rules, then [BOS] + Llama-2 ids + [EOS] (data/finepdfs_tokenize.py)
               into <work>/shards; a rerun skips the finished tasks
  3. select    exact-id dedup, URL groups, a keyed random order of the groups; validation >= --val-tokens, training
               >= --train-tokens tokens, whole groups only (data/finepdfs_select.py)
  4. write     <work>/out/finepdfs_edu_en_{train,val}.bin + .doc_lengths.npy (uint16, the stage-1 layout) and
               selection.parquet; the stream invariants; sha256 of every output (data/finepdfs_select.py)
  5. publish   copy <work>/out to <root>/.<out-name>.tmp-<job> (--copy-streams parallel writers per file), re-check
               bytes and sha256 there, move it to <root>/<out-name>/ (refused when that exists, unless --force: the old
               one is moved aside, not deleted), write README.md and, last, manifest.json (the provenance: source
               revision and files, every rule with its drops, dedup, URL groups, seed and key, targets and what was
               reached, pool sizes per edu threshold, tokenizer and software versions, script hashes, job, times,
               output sha256s). The default build (all files, the default targets, seed, threshold, revision and
               out-name) is refused when its outputs differ from data/finepdfs_edu_checksums.json
  6. verify    the published streams' bytes and sha256 against manifest.json (and finepdfs_edu_checksums.json), their
               invariants, and windows read through the training loader (DocAwareMemmapDataset, build_train_dataloader);
               the first default build that passes writes data/finepdfs_edu_checksums.json (--checksums)

Needs the Llama-2 tokenizer (data/fetch_tokenizer.py; --tokenizer, checked against data/checksums.json) and, for the
download, network access to the Hugging Face Hub (anonymous works; a login raises the rate limits). Disk: 299 GB of
parquet, ~190 GB of shards, ~125 GB of outputs in --work, ~125 GB under --root at the end. Time on a 96-CPU node:
data/pretrain_finepdfs_edu.md. The outputs depend neither on --workers nor on --threads-per. Downloaded files are
read with pyarrow only; run this with `python -I` (scripts/jobs do) and keep --raw / --work apart from the code.
"""
import argparse
import datetime
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify import load_checksums, sha256  # noqa: E402

REPO_ID = "HuggingFaceFW/finepdfs-edu"
REVISION = "9cfabe2127faca99b3d5c4dc6d1fcb397399ebde"
OUT_NAME = "finepdfs_edu_en_60b"
TRAIN_TOKENS, VAL_TOKENS, SEED, EDU_MIN = 60e9, 8e8, 11785, 1.5
TOKENIZER_SUB = "tokenizer/llama2_32k"
PINNED = {"transformers": "5.9.0", "tokenizers": "0.22.2", "pyarrow": "24.0.0", "numpy": "2.4.6",
          "huggingface_hub": "1.16.4"}
STEPS = ("download", "tokenize", "select", "write", "publish", "verify")
SCRIPTS = ("build_finepdfs_edu.py", "finepdfs_tokenize.py", "finepdfs_select.py", "finepdfs_edu_files.json", "verify.py")
STREAMS = ("finepdfs_edu_en_train.bin", "finepdfs_edu_en_train.doc_lengths.npy", "finepdfs_edu_en_val.bin",
           "finepdfs_edu_en_val.doc_lengths.npy")
PUBLISHED = STREAMS + ("selection.parquet",)


def log(msg):
    print(f"[finepdfs] {msg}", flush=True)


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def load_json(p, default=None):
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def dump_json(p, obj):
    with open(p + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(p + ".tmp", p)


def record_step(work, step, t0, t1, **extra):
    p = os.path.join(work, "steps.json")
    steps = load_json(p, {})
    steps[step] = dict(start=datetime.datetime.fromtimestamp(t0).astimezone().isoformat(timespec="seconds"),
                       end=datetime.datetime.fromtimestamp(t1).astimezone().isoformat(timespec="seconds"),
                       seconds=round(t1 - t0, 1), host=socket.gethostname(), job=os.environ.get("SLURM_JOB_ID"), **extra)
    dump_json(p, steps)


# ---- 1. download ---------------------------------------------------------------------------------------------------
def _marker(dst):
    return dst + ".sha256ok"


def _verified(dst, ent):
    m = _marker(dst)
    if not (os.path.isfile(dst) and os.path.isfile(m) and os.path.getsize(dst) == ent["bytes"]):
        return False
    with open(m) as f:
        return f.read().strip() == ent["sha256"]


def _check_and_mark(dst, ent):
    """Size and sha256 against the listing; a good file gets its marker, a bad one is deleted."""
    if os.path.getsize(dst) == ent["bytes"] and sha256(dst) == ent["sha256"]:
        with open(_marker(dst) + ".tmp", "w") as f:
            f.write(ent["sha256"] + "\n")
        os.replace(_marker(dst) + ".tmp", _marker(dst))
        return True
    os.remove(dst)
    return False


def download(listing, files, raw, revision, concurrency, work):
    from huggingface_hub import hf_hub_download
    todo, t0 = [], time.time()
    for fi in files:
        ent = listing["files"][fi]
        dst = os.path.join(raw, ent["path"])
        if _verified(dst, ent):
            continue
        if os.path.isfile(dst) and _check_and_mark(dst, ent):      # complete from an earlier attempt, not yet marked
            log(f"{ent['path']}: present, sha256 OK")
            continue
        todo.append(ent)
    log(f"download {REPO_ID}@{revision[:12]}: {len(files) - len(todo)}/{len(files)} files already verified, "
        f"{len(todo)} to fetch ({sum(e['bytes'] for e in todo) / 1e9:.1f} GB) -> {raw}, {concurrency} at a time")
    summ_p = os.path.join(work, "download_summary.json")
    summ = load_json(summ_p, {"files": {}, "attempts": 0})
    summ["attempts"] += 1

    def one(ent):
        t = time.time()
        p = hf_hub_download(REPO_ID, ent["path"], repo_type="dataset", revision=revision, local_dir=raw)
        dt = time.time() - t
        if os.path.abspath(p) != os.path.abspath(os.path.join(raw, ent["path"])):
            raise RuntimeError(f"hf_hub_download put {ent['path']} at {p}")
        ok = _check_and_mark(p, ent)
        log(f"{ent['path']}: {ent['bytes'] / 1e9:.2f} GB in {dt:.0f}s ({ent['bytes'] / 1e6 / max(dt, 1e-3):.0f} MB/s) "
            f"sha256 {'OK' if ok else 'MISMATCH: deleted'}")
        if not ok:
            raise RuntimeError(f"{ent['path']}: size or sha256 differs from data/finepdfs_edu_files.json")
        return ent["path"], {"bytes": ent["bytes"], "seconds": round(dt, 1), "MBps": round(ent["bytes"] / 1e6 / max(dt, 1e-3), 1)}

    errors, got, t1 = [], 0, time.time()
    with ThreadPoolExecutor(max(1, concurrency)) as ex:
        futs = [ex.submit(one, e) for e in todo]
        for f in futs:
            try:
                k, v = f.result()
                summ["files"][k] = v
                got += v["bytes"]
            except Exception as e:                 # noqa: BLE001  (reported, retried by the next attempt)
                errors.append(repr(e))
    wall = time.time() - t1
    if todo:
        summ.setdefault("rounds", []).append({"files": len(todo) - len(errors), "bytes": got, "seconds": round(wall, 1),
                                              "MBps": round(got / 1e6 / max(wall, 1e-3), 1), "concurrency": concurrency,
                                              "errors": len(errors), "end": now()})
        log(f"fetched {got / 1e9:.1f} GB in {wall:.0f}s: {got / 1e6 / max(wall, 1e-3):.0f} MB/s aggregate")
    summ["revision"] = revision
    dump_json(summ_p, summ)
    missing = [listing["files"][fi]["path"] for fi in files
               if not _verified(os.path.join(raw, listing["files"][fi]["path"]), listing["files"][fi])]
    if errors or missing:
        for e in errors[:10]:
            log(f"  error: {e}")
        sys.exit(f"[finepdfs] download incomplete: {len(missing)} of {len(files)} files missing or unverified")
    record_step(work, "download", t0, time.time(), files=len(files))
    log(f"download complete: {len(files)} files verified")


# ---- 2.-4. tokenize, select, write ---------------------------------------------------------------------------------
def check_tokenizer(tok_dir, allow_other):
    want = {os.path.basename(k): v for k, v in load_checksums().items() if k.startswith(TOKENIZER_SUB + "/")}
    got, bad = {}, []
    for name, v in sorted(want.items()):
        p = os.path.join(tok_dir, name)
        got[name] = sha256(p) if os.path.isfile(p) else None
        if got[name] != v["sha256"]:
            bad.append(name)
    if bad and not allow_other:
        sys.exit(f"[finepdfs] the tokenizer at {tok_dir} differs from data/checksums.json ({', '.join(bad)}): run "
                 "data/fetch_tokenizer.py, or pass --allow-other-tokenizer (tests only)")
    return {"sha256": got, "matches_checksums_json": not bad}


def run(cmd):
    cmd = [sys.executable, "-I", *map(str, cmd)]
    log(" ".join(cmd))
    subprocess.run(cmd, check=True)


# ---- 5. publish ----------------------------------------------------------------------------------------------------
def parallel_copy(src, dst, streams=8, chunk=64 << 20):
    """Copy src to dst with `streams` threads, each pread/pwrite-ing its own contiguous byte range (a striped lustre
    file takes the writes on several OSTs at once)."""
    n = os.path.getsize(src)
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, n)
        step = max(-(-n // max(streams, 1)), 1)
        step = -(-step // (1 << 20)) * (1 << 20)               # whole 1 MiB stripes per writer

        def part(lo):
            hi = min(lo + step, n)
            fi = os.open(src, os.O_RDONLY)
            try:
                off = lo
                while off < hi:
                    buf = os.pread(fi, min(chunk, hi - off), off)
                    if not buf:
                        raise OSError(f"short read of {src} at {off}")
                    mv, done = memoryview(buf), 0
                    while done < len(mv):
                        done += os.pwrite(fd, mv[done:], off + done)
                    off += len(buf)
            finally:
                os.close(fi)
        with ThreadPoolExecutor(max(streams, 1)) as ex:
            list(ex.map(part, range(0, n, step)))
        os.fsync(fd)
    finally:
        os.close(fd)


def is_default_build(man, n_listing):
    """The canonical corpus: every file, the default targets, seed, threshold, revision and name (read from its
    manifest, i.e. from what the steps recorded, not from this invocation's flags)."""
    return (man["name"] == OUT_NAME and man["source"]["files_used"] == n_listing
            and man["filters"]["edu_min"] == EDU_MIN and man["sampling"]["seed"] == SEED
            and man["targets"] == {"train_tokens": int(TRAIN_TOKENS), "val_tokens": int(VAL_TOKENS)}
            and man["source"]["revision"] == REVISION)


def write_checksums(path, man):
    dump_json(path, {
        "about": ("Byte size and sha256 of the four stream files of the default FinePDFs-Edu English build "
                  "(data/pretrain_finepdfs_edu.md), written by its first verified build; later builds of the same "
                  "parameters are checked against it (data/build_finepdfs_edu.py). Kept apart from checksums.json."),
        "build": {"out_name": man["name"], "revision": man["source"]["revision"], "files": man["source"]["files_used"],
                  "edu_min": man["filters"]["edu_min"], "train_tokens": man["targets"]["train_tokens"],
                  "val_tokens": man["targets"]["val_tokens"], "seed": man["sampling"]["seed"],
                  "job_id": man["build"]["job_id"], "created": now()},
        "files": checksum_entries(man["name"], man["outputs"])})


def checksum_entries(out_name, outputs):
    return {f"{out_name}/{nm}": {"bytes": outputs[nm]["bytes"], "sha256": outputs[nm]["sha256"]} for nm in STREAMS}


def versions():
    import numpy
    import pyarrow
    import tokenizers
    import transformers
    import huggingface_hub
    v = {"python": sys.version.split()[0], "transformers": transformers.__version__,
         "tokenizers": tokenizers.__version__, "pyarrow": pyarrow.__version__, "numpy": numpy.__version__,
         "huggingface_hub": huggingface_hub.__version__}
    v["matches_pins"] = all(v[k] == p for k, p in PINNED.items())
    v["pins"] = PINNED
    return v


def build_manifest(a, listing, work, tok_info, out_dir_final):
    tsum = load_json(os.path.join(work, "tokenize_summary.json"))
    ssum = load_json(os.path.join(work, "select_summary.json"))
    wsum = load_json(os.path.join(work, "write_summary.json"))
    dsum = load_json(os.path.join(work, "download_summary.json"))
    steps = load_json(os.path.join(work, "steps.json"), {})
    P = tsum["params"]
    used = [listing["files"][i] for i in tsum["files"]]
    scripts = {nm: sha256(os.path.join(HERE, nm)) for nm in SCRIPTS if os.path.isfile(os.path.join(HERE, nm))}
    scripts["src/pretrain/data/doc_dataset.py"] = (sha256(os.path.join(HERE, "..", "src", "pretrain", "data", "doc_dataset.py"))
                                                   if os.path.isfile(os.path.join(HERE, "..", "src", "pretrain", "data", "doc_dataset.py")) else None)
    sb = None
    if a.sbatch and os.path.isfile(a.sbatch):
        sb = {"path": a.sbatch, "sha256": sha256(a.sbatch), "name": os.environ.get("DM_FP_SBATCH_NAME")}
    sel = ssum["selection"]
    tot = tsum["totals"]

    def with_l2(d):                          # cascade entries: the Llama-2 tokens are known for G4 only
        return {r: dict(v, share_of_docs=round(v["docs"] / max(tot["docs"], 1), 5),
                        share_of_token_count=round(v["token_count"] / max(tot["token_count"], 1), 5))
                for r, v in d.items()}
    return {
        "name": a.out_name,
        "about": ("A quality-filtered, English-only, document-level random sample of HuggingFaceFW/finepdfs-edu "
                  "(eng_Latn), tokenized with the Llama-2 32k tokenizer in the stage-1 format of delta-matching-train "
                  "(data/pretrain_finepdfs_edu.md; built by data/build_finepdfs_edu.py)."),
        "created": now(),
        "source": {
            "repo": REPO_ID, "repo_type": "dataset", "revision": a.revision, "pinned_revision": REVISION,
            "config": listing.get("config"), "split": listing.get("split"), "base_card": listing.get("base_card"),
            "license": listing.get("license"), "files_used": len(used), "files_in_listing": len(listing["files"]),
            "bytes": sum(e["bytes"] for e in used), "rows": sum(e.get("rows", 0) for e in used),
            "files": used, "download": dsum,
        },
        "filters": {
            "order": list(P["rules"]), "rules": P["rules"], "edu_min": P["edu_min"],
            "thresholds": {"L3_nonenglish_max": 0.10, "english_like_min_ascii_words": 20, "english_like_min_fw28": 0.08,
                           "G1_dup_line_chars_max": 0.50, "G2_ascii_letters_min": 0.30, "G3_fw_rate_min": 0.02,
                           "G4_min_content_tokens": 50},
            "totals": tot,
            "cascade": with_l2(tsum["cascade"]),
            "standalone": with_l2(tsum["standalone"]),
            "note": ("cascade: each dropped document counted under the first rule it fails, in this order; "
                     "stand-alone: under every rule it fails (G4 only for tokenized documents, so it is the cascade "
                     "figure). Llama-2 tokens (l2_tokens, with BOS/EOS) are known only from G4 on."),
            "pool_before_dedup": tsum["pool_before_dedup"],
        },
        "dedup": ssum["dedup"],
        "url_groups": ssum["url_groups"],
        "special_literal_docs": {"pool": ssum["pool"]["special_literal_docs"], "train": ssum["train"]["special_literal_docs"],
                                 "val": ssum["val"]["special_literal_docs"],
                                 "note": "documents containing a literal '<s>', '</s>' or '<unk>' (encoded as text)"},
        "sampling": {
            "seed": ssum["seed"], "namespace": "finepdfs-edu-en",
            "key": "BLAKE2b-64(b'finepdfs-edu-en|' + str(seed) + '|' + norm_url as UTF-8), big-endian uint64",
            "norm_url": ("url.strip() without http:// or https:// (any case), host (up to the first '/', '?' or '#') "
                         "lowercased, a leading 'www.' removed; path and query unchanged"),
            "order": "pool sorted by (key, norm_url, id, file, row); whole norm_url groups, val first, then train",
            "train_order": "source order (file, then row)", "val_order": "key order",
        },
        "targets": ssum["targets"],
        "pool": dict(ssum["pool"], headroom=ssum["headroom"]),
        "train": dict(ssum["train"], overshoot=sel["train_overshoot"], last_group_tokens=sel["train_last_group_tokens"],
                      groups=sel["train_groups"]),
        "val": dict(ssum["val"], overshoot=sel["val_overshoot"], last_group_tokens=sel["val_last_group_tokens"],
                    groups=sel["val_groups"]),
        "unused": dict(ssum["unused"], groups=sel["unused_groups"]),
        "pool_by_edu_threshold": ssum["pool_by_edu_threshold"],
        "val_coverage": ssum["val_coverage"],
        "tokenizer": dict(tok_info, dir=a.tokenizer, repo="meta-llama/Llama-2-7b-hf",
                          revision="01c7f73d771dfac7d292323805ebc428287df4f9", vocab=P["vocab"], bos=P["bos"],
                          eos=P["eos"], unk=P["unk"],
                          call="AutoTokenizer.from_pretrained(dir)(texts, add_special_tokens=False, "
                               "split_special_tokens=True)['input_ids']"),
        "format": {"dtype": "uint16", "byte_order": "little", "header": None,
                   "document": "[BOS] + ids + [EOS], documents back to back, the stream starts with BOS",
                   "doc_lengths": "np.save of a 1-D int64 array, len(ids) + 2 per document",
                   "loader": "src/pretrain/data/doc_dataset.py DocAwareMemmapDataset (data.paths, data.doc_lengths, "
                             "dtype uint16, position_mode packed)",
                   "invariants": wsum["invariants"]},
        "software": versions(),
        "build": {"scripts": scripts, "sbatch": sb, "job_id": os.environ.get("SLURM_JOB_ID"),
                  "job_name": os.environ.get("SLURM_JOB_NAME"), "node": socket.gethostname(),
                  "staging": os.path.abspath(work), "raw": os.path.abspath(a.raw), "steps": steps,
                  "args": {k: v for k, v in vars(a).items() if k not in ("step",)},
                  "tokenize": {"workers": tsum["workers"], "threads_per": tsum["threads_per"], "timing": tsum["timing"],
                               "tokenized": tsum["tokenized"], "extractor_pool_before_dedup": tsum["extractor"]},
                  "select_seconds": ssum["seconds"], "write_seconds": wsum["seconds"]},
        "outputs": wsum["outputs"],
        "published_to": out_dir_final,
    }


README = """# {name}

A quality-filtered, English-only, document-level random sample of the Hugging Face dataset
[HuggingFaceFW/finepdfs-edu](https://huggingface.co/datasets/HuggingFaceFW/finepdfs-edu) (config eng_Latn, revision
`{rev}`), tokenized with the Llama-2 32k tokenizer for delta-matching-train. Built by `data/build_finepdfs_edu.py`
(recipe: `data/pretrain_finepdfs_edu.md`); `manifest.json` holds the full provenance.

| file | content |
|---|---|
| `finepdfs_edu_en_train.bin` | training stream: {train_tok:,} uint16 tokens, {train_docs:,} documents, in source order |
| `finepdfs_edu_en_train.doc_lengths.npy` | int64 length of each training document |
| `finepdfs_edu_en_val.bin` | held-out stream: {val_tok:,} tokens, {val_docs:,} documents, in random (key) order |
| `finepdfs_edu_en_val.doc_lengths.npy` | int64 length of each held-out document |
| `selection.parquet` | one row per selected document: split, file, row, id, norm_url, key, n_tokens, edu |
| `manifest.json` | source files and revision, every rule and its drops, seed and key, counts, versions, sha256s |

Each document is `[BOS=1] + ids + [EOS=2]`; documents are back to back. Train and val share no document and no URL.
Train on it with `--data.paths <this dir>/finepdfs_edu_en_train.bin --data.doc_lengths
<this dir>/finepdfs_edu_en_train.doc_lengths.npy` (dtype uint16, position_mode packed: as the stage-1 configs).

Terms: the data is under the Open Data Commons Attribution License (ODC-By) v1.0 and its use is also subject to the
CommonCrawl Terms of Use; attribute HuggingFaceFW/finepdfs-edu (Kydlicek, Penedo, von Werra, 2025). It contains
copyrighted documents; the dataset card links an opt-out / PII removal form. The tokenizer is under the Llama 2
Community License. Build the data yourself; do not re-publish it.
"""


def publish(a, listing, work, tok_info):
    t0 = time.time()
    out = os.path.join(work, "out")
    wsum = load_json(os.path.join(work, "write_summary.json"))
    if wsum is None or not all(os.path.isfile(os.path.join(out, nm)) for nm in PUBLISHED):
        sys.exit(f"[finepdfs] nothing to publish in {out}: run the write step first")
    final = os.path.join(a.root, a.out_name)
    tag = os.environ.get("SLURM_JOB_ID") or f"pid{os.getpid()}"
    if os.path.exists(final) and not a.force:
        sys.exit(f"[finepdfs] {final} exists: refusing to replace it (--force moves it aside)")
    man = build_manifest(a, listing, work, tok_info, final)
    default = is_default_build(man, len(listing["files"]))
    os.makedirs(a.root, exist_ok=True)
    others = [x for x in os.listdir(a.root) if x.startswith(f".{a.out_name}.tmp-") and x != f".{a.out_name}.tmp-{tag}"]
    if others:
        log(f"note: leftovers of earlier publish attempts under {a.root}: {', '.join(sorted(others))}")
    cks = load_json(a.checksums) if default else None
    if cks is not None:                       # a rebuild of the default corpus must reproduce the first one
        want = cks["files"]
        got = checksum_entries(a.out_name, wsum["outputs"])
        if got != {k: want.get(k) for k in got}:
            sys.exit(f"[finepdfs] FAILED: the outputs differ from {a.checksums} (the first default build): "
                     + "; ".join(f"{k}: {got[k]['sha256'][:12]} != {(want.get(k) or {}).get('sha256', 'missing')[:12]}"
                                 for k in got if got[k] != want.get(k)))
    tmp = os.path.join(a.root, f".{a.out_name}.tmp-{tag}")
    os.makedirs(tmp, exist_ok=True)
    for nm in PUBLISHED:
        t = time.time()
        src, dst = os.path.join(out, nm), os.path.join(tmp, nm)
        parallel_copy(src, dst, streams=a.copy_streams if os.path.getsize(src) > (1 << 30) else 1)
        dt = time.time() - t
        log(f"copied {nm} ({os.path.getsize(src) / 1e9:.2f} GB) to {tmp} in {dt:.0f}s "
            f"({os.path.getsize(src) / 1e6 / max(dt, 1e-3):.0f} MB/s)")
    t_copy = time.time()
    with ThreadPoolExecutor(len(PUBLISHED)) as ex:
        hashes = {nm: ex.submit(sha256, os.path.join(tmp, nm)) for nm in PUBLISHED}
        for nm, h in hashes.items():
            want = wsum["outputs"][nm]
            if os.path.getsize(os.path.join(tmp, nm)) != want["bytes"] or h.result() != want["sha256"]:
                sys.exit(f"[finepdfs] FAILED: {tmp}/{nm} differs from the staged copy after the copy")
    t_hash = time.time()
    log(f"re-checked bytes and sha256 of {len(PUBLISHED)} files in {tmp} ({t_hash - t_copy:.0f}s)")
    record_step(work, "publish", t0, time.time(), copy_s=round(t_copy - t0, 1), recheck_s=round(t_hash - t_copy, 1))
    with open(os.path.join(tmp, "README.md"), "w") as f:
        f.write(README.format(name=a.out_name, rev=a.revision, train_tok=man["train"]["tokens"],
                              train_docs=man["train"]["docs"], val_tok=man["val"]["tokens"], val_docs=man["val"]["docs"]))
    if os.path.exists(final):
        aside = os.path.join(a.root, f".{a.out_name}.replaced-{tag}-{time.strftime('%Y%m%d%H%M%S')}")
        os.rename(final, aside)
        log(f"--force: moved the existing {final} aside to {aside} (delete it when nothing reads it)")
    os.rename(tmp, final)
    man["build"]["steps"] = load_json(os.path.join(work, "steps.json"), {})
    dump_json(os.path.join(final, "manifest.json"), man)    # last: a published corpus has its manifest
    log(f"published {final} (manifest.json written last)")
    if default and cks is not None:
        log(f"the outputs match {a.checksums}")
    elif default:
        log(f"{a.checksums} does not exist yet: the verify step writes it once this build passes")
    else:
        log("not the default build: data/finepdfs_edu_checksums.json neither written nor checked")


# ---- 6. verify -----------------------------------------------------------------------------------------------------
def loader_check(binp, lenp, vocab, bos, seq_len, seed=SEED):
    """Windows read through the training loader: DocAwareMemmapDataset directly and build_train_dataloader (world size 1,
    CPU, no worker processes)."""
    src = os.path.join(HERE, "..", "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    import numpy as np
    from pretrain.data.doc_dataset import DocAwareMemmapDataset
    from pretrain.data.data_loader import build_train_dataloader
    ds = DocAwareMemmapDataset(binp, seq_len, lenp, "uint16", "packed")
    picks = sorted({0, 1, len(ds) // 2, len(ds) - 1} & set(range(len(ds))))
    for i in picks:
        it = ds[i]
        ids, pos, cu = it["input_ids"].numpy(), it["position_ids"].numpy(), it["cu_seqlens"].numpy()
        assert ids.shape == (seq_len,) and int(ids.max()) < vocab and int(ids.min()) >= 0, i
        assert (pos == np.arange(seq_len)).all() and cu[0] == 0 and cu[-1] == seq_len and (np.diff(cu) > 0).all(), i
        for c in cu[1:-1]:
            assert ids[c] == bos, f"window {i}: a document boundary at {c} without BOS"
    assert int(ds[0]["input_ids"][0]) == bos
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    dl = build_train_dataloader({"paths": binp, "doc_lengths": lenp, "dtype": "uint16", "position_mode": "packed",
                                 "seed": seed, "num_workers": 0, "pin_memory": False},
                                max_seq_len=seq_len, global_batch_size=2)
    it = iter(dl)
    batches = []
    for _ in range(3):
        b = next(it)
        assert tuple(b["input_ids"].shape) == (2, seq_len) and int(b["input_ids"].max()) < vocab
        batches.append([int(x) for x in b["index"]])
    return {"windows": len(ds), "seq_len": seq_len, "checked_windows": picks, "loader_batches": batches}


def verify(a, listing, files):
    t0 = time.time()
    import numpy as np
    sys.path.insert(0, HERE)
    from finepdfs_select import check_stream
    d = os.path.join(a.root, a.out_name)
    man = load_json(os.path.join(d, "manifest.json"))
    if man is None:
        sys.exit(f"[finepdfs] {d}/manifest.json missing: not a published corpus")
    missing = [nm for nm in PUBLISHED if not os.path.isfile(os.path.join(d, nm))]
    if missing:
        sys.exit(f"[finepdfs] FAILED: {d} lacks {', '.join(missing)}")
    with ThreadPoolExecutor(len(PUBLISHED)) as ex:          # each file hashed once, compared with both references
        got = {nm: ex.submit(lambda p: {"bytes": os.path.getsize(p), "sha256": sha256(p)}, os.path.join(d, nm))
               for nm in PUBLISHED}
        got = {nm: f.result() for nm, f in got.items()}
    refs = [("manifest.json", {nm: man["outputs"][nm] for nm in PUBLISHED})]
    cks = load_json(a.checksums)
    if cks is not None and any(k.startswith(a.out_name + "/") for k in cks["files"]):
        refs.append((a.checksums, {k.split("/", 1)[1]: v for k, v in cks["files"].items()
                                   if k.startswith(a.out_name + "/")}))
    bad = [f"{nm} vs {src}" for src, ref in refs for nm, v in ref.items()
           if {"bytes": v["bytes"], "sha256": v["sha256"]} != got[nm]]
    if bad:
        sys.exit(f"[finepdfs] FAILED: bytes or sha256 differ: {'; '.join(bad)}")
    log(f"{len(PUBLISHED)} files match " + " and ".join(src for src, _ in refs))
    tk = man["tokenizer"]
    inv = {}
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(max(1, a.workers or 8)) as pool:
        for split in ("train", "val"):
            inv[split] = check_stream(os.path.join(d, f"finepdfs_edu_en_{split}.bin"),
                                      np.load(os.path.join(d, f"finepdfs_edu_en_{split}.doc_lengths.npy")),
                                      tk["vocab"], tk["bos"], tk["eos"], tk["unk"], pool=pool)
    lc = {split: loader_check(os.path.join(d, f"finepdfs_edu_en_{split}.bin"),
                              os.path.join(d, f"finepdfs_edu_en_{split}.doc_lengths.npy"), tk["vocab"], tk["bos"],
                              a.seq_len) for split in ("train", "val")}
    record_step(a.work, "verify", t0, time.time())
    if is_default_build(man, len(listing["files"])) and not any(k.startswith(a.out_name + "/") for k in (cks or {}).get("files", {})):
        write_checksums(a.checksums, man)
        log(f"wrote {a.checksums}: the reference of later default builds")
    log(f"verify PASS: {d}: train {inv['train']['tokens']:,} tokens / {inv['train']['docs']:,} documents, val "
        f"{inv['val']['tokens']:,} tokens / {inv['val']['docs']:,} documents; sha256, invariants and the loader "
        f"({lc['train']['windows']:,} train windows of {a.seq_len}) OK ({time.time() - t0:.0f}s)")
    if a.delete_raw:
        for fi in files:
            p = os.path.join(a.raw, listing["files"][fi]["path"])
            for q in (p, _marker(p)):
                if os.path.exists(q):
                    os.remove(q)
        shutil.rmtree(os.path.join(a.raw, ".cache"), ignore_errors=True)
        log(f"--delete-raw: removed the downloaded parquet under {a.raw}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the data root (DM_DATA_ROOT): the corpus goes to <root>/<out-name>")
    ap.add_argument("--raw", default=None, help="where the parquet goes (default <root>/raw/finepdfs-edu)")
    ap.add_argument("--work", default=None, help="shards, summaries and the staged outputs (default "
                                                 "<root>/raw/finepdfs-edu-work)")
    ap.add_argument("--step", choices=STEPS, action="append", help="run only this step (repeatable; default all)")
    ap.add_argument("--files", default=None, help="indices into data/finepdfs_edu_files.json, e.g. '0-5,23' "
                                                  "(default: all 100; a subset is never the default build)")
    ap.add_argument("--files-json", default=os.path.join(HERE, "finepdfs_edu_files.json"))
    ap.add_argument("--tokenizer", default=None, help=f"the Llama-2 tokenizer (default <root>/{TOKENIZER_SUB})")
    ap.add_argument("--allow-other-tokenizer", action="store_true", help="skip the tokenizer checksum (tests only)")
    ap.add_argument("--workers", type=int, default=None, help="tokenize processes (default max(1, cpus // threads-per));"
                                                             " also the write / verify processes (at most 16)")
    ap.add_argument("--threads-per", type=int, default=4, help="tokenizer threads per process")
    ap.add_argument("--edu-min", type=float, default=EDU_MIN, help="the edu threshold (1.55 and 1.6 are stricter)")
    ap.add_argument("--edu-thresholds", default="1.4,1.45,1.5,1.55,1.6,1.65",
                    help="edu thresholds at which the manifest reports the exact pool size")
    ap.add_argument("--train-tokens", type=float, default=TRAIN_TOKENS)
    ap.add_argument("--val-tokens", type=float, default=VAL_TOKENS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--revision", default=REVISION, help="the dataset revision to download (the files are checked "
                                                         "against the listing's sha256 whatever the revision)")
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--checksums", default=os.path.join(HERE, "finepdfs_edu_checksums.json"),
                    help="the reference sha256s of the default build (written by the first one)")
    ap.add_argument("--dl-concurrency", type=int, default=4)
    ap.add_argument("--copy-streams", type=int, default=8, help="parallel writers per big file when publishing")
    ap.add_argument("--seq-len", type=int, default=8192, help="window length of the loader check")
    ap.add_argument("--sbatch", default=None, help="the job script, hashed into the manifest")
    ap.add_argument("--delete-raw", action="store_true", help="delete the downloaded parquet after a verified build")
    ap.add_argument("--force", action="store_true", help="replace an existing <root>/<out-name> (moved aside)")
    a = ap.parse_args(argv)
    a.raw = a.raw or os.path.join(a.root, "raw", "finepdfs-edu")
    a.work = a.work or os.path.join(a.root, "raw", "finepdfs-edu-work")
    a.tokenizer = a.tokenizer or os.path.join(a.root, TOKENIZER_SUB)
    steps = a.step or list(STEPS)
    sys.path.insert(0, HERE)
    from finepdfs_tokenize import parse_files
    with open(a.files_json) as f:
        listing = json.load(f)
    files = parse_files(a.files, len(listing["files"]))
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 8)
    workers = a.workers or max(1, cpus // max(a.threads_per, 1))
    os.makedirs(a.work, exist_ok=True)
    log(f"steps {','.join(steps)}: {len(files)} files, root {a.root}, raw {a.raw}, work {a.work}, out {a.out_name}; "
        f"targets train {a.train_tokens:.4g} val {a.val_tokens:.4g}, edu >= {a.edu_min}, seed {a.seed}; "
        f"{cpus} cpus, {workers} workers x {a.threads_per} threads")
    tok_info = None
    if "download" in steps:
        download(listing, files, a.raw, a.revision, a.dl_concurrency, a.work)
    if any(s in steps for s in ("tokenize", "publish")):
        tok_info = check_tokenizer(a.tokenizer, a.allow_other_tokenizer)
    if "tokenize" in steps:
        t0 = time.time()
        run([os.path.join(HERE, "finepdfs_tokenize.py"), "--raw", a.raw, "--files-json", a.files_json, "--files",
             ",".join(map(str, files)), "--tokenizer", a.tokenizer, "--work", a.work, "--workers", workers,
             "--threads-per", a.threads_per, "--edu-min", a.edu_min, "--edu-thresholds", a.edu_thresholds,
             "--seed", a.seed])
        record_step(a.work, "tokenize", t0, time.time())
    if "select" in steps:
        t0 = time.time()
        run([os.path.join(HERE, "finepdfs_select.py"), "--work", a.work, "--out", os.path.join(a.work, "out"),
             "--step", "select", "--train-tokens", a.train_tokens, "--val-tokens", a.val_tokens, "--seed", a.seed])
        record_step(a.work, "select", t0, time.time())
    if "write" in steps:
        t0 = time.time()
        run([os.path.join(HERE, "finepdfs_select.py"), "--work", a.work, "--out", os.path.join(a.work, "out"),
             "--step", "write", "--workers", min(workers, 16)])
        record_step(a.work, "write", t0, time.time())
    if "publish" in steps:
        publish(a, listing, a.work, tok_info)
    if "verify" in steps:
        a.workers = min(workers, 16)
        verify(a, listing, files)
    log("DONE " + ",".join(steps))


if __name__ == "__main__":
    main()
