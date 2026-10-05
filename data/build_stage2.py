#!/usr/bin/env python3
"""Build the stage-2 (context extension) corpus from ProLong's 512K release (data/pretrain_stage2.md):

    python data/build_stage2.py --root $DM_DATA_ROOT                   # download, repack, val, verify
    python data/build_stage2.py --root $DM_DATA_ROOT --step repack     # one step (download | repack | val | verify)

  1. download  the Hugging Face dataset princeton-nlp/prolong-data-512K at a pinned revision (12 directories of MDS
               shards, 122.4 GB of Llama-3 ids) into --raw, and Meta's Llama-3 tokenizer.model (the gated model
               repository meta-llama/Meta-Llama-3-8B: accept the Llama 3 Community License on its page and `hf auth
               login` first) into <root>/tokenizer/llama3
  2. repack    decode every source document, re-encode it with the Llama-2 tokenizer, pack 65,536-token windows and
               flatten them (data/stage2_repack.py): <root>/prolong_64k_llama2/prolong64k_train.bin (37.16 B tokens,
               566,949 windows) + .doc_lengths.npy
  3. val       the 256 held-out validation windows (data/stage2_val.py): prolong64k_val.bin + .doc_lengths.npy
  4. verify    sizes and sha256 of the four outputs against data/checksums.json

Needs the Llama-2 tokenizer first (data/fetch_tokenizer.py). Disk: 122.4 GB for the release + 74.3 GB for the output;
the release is deleted only with --delete-raw. Time on a 208-CPU node (measured): 17 minutes for the download
(network-bound), 18 minutes to repack (192 processes; the output does not depend on --workers), a minute for the rest."""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify import check_files, load_checksums, sha256  # noqa: E402

REPO_ID = "princeton-nlp/prolong-data-512K"
REVISION = "0bb2becdbb958cf29804da01ae615eb838170aa4"
L3_REPO, L3_REVISION, L3_FILE = "meta-llama/Meta-Llama-3-8B", "8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920", "original/tokenizer.model"
L3_SHA256 = "82e9d31979e92ab929cd544440f129d9ecd797b69e327f80f17e1c50d5551b55"
SUB = "prolong_64k_llama2"
TRAIN, VAL = "prolong64k_train", "prolong64k_val"
STEPS = ("download", "repack", "val", "verify")


def llama3_path(root):
    return os.path.join(root, "tokenizer", "llama3", "tokenizer.model")


def download(root, raw):
    import shutil
    from huggingface_hub import hf_hub_download, snapshot_download
    from stage2_repack import SOURCE_DIRS, units
    snapshot_download(REPO_ID, repo_type="dataset", revision=REVISION, local_dir=raw,
                      allow_patterns=[f"{d}/*" for d in SOURCE_DIRS])
    units(raw)                                           # exits unless every source directory is complete
    print(f"[stage2] downloaded {REPO_ID}@{REVISION[:12]} -> {raw}")
    dst = llama3_path(root)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(hf_hub_download(L3_REPO, L3_FILE, revision=L3_REVISION), dst)
    if sha256(dst) != L3_SHA256:
        sys.exit(f"[stage2] {dst}: sha256 differs from {L3_SHA256}")
    print(f"[stage2] {L3_REPO}@{L3_REVISION[:12]} {L3_FILE} -> {dst}")


def run(cmd):
    print("[stage2] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the data root (DM_DATA_ROOT)")
    ap.add_argument("--raw", default=None, help="where the release goes (default <root>/raw/prolong-data-512K)")
    ap.add_argument("--step", choices=STEPS, action="append", help="run only this step (repeatable; default all four)")
    ap.add_argument("--workers", type=int, default=None, help="repack processes (default: every CPU)")
    ap.add_argument("--delete-raw", action="store_true", help="delete the release after a verified build")
    a = ap.parse_args()
    raw = a.raw or os.path.join(a.root, "raw", "prolong-data-512K")
    out = os.path.join(a.root, SUB)
    l2 = os.path.join(a.root, "tokenizer", "llama2_32k", "tokenizer.model")
    steps = a.step or list(STEPS)
    if "download" in steps:
        download(a.root, raw)
    if "repack" in steps:
        if not os.path.isfile(l2):
            sys.exit(f"[stage2] no tokenizer at {l2}: run data/fetch_tokenizer.py --root {a.root} first")
        if not os.path.isfile(llama3_path(a.root)):
            sys.exit(f"[stage2] no Llama-3 tokenizer at {llama3_path(a.root)}: run the download step first")
        os.makedirs(out, exist_ok=True)
        run([sys.executable, os.path.join(HERE, "stage2_repack.py"), "--src", raw, "--llama3", llama3_path(a.root),
             "--llama2", l2, "--out", os.path.join(out, TRAIN), "--workers", str(a.workers or os.cpu_count() or 8)])
    if "val" in steps:
        run([sys.executable, os.path.join(HERE, "stage2_val.py"), "--train", os.path.join(out, TRAIN),
             "--out", os.path.join(out, VAL)])
    if "verify" in steps:
        files = {k: v for k, v in load_checksums().items() if k.startswith(SUB + "/")}
        if not check_files(a.root, files, hashes=True):
            sys.exit("[stage2] FAILED: the outputs differ from data/checksums.json (see above)")
        if a.delete_raw and os.path.isdir(raw):
            import shutil
            shutil.rmtree(raw)
        print("[stage2] DONE: the stage-2 corpus matches data/checksums.json")


if __name__ == "__main__":
    main()
