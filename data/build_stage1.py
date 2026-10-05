#!/usr/bin/env python3
"""Build the stage-1 (pretraining) corpus from Nemotron-CC-v2.1 (data/pretrain_stage1.md):

    python data/build_stage1.py --root $DM_DATA_ROOT                     # download, tokenize, split, verify
    python data/build_stage1.py --root $DM_DATA_ROOT --step tokenize     # one step (download | tokenize | split | verify)

  1. download  the High-Quality (20 parquet parts, 46.8 GB) and High-Quality-DQA (6 parts, 11.7 GB) buckets of the gated
               Hugging Face dataset nvidia/Nemotron-CC-v2.1 at a pinned revision into --raw (accept the NVIDIA Data
               Agreement for Model Training on the dataset page and `hf auth login` first)
  2. tokenize  every document -> [BOS] + Llama-2 ids + [EOS], one uint16 stream in a fixed order (data/stage1_tokenize.py):
               <root>/nemotron_cc_v2d1_hq_dqa/nemotron_hqdqa.bin + .doc_lengths.npy (40.47 B tokens, 36.98 M documents)
  3. split     the 2 % document-level holdout, seed 11785 (data/stage1_split.py): nemotron_rs_train.bin (39.66 B tokens)
               and nemotron_rs_val.bin (807 M tokens, shuffled), each with its .doc_lengths.npy
  4. verify    sizes and sha256 of the four outputs against data/checksums.json

Needs the tokenizer first (data/fetch_tokenizer.py). Disk: 58.5 GB of parquet + 81 GB for the tokenized stream + 81 GB
for the split; the tokenized stream is deleted after a successful split unless --keep-stream, the parquet only with
--delete-raw. Time on a 208-CPU node (measured): 2 minutes for the download (network-bound), 14 minutes to tokenize (26
processes x 8 threads; the output does not depend on --workers), a minute each for the split and the check."""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify import check_files, load_checksums  # noqa: E402

REPO_ID = "nvidia/Nemotron-CC-v2.1"
REVISION = "ba6f2aaef7ada865bb08fc08640ca292150097db"
CONFIGS = ("High-Quality", "High-Quality-DQA")
N_PARTS = {"High-Quality": 20, "High-Quality-DQA": 6}
SUB = "nemotron_cc_v2d1_hq_dqa"
STREAM, SPLIT = "nemotron_hqdqa", "nemotron_rs"
VAL_FRAC, SEED = 0.02, 11785
STEPS = ("download", "tokenize", "split", "verify")


def download(raw):
    from huggingface_hub import snapshot_download
    snapshot_download(REPO_ID, repo_type="dataset", revision=REVISION, local_dir=raw,
                      allow_patterns=[f"{c}/*.parquet" for c in CONFIGS])
    for c in CONFIGS:
        n = len([f for f in os.listdir(os.path.join(raw, c)) if f.endswith(".parquet")])
        if n != N_PARTS[c]:
            sys.exit(f"[stage1] {raw}/{c} holds {n} parquet parts, expected {N_PARTS[c]}: remove the stray files")
    print(f"[stage1] downloaded {REPO_ID}@{REVISION[:12]} {'/'.join(CONFIGS)} -> {raw}")


def run(cmd):
    print("[stage1] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the data root (DM_DATA_ROOT)")
    ap.add_argument("--raw", default=None, help="where the parquet goes (default <root>/raw/nemotron-cc-v2.1)")
    ap.add_argument("--step", choices=STEPS, action="append", help="run only this step (repeatable; default all four)")
    ap.add_argument("--workers", type=int, default=None, help="tokenizer processes (default min(26, cpus // 8))")
    ap.add_argument("--keep-stream", action="store_true", help="keep the unsplit tokenized stream")
    ap.add_argument("--delete-raw", action="store_true", help="delete the parquet after a verified build")
    a = ap.parse_args()
    raw = a.raw or os.path.join(a.root, "raw", "nemotron-cc-v2.1")
    out = os.path.join(a.root, SUB)
    tok = os.path.join(a.root, "tokenizer", "llama2_32k")
    steps = a.step or list(STEPS)
    cpus = os.cpu_count() or 8
    workers = a.workers or max(1, min(26, cpus // 8))
    threads = max(1, cpus // workers)
    if "download" in steps:
        download(raw)
    if "tokenize" in steps:
        if not os.path.isfile(os.path.join(tok, "tokenizer.model")):
            sys.exit(f"[stage1] no tokenizer at {tok}: run data/fetch_tokenizer.py --root {a.root} first")
        os.makedirs(out, exist_ok=True)
        run([sys.executable, os.path.join(HERE, "stage1_tokenize.py"), "--raw-dir", raw, "--configs", *CONFIGS,
             "--tokenizer", tok, "--out", os.path.join(out, STREAM), "--workers", str(workers), "--threads-per", str(threads)])
    if "split" in steps:
        run([sys.executable, os.path.join(HERE, "stage1_split.py"), "--bin", os.path.join(out, STREAM + ".bin"),
             "--doc-lengths", os.path.join(out, STREAM + ".doc_lengths.npy"), "--out", os.path.join(out, SPLIT),
             "--val-frac", str(VAL_FRAC), "--seed", str(SEED)])
    if "verify" in steps:
        files = {k: v for k, v in load_checksums().items() if k.startswith(SUB + "/")}
        if not check_files(a.root, files, hashes=True):
            sys.exit("[stage1] FAILED: the outputs differ from data/checksums.json (see above)")
        if not a.keep_stream:
            for ext in (".bin", ".doc_lengths.npy"):
                p = os.path.join(out, STREAM + ext)
                if os.path.exists(p):
                    os.remove(p)
        if a.delete_raw and os.path.isdir(raw):
            import shutil
            shutil.rmtree(raw)
        print("[stage1] DONE: the stage-1 corpus matches data/checksums.json")


if __name__ == "__main__":
    main()
