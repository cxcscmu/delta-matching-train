#!/usr/bin/env python3
"""Check a data root against data/checksums.json (the byte size and sha256 of every file the builds produce).

    python data/verify.py --root $DM_DATA_ROOT                 # every part: sizes and sha256 (a few minutes: 165 GB)
    python data/verify.py --root $DM_DATA_ROOT --fast          # sizes only
    python data/verify.py --root $DM_DATA_ROOT --part stage1   # one part: tokenizer | stage1 | stage2 | ruler

A missing part is reported and fails the check; `--part` restricts the check to what has been built. The evaluation
datasets cache (<root>/hf_cache) has no checksums (its layout is the datasets library's); its presence is checked.
stdlib only."""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKSUMS = os.path.join(HERE, "checksums.json")
PARTS = {"tokenizer": "tokenizer/llama2_32k/", "stage1": "nemotron_cc_v2d1_hq_dqa/", "stage2": "prolong_64k_llama2/",
         "ruler": "ruler-500-llama2/"}


def load_checksums() -> dict:
    """{path relative to the data root: {"bytes": int, "sha256": str}}"""
    with open(CHECKSUMS) as f:
        return json.load(f)["files"]


def sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def check_files(root: str, files: dict, hashes: bool = True) -> bool:
    """Check every file of `files` (as load_checksums) under root; prints one line per problem and a summary."""
    bad = 0
    for rel, want in sorted(files.items()):
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            print(f"  MISSING {rel}")
            bad += 1
            continue
        n = os.path.getsize(p)
        if n != want["bytes"]:
            print(f"  SIZE    {rel}: {n} bytes, expected {want['bytes']}")
            bad += 1
            continue
        if hashes:
            h = sha256(p)
            if h != want["sha256"]:
                print(f"  SHA256  {rel}: {h}, expected {want['sha256']}")
                bad += 1
    what = "sizes and sha256" if hashes else "sizes"
    print(f"[verify] {len(files) - bad}/{len(files)} files OK ({what})" + ("" if not bad else f", {bad} FAILED"))
    return bad == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the data root (DM_DATA_ROOT)")
    ap.add_argument("--part", choices=list(PARTS), action="append", help="check only this part (repeatable)")
    ap.add_argument("--fast", action="store_true", help="sizes only, no sha256")
    a = ap.parse_args()
    files = load_checksums()
    ok = True
    for part in a.part or list(PARTS):
        sel = {k: v for k, v in files.items() if k.startswith(PARTS[part])}
        print(f"[verify] {part}: {len(sel)} files under {os.path.join(a.root, PARTS[part])}")
        ok &= check_files(a.root, sel, hashes=not a.fast)
    if not a.part:
        cache = os.path.join(a.root, "hf_cache", "datasets")
        if os.path.isdir(os.path.join(cache, "cais___mmlu")):
            print(f"[verify] evaluation datasets: {cache} present")
        else:
            print(f"[verify] evaluation datasets: {cache}/cais___mmlu MISSING (python data/prepare_eval.py --root {a.root})")
            ok = False
    print("[verify] " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
