#!/usr/bin/env python3
"""Fetch the Llama-2 tokenizer (32,000 ids) into <root>/tokenizer/llama2_32k, the tokenizer every corpus of this study is
encoded with and every export carries.

    python data/fetch_tokenizer.py --root $DM_DATA_ROOT

The files come from the Hugging Face model repository meta-llama/Llama-2-7b-hf at a pinned revision; the repository is
gated: accept Meta's Llama 2 Community License on its page and log in (`hf auth login`) first. The four files are
checked against data/checksums.json. The tokenizer is under the Llama 2 Community License, not this repository's
license (data/README.md)."""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify import check_files, load_checksums  # noqa: E402

REPO_ID = "meta-llama/Llama-2-7b-hf"
REVISION = "01c7f73d771dfac7d292323805ebc428287df4f9"
FILES = ("tokenizer.model", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
SUBDIR = "tokenizer/llama2_32k"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the data root (DM_DATA_ROOT)")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download
    out = os.path.join(a.root, SUBDIR)
    os.makedirs(out, exist_ok=True)
    for f in FILES:
        p = hf_hub_download(REPO_ID, f, revision=REVISION)
        shutil.copyfile(p, os.path.join(out, f))
        print(f"[tokenizer] {f} -> {out}")
    ok = check_files(a.root, {k: v for k, v in load_checksums().items() if k.startswith(SUBDIR + "/")}, hashes=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
