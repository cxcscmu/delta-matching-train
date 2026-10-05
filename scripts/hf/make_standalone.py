#!/usr/bin/env python3
"""make_standalone.py: an HF export -> a standalone HF model dir (trust_remote_code), optionally uploaded.

    python scripts/hf/make_standalone.py <export dir> --out <dir> [--card README.md] [--upload <user>/<repo>]

Takes the export a run writes to $DM_WORKSPACE/qwen35_m/hf/<run>/ (or one made by export_checkpoint.py) and
builds a directory that stock `transformers` loads with trust_remote_code, without this repository:

  * model.safetensors (hard-linked when possible, else copied), tokenizer files and config.train.json;
  * the model code from scripts/hf/remote_code (modeling_dmlm.py, configuration_dmlm.py);
  * config.json plus `architectures` and `auto_map` pointing at that code, and the tokenizer's `bos_token_id` /
    `eos_token_id` (so `generate()` stops at the end-of-sequence token), without the training-only keys
    (TRAIN_ONLY_KEYS below; config.train.json keeps them);
  * README.md: --card, or a short generated card (architecture, parameter count, usage) to extend.

It supports exports of this repository only: a config.json (or config.train.json) whose model_type is not `dmlm`, or
that names keys or mixers this repository does not define, is refused. The pipeline's own files (TRAIN_COMPLETE,
valce.txt) are left out. The shipped code covers the GQA, MLA, Mamba2, GatedDeltaNet and KDA mixers; an export using
any other mixer is refused until the code is extended (HF_CONVERSION.md, "Other mixers").

--upload pushes the directory to a model repo in one commit (the repo is created PRIVATE if it does not exist) and
checks every file on the Hub against the local copy (SHA-256 for LFS files, the git blob hash otherwise). Files
already in the repo that this build does not contain (a previous upload's) are reported and fail the check;
--delete-stale deletes them in the same commit. No GPU and no import of this repository needed; verify the result
with verify_standalone.py before sharing it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = ("modeling_dmlm.py", "configuration_dmlm.py")
SUPPORTED_MIXERS = {"gqa", "mla", "mamba2", "gated_deltanet", "kda"}    # MIXER_REGISTRY in remote_code/modeling_dmlm.py
MIXER_PACKAGES = {"mamba2": "mamba-ssm, causal-conv1d", "gated_deltanet": "flash-linear-attention",
                  "kda": "flash-linear-attention"}
TESTED_VERSIONS = {"torch": "2.12", "transformers": "5.9.0", "mamba-ssm": "2.3.2", "causal-conv1d": "1.6.2",
                   "flash-linear-attention": "0.5.2"}   # the versions the conversion was verified with
AUTO_MAP = {"AutoConfig": "configuration_dmlm.DMLMConfig", "AutoModelForCausalLM": "modeling_dmlm.DMLMForCausalLM"}
CARRY = ("model.safetensors", "config.train.json", "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
         "special_tokens_map.json", "added_tokens.json")
TRAIN_ONLY_KEYS = ("dm_mode", "dm_fused_producers")   # the fp8 training arms' knobs: not read by the shipped code
TRAIN_MIXERS = SUPPORTED_MIXERS | {"dm", "mla_dm"}    # config.train.json may also name the fp8 training mixers
# keys a config may carry besides DMLMConfig's parameters and transformers' standard keys: the HF aliases DMLMConfig
# sets, the rope parameters transformers derives, and what this script adds
EXTRA_CONFIG_KEYS = {"hidden_size", "max_position_embeddings", "num_attention_heads", "num_hidden_layers",
                     "tie_word_embeddings", "rope_parameters", "auto_map", "bos_token_id", "eos_token_id",
                     "pad_token_id", "torch_dtype"}
KEEP_REMOTE = {".gitattributes"}                       # created by the Hub with every repo; never stale
NOT_OURS = "this export was not written by this repository; convert its config to model_type dmlm first"


def safetensors_params(path: str) -> int:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return sum(math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__")


def config_keys() -> set:
    """The keys this repository's configs may carry: DMLMConfig's parameters (read from the shipped code, not imported),
    TRAIN_ONLY_KEYS, transformers' standard keys and EXTRA_CONFIG_KEYS."""
    import ast
    from transformers import PretrainedConfig
    tree = ast.parse(open(os.path.join(HERE, "remote_code", "configuration_dmlm.py")).read())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DMLMConfig")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    return ({a.arg for a in init.args.args[1:]} | set(TRAIN_ONLY_KEYS) | set(PretrainedConfig().to_dict())
            | EXTRA_CONFIG_KEYS)


def check_config(path: str, cfg: dict, known: set) -> None:
    """Refuse a config this repository did not write: another model_type, or keys / mixers it does not define."""
    if cfg.get("model_type") != "dmlm":
        sys.exit(f"{path} has model_type {cfg.get('model_type')!r}: {NOT_OURS}")
    keys = sorted(set(cfg) - known)
    mixers = sorted((set(cfg.get("layer_mixers") or []) | {cfg.get("mixer") or "gqa"}) - TRAIN_MIXERS)
    parts = ([f"keys {keys}"] if keys else []) + ([f"mixers {mixers}"] if mixers else [])
    if parts:
        sys.exit(f"{path} names {' and '.join(parts)} that this repository does not define (remote_code/"
                 "configuration_dmlm.py, the training mixers): convert the export first if this repository did not "
                 "write it, or extend remote_code if its config gained a field")


def special_token_ids(export: str) -> dict:
    """bos_token_id / eos_token_id of the export's tokenizer, as AutoTokenizer reads them ({} without a tokenizer)."""
    if not any(os.path.exists(os.path.join(export, fn)) for fn in ("tokenizer.json", "tokenizer.model")):
        return {}
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging
    level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()   # quiet: transformers does not know the export's model_type (dmlm) here
    try:
        tok = AutoTokenizer.from_pretrained(export, local_files_only=True)
    except Exception as e:
        sys.exit(f"{export}: its tokenizer does not load with AutoTokenizer ({type(e).__name__}: {e})")
    finally:
        hf_logging.set_verbosity(level)
    ids = {}
    for k in ("bos_token_id", "eos_token_id"):
        if getattr(tok, k, None) is not None:
            ids[k] = int(getattr(tok, k))
    return ids


def default_card(name: str, c: dict, n_params: int, repo: str) -> str:
    pattern = c.get("layer_mixers") or [c.get("mixer", "gqa")]
    extra = list(dict.fromkeys(p for m in pattern if m in MIXER_PACKAGES for p in MIXER_PACKAGES[m].split(", ")))
    pkgs = ", ".join(["torch", "transformers >= 5"] + extra)
    tested = ", ".join(f"{p} {TESTED_VERSIONS[p]}" for p in ["torch", "transformers"] + extra if p in TESTED_VERSIONS)
    return f"""---
library_name: transformers
pipeline_tag: text-generation
---

# {name}

A {n_params / 1e9:.2f} B-parameter DMLM decoder ({c['n_layers']} layers, repeating pattern {pattern}, d_model
{c['d_model']}, {c['vocab_size']:,}-token vocabulary, {c['max_seq_len']:,}-token context) exported from the
delta-matching-train pipeline. The model code ships with the repo, so it loads with stock `transformers` (CUDA GPU;
{pkgs}; tested with {tested}):

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{repo}"
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
print(tokenizer.decode(model.generate(**inputs, max_new_tokens=32)[0], skip_special_tokens=True))
```

`generate()` keeps a cache (greedy decoding and sampling, `num_beams=1`; batched prompts must share one length) and
stops at the tokenizer's end-of-sequence token. `model(input_ids=ids, labels=ids).loss` is the next-token
cross-entropy for fine-tuning in bf16 (gradient checkpointing is not supported). `config.train.json` records the
training-time layer configuration.
"""


def build(export: str, out: str, card: str | None, repo: str | None, copy: bool) -> list[str]:
    c = json.load(open(os.path.join(export, "config.json")))
    known = config_keys()
    check_config(os.path.join(export, "config.json"), c, known)
    train_cfg = os.path.join(export, "config.train.json")
    if os.path.exists(train_cfg):
        check_config(train_cfg, json.load(open(train_cfg)), known)
    mixers = set(c.get("layer_mixers") or []) | {c.get("mixer") or "gqa"}
    if mixers - SUPPORTED_MIXERS:
        sys.exit(f"mixers {sorted(mixers - SUPPORTED_MIXERS)} are not in the shipped model code "
                 f"(scripts/hf/remote_code supports {sorted(SUPPORTED_MIXERS)}); extend it first "
                 "(HF_CONVERSION.md, 'Other mixers')")
    if c.get("attn_impl") != "sdpa":
        sys.exit("config.json is not the clean-kernel eval config (attn_impl != 'sdpa'): the export was not "
                 "patched; re-export with export_checkpoint.py (it applies train_arm.sh's patch)")
    if not os.path.exists(os.path.join(export, "model.safetensors")):
        sys.exit(f"{export}/model.safetensors missing")
    if os.path.isdir(out) and os.listdir(out):
        sys.exit(f"{out} exists and is not empty")
    tok_ids = special_token_ids(export)
    os.makedirs(out, exist_ok=True)

    dropped = [k for k in TRAIN_ONLY_KEYS if k in c]
    for k in dropped:
        del c[k]
    if dropped:
        print(f"config.json: dropped the training-only keys {dropped}"
              + (" (config.train.json keeps them)" if os.path.exists(train_cfg) else ""))
    if tok_ids:
        c.update(tok_ids)
    else:
        print("config.json: no tokenizer in the export, so no bos / eos ids (generate() will not stop at an eos)")
    c["architectures"] = ["DMLMForCausalLM"]
    c["auto_map"] = AUTO_MAP
    with open(os.path.join(out, "config.json"), "w") as f:
        f.write(json.dumps(c, indent=2) + "\n")
    written = ["config.json"]
    for fn in CARRY:
        src = os.path.join(export, fn)
        if not os.path.exists(src):
            continue
        dst = os.path.join(out, fn)
        if copy:
            shutil.copy2(src, dst)
        else:
            try:
                os.link(os.path.realpath(src), dst)
            except OSError:
                shutil.copy2(src, dst)
        written.append(fn)
    for fn in CODE:
        shutil.copy2(os.path.join(HERE, "remote_code", fn), os.path.join(out, fn))
        written.append(fn)
    name = (repo or os.path.basename(os.path.normpath(export))).split("/")[-1]
    text = open(card).read() if card else default_card(name, c, safetensors_params(os.path.join(out, "model.safetensors")),
                                                        repo or f"<user>/{name}")
    with open(os.path.join(out, "README.md"), "w") as f:
        f.write(text)
    written.append("README.md")
    return written


def upload(out: str, files: list[str], repo: str, message: str, delete_stale: bool = False) -> None:
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    api = HfApi()
    if not api.repo_exists(repo, repo_type="model"):
        api.create_repo(repo, repo_type="model", private=True)
        print(f"created {repo} (private)")
    ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=os.path.join(out, f)) for f in files]
    if delete_stale:
        stale = sorted({s.rfilename for s in (api.model_info(repo).siblings or [])} - set(files) - KEEP_REMOTE)
        if stale:
            print(f"deleting {len(stale)} remote file(s) this build does not contain: {stale}")
            ops += [CommitOperationDelete(path_in_repo=f) for f in stale]
    commit = api.create_commit(repo_id=repo, repo_type="model", operations=ops, commit_message=message)
    info = api.model_info(repo, files_metadata=True)
    remote = {s.rfilename: s for s in (info.siblings or [])}
    bad = []
    for f in files:
        s, p = remote.get(f), os.path.join(out, f)
        if s is None:
            bad.append(f"{f} missing")
            continue
        lfs = s.lfs.get("sha256") if isinstance(s.lfs, dict) else getattr(s.lfs, "sha256", None)
        if lfs:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            ok = h.hexdigest() == lfs
        else:
            d = open(p, "rb").read()
            ok = hashlib.sha1(b"blob %d\0" % len(d) + d).hexdigest() == s.blob_id
        if not ok:
            bad.append(f"{f} differs")
    stale = sorted(set(remote) - set(files) - KEEP_REMOTE)
    if stale:
        bad.append(f"the repo also holds files this build does not contain: {stale} (--delete-stale deletes them)")
    print(f"uploaded to {repo}: {commit.commit_url} | private={info.private} | "
          + ("all files verified on the Hub" if not bad else f"PROBLEMS: {bad}"))
    if bad:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("export", help="HF export dir (config.json + model.safetensors + tokenizer)")
    ap.add_argument("--out", required=True, help="standalone dir to create (must not exist or be empty)")
    ap.add_argument("--card", default=None, help="model card (README.md) to ship; default: a short generated card")
    ap.add_argument("--upload", default=None, metavar="REPO_ID", help="push to this model repo, e.g. <user>/<repo>")
    ap.add_argument("--message", default=None, help="commit message for --upload")
    ap.add_argument("--delete-stale", action="store_true",
                    help="with --upload: delete files in the repo that this build does not contain (default: report "
                         "them and exit 1)")
    ap.add_argument("--copy", action="store_true", help="copy the weights instead of hard-linking them")
    a = ap.parse_args()
    files = build(a.export, a.out, a.card, a.upload, a.copy)
    print(f"built {a.out}: {', '.join(files)}")
    if a.upload:
        upload(a.out, files, a.upload,
               a.message or f"Add {os.path.basename(os.path.normpath(a.export))} (standalone, trust_remote_code)",
               delete_stale=a.delete_stale)


if __name__ == "__main__":
    main()
