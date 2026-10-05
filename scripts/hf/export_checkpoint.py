#!/usr/bin/env python3
"""export_checkpoint.py: a training checkpoint -> an HF export dir, exactly as a finished run writes one.

    PYTHONPATH=src python scripts/hf/export_checkpoint.py \\
        --ckpt $DM_WORKSPACE/qwen35_m/ckpts/<run>/checkpoint-<step> --out $DM_WORKSPACE/qwen35_m/hf/<run>_step<step>

A finished run already has its export at $DM_WORKSPACE/qwen35_m/hf/<run>/ (the trainer writes it after the last step
and scripts/campaign/train_arm.sh patches it). Use this for a run that stopped early, or for an intermediate step.

It reads the checkpoint's consolidated weights (pytorch_model.bin) and model_config.json, writes config.json +
model.safetensors (bf16, the tied lm_head dropped) + the tokenizer through the trainer's own save_hf_checkpoint, and
then applies train_arm.sh's clean-kernel patch: config.train.json keeps the training config, while config.json runs
attention through SDPA with the bf16 mixers (dm -> gqa, mla_dm -> mla), the form every evaluation and the
standalone conversion expect. CPU only; no GPU needed.

It supports checkpoints written by this repository only: a model_config.json with a field this repository's
ModelConfig does not define, or a mixer its registry does not know, is refused.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


def eval_patch(out_dir: str) -> None:
    """scripts/campaign/train_arm.sh's 'export patched -> clean-kernel eval' step, verbatim."""
    p = os.path.join(out_dir, "config.json")
    shutil.copy(p, os.path.join(out_dir, "config.train.json"))
    c = json.load(open(p))
    c["attn_impl"] = "sdpa"
    if c.get("mixer") == "dm":
        c["mixer"] = "gqa"
    if c.get("layer_mixers"):
        c["layer_mixers"] = [{"dm": "gqa", "mla_dm": "mla"}.get(m, m) for m in c["layer_mixers"]]
    if c.get("mixer") == "mla_dm":
        c["mixer"] = "mla"
    for k in ("dm_mode", "dm_fused_producers"):   # the fp8-only keys; config.train.json keeps them
        c.pop(k, None)
    json.dump(c, open(p, "w"), indent=2, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="a complete checkpoint dir: <ckpt_root>/<run>/checkpoint-<step>")
    ap.add_argument("--out", required=True, help="the export dir to create (must not exist or be empty)")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer dir (default $DM_TOKENIZER_DIR, else $DM_DATA_ROOT/tokenizer/llama2_32k)")
    ap.add_argument("--no-eval-patch", action="store_true",
                    help="keep the training mixers in config.json (the raw trainer export, without the patch)")
    a = ap.parse_args()

    import dataclasses

    import torch
    from transformers import AutoTokenizer
    try:
        from pretrain.hf.export import save_hf_checkpoint
        from pretrain.modeling.config import ModelConfig
        from pretrain.modeling.mixers import MIXER_REGISTRY
    except ImportError:
        sys.exit("needs this repository's source on the path: run with PYTHONPATH=src from the repo root")

    ck = a.ckpt.rstrip("/")
    if not os.path.exists(os.path.join(ck, ".complete")):
        sys.exit(f"{ck}: no .complete marker (a save that never finished); use a complete checkpoint")
    if os.path.isdir(a.out) and os.listdir(a.out):
        sys.exit(f"{a.out} exists and is not empty")
    tok_dir = a.tokenizer or os.environ.get("DM_TOKENIZER_DIR") or os.path.join(
        os.environ.get("DM_DATA_ROOT", "workspace/data"), "tokenizer", "llama2_32k")

    mcd = json.load(open(os.path.join(ck, "model_config.json")))
    mcd.pop("_resolved", None)
    unknown = sorted(set(mcd) - {f.name for f in dataclasses.fields(ModelConfig)})
    bad_mixers = sorted((set(mcd.get("layer_mixers") or []) | {mcd.get("mixer", "gqa")}) - set(MIXER_REGISTRY))
    parts = ([f"fields {unknown}"] if unknown else []) + ([f"mixers {bad_mixers}"] if bad_mixers else [])
    if parts:
        sys.exit(f"{ck}/model_config.json names {' and '.join(parts)} that this repository does not define: "
                 "this checkpoint was not written by this repository")
    mc = ModelConfig(**mcd)
    sd = torch.load(os.path.join(ck, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    tok = AutoTokenizer.from_pretrained(tok_dir, local_files_only=True)
    save_hf_checkpoint(sd, mc, a.out, tokenizer=tok)
    if not a.no_eval_patch:
        eval_patch(a.out)
    n = sum(v.numel() for k, v in sd.items() if not (mc.tie_embeddings and k == "lm_head.weight"))
    mixers = json.load(open(os.path.join(a.out, "config.json"))).get("layer_mixers")
    print(f"exported {ck} -> {a.out}: {len(sd)} tensors, {n:,} parameters, layer pattern {mixers}"
          + ("" if a.no_eval_patch else " (clean-kernel config; training config in config.train.json)"))


if __name__ == "__main__":
    main()
