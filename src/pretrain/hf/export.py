"""Export a trained `src/pretrain` model to a HuggingFace checkpoint directory
(config.json + model.safetensors [+ tokenizer]) loadable by AutoModelForCausalLM:
the format the evaluation engine (src/evaluation) consumes.

The export is atomic file by file, with config.json LAST, so "config.json is present" means "the export is whole"
(scripts/campaign/eval_arm*.sh test config.json; `python -m pretrain.hf.check_export <dir>` checks the weights too):
  1. an existing config.json is unlinked first (a re-export into a finished export reads as incomplete until it is done);
  2. model.safetensors is written under a temporary name in a staging directory inside the export dir (same file
     system), fsynced, then os.replace'd into place;
  3. the tokenizer is saved into the staging directory, and each of its files fsynced and os.replace'd into place;
  4. config.json, written by DMLMConfig.save_pretrained into the staging directory, is fsynced and os.replace'd LAST.
A failure at any step leaves no config.json (and removes the staging directory). Only these files are touched: the
other contents of an export dir (TRAIN_COMPLETE, valce.txt, config.train.json, wandb_run_id.txt, stage 2's step<N>/)
stay as they are.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import torch

from ..modeling.config import ModelConfig
from .configuration_dmlm import DMLMConfig

STAGE_PREFIX = ".export-tmp."


def _fsync(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: str) -> None:
    """Best effort: the directory entries of the files just renamed into it."""
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _move_tree_into(src: str, dst: str, skip: tuple = ()) -> None:
    """fsync every file under `src` and os.replace it to the same relative path under `dst` (a directory is created
    only for files that go into it); names in `skip` (top level) are left behind."""
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target_dir = dst if rel == "." else os.path.join(dst, rel)
        for name in sorted(files):
            if rel == "." and name in skip:
                continue
            p = os.path.join(root, name)
            _fsync(p)
            os.makedirs(target_dir, exist_ok=True)
            os.replace(p, os.path.join(target_dir, name))


def _remove_stale_stages(out_dir: str) -> None:
    """Remove the staging directories of an export that was killed before its cleanup (best effort)."""
    try:
        names = os.listdir(out_dir)
    except OSError:
        return
    for name in names:
        if name.startswith(STAGE_PREFIX):
            shutil.rmtree(os.path.join(out_dir, name), ignore_errors=True)


def save_hf_checkpoint(state_dict: dict, model_cfg: ModelConfig, out_dir: str,
                       tokenizer=None, dtype: torch.dtype = torch.bfloat16) -> str:
    """Write a HF checkpoint (atomically, config.json last: module docstring). `state_dict` is the consolidated
    (unsharded) model state dict with `src/pretrain` naming (model.* / lm_head.weight); names match
    `DMLMForCausalLM`. Tied `lm_head.weight` is dropped (re-tied on load) so
    safetensors doesn't choke on shared storage."""
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    cfg_path = os.path.join(out_dir, "config.json")
    try:
        os.unlink(cfg_path)          # from here until the new config.json is in place the export reads as incomplete
    except FileNotFoundError:
        pass
    _remove_stale_stages(out_dir)
    stage = tempfile.mkdtemp(prefix=STAGE_PREFIX, dir=out_dir)
    try:
        cfg = DMLMConfig.from_model_config(model_cfg)
        cfg.save_pretrained(stage)   # config.json with model_type=dmlm + tie_word_embeddings (moved in LAST, below)

        sd = {k: v for k, v in state_dict.items()}
        if model_cfg.tie_embeddings:
            sd.pop("lm_head.weight", None)  # tied -> re-created on load (_tied_weights_keys)
        # clone+contiguous in the target dtype; clone severs any shared storage safetensors rejects
        sd = {k: v.detach().to(dtype).contiguous().clone() for k, v in sd.items()}
        tmp_model = os.path.join(stage, "model.safetensors")
        save_file(sd, tmp_model, metadata={"format": "pt"})
        _fsync(tmp_model)
        os.replace(tmp_model, os.path.join(out_dir, "model.safetensors"))

        if tokenizer is not None:
            tok_stage = os.path.join(stage, "tokenizer")
            os.makedirs(tok_stage)
            tokenizer.save_pretrained(tok_stage)
            _move_tree_into(tok_stage, out_dir, skip=("config.json",))   # the model's config.json comes last, below
            shutil.rmtree(tok_stage, ignore_errors=True)

        # whatever else save_pretrained wrote (nothing today), then config.json: the last file of a whole export
        _move_tree_into(stage, out_dir, skip=("config.json",))
        _fsync(os.path.join(stage, "config.json"))
        _fsync_dir(out_dir)
        os.replace(os.path.join(stage, "config.json"), cfg_path)
        _fsync_dir(out_dir)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return out_dir
