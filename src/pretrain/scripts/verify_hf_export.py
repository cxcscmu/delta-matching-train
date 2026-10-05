#!/usr/bin/env python
"""Verify the HF export round-trips: build a src/pretrain model -> save_hf_checkpoint
-> AutoModelForCausalLM.from_pretrained -> logits match the original, for the bf16 mixers.
Runs on CPU for the GQA cases; the GatedDeltaNet / Mamba2 cases need CUDA (their fla /
causal_conv1d kernels) and are skipped without it. Checkpoints go to a temporary directory.

Run: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m pretrain.scripts.verify_hf_export
"""
import os
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM

from ..modeling.config import ModelConfig
from ..modeling.model import build_model
from ..hf import save_hf_checkpoint, register_auto

CASES = [   # (name, mixer, extra ModelConfig kwargs, needs CUDA)
    ("gqa",            "gqa",            {},                   False),
    ("gqa_qknorm",     "gqa",            dict(qk_norm=True),   False),  # regression: q_norm/k_norm MUST round-trip
    ("gated_deltanet", "gated_deltanet", {},                   True),
    ("mamba2",         "mamba2",         {},                   True),
]


def main() -> int:
    register_auto()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ok_all, n_skip = True, 0
    with tempfile.TemporaryDirectory(prefix="hf_verify_") as tmp:
        for name, mixer, extra, needs_cuda in CASES:
            if needs_cuda and dev != "cuda":
                print(f"{name:16s} SKIPPED (needs CUDA)", flush=True)
                n_skip += 1
                continue
            mc = ModelConfig(vocab_size=128, d_model=256, n_layers=2, n_heads=4, n_kv_heads=2,
                             head_dim=64, max_seq_len=128, attn_impl="sdpa", z_loss_weight=0.0,
                             mixer=mixer, **extra)
            m = build_model(mc).to(dev).to(torch.bfloat16).eval()
            out_dir = os.path.join(tmp, name)
            save_hf_checkpoint(m.state_dict(), mc, out_dir)
            # load with sdpa to match the original's kernel (flash vs sdpa differ ~8e-2 in bf16: both
            # valid; the eval uses flash_attention_2 by default, which is the intended GQA path).
            hf, info = AutoModelForCausalLM.from_pretrained(
                out_dir, dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
            hf = hf.to(dev).eval()
            # Key coverage: no trained tensor silently dropped. Catches ModelConfig->DMLMConfig field
            # drift, a newly-added ModelConfig field the bridge forgot to carry (as qk_norm once did,
            # which silently un-normalized q/k in every eval).
            unexpected = list(info["unexpected_keys"])
            missing = [k for k in info["missing_keys"] if k != "lm_head.weight"]  # tied; re-tied on load
            x = torch.randint(0, 128, (2, 96), device=dev)
            with torch.no_grad():
                l0 = m(x)["logits"].float()
                l1 = hf(x).logits.float()
                loss = hf(x, labels=x.clone()).loss
            d = (l0 - l1).abs().max().item()
            ok = d < 1e-2 and bool(torch.isfinite(loss)) and not unexpected and not missing
            ok_all &= ok
            print(f"{name:16s} max|Δlogits|={d:.2e} hf_loss={float(loss):.3f} "
                  f"unexpected={len(unexpected)} missing={len(missing)} ok={ok}", flush=True)
            if unexpected or missing:
                print(f"    DROPPED -> unexpected={unexpected[:4]} missing={missing[:4]}")
    print("HF EXPORT ROUND-TRIP", "OK" if ok_all else "FAILED",
          f"({n_skip} case(s) skipped without CUDA)" if n_skip else "")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
