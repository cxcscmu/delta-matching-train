#!/usr/bin/env python
"""Mixer micro-benchmark: latency + peak memory vs sequence length.

Compares, at matched width (one mixer layer, bf16):
  gqa-sdpa   dense GQA via torch SDPA (fused flash/cuDNN backend)
  gqa-flash  dense GQA via flash-attn (FA2)
  gdn        Gated DeltaNet (fla, linear attention)
  mamba2     Mamba2 (mamba_ssm, selective SSM)

The point at long context: attention is O(L²) compute (flash makes it O(L) *memory*
but not compute), while GDN/Mamba2 are O(L) compute with fixed state: so the
recurrent mixers should pull ahead in latency (and often memory) as L grows.

Run:  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m pretrain.scripts.bench_mixers
"""
from __future__ import annotations

import argparse

import torch

from ..modeling.config import ModelConfig
from ..modeling.mixers import build_mixer
from ..modeling.layers import RotaryEmbedding

BF16 = torch.bfloat16


def _time(fn, n_warmup: int, n_iter: int) -> float:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter  # ms/iter


def bench_one(name: str, mixer_key: str, attn_impl: str, cfg: ModelConfig,
              B: int, T: int, device, n_warmup=3, n_iter=10) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    cfg.attn_impl = attn_impl
    mixer = build_mixer(mixer_key, cfg, layer_idx=0).to(device).to(BF16)
    x = torch.randn(B, T, cfg.d_model, device=device, dtype=BF16, requires_grad=True)
    rope = RotaryEmbedding(cfg.head_dim, T, cfg.rope_theta).to(device)
    pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    cos, sin = rope(pos)
    cos, sin = cos.to(BF16), sin.to(BF16)

    def fwd():
        with torch.no_grad():
            return mixer(x, cos, sin)

    def fwd_bwd():
        x.grad = None
        out = mixer(x, cos, sin)
        out.float().pow(2).mean().backward()

    t_fwd = _time(fwd, n_warmup, n_iter)
    # peak memory of a single training (fwd+bwd) iteration
    torch.cuda.reset_peak_memory_stats(device)
    fwd_bwd()
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    t_fb = _time(fwd_bwd, n_warmup, n_iter)
    params_m = sum(p.numel() for p in mixer.parameters()) / 1e6

    tok_per_s = B * T / (t_fb / 1e3)
    # the tensors are freed when this frame returns; the next call starts with empty_cache() + a peak reset
    return dict(name=name, t_fwd=t_fwd, t_fb=t_fb, peak=peak_gb, params=params_m, tok_s=tok_per_s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, nargs="+", default=[2048, 16384])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--d_model", type=int, default=1024)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--n_kv_heads", type=int, default=4)
    ap.add_argument("--head_dim", type=int, default=64)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "benchmark needs a GPU"
    device = torch.device("cuda")
    torch.manual_seed(0)

    entries = [
        ("gqa-sdpa", "gqa", "sdpa"),
        ("gqa-flash", "gqa", "flash"),
        ("gdn", "gated_deltanet", "sdpa"),
        ("mamba2", "mamba2", "sdpa"),
    ]
    print(f"# device={torch.cuda.get_device_name(0)} batch={a.batch} d_model={a.d_model} "
          f"heads={a.n_heads}/{a.n_kv_heads} head_dim={a.head_dim} dtype=bf16")

    for T in a.seqs:
        cfg = ModelConfig(vocab_size=32, d_model=a.d_model, n_layers=1, n_heads=a.n_heads,
                          n_kv_heads=a.n_kv_heads, head_dim=a.head_dim, max_seq_len=T)
        rows = []
        for name, key, impl in entries:
            try:
                rows.append(bench_one(name, key, impl, cfg, a.batch, T, device))
            except Exception as ex:
                rows.append(dict(name=name, err=f"{type(ex).__name__}: {str(ex)[:80]}"))
                torch.cuda.empty_cache()
        base = next((r for r in rows if r["name"] == "gqa-flash" and "err" not in r), None)
        print(f"\n=== seq_len = {T} (batch {a.batch}) ===")
        print(f"{'mixer':<11}{'fwd ms':>9}{'fwd+bwd ms':>12}{'peak GB':>9}{'params M':>10}"
              f"{'tok/s':>11}{'speedup':>9}{'mem save':>9}")
        for r in rows:
            if "err" in r:
                print(f"{r['name']:<11}  ERROR: {r['err']}")
                continue
            sp = (base["t_fb"] / r["t_fb"]) if base else float("nan")
            ms = (1 - r["peak"] / base["peak"]) if base else float("nan")
            print(f"{r['name']:<11}{r['t_fwd']:>9.2f}{r['t_fb']:>12.2f}{r['peak']:>9.2f}"
                  f"{r['params']:>10.2f}{r['tok_s']:>11.0f}{sp:>8.2f}x{ms*100:>7.0f}%")
        print("  (speedup / mem-save are fwd+bwd relative to gqa-flash)")


if __name__ == "__main__":
    main()
