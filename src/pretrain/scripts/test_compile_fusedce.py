#!/usr/bin/env python
"""Correctness tests for torch.compile (regional) + fused linear cross-entropy (liger).

Single GPU, no distributed: isolates the two features' numerics against the unfused/eager
reference on the SAME weights. Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m pretrain.scripts.test_compile_fusedce
"""
from __future__ import annotations

import torch

from pretrain.modeling.config import ModelConfig
from pretrain.modeling.model import build_model


def _grad_snapshot(model):
    return {n: (p.grad.detach().float().clone() if p.grad is not None else None)
            for n, p in model.named_parameters()}


def _max_grad_diff(ga, gb):
    worst, worst_name = 0.0, ""
    for n in ga:
        if ga[n] is None or gb.get(n) is None:
            continue
        d = (ga[n] - gb[n]).abs().max().item()
        scale = ga[n].abs().max().item() + 1e-8
        rel = d / scale
        if rel > worst:
            worst, worst_name = rel, n
    return worst, worst_name


def _run(model, ids, labels, autocast, dev):
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=autocast):
        out = model(ids, labels=labels)
        loss = out["loss"]
    loss.backward()
    return float(loss.detach()), _grad_snapshot(model)


def main():
    dev = torch.device("cuda")
    torch.manual_seed(0)
    results = []

    def check(name, ok, detail=""):
        results.append((name, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")

    for zw in (0.0, 1e-4):
        cfg = ModelConfig(vocab_size=512, d_model=256, n_layers=4, n_heads=8, n_kv_heads=2,
                          head_dim=32, max_seq_len=128, mixer="gqa", attn_impl="sdpa",
                          z_loss_weight=zw)
        model = build_model(cfg).to(dev)
        ids = torch.randint(0, cfg.vocab_size, (4, 128), device=dev)
        labels = ids.clone()
        labels[:, : 3] = -100  # exercise ignore_index

        # ---- fused vs unfused, fp32 (clean correctness) ----
        model.enable_fused_ce(False)
        lu, gu = _run(model, ids, labels, autocast=False, dev=dev)
        model.enable_fused_ce(True)
        lf, gf = _run(model, ids, labels, autocast=False, dev=dev)
        dl = abs(lu - lf)
        dg, gname = _max_grad_diff(gu, gf)
        check(f"fused==unfused fp32 (z={zw})", dl < 1e-4 and dg < 1e-4,
              f"|Δloss|={dl:.2e} max-rel-Δgrad={dg:.2e}@{gname}")

        # ---- fused vs unfused, bf16 autocast (realistic) ----
        model.enable_fused_ce(False)
        lu2, gu2 = _run(model, ids, labels, autocast=True, dev=dev)
        model.enable_fused_ce(True)
        lf2, gf2 = _run(model, ids, labels, autocast=True, dev=dev)
        dl2 = abs(lu2 - lf2)
        dg2, gname2 = _max_grad_diff(gu2, gf2)
        check(f"fused==unfused bf16 (z={zw})", dl2 < 5e-2 and dg2 < 5e-2,
              f"|Δloss|={dl2:.2e} max-rel-Δgrad={dg2:.2e}@{gname2}")

    # ---- torch.compile (regional) vs eager, and compile+fused ----
    from pretrain.parallel.wrap import apply_compile
    cfg = ModelConfig(vocab_size=512, d_model=256, n_layers=4, n_heads=8, n_kv_heads=2,
                      head_dim=32, max_seq_len=128, mixer="gqa", attn_impl="sdpa", z_loss_weight=1e-4)
    ids = torch.randint(0, cfg.vocab_size, (4, 128), device=dev)
    labels = ids.clone(); labels[:, :3] = -100

    torch.manual_seed(1); m_eager = build_model(cfg).to(dev)
    le, ge = _run(m_eager, ids, labels, autocast=True, dev=dev)

    torch.manual_seed(1); m_comp = build_model(cfg).to(dev)
    apply_compile(m_comp)  # compile the blocks
    lc, gc = _run(m_comp, ids, labels, autocast=True, dev=dev)
    dlc = abs(le - lc); dgc, gnc = _max_grad_diff(ge, gc)
    check("compile==eager bf16 (unfused)", dlc < 5e-2 and dgc < 1e-1,
          f"|Δloss|={dlc:.2e} max-rel-Δgrad={dgc:.2e}@{gnc}")

    torch.manual_seed(1); m_cf = build_model(cfg).to(dev)
    apply_compile(m_cf); m_cf.enable_fused_ce(True)
    lcf, gcf = _run(m_cf, ids, labels, autocast=True, dev=dev)
    dlcf = abs(le - lcf); dgcf, gncf = _max_grad_diff(ge, gcf)
    check("compile+fused==eager-unfused bf16", dlcf < 5e-2 and dgcf < 1e-1,
          f"|Δloss|={dlcf:.2e} max-rel-Δgrad={dgcf:.2e}@{gncf}")

    # ---- fused path really skips logits ----
    m_cf.enable_fused_ce(True)
    out = m_cf(ids, labels=labels)
    check("fused returns logits=None", out["logits"] is None and "loss" in out)
    m_cf.enable_fused_ce(False)
    out2 = m_cf(ids, labels=labels)
    check("unfused returns logits tensor", torch.is_tensor(out2["logits"]))

    npass = sum(ok for _, ok, _ in results)
    print(f"\n==== {npass}/{len(results)} checks PASS ====")
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
