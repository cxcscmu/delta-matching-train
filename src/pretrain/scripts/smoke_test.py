#!/usr/bin/env python
"""Harsh smoke test for the pretraining engine.

Run (2 GPUs recommended so DDP/FSDP2 collectives are real):
    torchrun --standalone --nproc_per_node=2 -m pretrain.scripts.smoke_test
    # or single process:
    python -m pretrain.scripts.smoke_test

Checks (all ranks run all checks in lockstep; scalars compared are already
all-reduced, so pass/fail is identical across ranks):
  1. param count == closed-form analytic
  2. causal masking: changing a future token never changes earlier-position logits
  3. forward/backward finite (loss + grads)
  4. learning: overfit a fixed batch -> CE collapses (fsdp2 & ddp)
  5. gradient-accumulation equivalence (token-weighted): (pdbs=4,accum=1) == (pdbs=2,accum=2)
  6. DDP <-> FSDP2 parity: identical loss trajectory within bf16 tolerance
  7. exact checkpoint/resume: resumed loss trajectory == uninterrupted reference
Exit code is nonzero if any check fails.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import torch

from ..config import TrainConfig, OptimConfig, DataConfig
from ..modeling.config import ModelConfig
from ..modeling import build_model
from ..data.data_loader import build_train_dataloader
from ..parallel.dist import dist_init, dist_close, dist_barrier, is_master, get_world_size
from ..parallel.wrap import build_ddp_model, build_fsdp2_model
from ..trainer import DDPTrainer, FSDP2Trainer
from ..utils.misc import seed_all, count_parameters
from .gen_synthetic_data import write_tokens

# Scratch dir under $TMPDIR, one per user; not mkdtemp, because every torchrun rank must see the same path.
WS = os.path.join(tempfile.gettempdir(), f"pretrain_smoke_{os.getuid()}")
QUIET = lambda s, level="info": None


# ----------------------------------------------------------------- builders
def make_cfg(backend, *, vocab, seq, gbs, pdbs, steps, bf16, data_path, dtype,
             run, ckpt_root, resume=False, force_check=False, n_layers=2, d_model=128):
    m = ModelConfig(vocab_size=vocab, d_model=d_model, n_layers=n_layers, n_heads=4,
                    n_kv_heads=2, max_seq_len=seq, attn_impl="sdpa", z_loss_weight=1e-4)
    o = OptimConfig(lr=3e-3, weight_decay=0.1, warmup_ratio=0.0, decay_ratio=0.0,
                    lr_scheduler_type="constant_with_warmup", max_grad_norm=1.0)
    d = DataConfig(paths=data_path, dtype=dtype, num_workers=0, pin_memory=False,
                   persistent_workers=False, seed=42, pad_token_id=None)
    cfg = TrainConfig(run_name=run, backend=backend, seed=1234,
                      per_device_batch_size=pdbs, global_batch_size=gbs, num_steps=steps,
                      bf16=bf16, fp16=False, resume=resume, save_interval=10 ** 9,
                      save_total_limit=None, logging_steps=10 ** 9, force_resume_check=force_check,
                      ckpt_root_dir=ckpt_root, log_root_dir=os.path.join(ckpt_root, "logs"),
                      model=m, optim=o, data=d)
    cfg.derive(get_world_size())
    return cfg


def make_trainer(cfg, device, shuffle):
    seed_all(cfg.seed)  # identical model init on every rank
    loader = build_train_dataloader(cfg.data_cfg, cfg.chunk_size, cfg.global_batch_size,
                                    total_size=cfg.total_size, seed=cfg.data.seed, shuffle=shuffle)
    model = build_model(cfg.model)
    if cfg.backend == "fsdp2":
        pdt = torch.bfloat16 if cfg.bf16 else torch.float32
        model = build_fsdp2_model(model, device, param_dtype=pdt)
        T = FSDP2Trainer
    else:
        model = build_ddp_model(model, device)
        T = DDPTrainer
    return T(model, cfg, loader, device, main_LOG=QUIET)


def run_steps(trainer, n):
    losses = []
    for _ in range(n):
        m = trainer.train_step(grad_clip=trainer.cfg.opt_cfg.max_grad_norm)
        losses.append(float(m["train/ce_loss"]))
        trainer.global_step += 1
        trainer._set_train_state_step(trainer.global_step)
    return losses


def _free(trainer):
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------- checks
def check_param_count():
    cfg = ModelConfig(vocab_size=128, d_model=128, n_layers=3, n_heads=4, n_kv_heads=2, max_seq_len=64)
    model = build_model(cfg)
    total, _ = count_parameters(model)
    analytic = cfg.num_params_analytic()
    return total == analytic, f"live={total} analytic={analytic}"


def check_causal(device):
    seed_all(0)
    cfg = ModelConfig(vocab_size=128, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=64, attn_impl="sdpa", z_loss_weight=0.0)
    model = build_model(cfg).to(device).eval()
    x = torch.randint(0, 128, (2, 32), device=device)
    with torch.no_grad():
        a = model(x)["logits"]
        x2 = x.clone(); x2[:, -1] = (x2[:, -1] + 1) % 128
        b = model(x2)["logits"]
    diff = (a[:, :-1] - b[:, :-1]).abs().max().item()   # earlier positions must be identical
    return diff < 1e-4, f"max|Δlogits[:-1]|={diff:.2e} (future token must not leak)"


def check_fwd_bwd(device):
    seed_all(0)
    cfg = ModelConfig(vocab_size=128, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2,
                      max_seq_len=64, attn_impl="sdpa")
    model = build_model(cfg).to(device)
    x = torch.randint(0, 128, (2, 64), device=device)
    out = model(x, labels=x.clone())
    out["loss"].backward()
    gfin = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    return bool(torch.isfinite(out["loss"]) and gfin), f"loss={out['loss'].item():.3f} grads_finite={gfin}"


def check_overfit(device, backend, data_path, vocab, seq, gbs):
    cfg = make_cfg(backend, vocab=vocab, seq=seq, gbs=gbs, pdbs=gbs // get_world_size(), steps=1,
                   bf16=True, data_path=data_path, dtype="uint16", run=f"overfit_{backend}",
                   ckpt_root=os.path.join(WS, "ckpt"))
    tr = make_trainer(cfg, device, shuffle=False)  # 1 batch/epoch -> same batch each step
    losses = run_steps(tr, 120)
    _free(tr)
    init, final = losses[0], min(losses[-3:])
    return final < 0.5 and final < 0.4 * init, f"{backend}: ce {init:.3f} -> {final:.3f}"


def check_grad_accum(device, data_path):
    W = get_world_size()
    common = dict(vocab=128, seq=64, gbs=4 * W, steps=2, bf16=False, data_path=data_path,
                  dtype="uint16", ckpt_root=os.path.join(WS, "ckpt"))
    a = make_trainer(make_cfg("fsdp2", pdbs=4, run="ga_a1", **common), device, shuffle=False)
    la = run_steps(a, 2); _free(a)
    b = make_trainer(make_cfg("fsdp2", pdbs=2, run="ga_a2", **common), device, shuffle=False)
    lb = run_steps(b, 2); _free(b)
    err = max(abs(x - y) for x, y in zip(la, lb))
    return err < 2e-3, f"accum1 {[round(x,4) for x in la]} vs accum2 {[round(x,4) for x in lb]} maxΔ={err:.2e}"


def check_parity(device, data_path):
    common = dict(vocab=128, seq=64, gbs=4 * get_world_size(), pdbs=4, steps=5, bf16=True,
                  data_path=data_path, dtype="uint16", ckpt_root=os.path.join(WS, "ckpt"))
    d = make_trainer(make_cfg("ddp", run="par_ddp", **common), device, shuffle=False)
    ld = run_steps(d, 5); _free(d)
    f = make_trainer(make_cfg("fsdp2", run="par_fsdp", **common), device, shuffle=False)
    lf = run_steps(f, 5); _free(f)
    err = max(abs(x - y) for x, y in zip(ld, lf))
    return err < 4e-2, f"ddp {[round(x,3) for x in ld]} vs fsdp2 {[round(x,3) for x in lf]} maxΔ={err:.2e}"


def check_resume(device, data_path):
    W = get_world_size()
    common = dict(vocab=128, seq=64, gbs=4 * W, pdbs=4, bf16=False, data_path=data_path,
                  dtype="uint16", ckpt_root=os.path.join(WS, "ckpt_resume"))
    # uninterrupted reference
    a = make_trainer(make_cfg("fsdp2", steps=8, run="ref", **common), device, shuffle=False)
    la = run_steps(a, 8); _free(a)
    # interrupted: run 4, save
    if is_master():
        shutil.rmtree(os.path.join(WS, "ckpt_resume", "rb"), ignore_errors=True)
    dist_barrier()
    b = make_trainer(make_cfg("fsdp2", steps=8, run="rb", **common), device, shuffle=False)
    run_steps(b, 4)
    b.save_ckpt()
    _free(b)
    dist_barrier()
    # resume and continue 4 more
    c = make_trainer(make_cfg("fsdp2", steps=8, run="rb", resume=True, force_check=True, **common),
                     device, shuffle=False)
    resumed_ok = c.global_step == 4
    lc = run_steps(c, 4)   # steps 4..7; sanity check runs on first step (force_check=True raises on mismatch)
    _free(c)
    err = max(abs(x - y) for x, y in zip(la[4:8], lc))
    return resumed_ok and err < 2e-3, f"resumed@{c_step(resumed_ok)} ref[4:8]={[round(x,4) for x in la[4:8]]} resumed={[round(x,4) for x in lc]} maxΔ={err:.2e}"


def c_step(ok):
    return "4" if ok else "WRONG"


# ----------------------------------------------------------------- main
def main():
    device = dist_init()
    rank0 = is_master()
    os.makedirs(os.path.join(WS, "data"), exist_ok=True)
    big = os.path.join(WS, "data", "big.bin")
    over = os.path.join(WS, "data", "over.bin")
    over_gbs = 8
    if rank0:
        write_tokens(big, n_tokens=500_000, vocab=128, dtype="uint16", seed=1)
        write_tokens(over, n_tokens=over_gbs * 64, vocab=64, dtype="uint16", seed=2)
    dist_barrier()

    results = []

    def run(name, fn):
        try:
            ok, detail = fn()
        except Exception as e:
            import traceback
            ok, detail = False, f"EXCEPTION {type(e).__name__}: {e}\n{traceback.format_exc()}"
        results.append((name, ok, detail))
        if rank0:
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        dist_barrier()

    run("1.param_count", check_param_count)
    run("2.causal_no_leak", lambda: check_causal(device))
    run("3.fwd_bwd_finite", lambda: check_fwd_bwd(device))
    run("4a.overfit_fsdp2", lambda: check_overfit(device, "fsdp2", over, 64, 64, over_gbs))
    run("4b.overfit_ddp", lambda: check_overfit(device, "ddp", over, 64, 64, over_gbs))
    run("5.grad_accum_equiv", lambda: check_grad_accum(device, big))
    run("6.ddp_fsdp2_parity", lambda: check_parity(device, big))
    run("7.resume_exactness", lambda: check_resume(device, big))

    dist_close()
    n_fail = sum(1 for _, ok, _ in results if not ok)
    if rank0:
        print(f"\n==== {len(results)-n_fail}/{len(results)} checks PASSED ====", flush=True)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
