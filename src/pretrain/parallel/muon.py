"""Muon for the FSDP2 trainer (2026-09-26, opt-in: `--optim.name muon`; the default stays AdamW).

One optimizer object, so the trainer, the WSD LambdaLR and the DCP checkpoint see one thing: Muon updates the 2-D hidden
weights, AdamW (torch's own implementation, sharing this optimizer's state and param groups) updates everything else, the
standard split of Keller Jordan's Muon, Moonlight and Kimi K2: embeddings / LM head, norm and qk-norm gains, biases, conv
and SSM scalars stay on AdamW.

Muon step, per 2-D parameter (torch.optim.Muon's math: `_single_tensor_muon`, same Newton-Schulz and coefficients):
  buf <- lerp(buf, g, 1 - momentum);  u <- lerp(g, buf, momentum) (Nesterov) or buf
  O   <- NewtonSchulz5(u) on the FULL matrix;  p <- p (1 - lr wd) - lr_adj O,  lr_adj = lr * 0.2 * sqrt(max(m, n))
(Moonlight's "match_rms_adamw" scale, so the AdamW lr, weight decay and schedule carry over unchanged.) Under FSDP2 the
parameter and its gradient are row shards (DTensor Shard(0)); Newton-Schulz needs the whole matrix, so every rank gathers
the momentum-corrected matrix (`full_tensor()`), orthogonalizes it (the same deterministic computation on every rank) and
applies its own rows (`distribute_tensor(..., src_data_rank=None)`: a local slice, no communication). The momentum buffer
is created from the gradient, a plain DTensor even when the parameter carries torchao's fp8 all-gather wrapper.
"""
from __future__ import annotations

import torch
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.optim._muon import DEFAULT_A, DEFAULT_B, DEFAULT_C, _adjust_lr, _zeropower_via_newtonschulz


class MuonWithAdamW(torch.optim.Optimizer):
    """Param groups carry `use_muon` (True: Muon; False: AdamW). The AdamW groups are stepped by an inner torch.optim.AdamW
    that is re-linked to this optimizer's current group dicts and state on every step (a DCP load replaces both)."""

    def __init__(self, param_groups, lr: float, betas=(0.9, 0.95), eps: float = 1e-8, weight_decay: float = 0.1,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 ns_coefficients=(DEFAULT_A, DEFAULT_B, DEFAULT_C), ns_eps: float = 1e-7,
                 adjust_lr_fn: str = "match_rms_adamw"):
        groups = [dict(g) for g in param_groups]
        adam_groups = [g for g in groups if not g.get("use_muon", False)]
        for g in groups:
            if g.get("use_muon", False):
                for p in g["params"]:
                    if p.ndim != 2:
                        raise ValueError(f"Muon group holds a {p.ndim}-D parameter {tuple(p.shape)}")
        # the inner AdamW fills its own defaults (amsgrad, foreach, fused, ...) into the AdamW group dicts in place
        self._adamw = torch.optim.AdamW(adam_groups, lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay) if adam_groups else None
        defaults = dict(lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay, use_muon=False, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps, ns_coefficients=tuple(ns_coefficients), ns_eps=ns_eps,
                        adjust_lr_fn=adjust_lr_fn)
        super().__init__(groups, defaults)

    def _link_adamw(self):
        if self._adamw is not None:
            self._adamw.param_groups = [g for g in self.param_groups if not g["use_muon"]]
            self._adamw.state = self.state

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_group(group)
        if self._adamw is not None:
            self._link_adamw()
            self._adamw.step()
        return loss

    def _muon_group(self, group):
        lr, wd, mom = group["lr"], group["weight_decay"], group["momentum"]
        lr = float(lr.item()) if torch.is_tensor(lr) else float(lr)
        for p in group["params"]:
            g = p.grad
            if g is None:
                continue
            st = self.state[p]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(g, memory_format=torch.preserve_format)
            buf = st["momentum_buffer"]
            buf.lerp_(g, 1 - mom)
            u = g.lerp(buf, mom) if group["nesterov"] else buf
            full = u.full_tensor() if isinstance(u, DTensor) else u
            o = _zeropower_via_newtonschulz(full, group["ns_coefficients"], group["ns_steps"], group["ns_eps"]).to(p.dtype)
            if isinstance(p, DTensor):
                o = distribute_tensor(o, p.device_mesh, p.placements, src_data_rank=None)
            p.mul_(1 - lr * wd)
            p.add_(o, alpha=-_adjust_lr(lr, group["adjust_lr_fn"], p.shape))


def muon_param_groups(model, opt_cfg):
    """(groups, names): Muon on every 2-D parameter whose name matches no `opt_cfg.muon_exclude_regex` (embeddings / LM
    head); AdamW on the rest, with build_optimizer's decay rule (no decay on 1-D params and `no_wd_regex` names)."""
    import re
    excl = [re.compile(p) for p in opt_cfg.muon_exclude_regex]
    nowd = [re.compile(p) for p in opt_cfg.no_wd_regex]
    buckets = {"muon": [], "decay": [], "no_decay": []}
    names = {k: [] for k in buckets}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and not any(r.search(name) for r in excl):
            k = "muon"
        elif p.ndim <= 1 or any(r.search(name) for r in nowd):
            k = "no_decay"
        else:
            k = "decay"
        buckets[k].append(p); names[k].append(name)
    groups = []
    if buckets["muon"]:
        groups.append({"params": buckets["muon"], "weight_decay": opt_cfg.weight_decay, "use_muon": True})
    if buckets["decay"]:
        groups.append({"params": buckets["decay"], "weight_decay": opt_cfg.weight_decay, "use_muon": False})
    if buckets["no_decay"]:
        groups.append({"params": buckets["no_decay"], "weight_decay": 0.0, "use_muon": False})
    return groups, names
