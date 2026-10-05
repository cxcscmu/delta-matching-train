"""Grouped-Query Attention (GQA) with a swappable kernel backend.

Backends (`attn_impl`):
  * "flash"   flash-attn's `flash_attn_func` if importable (FA2 by default; DM_FLASH_IMPL=fa3 pins the FA3 hopper
              wheel, fa2 pins FA2). GQA is native: K,V carry n_kv_heads and are broadcast inside the kernel.
  * "sdpa"    torch `F.scaled_dot_product_attention`. On H100 this dispatches to a fused flash / cuDNN kernel
              (DM_SDPA_BACKEND=cudnn|flash|efficient|math pins the backend); correct everywhere (incl. CPU), so it is
              the default when flash-attn is not installed. The evaluation loaders pick the backend of an export per
              run (EVAL_ATTN_IMPL, default flash_attention_2, i.e. "flash"; hf/modeling_dmlm.py maps it).
  * "dm"      the Delta-Matching fp8 attention kernel (the `delta_matching` package, `dm_attn_func_bhnd`) on this
              mixer's own q/k/v, WITHOUT the fused fp8 producers: a drop-in fp8 core with the delta chosen by
              ModelConfig.dm_mode (stale | matched). The paper's fp8 arms use the `dm` mixer (modeling/dm_attention.py),
              which adds the fused producers around the same kernel; this route is the plain one.
  * "te_fp8"  the core from Transformer Engine's cuDNN fp8 fused attention (modeling/te_attn.py).
  * "auto"    flash if importable, else sdpa.

GQA math: n_heads query heads share n_kv_heads key/value heads, group size n_rep = n_heads / n_kv_heads. K/V
projections are d_model x (n_kv_heads*head_dim), shrinking the KV cache by n_rep at fixed quality. Attention is the
usual softmax(QK^T / sqrt(head_dim) + causal_mask) V.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import (apply_output_gate, apply_rotary, q_proj_out_features, RMSNorm,
                     split_q_gate, yarn_mscale)
from .te_attn import TE_FP8_IMPL   # the 'te_fp8' route (lazy TE import; inert unless selected)

DM_IMPL = "dm"


# ---- optional flash-attn (FA2/3/4) detection -------------------------------
# The candidate order below is FA4/FA3 first, then FA2, but a candidate is only ACCEPTED if it actually supports the
# `dropout_p` kwarg the forward passes. Feature-detect, do not assume: installing the FA3 wheel once made
# `flash_attn_interface` importable and silently switched the engine's GQA path from FA2 to FA3 in every newly started
# process, and FA3's flash_attn_func has no `dropout_p` parameter.
def _pick_flash():
    import inspect
    import os
    # DM_FLASH_IMPL: OPT-IN backend pin.
    #   unset  -> the dropout_p feature-detect (which rejects FA3 and lands on FA2).
    #   fa3    -> flash_attn_interface accepted WITHOUT the dropout_p requirement; the forward then omits dropout_p
    #             (asserting dropout==0). Never the default: silently switching the clean arm's kernel mid-campaign is
    #             the documented failure mode.
    #   fa2    -> pin FA2 explicitly.
    pref = os.environ.get("DM_FLASH_IMPL", "").lower()
    cands = (("flash_attn_interface", "FA3/FA4"), ("flash_attn", "FA2"))
    if pref == "fa2":
        cands = (("flash_attn", "FA2"),)
    elif pref == "fa3":
        cands = (("flash_attn_interface", "FA3/FA4"),)
    for mod, name in cands:
        try:
            fn = getattr(__import__(mod, fromlist=["flash_attn_func"]), "flash_attn_func")
            has_dropout = "dropout_p" in inspect.signature(fn).parameters
            if has_dropout or pref == "fa3":
                return fn, f"{name} ({mod})", has_dropout
        except Exception:
            continue
    return None, "none", True


_FLASH_FN, _FLASH_IMPL, _FLASH_HAS_DROPOUT = _pick_flash()


# DM_SDPA_BACKEND pins the backend torch's SDPA may use on the attn_impl='sdpa' route (cudnn | flash | efficient |
# math). Unset = torch's own dispatch. The paper's Standard arm at head_dim 128 is defined as cuDNN attention and
# torch's pick can differ by head_dim, so the pin lets a run state which kernel it actually ran instead of assuming.
def _sdpa_backend_ctx():
    import contextlib
    import os as _os
    name = _os.environ.get("DM_SDPA_BACKEND", "").strip().lower()
    if not name:
        return contextlib.nullcontext()
    from torch.nn.attention import sdpa_kernel, SDPBackend
    table = {"cudnn": SDPBackend.CUDNN_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION,
             "efficient": SDPBackend.EFFICIENT_ATTENTION, "math": SDPBackend.MATH}
    if name not in table:
        raise ValueError(f"DM_SDPA_BACKEND must be one of {sorted(table)}, got {name!r}")
    return sdpa_kernel(table[name])


FLASH_AVAILABLE = _FLASH_FN is not None


def resolve_attn_impl(impl: str) -> str:
    if impl == "auto":
        return "flash" if FLASH_AVAILABLE else "sdpa"
    if impl == "flash" and not FLASH_AVAILABLE:
        raise RuntimeError("attn_impl='flash' but flash-attn is not importable. "
                           "Install it (FA4 on Blackwell / build-from-source on Hopper) or use 'sdpa'.")
    # Closed set: the forward dispatches "flash" / "dm" / "te_fp8" / everything-else->sdpa, so a typo would otherwise
    # SILENTLY train the sdpa backend; in a study about comparing attention backends that is the worst failure mode
    # available. Raise instead.
    if impl not in ("flash", "sdpa", DM_IMPL, TE_FP8_IMPL):
        raise ValueError(f"attn_impl must be auto|flash|sdpa|{DM_IMPL}|{TE_FP8_IMPL}, got {impl!r}")
    return impl


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[B, n_kv, T, D] -> [B, n_kv*n_rep, T, D] by broadcasting each KV head n_rep times."""
    if n_rep == 1:
        return x
    B, n_kv, T, D = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, T, D).reshape(B, n_kv * n_rep, T, D)


class GQAAttention(nn.Module):
    _dm_announced = False
    _noqk_announced = False

    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.n_rep = cfg.n_rep
        self.head_dim = cfg.head_dim
        self.scale = self.head_dim ** -0.5
        self.attn_dropout = cfg.attn_dropout
        self.impl = resolve_attn_impl(cfg.attn_impl)
        # YaRN attention temperature: multiply post-RoPE q by mscale (1.0 == no YaRN).
        self.attn_mscale = yarn_mscale(cfg.rope_scaling)

        self.attn_output_gate = cfg.attn_output_gate
        self.q_proj = nn.Linear(cfg.d_model, q_proj_out_features(cfg), bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.q_dim, cfg.d_model, bias=False)

        # qk-norm: per-head RMSNorm over head_dim, applied to q,k BEFORE RoPE (Qwen2/3 idiom).
        # Stabilizes attention logits at scale; weight is (head_dim,), shared across heads.
        self.qk_norm = cfg.qk_norm
        self.fused_rope = cfg.fused_rope
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps, cfg.rms_norm_in_fp32, fused=cfg.fused_rmsnorm)
            self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps, cfg.rms_norm_in_fp32, fused=cfg.fused_rmsnorm)
        elif not GQAAttention._noqk_announced:          # the no-qk-norm ablation: one banner per process
            GQAAttention._noqk_announced = True
            print(f"[attention] qk_norm OFF: no q_norm / k_norm on the attention layers (config qk_norm=false; "
                  f"head_dim {self.head_dim})", flush=True)

        # --- attn_impl='dm': the plain Delta-Matching fp8 kernel on this mixer's q/k/v ------------------------------
        # The python chain loads here (fails fast on a broken env); the kernel library loads eagerly so a broken build
        # fails at model build and not mid-step. DISTINCT NUMERICAL BACKEND: never mix or resume across backends.
        self._dm_fn = None
        if self.impl == DM_IMPL:
            if self.head_dim not in (128, 256):
                raise ValueError(f"attn_impl='dm' runs head_dim 128 or 256 (the Delta-Matching kernels), got {self.head_dim}")
            if cfg.attn_dropout != 0.0:
                raise ValueError("attn_impl='dm' does not support dropout (compiled out)")
            self._dm_delta = str(getattr(cfg, "dm_mode", "matched"))
            if self._dm_delta not in ("stale", "matched"):
                raise ValueError(f"attn_impl='dm' takes dm_mode stale|matched (the plain kernel route has no bf16_dp), "
                                 f"got {self._dm_delta!r}")
            try:
                import delta_matching  # noqa: F401
            except ImportError as e:
                raise ImportError("attn_impl='dm' needs the Delta-Matching kernel package: install the delta-matching "
                                  "kernel package (see docs/SETUP.md section 3: git clone --recursive, then "
                                  "pip install --no-build-isolation .)") from e
            from delta_matching import dm_attn_func_bhnd, kernel_version
            from delta_matching.attn import _ops_gt
            from delta_matching.core import _check_hd
            ops = _ops_gt()          # load now (and refuse a build without the global-memory twins)
            _check_hd(self.head_dim, ops)
            self._dm_fn = dm_attn_func_bhnd
            if not GQAAttention._dm_announced:
                GQAAttention._dm_announced = True
                kv = kernel_version()
                print(f"[gqa] attn_impl=dm  {kv}  delta_mode={self._dm_delta}  head_dim={self.head_dim}  "
                      f"build_hd128={int(ops.build_hd128())} build_gt={int(ops.build_gt())}", flush=True)
                try:  # stamp the run's wandb config so logs show which kernel trained which arm
                    import wandb
                    if wandb.run is not None:
                        wandb.config.update({"dm_kernel": kv, "dm_impl": "dm_attn_func_bhnd", "dm_mode": self._dm_delta},
                                            allow_val_change=True)
                except Exception:
                    pass

        # --- attn_impl='te_fp8': the attention core from Transformer Engine's cuDNN fp8 fused attention -------------
        # (DelayedScaling), hd128-only on sm90; modeling/te_attn.py holds the whole route (lazy TE import, layout, the
        # ONE-autocast-per-forward contract the trainer honours via `mixer_forward_ctx`, the fail-loud backend proof, the
        # no-extra-state checkpoint policy). The core has no parameters, so the registration order and the seed-matched
        # init of every other arm are untouched.
        self._te_core = None
        if self.impl == TE_FP8_IMPL:
            from .te_attn import build_te_core, te_forward_ctx
            self._te_core = build_te_core(cfg, layer_idx)
            self.mixer_forward_ctx = te_forward_ctx

    def forward(self, x, cos, sin, attention_mask=None):
        if self._te_core is not None:                      # te_fp8 route (bshd end to end)
            from .te_attn import te_fp8_forward
            return te_fp8_forward(self, x, cos, sin, attention_mask)
        B, T, _ = x.shape
        q, gate = split_q_gate(self.q_proj(x), self.n_heads, self.head_dim, self.attn_output_gate)
        q = q.transpose(1, 2)                                                          # [B,H,T,D]
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B,Hkv,T,D]
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)   # normalize over head_dim, then rotate
        q, k = apply_rotary(q, k, cos, sin, fused=self.fused_rope)
        if self.attn_mscale != 1.0:               # YaRN temperature: logits x mscale^2 (q only, so k untouched)
            q = q * (self.attn_mscale * self.attn_mscale)

        drop = self.attn_dropout if self.training else 0.0

        if self._dm_fn is not None:
            # (B, H, T, D) bf16 in, K/V at their own head count (the kernel reads GQA natively and folds dk/dv to the
            # KV heads), causal only, no mask (packed windows), output (B, T, H, D) contiguous so the reshape into the
            # out-projection is free.
            assert attention_mask is None, "attn_impl='dm' uses packed windows (no padding)"
            out = self._dm_fn(q.contiguous(), k.contiguous(), v.contiguous(), causal=True, delta_mode=self._dm_delta)
            out = out.reshape(B, T, self.n_heads * self.head_dim)
        elif self.impl == "flash" and attention_mask is None:
            # flash_attn_func wants [B, T, H, D] and native GQA (k,v keep n_kv_heads)
            if _FLASH_HAS_DROPOUT:
                out = _FLASH_FN(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    dropout_p=drop, causal=True,
                )
            else:
                # DM_FLASH_IMPL=fa3: FA3's flash_attn_func has no dropout_p at all
                assert drop == 0.0, f"{_FLASH_IMPL} has no dropout support (attn_dropout must be 0)"
                out = _FLASH_FN(
                    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                    causal=True,
                )
            if isinstance(out, tuple):  # FA3/4 returns (out, lse)
                out = out[0]
            out = out.reshape(B, T, self.n_heads * self.head_dim)
        else:
            # native GQA via enable_gqa (torch>=2.5): K,V keep n_kv_heads, no repeat_kv materialization. On H100 SDPA
            # dispatches this to cuDNN's fused attention.
            gqa = self.n_rep > 1
            with _sdpa_backend_ctx():   # no-op unless DM_SDPA_BACKEND is set
                if attention_mask is None:
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop, enable_gqa=gqa)
                else:
                    # attention_mask: [B, T] (1=keep,0=pad). Build additive [B,1,T,T] causal+pad bias.
                    bias = self._build_additive_mask(attention_mask, T, q.dtype, q.device)
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=drop, enable_gqa=gqa)
            out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)

        return self.o_proj(apply_output_gate(out, gate))

    @staticmethod
    def _build_additive_mask(attention_mask, T, dtype, device):
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
        keep = attention_mask.bool()[:, None, None, :] & causal[None, None]  # [B,1,T,T]
        bias = torch.zeros(keep.shape, dtype=dtype, device=device)
        bias.masked_fill_(~keep, torch.finfo(dtype).min)
        return bias
