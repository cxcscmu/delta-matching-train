"""Side-channel diagnostics for the stale-vs-matched delta study (2026-09-15). DEFAULT OFF.

Everything here runs under no_grad on DETACHED tensors and writes to its own files, so an instrumented
run is numerically identical to an uninstrumented one. Nothing in this module feeds back into training.

WHAT IT MEASURES, and why each quantity is the right one.

  The invariant.  Softmax is invariant to a per-row constant shift of the logits, so the gradient row
  must be orthogonal to 1:  dS_ij = P_ij (dP_ij - delta_i) with delta_i = sum_j P_ij dP_ij  gives
  sum_j dS_ij = 0 EXACTLY (verified in fp64 against autograd; causal masking does not break it).
  The stale delta takes delta from the FORWARD's O while dP comes from the BACKWARD's own realized
  P-hat and V-hat, so sum_j dS_ij = eps_i != 0.  The matched delta recomputes delta from the backward's
  own quantities, so eps_i = 0 by construction.

  eps          = delta_matched - delta_stale, per (b, h, row), in true dP units.  This is the injection
                 itself.  Both deltas are computed here on every logged step, for BOTH arms, so the two
                 arms are measured with the same instrument.
  commonmode   = sum_j dK_j, per (b, kv head).  Exactly 0 in true attention, since
                 sum_j dK_j = sum_i Q_i (sum_j dS_ij) = 0.  Under the stale delta it becomes the
                 coherent vector sum_i eps_i Q_i.  Read straight off the kernel's own dk, so it is the
                 gradient that actually reaches the optimizer, quantization floor included.
  dv_commonmode = the same statistic on dv, as a NEGATIVE CONTROL: dV_j = sum_i P_ij dO_i contains no
                 dS at all, so the violation cannot reach it and this must not separate the arms.

  The point of saving the common-mode VECTOR and not just its norm is coherence.  A per-step ratio of
  ~1.7 between the arms is not what kills training; what matters is whether the excess points the same
  way every step.  Incoherent noise accumulates as sqrt(T), a systematic drift as T, so over 10k steps a
  1.7x per-step ratio becomes a ~100x difference in the DISPLACEMENT of W_K.  The analysis computes
  ||sum_t u_t|| / sum_t ||u_t|| over training for each arm; that single curve is the mechanism.

  Row-level table (a stratified sample of rows) carries eps alongside the row's attention entropy,
  top-1 mass, sequence position, layer and head, so the heavy tail in eps can be attributed to all
  three candidate explanations at once rather than one pre-chosen one.

  env:  DM_DIAG=<interval in BACKWARD PASSES>   DM_DIAG_DIR=<dir>   DM_DIAG_ROWS=<rows/head>

  NOTE the unit: the counter ticks once per backward, and with gradient accumulation there are
  grad_accum backwards per optimizer step (8 at the M-tier cell), so an interval of 200 logs roughly
  every 25 optimizer steps: set DM_DIAG = (optimizer steps between logs) x grad_accum.
"""
from __future__ import annotations

import os

import torch

_INTERVAL = int(os.environ.get("DM_DIAG", "0") or 0)
_DIR = os.environ.get("DM_DIAG_DIR", "")
_ROWS = int(os.environ.get("DM_DIAG_ROWS", "256"))
# Rank 0 only: every rank runs the same layers, so without this the 8 ranks would race on one path and
# the files would be a mixture. Rank 0's shard of the batch is a fair sample for every statistic here.
_RANK = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
ENABLED = _INTERVAL > 0 and bool(_DIR) and _RANK == 0

_step = -1          # advanced by the layer that calls first each backward
_seen: set = set()
_buf: dict = {}


def enabled() -> bool:
    return ENABLED


_params: dict = {}     # layer -> the latest stashed parameter snapshot (gains, projection norms), GPU tensors


def next_is_logged() -> bool:
    """At FORWARD time: will the backward of this microbatch be a logged step?  The counter advances at the
    start of each backward (begin -> flush), so during forward m the last flush set _step = m - 2; the first
    microbatch (nothing seen yet) is logged by construction."""
    if not ENABLED:
        return False
    if _step == -1 and not _seen:
        return True
    return (_step + 2) % _INTERVAL == 0


@torch.no_grad()
def stash_params(layer: int, g_q, g_k, w_q=None, w_k=None) -> None:
    """From the mixer's forward on a logged microbatch, snapshot the layer's qk-norm gain
    vectors and the Frobenius norms of W_Q / W_K (the carrier of link 4 of the chain) so they become per-step
    curves in the diag files instead of checkpoint snapshots.  Cheap (two 128-vectors, two norms); merged into
    the layer's record by `record`.  Under FSDP2 the tensors seen inside forward are the unsharded ones; under
    torchao's fp8 all-gather they may be Float8 tensors, whose norm is that of the fp8-rounded weight (fine
    for a trajectory); any failure is recorded as a string, never raised."""
    if not next_is_logged():
        return
    d = {"g_q": g_q.detach().float().clone() if g_q is not None else None,   # None on no-qk-norm arms
         "g_k": g_k.detach().float().clone() if g_k is not None else None}
    for nm, w in (("wq", w_q), ("wk", w_k)):
        if w is None:
            continue
        try:
            wf = w.detach()
            if hasattr(wf, "full_tensor"):
                wf = wf.full_tensor()
            if hasattr(wf, "to_original_precision"):          # torchao Float8TrainingTensor inside forward
                wf = wf.to_original_precision()
            elif hasattr(wf, "_data") and hasattr(wf, "_scale"):
                wf = wf._data.float() / wf._scale.float()
            with torch.autocast("cuda", enabled=False):
                d[nm + "_fro"] = wf.float().norm().reshape(())
        except Exception as e:  # noqa
            d[nm + "_err"] = repr(e)[:120]
    _params[layer] = d


def begin(layer: int) -> bool:
    """Call once per layer per backward, BEFORE doing any diagnostic work.

    Advances the step bookkeeping (a repeat of a layer id means a new backward has started) and returns
    whether this step should be logged, so the caller can skip the extra delta computation entirely on
    the ~99% of steps that are not logged.
    """
    global _seen
    if not ENABLED:
        return False
    if layer in _seen:
        flush()
        _seen = set()
    _seen.add(layer)
    return (_step + 1) % _INTERVAL == 0


@torch.no_grad()
def record(layer: int, *, dk, dv, dq, eps, q8, lse=None, rows=None, kdc=None, qdc=None, force=None) -> None:
    """dk/dv are the kernel's own (B, N, H, D) grads BEFORE the n_rep fold; eps is (B*H, N) true units."""
    if not ENABLED:
        return
    def cm(t):                                   # (B, N, H, D) -> common mode over the sequence axis
        f = t.detach().float()
        s = f.sum(dim=1)                         # (B, H, D)  exactly 0 for dk in true attention
        a = f.abs().sum(dim=1)
        return s.cpu(), (s.norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-30)).cpu()
    dk_s, dk_r = cm(dk)
    dv_s, dv_r = cm(dv)
    e = eps.detach().float()
    rec = dict(
        dk_commonmode=dk_s, dk_ratio=dk_r,       # the observable
        dv_commonmode=dv_s, dv_ratio=dv_r,       # the negative control
        eps_rms=e.pow(2).mean().sqrt().cpu(), eps_absmax=e.abs().max().cpu(),
        eps_q=torch.quantile(e.abs().flatten().float(),
                             torch.tensor([0.5, 0.9, 0.99, 0.999], device=e.device)).cpu(),
        dq_norm=dq.detach().float().norm().cpu(), dk_norm=dk.detach().float().norm().cpu(),
    )
    if kdc is not None:
        # The mechanism's fingerprint in ACTIVATION space: a coherent common-mode in dK adds a shared
        # constant vector to every key, so the key cloud acquires a DC component. ||mean_t k_t|| / rms(k)
        # is exactly that and needs no weight gather. The caller passes it from the RAW post-RoPE keys
        # (dc_raw): the K codes are mean-centered by the smooth-K quantizer, so they cannot show it.
        rec["k_dc"], rec["q_dc"] = kdc, qdc
    if rows is not None:
        rec["rows"] = rows                       # the stratified per-row table
    if force is not None:
        rec.update(force)                        # dg_total, force_cr, force_ok, n_crushed, ent_full (gain_force)
    for k, v in _params.get(layer, {}).items():  # g_q, g_k, wq_fro, wk_fro from the mixer's forward (stash_params)
        rec[k] = v.cpu() if torch.is_tensor(v) else v
    _buf[layer] = rec


@torch.no_grad()
def gain_force(q8, q8s, k8, k8s, k_rot, dk, eps, dO8, B, H, HK, N, D, bm, bn):
    """Read links 2 and 3 of the mechanism chain IN TRAINING.

    dg_total[d] = sum_{b,j,h} dk[b,j,h,d] * k_rot[b,h//nrep,j,d]: the gradient of the k-norm gain of channel d
    in the kernel frame (rotation preserves each rotary pair's dot product, so the pair sums equal the model's
    own gain gradients up to the 1/g factor); it is what Adam integrates, from the arm's OWN dk.
    force_cr / force_ok[d] = sum_i eps_i * q_id (P k)_id over rows whose dO codes are mostly flushed (crushed)
    vs intact: the residual's spurious push on that gain, per row population.  P is rebuilt in fp32 from the
    codes (the Q codes carry the softmax scale; no further scale).  ent_full[h] = the true-scale mean entropy
    of head h over all rows.  Cost ~1 s per layer per logged step; memory one N x N fp32 at a time."""
    dev = q8.device
    nrep = H // HK
    ar = torch.arange(N, device=dev)
    kr = k_rot.float()                                                   # (B, HK, N, D) raw post-RoPE keys
    dkf = dk.float().permute(0, 2, 1, 3)                                 # (B, N, H, D) -> (B, H, N, D)
    dg_total = (dkf * kr.repeat_interleave(nrep, dim=1)).sum(dim=(0, 1, 2))
    qs = q8s.reshape(B * H, -1); ks = k8s.reshape(B * HK, -1)
    F_cr = torch.zeros(D, device=dev); F_ok = torch.zeros(D, device=dev); n_cr = 0
    ent = torch.zeros(H, device=dev)
    causal = ar[None, :] > ar[:, None]
    for b in range(B):
        for h in range(H):
            bh, bhk = b * H + h, b * HK + h // nrep
            Qd = q8[bh].float() * qs[bh, ar // bm].unsqueeze(-1)
            Kd = k8[bhk].float() * ks[bhk, ar // bn].unsqueeze(-1)
            S = Qd @ Kd.t(); S.masked_fill_(causal, float("-inf"))
            P = S.softmax(-1); del S
            ent[h] += -(P.clamp_min(1e-30).log() * P).sum(-1).mean() / B
            cbar = Qd * (P @ kr[b, h // nrep]); del P                   # (N, D): channel d's share of the attended logit
            e = eps[bh].float()
            crushed = (dO8[bh] == 0).float().mean(-1) > 0.5
            F_cr += (e[crushed, None] * cbar[crushed]).sum(0)
            F_ok += (e[~crushed, None] * cbar[~crushed]).sum(0)
            n_cr += int(crushed.sum())
    return dict(dg_total=dg_total.cpu(), force_cr=F_cr.cpu(), force_ok=F_ok.cpu(), n_crushed=n_cr, ent_full=ent.cpu())


@torch.no_grad()
def dc(codes, scales, nb, blk):
    """||mean over the sequence of the dequantized activation|| / its rms, per (batch*head)."""
    f = codes.float() * scales.reshape(nb, -1)[:, (torch.arange(codes.shape[1], device=codes.device) // blk)].unsqueeze(-1)
    return (f.mean(dim=1).norm(dim=-1) / f.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-30)).cpu()


@torch.no_grad()
def dc_raw(x):
    """dc() of an unquantized activation x (G, N, D): ||mean over the sequence|| / rms, per group."""
    f = x.float()
    return (f.mean(dim=1).norm(dim=-1) / f.pow(2).mean(dim=(1, 2)).sqrt().clamp_min(1e-30)).cpu()


@torch.no_grad()
def row_table(q8, q8s, k8, k8s, eps, B, H, HK, N, D, bm, bn, sm, stride=None):
    """A stratified per-row table so the heavy tail in eps can be attributed to entropy, position and
    head at once instead of one pre-chosen explanation.

    Rows are taken every `stride` positions (so they span the whole context uniformly), and for each the
    exact fp32 attention row is rebuilt from the dequantized codes to get its entropy and top-1 mass.
    Returned columns: position, head, eps, entropy, top1. Cost is O(rows * N * D) per layer, which is a
    rounding error next to the step itself.
    """
    # ~_ROWS sampled rows per (batch, head) whatever the context length, so the file size and the cost
    # do not blow up at N = 8192 (the campaign's training length).
    stride = stride or max(1, N // _ROWS)
    idx = torch.arange(0, N, stride, device=q8.device)
    qs = q8s.reshape(B * H, -1)
    ks = k8s.reshape(B * HK, -1)
    out = []
    for b in range(B):
        for h in range(H):
            bh, bhk = b * H + h, b * HK + (h * HK) // H
            qr = q8[bh, idx].float() * qs[bh, idx // bm].unsqueeze(-1)
            ka = k8[bhk].float() * ks[bhk, (torch.arange(N, device=q8.device) // bn)].unsqueeze(-1)
            # The Q codes from ropeq_q carry the softmax scale FOLDED IN (dequant(q8) == q * sm to 3%; the emulated
            # O matches the kernel's to 0.4% without any further scale, 17% with it: a layout check of the study).
            # Multiplying by `sm` again would flatten every row (entropies ~1.7x too high), so `sm` is kept in the
            # signature for the call sites and is deliberately unused.
            sc = qr @ ka.t()
            sc = sc.masked_fill(torch.arange(N, device=q8.device)[None, :] > idx[:, None], float("-inf"))
            p = sc.softmax(-1)
            ent = -(p.clamp_min(1e-30).log() * p).sum(-1)
            out.append(torch.stack([idx.float(), torch.full_like(idx, h, dtype=torch.float32),
                                    eps[bh, idx].float(), ent, p.max(-1).values], dim=-1))
    return torch.cat(out).cpu()          # (rows, 5): position, head, eps, entropy, top1


def flush() -> None:
    global _step, _buf
    _step += 1
    if _buf and ENABLED:
        os.makedirs(_DIR, exist_ok=True)
        torch.save({"step": _step, "layers": _buf}, os.path.join(_DIR, f"diag_{_step:07d}.pt"))
    _buf = {}
