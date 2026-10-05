# Adapted from Liger-Kernel (liger_kernel/ops/rms_norm.py), Copyright 2024 LinkedIn Corporation, BSD 2-Clause License.
# https://github.com/linkedin/Liger-Kernel. `_fp8glue_rmsnorm_fwd_kernel` below reproduces the per-row arithmetic of
# Liger's `_rms_norm_forward_kernel` (llama casting mode); the backward calls the installed Liger's `rms_norm_backward`.
# Liger-Kernel's license:
#
# BSD 2-CLAUSE LICENSE
# Copyright 2024 LinkedIn Corporation
# All Rights Reserved.
# Redistribution and use in source and binary forms, with or
# without modification, are permitted provided that the following
# conditions are met:
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""P2 (TE `apply_normalization(..., output_quantizer)`): RMSNorm forward emitting the amax for the fp8 cast.

The per-row arithmetic is Liger's `_rms_norm_forward_kernel` (llama casting mode, elementwise
affine) VERBATIM, launched with Liger's own `calculate_settings` (BLOCK_SIZE = next pow2 of
n_cols, its num_warps) and the same 1-D block shape, so the fp32 reduction tree of
`tl.sum(X*X, axis=0)` and every subsequent op are identical: the bf16 normalized values are
BIT-IDENTICAL to Liger's output (checked in test_fp8glue.py).  Each program handles ROWS
consecutive rows in a loop (same per-row code; one atomic amax per program instead of per row).
Epilogue (constexpr): EMIT_AMAX = the exact max|Y_bf16| side output (one fp32 atomic max per
program).  A direct-fp8 epilogue was tried and REJECTED: adding it changes the compiled
reduction (1-ulp rstd differences from Liger in some rows), so both scaling modes write the
bf16 Y here and cast it in the separate cast_transpose pass (the amax still comes from this
kernel, so no separate amax pass exists).  The backward is Liger's `rms_norm_backward`, untouched.
Only the 1-D (row-mode) Liger path is mirrored: Liger picks it when BLOCK_SIZE > 256 or
n_rows < 32768, which covers the block norms (d >= 512); q/k head norms are not ours.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.distributed.tensor  # noqa: F401  (Liger's RMSNorm references torch.distributed.tensor.DTensor)
import triton
import triton.language as tl
from triton.language.extra.libdevice import rsqrt

from liger_kernel.ops.rms_norm import _CASTING_MODE_LLAMA, rms_norm_backward
from liger_kernel.ops.utils import calculate_settings



@triton.jit
def _fp8glue_rmsnorm_fwd_kernel(
    Y_ptr, Y_row_stride, X_ptr, X_row_stride, W_ptr, RSTD_ptr, amax_ptr, n_rows, n_cols, eps, offset,
    EMIT_AMAX: tl.constexpr, ROWS: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    acc = tl.full([], 0.0, tl.float32)
    for i in range(ROWS):
        row_idx = pid * ROWS + i
        rok = row_idx < n_rows
        rmask = mask & rok
        x_base = X_ptr + row_idx * X_row_stride
        # ---- Liger _rms_norm_forward_kernel, llama casting mode, verbatim ----
        X_row = tl.load(x_base + col_offsets, mask=rmask, other=0)
        X_row_dtype = X_row.dtype
        X_row = X_row.to(tl.float32)
        mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
        rstd = rsqrt(mean_square + eps)
        tl.store(RSTD_ptr + row_idx, rstd, mask=rok)
        X_row = X_row * rstd
        X_row = X_row.to(X_row_dtype)
        Y_row = X_row * (offset + W_row)
        tl.store(Y_ptr + row_idx * Y_row_stride + col_offsets, Y_row, mask=rmask)
        # ---- epilogue: exact amax of the stored bf16 value (what torchao would cast) ----
        if EMIT_AMAX:
            Yf = Y_row.to(X_row_dtype).to(tl.float32)
            acc = tl.maximum(acc, tl.max(tl.abs(Yf), axis=0))
    if EMIT_AMAX:
        tl.atomic_max(amax_ptr, acc)


ROWS_PER_PROG = 4


@torch.no_grad()
def rmsnorm_fwd(X: torch.Tensor, W: torch.Tensor, eps: float, *, amax_out: Optional[torch.Tensor] = None,
                Y: Optional[torch.Tensor] = None):
    """X: [n_rows, n_cols] contiguous bf16. Returns (Y bf16 [n_rows, n_cols], RSTD fp32 [n_rows])."""
    assert X.dim() == 2 and X.is_contiguous()
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    assert BLOCK_SIZE > 256 or n_rows < 4096 * 8, "only Liger's 1-D row-mode path is mirrored"
    RSTD = torch.empty(n_rows, dtype=torch.float32, device=X.device)
    if Y is None:
        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    grid = (triton.cdiv(n_rows, ROWS_PER_PROG),)
    _fp8glue_rmsnorm_fwd_kernel[grid](
        Y, n_cols, X, X.stride(0), W, RSTD, amax_out if amax_out is not None else RSTD,
        n_rows, n_cols, eps, 0.0, EMIT_AMAX=amax_out is not None,
        ROWS=ROWS_PER_PROG, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    return Y, RSTD


class RMSNormCastFn(torch.autograd.Function):
    """Forward: the fused norm (Liger-bitwise bf16 Y) with the amax side output into amax_out;
    returns Y (autograd routes dY through it).  Backward: Liger's rms_norm_backward, exactly the
    call LigerRMSNormFunction makes (in_place=True, row_mode=None, llama casting)."""

    @staticmethod
    def forward(ctx, X, W, eps, amax_out):
        shape = X.shape
        X2 = X.contiguous().view(-1, shape[-1])
        Y, RSTD = rmsnorm_fwd(X2, W, eps, amax_out=amax_out)
        BLOCK_SIZE, num_warps = calculate_settings(shape[-1])
        ctx.save_for_backward(X2, W, RSTD)
        ctx.meta = (eps, BLOCK_SIZE, num_warps)
        return Y.view(shape)

    @staticmethod
    def backward(ctx, dY):
        X2, W, RSTD = ctx.saved_tensors
        eps, BLOCK_SIZE, num_warps = ctx.meta
        dY = dY.contiguous()
        dX, dW = rms_norm_backward(dY, X2, W, RSTD, 0.0, _CASTING_MODE_LLAMA.value, BLOCK_SIZE,
                                   num_warps, True, None)
        return dX, dW, None, None
