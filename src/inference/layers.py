"""Engine subclasses of the two stateful Nemotron-H mixers.

Adapted from transformers' src/transformers/models/nemotron_h/modeling_nemotron_h.py
(`NemotronHMamba2Mixer.cuda_kernels_forward`, `NemotronHAttention.forward`; Copyright 2024
The HuggingFace Inc. team, Copyright (c) 2025, NVIDIA CORPORATION, Apache-2.0). Changes:
the cache interaction only (the per-slot pool of cache.py instead of the HF cache).

These override ONLY the cache interaction; every matmul/kernel is identical to the
HF reference (`modeling_nemotron_h.py`), so the engine is numerically faithful by
construction. The instances are produced by rebinding `__class__` onto the loaded HF
modules (see model.py), so all weights/config-derived attrs are inherited unchanged.

Mamba decode replicates the cached branch of `NemotronHMamba2Mixer.cuda_kernels_forward`
plus `state_batch_indices` / `conv_state_indices`; Mamba prefill replicates its uncached
branch, writing the final states into the slot. Attention uses `flash_attn_with_kvcache`
(decode, gather by `cache_batch_idx`) and `flash_attn_func` (prefill): q is built as
(B,T,H,D) (NOT HF's (B,H,T,D) transpose), and there is NO RoPE (Nemotron-H attention is NoPE).
"""

from __future__ import annotations

import torch

from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHAttention,
    NemotronHMamba2Mixer,
)

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from flash_attn import flash_attn_func, flash_attn_with_kvcache
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

from .context import PHASE_DECODE, EngineContext


class EngineMamba2Mixer(NemotronHMamba2Mixer):
    """Continuous-batching Mamba2 mixer: pool-backed state gathered by slot id."""

    def forward(self, hidden_states: torch.Tensor, ctx: EngineContext):  # type: ignore[override]
        if ctx.phase == PHASE_DECODE:
            return self._decode(hidden_states, ctx)
        return self._prefill(hidden_states, ctx)

    def _decode(self, hidden_states: torch.Tensor, ctx: EngineContext) -> torch.Tensor:
        # hidden_states: (N, 1, H); slot_ids: (N,) int32
        pool = ctx.pool
        slot_ids = ctx.slot_ids
        batch_size = hidden_states.shape[0]
        gtss = self.n_groups * self.ssm_state_size

        in_proj_out = self.in_proj(hidden_states.squeeze(1))  # (N, intermediate+conv_dim+n_heads)
        gate, hidden_states_B_C, dt = torch.split(
            in_proj_out, [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        hidden_states_B_C = causal_conv1d_update(
            hidden_states_B_C,
            pool.conv_pool[self.layer_idx],
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
            conv_state_indices=slot_ids,
        )

        hs, B, C = torch.split(
            hidden_states_B_C, [self.intermediate_size, gtss, gtss], dim=-1
        )
        A = -torch.exp(self.A_log.float())  # (n_heads,)
        A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size).to(torch.float32)
        dt = dt[:, :, None].expand(-1, -1, self.head_dim)
        dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
        D = self.D[:, None, ...].expand(-1, self.head_dim)
        B = B.view(batch_size, self.n_groups, B.shape[1] // self.n_groups)
        C = C.view(batch_size, self.n_groups, C.shape[1] // self.n_groups)
        hs = hs.view(batch_size, self.num_heads, self.head_dim)

        hs = selective_state_update(
            pool.ssm_pool[self.layer_idx],
            hs, dt, A, B, C, D,
            z=None, dt_bias=dt_bias, dt_softplus=True,
            state_batch_indices=slot_ids,
        )
        hs = hs.view(batch_size, self.num_heads * self.head_dim)
        hs = self.norm(hs, gate)
        return self.out_proj(hs)[:, None, ...]

    def _prefill(self, hidden_states: torch.Tensor, ctx: EngineContext) -> torch.Tensor:
        # hidden_states: (1, L, H): a single unpadded sequence
        pool = ctx.pool
        slot = ctx.prefill_slot
        batch_size, seq_len, _ = hidden_states.shape
        gtss = self.n_groups * self.ssm_state_size

        projected_states = self.in_proj(hidden_states)
        A = -torch.exp(self.A_log.float())  # (n_heads,)
        dt_limit_kwargs = {} if self.time_step_limit is None else {"dt_limit": self.time_step_limit}

        gate, hidden_states_B_C, time_step = torch.split(
            projected_states, [self.intermediate_size, self.conv_dim, self.num_heads], dim=-1
        )

        # conv state for the slot = last conv_kernel cols of the PRE-conv input (left zero-padded)
        hsbc_t = hidden_states_B_C.transpose(1, 2)  # (1, conv_dim, L)
        conv_state = torch.nn.functional.pad(hsbc_t, (self.conv_kernel_size - hsbc_t.shape[-1], 0))
        pool.conv_pool[self.layer_idx][slot].copy_(conv_state[0])

        hidden_states_B_C = causal_conv1d_fn(
            x=hsbc_t,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
        ).transpose(1, 2)[:, :seq_len]

        hs, B, C = torch.split(
            hidden_states_B_C, [self.intermediate_size, gtss, gtss], dim=-1
        )
        scan_output, ssm_state = mamba_chunk_scan_combined(
            hs.view(batch_size, seq_len, -1, self.head_dim),
            time_step,
            A,
            B.view(batch_size, seq_len, self.n_groups, -1),
            C.view(batch_size, seq_len, self.n_groups, -1),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=None,
            return_final_states=True,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            **dt_limit_kwargs,
        )
        pool.ssm_pool[self.layer_idx][slot].copy_(ssm_state[0])

        scan_output = scan_output.view(batch_size, seq_len, -1)
        scan_output = self.norm(scan_output, gate)
        return self.out_proj(scan_output)


class EngineAttention(NemotronHAttention):
    """Continuous-batching GQA attention (NoPE): per-slot KV pool via FlashAttention."""

    def forward(self, hidden_states: torch.Tensor, ctx: EngineContext):  # type: ignore[override]
        pool = ctx.pool
        B, T, _ = hidden_states.shape
        # (B, T, H, D): seqlen-second layout for flash_attn (NOT HF's (B,H,T,D))
        q = self.q_proj(hidden_states).view(B, T, -1, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, -1, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, -1, self.head_dim)

        if ctx.phase == PHASE_DECODE:
            slot_ids = ctx.slot_ids
            lens = pool.seqlens.index_select(0, slot_ids)  # pre-write lengths (incremented after the layer loop)
            out = flash_attn_with_kvcache(
                q,
                pool.k_pool[self.layer_idx],
                pool.v_pool[self.layer_idx],
                k=k,
                v=v,
                cache_seqlens=lens,
                cache_batch_idx=slot_ids,
                softmax_scale=self.scaling,
                causal=True,
            )
        else:
            slot = ctx.prefill_slot
            out = flash_attn_func(q, k, v, softmax_scale=self.scaling, causal=True)
            pool.k_pool[self.layer_idx][slot, :T].copy_(k[0])
            pool.v_pool[self.layer_idx][slot, :T].copy_(v[0])

        out = out.reshape(B, T, -1)
        return self.o_proj(out), None
