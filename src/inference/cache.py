"""Hybrid state pool for the Nemotron-H path of the continuous-batching engine.

A Nemotron-H model interleaves Mamba2 layers (O(1) recurrent state), MLP / MoE layers
(stateless) and a few attention layers (growing KV); NVIDIA-Nemotron-3-Nano-30B-A3B, for
example, has 23, 23 and 6. We pre-allocate one *slot* per concurrent sequence (a contiguous
per-layer buffer, not paged) and index into the pool by slot id during batched decode: the
installed kernels (`selective_state_update(state_batch_indices=)`,
`causal_conv1d_update(conv_state_indices=)`, `flash_attn_with_kvcache(cache_batch_idx=)`)
gather/scatter by index in-place, so a sequence's state never moves once admitted.

Dtype note: HF builds `NemotronHHybridDynamicCache` with `dtype=inputs_embeds.dtype`
(= bf16), so the HF generation path this engine reproduces keeps a bf16 Mamba state and the
pool defaults to the model dtype. `state_dtype=torch.float32` (the fp32 SSM state of vLLM's
`mamba_ssm_cache_dtype=float32`) is available as a quality lever but *diverges* from the bf16
HF path: use it to follow that serving setting, not for parity checks.
"""

from __future__ import annotations

import torch


class HybridStatePool:
    """Pre-allocated per-slot Mamba state + attention KV, with a free-list allocator."""

    def __init__(
        self,
        config,
        max_num_seqs: int,
        max_len: int,
        device: torch.device,
        kv_dtype: torch.dtype = torch.bfloat16,
        state_dtype: torch.dtype | None = None,
    ):
        self.S = int(max_num_seqs)
        self.max_len = int(max_len)
        self.device = device
        self.kv_dtype = kv_dtype
        # Match HF: Mamba state lives in the model/cache dtype (bf16) unless overridden.
        self.state_dtype = state_dtype if state_dtype is not None else kv_dtype
        self.layers_block_type = list(config.layers_block_type)

        # --- Mamba dims (from the config)
        n_heads = config.mamba_num_heads
        head_dim = config.mamba_head_dim
        d_state = config.ssm_state_size
        n_groups = config.n_groups
        conv_kernel = config.conv_kernel
        intermediate = n_heads * head_dim
        conv_dim = intermediate + 2 * n_groups * d_state

        # --- Attention dims (from the config)
        n_kv = config.num_key_value_heads
        attn_head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        self.ssm_pool: dict[int, torch.Tensor] = {}
        self.conv_pool: dict[int, torch.Tensor] = {}
        self.k_pool: dict[int, torch.Tensor] = {}
        self.v_pool: dict[int, torch.Tensor] = {}

        for i, bt in enumerate(self.layers_block_type):
            if bt == "mamba":
                # Only the SSM recurrent state honors state_dtype (fp32 = the quality knob,
                # = vLLM's mamba_ssm_cache_dtype). The conv state MUST stay in the compute
                # dtype (bf16): causal_conv1d_update requires it to match x/weight.
                self.ssm_pool[i] = torch.zeros(
                    self.S, n_heads, head_dim, d_state, device=device, dtype=self.state_dtype
                )
                self.conv_pool[i] = torch.zeros(
                    self.S, conv_dim, conv_kernel, device=device, dtype=kv_dtype
                )
            elif bt == "attention":
                # Separate k/v tensors, each (S, max_len, n_kv_heads, head_dim): do NOT pack k,v.
                self.k_pool[i] = torch.zeros(
                    self.S, self.max_len, n_kv, attn_head_dim, device=device, dtype=kv_dtype
                )
                self.v_pool[i] = torch.zeros(
                    self.S, self.max_len, n_kv, attn_head_dim, device=device, dtype=kv_dtype
                )

        # Per-slot current KV length (also the write offset for the next decode token).
        self.seqlens = torch.zeros(self.S, dtype=torch.int32, device=device)
        self._free: list[int] = list(range(self.S))

    # --- allocator ---------------------------------------------------------
    @property
    def num_free(self) -> int:
        return len(self._free)

    def alloc(self) -> int:
        """Take a free slot. Prefill fully (re)initializes every tensor for the slot,
        so no zeroing is needed here. Raises IndexError if exhausted (caller gates on num_free)."""
        return self._free.pop()

    def free(self, slot: int) -> None:
        self.seqlens[slot] = 0
        self._free.append(slot)
