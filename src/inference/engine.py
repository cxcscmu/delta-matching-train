"""EngineModel: drop-in replacement for modeling.wrapper.HFModel.

Exposes the same `generate(...)` token-in/token-out contract the eval harness calls
(`evaluation/lm_eval_harness/wrapper.py::generate_until` and `ruler/ruler.py`):
returns `sequences` (N, L+max_new) with the left-padded prompt echoed in [:, :L] and
generated tokens after column L (finished rows right-padded with pad_token_id), so both
RULER's fixed-width `[:, L:]` slice and lm_eval's argmax-of-non-pad start work.

Construction rebinds the HF model's stateful mixers in place (dm_model.py for dmlm models, model.py for
Nemotron-H), so the plain HF `.forward()` path (loglikelihood) is no longer valid: only `.generate()`. The
evaluation wraps a model in the engine only for generation benchmarks (RULER, extraction, TriviaQA with
EVAL_ENGINE=mini_eval_dev); loglikelihood and recall stay on the HF path.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

from .scheduler import Request, Scheduler

# Stop ids used when the model carries no eos id (neither its generation_config nor its config). [2, 11] is the eos
# list of Nemotron-H's generation_config. This repository's exports carry no eos id either, so every engine-scored
# dmlm benchmark of the study (RULER, extraction, TriviaQA) stopped at these ids as well: with the study's Llama-2
# tokenizer 2 is `</s>` (the harness stop lists contain it too) and 11 is the byte token `<0x08>`, which only the
# engine stops at (the HF generation path does not). The fallback is kept so those numbers reproduce. A model whose
# generation_config or config sets eos_token_id stops at that id instead, e.g. a standalone repo built by
# scripts/hf/make_standalone.py, which copies the tokenizer's eos id into its config.
_FALLBACK_EOS = (2, 11)


def _eos_set(hf_model) -> set[int]:
    eos = getattr(getattr(hf_model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(hf_model.config, "eos_token_id", None)
    if eos is None:
        return set(_FALLBACK_EOS)
    return {int(eos)} if isinstance(eos, int) else {int(e) for e in eos}


def _state_bytes_per_seq(config, dtype_size: int) -> int:
    n_mamba = sum(1 for b in config.layers_block_type if b == "mamba")
    n_heads, head_dim = config.mamba_num_heads, config.mamba_head_dim
    d_state, n_groups, conv_k = config.ssm_state_size, config.n_groups, config.conv_kernel
    conv_dim = n_heads * head_dim + 2 * n_groups * d_state
    per = (conv_dim * conv_k + n_heads * head_dim * d_state) * dtype_size
    return n_mamba * per


def _kv_bytes_per_token(config, dtype_size: int) -> int:
    n_attn = sum(1 for b in config.layers_block_type if b == "attention")
    n_kv = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return n_attn * 2 * n_kv * head_dim * dtype_size  # k + v


class EngineModel(nn.Module):
    def __init__(self, hf_model, max_num_seqs_cap: int | None = None,
                 reserve_gb: float | None = None, state_dtype: torch.dtype | None = None):
        super().__init__()
        self.model = hf_model
        self.config = hf_model.config
        # dmlm (this repository's GQA / MLA / Mamba2 / GDN / KDA hybrids) vs Nemotron-H: pick the engine + pool path.
        self._is_dm = getattr(hf_model.config, "model_type", "") == "dmlm"
        if self._is_dm:
            from .dm_model import EngineDMLM
            self.engine = EngineDMLM(hf_model)   # rebinds mixers in place
        else:
            # imported here: the Nemotron-H path needs transformers' nemotron_h module, mamba_ssm and causal_conv1d
            from .model import EngineNemotronH
            self.engine = EngineNemotronH(hf_model)  # rebinds mixers in place
        self.eos_ids = _eos_set(hf_model)
        self.kv_dtype = self.engine.dtype
        # Mamba2 SSM state precision: the model dtype (bf16) by default, as transformers' Nemotron-H cache keeps it.
        # ENGINE_STATE_DTYPE=fp32 keeps it in float32, the setting of vLLM's `mamba_ssm_cache_dtype=float32` that
        # NVIDIA's serving instructions for Nemotron-H use; it may reduce rambling in long reasoning decodes, and it
        # changes the numerics. (GatedDeltaNet / KDA recurrent states are always float32.)
        if state_dtype is None and os.environ.get("ENGINE_STATE_DTYPE", "").lower() in ("fp32", "float32"):
            state_dtype = torch.float32
        self.state_dtype = state_dtype
        self.max_num_seqs_cap = int(max_num_seqs_cap or int(os.environ.get("ENGINE_MAX_SEQS", "64")))
        # Base decode headroom; the (much larger) long-prompt prefill activation is added
        # dynamically per call (measured ~110 KB/token peak at 131k).
        self.reserve_gb = float(reserve_gb if reserve_gb is not None else os.environ.get("ENGINE_RESERVE_GB", "3"))
        self.prefill_bytes_per_token = float(os.environ.get("ENGINE_PREFILL_BYTES_PER_TOK", "1.1e5"))

    def forward(self, *a, **k):
        # Valid only for .parameters()/get_device(); the HF compute path is rebound.
        return self.model(*a, **k)

    def _dm_kv_bytes_per_token(self, dtype_size: int) -> int:
        from .dm_cache import mixer_kind
        tot = 0
        for blk in self.engine.layers:
            k = mixer_kind(blk.mixer)
            if k == "gqa":
                tot += 2 * blk.mixer.n_kv_heads * blk.mixer.head_dim * dtype_size  # k + v
            elif k == "mla":
                # MHA on the wire: the pool holds n_heads of assembled K and V. Omitting this would
                # make an MLA model look like it costs no KV at all and oversize max_num_seqs.
                tot += 2 * blk.mixer.n_heads * blk.mixer.head_dim * dtype_size
        return tot

    def _dm_state_bytes_per_seq(self, dtype_size: int) -> int:
        from .dm_cache import mixer_kind
        tot = 0
        for blk in self.engine.layers:
            k = mixer_kind(blk.mixer)
            if k == "mamba2":
                m = blk.mixer.mamba
                tot += (m.conv1d.weight.shape[0] * m.d_conv + m.nheads * m.headdim * m.d_state) * dtype_size
            elif k in ("gdn", "kda"):
                g = blk.mixer.gdn if k == "gdn" else blk.mixer.kda
                # fp32 recurrent state, one per V head (GVA: num_v_heads > num_heads), as DMStatePool allocates it
                n_state_heads = getattr(g, "num_v_heads", None) or g.num_heads
                tot += n_state_heads * g.head_k_dim * g.head_v_dim * 4
                if g.use_short_conv:
                    tot += (g.q_conv1d.weight.shape[0] + g.k_conv1d.weight.shape[0]
                            + g.v_conv1d.weight.shape[0]) * g.conv_size * dtype_size
        return tot

    def _max_num_seqs(self, max_len: int, max_prompt_len: int) -> int:
        kv_sz = torch.finfo(self.kv_dtype).bits // 8
        st_sz = torch.finfo(self.state_dtype or self.kv_dtype).bits // 8
        if self._is_dm:
            per_seq = self._dm_state_bytes_per_seq(st_sz) + self._dm_kv_bytes_per_token(kv_sz) * max_len
        else:
            per_seq = _state_bytes_per_seq(self.config, st_sz) + _kv_bytes_per_token(self.config, kv_sz) * max_len
        # One per-sequence prefill activation peak is live at a time (MVP prefills per-seq).
        prefill_act = int(max_prompt_len * self.prefill_bytes_per_token)
        # Release the previous generate() call's freed pool back to the driver so mem_get_info
        # reflects true free memory: otherwise the caching allocator's reserved-but-free blocks
        # make every batch after the first see far less "free" and shrink the slot count.
        torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(self.engine.device)
        budget = max(0, free - int(self.reserve_gb * 1e9) - prefill_act)
        return int(min(self.max_num_seqs_cap, max(1, budget // per_seq)))

    def _make_pool(self, S: int, max_len: int, dev):
        if self._is_dm:
            from .dm_cache import DMStatePool
            return DMStatePool(self.engine.layers, S, max_len, dev,
                                kv_dtype=self.kv_dtype, state_dtype=self.state_dtype)
        from .cache import HybridStatePool
        return HybridStatePool(self.config, S, max_len, dev,
                               kv_dtype=self.kv_dtype, state_dtype=self.state_dtype)

    @torch.inference_mode()
    def generate(self, input_ids, attention_mask=None, max_new_tokens=32, temperature=1.0,
                 top_k=50, top_p=None, repetition_penalty=1.0, do_sample=False,
                 stop_token_list=None, pad_token_id=None, return_dict=False,
                 return_unfinished=False, return_kv_cache=False, prefill_chunk_size=None,
                 tokenizer=None, **kwargs):
        dev = self.engine.device
        input_ids = input_ids.to(dev)
        N, L = input_ids.shape
        if pad_token_id is None:
            pad_token_id = 0
        if attention_mask is None:
            attention_mask = (input_ids != pad_token_id).long()
        attention_mask = attention_mask.to(dev)
        max_new = int(max_new_tokens)
        if max_new < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new}")

        reqs = []
        for i in range(N):
            prompt = input_ids[i][attention_mask[i].bool()].tolist()
            reqs.append(Request(i, prompt, max_new))

        stop_seqs = [list(s) for s in (stop_token_list or [])]
        max_prompt = max((len(r.prompt_ids) for r in reqs), default=L)
        max_len = max_prompt + max_new + 1
        S = min(self._max_num_seqs(max_len, max_prompt), len(reqs))
        pool = self._make_pool(S, max_len, dev)
        sched = Scheduler(self.engine, pool, do_sample=do_sample, temperature=temperature,
                          top_k=top_k, top_p=top_p, stop_seqs=stop_seqs, eos_ids=self.eos_ids)
        done = sched.run(reqs)

        out = torch.full((N, L + max_new), pad_token_id, dtype=torch.long, device=dev)
        out[:, :L] = input_ids
        unfinished = torch.zeros(N, dtype=torch.bool, device=dev)
        for r in done:
            if r.gen:
                out[r.rid, L:L + len(r.gen)] = torch.tensor(r.gen, device=dev, dtype=torch.long)
            unfinished[r.rid] = r.finish_reason == "length"

        if return_dict:
            res = {"sequences": out}
            if return_unfinished:
                res["unfinished_sequences"] = unfinished
            return res
        return out

    @torch.inference_mode()
    def generate_stealing(self, global_prompts: dict, claim_fn, *, max_new_tokens=32,
                          max_prompt_len=None, do_sample=False, temperature=1.0, top_k=50,
                          top_p=None, stop_token_list=None):
        """Work-stealing generate for cross-rank load balancing (see wrapper._generate_until_stealing).

        Processes a GLOBAL pool of prompts where admission is driven by `claim_fn()`: a callable
        returning the next global instance id to run (or None when the distributed pool is exhausted).
        The scheduler keeps slots full by pulling from claim_fn, so a fast rank steals the work a
        straggler rank would otherwise grind alone. No cross-rank collective runs during generation
        (the counter is a torch.distributed Store/TCP service); each instance is prefilled once and
        stays resident in its slot until done (no re-prefill). Returns the done Requests it processed,
        each with rid = the global instance id and .gen = generated tokens.

        global_prompts: {global_id: list[int] prompt token ids}: available on every rank.
        """
        dev = self.engine.device
        max_new = int(max_new_tokens)
        if max_new < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {max_new}")
        if max_prompt_len is None:
            max_prompt_len = max((len(p) for p in global_prompts.values()), default=1)
        max_len = int(max_prompt_len) + max_new + 1
        S = min(self._max_num_seqs(max_len, int(max_prompt_len)), max(1, len(global_prompts)))
        pool = self._make_pool(S, max_len, dev)
        stop_seqs = [list(s) for s in (stop_token_list or [])]
        sched = Scheduler(self.engine, pool, do_sample=do_sample, temperature=temperature,
                          top_k=top_k, top_p=top_p, stop_seqs=stop_seqs, eos_ids=self.eos_ids)

        def _claim():
            gid = claim_fn()
            if gid is None:
                return None
            return Request(int(gid), global_prompts[int(gid)], max_new)

        return sched.run([], claim=_claim)
