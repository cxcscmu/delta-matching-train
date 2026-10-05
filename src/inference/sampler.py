"""Batched sampling + stop detection for the engine.

Greedy (`do_sample=False`, RULER) is exact argmax. Sampling replicates the HF warper
order/math (temperature -> top_k -> top_p) so the distribution matches the oracle
(`top_k=0` means *disabled*). Stops match the harness `TokensStoppingCriteria` suffix
semantics plus the engine's eos ids (engine.py::_eos_set: the model's eos_token_id, else
[2, 11]) and `max_new_tokens` (the checks run in scheduler.py::Scheduler._finish).

The top-k / top-p filters are adapted from transformers' TopKLogitsWarper / TopPLogitsWarper
(src/transformers/generation/logits_process.py; Copyright 2024 The HuggingFace Inc. team and
Google DeepMind, Apache-2.0).
"""

from __future__ import annotations

import torch


@torch.inference_mode()
def sample_tokens(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float = 1.0,
    top_k: int | None = 0,
    top_p: float | None = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits: (N, vocab) fp32 -> (N,) long token ids."""
    if not do_sample:
        return logits.argmax(dim=-1)

    logits = logits / max(float(temperature), 1e-6)

    if top_k and top_k > 0:
        k = min(int(top_k), logits.shape[-1])
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if top_p is not None and 0.0 < top_p < 1.0:
        # HF TopPLogitsWarper: ascending sort, drop tokens whose bottom-cumulative
        # prob <= (1 - top_p); always keep at least one token.
        sorted_logits, sorted_idx = torch.sort(logits, descending=False, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove_sorted = cum <= (1.0 - top_p)
        remove_sorted[..., -1:] = False  # min_tokens_to_keep=1
        remove = remove_sorted.scatter(-1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def suffix_hits(gen: list[int], stop_seqs: list[list[int]]) -> bool:
    """True iff `gen` ends with any stop sequence (TokensStoppingCriteria semantics)."""
    for s in stop_seqs:
        n = len(s)
        if n and len(gen) >= n and gen[-n:] == s:
            return True
    return False
