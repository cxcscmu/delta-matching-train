# make_mqar is adapted from HazyResearch/zoology (https://github.com/HazyResearch/zoology),
# zoology/data/multiquery_ar.py (multiquery_ar), Apache License, Version 2.0; zoology's LICENSE carries
# the notice "Copyright 2021 The Meerkat Team." You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Changes: numpy only (a per-call RandomState instead of the global seed and torch), no num_passes or
# DataSegment, the answer values also written into the input after each query key, labels aligned for
# the standard next-token shift; make_phonebook is new.
"""Token-space synthetic associative-recall data (MQAR + phonebook), tokenizer-free.

Reimplements the zoology multi-query associative-recall (MQAR) generator
(zoology/data/multiquery_ar.py, Arora/Eyuboglu et al.,
"Zoology: Measuring and improving recall in efficient language models") as an
*in-context probe of a pretrained LM*: the model was NOT trained on this format,
we put N key->value pairs in the context and check whether it can recall a value
when its key is re-queried. Everything is in raw token-id space, so NO tokenizer
is needed: it works on any `dmlm` checkpoint given its `vocab_size`.

Vocabulary split (disjoint, as in zoology):
    keys   = [1, V//2)
    values = [V//2, V)
Layout per example: interleaved pairs ``[k0, v0, k1, v1, ...]`` (the "context"),
then a queries region where query keys are scattered at power-law-spaced gaps and
each query key is immediately followed by the value it should recall.

Label / shift convention (the crux: see `make_mqar`):
    ``labels`` has -100 everywhere EXCEPT at the index of each *value token that
    follows a query key*, where it holds that value id. So ``labels`` is literally
    the input with everything but the answer-value positions masked out. An
    autoregressive LM predicts token at position ``p`` from ``logits[:, p-1]``;
    therefore the harness scores ``logits[:, :-1].argmax(-1) == labels[:, 1:]`` at
    the non-(-100) positions (the standard next-token shift). This matches
    `pretrain.modeling.model.PretrainLM._loss` and `evaluation/ppl/memmap_ppl.py`.
"""
from __future__ import annotations

import numpy as np


def _key_value_choices(vocab_size: int):
    """Disjoint key / value id pools, matching zoology's split."""
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)        # [1, V//2)
    value_choices = np.arange(key_vocab_size, vocab_size)  # [V//2, V)
    return key_choices, value_choices


def make_mqar(
    vocab_size: int,
    num_kv_pairs: int,
    seq_len: int,
    num_examples: int,
    seed: int,
    power_a: float = 0.01,
    random_non_queries: bool = True,
):
    """Generate a batch of MQAR examples directly in token-id space.

    Returns
    -------
    input_ids : np.int64 array  [num_examples, seq_len]
    labels    : np.int64 array  [num_examples, seq_len]
        -100 everywhere except at each *value-token* position that follows a
        query key, where it holds the gold value id. Aligned for the standard
        next-token shift: score ``logits[:, :-1].argmax == labels[:, 1:]``.

    Mirrors zoology ``multiquery_ar``: keys=[1,V//2), values=[V//2,V), interleaved
    ``[k,v,...]`` context of ``2*num_kv_pairs`` tokens, then queries placed at
    power-law gaps with the answer value written right after each query key.
    """
    assert seq_len % 2 == 0, "seq_len must be even"
    assert vocab_size > seq_len, (
        f"vocab_size ({vocab_size}) must exceed seq_len ({seq_len}); the key/value "
        "pools must each be large enough to draw distinct ids")
    context_size = num_kv_pairs * 2
    # need room for the 2*num_kv_pairs context tokens PLUS the queries region (a
    # query key + its answer value per pair => >= 2*num_kv_pairs more positions).
    assert context_size * 2 <= seq_len, (
        f"infeasible cell: 2*context ({context_size * 2}) > seq_len ({seq_len}); "
        f"reduce num_kv_pairs or raise seq_len")

    rng = np.random.RandomState(seed)
    key_choices, value_choices = _key_value_choices(vocab_size)

    # each key/value present exactly once per example (sampled without replacement)
    keys = np.stack([rng.choice(key_choices, size=num_kv_pairs, replace=False)
                     for _ in range(num_examples)])
    values = np.stack([rng.choice(value_choices, size=num_kv_pairs, replace=False)
                       for _ in range(num_examples)])

    # interleaved [k0, v0, k1, v1, ...] context block
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # power-law-spaced query slots over the queries region. `space` slot-pairs are
    # available; gaps index which slot each query key lands in (sampled w/o replace).
    space = (seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1.0)
    p = p / p.sum()
    gaps = np.stack([rng.choice(np.arange(space), size=num_kv_pairs, replace=False, p=p)
                     for _ in range(num_examples)])

    # queries region is laid AFTER the context. Width seq_len - context_size. A query
    # key goes at offset (gap*2); its answer value goes right after at (gap*2)+1.
    queries = np.zeros((num_examples, seq_len - context_size), dtype=np.int64)
    rows = np.arange(num_examples)[:, None]
    np.put_along_axis(queries, gaps * 2, keys, axis=1)
    np.put_along_axis(queries, gaps * 2 + 1, values, axis=1)

    input_ids = np.concatenate([kvs, queries], axis=1)  # [N, seq_len]
    assert input_ids.shape == (num_examples, seq_len)

    # labels: -100 except at the value-token positions in the queries region (the
    # token the model must produce). Absolute index = context_size + gap*2 + 1.
    labels = np.full((num_examples, seq_len), -100, dtype=np.int64)
    answer_pos = context_size + gaps * 2 + 1               # [N, num_kv_pairs]
    labels[rows, answer_pos] = values

    if random_non_queries:
        # fill structural zeros in the queries region with random filler so the model
        # can't trivially attend to a unique zero token. Following zoology, the filler
        # is drawn uniformly from the full vocab [0, V), so it includes key ids, value
        # ids and special ids (e.g. BOS / EOS = 1 / 2 in the Llama-2 vocab, which also
        # lie in the key pool): with 512 pairs in 16384 tokens (V = 32000) about 230
        # fillers per example equal one of that example's own keys. The scored answer
        # positions are not affected (already populated, never overwritten).
        zero_mask = input_ids == 0
        if zero_mask.any():
            rand = rng.randint(0, vocab_size, size=input_ids.shape)
            input_ids[zero_mask] = rand[zero_mask]

    return input_ids, labels


def make_phonebook(
    vocab_size: int,
    num_entries: int,
    seq_len: int,
    num_examples: int,
    seed: int,
    random_non_queries: bool = True,
):
    """Phonebook lookup: a list of key->value entries, then ONE queried key whose
    value the model must recall (a single-query special case of MQAR).

    Layout: interleaved ``[k0, v0, ..., k_{n-1}, v_{n-1}]`` (2*num_entries tokens),
    optional filler, then the query key appended at the end followed by the answer
    value. Returns the same ``(input_ids[N,seq_len], labels[N,seq_len])`` contract
    as `make_mqar` (labels -100 except the single answer-value position).
    """
    assert seq_len % 2 == 0, "seq_len must be even"
    assert vocab_size > seq_len
    context_size = num_entries * 2
    # context + a trailing "query_key value" pair
    assert context_size + 2 <= seq_len, (
        f"infeasible phonebook cell: 2*entries+2 ({context_size + 2}) > seq_len ({seq_len})")

    rng = np.random.RandomState(seed)
    key_choices, value_choices = _key_value_choices(vocab_size)

    keys = np.stack([rng.choice(key_choices, size=num_entries, replace=False)
                     for _ in range(num_examples)])
    values = np.stack([rng.choice(value_choices, size=num_entries, replace=False)
                       for _ in range(num_examples)])

    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    input_ids = np.zeros((num_examples, seq_len), dtype=np.int64)
    input_ids[:, :context_size] = kvs

    # pick one entry per example to query; place "query_key answer_value" at the very end
    q_idx = rng.randint(0, num_entries, size=num_examples)
    rows = np.arange(num_examples)
    q_keys = keys[rows, q_idx]
    q_vals = values[rows, q_idx]
    input_ids[:, seq_len - 2] = q_keys
    input_ids[:, seq_len - 1] = q_vals

    labels = np.full((num_examples, seq_len), -100, dtype=np.int64)
    labels[:, seq_len - 1] = q_vals   # answer value is the last token

    if random_non_queries:
        zero_mask = input_ids == 0
        if zero_mask.any():
            rand = rng.randint(0, vocab_size, size=input_ids.shape)
            input_ids[zero_mask] = rand[zero_mask]

    return input_ids, labels
