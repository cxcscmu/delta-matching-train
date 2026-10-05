"""meta_eval --add_bos: the BOS-prompt eval variant for models trained on BOS-anchored windows. The harness wrapper
prefixes BOS through its add_prefix path and RULER through encode_left_padded_with_bos; with the flag off both encodings
are the protocol's (no BOS). CPU only; needs the campaign tokenizer (the tok_dir fixture skips when none is found)."""
import pytest
from transformers import AutoTokenizer

from evaluation.lm_eval_harness.wrapper import LMEvalWrapper
from evaluation.ruler.ruler import encode_left_padded_with_bos

PROMPTS = ["The quick brown fox jumps over the lazy dog.", "A needle:", "Question: what is the capital of France?\nAnswer:"]


@pytest.fixture(scope="module")
def tok(tok_dir):
    t = AutoTokenizer.from_pretrained(str(tok_dir))
    if t.pad_token is None:                       # what load_model_for_eval does
        t.pad_token = t.eos_token
    return t


def test_ruler_bos_encoding(tok):
    ref = tok(PROMPTS, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    got = encode_left_padded_with_bos(tok, PROMPTS)
    assert got["input_ids"].shape[1] == ref["input_ids"].shape[1] + 1
    for i, p in enumerate(PROMPTS):
        ids = tok.encode(p, add_special_tokens=False)
        n = len(ids) + 1
        row, mask = got["input_ids"][i].tolist(), got["attention_mask"][i].tolist()
        assert row[-n:] == [tok.bos_token_id] + ids                                  # [bos] + the default tokens, right-aligned
        assert all(x == tok.pad_token_id for x in row[:-n]) and mask == [0] * (len(row) - n) + [1] * n
        r = ref["input_ids"][i].tolist()
        assert r[-len(ids):] == ids                                                  # the default path: same tokens, no BOS
        assert tok.bos_token_id not in ref["input_ids"][i][ref["attention_mask"][i].bool()].tolist()


def _wrapper(tok, add_prefix):
    return LMEvalWrapper(model=None, tokenizer=tok, max_seq_len=512, batch_size=1, max_new_tokens=16, add_prefix=add_prefix)


def test_harness_add_prefix(tok):
    from lm_eval.api.instance import Instance
    reqs = [Instance(request_type="loglikelihood", doc={}, arguments=("Paris is the capital of", " France"), idx=0)]
    enc = {}
    for flag in (False, True):
        w = _wrapper(tok, flag)
        seen = []
        w._loglikelihood_tokens = lambda new_reqs, **kw: seen.extend(new_reqs) or [(0.0, True)] * len(new_reqs)
        w.loglikelihood(reqs)
        (_, ctx, cont), = seen
        enc[flag] = (ctx, cont)
        gen = w.tok_encode("Answer:")                                                  # the generation path's encoder
        assert (gen[0] == tok.bos_token_id) == flag
    (c0, k0), (c1, k1) = enc[False], enc[True]
    assert c1 == [tok.bos_token_id] + c0 and k1 == k0 and c0[0] != tok.bos_token_id  # BOS on the context only, same continuation
