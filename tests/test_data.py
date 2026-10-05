"""Data checks against the data root (data/README.md): every file's size against data/checksums.json (and its sha256:
the small files always, everything with DM_TEST_SHA256=1), the token streams, the tokenizer, RULER, the HF caches. The
stage-2 corpus is checked when it is built (only scripts/experiments/stage2_context.sh needs it)."""
import importlib.util
import os

import numpy as np
import pytest

from _helpers import REPO, NEMO, hf_home

pytestmark = pytest.mark.data
VOCAB = 32000
PROTOCOL_TOKENS = 2000 * 8192 + 1
S2 = "prolong_64k_llama2"
SMALL = 64 << 20        # files up to this size are hashed on every run


def _verify():
    spec = importlib.util.spec_from_file_location("dm_data_verify", REPO / "data" / "verify.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("part", ["tokenizer", "stage1", "stage2", "ruler"])
def test_files_against_checksums(root, part):
    v = _verify()
    files = {k: x for k, x in v.load_checksums().items() if k.startswith(v.PARTS[part])}
    if part == "stage2" and not (root / S2).is_dir():
        pytest.skip(f"{root / S2} not built (python data/build_stage2.py; only the stage-2 experiment needs it)")
    fix = {"tokenizer": "data/fetch_tokenizer.py", "stage1": "data/build_stage1.py", "stage2": "data/build_stage2.py",
           "ruler": "data/prepare_eval.py"}[part]
    assert v.check_files(str(root), files, hashes=False), f"{part}: missing or wrong-sized files (see above; {fix})"
    full = os.environ.get("DM_TEST_SHA256") == "1"
    hashed = {k: x for k, x in files.items() if full or x["bytes"] <= SMALL}
    assert v.check_files(str(root), hashed, hashes=True), f"{part}: sha256 differs (see above; rebuild with {fix})"


def _stream(root, sub, name, starts_at_document=True):
    p = root / sub / f"{name}.bin"
    if not p.exists():
        pytest.skip(f"{p} not built")
    mm = np.memmap(p, dtype=np.uint16, mode="r")
    dl = np.load(root / sub / f"{name}.doc_lengths.npy", mmap_mode="r")
    assert dl.dtype == np.int64 and dl.ndim == 1 and (dl > 0).all()
    assert int(dl.sum()) == mm.size, f"{name}: doc_lengths sum {int(dl.sum())} != token count {mm.size}"
    rng = np.random.default_rng(0)
    for off in [0] + list(rng.integers(0, mm.size - 8192, size=8)):
        w = np.asarray(mm[off:off + 8192])
        assert int(w.max()) < VOCAB, f"{name}: token id {int(w.max())} >= {VOCAB} at offset {off}"
    if starts_at_document:   # the stage-2 validation windows are cut from the stream wherever they fall
        assert int(mm[0]) == 1, f"{name}: the stream must start with BOS (id 1)"
    return mm


@pytest.mark.parametrize("split", ["train", "val"])
def test_stage1_stream(root, split):
    n = _stream(root, NEMO, f"nemotron_rs_{split}").size
    if split == "val":
        assert n >= PROTOCOL_TOKENS, f"val stream too short for the 2000 x 8192 protocol: {n} tokens"
        assert n == 807013275, f"val stream has {n} tokens, expected 807013275 (data/pretrain_stage1.md)"
    else:
        assert n == 39660504515, f"train stream has {n} tokens, expected 39660504515 (data/pretrain_stage1.md)"


@pytest.mark.parametrize("split", ["train", "val"])
def test_stage2_stream(root, split):
    n = _stream(root, S2, f"prolong64k_{split}", starts_at_document=split == "train").size
    assert n % 65536 == 0, f"the stage-2 {split} stream is not whole 65,536-token windows: {n} tokens"
    assert n == {"train": 566949, "val": 256}[split] * 65536, (split, n)


def test_tokenizer(tok_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    assert len(tok) == VOCAB, len(tok)
    assert tok.bos_token_id == 1 and tok.eos_token_id == 2, (tok.bos_token_id, tok.eos_token_id)
    s = "The quick brown fox jumps over the lazy dog."
    ids = tok(s, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids) == s


def test_val_stream_decodes(root, tok_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    mm = np.memmap(root / NEMO / "nemotron_rs_val.bin", dtype=np.uint16, mode="r")
    text = tok.decode(np.asarray(mm[1:200]).tolist())
    assert len(text.split()) > 20, text[:200]


@pytest.mark.parametrize("length", [4096, 8192])
def test_ruler_layout(root, length):
    if not (root / "ruler-500-llama2").exists():
        pytest.fail(f"{root}/ruler-500-llama2 missing: python data/prepare_eval.py --root {root}")
    d = root / "ruler-500-llama2" / str(length) / "data"
    tasks = {"niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
             "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe", "qa_1", "qa_2"}
    have = {p.name for p in d.iterdir() if p.is_dir()}
    assert tasks <= have, f"missing RULER tasks at {d}: {sorted(tasks - have)}"
    for t in tasks:
        assert any((d / t).iterdir()), f"{d / t} is empty"


def test_hf_datasets_cache():
    h = hf_home()
    want = ["EleutherAI___lambada_openai", "Rowan___hellaswag", "baber___piqa", "allenai___ai2_arc", "allenai___sciq",
            "allenai___openbookqa", "allenai___winogrande", "aps___super_glue", "cais___mmlu",
            "hazyresearch___based-swde-v2", "hazyresearch___based-fda", "hazyresearch___based-squad",
            "mandarjoshi___trivia_qa"]
    if not (h / "datasets").is_dir():
        pytest.fail(f"no datasets cache at {h}/datasets (HF_HOME): python data/prepare_eval.py --root <data root>")
    missing = [w for w in want if not (h / "datasets" / w).is_dir()]
    assert not missing, f"eval datasets missing from {h}/datasets: {missing} (python data/prepare_eval.py)"
