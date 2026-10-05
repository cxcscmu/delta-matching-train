"""The data scripts of data/ on small synthetic inputs, CPU only, no network and no data root: the stage-1 tokenization
and document-level split, the stage-2 shard reader, document reconstruction, re-tokenization, packing, flattening and
validation carve, the checksum verifier, and the agreement of the recipe documents with data/checksums.json. The real
builds are checked end to end by their last step (the outputs against data/checksums.json)."""
import base64
import importlib.util
import json
import re
import subprocess
import sys

import numpy as np
import pytest

from _helpers import REPO

DATA = REPO / "data"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"dm_data_{name}", DATA / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S2 = _load("stage2_repack")
V = _load("verify")


def _py(*args, cwd=None):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, timeout=600, cwd=cwd)


# ---- stage 1 --------------------------------------------------------------------------------------------------------
def _word_tokenizer(texts, out):
    """A tiny word-level fast tokenizer with <unk> = 0, BOS <s> = 1, EOS </s> = 2 (the Llama-2 special ids)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(texts, trainers.WordLevelTrainer(special_tokens=["<unk>", "<s>", "</s>"]))
    PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", unk_token="<unk>").save_pretrained(out)
    return PreTrainedTokenizerFast.from_pretrained(out)


def test_stage1_tokenize(tmp_path):
    """[BOS] + ids + [EOS] per document, the configs in --configs order and the parts in name order, empty documents
    dropped, and the same bytes whatever --workers."""
    pa, pq = pytest.importorskip("pyarrow"), pytest.importorskip("pyarrow.parquet")
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(50)]
    parts = {("High-Quality", "part_000001.parquet"): [], ("High-Quality", "part_000000.parquet"): [],
             ("High-Quality-DQA", "part_000000.parquet"): []}
    for key in parts:
        parts[key] = [" ".join(rng.choice(words, size=rng.integers(1, 30))) for _ in range(40)] + [""]
        (tmp_path / "raw" / key[0]).mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"text": parts[key], "uuid": [str(i) for i in range(len(parts[key]))]}),
                       tmp_path / "raw" / key[0] / key[1])
    tok = _word_tokenizer([t for v in parts.values() for t in v], tmp_path / "tok")
    order = [("High-Quality", "part_000000.parquet"), ("High-Quality", "part_000001.parquet"),
             ("High-Quality-DQA", "part_000000.parquet")]
    want = [[1] + tok(t, add_special_tokens=False)["input_ids"] + [2] for k in order for t in parts[k] if t]
    outs = []
    for w in (1, 2):
        out = tmp_path / f"out{w}" / "s"
        p = _py(DATA / "stage1_tokenize.py", "--raw-dir", tmp_path / "raw", "--configs", "High-Quality", "High-Quality-DQA",
                "--tokenizer", tmp_path / "tok", "--out", out, "--workers", w, "--threads-per", 1, "--batch-size", 7)
        assert p.returncode == 0, p.stderr[-3000:]
        outs.append((out.with_suffix(".bin").read_bytes(), np.load(str(out) + ".doc_lengths.npy")))
    assert outs[0][0] == outs[1][0] and np.array_equal(outs[0][1], outs[1][1])
    assert np.array_equal(np.frombuffer(outs[0][0], np.uint16), np.concatenate(want).astype(np.uint16))
    assert outs[0][1].tolist() == [len(d) for d in want]


def test_stage1_split(tmp_path):
    """Bernoulli(val-frac) documents from RandomState(seed) go to val in the generator's permutation order; the rest keep
    their order; the two streams hold every token once."""
    rng = np.random.default_rng(1)
    dl = rng.integers(1, 50, size=500).astype(np.int64)
    toks = rng.integers(0, 32000, size=int(dl.sum())).astype(np.uint16)
    toks.tofile(tmp_path / "s.bin")
    np.save(tmp_path / "s.doc_lengths.npy", dl)
    p = _py(DATA / "stage1_split.py", "--bin", tmp_path / "s.bin", "--doc-lengths", tmp_path / "s.doc_lengths.npy",
            "--out", tmp_path / "r", "--val-frac", 0.1, "--seed", 7)
    assert p.returncode == 0, p.stderr[-3000:]
    docs = np.split(toks, np.cumsum(dl)[:-1])
    g = np.random.RandomState(7)
    is_val = g.random(len(dl)) < 0.1
    val_order = g.permutation(np.nonzero(is_val)[0])
    train = [docs[i] for i in range(len(dl)) if not is_val[i]]
    assert np.array_equal(np.fromfile(tmp_path / "r_train.bin", np.uint16), np.concatenate(train))
    assert np.array_equal(np.load(tmp_path / "r_train.doc_lengths.npy"), dl[~is_val])
    assert np.array_equal(np.fromfile(tmp_path / "r_val.bin", np.uint16), np.concatenate([docs[i] for i in val_order]))
    assert np.array_equal(np.load(tmp_path / "r_val.doc_lengths.npy"), dl[val_order])


# ---- stage 2 --------------------------------------------------------------------------------------------------------
def _mds_ndarray(a):
    """mosaicml-streaming's ndarray encoding with a static dtype and a dynamic shape."""
    sdt = 0 if max(a.shape) < 256 else 1 if max(a.shape) < 65536 else 2
    return bytes([(a.ndim << 2) | sdt]) + np.asarray(a.shape, ("u1", "u2", "u4")[sdt]).tobytes() + a.tobytes()


def _write_mds(d, windows, per_shard=2):
    """An uncompressed version-2 MDS directory with the release's columns (domain, indices, input_ids)."""
    d.mkdir(parents=True, exist_ok=True)
    shards = []
    for si in range(0, len(windows), per_shard):
        samples = []
        for w in windows[si:si + per_shard]:
            w = np.asarray(w, dtype=np.uint32)
            cols = [b"dom", _mds_ndarray(np.array([[0, len(w)]], dtype=np.uint32)), _mds_ndarray(w)]
            samples.append(b"".join(np.uint32(len(c)).tobytes() for c in cols) + b"".join(cols))
        head = 4 + 4 * (len(samples) + 1)
        offs = np.cumsum([head] + [len(s) for s in samples]).astype(np.uint32)
        name = f"shard.{si // per_shard:05d}.mds"
        blob = np.uint32(len(samples)).tobytes() + offs.tobytes() + b"".join(samples)
        (d / name).write_bytes(blob)
        shards.append({"column_encodings": ["str", "ndarray:uint32", "ndarray:uint32"],
                       "column_names": ["domain", "indices", "input_ids"], "column_sizes": [None, None, None],
                       "compression": None, "format": "mds", "hashes": [],
                       "raw_data": {"basename": name, "bytes": len(blob), "hashes": {}}, "samples": len(samples),
                       "size_limit": 1 << 26, "version": 2, "zip_data": None})
    (d / "index.json").write_text(json.dumps({"shards": shards, "version": 2}))


BOS, EOS = S2.L3_BOS, S2.L3_EOS


def _source_windows(rng, n_windows=6, width=64):
    """Llama-3-style windows: documents with a leading BOS, most ending on EOS, some without (length-chunked), some
    without a BOS, one made of specials only, documents crossing windows and one ending exactly at a window seam."""
    stream = []
    while len(stream) < n_windows * width:
        kind = rng.integers(0, 5)
        body = list(rng.integers(0, 1000, size=rng.integers(1, 90)))
        if kind == 0:
            stream += [BOS] + body                      # length-chunked: no EOS
        elif kind == 1:
            stream += body + [EOS]                      # no BOS: separated by the after-EOS rule
        elif kind == 2:
            stream += [BOS, EOS]                        # nothing but specials: skipped
        else:
            stream += [BOS] + body + [EOS]
    stream = stream[:n_windows * width]
    seam = 2 * width                                     # force a document to end exactly at the second seam
    stream[seam - 1], stream[seam] = EOS, 7              # ... followed by a BOS-less document
    return [np.asarray(stream[i:i + width], dtype=np.uint32) for i in range(0, len(stream), width)]


def _reference_documents(windows):
    """Split the concatenated stream at every BOS and after every EOS."""
    s = np.concatenate(windows)
    cuts = sorted(set(np.nonzero(s == BOS)[0].tolist()) | set((np.nonzero(s[:-1] == EOS)[0] + 1).tolist()) | {0})
    out = []
    for a, b in zip(cuts, cuts[1:] + [len(s)]):
        seg = s[a:b]
        if len(seg):
            out.append((seg[seg < S2.L3_SPECIAL], bool(seg[-1] == EOS)))
    return out


def test_mds_reader(tmp_path):
    rng = np.random.default_rng(2)
    wins = [rng.integers(0, 128256, size=int(n)).astype(np.uint32) for n in (5, 300, 70000, 1)]
    _write_mds(tmp_path / "u", wins, per_shard=3)
    got = list(S2.mds_column(str(tmp_path / "u"), "input_ids"))
    assert len(got) == len(wins) and all(np.array_equal(a, b) and a.dtype == np.uint32 for a, b in zip(got, wins))


def test_mds_reader_matches_mosaicml_streaming(tmp_path):
    """Shards written by mosaicml-streaming itself read back identically (skipped where it is not installed)."""
    pytest.importorskip("streaming")
    from streaming import MDSWriter
    rng = np.random.default_rng(3)
    wins = [rng.integers(0, 128256, size=int(n)).astype(np.uint32) for n in (10, 5000, 300)]
    cols = {"domain": "str", "indices": "ndarray:uint32", "input_ids": "ndarray:uint32"}
    with MDSWriter(out=str(tmp_path / "m"), columns=cols, compression=None, size_limit=1 << 14) as w:
        for a in wins:
            w.write({"domain": "x", "indices": np.array([[0, len(a)]], dtype=np.uint32), "input_ids": a})
    got = list(S2.mds_column(str(tmp_path / "m"), "input_ids"))
    assert len(got) == len(wins) and all(np.array_equal(a, b) for a, b in zip(got, wins))


def test_source_documents():
    rng = np.random.default_rng(4)
    for _ in range(20):
        wins = _source_windows(rng)
        got = list(S2.source_documents(iter(wins)))
        want = _reference_documents(wins)
        assert len(got) == len(want)
        assert all(np.array_equal(a[0], b[0]) and a[1] == b[1] for a, b in zip(got, want))


class _StubDecoder:
    def decode(self, ids):
        return " ".join(map(str, ids.tolist()))


class _StubEncoder:
    """Re-encodes each source id as one or two target ids in 3..999 (an inflating tokenizer, no specials)."""

    def encode(self, text):
        out = []
        for x in map(int, text.split()):
            out += [3 + x % 997] * (1 + x % 2)
        return out


def _reference_unit(windows, pack):
    docs = [np.asarray([S2.L2_BOS] + _StubEncoder().encode(_StubDecoder().decode(c)) + ([S2.L2_EOS] if g else []))
            for c, g in _reference_documents(windows) if len(c)]
    stream = np.concatenate(docs)
    kept = len(stream) // pack * pack
    starts = set(np.cumsum([0] + [len(d) for d in docs[:-1]]).tolist())
    cand = sorted({s for s in starts if 0 < s < kept} | set(range(pack, kept, pack)))
    bounds = [p for p in cand if p % pack or stream[p - 1] == S2.L2_EOS]
    return stream[:kept].astype(np.uint16), np.diff([0] + bounds + [kept]) if kept else np.zeros(0, np.int64)


@pytest.mark.parametrize("pack", [16, 64, 200])
def test_repack_unit_matches_reference(tmp_path, monkeypatch, pack):
    """Packing (carry-over, dropped partial tail) and the document lengths the flatten step recovers, against an
    independent computation on the concatenated document stream."""
    monkeypatch.setattr(S2, "PACK", pack)
    monkeypatch.setattr(S2, "_DEC", _StubDecoder())
    monkeypatch.setattr(S2, "_ENC", _StubEncoder())
    rng = np.random.default_rng(pack)
    for trial in range(5):
        wins = _source_windows(rng, n_windows=8, width=97)
        _write_mds(tmp_path / f"u{trial}", wins)
        prefix = str(tmp_path / f"out{trial}")
        _, ntok, ndocs, _ = S2.repack_unit((str(tmp_path / f"u{trial}"), prefix))
        toks, lens = np.fromfile(prefix + ".bin", np.uint16), np.load(prefix + ".lengths.npy")
        want_t, want_l = _reference_unit(wins, pack)
        assert np.array_equal(toks, want_t) and np.array_equal(lens, want_l), trial
        assert ntok == len(toks) == int(lens.sum()) and ndocs == len(lens)
        again = S2.repack_unit((str(tmp_path / f"u{trial}"), prefix))    # a finished unit is not redone
        assert again[1:3] == (ntok, ndocs)


def test_units_order(tmp_path, monkeypatch):
    """The ten output domains in order, each domain's streams in name order (the 524288 bin sorts first), and a missing
    stream stops the build."""
    monkeypatch.setattr(S2, "PROCS_PER_DIR", 2)
    for s in S2.SOURCE_DIRS:
        for p in ("proc01-64", "proc00-64"):
            (tmp_path / s / p).mkdir(parents=True)
            (tmp_path / s / p / "index.json").write_text("{}")
    got = [(dom, name) for dom, name, _ in S2.units(str(tmp_path))]
    assert [d for d, _ in got] == [dom for dom, srcs in S2.DOMAINS for _ in range(2 * len(srcs))]
    assert [n for _, n in got[:4]] == ["thestackv1_concat_by_repo-524288__proc00-64", "thestackv1_concat_by_repo-524288__proc01-64",
                                       "thestackv1_concat_by_repo-65536__proc00-64", "thestackv1_concat_by_repo-65536__proc01-64"]
    (tmp_path / "arxiv" / "proc01-64" / "index.json").unlink()
    with pytest.raises(SystemExit, match="arxiv: 1 proc"):
        S2.units(str(tmp_path))


def test_llama3_decoder(tmp_path):
    """Byte-level decoding with UTF-8 replacement, as tiktoken decodes (compared with tiktoken where it is installed)."""
    table = [bytes([i]) for i in range(256)] + [f"tok{i}".encode() for i in range(256, S2.L3_SPECIAL)]
    table[300] = "é".encode()                        # a multi-byte character in one token
    (tmp_path / "t.model").write_bytes(b"\n".join(base64.b64encode(t) + b" " + str(i).encode() for i, t in enumerate(table)))
    dec = S2.Llama3Decoder(str(tmp_path / "t.model"))
    ids = np.array([72, 105, 32, 300, 0xC3, 0xA9, 0xFF, 1000], dtype=np.uint32)   # "Hi é" + é split in bytes + a bad byte
    assert dec.decode(ids) == "Hi éé�tok1000"
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.Encoding("t", pat_str=r"\S+|\s+", mergeable_ranks={t: i for i, t in enumerate(table)}, special_tokens={})
    rng = np.random.default_rng(5)
    for _ in range(50):
        ids = rng.integers(0, S2.L3_SPECIAL, size=rng.integers(1, 200)).astype(np.uint32)
        ids[rng.random(len(ids)) < 0.3] = rng.integers(128, 256)             # stray continuation / lead bytes
        assert dec.decode(ids) == enc.decode(ids.tolist())
    (tmp_path / "bad.model").write_bytes(b"\n".join(base64.b64encode(t) + b" " + str(i).encode() for i, t in enumerate(table[:10])))
    with pytest.raises(ValueError, match="10 ids"):
        S2.Llama3Decoder(str(tmp_path / "bad.model"))


def test_stage2_val(tmp_path):
    """The held-out windows are the first N of the seed's permutation from --start on, sorted, copied whole."""
    n = 50
    rng = np.random.default_rng(6)
    toks = rng.integers(0, 32000, size=n * 65536).astype(np.uint16)
    toks.tofile(tmp_path / "t.bin")
    np.save(tmp_path / "t.doc_lengths.npy", np.full(n, 65536, dtype=np.int64))
    p = _py(DATA / "stage2_val.py", "--train", tmp_path / "t", "--out", tmp_path / "v", "--seed", 11, "--start", 0.9,
            "--windows", 3)
    assert p.returncode == 0, p.stderr[-3000:]
    pick = np.sort(np.random.Generator(np.random.PCG64(11)).permutation(n)[int(0.9 * n):][:3])
    want = np.concatenate([toks[c * 65536:(c + 1) * 65536] for c in pick])
    assert np.array_equal(np.fromfile(tmp_path / "v.bin", np.uint16), want)
    assert np.load(tmp_path / "v.doc_lengths.npy").tolist() == [65536] * 3


# ---- checksums ------------------------------------------------------------------------------------------------------
def test_verify_check_files(tmp_path, capsys):
    (tmp_path / "a").write_bytes(b"hello")
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "b").write_bytes(b"x" * 10)
    import hashlib
    files = {"a": {"bytes": 5, "sha256": hashlib.sha256(b"hello").hexdigest()},
             "d/b": {"bytes": 10, "sha256": hashlib.sha256(b"x" * 10).hexdigest()}}
    assert V.check_files(str(tmp_path), files)
    (tmp_path / "a").write_bytes(b"hellO")
    assert not V.check_files(str(tmp_path), files) and "SHA256  a" in capsys.readouterr().out
    assert V.check_files(str(tmp_path), files, hashes=False)
    (tmp_path / "d" / "b").write_bytes(b"x")
    assert not V.check_files(str(tmp_path), files, hashes=False) and "SIZE    d/b" in capsys.readouterr().out
    (tmp_path / "d" / "b").unlink()
    assert not V.check_files(str(tmp_path), files, hashes=False) and "MISSING d/b" in capsys.readouterr().out


def test_checksums_cover_every_part():
    files = V.load_checksums()
    assert all(re.fullmatch(r"[0-9a-f]{64}", v["sha256"]) and v["bytes"] > 0 for v in files.values())
    per = {part: [k for k in files if k.startswith(pre)] for part, pre in V.PARTS.items()}
    assert {p: len(v) for p, v in per.items()} == {"tokenizer": 4, "stage1": 4, "stage2": 4, "ruler": 110}
    assert sum(map(len, per.values())) == len(files)


@pytest.mark.parametrize("doc,sub", [("pretrain_stage1.md", "nemotron_cc_v2d1_hq_dqa"),
                                     ("pretrain_stage2.md", "prolong_64k_llama2")])
def test_recipe_tables_match_checksums(doc, sub):
    """Every output row of a recipe's Result table names a file of checksums.json with its byte count and hash."""
    files = V.load_checksums()
    rows = re.findall(r"^\| `([^`]+)` \|.*?([\d,]+) bytes.*?\| `([0-9a-f]{8})\.\.\.([0-9a-f]{6})` \|$",
                      (DATA / doc).read_text(), flags=re.M)
    assert len(rows) == 4, rows
    for name, nbytes, head, tail in rows:
        want = files[f"{sub}/{name}"]
        assert int(nbytes.replace(",", "")) == want["bytes"], name
        assert want["sha256"].startswith(head) and want["sha256"].endswith(tail), name


@pytest.mark.parametrize("script", ["fetch_tokenizer.py", "build_stage1.py", "build_stage2.py", "prepare_eval.py",
                                    "verify.py", "stage1_tokenize.py", "stage1_split.py", "stage2_repack.py",
                                    "stage2_val.py"])
def test_scripts_start(script, tmp_path):
    p = _py(DATA / script, "--help", cwd=tmp_path)
    assert p.returncode == 0 and "usage" in p.stdout, p.stderr[-2000:]
