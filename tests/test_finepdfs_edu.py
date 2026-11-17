"""The FinePDFs-Edu English build (data/build_finepdfs_edu.py, data/finepdfs_tokenize.py, data/finepdfs_select.py;
data/pretrain_finepdfs_edu.md) on small synthetic parquet files with the dataset's schema and a tiny word-level
tokenizer that has the Llama-2 special ids (<unk> 0, <s> 1, </s> 2): CPU only, no network, no data root.

Covers each rule at its threshold, the tokenizer call (literal special strings stay text), the [BOS] + ids + [EOS]
layout against data/stage1_tokenize.py, the id dedup, the URL groups, the keyed sampling against an independent
re-computation (targets, whole-group overshoot, nesting, a pool too small), the independence of the outputs from
--workers and --out-name, the publish / verify steps with the manifest and finepdfs_edu_checksums.json, and the
outputs read back through the training loader (DocAwareMemmapDataset, build_train_dataloader). The real build is
checked end to end by its own last steps (the stream invariants, the loader, the checksums)."""
import hashlib
import importlib.util
import itertools
import json
import os
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


T = _load("finepdfs_tokenize")
S = _load("finepdfs_select")
B = _load("build_finepdfs_edu")
SEED = 11785


def _py(*args, cwd=None):
    return subprocess.run([sys.executable, "-I", *map(str, args)], capture_output=True, text=True, timeout=900, cwd=cwd)


# ---- synthetic finepdfs-edu ----------------------------------------------------------------------------------------
EN = ("the history of the river valley is a story of farms and towns that grew with the trade of grain and wool as "
      "the people of the region learned to build mills on the banks of the water for their children in winter").split()
ES = ("el gato se sienta sobre la alfombra mientras el perro duerme dentro de la casa con una pelota roja para los "
      "chicos que juegan fuera durante la tarde").split()
CONTENT = "river valley farms towns grain wool mills banks water people region trade history story".split()


def _words(rng, words, n):
    return " ".join(rng.choice(words, size=n))


def _doc(pages, labels=None, edu=2.0, url=None, did=None, lid="eng_Latn", page_ends=None, extractor="docling"):
    text = "".join(pages)
    ends = page_ends if page_ends is not None else list(np.cumsum([len(p) for p in pages]).tolist())
    return {"text": text, "id": did, "url": url, "full_doc_lid": lid,
            "per_page_languages": labels or ["eng_Latn"] * len(pages), "page_ends": ends, "fw_edu_scores": [edu],
            "extractor": extractor}


def _good(rng, f, i, n=None, edu=None):
    n = n or int(rng.integers(60, 200))
    pages = [_words(rng, EN, n // 2) + " ", _words(rng, EN, n - n // 2)]
    return _doc(pages, edu=float(edu if edu is not None else rng.choice([1.5, 1.8, 2.2, 2.9, 3.4])),
                url=f"https://site{int(rng.integers(0, 4))}.org/f{f}/doc{i}.pdf", did=f"<urn:uuid:{f}-{i}>")


def corpus(n_per_file=40, seed=0):
    """Three files. File 0 opens with one document per rule that fails it (and nothing earlier), a counted document
    (edu 1.45), an English page labeled por_Latn with unknown and zxx pages, literal special strings, an overshooting
    page_ends, and URL / id duplicates whose partners are in files 1 and 2."""
    rng = np.random.default_rng(seed)
    files = [[], [], []]
    f0 = files[0]
    f0.append(_doc([_words(rng, ES, 80)], labels=["spa_Latn"], lid="spa_Latn", url="https://a.es/1", did="L1"))
    f0.append(_doc([_words(rng, EN, 30) + " "] * 4, labels=["eng_Latn", "spa_Latn", "spa_Latn", "spa_Latn"],
                   url="https://b.org/2", did="L2"))
    f0.append(dict(_good(rng, 0, 2), id="E", fw_edu_scores=[1.3]))
    f0.append(dict(_good(rng, 0, 3), id="count", fw_edu_scores=[1.45, 1.2]))
    f0.append(_doc([_words(rng, EN, 100) + " ", _words(rng, ES, 70)], labels=["eng_Latn", "spa_Latn"],
                   url="https://c.org/4", did="L3"))
    f0.append(_doc(["chapter one of the river valley\n" * 12 + _words(rng, EN, 30)], url="https://d.org/5", did="G1"))
    f0.append(_doc([" ".join(f"{int(x):06d}" for x in rng.integers(0, 10 ** 6, 60)) + " " + _words(rng, EN, 25)],
                   url="https://e.org/6", did="G2"))
    f0.append(_doc([_words(rng, CONTENT, 90)], url="https://f.org/7", did="G3"))
    f0.append(_doc([_words(rng, EN, 30)], url="https://g.org/8", did="G4"))
    f0.append(_doc([_words(rng, EN, 60) + " ", _words(rng, EN, 50) + " ", _words(rng, EN, 40) + " ", "12 34 56 ",
                    "--- ... ---"], labels=["eng_Latn", "eng_Latn", "por_Latn", "unknown", "zxx_Latn"],
                   url="https://h.org/9", did="por"))
    f0.append(_doc([_words(rng, EN, 40) + " <s> the </s> of <unk> and " + _words(rng, EN, 40)], url="https://i.org/10",
                   did="literal"))
    t = _words(rng, EN, 80)
    f0.append(_doc([t], page_ends=[len(t) // 2, len(t) + 500], labels=["eng_Latn", "eng_Latn"], url="https://j.org/11",
                   did="overshoot"))
    f0.append(dict(_good(rng, 0, 12), url="http://x.org/a.pdf", id="urlA"))
    f0.append(dict(_good(rng, 0, 13, n=80), id="dup"))
    files[1].append(dict(_good(rng, 1, 0), url="https://WWW.X.org/a.pdf", id="urlB"))
    files[1].append(dict(_good(rng, 1, 1, n=190), id="dup"))
    tie = dict(_good(rng, 1, 2, n=100), id="tie")
    files[1].append(tie)
    files[2].append(dict(tie, url="https://k.org/tie2"))
    for f in range(3):
        while len(files[f]) < n_per_file:
            files[f].append(_good(rng, f, len(files[f])))
    return files


def write_parquet(path, docs, row_group_size=5):
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = len(docs)
    cols = {
        "text": pa.array([d["text"] for d in docs], pa.string()),
        "id": pa.array([d["id"] for d in docs], pa.string()),
        "dump": pa.array(["CC-MAIN-2024-10"] * n), "url": pa.array([d["url"] for d in docs], pa.string()),
        "date": pa.array(["2024-01-01"] * n), "file_path": pa.array(["s3://x"] * n),
        "offset": pa.array(np.arange(n, dtype=np.int64)),
        "token_count": pa.array([len(d["text"].split()) for d in docs], pa.int64()),
        "language": pa.array(["eng_Latn"] * n), "page_average_lid": pa.array(["eng_Latn"] * n),
        "page_average_lid_score": pa.array([0.9] * n), "full_doc_lid": pa.array([d["full_doc_lid"] for d in docs]),
        "full_doc_lid_score": pa.array([0.9] * n),
        "per_page_languages": pa.array([d["per_page_languages"] for d in docs], pa.list_(pa.string())),
        "is_truncated": pa.array([False] * n),
        "extractor": pa.array([d["extractor"] for d in docs]).dictionary_encode(),
        "page_ends": pa.array([d["page_ends"] for d in docs], pa.list_(pa.int64())),
        "fw_edu_scores": pa.array([d["fw_edu_scores"] for d in docs], pa.list_(pa.float64())),
        "minhash_cluster_size": pa.array([1] * n, pa.int64()), "duplicate_count": pa.array([1] * n, pa.int64()),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.table(cols), path, row_group_size=row_group_size)


def word_tokenizer(texts, out):
    """A word-level fast tokenizer over every piece of `texts` with <unk> = 0, <s> = 1, </s> = 2."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    pre = pre_tokenizers.Whitespace()
    pieces = sorted({p for t in texts for p, _ in pre.pre_tokenize_str(t)})
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
    for p in pieces:
        vocab.setdefault(p, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre
    PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", unk_token="<unk>").save_pretrained(out)
    return AutoTokenizer.from_pretrained(str(out))


def make_listing(raw, files, path):
    import pyarrow.parquet as pq
    ents = []
    for f in files:
        p = os.path.join(raw, f)
        md = pq.ParquetFile(p).metadata
        ents.append({"path": f, "bytes": os.path.getsize(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                     "rows": md.num_rows, "row_groups": md.num_row_groups})
    with open(path, "w") as fh:
        json.dump({"repo": "synthetic", "config": "eng_Latn", "split": "train", "files": ents}, fh)


REL = [f"data/eng_Latn/train/000_{i:05d}.parquet" for i in range(3)]
TRAIN_T, VAL_T = 6000, 1500


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("finepdfs")
    files = corpus()
    for rel, docs in zip(REL, files):
        write_parquet(str(d / "raw" / rel), docs)
    tok = word_tokenizer([x["text"] for docs in files for x in docs], d / "tok")
    make_listing(str(d / "raw"), REL, d / "files.json")
    return {"dir": d, "files": files, "tok": tok}


def _build_args(synth, root, work, *extra, out_name="test_corpus", workers=2):
    d = synth["dir"]
    return ["--root", root, "--raw", d / "raw", "--work", work, "--files-json", d / "files.json", "--tokenizer",
            d / "tok", "--allow-other-tokenizer", "--workers", workers, "--threads-per", 1, "--train-tokens", TRAIN_T,
            "--val-tokens", VAL_T, "--seed", SEED, "--out-name", out_name, "--checksums", d / "checksums.json",
            "--seq-len", 64, *extra]


@pytest.fixture(scope="module")
def built(synth):
    """One full build (download = verify only, tokenize, select, write, publish, verify) with 2 workers."""
    d = synth["dir"]
    p = _py(DATA / "build_finepdfs_edu.py", *_build_args(synth, d / "root", d / "work"), cwd=d)
    assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-4000:]
    out = d / "root" / "test_corpus"
    return {"out": out, "work": d / "work", "log": p.stdout, "manifest": json.loads((out / "manifest.json").read_text())}


def _encode(tok, texts):
    return tok(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]


def expected_pool(synth, edu_min=1.5):
    """Every document passing the rules (decided by the unit-tested T.text_rules) and G4, from scratch."""
    tok, out = synth["tok"], []
    for f, docs in enumerate(synth["files"]):
        for r, x in enumerate(docs):
            ok, edu = T.text_rules(x["text"], x["full_doc_lid"], x["per_page_languages"], x["page_ends"],
                                   x["fw_edu_scores"], edu_min)
            if not all(ok.values()):
                continue
            ids = _encode(tok, [x["text"]])[0]
            if len(ids) >= 50:
                out.append({"file": f, "row": r, "id": x["id"], "nu": T.norm_url(x["url"]), "n": len(ids) + 2,
                            "ids": ids, "edu": edu})
    return out


def expected_selection(pool, seed, train_t, val_t):
    best = {}
    for x in pool:
        b = best.get(x["id"])
        if b is None or (x["n"], -x["file"], -x["row"]) > (b["n"], -b["file"], -b["row"]):
            best[x["id"]] = x
    docs = sorted(best.values(), key=lambda x: (int.from_bytes(hashlib.blake2b(
        b"finepdfs-edu-en|" + str(seed).encode() + b"|" + x["nu"].encode(), digest_size=8).digest(), "big"),
        x["nu"], x["id"], x["file"], x["row"]))
    val, train, vt, tt = [], [], 0, 0
    for _, g in itertools.groupby(docs, key=lambda x: x["nu"]):
        g = list(g)
        n = sum(x["n"] for x in g)
        if vt < val_t:
            val += g
            vt += n
        elif tt < train_t:
            train += g
            tt += n
        else:
            break
    return val, sorted(train, key=lambda x: (x["file"], x["row"])), vt, tt


# ---- the rules -----------------------------------------------------------------------------------------------------
def test_norm_url_and_key():
    for u in ("http://x/a.pdf", "https://www.x/a.pdf", "HTTPS://WWW.X/a.pdf", "  https://X/a.pdf \n", "www.x/a.pdf"):
        assert T.norm_url(u) == "x/a.pdf", u
    assert T.norm_url("https://www.Example.COM/Path/A.PDF?Q=1#Frag") == "example.com/Path/A.PDF?Q=1#Frag"
    assert T.norm_url("https://wwwx.com/a") == "wwwx.com/a"
    assert T.norm_url("ftp://www.x/a") == "ftp://www.x/a"            # only http(s) is stripped; the rest is the host
    assert T.norm_url("https://Host.org?a=B") == "host.org?a=B"
    assert T.norm_url(None) == T.norm_url("") == ""
    want = int.from_bytes(hashlib.blake2b(b"finepdfs-edu-en|11785|x/a.pdf", digest_size=8).digest(), "big")
    assert T.url_key("x/a.pdf", 11785) == want
    assert T.url_key("x/a.pdf", 11786) != want and 0 <= want < 2 ** 64


def test_language_rules():
    en = " ".join(["the cat is in the house and the dog is with the child of the man"] * 3)
    es = " ".join(["el gato se sienta sobre la alfombra mientras el perro duerme dentro de la casa"] * 3)
    assert T.english_like(en) and not T.english_like(es)
    assert not T.english_like("the of and to in is that for with")            # fewer than 20 words
    # L3: an English page labeled por_Latn is ignored, a Spanish page counts; unknown and zxx pages are neutral
    text = en + es
    ends = [len(en), len(text)]
    assert T.nonenglish_chars(text, ends, ["eng_Latn", "spa_Latn"]) == (len(es), len(text))
    assert T.nonenglish_chars(en + en, [len(en), 2 * len(en)], ["eng_Latn", "por_Latn"]) == (0, 2 * len(en))
    assert T.nonenglish_chars(text, ends, ["eng_Latn", "unknown"])[0] == 0
    assert T.nonenglish_chars(text, ends, ["eng_Latn", "zxx_Latn"])[0] == 0
    assert T.nonenglish_chars(text, ends, ["eng_Latn", None])[0] == 0
    # page_ends past the text are clipped; non-monotone ends attribute nothing twice
    assert T.nonenglish_chars(es, [10, 10 ** 6], ["eng_Latn", "spa_Latn"]) == (len(es) - 10, len(es))
    assert T.nonenglish_chars(es, [30, 20, len(es)], ["spa_Latn", "spa_Latn", "spa_Latn"]) == (len(es), len(es))
    # the 10 % threshold
    pad = "x" * 9 * len(es)
    ok, _ = T.text_rules(pad + es, "eng_Latn", ["eng_Latn", "spa_Latn"], [len(pad), len(pad) + len(es)], [2.0], 1.5)
    assert ok["L3"]
    ok, _ = T.text_rules(pad[:-1] + es, "eng_Latn", ["eng_Latn", "spa_Latn"], [len(pad) - 1, len(pad) - 1 + len(es)],
                         [2.0], 1.5)
    assert not ok["L3"]
    # L1 / L2: the card's keep_document; E: max of the scores, at the threshold
    def rules(labels, lid="eng_Latn", scores=(2.0,)):
        return T.text_rules(en, lid, labels, [len(en)] * len(labels), list(scores), 1.5)[0]
    assert rules(["eng_Latn", "eng_Latn", "spa_Latn", "spa_Latn"])["L2"]
    assert rules(["eng_Latn", "spa_Latn", "spa_Latn"])["L2"]
    assert not rules(["spa_Latn", "spa_Latn", "eng_Latn", "spa_Latn", "spa_Latn"])["L2"]
    assert rules([])["L2"] and not rules(["eng_Latn"], lid="spa_Latn")["L1"]
    assert rules(["eng_Latn"], scores=(1.2, 1.5))["E"] and not rules(["eng_Latn"], scores=(1.49999,))["E"]
    assert not rules(["eng_Latn"], scores=())["E"]


def test_quality_rules():
    # G3: the function-word rate, NBSP-separated words included; Unicode word characters block a match
    nb = " ".join(["the", "river", "of", "time", "and", "space"] * 10)
    assert T.fw_rate(nb) == pytest.approx(0.5) and T.text_rules(nb, "eng_Latn", [], [], [2.0], 1.5)[0]["G3"]
    assert T.fw_rate("théory ofé andy") == 0
    assert T.fw_rate("The OF And") == 1.0
    words = ["river"] * 98 + ["the", "of"]
    assert T.text_rules(" ".join(words), "eng_Latn", [], [], [2.0], 1.5)[0]["G3"]
    assert not T.text_rules(" ".join(["river"] * 99 + ["the"]), "eng_Latn", [], [], [2.0], 1.5)[0]["G3"]
    # G1: characters of repeated lines, at 50 %
    line = "abcdefghij"                                  # 10 characters
    text = "\n".join([line] * 3 + ["k" * 19])           # 3 x 10 + 19 + 3 newlines = 52; 20 repeated <= 26
    assert T.dup_line_chars(text) == 20 and T.text_rules(text, "eng_Latn", [], [], [2.0], 1.5)[0]["G1"]
    text = "\n".join([line] * 4 + ["k" * 5])            # 49 characters, 30 repeated > 24.5
    assert not T.text_rules(text, "eng_Latn", [], [], [2.0], 1.5)[0]["G1"]
    assert T.dup_line_chars("  a b \n\na b\n \n") == 3     # stripped, empty lines ignored
    exact = "ab\nab\nab"                                 # 8 characters, 4 repeated: exactly 50 % passes
    assert T.dup_line_chars(exact) == 4 and T.text_rules(exact, "eng_Latn", [], [], [2.0], 1.5)[0]["G1"]
    assert not T.text_rules(exact + "\nab", "eng_Latn", [], [], [2.0], 1.5)[0]["G1"]     # 6 of 11
    # G2: ASCII letters >= 30 %
    assert T.ascii_letters("aé1Z") == 2
    assert T.text_rules("abc" + "1" * 7, "eng_Latn", [], [], [2.0], 1.5)[0]["G2"]                 # exactly 30 %
    assert not T.text_rules("abc" + "1" * 8, "eng_Latn", [], [], [2.0], 1.5)[0]["G2"]
    assert not T.text_rules("", "eng_Latn", [], [], [2.0], 1.5)[0]["G2"]


def test_special_literals_stay_text(synth, monkeypatch):
    """split_special_tokens: a literal <s>, </s> or <unk> in a document is text, never the id 0, 1 or 2."""
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")          # T._init sets both; restored after the test
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    tok = synth["tok"]
    text = "the <s> river </s> of <unk> water"
    assert not {0, 1, 2} & set(_encode(tok, [text])[0])
    assert {0, 1, 2} <= set(tok(text, add_special_tokens=False)["input_ids"])   # what stage1's call would give
    T._init(str(synth["dir"] / "tok"), 1)
    assert T.encode([text]) == _encode(tok, [text])


def test_special_literals_llama2(tok_dir, monkeypatch):
    """The same with the real Llama-2 tokenizer (skipped where it is not installed)."""
    monkeypatch.setenv("RAYON_NUM_THREADS", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    T._init(str(tok_dir), 1)
    texts = ["Use <s> and </s> as markers; <unk> is unknown.", "<s></s><unk>", "a<s>b</s>c<unk>d"]
    for t, ids in zip(texts, T.encode(texts)):
        assert ids and not {0, 1, 2} & set(ids), (t, ids)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    assert 1 in tok(texts[1], add_special_tokens=False)["input_ids"]
    plain = "The quick brown fox jumps over the lazy dog."
    assert T.encode([plain])[0] == tok(plain, add_special_tokens=False)["input_ids"]


# ---- the steps on the synthetic corpus -----------------------------------------------------------------------------
def test_layout_matches_stage1(synth, tmp_path):
    """On a file whose documents all pass, the task shards are byte for byte the stream of data/stage1_tokenize.py."""
    rng = np.random.default_rng(7)
    docs = [_good(rng, 0, i) for i in range(23)]
    raw = tmp_path / "raw"
    write_parquet(str(raw / REL[0]), docs, row_group_size=4)
    make_listing(str(raw), REL[:1], tmp_path / "files.json")
    tok = word_tokenizer([x["text"] for x in docs], tmp_path / "tok")
    p = _py(DATA / "finepdfs_tokenize.py", "--raw", raw, "--files-json", tmp_path / "files.json", "--tokenizer",
            tmp_path / "tok", "--work", tmp_path / "work", "--workers", 2, "--threads-per", 1)
    assert p.returncode == 0, p.stderr[-3000:]
    ours = b"".join((tmp_path / "work" / "shards" / f"task_{t:04d}.bin").read_bytes() for t in range(4))
    lens = np.concatenate([np.load(tmp_path / "work" / "shards" / f"task_{t:04d}.lengths.npy") for t in range(4)])
    (tmp_path / "s1" / "c").mkdir(parents=True)
    os.link(raw / REL[0], tmp_path / "s1" / "c" / "part_000000.parquet")
    p = _py(DATA / "stage1_tokenize.py", "--raw-dir", tmp_path / "s1", "--configs", "c", "--tokenizer", tmp_path / "tok",
            "--out", tmp_path / "s1" / "out", "--workers", 1, "--threads-per", 1)
    assert p.returncode == 0, p.stderr[-3000:]
    assert ours == (tmp_path / "s1" / "out.bin").read_bytes()
    assert np.array_equal(lens, np.load(tmp_path / "s1" / "out.doc_lengths.npy"))
    want = [[1] + ids + [2] for ids in _encode(tok, [x["text"] for x in docs])]
    assert np.array_equal(np.frombuffer(ours, np.uint16), np.concatenate(want))


def test_rule_accounting(built):
    """Each planted document is dropped by its own rule (cascade), the counted one is tokenized but not in a shard."""
    w = json.loads((built["work"] / "tokenize_summary.json").read_text())
    casc = {r: v["docs"] for r, v in w["cascade"].items()}
    assert casc == {"L1": 1, "L2": 1, "E": 2, "L3": 1, "G1": 1, "G2": 1, "G3": 1, "G4": 1}, casc
    assert w["cascade"]["G4"]["l2_tokens"] > 0 and "l2_tokens" not in w["cascade"]["L1"]
    sa = {r: v["docs"] for r, v in w["standalone"].items()}
    assert sa["L1"] == 1 and sa["E"] == 2 and all(sa[r] >= casc[r] for r in sa)
    assert w["totals"]["docs"] == 120
    import pyarrow.parquet as pq
    docs = pq.read_table([str(p) for p in sorted((built["work"] / "shards").glob("task_*.docs.parquet"))]).to_pylist()
    ids = {x["id"]: x for x in docs}
    for gone in ("L1", "L2", "E", "L3", "G1", "G2", "G3", "G4"):
        assert gone not in ids, gone
    assert ids["count"]["in_shard"] is False and ids["count"]["tok_off"] == -1 and ids["count"]["edu"] == 1.45
    for kept in ("por", "literal", "overshoot"):
        assert ids[kept]["in_shard"], kept
    assert ids["literal"]["special_literal"] and w["special_literal_docs"] == 1
    thr = built["manifest"]["pool_by_edu_threshold"]
    assert thr["1.4"]["docs"] == thr["1.45"]["docs"] == thr["1.5"]["docs"] + 1
    assert thr["1.5"]["tokens"] == built["manifest"]["pool"]["tokens"]


def test_selection_matches_recomputation(built, synth):
    """The keyed whole-group sampling, the id dedup and the targets, against an independent re-computation; the
    streams hold exactly [BOS] + ids + [EOS] of the selected documents, train in source order, val in key order."""
    import pyarrow.parquet as pq
    out = built["out"]
    pool = expected_pool(synth)
    val, train, vt, tt = expected_selection(pool, SEED, TRAIN_T, VAL_T)
    sel = pq.read_table(out / "selection.parquet").to_pylist()
    got_v = [(x["file"], x["row"]) for x in sel if x["split"] == "val"]
    got_t = [(x["file"], x["row"]) for x in sel if x["split"] == "train"]
    assert got_v == [(x["file"], x["row"]) for x in val]
    assert got_t == [(x["file"], x["row"]) for x in train]
    m = built["manifest"]
    assert m["val"]["tokens"] == vt >= VAL_T and m["train"]["tokens"] == tt >= TRAIN_T
    assert 0 <= m["train"]["overshoot"] < m["train"]["last_group_tokens"]
    for split, docs in (("train", train), ("val", val)):
        toks = np.fromfile(out / f"finepdfs_edu_en_{split}.bin", np.uint16)
        assert np.array_equal(toks, np.concatenate([[1] + x["ids"] + [2] for x in docs]))
        lens = np.load(out / f"finepdfs_edu_en_{split}.doc_lengths.npy")
        assert lens.dtype == np.int64 and lens.tolist() == [x["n"] for x in docs] and int(lens.sum()) == len(toks)
    # dedup: "dup" stays as the longer copy (file 1), "tie" as the lower (file, row)
    rows = {x["id"]: (x["file"], x["row"]) for x in sel}
    if "dup" in rows:
        assert rows["dup"] == (1, 1)
    if "tie" in rows:
        assert rows["tie"] == (1, 2)
    assert m["dedup"]["docs_dropped"] == 2
    # http://x.org/a.pdf and https://WWW.X.org/a.pdf are one group, so one split; train and val share no id or URL
    sp = {x["id"]: x["split"] for x in sel}
    if "urlA" in sp or "urlB" in sp:
        assert sp.get("urlA") == sp.get("urlB")
    tv = [x for x in sel if x["split"] == "val"]
    tr = [x for x in sel if x["split"] == "train"]
    assert not {x["id"] for x in tv} & {x["id"] for x in tr}
    assert not {x["norm_url"] for x in tv} & {x["norm_url"] for x in tr}
    assert [x["key"] for x in tv] == sorted(x["key"] for x in tv)


def test_url_group_and_duplicates_are_selected():
    """The planted partners always land together: with the targets above the whole pool of their keys is used, so
    check the grouping rule on its own (select_groups) with two norm_urls of one site."""
    key = np.array([5, 5, 3, 9], dtype=np.uint64)
    url = np.array([1, 1, 0, 2])
    order, n_val, n_train, info = S.select_groups(key, url, np.arange(4), np.zeros(4, np.int64), np.arange(4),
                                                  np.array([10, 10, 10, 10]), train_target=15, val_target=5)
    assert order.tolist() == [2, 0, 1, 3] and n_val == 1 and n_train == 2       # the two docs of url 1 stay together
    assert info["train_tokens"] == 20 and info["train_overshoot"] == 5 and info["train_last_group_tokens"] == 20


def test_select_targets_overshoot_nesting_and_failure():
    rng = np.random.default_rng(3)
    n = 2000
    url = rng.integers(0, 1500, n)                       # shared URLs make multi-document groups
    key = np.array([T.url_key(str(u), 1) for u in url], dtype=np.uint64)
    ntok = rng.integers(52, 3000, n)
    args = (key, url, np.arange(n), np.zeros(n, np.int64), np.arange(n), ntok)
    order, n_val, n_train, info = S.select_groups(*args, train_target=400_000, val_target=50_000)
    assert 400_000 <= info["train_tokens"] < 400_000 + info["train_last_group_tokens"]
    assert 50_000 <= info["val_tokens"] < 50_000 + info["val_last_group_tokens"]
    tr = order[n_val:n_val + n_train]
    assert int(ntok[tr].sum()) == info["train_tokens"] and int(ntok[order[:n_val]].sum()) == info["val_tokens"]
    assert not set(url[tr]) & set(url[order[:n_val]])
    # a smaller train target selects a prefix of the same order (nested subsets)
    o2, v2, t2, _ = S.select_groups(*args, train_target=100_000, val_target=50_000)
    assert v2 == n_val and t2 < n_train and np.array_equal(o2[:v2 + t2], order[:v2 + t2])
    with pytest.raises(ValueError, match="training target"):
        S.select_groups(*args, train_target=int(ntok.sum()), val_target=50_000)
    with pytest.raises(ValueError, match="validation target"):
        S.select_groups(*args, train_target=10, val_target=int(ntok.sum()) + 1)


def test_dedup_rule():
    id_rank = np.array([0, 1, 0, 2, 2, 0])
    ntok = np.array([10, 5, 30, 7, 7, 30])
    file = np.array([0, 0, 1, 0, 1, 2])
    row = np.array([0, 1, 0, 2, 3, 0])
    assert S.dedup_ids(id_rank, ntok, file, row).tolist() == [1, 2, 3]   # id 0: 30 tokens, file 1 < 2; id 2: tie -> (0, 2)


def test_val_coverage():
    c = S.val_coverage([8192 * 1000 + 1, 8192 * 1500], seq_len=8192, windows=2000)
    assert c["docs"] == 2 and c["windows"] == 2000
    assert c["largest_doc_share"] == pytest.approx(8192 * 1000 / (8192 * 2000), abs=1e-4)
    assert S.val_coverage([100], seq_len=8192)["windows"] == 0


def test_outputs_independent_of_workers_and_out_name(built, synth, tmp_path):
    """One worker instead of two, and another --out-name: the same bytes, keys and selection."""
    p = _py(DATA / "build_finepdfs_edu.py", *_build_args(synth, tmp_path / "root", tmp_path / "work", "--step", "tokenize",
                                                         "--step", "select", "--step", "write", "--step", "publish",
                                                         out_name="other_name", workers=1), cwd=tmp_path)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    for nm in B.PUBLISHED:
        assert (tmp_path / "root" / "other_name" / nm).read_bytes() == (built["out"] / nm).read_bytes(), nm
    m2 = json.loads((tmp_path / "root" / "other_name" / "manifest.json").read_text())
    assert m2["outputs"] == built["manifest"]["outputs"]


def test_manifest_and_loader(built):
    m = built["manifest"]
    for k in ("source", "filters", "dedup", "url_groups", "special_literal_docs", "sampling", "targets", "pool", "train",
              "val", "unused", "pool_by_edu_threshold", "val_coverage", "tokenizer", "software", "build", "outputs"):
        assert k in m, k
    assert m["source"]["files_used"] == 3 and len(m["source"]["files"]) == 3 and m["source"]["files"][0]["sha256"]
    assert m["filters"]["order"] == ["L1", "L2", "E", "L3", "G1", "G2", "G3", "G4"]
    assert m["sampling"]["seed"] == SEED and m["sampling"]["namespace"] == "finepdfs-edu-en"
    assert m["targets"] == {"train_tokens": TRAIN_T, "val_tokens": VAL_T}
    assert m["pool"]["tokens"] == m["train"]["tokens"] + m["val"]["tokens"] + m["unused"]["tokens"]
    assert set(m["build"]["scripts"]) >= {"build_finepdfs_edu.py", "finepdfs_tokenize.py", "finepdfs_select.py"}
    assert m["format"]["invariants"]["train"]["bos_exactly_at_starts"] and m["tokenizer"]["bos"] == 1
    assert set(m["outputs"]) == set(B.PUBLISHED)
    assert "verify PASS" in built["log"] and (built["out"] / "README.md").exists()
    # the training loader reads it (a few batches)
    sys.path.insert(0, str(REPO / "src"))
    from pretrain.data.doc_dataset import DocAwareMemmapDataset
    from pretrain.data.data_loader import build_train_dataloader
    binp, lenp = str(built["out"] / "finepdfs_edu_en_train.bin"), str(built["out"] / "finepdfs_edu_en_train.doc_lengths.npy")
    ds = DocAwareMemmapDataset(binp, 64, lenp, "uint16", "packed")
    assert len(ds) == m["train"]["tokens"] // 64 and int(ds[0]["input_ids"][0]) == 1
    dl = build_train_dataloader({"paths": binp, "doc_lengths": lenp, "dtype": "uint16", "position_mode": "packed",
                                 "seed": SEED, "num_workers": 0, "pin_memory": False}, max_seq_len=64, global_batch_size=4)
    it = iter(dl)
    for _ in range(3):
        b = next(it)
        assert tuple(b["input_ids"].shape) == (4, 64) and int(b["input_ids"].max()) < m["tokenizer"]["vocab"]


def test_publish_refuses_and_checksums(built, synth, tmp_path, monkeypatch):
    """A published corpus is not replaced without --force; the first verified default build writes
    finepdfs_edu_checksums.json, and later default builds are checked against it (here with the module's defaults
    patched to the synthetic build's targets)."""
    base = _build_args(synth, tmp_path / "root", built["work"], "--step", "publish", out_name="finepdfs_edu_en_60b")
    base[base.index("--checksums") + 1] = tmp_path / "cks.json"
    base = [str(x) for x in base]
    vargs = list(base)
    vargs[vargs.index("publish")] = "verify"
    monkeypatch.setattr(B, "TRAIN_TOKENS", float(TRAIN_T))
    monkeypatch.setattr(B, "VAL_TOKENS", float(VAL_T))
    B.main(base)
    assert not (tmp_path / "cks.json").exists()                      # published, not yet verified
    B.main(vargs)
    cks = json.loads((tmp_path / "cks.json").read_text())
    assert set(cks["files"]) == {f"finepdfs_edu_en_60b/{nm}" for nm in B.STREAMS}
    assert cks["build"]["train_tokens"] == TRAIN_T and cks["build"]["seed"] == SEED
    assert cks["files"]["finepdfs_edu_en_60b/finepdfs_edu_en_train.bin"] == built["manifest"]["outputs"]["finepdfs_edu_en_train.bin"]
    with pytest.raises(SystemExit, match="exists"):
        B.main(base)
    B.main(base + ["--force"])                                        # same bytes: matches the reference
    assert any(x.startswith(".finepdfs_edu_en_60b.replaced-") for x in os.listdir(tmp_path / "root"))
    B.main(vargs)                                                     # checked against the reference, not rewritten
    assert json.loads((tmp_path / "cks.json").read_text()) == cks
    bad = json.loads((tmp_path / "cks.json").read_text())
    bad["files"]["finepdfs_edu_en_60b/finepdfs_edu_en_val.bin"]["sha256"] = "0" * 64
    (tmp_path / "cks.json").write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="differ from"):
        B.main(base + ["--force"])
    with pytest.raises(SystemExit, match="differ"):
        B.main(vargs)
    # verify catches a corrupted published file
    (tmp_path / "cks.json").write_text(json.dumps(cks))
    f = tmp_path / "root" / "finepdfs_edu_en_60b" / "finepdfs_edu_en_val.bin"
    raw = bytearray(f.read_bytes())
    raw[10] ^= 1
    f.write_bytes(bytes(raw))
    with pytest.raises(SystemExit, match="differ"):
        B.main(vargs)
    # not the default build (another name): no reference written or read
    other = [x if x != "finepdfs_edu_en_60b" else "other" for x in base]
    other[other.index("--checksums") + 1] = str(tmp_path / "cks_other.json")
    B.main(other)
    o2 = list(other)
    o2[o2.index("publish")] = "verify"
    B.main(o2)
    assert not (tmp_path / "cks_other.json").exists()


def test_check_stream_catches_bad_layout(tmp_path):
    good = np.array([1, 5, 6, 2, 1, 7, 2], dtype=np.uint16)
    lens = np.array([4, 3], dtype=np.int64)
    good.tofile(tmp_path / "g.bin")
    r = S.check_stream(str(tmp_path / "g.bin"), lens, vocab=10, bos=1, eos=2, unk=0)
    assert r["bos_exactly_at_starts"] and r["eos_exactly_at_ends"] and r["tokens"] == 7
    for bad, msg in (([1, 5, 1, 2, 1, 7, 2], "BOS"), ([1, 5, 6, 2, 1, 2, 2], "EOS"), ([1, 0, 6, 2, 1, 7, 2], "unk"),
                     ([1, 5, 12, 2, 1, 7, 2], "vocab")):
        np.array(bad, dtype=np.uint16).tofile(tmp_path / "b.bin")
        with pytest.raises(AssertionError, match=msg):
            S.check_stream(str(tmp_path / "b.bin"), lens, vocab=10, bos=1, eos=2, unk=0)
    with pytest.raises(AssertionError, match="bytes"):
        S.check_stream(str(tmp_path / "g.bin"), np.array([4, 4], dtype=np.int64), vocab=10, bos=1, eos=2, unk=0)


def test_download_verifies_present_files(synth, tmp_path):
    """The download step only checks files that are already there (no network): a good file gets its marker, a
    corrupted one is deleted and reported."""
    import shutil
    raw = tmp_path / "raw"
    shutil.copytree(synth["dir"] / "raw", raw)
    listing = json.loads((synth["dir"] / "files.json").read_text())
    B.download(listing, [0, 1, 2], str(raw), B.REVISION, 2, str(tmp_path))
    assert all((raw / (r + ".sha256ok")).exists() for r in REL)
    s = json.loads((tmp_path / "download_summary.json").read_text())
    assert s["attempts"] == 1 and s["files"] == {}
    p = raw / REL[1]
    data = bytearray(p.read_bytes())
    data[100] ^= 1
    p.write_bytes(bytes(data))
    (raw / (REL[1] + ".sha256ok")).unlink()
    assert not B._check_and_mark(str(p), listing["files"][1]) and not p.exists()


@pytest.mark.parametrize("script", ["build_finepdfs_edu.py", "finepdfs_tokenize.py", "finepdfs_select.py"])
def test_scripts_start(script, tmp_path):
    p = _py(DATA / script, "--help", cwd=tmp_path)
    assert p.returncode == 0 and "usage" in p.stdout, p.stderr[-2000:]


def test_files_listing():
    m = json.loads((DATA / "finepdfs_edu_files.json").read_text())
    assert m["repo"] == B.REPO_ID and m["revision"] == B.REVISION and m["config"] == "eng_Latn"
    assert [f["path"] for f in m["files"]] == [f"data/eng_Latn/train/000_{i:05d}.parquet" for i in range(100)]
    assert m["totals"] == {"files": 100, "bytes": sum(f["bytes"] for f in m["files"]),
                           "rows": sum(f["rows"] for f in m["files"]), "row_groups": sum(f["row_groups"] for f in m["files"])}
    assert m["totals"]["bytes"] == 298_670_229_506 and m["totals"]["rows"] == 23_023_372
    assert all(len(f["sha256"]) == 64 for f in m["files"])
