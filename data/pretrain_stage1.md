# Stage 1: the pretraining corpus (Nemotron-CC-v2.1 HQ + HQ-DQA, Llama-2 32k)

Every stage-1 run of the paper (every arm of `scripts/experiments/stage1_*.sh`) trains on the same token stream in the
same order: 28,610 steps of 128 x 8,192 tokens (30.0 B tokens) drawn from the training stream below, and every
validation CE is measured on the held-out stream. This page is the recipe; `data/build_stage1.py` runs it.

## Result

`<root>/nemotron_cc_v2d1_hq_dqa/` (`<root>` = `DM_DATA_ROOT`):

| file | content | size | sha256 |
|---|---|---|---|
| `nemotron_rs_train.bin` | the training stream: uint16 Llama-2 ids, documents back to back | 39,660,504,515 tokens, 79,321,009,030 bytes | `29de1fee...82bdb4` |
| `nemotron_rs_train.doc_lengths.npy` | int64, one length per document, sums to the token count | 36,239,772 documents, 289,918,304 bytes | `014f4a4b...c5ec7c` |
| `nemotron_rs_val.bin` | the held-out stream (2 % of the documents, shuffled) | 807,013,275 tokens, 1,614,026,550 bytes | `c3dbf223...5f5c35` |
| `nemotron_rs_val.doc_lengths.npy` | its document lengths | 740,467 documents, 5,923,864 bytes | `ed1363aa...53ed4b` |

The full hashes are in `data/checksums.json`; `python data/verify.py --root <root> --part stage1` checks them.

## Recipe

1. **Source.** The Hugging Face dataset `nvidia/Nemotron-CC-v2.1` at revision
   `ba6f2aaef7ada865bb08fc08640ca292150097db`, two buckets of its English raw-text parquet (one document per row, in the
   `text` column):
   - `High-Quality/part_000000.parquet` .. `part_000019.parquet` (20 files, 46.8 GB): organic English Common Crawl text
     from three 2025 snapshots, 29,173,164 documents;
   - `High-Quality-DQA/part_000000.parquet` .. `part_000005.parquet` (6 files, 11.7 GB): synthetic STEM question-answer
     text (the "diverse QA" rephrasings of Essential-Web STEM documents by Qwen3-30B-A3B), 7,807,075 documents.
2. **Tokenizer.** The Llama-2 tokenizer (`meta-llama/Llama-2-7b-hf`, revision
   `01c7f73d771dfac7d292323805ebc428287df4f9`, 32,000 ids, BOS = 1, EOS = 2), loaded with transformers'
   `AutoTokenizer` (the fast tokenizer); `data/fetch_tokenizer.py` puts it at `<root>/tokenizer/llama2_32k`.
3. **Tokenize** (`data/stage1_tokenize.py`). Each document's text is encoded without special tokens and written as
   `[BOS] + ids + [EOS]`; a document that encodes to nothing is dropped. The parts are processed in a fixed order,
   `High-Quality` then `High-Quality-DQA`, each in file-name order, and the documents keep their order within a part,
   so the result does not depend on the parallelism. Result: one stream of 40,467,517,790 tokens in 36,980,239
   documents (every document of both buckets; mean 1,094 tokens, median 768, max 4,066,685), and its length sidecar.
   10.8 % of the tokens are in documents longer than 8,192 tokens.
4. **Split** (`data/stage1_split.py`). A document-level holdout: `numpy.random.RandomState(11785)` draws one uniform
   number per document and a document with a draw below 0.02 goes to the held-out set; the held-out documents are then
   shuffled with the same generator (`permutation`), the training documents keep their order. A contiguous tail split
   would have put only DQA documents in the held-out set; the random split samples every region of the stream, and
   the shuffle makes any window range of the held-out stream representative (the val-CE protocol scores its first
   2,000 windows of 8,192 tokens, `docs/benchmark.md`).

No deduplication, filtering or reordering beyond this is applied. Packing is naive: the training loader cuts the stream
into 8,192-token windows regardless of document boundaries (`position_mode: packed`, no intra-document attention
mask), and `data.seed 11785` fixes the window order of every run.

## Commands

    export DM_DATA_ROOT=$PWD/workspace/data
    python data/fetch_tokenizer.py --root $DM_DATA_ROOT      # once (gated: accept Meta's license, hf auth login)
    python data/build_stage1.py --root $DM_DATA_ROOT         # download, tokenize, split, verify

`--step download|tokenize|split|verify` runs one step; `--raw DIR` puts the parquet elsewhere (default
`<root>/raw/nemotron-cc-v2.1`); `--delete-raw` removes it after a verified build; `--keep-stream` keeps the unsplit
stream (`nemotron_hqdqa.bin`). Needs about 220 GB of free disk during the build (58.5 GB of parquet, 81 GB for the
tokenized stream, 81 GB for the split) and 81 GB after it. On a 208-CPU node the build takes 17 minutes (2 to download,
14 to tokenize, 1 each to split and check).

The build is exact: with the locked environment (`env/requirements-cu130-hopper.lock.txt`: transformers 5.9.0,
tokenizers 0.22.2, pyarrow 24.0.0, numpy 2.4.6) it reproduces the four files above bit for bit, which the last step
checks. A different tokenizers version can change the ids of rare inputs; the checksums will then say so.

## Terms

`nvidia/Nemotron-CC-v2.1` is gated and governed by the NVIDIA Data Agreement for Model Training: the data may be used to
train models internally and may not be distributed, published or made available to others, in any form (a tokenized
copy included). Every user must accept the agreement and build the corpus from the source. Models trained on it and
their outputs are the trainer's own. The synthetic DQA subset was generated with Qwen3-30B-A3B (Apache-2.0).
