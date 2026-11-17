# FinePDFs-Edu English: a 60 B-token pretraining corpus (Llama-2 32k)

An alternative stage-1 corpus: 60 B Llama-2 tokens of English, educational PDF text, sampled at random (whole
documents) from the Hugging Face dataset HuggingFaceFW/finepdfs-edu after English and extraction-quality filters, in
exactly the format of the stage-1 Nemotron streams (`pretrain_stage1.md`), so that any stage-1 arm trains on it with two
flags and no code change. This page is the recipe; `data/build_finepdfs_edu.py` runs it and writes a `manifest.json`
with the full provenance next to the streams.

## Result

`<root>/finepdfs_edu_en_60b/` (`<root>` = `DM_DATA_ROOT`):

| file | content | size |
|---|---|---|
| `finepdfs_edu_en_train.bin` | the training stream: uint16 Llama-2 ids, documents back to back, in source order (file, then row) | >= 60,000,000,000 tokens (~120.0 GB), ~10.2 M documents |
| `finepdfs_edu_en_train.doc_lengths.npy` | int64, one length per document, sums to the token count | ~83 MB |
| `finepdfs_edu_en_val.bin` | the held-out stream, in random (key) order | >= 800,000,000 tokens (~1.6 GB), ~0.14 M documents |
| `finepdfs_edu_en_val.doc_lengths.npy` | its document lengths | ~1 MB |
| `selection.parquet` | one row per selected document: split, file, row, id, norm_url, key, n_tokens, edu | ~0.5-1 GB |
| `manifest.json` | provenance: source revision and files, every rule with its drops, dedup, URL groups, seed and key, targets and what was reached, exact pool sizes per edu threshold, tokenizer and software versions, script hashes, job, times, output sha256s | |
| `README.md` | a short description and the terms | |

The exact token and document counts and sha256s of a build are in its `manifest.json`; the first default build writes
the sizes and sha256s of the four stream files to `data/finepdfs_edu_checksums.json` (kept apart from
`data/checksums.json`), and every later default build is checked against it (`--step verify` checks a built corpus
again).

The default build of 2026-11-09 (job 21596272) holds 60,000,005,485 training tokens in 10,162,083 documents and
800,016,252 held-out tokens in 136,211 documents. They were drawn from a pool of 93,914,362,910 tokens, 1.54x the
need. The corpus's own `README.md` gives its counts, its filter drops and the checks run on it.

## Recipe

1. **Source.** HuggingFaceFW/finepdfs-edu at revision `9cfabe2127faca99b3d5c4dc6d1fcb397399ebde` (2025-11-11, the
   repository's only commit: it is squashed on update, so `--revision` can point at a newer one; the files are then
   still checked against the pinned sha256s), config `eng_Latn` (the only English one of the 69; train split only):
   all 100 files `data/eng_Latn/train/000_00000.parquet` .. `000_00099.parquet`, 298.7 GB, 23,023,372 documents.
   `data/finepdfs_edu_files.json` pins each file's path, bytes, LFS sha256, rows and row groups. The set is the top
   10 % of FinePDFs English by an educational classifier (ModernBERT-large, trained on Qwen3-235B-A22B scores of the
   FineWeb-Edu prompt; max of the scores of the first and the last 2,048 tokens, `fw_edu_scores`); the base card is
   HuggingFaceFW/finepdfs at `220bac3acbf07789502c621d2d33952f51ac7f86`. Every file is used: the pool is only ~1.6x
   the need, and no shard-selection bias.
2. **Per-document rules** (`data/finepdfs_tokenize.py`), in this order; the manifest counts the drops of each rule in
   this order (cascade) and stand-alone, in documents, characters and the dataset's `token_count`, plus Llama-2 tokens
   where known (G4):

   | rule | keeps a document when | why |
   |---|---|---|
   | L1 | `full_doc_lid == eng_Latn` | with L2, the dataset card's `keep_document()` for English |
   | L2 | #pages labeled `eng_Latn` >= #pages // 2 (`per_page_languages`) | mostly drops table-dominated documents whose pages GlotLID calls `unknown` |
   | E | `edu = max(fw_edu_scores) >= 1.5` (`--edu-min`; 1.55 and 1.6 are the stricter options) | the top ~8 % of FinePDFs English (the -edu set starts at 1.32) |
   | L3 | the non-English pages hold <= 10 % of the attributed characters | code-switching (bilingual documents); see below |
   | G1 | the characters of repeated lines (stripped, non-empty; `sum len(line) * (count - 1)`) <= 50 % | extraction junk: running headers, table fill |
   | G2 | ASCII letters >= 30 % of the characters | numeric tables, garbled OCR |
   | G3 | function-word rate >= 2 %: matches of `(?<!\w)(the\|of\|and\|to\|in\|is\|that\|for\|with\|as)(?!\w)` in `text.lower()` over `len(text.split())` | not running English prose |
   | G4 | >= 50 Llama-2 content tokens (after tokenization) | fragments |

   L3: pages are cut from the text at `page_ends` clipped to `len(text)` (`prev = max(prev, min(end, n))`); a page is
   non-English only if its label is not `eng_Latn`, not `unknown`, not `zxx_*`, and its text is not English-like
   (English-like: >= 20 ASCII words `[A-Za-z]+`, >= 8 % of them, lowercased, in FW28 = the, of, and, to, in, is, that,
   for, with, as, are, be, this, by, on, or, an, it, from, at, which, was, were, not, can, will, have, has): GlotLID's
   page labels are noisy, and an English page labeled `por_Latn` is not dropped for it. No length cap, no filter on
   minhash, truncation or extractor, no text rewriting.
   Measured by this pipeline on 15 of the 100 files (a 12-file trial build, files 0-11, 2,841,229 documents, plus the
   3-file planning sample 23/44/86, 686,309 documents), as shares of the dataset's `token_count`:

   | | L1 | L2 | E | L3 | G1 | G2 | G3 | G4 |
   |---|---|---|---|---|---|---|---|---|
   | dropped in this order (12 files) | 1.58 % | 3.11 % | 32.13 % | 2.11 % | 0.27 % | 0.72 % | 0.05 % | 0.00 % |
   | failing the rule at all (12 files) | 1.58 % | 3.72 % | 33.86 % | 4.94 % | 0.69 % | 3.15 % | 1.40 % | |

   The pool keeps 69 % of the documents: ~96.7 B Llama-2 tokens (with BOS/EOS) scaled to the 100 files (12 files:
   97.6 B, 3 files: 94.0 B), 1.59x the 60.8 B needed; token-weighted edu 2.26; 23.4 % of its tokens in documents over
   100k tokens and 2.9 % over 1M; 65 % of its tokens from rolmOCR (OCR'd PDFs), the rest docling; ~0.5 % of the
   characters in residual non-English text (planning estimate). Exact pool sizes by edu threshold (12 files, scaled to
   100): 1.4: 118.9 B, 1.45: 107.5 B, 1.5: 97.6 B, 1.55: 87.8 B, 1.6: 80.5 B, 1.65: 73.1 B; the build reports them
   exactly for the whole config (documents down to 1.4 are tokenized for that: ~22 % more tokenization than the pool).
3. **Tokenizer and layout.** The Llama-2 tokenizer (`meta-llama/Llama-2-7b-hf` at
   `01c7f73d771dfac7d292323805ebc428287df4f9`, 32,000 ids, BOS 1, EOS 2; `data/fetch_tokenizer.py`), checked against
   `data/checksums.json` before the step runs, and called as
   `AutoTokenizer.from_pretrained(dir)(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]`.
   That is the stage-1 call plus `split_special_tokens`: a literal `<s>`, `</s>` or `<unk>` in a PDF (~0.004 % of the
   documents) is encoded as text, never as the id 0, 1 or 2; for every other document the ids are those of
   `data/stage1_tokenize.py`. Each document is written as `[BOS] + ids + [EOS]` (`doc_lengths[i] = len(ids) + 2`).
4. **Dedup and URL groups** (`data/finepdfs_select.py`). Exact-id dedup across all files (of the documents sharing an
   `id`, the one with the most tokens stays; ties go to the lower (file, row)). `norm_url` = `url.strip()` without
   `http://` or `https://` (any case), the host lowercased and a leading `www.` removed (path and query unchanged);
   the documents of one `norm_url` form a group that always goes to one split (~1.8 % of the documents share a URL
   across the 100 files, a planning estimate; 0.27 % within the 12 trial files).
5. **Sampling.** key = BLAKE2b-64(`b'finepdfs-edu-en|11785|' + norm_url`), read big-endian (11785 = `--seed`). The pool
   is sorted by (key, norm_url, id, file, row); the validation set takes whole groups in that order until it holds
   >= 800,000,000 tokens, the training set the following whole groups until it holds >= 60,000,000,000 tokens, and the
   rest is unused. So 60e9 <= train < 60e9 + the tokens of its last group (the expected overshoot is ~6e4 tokens, the
   bound the largest group, a few M tokens), a smaller `--train-tokens` selects a prefix of the same order (nested
   subsets), and the build stops if the pool cannot reach both targets. Train and val share no document and no URL;
   the outputs do not depend on the number of processes. Train is written in source order (as the stage-1 training
   stream), val in key order (already shuffled: any window range of it samples the whole set).
6. **Checks** (`--step write`, again by `--step verify` on the published copy): bytes == 2 x sum(lengths), every id
   < 32,000, BOS exactly at the document starts and EOS exactly at the document ends, no id 0, the stream starts with
   BOS, and windows read through `DocAwareMemmapDataset(path, 8192, lengths, "uint16", "packed")` and
   `build_train_dataloader` (world size 1, CPU). The pinned environment (transformers 5.9.0, tokenizers 0.22.2, pyarrow
   24.0.0, numpy 2.4.6, huggingface_hub 1.16.4) is recorded in the manifest; a different tokenizers version can change
   rare ids, which `data/finepdfs_edu_checksums.json` then reports.

## Commands

    export DM_DATA_ROOT=$PWD/workspace/data
    python -I data/fetch_tokenizer.py --root $DM_DATA_ROOT        # once
    python -I data/build_finepdfs_edu.py --root $DM_DATA_ROOT     # download, tokenize, select, write, publish, verify

`--step download|tokenize|select|write|publish|verify` runs one step (repeatable); `--raw` / `--work` put the parquet
and the intermediates elsewhere (default `<root>/raw/finepdfs-edu[-work]`); `--files 0-5,23` builds from a subset (never
the default build); `--edu-min`, `--train-tokens`, `--val-tokens`, `--seed`, `--out-name` change the corpus;
`--workers` / `--threads-per` only the speed. The download needs no login (the dataset is public and ungated; a login
raises the rate limits) but an anonymous download can stall silently: the job script restarts it. Disk: 299 GB of
parquet, ~190 GB of shards and ~125 GB of outputs in the work dir, ~125 GB in the data root at the end.

On this cluster, one cpu_datamover node does everything on its node-local `/raid/scratch` and publishes to the data
root (`dm-workspace/build/jobs/finepdfs_build.sbatch`; job name `dm-data-finepdfs-build`; the code runs from a copy
under `dm-workspace/build/finepdfs/<job>/`):

    sbatch dm-workspace/build/jobs/finepdfs_build.sbatch

Measured on cpu_datamover nodes (2x Xeon Gold 6442Y, 96 CPUs) on 2026-11-09, with this pipeline:

| step | measured | full build (100 files, 96 CPUs) |
|---|---|---|
| download (299 GB) | 157 MB/s with 16 parallel files and `HF_XET_HIGH_PERFORMANCE=1` (the job's default), 116 MB/s with 8; 26-72 MB/s with 4 plain downloads; 4-40 MB/s per file, varying over the day | ~35 min (3 h at the slowest rate seen) |
| filter + tokenize (~118 B tokens, the pool plus the documents counted down to edu 1.4) | 15.9 M tok/s with 16 processes x 4 threads on 64 CPUs (16.5 with 32 x 2), 7.0 M tok/s with 8 x 4 on 32; ~2.4 GB of memory per process; per task 68 % tokenizer, 25 % text statistics | ~85 min with 24 x 4 (~2 h on 64 CPUs) |
| select | 11 s for 2.3 M counted documents | ~2 min |
| write + invariants + sha256 | 1 B training tokens in 2 s, checks in 2 s (node-local NVMe) | ~5 min |
| publish (125 GB to lustre) + re-check + verify | 2.5 GB/s for a 2 GB file | ~10 min (~1 h at 40 MB/s) |

So about 2.5 h on a whole node once it runs (12 h limit); the node-local disk peaks at ~610 GB (299 GB of parquet,
~185 GB of shards, ~125 GB of outputs). A whole 96-CPU / 220 GB node can wait for hours when the partition is busy
(it did on 2026-11-09); `--cpus-per-task=64 --mem=160G` also works (the workers follow the CPU count; ~2.4 GB each).
A 12-file trial (`DM_FP_FILES=0-11 DM_FP_TRAIN_TOKENS=1e9 DM_FP_VAL_TOKENS=5e7 DM_FP_ROOT=<scratch>`, 64 CPUs) took
25 minutes end to end.

## Training on it

The stage-1 configs read `data.paths` / `data.doc_lengths`; point them at this corpus (dtype `uint16`, `position_mode:
packed` and `data.seed 11785` stay as in the yaml):

    FP=$DM_DATA_ROOT/finepdfs_edu_en_60b
    PYTHONPATH=src torchrun --standalone --nproc_per_node=8 -m pretrain.train --config src/pretrain/configs/<yaml> \
        --data.paths $FP/finepdfs_edu_en_train.bin --data.doc_lengths $FP/finepdfs_edu_en_train.doc_lengths.npy ...

Through the launcher (`scripts/campaign/train_arm.sh` puts its Nemotron `DATA_ARGS` before `EXTRA_ARGS`, and the last
occurrence of a flag wins):

    FP=$DM_DATA_ROOT/finepdfs_edu_en_60b
    TAG=_fpe60b NUM_STEPS=57220 SEEDS=11785 \
    EXTRA_ARGS="--data.paths $FP/finepdfs_edu_en_train.bin --data.doc_lengths $FP/finepdfs_edu_en_train.doc_lengths.npy" \
    scripts/slurm/submit.sh -A nvr_elm_llm -p batch -t 4:00:00 -- scripts/campaign/train_arm.sh <arm> 11785

57,220 steps x 128 x 8,192 = 59,999,518,720 tokens: one epoch of the training stream (for any world size that divides
128; 59 windows of the 2026-11-09 build are left over). Run `SMOKE=1 DM_RESUBMIT=0 ... -p batch_short -t 1:00:00` first. Pitfalls:

- `TAG` is mandatory: without it the run name, hence the checkpoint directory, is that of the Nemotron run of the arm.
- `EXTRA_ARGS` must carry both flags (the loader checks that the lengths describe the file, so a mix fails loudly).
- Do not repoint `DM_DATA_ROOT`: the evaluation reads its other parts.
- The end-of-run val CE and `scripts/campaign/eval_arm.sh` stay on the Nemotron validation stream, and the LAUNCH log
  line still prints the Nemotron data directory. The FinePDFs-Edu validation CE of an export, on one GPU from the
  repository root (`memmap_ppl` initializes torch.distributed, so it runs under `torchrun`, as in `docs/eval_engine.md`):

      PYTHONPATH=src torchrun --standalone --nproc_per_node=1 -m evaluation.ppl.memmap_ppl --model_name_or_path <hf export> \
          --data $FP/finepdfs_edu_en_val.bin --dtype uint16 --seq_len 8192 --max_windows 2000 --batch_size 4

## Caveats

- **Pool size.** The ~97 B estimate is scaled from 15 of the 100 files (12 files: 97.6 B, 3 others: 94.0 B), so
  about +-3 %; the headroom is ~1.6x. The build reports the exact pool and stops if it is too small. The full build
  measured 93.91 B, 1.54x the need.
- **Language.** GlotLID page labels are noisy; L3 leaves ~0.5 % of the characters in non-English text. The dag, kiu and
  pcm configs probably hold some English that GlotLID mislabeled; they are not used.
- **Quality.** The edu score sees only the first and the last 2,048 tokens: ~23 % of the pool's tokens are in
  documents over 100k tokens (98 % of them OCR'd by rolmOCR: textbooks, encyclopedias, reports, some table-heavy
  documents), whose middles are unscored. ~65 % of the pool's tokens are OCR output, which can hallucinate. Base
  FinePDFs v1.5.0 also has `ocr_quality_scores`, `dclm_scores` and `fw_edu_v2_scores`, which the -edu set lacks; joining
  them by id (column reads of 579 base files, 2.48 TB) would be a possible v2 filter, out of scope here.
- **Order.** Within a file, refetched documents cluster by position; irrelevant here (every file is used and the
  selection hashes each document).
- **Validation CE.** The first 2,000 windows of the val stream cover ~2.9k documents (2,950 in the trial), and one
  long document can take a sizable share of the scored tokens (5.9 % in the trial; up to ~12 % at the 95th percentile
  in planning; manifest `val_coverage`); compared arms still score the same windows.
- **Mixture.** The FinePDFs card recommends keeping PDFs at or below 25 % of a training mixture; this corpus is 100 %
  PDF.
- **Reproducibility.** Tokenizer or library drift changes ids: the manifest records the versions and
  `data/finepdfs_edu_checksums.json` catches it. The pinned revision can be squashed away; the per-file sha256s pin the
  inputs regardless.

## Terms

HuggingFaceFW/finepdfs-edu is released under the Open Data Commons Attribution License (ODC-By) v1.0, and its use is
also subject to the CommonCrawl Terms of Use. Attribute it (Kydlicek, Penedo and von Werra, "FinePDFs", Hugging Face,
2025). It contains copyrighted documents (books among them) and possibly personal information; the dataset card links
an opt-out / PII removal form. The tokenizer is under the Llama 2 Community License. Build the corpus yourself from the
source; do not re-publish it.
