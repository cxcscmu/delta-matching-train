# The data

Everything the experiments read comes from public sources and is rebuilt here, bit for bit, by the scripts of this
directory. No data is shipped: several sources forbid redistribution (see Terms), so every user builds their own copy.

| document | what |
|---|---|
| [`pretrain_stage1.md`](pretrain_stage1.md) | the stage-1 (pretraining) corpus: Nemotron-CC-v2.1 High-Quality + High-Quality-DQA, Llama-2 ids |
| [`pretrain_stage2.md`](pretrain_stage2.md) | the stage-2 (context extension) corpus: ProLong's long-context data re-tokenized with Llama-2, 65,536-token windows |
| [`pretrain_finepdfs_edu.md`](pretrain_finepdfs_edu.md) | an alternative pretraining corpus (optional): 60 B tokens of English FinePDFs-Edu, quality-filtered and sampled at random, Llama-2 ids in the stage-1 format |
| `docs/benchmark.md` | the evaluation protocol that reads the evaluation data |

## Building the data root

The data root is `DM_DATA_ROOT` (default `workspace/data` in the repository). The scripts need the pinned environment
(`docs/SETUP.md` section 2; stdlib, numpy, pyarrow, transformers, sentencepiece, huggingface_hub and lm-eval are what
they use) and a Hugging Face login with access to three gated repositories: accept the terms on each page first.

| gated repository | needed by | terms |
|---|---|---|
| `meta-llama/Llama-2-7b-hf` | the tokenizer (every part) | Llama 2 Community License |
| `nvidia/Nemotron-CC-v2.1` | stage 1 | NVIDIA Data Agreement for Model Training |
| `meta-llama/Meta-Llama-3-8B` | stage 2 (decoding ProLong's Llama-3 ids) | Llama 3 Community License |

    hf auth login
    export DM_DATA_ROOT=$PWD/workspace/data
    python data/fetch_tokenizer.py --root $DM_DATA_ROOT      # tokenizer/llama2_32k
    python data/build_stage1.py   --root $DM_DATA_ROOT       # nemotron_cc_v2d1_hq_dqa/   (every stage-1 experiment)
    python data/build_stage2.py   --root $DM_DATA_ROOT       # prolong_64k_llama2/        (scripts/experiments/stage2_context.sh only)
    python data/prepare_eval.py   --root $DM_DATA_ROOT       # ruler-500-llama2/ and hf_cache/ (the evaluation)
    python -I data/build_finepdfs_edu.py --root $DM_DATA_ROOT  # finepdfs_edu_en_60b/ (optional; public data, no login)
    python data/verify.py         --root $DM_DATA_ROOT       # sizes + sha256 of everything (--part: only what you built)

Each build ends by checking its outputs against `checksums.json` and fails loudly on a difference; each step can be run
alone (`--step` of the build scripts, `--only` of `prepare_eval.py`; see `--help`). Measured on one 208-CPU node with a fast network:

| part | result | disk after (during) | time |
|---|---|---|---|
| tokenizer | `tokenizer/llama2_32k/` (4 files) | 2 MB | seconds |
| stage 1 | `nemotron_cc_v2d1_hq_dqa/`: 39.66 B training + 807 M validation tokens | 81 GB (220 GB) | 17 minutes |
| stage 2 | `prolong_64k_llama2/`: 37.16 B training tokens + 256 validation windows | 74.5 GB (200 GB) | 36 minutes |
| evaluation | `ruler-500-llama2/` (RULER, 13 tasks x 500 examples at 2K to 128K) and `hf_cache/` (the lm-eval datasets) | 8.3 GB | 3 minutes |

The downloads stay under `<root>/raw/` (`--delete-raw` removes them after a verified build). Fewer cores make the builds
slower but not different: the outputs do not depend on the number of processes.

## The evaluation data

- `ruler-500-llama2/`: RULER (Hsieh et al., 2024) regenerated with the Llama-2 tokenizer, 500 examples per task and
  length, from the Hugging Face dataset `tturing/ruler-500-llama2` at a pinned revision (CC BY-SA 4.0; its card
  describes the generation).
- `hf_cache/`: the datasets of the lm-eval tasks of the protocol (LAMBADA, HellaSwag, PIQA, ARC-Easy / Challenge,
  SciQ, OpenBookQA, WinoGrande, COPA, SWDE, FDA, SQuAD completion, TriviaQA, MMLU), fetched by lm-eval's own task
  definitions from the Hugging Face Hub; `scripts/campaign/env.sh` points `HF_HOME` at it. These have no checksums
  (the layout is the datasets library's); `verify.py` checks that the cache is there.

## The FinePDFs-Edu corpus (optional)

`finepdfs_edu_en_60b/` is an alternative pretraining corpus in the stage-1 format (`pretrain_finepdfs_edu.md`):
English FinePDFs-Edu PDF text, quality-filtered and sampled at random by document, in Llama-2 ids. The build of
2026-11-09 (job 21596272, 2 h 08 min on one 80-CPU cpu_datamover node) holds 60,000,005,485 training tokens in
10,162,083 documents and 800,016,252 held-out tokens in 136,211 documents: 122 GB, plus the 299 GB of source parquet
kept in `raw/finepdfs-edu/`. Its own `README.md` gives the filters, counts and provenance, and its `manifest.json` gives
everything else. On this cluster it is `workspace/data/finepdfs_edu_en_60b/`, which is
`dm-workspace/data/finepdfs_edu_en_60b/`.

To train a stage-1 arm on it, override two keys; no code change is needed. dtype `uint16`, `position_mode: packed` and
`data.seed` stay as in the yaml:

    FP=$DM_DATA_ROOT/finepdfs_edu_en_60b
    --data.paths $FP/finepdfs_edu_en_train.bin --data.doc_lengths $FP/finepdfs_edu_en_train.doc_lengths.npy

At global batch 128 x 8,192, one epoch of the training stream is 57,220 steps (59,999,518,720 tokens).
`pretrain_finepdfs_edu.md` ("Training on it") has the launcher command and its pitfalls: `TAG` is mandatory, and the
end-of-run val CE still scores the Nemotron stream.

## Terms

The code license does not cover the data. What you build is subject to the terms of its sources:

- `nvidia/Nemotron-CC-v2.1` (stage 1): the NVIDIA Data Agreement for Model Training. It allows training models and
  forbids distributing, publishing or otherwise making the data available to others in any form, which includes the
  tokenized stage-1 files. The High-Quality-DQA part was generated with Qwen3-30B-A3B (Apache-2.0).
- `princeton-nlp/prolong-data-512K` (stage 2): no license of its own; its parts come from The Stack, SlimPajama
  (books, arXiv, StackExchange), FineWeb, FineWeb-Edu, Dolma (Wikipedia), Tulu v2, OpenWebMath and TextbookChapters,
  each under its own terms (links on the dataset card).
- The Llama 2 tokenizer (every part, and baked into every model export): the Llama 2 Community License. The Llama 3
  tokenizer (stage-2 build only): the Llama 3 Community License.
- The evaluation datasets: the license on each dataset card.
- `HuggingFaceFW/finepdfs-edu` (the optional FinePDFs-Edu corpus, `pretrain_finepdfs_edu.md`): the Open Data Commons
  Attribution License (ODC-By) v1.0, which requires attribution, and the CommonCrawl Terms of Use. It contains
  copyrighted documents and possibly personal information; its card links an opt-out / PII removal form. The
  tokenized corpus carries the Llama 2 tokenizer's ids (Llama 2 Community License). Build it locally.

Build the data yourself; do not re-publish it.

## Files

| file | what |
|---|---|
| `fetch_tokenizer.py` | the Llama-2 tokenizer at a pinned revision |
| `build_stage1.py`, `stage1_tokenize.py`, `stage1_split.py` | the stage-1 corpus (`pretrain_stage1.md`) |
| `build_stage2.py`, `stage2_repack.py`, `stage2_val.py` | the stage-2 corpus (`pretrain_stage2.md`) |
| `prepare_eval.py` | RULER and the lm-eval datasets |
| `verify.py`, `checksums.json` | the size and sha256 of every file the builds produce (122 files) |
| `build_finepdfs_edu.py`, `finepdfs_tokenize.py`, `finepdfs_select.py`, `finepdfs_edu_files.json` | the optional FinePDFs-Edu corpus (`pretrain_finepdfs_edu.md`) and its pinned input files |
| `finepdfs_edu_checksums.json` | the size and sha256 of its four streams, written by its first default build; not in `checksums.json` and not read by `verify.py` (`build_finepdfs_edu.py --step verify` checks it) |
