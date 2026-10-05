# The data

Everything the experiments read comes from public sources and is rebuilt here, bit for bit, by the scripts of this
directory. No data is shipped: several sources forbid redistribution (see Terms), so every user builds their own copy.

| document | what |
|---|---|
| [`pretrain_stage1.md`](pretrain_stage1.md) | the stage-1 (pretraining) corpus: Nemotron-CC-v2.1 High-Quality + High-Quality-DQA, Llama-2 ids |
| [`pretrain_stage2.md`](pretrain_stage2.md) | the stage-2 (context extension) corpus: ProLong's long-context data re-tokenized with Llama-2, 65,536-token windows |
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

Build the data yourself; do not re-publish it.

## Files

| file | what |
|---|---|
| `fetch_tokenizer.py` | the Llama-2 tokenizer at a pinned revision |
| `build_stage1.py`, `stage1_tokenize.py`, `stage1_split.py` | the stage-1 corpus (`pretrain_stage1.md`) |
| `build_stage2.py`, `stage2_repack.py`, `stage2_val.py` | the stage-2 corpus (`pretrain_stage2.md`) |
| `prepare_eval.py` | RULER and the lm-eval datasets |
| `verify.py`, `checksums.json` | the size and sha256 of every file the builds produce (122 files) |
