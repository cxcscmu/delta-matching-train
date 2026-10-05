# Converting models between this pipeline and Hugging Face

How to turn a trained checkpoint into a standalone Hugging Face model that loads without this codebase, and how to
bring an HF model back into the pipeline for further training. The tools are in this directory and support models of
this repository only (`model_type: dmlm`): `make_standalone.py` refuses a `config.json` or `config.train.json` whose
`model_type` is not `dmlm` or that names keys or mixers this repository does not define, `verify_standalone.py`
refuses a config that is not `dmlm`, and `export_checkpoint.py` refuses a `model_config.json` with fields or mixers this
repository lacks.

## The three formats

The same weights live in three places. Tensor names are identical in all three (`model.embed_tokens`,
`model.layers.N.{input_layernorm, mixer.*, post_attention_layernorm, mlp.*}`, `model.norm`), so no conversion ever
renames a tensor.

| format | where | contents | read by | good for |
|---|---|---|---|---|
| training checkpoint | `$DM_WORKSPACE/qwen35_m/ckpts/<run>/checkpoint-<step>/` | FSDP shards `__<rank>_0.distcp` (weights + AdamW state), `pytorch_model.bin` (consolidated bf16 weights, tied `lm_head` included), `model_config.json` (the full `ModelConfig`), per-rank data-loader state, `.complete` | the trainer | resuming a run exactly |
| HF export | `$DM_WORKSPACE/qwen35_m/hf/<run>/` | `config.json` (`model_type: dmlm`, clean-kernel patched), `config.train.json` (as trained), `model.safetensors` (bf16, tied `lm_head` dropped), tokenizer; the pipeline adds `valce.txt` and `TRAIN_COMPLETE` | the eval chain, `--init_from_hf` | evaluation, warm starts |
| standalone repo | a Hub model repo | the export's weights, tokenizer and both configs, plus `modeling_dmlm.py`, `configuration_dmlm.py`, `auto_map` in `config.json`, a model card | stock `transformers` with `trust_remote_code=True` | inference and fine-tuning anywhere |

A run writes its HF export after its last step; `scripts/campaign/train_arm.sh` then saves the untouched config as
`config.train.json` and patches `config.json` for the clean-kernel eval (`attn_impl: sdpa`, `dm` → `gqa`,
`mla_dm` → `mla`, the `dm_mode` / `dm_fused_producers` keys dropped). The FP8 arms' attention layers keep their
parameter names, so their weights load unchanged. The evaluation loaders request
`attn_implementation="flash_attention_2"` (the default of `EVAL_ATTN_IMPL`), which overrides the export's
`attn_impl: sdpa`: the benchmarks ran FlashAttention-2 (`docs/benchmark.md` section 1).

## Files here

| file | does |
|---|---|
| `export_checkpoint.py` | training checkpoint → HF export, exactly as a finished run writes one (CPU) |
| `make_standalone.py` | HF export → standalone directory; `--upload` pushes it and checks every file on the Hub (CPU) |
| `verify_standalone.py` | checks a standalone directory or Hub repo against this repository's loader (GPU) |
| `remote_code/` | the model code shipped in standalone repos: GQA, MLA, Mamba2, GatedDeltaNet and KDA mixers, cached `generate()` |

## Before you start

- A checkout of this repo and its `.venv`. Tested with torch 2.12, transformers 5.9.0, mamba-ssm 2.3.2,
  causal-conv1d 1.6.2, flash-linear-attention 0.5.2, huggingface_hub 1.16.4.
- `DM_WORKSPACE` set; `source scripts/campaign/env.sh` derives `DM_DATA_ROOT` and `DM_TOKENIZER_DIR` from it.
- `hf auth login` with a token that can write to the target namespace. A job whose environment overrides `HF_HOME`
  (as `scripts/campaign/env.sh` does, to the data root's `hf_cache`) cannot see that token; set
  `HF_TOKEN_PATH=$HOME/.cache/huggingface/token` there.
- A CUDA GPU for `verify_standalone.py`: the Mamba2, GatedDeltaNet and KDA layers run GPU kernels (`mamba_ssm`,
  `fla`). The other tools need only CPU.

## A. Trained checkpoint → standalone HF model

1. **Get an HF export.** A finished run already has one at `$DM_WORKSPACE/qwen35_m/hf/<run>/` (with
   `TRAIN_COMPLETE`). For a run that stopped early, or an intermediate step, export a complete checkpoint:

   ```bash
   PYTHONPATH=src python scripts/hf/export_checkpoint.py \
       --ckpt $DM_WORKSPACE/qwen35_m/ckpts/<run>/checkpoint-<step> \
       --out  $DM_WORKSPACE/qwen35_m/hf/<run>_step<step>
   ```

   It calls the trainer's own `save_hf_checkpoint` and then applies the same patch as `train_arm.sh`
   (`--no-eval-patch` keeps the training mixers); the weights and the tokenizer come out byte-identical to the run's
   own export. The tokenizer is `$DM_TOKENIZER_DIR` unless `--tokenizer` names another.

2. **Build the standalone directory.**

   ```bash
   python scripts/hf/make_standalone.py $DM_WORKSPACE/qwen35_m/hf/<run> --out <dir> --card <README.md>
   ```

   It hard-links the weights (no extra disk; `--copy` copies them), copies the tokenizer, `config.train.json` and
   `remote_code/`, and writes `config.json` with `architectures`, `auto_map` and the tokenizer's `bos_token_id` /
   `eos_token_id` (so `generate()` stops at `</s>`), without the training-only keys `dm_mode` and
   `dm_fused_producers` (`config.train.json` keeps them). It leaves out `TRAIN_COMPLETE` and `valce.txt`. It refuses,
   before writing anything, an export that is not this repository's (see the top), one whose mixers the shipped code
   does not cover (every cell of the campaign is covered), and an unpatched export (`attn_impl` other than `sdpa`).
   Without `--card` it writes a short factual card. For a release, write a real one: the arm and what it means, the
   architecture, the training recipe, the evaluation and the licenses (step 5).

3. **Verify it on a GPU.**

   ```bash
   python scripts/hf/verify_standalone.py --export $DM_WORKSPACE/qwen35_m/hf/<run> --standalone <dir>
   ```

   The standalone side runs in a process that cannot import this repository's source. It must print `PASS`: logits
   bitwise identical on 4 held-out windows of 8,192 tokens, the same training loss with finite gradients, and the same
   greedy tokens from a plain-forward loop on both sides and from the standalone's `generate()` with and without its
   cache, compared up to the first end-of-sequence token. Both sides run SDPA attention, so like is compared with
   like. (The reference cannot use `generate()` for MLA: this repository's MLA mixer refuses an attention mask.) Pass
   `--val random` when the campaign's validation stream is not on the machine.

4. **Upload.** Run step 2 again, into a new (or emptied) `--out` directory since the build refuses a non-empty one,
   with `--upload <user>/<repo> --message "<plain message>"`. The repo is created private
   if it does not exist, everything goes in one commit, and each file is checked on the Hub against the local copy.
   Files already in the repo that this build does not contain (for example the code files of an older upload under
   other names) fail the check; `--delete-stale` deletes them in the same commit. Re-uploading unchanged files is a
   no-op: the Hub records no commit. Hub commit text cannot be edited afterwards, so keep commit messages plain from
   the first upload (`HfApi().super_squash_history` keeps the squashed commits' authors). To test the published repo
   itself, pass its id as `--standalone`. A GPU node without internet access needs the snapshot first: `hf download`
   on a login node, then pass the downloaded path.

5. **Before making a repo public,** add the licenses to the card: the model's own, and the Llama 2 Community License
   of the Llama 2 tokenizer every repo ships (`docs/SETUP.md` section 4). Publishing also publishes `remote_code/`.

## B. HF model → trainable in this pipeline

New runs start from HF weights with `--init_from_hf <dir>`. It loads `<dir>/model.safetensors` into the model the
arm builds, with a strict name check (only the tied `lm_head.weight` may be missing), a fresh optimizer and a fresh
LR schedule from step 0. It accepts the pipeline's exports and standalone repos alike; the shipped code and
`auto_map` are ignored.

1. **Download.** `hf download <user>/<repo> --local-dir $DM_WORKSPACE/qwen35_m/hf/<name>`. Only `model.safetensors` is
   read.

2. **Launch** with an arm that builds the same architecture, under a new `TAG`:

   ```bash
   TAG=_ft EXTRA_ARGS="--init_from_hf $DM_WORKSPACE/qwen35_m/hf/<name>" scripts/campaign/train_arm.sh mamba2_dm_matched_hd128 11785
   ```

   The architecture must match the
   repo's `config.json` (`layer_mixers` and the geometry): the `mamba2_*_hd128` arms for Mamba2 : GQA models, the
   `mla_*_hd128` arms for GDN : MLA, the `kda_*_hd128` arms for KDA : GQA, `clean_hd128` / `dm_*_hd128` for GDN : GQA.
   The numerics arm (clean, stale, matched) is free, since the weights have the same format in all three. Set the
   schedule for continued training through `EXTRA_ARGS` and the usual knobs (`NUM_STEPS`, `DM_LR`);
   `scripts/campaign/stage2_arm.sh` is a complete example (a 100-step re-warm, a linear decay).

3. **The new `TAG` is required.** If `ckpts/<run name>` already exists, the trainer resumes that checkpoint and
   ignores `--init_from_hf`.

4. **Exact resumption is not possible from HF.** A repo carries weights only; the optimizer moments and the data
   position live in the training checkpoint. Keep any checkpoint you may resume.

5. **Evaluating a downloaded repo** with the pipeline's benchmarks: put it at `$DM_WORKSPACE/qwen35_m/hf/<name>/`
   and run `scripts/campaign/eval_arm.sh <name>`. The eval checks only that `config.json` is the clean-kernel config,
   which a standalone repo's is. A standalone repo's `config.json` carries the tokenizer's end-of-sequence id, so the
   generation engine stops at that id only, while an export without one stops at the engine's fallback ids {2, 11}
   (`src/inference/engine.py`, `_FALLBACK_EOS`); results differ only if a generation emits id 11.

## Other mixers

`remote_code/` covers the GQA, MLA, Mamba2, GatedDeltaNet and KDA mixers, so every cell of the campaign (Mamba2 : GQA,
GDN : MLA, GDN : GQA with or without qk-norm, KDA : GQA) converts as it is. A cell with another mixer needs it added
first, the way KDA was:

- port it from `src/pretrain/modeling/<mixer>.py` into `remote_code/modeling_dmlm.py`, keeping only the bf16 path the
  eval runs, and add it to `MIXER_REGISTRY` there and to `SUPPORTED_MIXERS` (and `MIXER_PACKAGES`) in
  `make_standalone.py`; a new fp8 training mixer name also goes into `TRAIN_MIXERS`, and a new package into
  `TESTED_VERSIONS`;
- import its library inside a `try` with an install hint, as the Mamba2, GatedDeltaNet and KDA mixers do, so a repo
  only needs the packages its own layers use;
- give its forward the `cache` argument (fla layers keep their recurrent and conv states in fla's `FLACache`,
  `DMLMCache.fla`, as GatedDeltaNet and KDA do);
- pass `verify_standalone.py` on one export of the cell.

## Why the conversion is exact, and what to watch

- `remote_code/` is the bf16 code of this repository's HF loader with both of its attention paths: SDPA by default,
  flash-attn with `attn_implementation="flash_attention_2"`. This repository's loader never enables the fused kernels
  (`DMLMConfig.to_model_config` leaves `fused_rmsnorm`, `fused_rope` and `fused_swiglu` off), so both sides run eager
  RMSNorm, RoPE and SwiGLU, `mamba_ssm`'s Mamba2 and fla's GatedDeltaNet and KimiDeltaAttention called as `gdn(x)` /
  `kda(x)`. On the same attention path the logits therefore match bit for bit. `verify_standalone.py` checks the SDPA
  path; the study's evaluation loaded models with `flash_attention_2`, so pass that to reproduce its numerics (the two
  differ by attention rounding only).
- `transformers` refuses to load remote code until every package it imports is installed. Its scan covers imports
  inside functions and skips only those inside `try` blocks. The training modules import TransformerEngine,
  `torchao` and the Delta-Matching kernel, so `remote_code/` is a trimmed copy, not the training modules. It imports each
  mixer's library (`mamba_ssm`, `fla`) inside a `try` with an install hint, so the load-time requirements are only
  torch and transformers, and a repo needs just the packages its own layers use.
- Once `config.json` has `auto_map`, `AutoTokenizer.from_pretrained` also needs `trust_remote_code=True`, or it asks.
- An all-ones attention mask is treated as no mask, so an unpadded `generate()` runs the same kernels as a plain
  forward. This repository's MLA mixer asserts `attention_mask is None`, so `generate()` through its loader fails
  on MLA models; the standalone code (or the continuous-batching engine in `src/inference`) is the way to decode.
- `generate()` supports greedy decoding and sampling (`num_beams=1`, no assisted generation) and stops at the
  tokenizer's end-of-sequence token. Prompts batched together must share one length, because the recurrent layers
  (Mamba2, GatedDeltaNet, KDA) have no padding mask. In eager mode decoding runs at about 17 tokens/s on one Hopper GPU and is
  launch-bound: the cache made 64 new tokens 1.26× faster after a 4k prompt and 1.07× after a 1k one.
- Fine-tuning a standalone repo runs in bf16 and without gradient checkpointing (`gradient_checkpointing_enable()`
  refuses: no layer implements it). The FP8 Delta-Matching attention the stale and matched arms trained with is a
  compiled kernel that is not shipped; to continue training in those numerics, use part B inside this pipeline.

## Storage

- On shared storage with a group quota, delete test downloads when done, and let `make_standalone.py` hard-link the
  weights rather than copy them.
