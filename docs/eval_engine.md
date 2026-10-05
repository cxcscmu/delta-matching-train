> The engineering notes of the inference + evaluation engine (`src/{evaluation, inference, modeling, utils}`) and of the
> `dmlm` HF bridge (`src/pretrain/hf/`) that connects it to the pretraining engine. The Nemotron-H parity and speed
> numbers of §4 were measured before the engine moved into this repository (§11). The confirmed evaluation protocol of
> the study is `docs/benchmark.md`; the eval chain is `scripts/campaign/eval_arm.sh`.

# Eval engine: inference + evaluation for the study's models (and Nemotron-H)

## 1. What this is & the design principle

**Train and eval are deliberately decoupled.** `src/pretrain` (`docs/train_engine.md`) trains a model
and writes a **HuggingFace-format checkpoint**; it imports *nothing* from this engine. This engine
then **loads that checkpoint and evaluates it**. The seam is a directory of `safetensors` + a
`config.json` on shared NFS: nothing else.

The engine lives in **`src/{evaluation,inference,modeling,utils}`** (formerly a standalone package, moved into `src/`
so the train→eval loop is in-repo; the engine switch keeps its old name, `EVAL_ENGINE=mini_eval_dev`). It is two
subsystems on a shared infra layer:

1. **Eval harness** (`src/evaluation/`): an lm-eval-style harness that loads an HF
   `AutoModelForCausalLM`, wraps it, and runs reasoning tasks (GPQA/AIME/MMLU-Pro/LiveCodeBench),
   long-context retrieval (RULER), or perplexity (PG19 / **memmap**). Multi-node data-parallel via
   `torchrun`.
2. **Inference engine** (`src/inference/`): a continuous-batching engine that makes the 131072-token
   generation budget tractable on a single GPU, a drop-in for the HF wrapper's `generate`. Built for
   **Nemotron-H**, and **also drives the study's dmlm models** (GQA / MLA / Mamba2 / GDN / KDA engine
   mixers; `EngineModel` dispatches on `model_type`; §4). Use it for dmlm long-context generation (RULER)
   via `EVAL_ENGINE=mini_eval_dev`: KV/recurrent-state cache instead of the HF path's O(L²) recompute
   (~12× at 4096). Perplexity/loglikelihood/recall still use the HF `.forward` path (the engine is
   generate-only).

**The single most important architectural idea: everything reuses the loaded HF `transformers`
modules and replaces only the serving/cache behavior.** The bridge (§2) makes a pretrain checkpoint a
first-class HF model; the inference engine (§4) does an in-place `module.__class__` swap onto its
Mamba/attention subclasses, sharing weights zero-copy. This gives **numerical equivalence by
construction** and keeps the surface area small.

It serves two model families: **the study's pretrain checkpoints** (`model_type="dmlm"`: GQA flash-attn / MLA /
Gated-DeltaNet and KDA through `fla` / Mamba2), the primary use, and **Nemotron-H**, the engine's original target.
`flash_attn` (GQA) and `fla` (GDN) ride *inside the model*; the harness just loads it and calls
`.forward`/`.generate`.

## 2. The HF bridge: `src/pretrain/hf/` (how a pretrain checkpoint becomes evaluable)

Makes a trained `src/pretrain` model a first-class HF causal LM (`model_type="dmlm"`):

- `configuration_dmlm.py`: `DMLMConfig(PretrainedConfig)`: carries the **full `ModelConfig`
  surface**: every field **including `qk_norm` / `rms_norm_in_fp32`** (+ the fp8 arms' `dm_mode`, carried for provenance);
  `from_model_config` / `to_model_config` round-trip it. **(A fixed bug.)** `qk_norm` /
  `rms_norm_in_fp32` were once dropped by the bridge, so every eval silently loaded a GQA model with
  **un-normalized q/k** (the `q_norm`/`k_norm` weights came back "UNEXPECTED" and were discarded: a
  correctness bug, since every config uses `qk_norm: true`). `src/pretrain/scripts/verify_hf_export.py` now has a
  `gqa_qknorm` regression case **plus a dropped-key assertion** (loads with `output_loading_info=True`
  and fails on any unexpected/missing key besides the tied `lm_head.weight`) to catch future
  `ModelConfig`→`DMLMConfig` field-drift; eval now loads all weights with 0 UNEXPECTED.
- `modeling_dmlm.py`: `DMLMForCausalLM(PreTrainedModel, GenerationMixin)`: **reuses the
  `src/pretrain` `Transformer` backbone + tied head verbatim**, so parameter names
  (`model.embed_tokens.*`, `model.layers.N.*`, `model.norm.*`, `lm_head.weight`) match the training
  checkpoint and the consolidated weights load directly. `.forward → CausalLMOutputWithPast(logits,
  loss)`; it honors transformers' `logits_to_keep` (an int k keeps the logits of the last k positions, 0 or None all
  of them). Declares `_supports_flash_attn` / `_supports_sdpa` / `_supports_attention_backend` (the
  transformers-5.x flag names) and maps HF's `attn_implementation` → the GQA mixer's flash/sdpa path.
  `.generate` works via `GenerationMixin` **without a KV cache** (recompute: fine for short eval
  generations; §10).
- `export.py`: `save_hf_checkpoint(state_dict, model_cfg, out_dir, tokenizer=None)`: writes
  `config.json` + `model.safetensors` (+ tokenizer). Drops the tied `lm_head.weight` (re-tied on load).
- `register_auto()`: registers the config+model with the HF Auto classes so a `dmlm` checkpoint
  loads with `trust_remote_code=False` (no bundled modeling file): provided `pretrain` is on `PYTHONPATH`
  (`scripts/hf/make_standalone.py` builds the bundled-code variant for publishing).

**Verified bit-exact (Δlogits = 0)** for every mixer: build → `save_hf_checkpoint` →
`AutoModelForCausalLM.from_pretrained` → identical logits (`src/pretrain/scripts/verify_hf_export.py`; its GQA cases run
on CPU, the GatedDeltaNet / Mamba2 cases need CUDA). One bug
fixed in the process: `RotaryEmbedding` cached `cos/sin` in a non-persistent buffer, which HF's
**meta-device** `from_pretrained` left uninitialized (→ NaN RoPE); now computed in `forward` from
`inv_freq` (also enables arbitrary eval seq-len).

**Train-engine export.** `src/pretrain` writes the HF checkpoint when `--hf_save_dir <dir>` is set: the
`FSDP2Trainer`/`DDPTrainer` gathers the full (unsharded) state dict at end-of-training (`DTensor.
full_tensor()`) and `save_hf_checkpoint`s it (`base_trainer.py::save_hf`). Put `hf_save_dir` on **shared
storage** visible to the eval nodes; the DCP resume checkpoints may stay node-local. **Tokenizer baking:** set
`--tokenizer_dir <dir>` and
`save_hf` loads `AutoTokenizer.from_pretrained(dir, local_files_only=True)` → writes it into the export.
**This is required for ALL `meta_eval` evals**: `ruler/load_model.py::load_model_for_eval`
*unconditionally* `AutoTokenizer.from_pretrained(ckpt)`, so even tokenizer-free **recall** fails to load
without it (`ValueError: Couldn't instantiate the backend tokenizer`). `memmap_ppl` is the only
tokenizer-free path. The study uses the 32k **Llama-2** tokenizer (`tokenizer/llama2_32k` in the data root).

## 3. The eval harness: `src/evaluation/`

`meta_eval.py` is the entry point (`torchrun -m evaluation.meta_eval`, one rank = one DP replica).
`meta_eval.main()` (`src/evaluation/meta_eval.py`) → argument parsing (an unknown or misspelled flag exits with
"unrecognized arguments" before anything starts) → `dist_init(...)` → `build_config(eval_config)` →
`check_complete_marker(...)` skip-short-circuit → `load_model_for_eval(model, device_map=
EVAL_DEVICE_MAP)` (`AutoModelForCausalLM.from_pretrained(attn_implementation=EVAL_ATTN_IMPL [default
flash_attention_2; also sdpa/eager], dtype=bfloat16, experts_implementation=EVAL_EXPERTS_IMPL)`)
→ **wrapper choice**: `EngineModel(model)` iff `EVAL_ENGINE=mini_eval_dev` and the config's type is not `recall`
(for a dmlm model only when every layer's mixer has an engine twin; otherwise it prints why and falls back),
else `HFModel(model)`. Then dispatch on `eval_config["type"]`:

**Prompts carry no BOS by default** (the protocol, docs/benchmark.md). `--add_bos` (default off) starts every
prompt with the tokenizer's BOS: the harness through the wrapper's existing `add_prefix` (loglikelihood context, generation
and batch encodings; the continuation is unchanged), RULER through `ruler.py::encode_left_padded_with_bos` ([bos] + tokens,
then left padding); the recall evaluators (token space) refuse it. Banner `[meta_eval] add_bos ON`. Built for models trained
on BOS-anchored windows (`--data.prepend_bos`); tests `tests/test_eval_bos.py`.

- `lm_eval_harness`: `LMEvalWrapper(model, tokenizer, max_seq_len, max_new_tokens, ...)`
  (`lm_eval_harness/wrapper.py`) → `LMHarnessEvaluator(...).evaluate()` (`lm_eval_harness/evaluator.py`) →
  `lm_eval.simple_evaluate(model=wrapper, tasks=...)`. lm_eval round-robin-shards docs by
  `wrapper._rank/_world_size`, then calls `wrapper.loglikelihood` (MC; HF `.forward`) or
  `wrapper.generate_until`. Master extracts the per-task metric and
  `dist.broadcast_object_list`'s it to all ranks. On the plain HF path (`EVAL_ENGINE` unset) a dmlm model with Mamba2 /
  GatedDeltaNet / KDA layers generates only at `--eval_batch_size 1`: those layers cannot mask the left padding of a
  batch, so `ruler/load_model.py::padded_generation_error` refuses bigger batches; the engine has no such limit.
  `EVAL_WORK_STEAL=1` steals generation work across ranks only when every rank owns requests of the task, else it
  takes the static path.
- `ruler`: `RULEREvaluator(...).evaluate()` (`ruler/ruler.py`) loads pre-tokenized
  `{data_dir}/{length}/data/{task}/{subset}.jsonl`, pads to `world_size`,
  `chunk_list_interleaved(...)[rank]`, `model.generate(stop_token_list=[[EOS],"\n",...],
  max_new_tokens=tokens_to_generate)`, `dist.all_gather_object` predictions →
  `scorer.run_evaluation_per_task` (`ruler/scorer.py`) → `string_match_all`/`string_match_part`
  (`ruler/eval_constants.py`). Predictions and per-task markers go to
  `<output_dir>/<name>/full_results/<cfg>/<length>/<task>.{jsonl,complete}`. A run whose run id changed (another
  `--run_name` or `--version_str`) deletes only this config's results JSON, its lock, its marker and its predictions
  directory before it starts. The run id covers the run name and the version string, not the config's content: after
  editing a config, re-run with a new `--version_str`.
  **RULER data: `tturing/ruler-500-llama2`**: RULER-500 regenerated with the **Llama-2 32k
  tokenizer** (exact context lengths for our dmlm models), at
  `$DM_DATA_ROOT/ruler-500-llama2` (8 lengths 2048–131072 × 13 tasks × 500; docs/SETUP.md section 4).
  Config **`configs/ruler_llama2_8k.yaml`** (`data_dir` → it; `lengths: [4096, 8192]`; stage 2: `ruler_llama2_64k.yaml`). It uses
  RULER's **binary-search format**: each record stores `input` **without** the answer prefix + a separate
  `answer_prefix` field, so **`load_data` feeds `input + answer_prefix`** (the `ap = d.get("answer_prefix")`
  branch; falls back to the classic strip + NIAH `':'` when absent). RULER's W&B run writes to `WANDB_DIR`, else
  `$DM_WORKSPACE`. **Run with `EVAL_ENGINE=mini_eval_dev`** → the dmlm
  continuous-batching `EngineModel` (KV/recurrent-state cache; §4): **~12× faster than the cache-free HF
  `.generate` at 4096, identical scores** (validated end-to-end on a small trained GQA checkpoint: niah+qa@4096
  avg 56.25, engine 1.75 s vs cache-free 21.08 s). The HF path (`EVAL_ENGINE` unset) still works but is
  cache-free (O(L²)·G; use `max_samples`/short lengths). Validated: prompt construction + full RULER run.
- `recall` (`src/evaluation/recall/`; MQAR is computed by the eval chain but no longer reported): tokenizer-free
  in-context **synthetic-KV / MQAR** (+ `phonebook`) recall. `recall/synthetic_kv.py::make_mqar` lays
  out token-space `[k0,v0,…]` pairs (keys `[1,V/2)`, values `[V/2,V)`) + power-law queries with the gold
  value at the position **after** each query key; `recall/recall_eval.py::RecallEvaluator` sweeps a
  **`num_kv_pairs × seq_len` grid** ({64…8192}×{2K…16K}; infeasible cells are skipped: a cell needs
  4 × pairs ≤ seq_len, so the 8192-pair row never runs; the phonebook grid is 64…4096 entries), scores the
  standard next-token shift (`logits[:,:-1].argmax == labels[:,1:]` at non-`-100`, same as the train
  loss), DP-shards via `chunk_list_interleaved` + all-reduces, and writes a pairs×ctx **accuracy matrix**
  (the degradation curve). Plain HF `.forward` (EngineModel excluded). Configs
  `recall_synthetic.yaml` / `recall_phonebook.yaml`; their `batch_size` (8) wins over `--eval_batch_size`. wandb: after
  the completion marker, and only when W&B is on, it resumes `{ckpt}/wandb_run_id.txt` and logs
  `eval/recall_kv_acc_p{pairs}_c{ctx}` + `eval/recall_mean` to the same train run.

**The bridge into the harness**: two no-op-for-Nemotron-H touch-points:
`evaluation/ruler/load_model.py::load_model_for_eval` and `evaluation/ppl/load.py::load_model_for_ppl`
call `_maybe_register_pretrain_models()` (imports `pretrain.hf` + `register_auto()`; silent only when the `pretrain`
package is absent, any other failure is printed) **before** `AutoModelForCausalLM.from_pretrained`, so a `dmlm` dir
loads through the **existing** entry points unchanged.

**W&B never stops an evaluation.** Every `wandb.init` is guarded: `WANDB_MODE=disabled` or `WANDB_DISABLED=true` turn
W&B off, and a failing init only prints. Each evaluator writes its completion marker before its final W&B logging.
The default W&B run name (and the seed of the evaluators' run ids) is `--run_name`, else the model directory's own
name. For a hand-run evaluation without a W&B login set `WANDB_MODE=disabled` (on a terminal, wandb otherwise asks
for one); `scripts/campaign/eval_arm.sh` does.

**Reasoning grading** (the cross-cutting invariant): the model generates thinking-ON; the
`LMEvalWrapper` strips everything through the **last `</think>`** (`s.rsplit("</think>",1)[-1]` in its
`generate_until`, when the config's `strip_reasoning` is on, which defaults to `enable_thinking`) *before* the task
grader sees text. Each `*_reasoning_utils.py` uses a
`keep_all` identity filter (all N sampled repeats reach `process_results`) and extracts the answer:
GPQA/MMLU-Pro `\boxed{}` → `Answer:` → last standalone letter; AIME last balanced `\boxed{}` +
normalized integer equality; LCB the last fenced `python` block, executed against tests.
`exact_match`/`pass_at_1` = mean of per-sample 0/1 = pass@1[avg-of-N]. **LCB executes generated code**
(`check_correctness` → fresh `multiprocessing.Process` running `_lcb_testing_util.run_test` under
SIGALRM + `reliability_guard`, its address space capped at the inherited size + 4 GiB): *not a sandbox; isolated node
only.* Tasks that execute model-generated code (the lm-eval tasks marked `unsafe_code`, i.e. LiveCodeBench and
CRUXEval, and the HF `code_eval` metric) run only with the explicit opt-in `DM_ALLOW_CODE_EVAL=1`, the only thing that
sets `HF_ALLOW_CODE_EVAL=1` and lm-eval's `confirm_run_unsafe_code`; without it such a task fails with a message that
names the variable.

**Perplexity: `ppl/`.**
- `memmap_ppl.py` (**the primary eval**): perplexity over a flat pre-tokenized token `.bin`
  (`np.memmap`), **tokenizer-free and download-free** (reads ids directly), multi-node data-parallel.
  `main()` → `dist_init()` → `_load_model()` (registers `dmlm`, then `from_pretrained(
  attn_implementation=$EVAL_ATTN_IMPL)`, `.cuda().eval()`) → non-overlapping `seq_len` windows
  `chunk_list_interleaved`-sharded across ranks → per-token `cross_entropy` on fp32 logits, summed per window, under
  `inference_mode` → `all_reduce` NLL & token counts → master writes `{run_name}_results.json` (to
  `--output_dir`) with `nll_per_token`/`perplexity`, and the per-window artifact `{run_name}_perwindow.npy` (window ids
  and mean NLL, for paired comparisons). One rank = one DP replica; idle ranks if
  `n_windows < world` (the global token-sum still reduces correctly). Uses HF `.forward` only (no
  `EngineModel`). **With `--wandb`, logs to the same wandb run as training:** `_maybe_log_wandb` reads
  `{model_dir}/wandb_run_id.txt` (written by the trainer's HF export) and, if present, resumes that run
  (entity `WANDB_ENTITY`, default the account's; project `WANDB_PROJECT`, default `delta-matching`) to log
  `eval/{perplexity,nll_per_token}`: so a model's train curve and its eval ppl land on one run (`train_arm.sh` passes
  `--wandb` to its end-of-run val CE when the run logged to W&B). Point `--output_dir`
  at `workspace/result/{run_name}` to co-locate the eval json with the trainer's `train_summary.json`
  (`train_engine.md §16`).
- `pg19_ppl.py` + `growing_window.py` + `pg19_dataset.py`: anchored growing-window perplexity over
  PG19 books (`evaluate_books_batched_growing_window`, `iter_pg19_books(num_books, split)`; `pg19_ppl.yaml` reads
  `split: auto`, the HF test split unless a local curated file exists), marker-first resume.
  `load.py::load_model_for_ppl` loads with an optional `max_position_embeddings` override.

**Resumable eval checkpointing.** Per `(model,task,length)` a `*_results.json` is the unit of progress,
read with `locked_read_json` and published with `locked_atomic_write_json` (portalocker lock + tmp +
fsync + `os.replace`, in `utils/ckpt.py`); completion signaled **last** by
`create_complete_marker(path, version_str)`. A version mismatch counts as incomplete (re-run);
`check_complete_marker` is a **collective** (two barriers; `clean_dir` defaults to False, and no caller deletes
anything through it): never call on a subset of ranks. RULER additionally checkpoints per `(length,task)`.

## 4. The inference engine: `src/inference/` (continuous batching; dmlm and Nemotron-H)

A continuous-batching engine, a drop-in for `modeling/wrapper.py::HFModel.generate`,
gated by `EVAL_ENGINE=mini_eval_dev` (`meta_eval.main` constructs `EngineModel(model)`). It
**reuses every HF weight module** and replaces only cache interaction, so it is numerically faithful by
construction.

**The Nemotron-H path.** `src/inference/layers.py` subclasses transformers' `NemotronHAttention` /
`NemotronHMamba2Mixer`, and `model.py::EngineNemotronH` rebinds `blk.mixer.__class__` onto the engine subclasses.
`EngineModel` imports this path (and with it transformers' `nemotron_h` module, `mamba_ssm` and `causal_conv1d`) only
when it wraps a Nemotron-H model, so `from inference.engine import EngineModel` works without them. The
architectural gift it exploits: 52 layers = 23 Mamba2 (O(1) recurrent state) + 23 MoE (stateless) + only **6**
attention layers (growing KV), and **NoPE** (no RoPE → slot KV needs no rotation). For the model
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (31.6B; attn at `{5,12,19,26,33,42}`, 32 q / 2 kv heads,
`n_rep=16`, `head_dim=128`, `hidden=2688`, `vocab=131072`).

**dmlm port: `EngineModel` drives the study's models too.** `EngineModel.__init__`
dispatches on `config.model_type`: `"dmlm"` → **`EngineDMLM`** (`inference/dm_model.py`) + **`DMStatePool`**
(`inference/dm_cache.py`), else the Nemotron path. The dmlm engine mixers (`inference/dm_layers.py`)
subclass our mixers (weights shared via `__class__` rebind) and override only the cache interaction:
**`EngineGQA`** (RoPE: reuses our exact `rotate_half`, stores rotated K, `flash_attn_with_kvcache`),
**`EngineMamba2`** (`selective_state_update`/`causal_conv1d_update`, slot-indexed in place),
**`EngineGDN`** (fla `fused_recurrent_gated_delta_rule` + the 3 q/k/v short-conv caches, per-slot
gather/scatter; recurrent state fp32), **`EngineMLA`** (the latent cache) and **`EngineKDA`** (fla
`fused_recurrent_kda` with the gathered state). Per-slot RoPE positions come from `pool.seqlens`; a NoPE model
(`nope` in its config) gets the identity rotation (cos 1, sin 0), as in the training forward. **Parity
(teacher-forced vs HF `.forward`):** GQA **bit-exact** (Δ=0), Mamba2 Δ=7.8e-3, GDN Δ=1.6e-2, all **100%
argmax**; 3:1 hybrids pass (mixed-layer dispatch + pool correct, prefill bit-exact); the KDA decode is checked by
`tests/test_new_mixers.py`. `PYTHONPATH=src python -m inference.verify_engine_parity <case>` (GPU; `--help` lists the
cases, `nope` included) reruns the check, and `tests/test_engine_parity.py` runs six of the cases. RULER runs
end-to-end through it (`EVAL_ENGINE=mini_eval_dev`). So long-context *generation* (RULER) uses the cache, not the
O(L²) recompute.

| file | role |
|---|---|
| `cache.py` | `HybridStatePool` (the Nemotron-H path): per-slot Mamba state (ssm/conv) + separate k/v KV pools, one contiguous buffer per layer (not paged); free-list allocator |
| `context.py` | `EngineContext`: the per-forward phase (all-prefill *or* all-decode) + slot ids, threaded into the rebound mixers |
| `layers.py` | `EngineMamba2Mixer` / `EngineAttention`: rebound onto the HF mixers via `__class__` swap (weights shared). Decode passes `state_batch_indices`/`conv_state_indices`/`cache_batch_idx` = slot ids; `flash_attn_with_kvcache` for decode, `flash_attn_func`/`mamba_chunk_scan_combined` for prefill |
| `model.py` | `EngineNemotronH`: explicit 52-layer loop reusing `blk.norm`/`blk.mixer`; MoE/MLP verbatim; fp32 logits |
| `scheduler.py` | `Scheduler`/`Request`: phased continuous batching: per-seq prefill + one batched 1-token decode/step; admit on free slot; optional cross-rank work-stealing |
| `sampler.py` | greedy=argmax; HF-faithful temp→top_k→top_p; stop = the model's eos id, else the fallback `[2, 11]` (`engine.py::_FALLBACK_EOS`), + suffix-match + max_new |
| `engine.py` | `EngineModel.generate` drop-in; memory-aware `max_num_seqs` (reserve scales with prompt len); echo-prompt reassembly; `generate_stealing` work-steal variant. **Dispatches `model_type=="dmlm"` → `EngineDMLM`+`DMStatePool`, else `EngineNemotronH`** |
| `dm_model.py` | **`EngineDMLM`**: dmlm backbone loop (analogue of `EngineNemotronH` for our `Transformer`); rebinds each block's mixer, threads per-slot RoPE cos/sin into the context (§4 dmlm port) |
| `dm_layers.py` | **`EngineGQA`** (RoPE via our `rotate_half`, rotated-K cache, `flash_attn_with_kvcache`) / **`EngineMLA`** / **`EngineMamba2`** (`selective_state_update`/`causal_conv1d_update`, slot-indexed) / **`EngineGDN`** (fla `fused_recurrent_gated_delta_rule` + 3 q/k/v short-conv caches, gather/scatter) / **`EngineKDA`**: subclass our mixers via `__class__` rebind |
| `dm_cache.py` | **`DMStatePool`**: per-slot caches sized from the loaded mixers: GQA / MLA k/v, Mamba2 ssm/conv, GDN and KDA recurrent (fp32) + conv; `mixer_kind()` dispatch |
| `verify_engine_parity.py` | teacher-forced parity regression, engine vs HF `.forward` per position, all mixers + 3:1 hybrids + a NoPE model (GQA bit-exact): `PYTHONPATH=src python -m inference.verify_engine_parity <case>` |

**State dtype.** Mamba SSM state is **bf16** by default, matching the HF cache dtype the eval baseline
used (fp32 would *diverge* from that oracle). `ENGINE_STATE_DTYPE=fp32` /
`EngineModel(state_dtype=torch.float32)` matches NVIDIA's serving (fp32 SSM state: may reduce
long-decode rambling on hard traces) at the cost of parity. Conv state always stays in the compute
dtype (`causal_conv1d_update` requires it).

**Measured on Nemotron-H (DP8, 32k budget, same data/hardware as the HF baseline):**

| benchmark | HF baseline | engine | speedup | score parity |
|---|---:|---:|---:|---|
| GPQA-diamond @32k | 60.1 @ 22,782 s | **61.7 @ 2,853 s** | **8.0×** | +1.6 (within noise) ✓ |
| AIME-2025 @32k | 48.3 @ 7,082 s | **48.0 @ 1,288 s** | **5.5×** | −0.3 (within noise) ✓ |
| **combined @32k** | **8.3 h** | **1.15 h** | **7.2×** | parity ✓ |

8× over the *best* HF config (batch-16): continuous batching at batch-64 saturates the GPU (~97% util).
**131072 fits a single 80 GB H100** (prefill 131k = 15 s, peak 77.3 GB): no `device_map`, so 131k
reasoning runs on DP8.

**Validation (Nemotron-H):** prefill logits bit-identical (max|Δ|=0), teacher-forced decode argmax
matches HF (rare misses are proven near-ties); RULER@4096 engine 88.99 vs baseline 89.03;
GPQA@32k parity + 8.0× (above); the 131k card match is *pending* (`*_nemotron_131k.yaml`).

## 5. Shared infra: `src/modeling/`, `src/utils/`

**`modeling/wrapper.py`**: `HFModel(nn.Module)`: the baseline wrapper whose `generate(...)` signature
**every other generator matches** (`HFModel.generate` in `modeling/wrapper.py`, `EngineModel.generate` in
`inference/engine.py`, same args: `input_ids, attention_mask, max_new_tokens, temperature, top_k, top_p,
do_sample, stop_token_list, pad_token_id, return_dict, return_unfinished, prefill_chunk_size, ...`,
returning a left-padded `(N, L+max_new)` tensor with the prompt echoed in `[:, :L]`). This is why the
harness is agnostic to which it constructs. `HFModel` supports `.forward` (loglikelihood/MC) **and**
`.generate`; `TokensStoppingCriteria` for token-level stops.

**`utils/`** (every module defines `__all__`, and `utils/__init__` re-exports exactly those names): `dist.py`
(torchrun lifecycle, `dist_print`, the NCCL-under-sharding `device_id` fix; `dist_init(cudnn_benchmark=False)`, the
default that meta_eval, memmap_ppl and pg19_ppl use, switches cuDNN off for the whole process), `ckpt.py`
(portalocker-locked atomic JSON + `.complete` markers), `io.py` (loaders by file type + `get_filename_without_ext`),
`config.py` (`build_config` with `include:`; a missing file or a YAML that is not a mapping raises), `misc.py`
(**`chunk_list_interleaved`** = THE DP sharding primitive, `chunk_list_by_size`, `val2tuple`, `get_device`,
`to_uppercase_only_letters`), `debug.py` (`master_print`). The eval `version_str` lives in `evaluation/constants.py`
(§9).

## 6. Configuration

- **Eval configs**: `src/evaluation/configs/*.yaml` (28): `type:` ∈ {`lm_eval_harness`, `ruler`, `recall`} for
  `meta_eval`, and `pg19_ppl` for the separate `-m evaluation.ppl.pg19_ppl` entry; plus
  `tasks`/`metric`/`lengths`/`max_length`/`enable_thinking`/`apply_chat_template`/`max_samples`. `build_config`
  supports an `include:` directive (the included file is the base; the current file overrides); config keys have no
  command-line override, and `meta_eval` rejects unknown flags. Families: reasoning (AIME/GPQA/LCB/MMLU-Pro), code
  (CRUXEval), general (BBH, commonsense, MMLU, TriviaQA), long context (RULER 4k–64k; LongBench, whose metrics need
  `jieba`, `fuzzywuzzy` and `rouge`, which the `eval` extra does not install), math, retrieval and recall
  (extraction, MQAR, phonebook). `*_131k.yaml` = card-match recipes.
- **lm_eval task defs**: `src/evaluation/lm_eval_tasks/*.yaml` (`output_type: generate_until`,
  `process_results: !function <util>`, `generation_kwargs`, `repeats` GPQA 8 / AIME 16 (64 in the 131k task) /
  LCB 8 / MMLU-Pro 1, the `keep_all` filter). RULER variant→family map = `ruler/config_tasks.yaml`. Found at
  runtime via the config's `include_path: src/evaluation/lm_eval_tasks`, resolved against the working directory: run
  from the repository root.
- **Env knobs** (besides the command-line flags of `meta_eval --help`): `EVAL_ENGINE=mini_eval_dev`,
  `ENGINE_STATE_DTYPE=fp32`, `ENGINE_MAX_SEQS` (64), `ENGINE_RESERVE_GB` (3), `ENGINE_PREFILL_BYTES_PER_TOK` (1.1e5),
  `ENGINE_PROGRESS`, `EVAL_ATTN_IMPL` (flash_attention_2|sdpa|eager), `EVAL_EXPERTS_IMPL`
  (eager|grouped_mm|batched_mm), `EVAL_DEVICE_MAP`, `EVAL_RULER_MAX_LEN`, `EVAL_DIST_TIMEOUT`,
  `EVAL_WORK_STEAL`, `EVAL_GENLEN_DIR`, `EVAL_TRUST_REMOTE_CODE` (default 0), `DM_ALLOW_CODE_EVAL` (1 = run
  model-generated code; it sets `HF_ALLOW_CODE_EVAL`), `LCB_{START,END}_DATE` (override the per-task windows:
  2024-07..2024-12 for `livecodebench_reasoning`, release_v5; 2024-08..2025-05 for the 131k task, release_v6) /
  `LCB_TIMEOUT`, `WANDB_{MODE,DISABLED,RESUME,RUNTYPE,ENTITY,PROJECT}` (read by the logging path, with defaults).

## 7. Entry points & commands

**PYTHONPATH:** `src` (the harness and every dmlm checkpoint). The campaign's chain is
`scripts/campaign/eval_arm.sh <run>` (docs/benchmark.md).

**Memmap perplexity (the primary metric), single node:**
```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1,..,7 \
  torchrun --standalone --nproc_per_node=8 -m evaluation.ppl.memmap_ppl \
  --model_name_or_path <hf_ckpt_dir> --data <val.bin> --dtype uint16 \
  --seq_len 8192 --max_windows 512 --batch_size 4 --output_dir <out>
```
**Multi-node (4 nodes / 32 ranks)** (model + data on shared storage or pre-seeded to each node's scratch):
```bash
PYTHONPATH=src torchrun --nnodes=4 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=<head>:29531 -m evaluation.ppl.memmap_ppl \
  --model_name_or_path <hf_ckpt_dir> --data <val.bin> \
  --dtype uint16 --seq_len 8192 --max_windows 512 --batch_size 4 --output_dir <out>
```
(Set `EVAL_ATTN_IMPL=sdpa` to force the GQA SDPA path; add `--wandb` to log `eval/*` into the model's training W&B
run.)

**Text tasks** (need a tokenizer in the checkpoint dir, once the model is `dmlm`), from the repository root:
`PYTHONPATH=src torchrun --nproc_per_node=8 -m evaluation.meta_eval --model_name_or_path <ckpt> --eval_config
src/evaluation/configs/<cfg>.yaml` (`pretrain` must be importable; `EVAL_TRUST_REMOTE_CODE` stays at its default 0;
`WANDB_MODE=disabled` without a W&B login). PG19: `-m evaluation.ppl.pg19_ppl`.

**Nemotron-H engine reasoning @32k** (the inference engine's native use):
```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 EVAL_ENGINE=mini_eval_dev ENGINE_MAX_SEQS=64 EVAL_EXPERTS_IMPL=eager \
  torchrun --nproc_per_node=8 -m evaluation.meta_eval \
    --model_name_or_path <nemotron-h-instruct> \
    --eval_config src/evaluation/configs/gpqa_nemotron.yaml --eval_batch_size 256 --generation_num_chunks 1
```
(131k card-match: `*_nemotron_131k.yaml` + `ENGINE_RESERVE_GB=5`. The lm_eval task YAML's `max_gen_toks`
caps the budget: raise it there for 131k, not just `--max_new_tokens`.)

## 8. Tests / validation (verified)

- **HF round-trip bit-exact** (Δlogits=0) for every bf16 mixer: `src/pretrain/scripts/verify_hf_export.py`, `tests/test_new_mixers.py`.
- **train → HF export → eval-load → ppl**, single node: a gqa checkpoint, flash-attn, ppl computed.
- **4-node memmap-ppl (32 ranks), random data**: 512 windows × 256 tok: **32-rank result == single
  node** (DP reduction correct), result json on shared NFS.
- **End-to-end on a *real trained* checkpoint (the closing loop).** The 360M GQA model trained 4-node
  (`docs/train_engine.md §14`) was exported (node-local; seed each
  node's scratch for a multi-node read), then evaluated 4-node (32 ranks) on the held-out
  `val.bin`: **121 windows × 2048 tok = 247,808 tokens → nll/tok = 7.275, ppl = 1443.6**: **coherent
  with training** (last train CE 7.42 @ step 25 → val nll 7.27 @ step 30). So train → HF-export →
  multi-node-eval is verified on a model the engine actually produced (high ppl only because it is a
  30-step smoke run; the test is *pipeline correctness + cross-node DP*, not model quality).
  Reproduce: `PYTHONPATH=src torchrun --nnodes=4 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=<head>:29531 -m
  evaluation.ppl.memmap_ppl --model_name_or_path <hf_ckpt_dir> --data <val.bin>
  --dtype uint16 --seq_len 2048 --max_windows 512 --batch_size 4 --output_dir <out>`.

## 9. Conventions, invariants, gotchas

- **Train ↔ eval is decoupled through an HF checkpoint**: `src/pretrain` imports nothing from this
  engine; it only `save_hf_checkpoint`s. Everything here reuses loaded HF modules → parity by
  construction. Put `hf_save_dir` + eval results on **shared storage**; DCP resume shards may stay node-local.
- **`EngineModel` is generate-only and rebinds the HF mixers IN PLACE.** Supports Nemotron-H **and** dmlm
  (GQA/MLA/Mamba2/GDN/KDA; §4): `EngineModel(hf_model)` swaps each `blk.mixer.__class__` onto the engine subclass,
  so the plain HF `.forward` (loglikelihood) on that object is **no longer valid**: only `.generate`.
  **Run any HF `.forward` oracle BEFORE constructing the engine** (this bit the parity test). Use it for
  generate_until/ruler (set `EVAL_ENGINE=mini_eval_dev`); perplexity/loglikelihood/recall use the HF path /
  `memmap_ppl`.
- **`memmap_ppl` is the perplexity primary**: tokenizer-free, download-free, multi-node DP, HF
  `.forward`.
- **fp32 Mamba state** (`ENGINE_STATE_DTYPE=fp32`) is required to match Nemotron card numbers; bf16
  diverges at long context.
- **`</think>` stripping happens in the harness wrapper** (`wrapper.py`), *before* the task graders. If
  `</think>` never appears (budget exhausted) → extraction fails → score 0 (intended).
- **`chunk_list_interleaved(items, world)[rank]`** is THE DP sharding primitive (strided, not block);
  RULER pads the work list to a multiple of `world` first to avoid gather hangs.
- **Completion-marker `version_str`** (`evaluation/constants.py`) is the cache-invalidation key for
  `.complete` markers and the W&B `run_id` (`sha1(run_name+version_str)`); bumping it forces re-eval.
  `check_complete_marker` is **collective**: never call on a subset of ranks.
- **NCCL init under sharding** (`utils/dist.py`): with `device_index` set (device_map path) it pins the
  device + passes `device_id=` before `init_process_group` so NCCL doesn't equate device-id with global
  rank (hangs under per-rank `CUDA_VISIBLE_DEVICES`). `EVAL_DIST_TIMEOUT` raises the 1h default.
- **`EVAL_EXPERTS_IMPL=eager`** dodges a `grouped_mm` CUDA illegal access at S=262144; **`EVAL_RULER_
  MAX_LEN`** caps RULER ≤131072 (the 262144 bucket int32-overflows the `mamba_ssm` kernel).
- **LCB code execution** mutates process globals irreversibly (`reliability_guard`) → fresh child
  process per run, with its address space capped; it is **not** a security sandbox: isolated node only, and only
  with `DM_ALLOW_CODE_EVAL=1`.
- **Env deps beyond the core dependencies:** `portalocker` (eval `utils.ckpt`) + `datasets` (already in
  `pyproject` core); the lm-eval tasks need the `eval` extra (`lm_eval`); RULER needs nothing beyond it. The engine
  runs from the repository's one environment (not its own).
- **`lm_eval_harness` was written for a PATCHED lm_eval fork; the env has VANILLA `lm_eval` 0.4.12**
  (**4 bug classes, fixed** when lm_eval first ran in this repository; until then only perplexity and recall had been
  used, so none of these had fired). All in `wrapper.py`/`evaluator.py`:
  (1) **device**: `LMEvalWrapper(LM)` did `self.device = device`, but `LM` exposes `device` as a
  **getter-only `@property`** → `AttributeError`. Fix: store `self._device`; override `device` → `get_device(self.model)`.
  (2) **multi-rank collectives**: `LM.all_gather`/`gather_object` are **no-ops** (`return tensor` / `[obj]`);
  fine at world_size 1 but at DP>1 lm_eval's `evaluator.py` does `max(lm.all_gather(count))` → `TypeError:
  'int' object is not iterable`, and `gather_object` silently drops other ranks. Fix: override both on the
  wrapper with real `dist.all_gather` (→ `[world_size]` tensor) / `dist.all_gather_object` (→ per-rank list).
  **This is why single-GPU smoke PASSED but 8-GPU DP FAILED: smoke the real topology.**
  (3) **runtime_cache**: `evaluator.py` passed `return_runtime_cache=` + `**runtime_cache` (fork-only
  prefill-reuse) → unexpected-kwarg `TypeError`. Fix: gate behind `"return_runtime_cache" in
  inspect.signature(simple_evaluate).parameters` (degrades gracefully; MC loglikelihood doesn't need it).
  (4) **WANDB_RESUME**: `evaluator.finish_and_log` did `os.environ["WANDB_RESUME"]` (required) → `KeyError`
  when the 4 `WANDB_*` vars aren't pre-exported (every other eval path uses `.get(...)` with defaults). Fix:
  `.get` with the same defaults + `dir=$DM_WORKSPACE`. → the commonsense suite of that time
  (`arc_c/arc_e/piqa/winogrande/obqa/boolq/truthfulqa_mc2`, not today's CS9 of `commonsense.yaml`) then ran at 8-GPU
  DP (avg ~0.47–0.48 for the 360M model).

## 10. Limitations / future work

- **Continuous-batching `EngineModel` covers dmlm GQA / MLA / Mamba2 / GDN / KDA (§4).** Each has an
  engine mixer (`inference/dm_layers.py`), parity-verified vs HF `.forward` (GQA bit-exact). Use it for
  dmlm long-context *generation* (RULER) via `EVAL_ENGINE=mini_eval_dev`.
- **dmlm `.generate` (the plain HF path, `EVAL_ENGINE` unset) is cache-free**: each decode step re-runs
  the *full* forward (G× redundant): per re-forward O(L²) for the attention layers (all of GQA; the 6/24
  attention layers of the 3:1 hybrids), O(L) for the recurrent layers. So decode ≈ G×(prefill); the prefill
  itself is fine. **`EngineModel` removes the redundancy** (KV cache → O(G·L) attn decode; recurrent-state
  cache → O(G·L) Mamba2/GDN): that's the reason to use it for long-context RULER.
- Text-task evals (PG19, commonsense, RULER) need the model's **tokenizer** saved in the checkpoint and
  a vocab match; the study uses the 32k Llama-2 tokenizer.

## 11. Open questions / not-yet-verified

- `inference/layers.py` mirrors transformers' `NemotronHMamba2Mixer.cuda_kernels_forward` and
  `NemotronHAttention.forward` of the pinned version (5.9.0); check them again after a transformers upgrade.
- `ENGINE_PREFILL_BYTES_PER_TOK`/`ENGINE_RESERVE_GB` defaults are empirical (~110 KB/token at 131k);
  accuracy at other lengths unverified.
- **Provenance:** the GPQA / AIME / RULER parity and speed numbers of §4 were measured on Nemotron-H in the engine's
  upstream repository, **before** it moved into this one: re-confirm them here if you rely on them. (The bridge
  tests of §8 *were* run in this repository.)
