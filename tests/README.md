# tests/: environment, data, kernel, launcher, training and evaluation checks

```
pytest tests -m "not gpu"            # any machine, no GPU: the environment and its pins, CPU logic (loaders, Muon, BOS
                                     #   options, signal handling), the arm table, the launcher / stage-2 / experiment /
                                     #   eval-chain scripts in dry mode, the data scripts on synthetic data, the paired
                                     #   comparison; + the data root, tokenizer, RULER and caches when they are found
pytest tests                         # a GPU node (Hopper): + the Delta-Matching kernels vs fp32, the mixers, 3-step
                                     #   trainings of the arms, resume and launcher chains, the eval engine, HF conversion
pytest tests -m "not slow"           # skip the subprocess trainings and evaluations (minutes each)
DM_TEST_GPU=3 pytest tests/test_train_smoke.py     # the GPU the subprocess smokes run on (default 0)
```

Markers: `gpu` (needs a CUDA GPU; skipped when none is visible), `slow` (launches training or evaluation
subprocesses), `data` (needs the data root; skipped when none is found and `DM_DATA_ROOT` is unset). A test that needs
the tokenizer or the data root through a fixture skips when nothing is configured and nothing is found, and fails when
`DM_DATA_ROOT` / `DM_TOKENIZER_DIR` is set but incomplete.

What the CPU subset needs: the Python environment of docs/SETUP.md section 2 (torch, transformers, lm_eval, fla, ...),
`bash`, `tar` and GNU `cp`. No GPU, no network, no data: the launchers and experiment drivers run in their dry modes,
and every fake workspace, export or data file lives in pytest's temporary directory. `test_env.py` checks the pinned versions
(`env/requirements-cu130-hopper.lock.txt`), `nvcc` 13 and the `delta_matching` kernel package; the tests that need the
kernel package fail, not skip, when it is missing (docs/SETUP.md section 3).

Environment variables (locations follow `scripts/campaign/env.sh`):

| variable | default | used for |
|---|---|---|
| `DM_WORKSPACE` | `<repo>/workspace` | scratch; the smoke runs write to `$DM_WORKSPACE/tests_tmp/` (removed at the end of the session) |
| `DM_DATA_ROOT` | `$DM_WORKSPACE/data` | the training / val streams, RULER, the HF caches (data/README.md) |
| `DM_TOKENIZER_DIR` | `<data root>/tokenizer/llama2_32k` | the tokenizer |
| `HF_HOME` | `<data root>/hf_cache` | the eval datasets cache |
| `DM_PY`, `DM_TORCHRUN` | the interpreter running pytest, and `torchrun` beside it | the subprocesses |
| `DM_SO` | the kernel package's own library | the delta_matching library to load; on a `DM_BF16_DP=1` build (a study variant of the kernel package) the stale / matched kernel and training tests skip, since that build refuses them by design, and the bf16_dp flag check runs in the other direction |
| `DM_TEST_GPU` | `0` | the GPU of the subprocess tests; in-process GPU tests (test_kernels, test_new_mixers, test_muon, test_env) use GPU 0 of `CUDA_VISIBLE_DEVICES` |
| `DM_KEEP_TEST_RUNS` | `0` | `1` keeps `$DM_WORKSPACE/tests_tmp/` |
| `DM_TEST_EVAL_CONFIG` | `tiny_hybrid.yaml` | the tiny yaml of `tests/configs` whose export the eval and HF conversion tests use (e.g. `tiny_kda.yaml`) |
| `DM_TEST_BASH` | unset | a second bash (e.g. the training image's 5.2) for test_launcher_signals.py |

The subprocesses never inherit the launcher, trainer, kernel or evaluator knobs of the calling shell (`DM_*` except
the kernel library's location `DM_SO`, `DM_JIT_DIR`, `DM_FA_SRC`; `S2_*`, `SEEDS`, `DRY`, `TAG`, `EXTRA_ARGS`, `SMOKE`,
`STOP_AT_STEP`, `NUM_STEPS`, `EVAL_ENGINE`, ...): each test sets the ones it needs (tests/_helpers.py `base_env`,
`dry_env`). The dry launcher checks also run with every GPU hidden (`CUDA_VISIBLE_DEVICES` empty).

| file | marker | what a failure means |
|---|---|---|
| test_env.py | (cpu; gpu for the kernel library) | wrong Python/torch/CUDA build, a package off its pin, nvcc not 13, a repo import broken, the `delta_matching` package missing or its library built without the head_dim-128 instantiations |
| test_data.py | data | the data root is incomplete or corrupt (sizes and sha256 against data/checksums.json, token ranges, doc-length sums, tokenizer ids, RULER tasks, HF caches) |
| test_data_recipe.py | (cpu) | the data scripts of `data/` are broken: the stage-1 tokenization, the document-level split, the stage-2 re-tokenization, packing, flattening and validation carve, or the checksum verifier, on small synthetic inputs |
| test_experiments.py | (cpu) | an experiment script of `scripts/experiments/` runs the wrong arms, seeds or commands, accepts an arm outside its experiment, or its driver mishandles DRY / SMOKE / SEEDS |
| test_arms.py | (cpu) | `scripts/campaign/arms.sh` gives an arm the wrong run name, config, attention core, recurrent layers, geometry, dm_mode, fp8 scope, NoPE or Muon flag, an arm is missing from its header, or docs/SETUP.md disagrees with it |
| test_model_guards.py | (cpu) | the dm mixer builds a configuration it cannot run (bf16_dp or YaRN at head_dim 256, another head_dim, the plain route, fused RoPE), the bf16 MLA mixer accepts an fp8 attn_impl, the trainer accepts `--backend ddp` with a Delta-Matching route, the HF wrapper's `logits_to_keep` keeps the wrong positions, or the analytic parameter count is wrong |
| test_te_fp8.py | (cpu) | the te_fp8 arm's guards (head_dim 256, Transformer Engine missing, the launcher without its TE environment) do not refuse |
| test_kernels.py | gpu | the Delta-Matching-128 kernel does not match fp32 attention within the fp8 envelope, is non-deterministic, the model stack (fla, causal_conv1d, torchao) does not run, or the dm mixer does not build from the package or does not check the bf16_dp build flag (in both directions) |
| test_new_mixers.py | gpu (parity test cpu) | the MLA, Mamba2 or KDA hybrid is broken: parameter parity, MLA fp8 core vs bf16 core, HF round trip, the fused paths, the inference engine's KDA decode |
| test_engine_parity.py | gpu | the inference engine's prefill / decode differs from the HF forward for a mixer family (GDN, GQA, MLA, Mamba2 and KDA hybrids, NoPE) |
| test_train_smoke.py | gpu, slow, data | the trainer cannot run an arm end to end (fp8 pipeline at head_dim 128 and 256, compiled MLP, export), or an fp8 run does not announce the requested dm_mode (and the consistent-dO arm its delta from the dO codes) |
| test_resume.py | gpu, slow, data | the resume pipeline is broken: periodic checkpoints, rotation and exact resume, SIGTERM stop (checkpoint + exit 75) and resume, deadline stop, the idempotent finish, the metrics log events |
| test_launcher.py | gpu, slow, data | `scripts/campaign/train_arm.sh` cannot run an arm end to end under `env` launch (the attention core each arm trains, the consistent-dO, NoPE and Muon switches, the export patch), forward a SIGTERM, write `TRAIN_COMPLETE`, or skip a finished run |
| test_trainer_units.py | (cpu) | checkpoint rotation keeps more or fewer than `save_total_limit` checkpoints or touches a foreign directory, a save or a resume changes the data stream (epoch boundaries, persistent workers), an empty epoch is accepted, or the document-aware loader accepts a stream that does not match its sidecar |
| test_launcher_signals.py | (cpu) | the signal-forwarding block of train_arm.sh aborts, forwards twice or loses a rank's exit status |
| test_dataloader_sigterm.py | (cpu) | a data-loader worker dies of a scheduler's SIGTERM or blocks the exit |
| test_eval_smoke.py | gpu, slow, data | the val-CE scorer, the lm-eval path or the engine (RULER, TriviaQA) cannot score an export, or the engine did not serve the generation runs |
| test_eval_units.py | (cpu; the RULER case needs the tokenizer) | the PG19 growing-window scorer mis-scores a model that ignores `logits_to_keep`, the plain HF path generates padded batches for recurrent dmlm models, RULER does not resume or clears another config's results, or meta_eval ignores an unknown flag |
| test_eval_configs.py | (cpu) | an eval config does not parse or shares its name, eval_arm.sh checks markers meta_eval does not write, `utils` re-exports the wrong modules, or a reasoning task's answer extractor regressed |
| test_eval_arm.py | (cpu) | `scripts/campaign/eval_arm.sh` accepts an export not patched for the clean-kernel eval, or re-runs a completed evaluation |
| test_eval_bos.py | (cpu; tokenizer) | the `--add_bos` prompt option of the evaluators is broken |
| test_paired_analysis.py | (cpu) | the paired per-window comparison of two arms (`paired_analysis.py`, `paired_pair.py`) is wrong |
| test_hf_standalone.py | (cpu: the refusals, and a tiny conversion with the tokenizer); gpu, slow, data: the round trip | `scripts/hf/make_standalone.py` ships an unpatched or foreign export, or the standalone model differs from this repository's loader |
| test_prepend_bos.py | (cpu) | the BOS-anchored window option of the data loaders is broken |
| test_muon.py | (cpu); gpu (2 GPUs for the FSDP2 case) | the Muon optimizer differs from torch.optim.Muon + AdamW, its state does not round-trip, or its FSDP2 path differs from the unsharded reference |
| test_stage2.py | gpu (slow + data for the warm starts); the dry launcher checks are cpu | the context-extension path is broken: the YaRN temperature on the fused mixer, the per-head backward split, `--init_from_hf` warm starts at 8192, the stage-2 launcher (its arms, the warm start, the exact 28,000-step schedule) |
