# Setting up and running the Delta-Matching experiments (Hopper, CUDA 13.0, uv)

The target is **NVIDIA H100 / H200 (sm_90a), the CUDA 13.0 toolkit, Python 3.12, PyTorch 2.12.0+cu130.** The fp8
attention kernel (the `delta_matching` package) is an sm_90a build and does not run on other GPUs (no Blackwell path
exists).

Read in order: section 1 (what you need), 2 (the Python environment with uv), 3 (the kernel package), 4 (data),
5 (configuring, smoke-testing and launching a run), 6 (evaluation), 7 (the test suite), 8 (stopping and resuming),
9 (pitfalls).

## 1. Requirements and layout

- **Hardware.** 8 x H100 80 GB (or H200) per node for the experiment runs: one arm per node; a 28,610-step run of the
  1.665 B hybrid (`qwen35_m.yaml`) takes 23 to 26 hours on H100 (step times in section 5).
  The CPU tests need no GPU and the GPU tests one (section 7); a launcher smoke of a real arm starts `DM_NPROC` ranks
  (default 8). The data loader is memmap-based and light.
- **Software on the host.** An NVIDIA driver for CUDA 13 (580.x), the CUDA 13.0 toolkit at `/usr/local/cuda-13.0`
  (`nvcc` release 13.0 is needed to build the kernels and by Triton/Inductor; any location works with `CUDA_HOME`), git,
  tar, `uv`, and the `hf` command of `huggingface_hub` (in the lock) for the data downloads.
- **Scratch.** Every heavy artifact (data, checkpoints, exports, evals, caches) lives under the workspace,
  `DM_WORKSPACE` (default `<repo>/workspace`, usually a symlink to node-local or shared scratch):

      ln -s /path/to/scratch workspace

  The repository itself holds code only. The finished data root takes about 165 GB (the build needs more while it
  runs: `data/README.md`), plus ~35 GB per run for two retained checkpoints and the export, plus ~1 GB per eval.
- **Layout after setup** (`scripts/campaign/env.sh` encodes these defaults; every one can be overridden):

      workspace/
        data/                     the data root (section 4): DM_DATA_ROOT
        qwen35_m/{ckpts,hf,eval}/ per-run checkpoints, HF exports, evaluation artifacts
        result/                   per-run train_summary.json
        logs/                     trainer, launcher and eval logs
        triton_cache/ tilelang_cache/  compiler caches

## 2. The Python environment with uv

The pinned, tested set is `env/requirements-cu130-hopper.lock.txt`: the `pip freeze` of the environment the study's
runs were produced with, minus three lines that cannot be installed from it (its header says which). Install in this
order; the order matters because several packages compile against the installed torch.

    uv venv .venv --python 3.12 && source .venv/bin/activate
    uv pip install "torch==2.12.0" "torchvision==0.27.0" --index-url https://download.pytorch.org/whl/cu130
    uv pip install ninja packaging setuptools wheel psutil

Then the packages that are built against torch, each with `--no-build-isolation` so they see the installed torch
(set `MAX_JOBS=32`, `CUDA_HOME=/usr/local/cuda-13.0`, `TORCH_CUDA_ARCH_LIST="9.0a"`; each build takes 10 to 60 min):

| package (version in the lock) | what it provides | how to get it |
|---|---|---|
| `flash_attn==2.8.3.post1` | FlashAttention-2 (the `flash` attn_impl, the evaluation's attention, the engine's KV cache) | wheel or `uv pip install flash-attn==2.8.3.post1 --no-build-isolation` |
| `flash_attn_3==3.0.0` (module `flash_attn_interface`) | FlashAttention-3 for Hopper: the bf16 attention of the head_dim-256 Standard arms (`*clean_hd256`) | `git clone https://github.com/Dao-AILab/flash-attention && cd flash-attention/hopper && MAX_JOBS=32 uv pip install . --no-build-isolation` |
| `causal_conv1d==1.6.2.post1`, `mamba_ssm==2.3.2.post1` | the GatedDeltaNet / Mamba2 mixers' conv and scan kernels | `uv pip install causal-conv1d==1.6.2.post1 mamba-ssm==2.3.2.post1 --no-build-isolation` |
| `flash-linear-attention==0.5.2` + `fla-core==0.5.2` | GatedDeltaNet and KDA (18 of the 24 layers of the hybrids); 0.5.2 also runs GDN with more value heads than key heads, which the 5.289 B config needs (fla 0.5.0 refuses its backward on Hopper) | from the lock |
| `tilelang==0.1.8` | the TileLang backend fla needs for the GatedDeltaNet backward on Hopper (it refuses Triton >= 3.4 without it) | from the lock |
| `liger_kernel==0.8.0` | fused RMSNorm, RoPE, linear cross-entropy | from the lock |
| `torchao==0.17.0` | fp8 `Float8Linear` for the FFN and q/k/v/o projections | from the lock |
| `triton==3.7.0` | comes with torch 2.12; the cast glue (`src/kernel/fp8glue`) and the fused producers are Triton | with torch |
| `transformers==5.9.0`, `datasets==5.0.0`, `tokenizers==0.22.2`, `huggingface_hub==1.16.4`, `safetensors==0.8.0`, `sentencepiece==0.2.1` | the HF export/eval bridge and the data caches | from the lock |
| `lm_eval==0.4.12` | the commonsense, extraction, TriviaQA and MMLU benchmarks | from the lock |
| `wandb==0.27.0` | training curves (project `delta-matching` by default; any entity, or `DM_WANDB=0`) | from the lock |
| `numpy==2.4.6`, `PyYAML`, `einops`, `portalocker`, `pandas`, `matplotlib`, `scipy` | plumbing, analysis | from the lock |
| `pytest` (9.1.1 in the reference environment) | the test suite | `uv pip install pytest` (not in the lock) |

Everything else installs from the lock in one go, minus the torch and source-built lines above:

    grep -v -E '^(torch|torchvision|triton|flash_attn|flash_attn_3|causal_conv1d|mamba_ssm)==|^nvidia-' \
        env/requirements-cu130-hopper.lock.txt > .venv/requirements-rest.txt
    uv pip install -r .venv/requirements-rest.txt

The pattern is anchored on `name==`, so `torchao`, `torch_c_dlpack_ext` and `tilelang` stay in. The excluded `nvidia-*`
lines are the CUDA libraries that come with the torch wheel (cuDNN 9.20, NCCL 2.29, the CUDA 13.0 runtime). Any
environment with the same packages works (conda included); nothing in the repository depends on the environment manager.
(`scripts/build_kernels.sh` is a `uv` recipe for a development environment at a newer torch; it is not the
environment of the study. It builds in a fresh private temporary directory unless `BUILD_DIR` names a directory of
your own.)

The repository is used in place, not installed: `scripts/campaign/env.sh` sets `PYTHONPATH=<repo>/src`, and the
modules are imported as `pretrain.*`, `evaluation.*`, `inference.*`, `kernel.*`.

Transformer Engine (the `te_fp8` arm: cuDNN's fp8 fused attention with TE's DelayedScaling recipe) needs a separate
environment (TE 2.20 built against the same torch, cudnn-frontend 1.28, cmake 4.4, pybind11); it is not required for
the other arms. `scripts/build_kernels.sh` step 7 has the build flags (a failed TE build
there only warns); the launcher runs that arm through `DM_TE_PY` / `DM_TE_TORCHRUN`.

## 3. The kernel package

The fp8 attention kernel of the Delta-Matching arms is the separate **`delta-matching`** package (its own repository,
`https://github.com/jet-ai-projects/delta-matching`: C++/CUDA, CUTLASS/CuTe, a flash-attention-style drop-in; import name
`delta_matching`). Install it into the environment of section 2 from its checkout:

    export CUDA_HOME=/usr/local/cuda-13.0
    git clone https://github.com/jet-ai-projects/delta-matching && cd delta-matching
    git submodule update --init third_party/flash-attention                  # the pinned flash-attention headers
    git -C third_party/flash-attention submodule update --init csrc/cutlass   # and its CUTLASS
    MAX_JOBS=8 uv pip install --no-build-isolation .                         # the build imports the installed torch

`git clone --recursive` works too, but it also pulls about 1.3 GB of ROCm submodules. The build needs `nvcc` 13 and an
installed torch and ninja; it takes about 1.5 minutes with `MAX_JOBS=8` on a large node. In an environment that has
`pip`, `pip install --no-build-isolation .` is the same. The kernel's README has the API, its input contract (it
raises `ValueError`, `TypeError` or `NotImplementedError` naming the broken constraint) and its runtime knobs.

The default build is the certified flag set of the study (Delta-Matching-128 v3 for head_dim 128, Delta-Matching-256
v1 for head_dim 256; one binary serves both). It serves every fp8 arm of the experiments (stale, matched and the
consistent-dO arm, at both head dims). The trainer imports the kernel from the installed package; no binary paths are
pinned anywhere in this repository (`DM_SO` can point the package at another build of its library).

**nvcc builds are not byte-reproducible**, so a rebuilt library is verified by numerics: `pytest tests/test_env.py
tests/test_kernels.py` loads it, checks that it is a head_dim-128 build with gmem twins (`build_hd128() == 1`,
`build_gt() == 1`), runs both delta modes against fp32 attention within the fp8 envelope, checks determinism and the
structural identities between the modes, and builds the `dm` mixer from it. `python scripts/verify_env.py` (GPU)
prints the build flags of the loaded library, and the kernel repository's own `pytest tests/` checks the kernel alone.

## 4. Data

The data root (`DM_DATA_ROOT`, default `workspace/data`) is built from the original sources by the scripts in `data/`;
`data/README.md` is the guide, with one document per training corpus (`data/pretrain_stage1.md`,
`data/pretrain_stage2.md`). After the build the root holds:

| path | what | size |
|---|---|---|
| `nemotron_cc_v2d1_hq_dqa/nemotron_rs_train.bin` (+ `.doc_lengths.npy`) | the stage-1 training stream: Nemotron-CC-v2.1 HQ + HQ-DQA, Llama-2 32k ids, uint16, 39.66 B tokens | 79.3 GB |
| `nemotron_cc_v2d1_hq_dqa/nemotron_rs_val.bin` (+ `.doc_lengths.npy`) | the 2 % document holdout, 807 M tokens; the val-CE protocol scores its first 2000 x 8192 windows | 1.6 GB |
| `prolong_64k_llama2/prolong64k_{train,val}.bin` (+ `.doc_lengths.npy`) | the stage-2 corpus: ProLong's long-context data re-tokenized with the Llama-2 tokenizer, 65,536-token windows, 37.16 B tokens + 256 validation windows | 74.5 GB |
| `tokenizer/llama2_32k/` | the tokenizer (`--tokenizer_dir`; baked into every export) | 2 MB |
| `ruler-500-llama2/` | RULER regenerated with this tokenizer, 13 tasks x 500 examples at 2k .. 128k (`docs/benchmark.md` section 3) | 7.9 GB |
| `hf_cache/` | `HF_HOME` with the evaluation datasets | 0.3 GB |

`python data/verify.py --root $DM_DATA_ROOT` checks sizes and sha256 of every file against `data/checksums.json`, and
`pytest tests/test_data.py` runs them as tests (every size and the small files' sha256; every sha256 with
`DM_TEST_SHA256=1`). The launchers pass the data paths explicitly
(`--data.paths`, `--data.doc_lengths`, `--tokenizer_dir`, `HF_HOME`), so the `workspace/...` paths inside the yaml
configs are never used by a launched run; if you run the trainer by hand, pass the same overrides.

**Terms of the data.** The data root holds third-party material under its own terms, which the code license does not
cover: the Llama 2 tokenizer (`tokenizer/llama2_32k`, also baked into every export) is under the Llama 2 Community
License; the stage-1 streams are derived from NVIDIA's Nemotron-CC-v2.1 (the NVIDIA Data Agreement for Model Training:
internal model training only, no redistribution), the stage-2 corpus from ProLong's long-context data
(`princeton-nlp/prolong-data-512K`, a mixture whose parts keep their own licenses; decoding it takes the Llama 3
tokenizer, under the Llama 3 Community License); `hf_cache/` holds copies of the evaluation datasets, each under the
license of its own dataset card. `data/README.md` lists them. Build the data yourself; do not re-publish it.

## 5. Configure, smoke-test, launch a run

All launch settings come from `scripts/campaign/env.sh` (sourced by the scripts). Put your values in your shell
profile or a wrapper:

    export DM_WORKSPACE=/path/to/repo/workspace          # default: <repo>/workspace
    export DM_DATA_ROOT=$DM_WORKSPACE/data              # default
    export DM_PY=/path/to/.venv/bin/python               # default: the repo's .venv, else `python` on PATH; DM_TORCHRUN defaults to torchrun beside it
    export DM_NPROC=8                                    # GPUs per node
    export WANDB_ENTITY=... WANDB_PROJECT=...             # or DM_WANDB=0
    export CUDA_HOME=/usr/local/cuda-13.0                # wherever the CUDA 13.0 toolkit is

The paper's experiments are the scripts of `scripts/experiments/` (`scripts/experiments/README.md` maps each to the
paper). Each trains, then evaluates, every arm and seed of one experiment in turn on one node, and continues where it
stopped when run again:

| script | arms | seeds |
|---|---|---|
| `stage1_hd128.sh` | the head_dim-128 cell: `clean_hd128`, `dm_stale_hd128`, `dm_matched_hd128`, `dm_consistent_hd128`, `te_fp8` | 11785, 2 |
| `stage1_hd256_scale.sh` | head_dim 256 at 0.569 / 1.665 / 5.289 B: `{xs_,,l_}clean_hd256`, `{xs_,,l_}dm_stale_hd256`, `{xs_,,l_}dm_matched_hd256` | 11785, 2 |
| `stage1_architectures.sh` | `mla_*`, `kda_*`, `mamba2_*`, `gqa_*` (clean / dm_stale / dm_matched, head_dim 128) | 11785, 2 (pure GQA: 11785, 1) |
| `stage1_nope.sh` | `nope_clean_hd128`, `nope_dm_stale_hd128`, `nope_dm_matched_hd128` | 11785, 2 |
| `stage1_muon.sh` | `muon_clean_hd128`, `muon_dm_stale_hd128`, `muon_dm_matched_hd128` | 11785 |
| `stage2_context.sh` | the 8K to 64K extension of the head_dim-128 cell: `clean_hd128`, `dm_matched_hd128`, `dm_stale_hd128`, `te_fp8` | 11785, 2 |

`DRY=1` prints an experiment's plan (arms, seeds, run names, commands); arms given on the command line restrict it;
`SEEDS="..."` overrides the seeds. To use several nodes, give each its own arms or seeds, e.g.
`scripts/experiments/stage1_hd128.sh dm_matched_hd128` on one node and `... dm_stale_hd128` on another.

The arms of the head_dim-128 cell and their run names (`scripts/campaign/arms.sh` has every arm):

| arm | `train_arm.sh` name | attention core | run name |
|---|---|---|---|
| Standard BF16/FP32 Mix-precision | `clean_hd128` | cuDNN bf16 SDPA + fp8 FFN | `Q35M_clean_sdpa_ffn8_hd128_<seed>` |
| Delta-Matching | `dm_matched_hd128` | Delta-Matching-128, matched delta, full fp8 pipeline | `Q35M_dm_matched_ffn8_qkvo8_hd128_<seed>` |
| Naive FP8 (with Stale Delta) | `dm_stale_hd128` | Delta-Matching-128, stale delta, full fp8 pipeline | `Q35M_dm_stale_ffn8_qkvo8_hd128_<seed>` |
| Naive FP8 (Consistent dO) | `dm_consistent_hd128` | the stale arm with the delta from the dequantized dO codes (`DM_DELTA_Q8=1`) | `Q35M_dm_consistent_ffn8_qkvo8_hd128_<seed>` |
| cuDNN / Transformer Engine FP8 | `te_fp8` | cuDNN fp8 fused attention (TE env) | `Q35M_te_fp8_ffn8_qkvo8_hd128_<seed>` |
| NoPE / Muon variants | `nope_*_hd128` / `muon_*_hd128` | the Standard, stale and matched arms with `--model.nope true` / `--optim.name muon` | `Q35M_..._<seed>_nope` / `Q35M_..._<seed>_muon` |
| head_dim 256, 0.569 B | `xs_clean_hd256` / `xs_dm_stale_hd256` / `xs_dm_matched_hd256` | FlashAttention-3 bf16 / Delta-Matching-256 stale / matched (`qwen35_xs.yaml`) | `Q35XS_..._<seed>` |
| head_dim 256, 1.665 B | `clean_hd256` / `dm_stale_hd256` / `dm_matched_hd256` | the same three at the yaml's 8q/2kv x 256 geometry (`qwen35_m.yaml`) | `Q35M_..._<seed>` |
| head_dim 256, 5.289 B | `l_clean_hd256` / `l_dm_stale_hd256` / `l_dm_matched_hd256` | the same three (`qwen35_l.yaml`) | `Q35L_..._<seed>` |
| GDN:MLA hybrid | `mla_clean_hd128` / `mla_dm_stale_hd128` / `mla_dm_matched_hd128` | Multi-head Latent Attention, bf16 core / Delta-Matching stale / matched (`qwen35_m_mla.yaml`) | `Q35MLA_..._<seed>` |
| Mamba2:GQA hybrid | `mamba2_clean_hd128` / `mamba2_dm_stale_hd128` / `mamba2_dm_matched_hd128` | the GDN cell's attention arms with Mamba2 recurrent layers (`qwen35_m_mamba2.yaml`) | `Q35M2_..._<seed>` |
| KDA:GQA hybrid | `kda_clean_hd128` / `kda_dm_stale_hd128` / `kda_dm_matched_hd128` | the GDN cell's attention arms with Kimi Delta Attention recurrent layers (`qwen35_m_kda.yaml`) | `Q35KDA_..._<seed>` |
| pure GQA (Qwen3 geometry) | `gqa_clean_hd128` / `gqa_dm_stale_hd128` / `gqa_dm_matched_hd128` | 28 attention layers, 16q/8kv head_dim 128 (`qwen3_m.yaml`) | `Q3M_..._<seed>` |

One arm by hand: always smoke first (3 steps, wandb off, the export and a 128-window val CE at the end, then the dirs
are removed):

    SMOKE=1 scripts/campaign/train_arm.sh dm_matched_hd128 11785
    # expected tail: "... done (exit 0)", "export patched -> clean-kernel eval", a "[memmap-ppl RESULT] nll/tok=..." line

(`SMOKE=1 scripts/experiments/<experiment>.sh` smokes every arm of an experiment.) The step-0 train CE of the 1.665 B
GDN:GQA cell at seed 11785 is 10.7575 on every arm (same init, same data order); a smoke that prints a different step-0
value has a different model or data. Then the full run, one arm per node:

    scripts/campaign/train_arm.sh dm_matched_hd128 11785 > workspace/logs/driver_md_11785.log 2>&1

Knobs: `STOP_AT_STEP=n` ends the run at step n with the 28,610-step WSD schedule unchanged (never use `NUM_STEPS`
to shorten a run: it compresses the schedule); `EXTRA_ARGS="--flag value"` appends trainer flags (append a `TAG` to
keep the run name distinct; a misspelled key stops the run at config load with a ValueError naming it);
`SAVE_TOTAL_LIMIT` (default 2: the two newest complete checkpoints are kept); `SEED` is the second argument (11785 and 2
are the paper's; `data.seed` stays pinned at 11785 in the yaml, so a seed changes the initialization only). The
trainer resumes automatically from `workspace/qwen35_m/ckpts/<run>` if the same run name is relaunched.

`train_arm.sh` exits 0 when the run is done (or was already complete), 75 when the run stopped early with a complete
checkpoint (relaunch to resume), 143 when a signal stopped it before any rank saved, non-zero before any launch when a
pre-flight check fails (2 for the data, the tokenizer, the kernel build or an unknown arm), and otherwise with the
trainer's or torchrun's own failure status. The 128-window val CE at the end of a run uses the first GPU the launcher
was given (`CUDA_VISIBLE_DEVICES`, else GPU 0) and, when the run logs to W&B, reports to the same W&B run.

The head_dim-256 Standard arms (`*clean_hd256`) need the FA3 wheel; the head_dim-256 fp8 arms run the
`delta_matching` kernel like the head_dim-128 arms; `te_fp8` needs the TE environment (`DM_TE_PY`, `DM_TE_TORCHRUN`).
Stage 2 (`scripts/campaign/stage2_arm.sh <arm> [seed]`) continues the Standard arm's stage-1 export of the same seed at
65,536 tokens; its header states the recipe (28,000 steps, 8e-4, YaRN factor 8).

Measured step times on one H100 node (8 GPUs, 25-step probes, mean of steps 10 to 15): GDN:GQA clean / stale /
matched 3.155 / 3.078 / 3.120 s per step (47 GB peak), GDN:MLA 3.212 / 3.200 / 3.230 (48 GB), Mamba2:GQA 2.918 /
2.845 / 2.875 (39 GB); a 28,610-step run of the 1.665 B cell is 23 to 26 hours.

## 6. Evaluate a finished run

    scripts/campaign/eval_arm.sh Q35M_dm_matched_ffn8_qkvo8_hd128_11785        # ~25 min on 8 x H100
    scripts/campaign/eval_arm_64k.sh Q35M_dm_matched_ffn8_qkvo8_hd128_11785_s2_64k   # a stage-2 run

(the experiment scripts call them). `eval_arm.sh` runs the protocol of `docs/benchmark.md` on the export (val CE on 2000
x 8192 windows with the per-window artifact, RULER 4k and 8k, commonsense-9, extraction, TriviaQA 5-shot, MQAR, MMLU)
into `workspace/qwen35_m/eval/<run>/`. Verify by artifact: six `<cfg>/<cfg>_results.complete` markers
(`ruler_llama2_8k`, `commonsense`, `recall_extraction`, `triviaqa`, `recall_synthetic`, `mmlu`), a `memmap-ppl RESULT`
line in `valce_n2000.txt` with its finite per-window artifact `<run>_n2000_perwindow.npy`, and zero `Traceback` lines in
the `<cfg>.txt` logs; `EVAL_COMPLETE` is written when all seven are there. Re-running the script fills whatever is
missing. Compare arms with the per-window artifact, never with the scorer's printed standard error:

    python scripts/campaign/paired_pair.py <run_A> <run_B>                  # A - B per window: mean, SE, t, share of windows worse (every val set both runs have)
    python scripts/campaign/paired_analysis.py <baseline> <run> [<run> ...]  # several runs against one baseline (--val 64k for the stage-2 set)
    python scripts/campaign/collect_results.py <run> [<run> ...]            # the report's columns, mean +/- one SE over the runs given
    python scripts/campaign/report_table.py --per-seed                     # every experiment cell and arm, mean +/- SE over seeds
    python scripts/campaign/extract_ce_series.py workspace/logs/<run>.log out.tsv    # the per-step train CE

`eval_arm.sh` and `eval_arm_64k.sh` run with `WANDB_MODE=disabled`. For a hand-run evaluation (`python -m evaluation.meta_eval`,
`evaluation.ppl.memmap_ppl`) without a W&B account, set `WANDB_MODE=disabled` too (on a terminal, wandb otherwise asks
for a login); `memmap_ppl` logs to W&B only with `--wandb`. A W&B failure never stops an evaluation: every W&B call is
guarded, and each evaluator writes its completion marker before its final W&B logging (`docs/eval_engine.md` section 3).

## 7. The test suite

    pytest tests -m "not gpu"     # anywhere: the environment and its pins, CPU logic, the arm table, the launcher, stage-2 and
                                  #   eval scripts in dry mode, the data scripts on synthetic data; + the data root when found
    pytest tests                  # on a GPU node: + kernel numerics, the mixers, 3-step trainings of the arms (head_dim 128 and
                                  #   256, through the launcher too), resume, engine parity, the eval chain, the HF conversion

The CPU subset needs the environment of sections 2 and 3, bash, tar and GNU cp, but no GPU, network or data: the data
and tokenizer tests skip with a reason when nothing is found (and fail when `DM_DATA_ROOT` / `DM_TOKENIZER_DIR` is set
but incomplete). `-m "not slow"` skips the subprocess trainings and evaluations, `-m "not data"` the tests that need
the data root.

Run it after every step of this document. `tests/README.md` maps each file to what a failure means and lists the
variables it reads. The GPU tests run on `DM_TEST_GPU` (default 0) and write under `workspace/tests_tmp/`.

## 8. Stopping and resuming

A run can stop at any time and continue later, on the same node or another one that sees the same `DM_WORKSPACE`:

- The trainer saves every `DM_SAVE_INTERVAL` steps (1500 by default; lower it where runs are stopped often) and resumes
  from the newest complete checkpoint when the same arm and seed are launched again. A SIGTERM (for example from a job
  scheduler) makes it save a checkpoint at the next step boundary and exit 75; so does its time budget
  `DM_STOP_BEFORE=<epoch seconds>`. With `DM_LAUNCH=env` the ranks are started by `train_arm.sh` itself, so a signal
  sent to the launcher reaches them directly.
- The training is complete when the export carries `TRAIN_COMPLETE`; `eval_arm.sh` then skips each benchmark that has
  its `.complete` marker (RULER per task), an interrupted one restarts from scratch, the val-CE step is complete only
  with both its RESULT line and the per-window artifact, and `EVAL_COMPLETE` is written when all seven are done.
- Checkpoint safety: the `.complete` marker of a checkpoint is written last and atomically, resume only considers marked
  checkpoints, rotation runs right after a new checkpoint's marker and keeps the newest `save_total_limit` complete
  ones (the new one included) and removes stale partial directories, and a relaunch of a finished run only re-exports.
  `pytest tests/test_resume.py tests/test_launcher.py` exercises all of it on one GPU (periodic checkpoint and exact
  resume, SIGTERM stop with exit 75 and resume, deadline stop, idempotent finish, the launcher's env launch and signal
  forwarding).
- Local logs for plots and reports, all under `workspace/logs/`: `<run>.metrics.jsonl` (one JSON per logged step with
  `train/ce_loss`, the learning rate and the rest, plus `start` / `resume` / `checkpoint` / `stop` / `export` / `complete`
  events; keep, per step, the record after the last `resume`), `<run>.log` and `<run>.rank<r>.log` (the trainer),
  `eval_<run>.log` and `workspace/qwen35_m/eval/<run>/<cfg>.txt` (the benchmarks), `workspace/result/<run>/train_summary.json`.
  wandb is optional (`DM_WANDB=0`); with it on, a relaunched run resumes the same wandb run (its id is kept in
  `logs/<run>.wandb_id`).

## 9. Pitfalls

- Launch from a shell without `CUDA_VISIBLE_DEVICES` set (a leaked `CUDA_VISIBLE_DEVICES=0` makes torchrun's 8 ranks
  fail with `invalid device ordinal`); the scripts never set it for training.
- Never edit a bash script while a run is executing it (bash reads scripts incrementally); copy, edit, relaunch.
- `--num_steps` compresses the WSD schedule; use `STOP_AT_STEP` for shortened legs.
- A config key the trainer does not know (a yaml key or a `--section.key` flag) fails at load with a ValueError that
  names it and suggests the closest field.
- The fp8 arms fail at model build when the `delta_matching` package is missing or its library was built without the
  head_dim-128 instantiations; `arms.sh` checks the build flags before launching. The eval scorers' wandb directory
  follows `WANDB_DIR` or `DM_WORKSPACE`; the training-data and RULER paths inside the yaml configs are overridden by
  the launcher and the eval chain, never used directly.
- The exports of the fp8 arms are patched at the end of training (`attn_impl: sdpa`, `dm -> gqa`, `mla_dm -> mla`, the
  fp8-only `dm_mode` / `dm_fused_producers` keys dropped; `config.train.json` keeps the trained config), so every arm is
  scored through the same bf16 path (FlashAttention-2, the loaders' `EVAL_ATTN_IMPL` default); `eval_arm.sh` and
  `eval_arm_64k.sh` refuse an unpatched export.
- MMLU and MQAR run with `HF_HUB_OFFLINE=1`: they need the `cais___mmlu` cache in `HF_HOME` (`data/prepare_eval.py`
  fills `$DM_DATA_ROOT/hf_cache`).
- The fp8 GEMMs need every dimension divisible by 16; the Delta-Matching kernel needs head_dim 128 (sequence length a
  multiple of 256) or 256 (a multiple of 128) and self-attention (q, k and v of one length).
- The trainer logs `train/ce_loss` every 10 steps; it includes the z-loss term (`z_loss_weight * logsumexp^2`). A run
  maps to one wandb run through `logs/<run>.wandb_id`; if that file is lost, a relaunch starts a second wandb run under
  the same name (order by `created_at`, later segment wins). `scripts/campaign/extract_ce_series.py` (from the trainer
  log) and `pull_wandb_ce.py` (from wandb) handle both.
- Checkpoints on node-local disk die with the node; copy them to durable storage and keep `SAVE_TOTAL_LIMIT` at 2
  unless a later run needs an older checkpoint.
