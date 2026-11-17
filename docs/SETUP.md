# Setting up and running the Delta-Matching experiments (Hopper, CUDA 13.0, uv)

The target is **NVIDIA H100 / H200 (sm_90a), the CUDA 13.0 toolkit, Python 3.12, PyTorch 2.12.0+cu130.** The fp8
attention kernel (the `delta_matching` package) is an sm_90a build and does not run on other GPUs (no Blackwell path
exists).

Read in order: section 1 (what you need), 2 (the Python environment with uv), 3 (the kernel package), 4 (data),
5 (configuring, smoke-testing and launching a run), 6 (evaluation), 7 (the test suite), 8 (stopping and resuming),
9 (pitfalls).

## 1. Requirements and layout

- **Hardware.** 8 x H100 80 GB (or H200) per node for the experiment runs; a 28,610-step run of the 1.665 B hybrid
  (`qwen35_m.yaml`) takes 23 to 26 hours on one H100 node (step times in section 5). Under Slurm a run can also train on
  part of a node or on several nodes (`scripts/slurm/README.md`): the global batch and the data order stay, the
  gradient accumulation follows.
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
        logs/                     trainer, launcher and eval logs; slurm/ (the Slurm jobs' logs and chains/<name>/, each
                                  chain's state), nodes/ (the per-node reports of multi-node launches)
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

    grep -v -E '^(torch|torchvision|triton|flash_attn|flash_attn_3|causal_conv1d|mamba_ssm)==|^nvidia-(cublas|cuda-(cupti|nvrtc|runtime)|cudnn|cufft|cufile|curand|cusolver|cusparse|nccl|nvjitlink|nvshmem|nvtx)' \
        env/requirements-cu130-hopper.lock.txt > .venv/requirements-rest.txt
    uv pip install --no-deps -r .venv/requirements-rest.txt   # the lock is a full freeze: no resolution, torch untouched

The pattern is anchored on `name==`, so `torchao`, `torch_c_dlpack_ext` and `tilelang` stay in. The excluded `nvidia-*`
lines are the CUDA libraries that come with the torch wheel (cuDNN 9.20, NCCL 2.29, the CUDA 13.0 runtime); the lock's
other `nvidia-*` packages (`nvidia-cuda-nvdisasm` and the `nvidia-cutlass-dsl*` that `quack-kernels` needs) are kept.
Any environment with the same packages works (conda included); nothing in the repository depends on the environment manager.
(`scripts/build_kernels.sh` is a `uv` recipe for a development environment at a newer torch; it is not the
environment of the study. It builds in a fresh private temporary directory unless `BUILD_DIR` names a directory of
your own.)

The repository is used in place, not installed: `scripts/campaign/env.sh` sets `PYTHONPATH=<repo>/src`, and the
modules are imported as `pretrain.*`, `evaluation.*`, `inference.*`, `kernel.*`.

Transformer Engine (the `te_fp8` arm: cuDNN's fp8 fused attention with TE's DelayedScaling recipe) needs a separate
environment (TE 2.20 built against the same torch, cudnn-frontend 1.28, cmake 4.4, pybind11); it is not required for
the other arms. `scripts/build_kernels.sh` step 7 has the build flags (a failed TE build
there only warns); the launcher runs that arm through `DM_TE_PY` / `DM_TE_TORCHRUN`. A light way to get it that
reuses the main environment: a venv layered on it (`python -m venv --system-site-packages`) holding only
`transformer_engine[pytorch,core-cu13]==2.20.2` (installed with `--no-build-isolation`; it needs `pybind11` and
`nvidia-cudnn-frontend>=1.29`, and the main environment's pip `nvidia/nccl/include` and `nvidia/cudnn/include` on
`CPATH` to compile), then the packages pip pulled in that shadow the lock's pins (`apache-tvm-ffi`,
`nvidia-cutlass-dsl*`) uninstalled from the venv. At run time TE has to find the main environment's pip cuDNN:
`CUDNN_PATH=<site-packages>/nvidia/cudnn` (TE otherwise looks for an unversioned `libcudnn.so`) and its `lib`
directory, with `nvidia/nccl/lib`, on `LD_LIBRARY_PATH` (the core library links the cuDNN sub-libraries); point
`DM_TE_PY` at a wrapper script that sets both and execs the venv's python, and `DM_TE_TORCHRUN` at a second executable
script that runs `exec <wrapper> -m torch.distributed.run "$@"` (the launcher runs each as one executable, so neither can
be a command line with arguments).

## 3. The kernel package

The fp8 attention kernel of the Delta-Matching arms is the **`delta-matching`** package (C++/CUDA, CUTLASS/CuTe, a
flash-attention-style drop-in; import name `delta_matching`). Its source is bundled with this repository at
`kernels/delta-matching/`: the **KDA-evolved kernel** (the original release `jet-ai-projects/delta-matching` 3d4cd5b plus
the eight commits of `kda-sm90-clean`, `github.com/haok1402/delta-matching` 5a512e6), the same API and contract with
faster head_dim-128 passes (fwd+bwd 1.18x at the training shapes, a 0.8 % faster step of the 1.665 B hybrid) and named
numerics changes at head_dim 128. `kernels/README.md` lists every change, its speed and numerics evidence, and
`kernels/PROVENANCE.json` the exact commits. Build and install it into the environment of section 2:

    export CUDA_HOME=/usr/local/cuda-13.0     # any CUDA 13 toolkit
    scripts/install_kernel.sh                 # MAX_JOBS=16 by default; a few minutes, no GPU needed

The script checks for a cu130 torch and nvcc 13, fetches the flash-attention headers and CUTLASS at the commits the
package pins into `$DM_WORKSPACE/build/flash-attention` the first time (or uses `DM_FA_SRC`), runs `pip install
--no-build-isolation --no-deps` on the bundle, and checks that the installed package reports the bundle's source hash
(`kernel_version()["code_sha"]`, 9fe8da5f743a). The kernel's README (`kernels/delta-matching/README.md`) has the API, its
input contract (it raises `ValueError`, `TypeError` or `NotImplementedError` naming the broken constraint) and its
runtime knobs.

The build is the certified flag set of the bundled kernel (`kernels/delta-matching/delta_matching/_buildcfg.py`; one
binary serves head_dim 128 and 256). It serves every fp8 arm of the experiments (stale, matched and the consistent-dO
arm, at both head dims). The trainer imports the kernel from the installed package; no binary paths are pinned anywhere
in this repository (`DM_SO` can point the package at another build of its library). The paper's runs used the original
kernel (`code_sha` 5905517248c5); to reproduce them bit for bit, install that checkout instead with
`DM_KERNEL_SRC=/path/to/delta-matching scripts/install_kernel.sh` (and `DM_EXPECT_BUNDLED_KERNEL=0` for the test suite).

**nvcc builds are not byte-reproducible**, so a rebuilt library is verified by numerics: `pytest tests/test_env.py
tests/test_kernels.py` loads it, checks that it is a head_dim-128 build with gmem twins (`build_hd128() == 1`,
`build_gt() == 1`), runs both delta modes against fp32 attention within the fp8 envelope, checks determinism and the
structural identities between the modes, and builds the `dm` mixer from it; `pytest tests/test_kernel_bundle.py` checks
that the installed package is the bundled build. `python scripts/verify_env.py` (GPU) prints the build flags of the
loaded library, and the kernel's own suite checks the kernel alone (`cd /tmp && pytest <repo>/kernels/delta-matching/tests`,
run from outside the source tree so that the installed package is the one imported).

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
paper). Each trains, then evaluates, every arm and seed of one experiment in turn, one run at a time on all the GPUs it
is given (one node when run by hand; one node, part of one or several nodes in a Slurm job), and continues where it
stopped when run again:

| script | arms | seeds |
|---|---|---|
| `stage1_hd128.sh` | the head_dim-128 cell: `clean_hd128`, `dm_stale_hd128`, `dm_matched_hd128`, `dm_consistent_hd128`, `te_fp8` | 11785, 2 |
| `stage1_hd256_scale.sh` | head_dim 256 at 0.569 / 1.665 / 5.289 B: `{xs_,,l_}clean_hd256`, `{xs_,,l_}dm_stale_hd256`, `{xs_,,l_}dm_matched_hd256` | 11785, 2 |
| `stage1_l_kda.sh` | the 5.293 B KDA : GQA twin of the 5.289 B cell (head_dim 256): `l_kda_clean_hd256`, `l_kda_dm_stale_hd256`, `l_kda_dm_matched_hd256` | 11785 |
| `stage1_architectures.sh` | `mla_*`, `kda_*`, `mamba2_*`, `gqa_*` (clean / dm_stale / dm_matched, head_dim 128) | 11785, 2 (pure GQA: 11785, 1) |
| `stage1_nope.sh` | `nope_clean_hd128`, `nope_dm_stale_hd128`, `nope_dm_matched_hd128` | 11785, 2 |
| `stage1_muon.sh` | `muon_clean_hd128`, `muon_dm_stale_hd128`, `muon_dm_matched_hd128` | 11785 |
| `stage2_context.sh` | the 8K to 64K extension of the head_dim-128 cell: `clean_hd128`, `dm_matched_hd128`, `dm_stale_hd128`, `te_fp8` | 11785, 2 |

`DRY=1` prints an experiment's plan (arms, seeds, run names, commands); arms given on the command line restrict it;
`SEEDS="..."` overrides the seeds. The runs are independent (except stage 2, which warm-starts from `clean_hd128`'s
stage-1 export of the same seed): to train several at once, give each node, or each Slurm chain, its own arms or seeds,
e.g. `scripts/experiments/stage1_hd128.sh dm_matched_hd128` in one and `... dm_stale_hd128` in another. A Slurm job of
several nodes trains each of its runs on all of them.

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
| head_dim 256, 5.293 B KDA:GQA | `l_kda_clean_hd256` / `l_kda_dm_stale_hd256` / `l_kda_dm_matched_hd256` | the same three with Kimi Delta Attention recurrent layers (`qwen35_l_kda.yaml`) | `Q35LKDA_..._<seed>` |
| GDN:MLA hybrid | `mla_clean_hd128` / `mla_dm_stale_hd128` / `mla_dm_matched_hd128` | Multi-head Latent Attention, bf16 core / Delta-Matching stale / matched (`qwen35_m_mla.yaml`) | `Q35MLA_..._<seed>` |
| Mamba2:GQA hybrid | `mamba2_clean_hd128` / `mamba2_dm_stale_hd128` / `mamba2_dm_matched_hd128` | the GDN cell's attention arms with Mamba2 recurrent layers (`qwen35_m_mamba2.yaml`) | `Q35M2_..._<seed>` |
| KDA:GQA hybrid | `kda_clean_hd128` / `kda_dm_stale_hd128` / `kda_dm_matched_hd128` | the GDN cell's attention arms with Kimi Delta Attention recurrent layers (`qwen35_m_kda.yaml`) | `Q35KDA_..._<seed>` |
| pure GQA (Qwen3 geometry) | `gqa_clean_hd128` / `gqa_dm_stale_hd128` / `gqa_dm_matched_hd128` | 28 attention layers, 16q/8kv head_dim 128 (`qwen3_m.yaml`) | `Q3M_..._<seed>` |

On a Slurm cluster, `scripts/slurm/submit.sh [sbatch options] -- <command>` runs any of these commands as a chain of
time-limited jobs (each on one node, part of one, or several nodes) that stop with a checkpoint before their time limit
and hand over to the next, until the command is done (`scripts/slurm/README.md`).

One arm by hand: always smoke first (3 steps, wandb off, the export and a 128-window val CE at the end, then the dirs
are removed):

    SMOKE=1 scripts/campaign/train_arm.sh dm_matched_hd128 11785
    # expected tail: "... done (exit 0)", "export patched -> clean-kernel eval", a "[memmap-ppl RESULT] nll/tok=..." line

(`SMOKE=1 scripts/experiments/<experiment>.sh` smokes every arm of an experiment.) The step-0 train CE of the 1.665 B
GDN:GQA cell at seed 11785 is 10.7575 on every arm (same init, same data order; the bf16-attention clean arm prints
10.75741, the fp8 arms 10.75747-10.75748); a smoke whose step-0 value differs by more than 1e-4 has a different model or
data. Then the full run, one arm per node:

    scripts/campaign/train_arm.sh dm_matched_hd128 11785 > workspace/logs/driver_md_11785.log 2>&1

Knobs: `STOP_AT_STEP=n` ends the run at step n with the 28,610-step WSD schedule unchanged (never use `NUM_STEPS`
to shorten a run: it compresses the schedule, and a resume with another `NUM_STEPS` is refused, section 8);
`EXTRA_ARGS="--flag value"` appends trainer flags (append a `TAG` to keep the run name distinct; a misspelled key stops
the run at config load with a ValueError naming it); `DM_LR` (2.4e-3) the peak learning rate; `DM_SAVE_INTERVAL`
(default: the arm's `SAVE_IV`, about an hour of steps, section 8); `SAVE_TOTAL_LIMIT` (default 2: the two newest
complete checkpoints are kept); `SEED` is the second argument (11785 and 2 are the paper's; `data.seed` stays pinned at
11785 in the yaml, so a seed changes the initialization only). The trainer resumes automatically from
`workspace/qwen35_m/ckpts/<run>` if the same run name is relaunched, provided the run's identity is unchanged
(section 8). The full list of the launcher's knobs is in the header of `scripts/campaign/train_arm.sh`.

`train_arm.sh` exits 0 when the run is done (or was already complete: then before any other check), 75 when the run
stopped early with a complete checkpoint (relaunch to resume), 2 for a configuration error, never worth a retry (the
data, the tokenizer, the kernel build, an import, an unknown arm or a bad knob before any rank starts, or a resume the
trainer refused: `FATAL: RESUME REFUSED` in the log, section 8), 3 when another live job is training the run (its lock,
section 8), 1 when the trainer exited 0 but its export is not whole (relaunch: it resumes the finished run and exports
again) or the run's lock was not released in time, 143 when ranks had to be killed (after a stop signal, or stuck after
the others had exited) and none saved, and otherwise with the first failing rank's status (several nodes: the rule in
the header). The 128-window val CE at the end of a run uses the first GPU the launcher was given
(`CUDA_VISIBLE_DEVICES`, else GPU 0) and, when the run logs to W&B, reports to the same W&B run.

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

    pytest tests -m "not gpu"     # anywhere: the environment and its pins, CPU logic, the arm table, the launchers, the
                                  #   experiment driver and the Slurm chains with stand-ins (no GPU, no Slurm), the resume
                                  #   pipeline on CPU, stage-2 and eval scripts in dry mode, the data scripts on synthetic
                                  #   data; + the data root when found
    pytest tests                  # on a GPU node: + kernel numerics, the mixers, 3-step trainings of the arms (head_dim 128 and
                                  #   256, through the launcher too), resume, engine parity, the eval chain, the HF conversion

The CPU subset needs the environment of sections 2 and 3, bash, tar, GNU cp and util-linux's `flock` and `setsid`
(the run locks, the multi-node launch), but no GPU, network, data or Slurm: the data and tokenizer tests skip with a
reason when nothing is found (and fail when `DM_DATA_ROOT` / `DM_TOKENIZER_DIR` is set but incomplete).
`-m "not slow"` skips the subprocess trainings and evaluations, `-m "not data"` the tests that need the data root.

Run it after every step of this document. `tests/README.md` maps each file to what a failure means and lists the
variables it reads. The GPU tests run on `DM_TEST_GPU` (default 0) and write under `workspace/tests_tmp/`.

## 8. Stopping and resuming

A run can stop at any time and continue later, on the same node or others that see the same `DM_WORKSPACE`, on the same
number of GPUs or another one. Relaunching the same command resumes it; on a Slurm cluster a chain of jobs does that
for you (`scripts/slurm/README.md`).

- **Checkpoints.** The trainer saves `checkpoint-<step>/` under `workspace/qwen35_m/ckpts/<run>/` every
  `DM_SAVE_INTERVAL` steps (default: the arm's `SAVE_IV` in `scripts/campaign/arms.sh`, about an hour of training: 1000
  for the 1.5 to 1.7 B arms, 400 for the 5.3 B `l_*` arms, 2500 for the 0.569 B `xs_*` arms; 750 for stage 2), when it
  stops early, and at the end. A checkpoint holds the resume state (FSDP2: the DCP shards; DDP:
  `training_state_<rank>.pt`), `pytorch_model.bin`, `model_config.json`, `kernel_version.json`, the sidecar
  `trainer_state.json` (the skipped-update count, the world size, the last logged metrics, the training time) and each
  rank's `next_train_data_<rank>.pt`. Every one of them is fsynced, then the `.complete` marker is written last and
  atomically; the resume considers marked checkpoints only. Right after a marker the rotation keeps the newest
  `SAVE_TOTAL_LIMIT` (2) complete checkpoints, the new one included, and removes older partial directories; a victim is
  renamed to `.trash-<name>.<ns>` before it is deleted (a half-deleted directory never looks like a checkpoint;
  leftovers are removed at the next start), and a directory whose marker cannot be read is kept. In a Slurm chain each
  marker is followed by a progress line (`<epoch> <job id> ckpt <run> <step>` in the chain's `progress` file).
- **Stops.** A SIGTERM or SIGUSR1 on any rank, or the time budget `DM_STOP_BEFORE=<epoch seconds>`, stops every rank at
  the same step boundary (the stop flag is MAX-all-reduced every step): the trainer saves a checkpoint and exits 75. A
  stop that is pending before the first step (a signal during the startup, a budget already spent) exits 75 at once,
  without training or saving: the checkpoint the run resumed from holds its state. With `DM_LAUNCH=env` (which
  `scripts/slurm/job.sh` sets) the ranks are started by `train_arm.sh` itself, so a SIGTERM or SIGINT sent to the
  launcher reaches them directly, as SIGTERM (on several nodes the node launchers start them, and HUP, USR1 and USR2
  are stops as well: `scripts/slurm/README.md`); under torchrun the launcher passes it to torchrun's children, never to
  torchrun once it has started them (it would kill its ranks 30 s later). Under the env launch the launcher's collector
  (`scripts/campaign/ranks.sh`) ends ranks that cannot end themselves: after a rank failed (an exit other than 0 and
  75) before any stop signal, the others get `DM_RANK_GRACE` seconds (60) before they are SIGKILLed (they are most
  likely blocked in a collective; a rank that was already gone when a stop came failed before it, unless a stop signal
  killed it); under Slurm, ranks get `DM_TERM_GRACE` seconds (90, below Slurm's KillWait of 120 s) after a forwarded
  SIGTERM; after the first rank exited cleanly, the others get `DM_EXIT_GRACE` seconds (600). A save cut short leaves
  no marker, and the previous checkpoint holds.
- **Resume.** The newest complete checkpoint is chosen by the master and broadcast, so every rank loads the same one. A
  newest marker that exists but cannot be read (an I/O error after 4 tries, 1 s apart) fails the resume instead of
  falling back to an older checkpoint; once the file system reads it again a relaunch resumes from it. If it stays
  unreadable, set the checkpoint aside, `mv <ckpt dir>/checkpoint-N <ckpt dir>/.bad-checkpoint-N` (a name nothing reads
  or removes), and the next launch resumes from the newest older complete checkpoint. The data order is a function of
  the global step, not of the world size, and DCP reshards the state, so a run may resume on another number of GPUs or
  nodes; the per-rank next-batch check is then skipped. (A checkpoint written before the sidecar existed records no
  world size: resumed on another one, its check fails; `EXTRA_ARGS="--force_resume_check false"` makes the check a
  warning.)
- **Run identity.** A resume continues a run only when it is the same experiment. The master records the run's identity
  in `<ckpt dir>/run_identity.json` at its first start (or at the first resume of a run that has none) and checks it at
  every resume. It holds what defines the trajectory: `num_steps` and every optimizer field (the schedule and its
  horizon), the global batch and the sequence length, the data (the paths, compared as the file system resolves them;
  dtype, seed, chunk subset, ...), every model field, the precision and the kernels that set the round-off (bf16, fp8 and
  its recipe and scope, fused CE, compile), the backend, the seed, and the arm's numerics environment (`DM_FP8*`,
  `DM_SHARED_X8`, `DM_FLASH_IMPL`, `DM_SDPA_BACKEND`, `DM_DELTA_Q8`, ...; unset reads as the code default). Not in it,
  so free to differ between the launches of one run: `STOP_AT_STEP`, the per-device batch and the world size, the save,
  export, logging and W&B settings, `init_from_hf`, the activation-checkpointing knobs and the data loader's workers. A
  difference refuses the resume: every rank prints `FATAL: RESUME REFUSED ...`, the log lists each differing value
  (recorded -> now), and the trainer exits 2, a configuration error no retry can change (a Slurm chain of the run
  stops, an experiment script gives the run up; under torchrun, which reports 1, `train_arm.sh` maps that line back to
  2). `DM_ALLOW_RUN_CHANGE=1` continues with the new
  values: a warning, and the change is recorded in the file, so later launches need no knob. The software (the torch,
  triton, fla, torchao and Transformer Engine versions, and `DM_GIT_REV`) is recorded as well; a difference there is a
  warning only. The record of a resume is written only once the checkpoint has loaded, so a load that fails changes
  nothing.
- **Kernel identity.** Every checkpoint records the `delta_matching` build the run trains with (`kernel_version.json`:
  `code_sha` (its csrc/), `bf16_dp`, `py_sha` (its Python files: the fp8 producers, the backward split), `dm_so` (the
  resolved `DM_SO` library, which the package's `code_sha` stamp cannot describe) and the library path; `null` for the
  bf16-attention arms). A resume on another build (e.g. the original release instead of the bundled KDA-evolved kernel)
  is refused, since their head_dim-128 numerics differ, and so is a kernel run's resume of a checkpoint whose build
  cannot be read (no `kernel_version.json`: written before builds were recorded; or one that does not parse): the same
  `RESUME REFUSED` line and exit 2 (a record that cannot be read for an I/O error: exit 1, worth a retry).
  `DM_ALLOW_KERNEL_CHANGE=1` resumes anyway with a warning (the bf16 arms are never checked). The `start` event of
  `<run>.metrics.jsonl` names the kernel's `code_sha` too. Both allow knobs are meant for one launch: set while nothing
  differs, each is a warning, since it would let a later accidental change through. In a Slurm chain, which passes its
  environment to every later job, `train_arm.sh` drops each once the submission that set it has recorded the change
  (`run_identity.json` rewritten, or a checkpoint saved on the new build).
- **One trainer per run.** `train_arm.sh` takes an flock on `workspace/qwen35_m/ckpts/<run>.lock` before any rank starts
  (`<run>.lock.owner` names the holder). Another live job holding it: exit 3. A holder that is ending (Slurm starts a
  chain's next job while the previous one's ranks may still save), has ended or is this job is waited for,
  `DM_LOCK_WAIT` seconds (600); still held then: exit 1, a failure to retry. `eval_arm.sh` and `eval_arm_64k.sh` lock the
  run's eval directory the same way.
- **The end of a run.** After the final checkpoint and the export, `train_arm.sh` checks that the export is whole
  (`python -m pretrain.hf.check_export <export dir>`: `config.json`, which the exporter writes last, parses, and
  `model.safetensors` is as long as its header says), patches `config.json` for the clean-kernel evaluation (section 9;
  `config.train.json` keeps the trained one), writes `TRAIN_COMPLETE` (`<date> steps=<trained steps> num_steps=<N>`),
  and only then runs the informational 128-window val CE. A whole export written by this launch completes the run even
  when the trainer fails afterwards, in its teardown. A relaunch of a run with `TRAIN_COMPLETE` exits 0 at once.
  `STOP_AT_STEP=n` ends a run the same way, at step n (`steps=<n>`); training it on to the full horizon later takes
  removing that marker first (and setting its evaluation directory aside if the step-n export was evaluated: its
  `EVAL_COMPLETE` would be taken for the new export's).
- **The trainer's exit status** (`python -m pretrain.train`): 0 complete (the final checkpoint and the export written),
  75 stopped early with a complete checkpoint (whatever the process-group teardown or W&B do), 2 a refused resume, any
  other status a failure. W&B is initialized only after the resume has been decided, checked and loaded (a refused
  resume never touches the run's W&B id, kept in `logs/<run>.wandb_id`) and closed after the process group, by the
  master alone, within `DM_WANDB_FINISH_TIMEOUT` seconds (60; marked preempting after a stop; what is not uploaded then
  stays in `WANDB_DIR` for `wandb sync`). The process-group teardown after a stop or a refusal is bounded by
  `DM_DIST_CLOSE_TIMEOUT` seconds (30).
- **The code revision.** `DM_GIT_REV` (unless set: `git describe --always --dirty`, and for a tree with changes under
  `src/` or `scripts/`, uncommitted or untracked, `-dirty+<8 hex>` of those changes) is in the launch line, the
  trainer's `start` event and the run identity's software section.
- **Evaluations.** The training is complete when the export carries `TRAIN_COMPLETE`; `eval_arm.sh` then skips each
  benchmark that has its `.complete` marker (RULER per task), an interrupted one restarts from scratch, the val-CE step is
  complete only with both its RESULT line and the per-window artifact, and `EVAL_COMPLETE` is written when all seven are
  done.
- **Tests.** `pytest tests/test_resume.py tests/test_launcher.py` exercises the trainer and the launcher on one GPU
  (periodic checkpoints and exact resume, the run identity, a SIGTERM stop with exit 75 and the resume, a spent time
  budget, the idempotent finish, the env launch and its signal forwarding); on CPU, `tests/test_resume_cpu.py`,
  `test_resume_exactness_cpu.py`, `test_ckpt_units.py`, `test_resume_units.py`, `test_driver_launch.py` and
  `test_launcher_signals.py` cover the same paths with injected faults (`tests/README.md`).
- **Local logs** for plots and reports, all under `workspace/logs/`: `<run>.metrics.jsonl` (one JSON per logged step with
  `train/ce_loss`, the learning rate and the rest, plus `start` / `resume` / `checkpoint` / `stop` / `export` / `complete`
  events; keep, per step, the record after the last `resume`), `<run>.log` and `<run>.rank<r>.log` (the trainer),
  `eval_<run>.log` and `workspace/qwen35_m/eval/<run>/<cfg>.txt` (the benchmarks), `workspace/result/<run>/train_summary.json`.
  wandb is optional (`DM_WANDB=0`); with it on, a relaunched run resumes the same wandb run (its id is kept in
  `logs/<run>.wandb_id`).

## 9. Pitfalls

- Launch from a shell without `CUDA_VISIBLE_DEVICES` set (a leaked `CUDA_VISIBLE_DEVICES=0` makes torchrun's 8 ranks
  fail with `invalid device ordinal`); the scripts never set it for training.
- Never edit a bash script while a run is executing it (bash reads scripts incrementally); copy, edit, relaunch. Every
  link of a Slurm chain runs the launchers and experiment scripts from the repository (only `scripts/slurm/job.sh` is
  copied by Slurm): change them between links, or write the new file beside the old one and rename it into place
  (`scripts/slurm/README.md`, "Deploying this version").
- `--num_steps` compresses the WSD schedule (and a resume with another `num_steps` is refused, section 8); use
  `STOP_AT_STEP` for shortened legs.
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
  unless a later run needs an older checkpoint. A run on several nodes, and a chain whose next job may land on another
  node, needs `DM_WORKSPACE` on a file system all of them share.
