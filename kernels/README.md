# kernels/: the bundled fp8 attention kernel

`kernels/delta-matching/` is the `delta-matching` kernel package (import name `delta_matching`) that the fp8 arms of
this repository run on (the `dm` and `mla_dm` mixers, `src/kernel/dm_port/`). It is vendored here, so a checkout carries
the exact kernel source it was tested with, and `scripts/install_kernel.sh` builds it into the environment.

The bundled source is the **KDA-evolved kernel**: the original release (`jet-ai-projects/delta-matching` at 51ea382,
release 3d4cd5b) plus the eight commits of branch `kda-sm90-clean` of `github.com/haok1402/delta-matching`
(commit 5a512e6), found and validated by the kernel design agent (KDA). `PROVENANCE.json` holds the exact commits, the
source hash a build of this tree reports (`code_sha` 9fe8da5f743a), and the flag changes. The original release hashes to
5905517248c5. Excluded from the copy: `asset/` (blog media), `.github/` and `third_party/` (flash-attention, which
`scripts/install_kernel.sh` fetches at the pinned commit).

## What changed against the original kernel

The API, the input contract and the runtime knobs are unchanged (`kernels/delta-matching/README.md`); the training code
needed no change. The changes are in head-dim-128 scheduling and numerics, plus one bug fix at both head dims:

| commit | change | class |
|---|---|---|
| 43c2585 | pass 2 (dQ, D128): TMA pipeline depth 3 and psi loaded one tile early | bitwise |
| c50a6cb | pass 1 (dK / dV): dV overlapped with the psi barrier, deferred dS' stores, 8-shuffle dS' transpose; D256 dO depth 2 | bitwise |
| 0cd60ee | producers: Q / K / V quantized through their own strides, no contiguous copies (first use of each shape self-checked bitwise against the contiguous route, `DM_PRODUCER_SELFCHECK`) | bitwise |
| 644c787 | pass 0 (matched delta, D128): longest-first dispatch, four warpgroups per CTA (256-row CTAs when seqlen % 256 == 0, else 128-row), mask-free leading loop, one all-heads call in the split route | bitwise |
| 229bd69 | forward (D128): 192 x 128 tiles with three consumer warpgroups, lazy O rescale, S descale folded into the exp argument, scale staging | named numerics change |
| 87ec68d | passes 0 + 1 (D128): the 2^8 exp offset folded into the row bias (EXPBX64), clamp only on the masked tiles | named numerics change |
| 12d3d19 | pass 0 (D128): row sums reduced once per lane in fp64, dP descale per half partial; pass 1: 2048-row dV promote window | named numerics change |
| 5a512e6 | fix of the original release: an all-zero Q block times an all-zero K block no longer gives -inf * 0 = NaN on masked scores (NaN O / LSE at D256, wrong dV at both head dims) | fix |

Speed, measured by the KDA work on H100 (`kda-sm90-clean`, median; fwd+bwd = `dm_attn_func` with its producers):

| shape (B x N, H / Hkv) | forward | pass 0 | pass 1 | pass 2 | fwd+bwd |
|---|---|---|---|---|---|
| 2 x 8k, 16 / 4 | 1.14x | 1.37x | 1.22x | 1.14x | 1.18x |
| 2 x 8k, 32 / 4 | 1.24x | 1.35x | 1.15x | 1.12x | 1.19x |
| 1 x 64k, 16 / 4 | 1.18x | 1.33x | 1.17x | 1.10x | 1.18x |
| 1 x 64k, 32 / 4 | 1.17x | 1.33x | 1.18x | 1.11x | 1.18x |

Re-measured with this repository's environment (H100 80GB, driver 535 + the CUDA 13 forward-compat libcuda, torch
2.12.0+cu130, nvcc 13.0.88; both kernels built the same way; `benchmarks/bench.py`, median of 30 iterations, two
alternating repetitions each, ms of fwd+bwd through `dm_attn_func`):

| shape (B x N, H / Hkv, D) | matched: original -> bundled | speedup | stale: original -> bundled | speedup |
|---|---|---|---|---|
| 2 x 8k, 16 / 4, 128 | 4.547 -> 3.808 | 1.19x | 3.924 -> 3.345 | 1.17x |
| 2 x 8k, 32 / 4, 128 | 8.652 -> 7.335 | 1.18x | 7.435 -> 6.369 | 1.17x |
| 1 x 64k, 16 / 4, 128 | 122.68 -> 104.31 | 1.18x | 102.73 -> 89.16 | 1.15x |
| 2 x 8k, 8 / 2, 256 | 4.468 -> 4.246 | 1.05x | 3.953 -> 3.802 | 1.04x |

The head_dim-128 forward alone is 1.30x faster at 8k (1.095 -> 0.844 ms at 16 / 4).

Inside training (the 1.665 B GDN:GQA hybrid of `qwen35_m.yaml` at head_dim 128, 8 x H100, FSDP2, 2 x 8192 per GPU,
global batch 128 x 8192; fp8 FFN in both arms): the attention kernels take 192 ms of a ~3.26 s step instead of 231 ms,
and the whole optimizer step is 0.80 % / 0.83 % faster in two runs on two nodes. Attention is ~6-7 % of this hybrid's
step, which caps the end-to-end gain; models where attention is a larger share of the step (pure GQA, long context)
should gain more, but that was not measured.

Numerics: at head_dim 256 the outputs are bitwise those of the original kernel except on all-zero inputs (the fix). At
head_dim 128 the forward, pass 0 and pass 1 carry named numerics changes accepted under the KDA contract: every matched
row keeps the zero-sum identity rowsum(dS) = 0 within fp32 round-off of the original's residual, the relative error
against fp32 is at most 1.01x the original's per case in matched mode (1.10x in stale mode), the quantized codes are
unchanged, outputs stay deterministic, and the training loss matched the original kernel's within run-to-run noise.
**Runs on this kernel are not bitwise comparable with runs on the original kernel** (the paper's runs used the original):
compare arms within one kernel build, and read `kernel_version()` in the run log (`[dm] ... code_sha ...`).

## Install, verify, switch

    scripts/install_kernel.sh                 # build kernels/delta-matching and install it into the active environment

It needs torch 2.12+cu130 and nvcc 13 (`CUDA_HOME`) in the environment (docs/SETUP.md sections 2 and 3), fetches the
flash-attention / CUTLASS headers the package pins into `$DM_WORKSPACE/build/flash-attention` the first time (or uses
`DM_FA_SRC`; headers at other commits are refused unless `DM_FA_ALLOW_UNPINNED=1`), builds a fresh copy of the tree (no
object of an earlier build is reused, nothing is written into the tree), installs with `pip install
--no-build-isolation --no-deps`, and ends by checking that the installed package reports the source tree's `code_sha`,
loads its own compiled library and has the tree's Python files (`py_sha` in `PROVENANCE.json`: `code_sha` covers csrc/
only). Then:

    pytest tests/test_kernel_bundle.py        # the bundle is intact and the installed kernel is the bundled build
    pytest tests/test_env.py tests/test_kernels.py   # (GPU) the build flags, numerics vs fp32, determinism, the dm mixer
    python scripts/verify_env.py              # (GPU) every kernel library of the environment
    cd /tmp && pytest <repo>/kernels/delta-matching/tests   # (GPU) the kernel's own suite against the installed package

To run the original kernel instead (e.g. to reproduce the paper's numbers bit for bit), install its checkout:
`DM_KERNEL_SRC=/path/to/delta-matching scripts/install_kernel.sh`. `tests/test_kernel_bundle.py` then reports the
mismatch unless `DM_EXPECT_BUNDLED_KERNEL=0` is set. `DM_SO` swaps only the compiled library at run time; the Python side
(`producers.py`, `split.py`) differs between the two kernels too, so switch with a full install.

## Updating the bundle

    git -C <delta-matching checkout> archive <commit> benchmarks csrc delta_matching tests LICENSE MANIFEST.in NOTICE \
        pyproject.toml README.md setup.py THIRD_PARTY_NOTICES.md .gitmodules .gitignore | tar -x -C kernels/delta-matching

after emptying `kernels/delta-matching/`; then update `PROVENANCE.json` (commit; `code_sha`: `python -c "import sys;
sys.path.insert(0, 'kernels/delta-matching/delta_matching'); import _buildcfg; print(_buildcfg.csrc_sha())"`; `py_sha`:
`PYTHONPATH=src python -c "from pretrain.trainer.base_trainer import kernel_py_sha as h;
print(h('kernels/delta-matching/delta_matching'))"`), rerun `scripts/install_kernel.sh` and the tests above.
