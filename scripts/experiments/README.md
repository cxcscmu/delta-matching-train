# The experiments

One script per experiment of the paper. Each trains every arm and seed of its experiment on one node, one run after
another, then evaluates it; the same command continues an interrupted experiment (a finished training or evaluation is
skipped). Prerequisites: the environment and the kernel package (`docs/SETUP.md` sections 2 and 3) and the data root
(`data/README.md`).

| script | what it runs | arms | seeds | runs | time on 8 x H100 |
|---|---|---|---|---|---|
| `stage1_hd128.sh` | the head_dim-128 cell: 1.665 B GatedDeltaNet : GQA 3:1, 30.0 B tokens | `clean_hd128`, `dm_stale_hd128`, `dm_matched_hd128`, `dm_consistent_hd128`, `te_fp8` | 11785, 2 | 10 | about 1 day per run (3.1 s per step) plus 25 minutes of evaluation |
| `stage1_hd256_scale.sh` | head_dim 256 at three sizes, 30.0 B tokens each | `xs_*` (0.569 B), no prefix (1.665 B), `l_*` (5.289 B): `clean_hd256`, `dm_stale_hd256`, `dm_matched_hd256` | 11785, 2 | 18 | about 3 days per 5.289 B run (9.3 to 9.6 s per step), 1 day at 1.665 B, less at 0.569 B |
| `stage1_architectures.sh` | other architectures, head_dim 128, 30.0 B tokens each | `mla_*` (GDN : MLA), `kda_*` (KDA : GQA), `mamba2_*` (Mamba2 : GQA), `gqa_*` (pure GQA): `clean_hd128`, `dm_stale_hd128`, `dm_matched_hd128` | 11785, 2 (pure GQA: 11785, 1) | 24 | about 1 day per run |
| `stage1_nope.sh` | the head_dim-128 cell without positional encoding | `nope_clean_hd128`, `nope_dm_stale_hd128`, `nope_dm_matched_hd128` | 11785, 2 | 6 | about 1 day per run |
| `stage1_muon.sh` | the head_dim-128 cell trained with Muon | `muon_clean_hd128`, `muon_dm_stale_hd128`, `muon_dm_matched_hd128` | 11785 | 3 | about 1 day per run |
| `stage2_context.sh` | the 8K to 64K context extension of the head_dim-128 cell (28,000 steps x 1 M tokens) | `clean_hd128`, `dm_matched_hd128`, `dm_stale_hd128`, `te_fp8` | 11785, 2 | 8 | about 1.5 days per run (4.7 s per step) plus 3 hours of evaluation |

Arm names map to the paper as `clean` = Standard BF16/FP32 Mix-precision, `dm_matched` = Delta-Matching, `dm_stale` =
Naive FP8 (with Stale Delta), `dm_consistent` = Naive FP8 (Consistent dO), `te_fp8` = cuDNN / Transformer Engine FP8.
Every arm of an experiment trains the same model from the same initialization (the seed) on the same data order; only
the attention numerics differ. `scripts/campaign/arms.sh` defines every arm.

## Running

    scripts/experiments/stage1_hd128.sh                                  # every arm and seed, in order
    scripts/experiments/stage1_hd128.sh dm_matched_hd128 dm_stale_hd128  # only these arms
    SEEDS=11785 scripts/experiments/stage1_hd128.sh clean_hd128          # only this seed
    DRY=1 scripts/experiments/stage1_hd256_scale.sh                      # print the plan: arms, seeds, run names, commands
    SMOKE=1 scripts/experiments/stage1_architectures.sh                  # a 3-step training smoke of every arm, no evaluation
    EVAL=0 scripts/experiments/stage1_nope.sh                            # train only

One node runs one run at a time. To use several nodes, give each node its own arms or seeds (every run is independent
of the others, except stage 2, which warm-starts from `clean_hd128`'s stage-1 export of the same seed). The workspace
(`DM_WORKSPACE`) must be visible to every node that continues a run.

Exit status: 0 when every requested run is trained and evaluated; 75 when a training stopped early with a checkpoint
(run the same command again to resume); 1 with a list of what is missing otherwise. The logs are under
`$DM_WORKSPACE/logs/` and the results under `$DM_WORKSPACE/qwen35_m/eval/<run>/`; `scripts/campaign/report_table.py`
prints every experiment's table (mean +/- one standard error over seeds), `scripts/campaign/paired_pair.py` the paired
per-window val-CE contrasts (`docs/benchmark.md`).

## Notes

- `te_fp8` runs in a separate Python environment with Transformer Engine (`docs/SETUP.md` section 2): set `DM_TE_PY`
  and `DM_TE_TORCHRUN` before `stage1_hd128.sh` or `stage2_context.sh`, or leave `te_fp8` out of the arm list.
- The head_dim-256 Standard arms need the FlashAttention-3 package (`flash_attn_interface`).
- The GDN : MLA and KDA : GQA configs train 4 sequences per GPU per micro-step (the paper's runs had 141 GB GPUs). On
  GPUs with less than 100 GB the launcher runs 2 per GPU instead (`scripts/campaign/arms.sh`; 45 to 47 GB peak on an
  80 GB H100): the global batch of 128 sequences, the data order and the schedule stay the same, with twice the gradient
  accumulation. Pass
  `EXTRA_ARGS="--per_device_batch_size <n>"` to choose another value.
- `stage1_hd256_scale.sh` runs every size with the same recipe: the full fp8 pipeline (fp8 FFN and q/k/v/o, the
  Delta-Matching kernel at head_dim 256) for the fp8 arms, FlashAttention-3 bf16 attention with an fp8 FFN for the
  Standard arm. The paper's 1.665 B head_dim-256 cell instead ran fp8 attention only (bf16 FFN and projections) on an
  earlier realization of the kernel that is not part of these repositories.
- The per-run step times, memory and knobs are in `docs/SETUP.md` section 5; stage 2's recipe is in the header of
  `scripts/campaign/stage2_arm.sh`.
