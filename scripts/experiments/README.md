# The experiments

One script per experiment of the paper. Each trains every arm and seed of its experiment, one run after another on all
the GPUs it is given (one node when run by hand; one node, part of one or several nodes in a Slurm job), and evaluates
each trained run; the same command continues an interrupted experiment (a finished training or evaluation is
skipped). Prerequisites: the environment and the kernel package (`docs/SETUP.md` sections 2 and 3) and the data root
(`data/README.md`).

| script | what it runs | arms | seeds | runs | time on 8 x H100 |
|---|---|---|---|---|---|
| `stage1_hd128.sh` | the head_dim-128 cell: 1.665 B GatedDeltaNet : GQA 3:1, 30.0 B tokens | `clean_hd128`, `dm_stale_hd128`, `dm_matched_hd128`, `dm_consistent_hd128`, `te_fp8` | 11785, 2 | 10 | about 1 day per run (3.1 s per step) plus 25 minutes of evaluation |
| `stage1_hd256_scale.sh` | head_dim 256 at three sizes, 30.0 B tokens each | `xs_*` (0.569 B), no prefix (1.665 B), `l_*` (5.289 B): `clean_hd256`, `dm_stale_hd256`, `dm_matched_hd256` | 11785, 2 | 18 | about 3 days per 5.289 B run (9.3 to 9.6 s per step), 1 day at 1.665 B, less at 0.569 B |
| `stage1_l_kda.sh` | the 5.293 B KDA : GQA twin of the 5.289 B cell (head_dim 256), 30.0 B tokens | `l_kda_clean_hd256`, `l_kda_dm_stale_hd256`, `l_kda_dm_matched_hd256` | 11785 | 3 | about 2 days per run on four 4-GPU nodes (5.0 to 5.2 s per step, 70 GB peak), about 11 four-hour links |
| `stage1_xl.sh` | the 9.275 B GatedDeltaNet : GQA cell (qwen35_xl.yaml: the Qwen3.5-9B-Base block at 36 layers, head_dim 256), 30.0 B tokens, activation checkpointing on | `xl_clean_hd256`, `xl_dm_stale_hd256`, `xl_dm_matched_hd256` | 11785 | 3 | about 41-46 h per run on 32 GPUs (8 nodes x 4; estimated, 4.9-5.4 s per step with selective checkpointing) |
| `stage1_architectures.sh` | other architectures, head_dim 128, 30.0 B tokens each | `mla_*` (GDN : MLA), `kda_*` (KDA : GQA), `mamba2_*` (Mamba2 : GQA), `gqa_*` (pure GQA): `clean_hd128`, `dm_stale_hd128`, `dm_matched_hd128` | 11785, 2 (pure GQA: 11785, 1) | 24 | about 1 day per run |
| `stage1_nope.sh` | the head_dim-128 cell without positional encoding | `nope_clean_hd128`, `nope_dm_stale_hd128`, `nope_dm_matched_hd128` | 11785, 2 | 6 | about 1 day per run |
| `stage1_muon.sh` | the head_dim-128 cell trained with Muon | `muon_clean_hd128`, `muon_dm_stale_hd128`, `muon_dm_matched_hd128` | 11785 | 3 | about 1 day per run |
| `stage1_hd128_auxout.sh` | the reviewer's Attn-QAT comparison on the head_dim-128 cell: the auxiliary-output delta (dm_mode aux_out) beside a rerun of Delta-Matching on the same kernel build and node count, 30.0 B tokens | `dm_auxout_hd128`, `dm_matched_hd128` | 11785 | 2 | about 1 day per run on 8 GPUs; run on more nodes (same global batch, lower gradient accumulation) |
| `stage2_context.sh` | the 8K to 64K context extension of the head_dim-128 cell (28,000 steps x 1 M tokens) | `clean_hd128`, `dm_matched_hd128`, `dm_stale_hd128`, `te_fp8` | 11785, 2 | 8 | about 1.5 days per run (4.7 s per step) plus 3 hours of evaluation |

Arm names map to the paper as `clean` = Standard BF16/FP32 Mix-precision, `dm_matched` = Delta-Matching, `dm_stale` =
Naive FP8 (with Stale Delta), `dm_consistent` = Naive FP8 (Consistent dO), `dm_auxout` = the auxiliary-output delta (Attn-QAT's correction, a baseline), `te_fp8` = cuDNN / Transformer Engine FP8.
Every arm of an experiment trains the same model from the same initialization (the seed) on the same data order; only
the attention numerics differ. `scripts/campaign/arms.sh` defines every arm.

## Running

    scripts/experiments/stage1_hd128.sh                                  # every arm and seed, in order
    scripts/experiments/stage1_hd128.sh dm_matched_hd128 dm_stale_hd128  # only these arms
    SEEDS=11785 scripts/experiments/stage1_hd128.sh clean_hd128          # only this seed
    DRY=1 scripts/experiments/stage1_hd256_scale.sh                      # print the plan: arms, seeds, run names, commands
    SMOKE=1 scripts/experiments/stage1_architectures.sh                  # a 3-step training smoke of every arm, no evaluation
    EVAL=0 scripts/experiments/stage1_nope.sh                            # train only

    # on a Slurm cluster: one chain of 4-hour jobs per arm (scripts/slurm/README.md); -N 2 trains each run on two nodes
    scripts/slurm/submit.sh -A <account> -p batch -t 4:00:00 -- scripts/experiments/stage1_hd128.sh dm_matched_hd128
    scripts/slurm/submit.sh -A <account> -p batch -t 4:00:00 -N 2 -- scripts/experiments/stage1_l_kda.sh l_kda_dm_matched_hd256

One invocation trains one run at a time. Every run is independent of the others, except stage 2, which warm-starts from
`clean_hd128`'s stage-1 export of the same seed: to train several at once, give each node, or each Slurm chain, its
own arms or seeds. Inside a Slurm job of several nodes each training uses every node (`scripts/campaign/train_arm.sh`),
while the evaluations run on the job's first node only. The workspace (`DM_WORKSPACE`) must be visible to every node
that trains or continues a run.

The logs are under `$DM_WORKSPACE/logs/` and the results under `$DM_WORKSPACE/qwen35_m/eval/<run>/`;
`scripts/campaign/report_table.py` prints every experiment's table (mean +/- one standard error over seeds),
`scripts/campaign/paired_pair.py` the paired per-window val-CE contrasts (`docs/benchmark.md`).

## The driver

Every script names its arms and seeds and sources `_run.sh`, the shared driver (its header is the reference). One
invocation works in two phases:

- **A.** First, while the job has all its time, the evaluations still owed: every planned run whose export carries
  `TRAIN_COMPLETE` but has no evaluation marker (`EVAL_COMPLETE`; stage 2: `EVAL64K_COMPLETE`) is evaluated when the
  evaluation fits in the job, else it stays deferred.
- **B.** Then in plan order: a run whose export carries `TRAIN_COMPLETE` is skipped without calling its launcher; any
  other run is trained (no training starts less than `DM_MIN_TRAIN_TIME` seconds, 600, before the trainer's deadline),
  then evaluated when that fits in what is left of the job. An evaluation that does not fit is deferred to the next
  job's phase A, and the time left goes to the next run's training. A training that stops early with a checkpoint
  (exit 75) ends the invocation.

The evaluation budget: under a deadline (`DM_STOP_BEFORE` and `DM_JOB_END`, which `scripts/slurm/job.sh` exports) an
evaluation starts only when the job's end leaves `DM_EVAL_BUDGET` seconds: 2400 by default for stage 1 (about 25
minutes on 8 GPUs, plus slack), 4800 for `stage1_l_kda.sh` (a 5.3 B export takes longer), 11700 for stage 2 (about
3 hours). A job that started less than 10 minutes ago evaluates anyway (the next job would have no more time). A
SIGTERM (the time limit, scancel) lets the running training or evaluation shut down (the trainer saves a checkpoint),
then the driver exits 75.

Failures. Outside a chain a failure is reported and the other runs go on. In a Slurm chain (`DM_CHAIN_DIR`, which
job.sh sets only when a next job will run):

- an exit 2 (a configuration error: a refused resume, missing data, a bad knob; the same in every job) gives that
  training or evaluation up at once, for the rest of the chain;
- any other failure (an exit status other than 0, 75, 2 and 3) is counted in `$DM_CHAIN_DIR/runfails/<run>.train` or
  `<run>.eval` (`<metric> <count> <job>`; the metric is the run's newest complete checkpoint step, or the number of
  completion markers under its eval directory). The count restarts at 1 when the metric grew since the previous
  failure (progress), stays when the previous failure was in this same job (job.sh's in-place retry: a job counts once;
  a requeue is another job), and grows by one otherwise. At `DM_RUN_MAX_FAILS` (2) jobs in a row that step is given up
  for the rest of the chain, while the other runs go on. Success removes the counter; a new `scripts/slurm/submit.sh`
  clears them all;
- an exit 3 (another live job holds the run's lock) lists the run as owned by another job.

Exit status: 0 when every requested run is trained and evaluated (or smoke-tested); 75 when work remains (an early stop
or a stop with a checkpoint, a deferred evaluation, a training not started for lack of time, or, in a chain, a failure
that is retried while another step of this job succeeded): run the same command again (a Slurm chain does); 1 in a
chain when a failure is retried and no step of this job succeeded (a failed job: job.sh retries it in place, then keeps
the next job off the node(s) it failed on), and outside a chain when only failures and runs owned by another job are
left; 4 in a chain when only given-up runs and runs owned by another job are left (the chain stops: fix the cause,
then submit again); 2 for the driver's own configuration errors (an arm outside the experiment, a malformed knob). A
summary at the end lists what is done, deferred, retried, given up and owned.

Knobs: `SEEDS`, `DRY=1`, `SMOKE=1`, `EVAL=0` (above), `DM_EVAL_BUDGET`, `DM_MIN_TRAIN_TIME`, `DM_RUN_MAX_FAILS`; every
knob of `train_arm.sh`, `stage2_arm.sh` and `eval_arm*.sh` passes through (`TAG`, `DM_LR`, `EXTRA_ARGS`,
`DM_SAVE_INTERVAL`, ...).

## The 5.3 B KDA experiment (`stage1_l_kda.sh`)

The KDA : GQA 3:1 twin of the 5.289 B GatedDeltaNet : GQA cell of `stage1_hd256_scale.sh`, with that cell's recipe:
`src/pretrain/configs/qwen35_l_kda.yaml` keeps everything of `qwen35_l.yaml` but the recurrent mixer (Kimi Delta
Attention, 32 heads x 128) and the FFN width (parameter parity, 5,292,698,368 parameters); its header has the design and
the early-training diagnosis. The fp8 arms run the full fp8 pipeline with the Delta-Matching kernel at head_dim 256, the
Standard arm FlashAttention-3 bf16 attention with an fp8 FFN; the KDA projections stay bf16 in every arm. The learning
rate is the GDN cell's 2.4e-3. A 30-step probe of this cell showed grad-norm spikes (and CE spikes) at steps 4 to 20
that recover by about step 25, and 1.2e-3 halves them, so the clean arm also runs a hedge at 1.2e-3 through the warmup,
a run of its own (`TAG=_lr1p2e3`, so its name ends in `_lr1p2e3`) in a chain of its own. One chain per arm, here on four
4-GPU nodes:

    OPTS=(-A <account> -p batch -N 4 --gpus-per-node=4 --cpus-per-task=64 --mem=800G -t 4:00:00)
    for arm in l_kda_clean_hd256 l_kda_dm_stale_hd256 l_kda_dm_matched_hd256; do
      scripts/slurm/submit.sh "${OPTS[@]}" -- scripts/experiments/stage1_l_kda.sh $arm
    done
    TAG=_lr1p2e3 DM_LR=1.2e-3 scripts/slurm/submit.sh "${OPTS[@]}" -- scripts/experiments/stage1_l_kda.sh l_kda_clean_hd256

Measured with this layout (2026-11-09, 16 H100s per run, no activation checkpointing): 4.99 s per step for the Standard
arm, 5.11 for Naive FP8 and 5.19 for Delta-Matching, 70 GB peak on every arm; the whole run is about 40 hours of steps.
The hedge decided the rate: from the end of the warmup to step 2000 the 2.4e-3 arms showed no loss or grad-norm spike
and the 2.4e-3 Standard arm led the hedge by 0.013 to 0.020 nats, so every arm keeps 2.4e-3 and the hedge was stopped
(the yaml's header).

## Notes

- `te_fp8` runs in a separate Python environment with Transformer Engine (`docs/SETUP.md` section 2): set `DM_TE_PY`
  and `DM_TE_TORCHRUN` before `stage1_hd128.sh` or `stage2_context.sh`, or leave `te_fp8` out of the arm list.
- The head_dim-256 Standard arms need the FlashAttention-3 package (`flash_attn_interface`).
- The 1.665 B GDN : MLA and KDA : GQA configs (`qwen35_m_mla.yaml`, `qwen35_m_kda.yaml`; not the 5.3 B
  `qwen35_l_kda.yaml`, which trains 1) train 4 sequences per GPU per micro-step (the paper's runs had 141 GB GPUs). On
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
