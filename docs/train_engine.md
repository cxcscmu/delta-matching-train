> The engineering notes of the training engine. The measurements quoted below were taken while the engine was built,
> on a 360M model (its first launch), not on the study's runs; the fp8 attention mixers, the campaign launchers and the
> environment are documented in `docs/SETUP.md` and the module docstrings.

# `src/pretrain`: from-scratch pretraining engine (DDP + FSDP2)

## 1. Purpose

A self-contained LM pretraining engine for the Delta-Matching study: the same trainer runs the bf16 Standard arm, the
fp8 Delta-Matching arms (`dm_mode` stale / matched / bf16_dp) and the Transformer Engine arm, over dense GQA,
GatedDeltaNet, Mamba2, KDA and MLA mixers and their 3:1 hybrids. It is modeled on NVIDIA's **jetlm** training framework
(the FSDP2 sharding idiom, DCP checkpointing, resumable streaming data, the WSD schedule, a draccus-style nested config)
but rebuilt for **from-scratch pretraining** (jetlm is a LoRA / length-extrapolation *fine-tuning* engine) and made
**dependency-light** (no draccus / more_itertools; `fla`, `mamba_ssm`, `flash_attn`, `transformer_engine` and the
`delta_matching` kernel package are imported lazily by the mixers that need them).

**Status.** Every bf16 mixer trains end-to-end through DDP + FSDP2 (§13) with exact resume (§7); the Delta-Matching
fp8 routes (the `dm` and `mla_dm` mixers, `attn_impl dm`) need FSDP2 (they run on the bf16 activations of its
mixed-precision policy; DDP keeps them fp32), so `--backend ddp` with them is refused at startup (FSDP2 also runs on
one GPU). Real-data multi-node training, the HF export and the train-to-eval loop are verified (§14);
`--compile --fused_ce` are the throughput levers (§14.5). FP8 GEMMs (`--fp8`, torchao) give no gain at 360M (§14.6); at the study's 1.7 B and above they are the default
of every fp8 arm (`--fp8_scope ffn` for the Standard arm, `ffn_qkvo` for the Delta-Matching arms).

## 2. Module map

```
src/pretrain/
  config.py              TrainConfig/OptimConfig/DataConfig + dependency-free YAML/CLI loader
  train.py               entry point: dist_init -> data -> model -> wrap -> Trainer -> train
  modeling/
    config.py            ModelConfig (dims, GQA, SwiGLU sizing, z-loss, mixer-select, qk_norm, dm_mode) + analytic param count
    layers.py            RMSNorm, RotaryEmbedding (+ YaRN), apply_rotary (rotate_half), SwiGLUMLP, output gate
    attention.py         GQAAttention: attn_impl flash | sdpa | dm (the plain Delta-Matching kernel) | te_fp8
    dm_attention.py      DMAttention: the dm mixer (fp8 q/k/v/o, fused producers, the Delta-Matching kernel; dm_mode)
    te_attn.py           attn_impl te_fp8: Transformer Engine's fp8 fused attention core (DelayedScaling)
    mla.py               MLAAttention (bf16 core) / MLADMAttention (the Delta-Matching core)
    mixers.py            MIXER_REGISTRY {gqa, gated_deltanet, mamba2, kda, dm, mla, mla_dm} / register_mixer / build_mixer
    gated_deltanet.py    GatedDeltaNetMixer -> fla.layers.GatedDeltaNet      (lazy import)
    mamba2.py            Mamba2Mixer       -> mamba_ssm.Mamba2               (lazy import)
    kda.py               KDAMixer          -> fla.layers.kda.KimiDeltaAttention (lazy import)
    mlp_fp8glue.py       Fp8GlueSwiGLUMLP: the fp8 cast-glue MLP (kernel/fp8glue), installed when DM_FP8GLUE=1
    block.py             DecoderBlock (pre-norm: x += mixer(norm x); x += mlp(norm x))
    model.py             Transformer backbone + PretrainLM (tied head, CE+z-loss, fused linear-CE, config (de)serialize)
  data/
    memmap_dataset.py    MemmapTokenDataset: flat uint16/uint32 token files -> fixed chunks
    dist_sampler.py      ResumableDistSampler: (seed,epoch,iteration) -> exact-resume order
    data_loader.py       build_train_dataloader -> DistDataLoader (.sampler/.batch_size/.pad_token_id); doc-aware collate
    doc_dataset.py       DocAwareMemmapDataset: .bin + doc-length sidecar -> position_ids + cu_seqlens (§15)
  trainer/
    base_trainer.py      shared loop, token-weighted grad-accum, clip, logging (tok/s, peak_mem_gb), ckpt orchestration, resume
    fsdp2_trainer.py     FSDP2 backend (DCP sharded save/load + DTensor gather; set_requires_gradient_sync)
    ddp_trainer.py       DDP backend (per-rank replica save; require_backward_grad_sync gating)
    stateful.py          TrainState / ModelWrapper / OptimizerWrapper (torch DCP Stateful)
  parallel/
    dist.py              dist_init (nccl/gloo), is_master, sync_tensor, barriers
    wrap.py              build_ddp_model, build_fsdp2_model (fully_shard), AC (full/selective), compile, fp8, build_optimizer, build_lr_scheduler
    muon.py              MuonWithAdamW (opt-in optimizer, §6)
  hf/                    the dmlm HF bridge (configuration_dmlm / modeling_dmlm / export / register_auto) -> docs/eval_engine.md
  utils/
    ckpt.py              complete-marker atomic swap + version-string discovery
    logging.py, misc.py  rank-aware logging; seed, Peekable, amp dtype, param count
  configs/               the study's model configs (qwen35_*.yaml hybrids, qwen3_*.yaml pure attention) + README
  scripts/
    gen_synthetic_data.py   tiny token .bin generator
    smoke_test.py           the 8-check harness (§9)
    bench_mixers.py         single-layer mixer throughput/memory benchmark (§12)
    test_compile_fusedce.py compile + fused-CE correctness (§14.5)
    test_doc_loader.py      DocAwareMemmapDataset checks (§15)
    test_fused_layers.py    fused (Liger) RMSNorm / RoPE / SwiGLU vs the eager ops, forward and backward (GPU)
    verify_hf_export.py     dmlm round-trip bit-exactness + qk_norm case + dropped-key check (docs/eval_engine.md); the GQA
                            cases run on CPU, the GatedDeltaNet / Mamba2 cases need CUDA and are skipped without it
```
(The study's corpora, `.bin` + `.doc_lengths.npy` each, are built by the scripts in `data/`: `data/README.md`.)

## 3. The model (dense GQA): math

Llama-style pre-norm decoder. For hidden width `d`, `H` query heads, `H_kv` KV heads
(GQA group size `n_rep = H/H_kv`), head dim `d_h`:

- **RMSNorm** (scale-only, no mean-subtraction; reduction in fp32 for stability):
  `RMSNorm(x) = x / sqrt(mean(x²) + ε) · γ`.
- **RoPE** (NeoX/Llama rotate-half), applied to Q,K before attention:
  `inv_freq_i = θ^(−2i/d_h)`, angle at position `m` is `m·inv_freq`. With
  `rotate_half([a;b]) = [−b;a]`, `q' = q·cos + rotate_half(q)·sin`. Because the rotation
  is orthogonal and position-dependent, `⟨R_m q, R_n k⟩` depends only on `m−n` →
  relative-position attention. Built/applied in fp32, cast back.
- **GQA attention**: `Attn(Q,K,V) = softmax(QKᵀ/√d_h + causal_mask)·V`, with each KV head
  broadcast to `n_rep` query heads. K/V projections are `d×(H_kv·d_h)` instead of
  `d×(H·d_h)` → the KV cache shrinks by `n_rep` at fixed query resolution.
- **SwiGLU MLP**: `down(silu(gate(x)) ⊙ up(x))`, `silu(z)=z·σ(z)`. Intermediate size
  defaults to `round_up(8/3·d, 256)`, which reproduces the shapes of the engine's first launch (the 360M model):
  `d=768→2048` and `d=1024→2816`.
- **Decoder block**: `x += mixer(RMSNorm(x));  x += mlp(RMSNorm(x))`.
- **Head**: tied (`lm_head.weight is embed_tokens.weight`).

**Loss** = next-token cross-entropy + z-loss. Logits/labels are shifted (position `t`
predicts token `t+1`); CE in fp32 with `ignore_index=−100`:
`L_CE = −(1/N) Σ_t log p(x_{t+1}|x_{≤t})`. The **z-loss** regularizes the log-partition
toward 0 (prevents logit drift): `L_z = λ_z · E_t[(logsumexp(z_t))²]`,
default `λ_z = 1e−4`, averaged over the same non-ignored positions as CE.

**Init**: linear/embedding weights `N(0, 0.02)`; residual-output projections (`o_proj`,
`down_proj`) scaled by `1/√(2L)` (GPT-2/NeoX) to keep residual-stream variance ~constant
in depth; tie head after init.

**Closed-form parameter count** (tied embeddings), verified equal to the live module in
the smoke test:
```
N = V·d  +  L·( 2d² + 2d²·(H_kv/H) + 3·d·d_ff + 2d )  +  d
    embed      attn-proj             SwiGLU    2 norms   final norm
```

## 4. Attention backends (flash / SDPA / dm / te_fp8)

`modeling/attention.py` resolves `attn_impl ∈ {auto, flash, sdpa, dm, te_fp8}` at construction:
- **flash**: `flash_attn.flash_attn_func` (FA2) if importable; `DM_FLASH_IMPL=fa3` pins the FA3 hopper wheel
  (`flash_attn_interface`), `fa2` pins FA2. A candidate is accepted by default only if its `flash_attn_func` takes
  `dropout_p`, which FA3's does not, so installing the FA3 wheel never switches the default. Native GQA (K,V keep
  `H_kv` heads); causal.
- **sdpa**: `torch.nn.functional.scaled_dot_product_attention`. On H100 this dispatches to
  a fused flash/cuDNN kernel (`DM_SDPA_BACKEND=cudnn|flash|efficient|math` pins it); correct everywhere (incl. CPU).
  Native GQA through `enable_gqa` (no `repeat_kv` copy of K,V).
- **auto** = flash if `FLASH_AVAILABLE` else sdpa.

**flash-attn status (when the engine was built):** `flash-attn 2.8.3.post1` (FA2) is installed:
built from source against torch 2.12 + **CUDA 13** (no prebuilt wheel exists; sm_90-only).
It imports and runs GQA@16384 fine. **But the §12 benchmark
found torch SDPA is 1.13–1.33× *faster* than this FA2 on H100/torch-2.12** (SDPA dispatches
to a cuDNN/FA3-class fused kernel), so **SDPA stays the practical dense default here**; the
real flash win needs **FA3-hopper** (separate build from the flash-attention `hopper/` dir)
or **FA4-Blackwell** (different hardware). `attn_impl="auto"` currently prefers flash when
importable; the campaign's clean arms pin `sdpa` with `DM_SDPA_BACKEND=cudnn` at head_dim 128 and the FA3
wheel (`flash` + `DM_FLASH_IMPL=fa3`) at head_dim 256.
- **dm** (`DM_IMPL`): the plain Delta-Matching kernel wrapper `delta_matching.dm_attn_func_bhnd` on the GQA mixer's bf16
  q/k/v, with `dm_mode` stale or matched (the same plain route the `mla_dm` mixer takes). The campaign's
  attention arms use the **`dm` mixer** instead (`modeling/dm_attention.py`: fp8 q/k/v/o projections, the fused RoPE +
  quantization producers of `kernel/dm_port`, the same kernel), selected by `layer_mixers` / `mixer`. The kernel runs
  head_dim 128 (sequence length a multiple of 256) or 256 (a multiple of 128), self-attention only.
- **te_fp8**: Transformer Engine's fp8 fused attention core (`modeling/te_attn.py`, head_dim 128 only, the Transformer
  Engine FP8 arm).

## 5. Data pipeline

- **On disk**: flat token-id files, `uint16` (vocab ≤ 65535) or `uint32`, packed and
  chunked into fixed `max_seq_len` sequences (trailing remainder dropped). `.bin` (raw,
  via `np.memmap`) or `.npy` (`np.load(mmap_mode)`); a dir is globbed for `*.bin`/`*.npy`.
  Memmaps open lazily per worker (pickles cleanly to DataLoader workers).
- **One loader batch = one optimizer step's per-rank data**: `batch_size = global_batch //
  world`; the trainer micro-chunks it into `grad_accum` micro-batches.
- **Exact resume (`ResumableDistSampler`)**: the global sample order is a pure function of
  `(seed, epoch)`. Each optimizer step consumes `world × per_rank_bs` samples. To resume
  at completed-step `k`: `epoch = k // B`, `i = k mod B` (B = batches/epoch =
  `num_samples // batch_size`); set the sampler offset to `i · per_rank_bs` **samples**,
  which skips `offset · world` global samples, then rank-strides. So `(seed, epoch,
  iteration)`: all derived from `global_step`: pin the exact next batch with no extra
  state. (Verified bit-exact: smoke check 7, maxΔ=0.)

## 6. Trainers (DDP + FSDP2)

Shared `BaseTrainer` owns the loop; backends implement ~5 hooks.

- **Mixed precision.** Model built in fp32. **FSDP2**: `MixedPrecisionPolicy(param=bf16,
  reduce=fp32)`: sharded params kept fp32 (the **fp32 master**), all-gathered+cast to bf16
  for compute, grads reduce-scattered in fp32. **DDP**: fp32 replicated params + bf16
  `autocast` compute + fp32 grads. Both realize "**bf16 compute / fp32 master**".
  (fp16 loss-scaling is intentionally not wired: bf16 is the H100/Blackwell path.)
- **Gradient accumulation (token-weighted, exact).** Per-rank batch `local_bs` split into
  `A` micro-batches; micro `i` has `v_i` valid (non −100) target tokens, `V = Σ v_i`. The
  model returns `CE_i = (1/v_i)Σ_tok ℓ` (plus the z-loss term). We scale micro `i` by `r_i = v_i/V` before
  `backward`, so the accumulated grad equals `Σ_i r_i·CE_i = (1/V)Σ_tok ℓ`: the rank
  token-mean. DDP/FSDP then mean-reduce across ranks (exact global token-mean when each
  rank has equal `V`, i.e. packed fixed-length data; otherwise the mean-of-rank-means:
  switch to all-reduced token weights if you train on padded data). Verified exact:
  smoke check 5, (pdbs=4,accum=1) == (pdbs=2,accum=2), maxΔ=0.
- **Grad-sync gating.** Only the last micro-batch all-reduces/reduce-scatters grads:
  FSDP2 `set_requires_gradient_sync(last)`, DDP `require_backward_grad_sync = last`.
- **Clipping.** `clip_grad_norm_(params, max_grad_norm)`; the FSDP2 DTensor result is
  `.full_tensor()`-reduced to the global norm. Default `max_grad_norm = 1.0`.
- **Optimizer/schedule.** AdamW(betas=(0.9,0.95), eps=1e-8) with a 2-group weight-decay
  split (norms / embeddings / biases get wd=0; rest get `weight_decay`, default 0.1). LR =
  **WSD** via `transformers.get_wsd_schedule` (linear warmup → constant → cosine/linear
  decay), or `constant_with_warmup`. Tied embed/head appears once (named_parameters dedups).
- **Muon (opt-in `--optim.name muon`; default `adamw`).** `parallel/muon.py::MuonWithAdamW`, one optimizer
  object (the trainer, the WSD LambdaLR and DCP see one thing): Muon (torch.optim.Muon's math: EMA momentum 0.95, Nesterov,
  5 quintic Newton-Schulz steps in bf16) on every 2-D weight not matching `muon_exclude_regex` (embed_tokens / lm_head); an
  inner torch AdamW, re-linked to this optimizer's groups and state every step, on the rest (embeddings, norm and qk-norm
  gains, GDN conv weights and A_log / dt_bias), with the AdamW decay rule above. Muon's lr is `lr * 0.2 * sqrt(max(m, n))`
  (`muon_adjust_lr match_rms_adamw`, Moonlight's scale) with decoupled decay `1 - lr * weight_decay`, so the AdamW lr,
  decay and schedule carry over. Under FSDP2 each rank gathers the momentum-corrected matrix (`full_tensor()`), runs the
  same Newton-Schulz and applies its own rows (`distribute_tensor(..., src_data_rank=None)`); the momentum buffer is made
  from the gradient, so torchao's fp8 all-gather weight wrapper is never copied. Banner `[optim] MUON: <n> 2-D tensors on
  Muon ...`. Tests `tests/test_muon.py`: exact equality with torch.optim.Muon + AdamW on CPU, an exact state_dict round
  trip, and a 2-GPU FSDP2 run (an uneven row split included) equal to the unsharded reference within 1e-6.
- **Activation checkpointing** (`--gradient_checkpointing`): each block wrapped in a
  non-reentrant `checkpoint_wrapper` *before* sharding.

**FSDP2 sharding** (`build_fsdp2_model`): one `fully_shard`
per `DecoderBlock` + a root wrap; `reshard_after_forward=True` on all blocks except the
last (kept gathered for the immediate backward); default 1-D mesh over the process group
(pure data-parallel FSDP). Per-rank memory ≈ (fp32 params + fp32 AdamW moments +
grad)/world + bf16 transient unsharded layer + activations (cut by AC).

## 7. Checkpoint & resume

`checkpoint-<step>/` contains, written in this order with the `.complete` marker LAST
(atomic rename, content = `VERSION_STRING`):
- **resume state**: FSDP2: a DCP sharded save of `{train_state, model, optimizer,
  lr_scheduler}` (`__r_0.distcp` per rank + `.metadata`); DDP: per-rank
  `training_state_<rank>.pt` (replicas are identical; node-local-`/tmp` safe).
- **`pytorch_model.bin`**: consolidated full (unsharded) state dict, gathered on the
  master via `DTensor.full_tensor()`, keys stripped of wrapper prefixes → loadable by a
  fresh `PretrainLM` (verified: 0 missing/unexpected, finite loss). Plus `model_config.json`.
- per-rank **`next_train_data_<rank>.pt`** for the post-resume **sanity check** (the first resumed batch must equal
  the saved next batch; raises if `force_resume_check`, the default). The data position itself is re-derived from
  `global_step` (§5).

Resume finds the newest `checkpoint-*` whose `.complete` content matches `VERSION_STRING`
(a version mismatch is treated as incompatible). Rotation runs on the master right after a new checkpoint's `.complete`
marker: it keeps the newest `save_total_limit` complete checkpoints, the new one included (None or 0 keeps all),
removes partial directories older than the newest complete one, and leaves directories without a step number
(e.g. `checkpoint-final`) alone.

## 8. Config & launch

**Config** (`config.py`): nested `@dataclass` (`TrainConfig` ⊃ `ModelConfig`/`OptimConfig`/
`DataConfig`), `__post_init__`/`derive(world)` computes `gradient_accumulation_steps =
global_batch // (per_device_batch · world)` and the run dirs. Loader merges **dataclass
defaults < `--config a.yaml` < `--section.key value` CLI**. Values use an int→float→yaml
coercion ladder (so `--optim.lr 3e-3` parses as a float: PyYAML alone loads bare
scientific notation as a *string*). Unknown keys fail: a YAML key or a `--section.key` override that names no field
raises a ValueError listing every such key, dotted, with a close-match hint (e.g. `unknown config key(s):
model.dm_mdoe (did you mean model.dm_mode?)`); free-form dict fields (`rope_scaling`) take any sub-key.

**Launch** (single node, N GPUs):
```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=N -m pretrain.train \
  --backend fsdp2 --config my_run.yaml [--optim.lr 2.4e-3 --num_steps 30000 ...]
```
Single process (1 GPU / CPU): `PYTHONPATH=src python -m pretrain.train ...` (synthesizes a 1-rank world). The campaign's launcher
is `scripts/campaign/train_arm.sh <arm> <seed>` (one node, `DM_NPROC` GPUs; `docs/SETUP.md` section 5).
**Multi-node**: launch one `torchrun --nnodes=K --node_rank=<i> --master_addr=<head> ...` per node (under SLURM, for
example, with `srun --nodes=K --ntasks-per-node=1`). **Storage**: checkpoints + token data go to the workspace (`DM_WORKSPACE`: node-local or
shared scratch, never a small home quota); DCP shards land per-node, so keep the rank→node mapping stable across
resume or use shared storage (the preemptible-cluster setup of `docs/SETUP.md` section 8), and copy checkpoints off
the node for durability.

## 9. Smoke test (verified)

`PYTHONPATH=src torchrun --standalone --nproc_per_node=2 -m pretrain.scripts.smoke_test`
builds tiny synthetic data and runs 8 checks. **All pass (2×H100, when the engine was built):**

| # | check | result |
|---|---|---|
| 1 | param count == analytic formula | PASS (754560 == 754560) |
| 2 | causal masking (future token never changes earlier logits) | PASS (maxΔ = 0) |
| 3 | forward/backward finite (loss + grads) | PASS |
| 4 | learning: overfit a fixed batch → CE collapse (fsdp2 & ddp) | PASS (4.24 → 0.019) |
| 5 | grad-accum equivalence (accum1 vs accum2, token-weighted, fp32) | PASS (maxΔ = 0) |
| 6 | DDP ↔ FSDP2 loss-trajectory parity (bf16) | PASS (maxΔ = 8.4e-4) |
| 7 | exact checkpoint/resume (vs uninterrupted reference, fp32) | PASS (maxΔ = 0) |

Plus end-to-end `train.py`: WSD + activation checkpointing + mid-loop save + resume to a
higher step, and `pytorch_model.bin` reload into a fresh model: all verified.

## 10. Mixers beyond GQA: GDN, Mamba2, KDA, MLA and the fp8 attention mixers

Every mixer is registered in `MIXER_REGISTRY` (`modeling/mixers.py`), lazy-imports its backend (so a GQA-only run needs
none of them) and shares the `forward(x, cos, sin, attention_mask)` signature; the recurrent mixers ignore `cos/sin`:

- `modeling/gated_deltanet.py` (`GatedDeltaNetMixer` → `fla.layers.GatedDeltaNet`)
- `modeling/mamba2.py` (`Mamba2Mixer` → `mamba_ssm.Mamba2`)
- `modeling/kda.py` (`KDAMixer` → `fla.layers.kda.KimiDeltaAttention`, Kimi Delta Attention)
- `modeling/mla.py` (`MLAAttention`: Multi-head Latent Attention with a bf16 core; `MLADMAttention` (`mla_dm`): the same
  layer on the Delta-Matching fp8 core, `dm_mode` stale or matched)
- `modeling/dm_attention.py` (`DMAttention` (`dm`): the GQA mixer's parameters (same registration order, so the seed-matched
  init is shared with the clean arm) with fp8 q/k/v/o projections, the fused RoPE + block-quantization producers and the
  Delta-Matching kernel; `dm_mode` stale / matched / bf16_dp; `dm_fused_producers` must stay true)

Select via `--model.mixer {gated_deltanet|mamba2|kda|mla|dm}`, or a **3:1 hybrid** with
`--model.layer_mixers '[gated_deltanet,gated_deltanet,gated_deltanet,gqa]'` (`mixer_for_layer` cycles the list); the
fp8 arms replace the attention entry (`gqa` → `dm`, `mla` → `mla_dm`), which is what `scripts/campaign/arms.sh` does.
The FSDP2 sharder and optimizer are mixer-agnostic (the optimizer's `ndim<=1` no-weight-decay rule covers SSM
`A_log`/`dt_bias`/`D` automatically). The export patch of the launcher maps the fp8 mixers back to their bf16 twins
(`dm` → `gqa`, `mla_dm` → `mla`) so every arm is evaluated through the same bf16 kernels.

The general recipe to add another mixer:

1. Create `modeling/<mixer>.py` exposing `class X(nn.Module): __init__(self, cfg: ModelConfig, layer_idx);
   forward(self, x, cos, sin, attention_mask)`. Recurrent mixers ignore `cos/sin` (no RoPE): keep the signature for
   drop-in compatibility.
2. `register_mixer("<name>", X)` (import the module in `modeling/__init__.py` so registration runs).
3. Select per layer: `ModelConfig.mixer="<name>"` for a pure stack, or `layer_mixers=[...]` for a hybrid. The FSDP2
   sharder is mixer-agnostic (it shards `DecoderBlock`s), and `build_optimizer` already exempts Mamba
   `A_log`/`dt_bias`-style params from weight decay through its `p.ndim <= 1` rule (`optim.no_wd_regex` adds name
   patterns if needed).
4. Give it an inference-engine subclass (`src/inference/dm_layers.py`) and a slot in `DMStatePool` if the evaluation
   needs generation, and a `remote_code/` port if its exports are to be published (`scripts/hf/HF_CONVERSION.md`).

## 11. Known limitations

- The fp8 attention arms need the `delta_matching` kernel package (Hopper only, `docs/SETUP.md` section 3); the bf16
  arms and the evaluation do not.
- **Single packed dataset** (no weighted multi-domain data *mixture* yet). Document-boundary awareness for ProLong
  **is** implemented (`--data.doc_lengths`, §15: per-doc `position_ids` + `cu_seqlens`); the
  remaining piece is intra-document *attention* masking via flash-varlen (`cu_seqlens` is emitted).
- FP8 training is implemented (`--fp8`, §14.6) but a **measured no-op at 360M** (small GEMMs): off
  by default; the campaign launcher turns it on for every arm at the study's 1.7 B scale.
- bf16 only (no fp16 loss-scaling); no tensor/pipeline/sequence parallelism (pure FSDP/DDP
  data-parallel: sufficient for ≤~1B at 2K context on H100).
- Cross-rank token-weighting assumes equal valid-token counts per rank (true for packed
  fixed-length data); padded/variable data needs all-reduced token weights.
- Fused linear-cross-entropy (liger) and regional `torch.compile` are implemented and tested
  (§14.5): `--fused_ce true --compile true`. Defaults are OFF (unfused/eager) for bit-repro; turn
  both ON for throughput (compile helps GQA, is a no-op for the recurrent hybrids: §15).

## 12. Mixer benchmark (`src/pretrain/scripts/bench_mixers.py`)

One mixer layer, d_model 1024, 16q/4kv heads, head_dim 64, batch 2, bf16, **H100**:

| seq | mixer | fwd+bwd ms | peak GB | tok/s | speed vs FA2 |
|---|---|---:|---:|---:|---:|
| 2048 | gqa-sdpa | 1.21 | 0.18 | 3.39M | 1.13× |
| 2048 | gqa-flash (FA2) | 1.37 | 0.20 | 2.99M | 1.00× |
| 2048 | gdn | 4.18 | 0.62 | 0.98M | 0.33× |
| 2048 | mamba2 | 3.78 | 0.62 | 1.08M | 0.36× |
| **16384** | gqa-sdpa | 12.59 | 1.17 | 2.60M | 1.33× |
| **16384** | gqa-flash (FA2) | 16.69 | 1.06 | 1.96M | 1.00× |
| **16384** | **gdn** | **8.29** | 2.20 | 3.95M | **2.01×** |
| **16384** | **mamba2** | **6.47** | 2.16 | 5.06M | **2.58×** |

**Findings.** (1) **Long-context compute crossover is real**: at 16384, Mamba2/GDN are
2.0–2.6× faster than attention (fwd+bwd attention scales 1.2→12.6 ms = 10.4× for 8× seq ≈
O(L²); Mamba2 3.8→6.5 ms = 1.7× ≈ O(L)). At 2048 attention wins 3× (recurrent overhead
dominates): motivating the 3:1 hybrid. (2) **Memory: flash attention is the training
champion** (1.06 GB); GDN/Mamba2 use ~2× more (chunk-parallel intermediates + internal
expand → 6–8M vs 2.6M params). The SSM memory win is an *inference* property (fixed state
vs growing KV cache), not training activations: not captured here. (3) **SDPA > our FA2**
on this H100/torch-2.12 stack (1.13–1.33× faster, as lean) → keep SDPA as the dense default;
FA3-hopper/FA4-blackwell are the real flash wins. **Backend probe (fwd-only @16384):** SDPA
auto = 2.31 ms, dispatching to **cuDNN** (2.45 / enable_gqa 2.28); SDPA bundled-flash 3.61 ≈
**installed flash_attn FA2 3.59** (same Tri-Dao kernel); mem-efficient 8.28. So SDPA wins by
using cuDNN's Hopper FMHA (~1.5× faster than FA2), and the flash-attn install is *redundant*
(PyTorch already bundles the identical FA2; `flash_attn_interface`/FA3 is not installed:
FA2 only). Reproduce:
`CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m pretrain.scripts.bench_mixers`.

## 13. End-to-end mixer integration test (historical)

Trains a real 4-layer model through the **full FSDP2 / DDP trainers** for every mixer (not just
isolated layers), checking each shards, flows through the optimizer, overfits a fixed batch (learns),
and checkpoints/resumes. **PASS (2×H100):**

| backend · mixer | overfit CE (start→best) |
|---|---|
| fsdp2 · gqa / gated_deltanet / mamba2 / hybrid-3:1 | 4.6–4.7 → 0.013–0.094 |
| ddp · gated_deltanet / mamba2 | → 0.013–0.020 |
| resume · mamba2 (fp32) | bit-exact, maxΔ = 0 |

So full attention and the GDN/Mamba2 3:1 hybrids train end-to-end through both backends, custom params shard and
optimize correctly, and resume is bit-exact. (The harness that produced this table exercised the source repository's
research mixers too and is not shipped; the mixers of this repository, including KDA, MLA and the fp8 attention mixers,
are covered by `tests/test_new_mixers.py`, `tests/test_train_smoke.py` and `tests/test_resume.py`.)

## 14. Real-data, multi-node & throughput (the 360M launch)

The engine was **verified ready for pretraining**: real-corpus training, 4-node (32-GPU)
FSDP2, HF export gathered across nodes, the train→eval loop closed on an actually-trained model, and
a throughput characterization. Data was `gpt2`-tokenized wikitext as a flat uint16 token `.bin`:
`train.bin` 25M tok, `val.bin` 250k tok, on **node-local `$DM_WORKSPACE/`** (heavy data and checkpoints never go
to a small home quota; for multi-node, seed each node's scratch).

**14.1 Real-data non-overfit sanity (single node).** A tiny 50M model (`d=512 L=8 H=8/2 seq=1024`)
trained on real wikitext (not a memorizable fixed batch): **CE 6.65 → 5.66**, WSD LR rising then
decaying, ~430K tok/s, HF-exported to `…/hf_gqa` and reloadable. Confirms the loss/LR/data/export
path on real data (distinct from the §9 overfit check).

**14.2 Multi-node 360M, 4 nodes × 8 = 32 GPUs, FSDP2.**
`d=1024 L=24 H=16/4 head=64 seq=2048 ffn=2816` (322M params), `GBS=256`, real wikitext, WSD,
`gradient_checkpointing`, launched with a 4-node torchrun fan-out (c10d rendezvous on the head node). **Result:
cross-node FSDP shards + trains**: CE **11.03 → 7.42** over 30 steps (step-0 11.03 ≈ `ln(50257)`,
i.e. correct random init; monotone decrease after), grad-norm 16.8 → ~0.7 (no spikes/NaN). The
**HF export gathered the full state dict across all 4 nodes** → an export dir (`config.json` + 644MB
`model.safetensors`) on NFS. Then **eval'd 4-node** on `val.bin` (`memmap_ppl`, `docs/eval_engine.md`):
**ppl 1443.6, nll/tok 7.275**: coherent with the step-25 train CE 7.42, so **train → HF-export →
multi-node-eval is closed on a model the engine actually produced** (high ppl only because it is a
30-step smoke run). One cosmetic note: a `RendezvousConnectionError`/`recvBytes` line prints *after*
"Training complete": the **benign c10d store-teardown race** (one rank closes the store before
others' final barrier); exit code 0, checkpoint + export already on disk. Rank 0 is assigned by the
rendezvous, **not** necessarily the launch/rdzv host: the rank-0 step logs land in whichever node's
launch log got rank 0, and the trainer's own file logger only writes the master's
process (the per-step dict, incl. `train/tokens_per_sec`, prints to that rank's stdout).

**14.2b The other architectures train cross-node.** Beyond GQA (above), a 4-node (32-GPU) FSDP2
smoke of the recurrent mixers at the 360M dims (`pdbs=4`, ckpt-on, 10 steps, real data) confirms each
shards across nodes, loss-decreases, and **HF-exports its mixer-specific params gathered across all 4
nodes**: **GDN** 11.03→8.50 (923MB safetensors), **Mamba2** 11.06→7.74 (835MB): so GQA / GDN / Mamba2 are all
multi-node-ready, not just single-node (§13). (`fla`/`mamba_ssm` JIT-compile Triton kernels on the
first step, so cold-start step-0 is slow; steady steps are not.)

**14.3 Throughput & MFU.** GQA 360M, seq 2048, bf16, H100 (SXM, 989.5 TFLOP/s bf16 dense peak),
SDPA(cuDNN), **uncompiled**. Model FLOPs/token = `6N + 12·L·d·T` = `6·322M + 12·24·1024·2048` ≈
**2.536 GFLOP/tok**; MFU = tok/s/GPU · 2.536e9 / 989.5e12. Steady-state (post-warmup) from the
trainer's `train/tokens_per_sec`:

| # | GPUs (nodes) | backend | pdbs | g.accum | ckpt | step s | tok/s agg | tok/s·GPU⁻¹ | MFU |
|---|---|---|---:|---:|---|---:|---:|---:|---:|
| A | 8 (1) | DDP   |  8 | 1 | off | 0.252 | 519K | 64.9K | 16.6% |
| B | 8 (1) | FSDP2 |  8 | 1 | off | 0.252 | 520K | 65.0K | 16.6% |
| C | 8 (1) | FSDP2 | 16 | 1 | off | 0.465 | 564K | 70.5K | 18.1% |
| D | 8 (1) | DDP   | 16 | 1 | off | n/a | **OOM** (alloc 6.13 GiB = fp32 logits) | n/a | n/a |
| E | 8 (1) | FSDP2 | 32 | 1 | off | n/a | **OOM** (same logits alloc) | n/a | n/a |
| F | 32 (4) | FSDP2 |  1 | 8 | **on** | 1.85 | 280K | 8.7K | 2.2% |
| G | 32 (4) | FSDP2 |  8 | 1 | off | 0.357 | **1.47M** | 45.9K | 11.8% |

**Findings.** (1) **DDP ≈ FSDP2 at equal batch** (A vs B: 519K vs 520K): at 360M on single-node
NVLink, FSDP's all-gather/reduce-scatter fully overlaps compute, so its sharding is "free"; choose
FSDP2 for the **memory headroom**, not speed. (2) **The un-fused cross-entropy is the memory wall**:
both D and E OOM on the *same* 6.13 GiB allocation: the materialized `[b, T, V]` fp32 logits
(`16·2048·50257·4 B`). It caps the no-ckpt batch (FSDP2 ≤16, DDP ≤8); **fused linear-CE is the
single highest-value memory optimization** (§11, the fused linear-CE bullet): it directly unlocks larger batches /
higher MFU. (3) **Multi-node: feed the GPUs.** The production-style `pdbs=1, g.accum=8, ckpt=on`
config (F) is **inter-node-comms-bound** at 2.2% MFU: each GPU does one 2048-token microbatch
between every per-layer FSDP all-gather, and those collectives cross the (sub-NVLink) node network.
Switching to `pdbs=8, g.accum=1, ckpt=off` (G) amortizes 8× more compute per collective (and drops
recompute) → **5.3× tok/s, 1.47M/s aggregate**. So for the launch: prefer **large per-device batch
with `grad_accum=1`** over tiny per-device batch with accumulation. (4) **Scaling cost** at 360M:
per-GPU throughput is 65K (1 node) → 46K (4 nodes) at the same `pdbs=8` = 0.71×: the price of
inter-node bandwidth for FSDP collectives at this small model; it improves with model size (more
FLOPs per communicated byte). (5) **MFU ~17–18%** single-node is on the expected low side for an
**uncompiled 360M** (d=1024 GEMMs under-fill H100 tensor cores; norm/RoPE/SiLU are memory-bound):
`torch.compile` (untested here) and the fused CE are the levers; larger models will sit higher.
**Practical launch budget:** at 1.47M tok/s (32 GPU), a 30B-token 360M run ≈ **5.7 h/architecture**.

**14.4 MFU vs context length (why it's low, and whether 32k "fixes" it).** Measured single-node
(8×H100, GQA 360M, bf16, SDPA-cuDNN, grad-ckpt ON, holding ~32768 tok/GPU/step so only seq-len
varies; a sequence-length sweep script, not shipped). FLOP/tok shown both as **full-T²** (`6N+12LdT`, the standard/PaLM
count) and **causal-corrected** (`6N+6LdT`, the lower-triangle FlashAttention actually computes):

| seq | tok/s·GPU⁻¹ | FLOP/tok full / causal | MFU (full-T²) | MFU (causal) |
|---:|---:|---:|---:|---:|
| 2048 | 59.6K | 2.54 / 2.23 G | 15.3% | 13.4% |
| 8192 | 50.9K | 4.35 / 3.14 G | 22.4% | 16.1% |
| 16384 | 42.7K | 6.76 / 4.35 G | 29.2% | 18.8% |
| 32768 | 31.3K | 11.6 / 6.76 G | **36.6%** | **21.4%** |

**Why low at 2k:** d=1024 GEMMs underfill H100 tensor cores (small M/N/K), memory-bound
norm/RoPE/SiLU/softmax are a big fraction at small d, uncompiled eager (no fusion), and the
efficient-at-scale attention is only ~24% of FLOPs (14% causal). **Why it climbs with seq-len:**
attention FLOPs grow O(L²) vs the small dense-proj GEMMs O(L), so the efficient flash kernel (~45–50%
on this stack) comes to dominate the mix (24%→83% of the full count, 2k→32k). **The catch:** most of
the climb toward "40%" is the **full-T² convention crediting masked FLOPs flash never runs**: the
honest causal-corrected MFU only reaches ~21% at 32k (the gap between the two columns *is* the masked
upper triangle). And it is **not cheaper training**: tok/s/GPU nearly halves (59.6K→31.3K) because
actual FLOPs/token ~triple (causal 2.23→6.76 G): you buy utilization by doing quadratically more
work per token. **Caveats:** (i) grad-ckpt here lowers all rows ~15% vs no-ckpt (the 2k no-ckpt point
was 18.1%, §14.3-C): so full-count >40% at 32k is reachable without ckpt / with `torch.compile`;
(ii) **only full-attention benefits**: Mamba2/GDN are O(L), so 32k gives them ~no MFU lift (3:1
hybrid gets ¼). **The real levers (any context):** `torch.compile` (fuse the memory-bound glue),
fused linear-CE (also mandatory at 32k for memory), bigger per-GPU batch, and: most fundamentally:
a **bigger model (d≥2048)** which fixes the small-GEMM problem at every context.
**Takeaway: pick context length for the science, not to chase an MFU number.**

**14.5 `torch.compile` + fused linear-CE (the two MFU levers, landed + tested).** Both are now
wired: `--compile true` regionally compiles each decoder block (`wrap.py::apply_compile`, torchtitan
idiom: AC→compile→shard); `--fused_ce true` swaps the training loss to a liger fused
linear-cross-entropy (`model.py::_fused_loss`) that computes CE (+z-loss, via liger's
`lse_square_scale`) straight from hidden states and the tied head: **no `[B,T,V]` logits**. The HF
eval path keeps the unfused logits (eval needs them); fused-CE is training-only.

- **Correctness** (`src/pretrain/scripts/test_compile_fusedce.py`, 8/8 PASS): fused == unfused to **fp32 machine
  precision** (|Δloss|=0, max-rel-Δgrad 6e-7; z-loss exact), bf16 noise ~1e-2 as expected;
  `torch.compile` == eager (|Δloss|~1e-5); the two combined == the plain reference.
- **Memory wall removed.** The configs that OOM'd on the 6.13 GiB logits (§14.3 D/E) now **run** with
  `--fused_ce`: DDP pdbs=16 and FSDP2 pdbs=32 (seq 2048, no-ckpt) both train to completion.
- **Throughput** (same sweep as §14.4, grad-ckpt ON, `--compile --fused_ce`):

  | seq | tok/s·GPU⁻¹ base → opt | speedup | MFU(full) base → opt |
  |---:|---:|---:|---:|
  | 2048 | 59.6K → **106.9K** | 1.79× | 15.3% → 27.4% |
  | 8192 | 50.9K → **80.5K** | 1.58× | 22.4% → 35.4% |
  | 16384 | 42.7K → **60.5K** | 1.42× | 29.2% → 41.4% |
  | 32768 | 31.3K → **39.9K** | 1.28× | 36.6% → 46.8% |

  The win is **largest at short context** (1.79× at 2k): exactly where the memory-bound glue (compile
  fuses it) and the logit materialization (fused-CE removes it) were the biggest share; at 32k
  attention already dominates so there's less to recover. **No-ckpt 2k** (fused-CE enables pdbs=32):
  **1.037M tok/s/node = 33% MFU** (vs 18% before). Note compile is most valuable on the **GQA** path;
  the GDN/Mamba2 mixers wrap custom Triton/CUDA kernels (graph-break, less compile benefit): verify
  per-mixer before relying on it for the recurrent arms.

**14.6 The bf16 ceiling + FP8 (measured): and what 3.1M tok/s/node really takes.** At seq 2048 the
model is 2.536 GFLOP/tok, so the **bf16 hardware ceiling** is `8·989.5e12/2.536e9 ≈ 3.12M tok/s/node`
= 100% MFU: unreachable (real kernels peak ~50–55%). Measured (GQA 360M, 2k, compile+fused, no-ckpt):

| config | tok/s/node | tok/s/GPU | MFU | note |
|---|---:|---:|---:|---|
| bf16, pdbs=32 | 1.037M | 129.6K | 33.2% | |
| **bf16, pdbs=48** | **1.076M** | **134.5K** | **34.5%** | **single-node bf16 ceiling** (pdbs≥56 OOMs; `max-autotune` OOMs on its autotune workspace) |
| FP8 tensorwise, pdbs=32 | 1.066M | 133.3K | n/a | **1.03× over bf16: no meaningful gain** |
| FP8 rowwise, pdbs=32 | 0.910M | 113.8K | n/a | **0.88×: slower than bf16** |
| **bf16, pdbs=48, 4 nodes (32 GPU)** | 1.01M | 126K | 32.3% | **4.03M tok/s aggregate** (small ~6% inter-node penalty: compile+fused amortize it well) |

**FP8 does not help at 360M (measured, not estimated).** torchao float8 correctly converts all 168
block Linears and is accuracy-neutral (bf16-identical loss curve), but the d=1024 GEMMs are too small
for FP8 tensor cores to win: the dynamic-scaling overhead (amax/scale/cast) cancels the GEMM gain;
rowwise is net negative. So **the single-node bf16 ceiling here is ~1.08M tok/s/node (34.5% MFU)** and
no precision lever moves it at this scale.

**The ~3.1M tok/s throughput target is MET (job-aggregate).** It was the whole-job (4-node / 32-GPU)
number; the compile+fused GQA config delivers **4.03M tok/s aggregate** (>3.1M). (Per *node* it would
mean 100% dense-bf16 MFU: unreachable at 360M; FP8 can't bridge it because the d=1024 GEMMs are too
small. Per-node 3.1M is a larger-model goal where FP8 gives its full ~2×.)

**Recommended 360M GQA launch config (bf16, FP8 off):** `--backend fsdp2 --compile true --fused_ce true
--gradient_checkpointing false --per_device_batch_size 48 --global_batch_size $((48*NGPU))` at seq
2048 → ~4.03M tok/s on 32 GPUs ⇒ **~2.1 h per 360M/30B-token GQA run**. Drop `pdbs` to 32 for memory
margin (1.04M/node), or raise grad-ckpt + `pdbs` for longer context. `torch.compile` mainly helps the
**GQA** arm; re-check the GDN/Mamba2 arms individually (custom kernels graph-break) and turn
`--compile` off for them if it doesn't help. Keep `--fused_ce true` for all arms (pure memory+speed
win, mixer-agnostic).

Reproduce a node-saturating multi-node run:
```bash
PYTHONPATH=src torchrun --nnodes=4 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=<head>:29541 -m pretrain.train --backend fsdp2 \
  --model.vocab_size 50257 --model.d_model 1024 --model.n_layers 24 --model.n_heads 16 \
  --model.n_kv_heads 4 --model.head_dim 64 --model.max_seq_len 2048 --model.mixer gqa \
  --model.attn_impl sdpa --global_batch_size 256 --per_device_batch_size 8 --num_steps 30 \
  --gradient_checkpointing false --data.paths <a flat uint16 token .bin> \
  --data.dtype uint16 --optim.lr 3e-4 --optim.lr_scheduler_type wsd --resume false --logging_steps 5 \
  --ckpt_root_dir <shared dir>/ckpts --log_root_dir <shared dir>/logs
```
(Single-node throughput: `torchrun --standalone --nproc_per_node=8`.)

## 15. Long-context (32k) launch validation + document-aware loader

> **The study trains at 8k** (the 64k runs are the stage-2 context extension, `scripts/campaign/stage2_arm.sh`).
> The 32k measurements in this section are retained as a long-context preview; the AC / fused-CE / compile / FP8
> findings apply at 8k qualitatively (8k ≈ ¼ the 32k footprint, so no-AC fits at pdbs≫1). The launch configs are
> `src/pretrain/configs/*.yaml` (`configs/README.md`). `qk_norm` (per-head RMSNorm on q,k before RoPE,
> `--model.qk_norm true`) is wired into `GQAAttention`; GDN/Mamba2 use their native stabilization.

### 15.0 8k throughput ablation: pdbs × AC (measured)

GQA 360M (intermediate size 3584, **8k**, vocab 32k, qk-norm, compile+fused, single-node 8×H100, ga=1):

| pdbs | AC off (tok/s · GB) | AC selective | AC full |
|---:|---:|---:|---:|
| 1 | 582K · 8.5 | n/a | 494K · 1.7 |
| 2 | 794K · 16 | n/a | n/a |
| 4 | 880K · 31 | n/a | n/a |
| 8 | **938K · 61** | **930K · 40** | 758K · 8 |
| 16 | OOM (~122) | (~80) | 784K · 15 |

**Findings.** (1) **AC off wins at every matched pdbs**: full-AC costs **15–19%** (recompute the never-stored
flash work; pdbs=1 582K→494K, pdbs=8 938K→758K), and full-AC never recovers it with a bigger batch
(full pdbs=16 784K < no-AC pdbs=8 938K). Same conclusion as 32k (§15.1). (2) **Unlike 32k, pdbs matters a
lot at 8k**: 8192 tok/GPU under-saturates, so pdbs 1→8 is **+61%** (582K→938K, **34%→54% MFU**;
flops/tok = 6N+12LdT ≈ 4.58 GFLOP). Diminishing past 8 (p4→p8 +7%); pdbs=16 OOMs. (3) **Selective AC is the
sweet spot**: pdbs=8 gives 930K at **40 GB vs no-AC's 61 GB**: ~the same speed, 35% less memory (recomputes
the cheap norm/RoPE/SiLU, keeps GEMM+attn outputs). **⇒ a micro-batch of 1 leaves 1.6× on the
table at 8k; launch at pdbs=8.** On 16 GPU, `pdbs=8, ga=1` gives the same GBS=128 (1M-token batch) at
**938K tok/s/node ≈ 1.76M on 2 nodes → ~4.7 h per architecture** (vs ~5–6h extrapolated at pdbs=1). Use **no-AC
pdbs=8** (max speed, 61 GB) or **selective-AC pdbs=8** (recommended: 930K at 40 GB, robust headroom).

Per-arch probe at the param-matched 360M dims (24L/d1024, vocab 32k, pdbs=1, single-node 8×H100;
**fused-CE mandatory** at 32k: it removes the 32768×vocab logit tensor). The table below is with
**full** activation checkpointing; **the §15.1 result is to turn full-AC OFF: it's ~1.25× faster** and
fits easily, so the *real* launch numbers are the no-AC column there:

| arch (param-matched ~358M) | `intermediate_size` | tok/s/node (full-AC) | peak mem/GPU | vs GQA |
|---|---:|---:|---:|---:|
| GQA | 3584 | 341K | 4.3 GB | 1.00× (≈51% MFU) |
| GDN-3:1 (`[gdn,gdn,gdn,gqa]×6`) | 2176 | 434K | 4.9 GB | **1.27×** |
| Mamba2-3:1 (`[mamba2,…,gqa]×6`) | 2560 | 469K | 4.6 GB | **1.38×** |

**Findings.** (1) **The recurrent hybrids are *faster* than GQA at 32k**: the O(L) vs O(L²) crossover
(at 2k attention won 3×; §12). At matched params the 18 recurrent layers + smaller MLP beat pure
attention. (2) **`compile` is a no-op for the hybrids** (GDN 434K with vs 433K without: fla's Triton
kernels graph-break) → **run hybrids with `--compile false`** (skip the compile wait), keep
`--compile true` for **GQA** only. `--fused_ce true` for all. (3) **Don't full-checkpoint at 32k**
(§15.1): with flash attn it recomputes work that was never stored, costing ~25%. (4) Budget: 30B
tokens at 32k ≈ 19 h/arch (GQA, 1 node, no-AC) / ~5 h (32 GPU): hybrids ~1.3× faster (≈14 h / node).

### 15.1 Activation checkpointing at 32k: turn (full) AC off

With flash attention the O(L²) score matrix is never stored, so **full** AC re-runs the attention + all
matmuls for nothing. Measured (GQA 32k, single-node, compile+fused):

| AC mode | pdbs | tok/s/node | peak GB | note |
|---|---:|---:|---:|---|
| full | 1 | 341K | 4.3 | recompute-bound: slowest |
| **selective** (`--ac_mode selective`) | 1 | **425K** | 20 | recompute only norm/RoPE/SiLU; **keep matmul+attn outputs** |
| **off** (`--gradient_checkpointing false`) | 1 | **426K** | 29 | fastest, simplest; fits with room |
| off | 2 | 438K | 57 | no-AC ceiling (pdbs≥3 OOMs) |
| full | 8 | 354K | 28 | |
| full | 16 | 355K | 56 | AC max batch |

**Findings.** (1) **Full-AC off → 1.25×** (341K→426K) and still only 29 GB of 80. (2) **`pdbs` barely
moves throughput at 32k**: it's compute-bound, so full-AC pdbs 1→16 gains +4% and no-AC pdbs 1→2 gains
+3%; **no-AC pdbs=1 (426K) beats full-AC at *any* batch (≤355K)**. So the recompute cost, not batch
size, is the lever. (3) **Selective AC** (the new `--ac_mode selective`, a torch SAC policy that saves
`aten.mm/addmm/bmm` + `_scaled_dot_product_*` and recomputes the rest) is *strictly better than no-AC
here*: same speed (425K), less memory (20 vs 29 GB): it drops the cheap norm/RoPE/SiLU activations.
Its memory still scales with batch (it saves all the GEMM outputs), so SAC pdbs=8 OOMs; it's the lever
when you need a batch bigger than no-AC allows. (4) **The hybrids also fit no-AC** at pdbs=1: GDN
**543K @ 47.5 GB**, Mamba2 **589K @ 39 GB** (≈1.25× their full-AC numbers): drop AC for them too.
**The +1.25× is uniform across archs.** ⇒ **Launch with `--gradient_checkpointing false`** (or
`--ac_mode selective` if a future config needs the memory headroom), pdbs=1.

**BOS-anchored windows (default off).** `--data.prepend_bos true` (with `--data.bos_token_id`, default 1 =
Llama-2 `<s>`) makes every chunk `[bos] + chunk[:-1]` in both loaders below: the same chunks, sampler order and length, so a
BOS sits at position 0 of every window and predicts the chunk's first token (the chunk's last token drops); in the
document-aware loader the BOS joins the first segment of `cu_seqlens` (boundaries shift right by one; `position_mode reset`
raises). Rank 0 prints `[data] prepend_bos ON` at startup. Tests: `tests/test_prepend_bos.py`. The Nemotron stream
already carries BOS / EOS
at every document start / end, but a mid-window BOS sees the previous document, so only position 0 gives the identical
key and value in every window.

**Document-aware loader (ProLong).** `--data.doc_lengths <sidecar.npy>` switches the loader to
`data/doc_dataset.py::DocAwareMemmapDataset`: same fixed-`chunk_size` chunking (resume unchanged) over
a flat `.bin` plus an int64 doc-length sidecar, emitting per-token **`position_ids`** (threaded through
trainer→model) and **`cu_seqlens`** (document boundaries within each chunk). `--data.position_mode`:
`packed` (continuous positions, cross-document long-context: ProLong's intent, default) or `reset`
(per-document positions, for use with intra-document masking). The study's two corpora come with their sidecars
(`data/README.md`). Verified 7/7 (`src/pretrain/scripts/test_doc_loader.py`): position reset, cu_seqlens,
collate, and a model forward that consumes the per-doc positions. **Remaining piece:** intra-document
*attention* masking at 32k needs flash-varlen (`cu_seqlens`): a dense [T,T] mask is 4 GB at T=32768,
so it can't be materialized; the loader already emits `cu_seqlens`, the mixer-side varlen wiring (and
per-document SSM state reset) is the follow-up. **Tokenizer:** use an off-the-shelf **32k** tokenizer
(Llama-2/Mistral: the GDN paper's choice). Llama-3 (128k) / Qwen3 (151k) put 36–43% of a 360M model
in the embedding; a 32k vocab keeps it at ~9% and preserves the param-match numbers above.

**Configs (`src/pretrain/configs/`).** The 360M two-stage recipe described in this section (an LR screen on a smaller
model, then a decision run) was the engine's first launch; the study's configs are the Qwen3.5-shaped 3:1 hybrids
(`qwen35_*.yaml`) and the Qwen3-shaped pure-attention models (`qwen3_*.yaml`), described in `configs/README.md`.
Run them through the launcher `scripts/campaign/train_arm.sh`, which adds the data paths, the tokenizer and the arm's
flags; the yamls' own `workspace/data/...` paths are placeholders. By hand, pass the same overrides (the README's
by-hand command): `PYTHONPATH=src torchrun --standalone --nproc_per_node=8 -m pretrain.train --config
src/pretrain/configs/qwen35_m.yaml --data.paths <train.bin> --data.doc_lengths <train.doc_lengths.npy>
--tokenizer_dir <tokenizer dir> ...`.

**Equivalent explicit per-arch launch (32k, ProLong, 32k vocab):**
```
# shared: --backend fsdp2 --model.d_model 1024 --model.n_layers 24 --model.n_heads 16 \
#   --model.n_kv_heads 4 --model.head_dim 64 --model.max_seq_len 32768 --model.vocab_size 32000 \
#   --model.z_loss_weight 1e-4 --global_batch_size 32 --num_steps 30000 --gradient_checkpointing false \
#   --fused_ce true --optim.* (AdamW .9/.95 wd .1 clip 1; WSD warmup .05 decay .2 linear) \
#   (32 GPU: pdbs=1 ga=1; 8 GPU: pdbs=1 ga=4. AC off: §15.1: or --gradient_checkpointing true --ac_mode selective for headroom)
#   --data.paths <train.bin> --data.doc_lengths <train.doc_lengths.npy> --data.dtype uint16
# GQA :     --model.mixer gqa --model.intermediate_size 3584 --compile true
# GDN-3:1 : --model.layer_mixers '[gated_deltanet,gated_deltanet,gated_deltanet,gqa]' --model.intermediate_size 2176 --compile false
# Mamba2  : --model.layer_mixers '[mamba2,mamba2,mamba2,gqa]'                          --model.intermediate_size 2560 --compile false
```

## 16. wandb logging and the result dump

**wandb** (off by default; `--wandb true` to enable; entity = the account's default unless `--wandb_entity`, project `delta-matching`). Config fields
(`config.py`): `wandb` / `wandb_entity` / `wandb_project` / `wandb_job_type` (a free-form grouping label; the launcher
passes `WANDB_JT`). `train.py::_maybe_init_wandb` (master-only, guarded: a wandb failure never stops training) inits
with `name=run_name, job_type, config=asdict(cfg)`; `--wandb true` overrides an inherited `WANDB_MODE=disabled` with
`online` (an explicit `offline` is kept). A relaunched run resumes its wandb run through the id stored in
`{log_root}/{run_name}.wandb_id`, and `run_name<TAB>run_id` is appended to `{log_root}/wandb_runs.txt`
(portalocker-locked) as an index.
`base_trainer.py::log_metrics` does `wandb_run.log(metrics, step)` each logging step: metrics carry
`train/*`: `ce_loss` (the training loss: cross-entropy plus the z-loss term), `grad_norm, step_time, tokens_per_sec,
learning_rate, valid_tokens, peak_mem_gb`.
On HF export the trainer writes `{hf_save_dir}/wandb_run_id.txt`, so an evaluation can log `eval/*` to the SAME run:
`memmap_ppl` does it only with `--wandb` (`memmap_ppl::_maybe_log_wandb`; `train_arm.sh` passes it to its end-of-run
val CE when the run logged to W&B), the recall evaluator whenever W&B is on (`eval_engine.md §3`; `eval_arm.sh` runs
with W&B off).
`utils/logging.py::tee_stdio` tees the master's stdout/stderr to `{log_root}/{run_name}.log`.

**Result dump** (always-on, master-only, try/except-guarded so a write failure never crashes the run).
`TrainConfig.result_root_dir: str | None = None` (CLI-settable). At end-of-training
(`base_trainer.py::_dump_result_summary`, called right after "Training complete") it writes
`<result_root_dir or dirname(log_root_dir)/result>/{run_name}/train_summary.json`: a curated, durable
per-baseline record: `{run_name, wandb_run_id (or None), global_step, num_steps, final_metrics (last
logged train metrics), hf_save_dir, ckpt_dir, config: dataclasses.asdict(cfg)}`. With `result_root_dir`
pointed at `workspace/result` (= `$DM_WORKSPACE/result`) and the eval driver's
`--output_dir` at `workspace/result/{run_name}`, each run's `train_summary.json` + `{run_name}_results.json`
co-locate under one durable dir, so a baseline never has to be re-run.

**Tokenizer baking into HF exports.** `TrainConfig.tokenizer_dir: str | None = None` (CLI). When
set, `base_trainer.save_hf` loads `AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)` on the
master and passes it to `save_hf_checkpoint(..., tokenizer=...)`, so every export carries the tokenizer.
**Required for eval:** `meta_eval`→`load_model_for_eval` always loads a tokenizer (even tokenizer-free
**recall** fails without one); only `memmap_ppl` is tokenizer-free. The launchers pass
`--tokenizer_dir $DM_TOKENIZER_DIR` (the 32k Llama-2 tokenizer of the data root).

**Token-budget ladder exports.** `TrainConfig.hf_export_steps: list[int]` (default `[]`). In the
train loop, when `global_step` hits a listed step (and `hf_save_dir` is set), the trainer does a collective
`save_hf(f"{hf_save_dir}/step{N}")`: a full HF snapshot (config + safetensors + baked tokenizer) mid-training,
so a training trajectory can be evaluated at 1.25/2.5/5/10/20B tokens. The
final export still goes to `{hf_save_dir}/` at the end. `qwen35_m.yaml` bakes
`hf_export_steps: [1192, 2384, 4768, 9537, 19073]` (= 1.25/2.5/5/10/20B at 1.048576M tok/step; tied to
`num_steps: 28610`); the campaign launcher overrides it with `[]` for stage 1 and with an export every
`S2_EXPORT_EVERY` steps for stage 2, so the yaml's milestones apply to direct runs only. Smoke-validated (4-step run,
export at step 2 → `hf/step2/` full + final). Each snapshot is a collective DTensor gather (~1.4GB fp32 on master at
360M, ~seconds): negligible over a run.
