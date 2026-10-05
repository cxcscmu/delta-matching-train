# `src/pretrain/configs`: the study's model configs

Every yaml is self-contained (`--config FILE`, then any `--section.key value` override wins). The launcher
(`scripts/campaign/arms.sh` + `train_arm.sh`, driven by the experiment scripts in `scripts/experiments/`) runs all seven
of them and adds the attention geometry, the mixer list, the numerics flags and the data paths on the command line, so
for a launched run the yaml supplies the architecture, the schedule and the budget. Two data facts are shared by all:
the Nemotron-CC-v2.1 HQ+DQA corpus in Llama-2 32k ids (naive packing, `position_mode: packed`, BOS/EOS in-stream;
`data/pretrain_stage1.md`) and `data.seed 11785` (the data order; the `--seed` argument changes the initialization only,
and every arm of a cell starts from the same seed).

| file | model | params | attention | recurrent layers | budget |
|---|---|---|---|---|---|
| `qwen35_m.yaml` | the Qwen3.5-2B-Base block, 24 layers, d 2048, 3:1 GatedDeltaNet : GQA | 1.665 B | 8q/2kv head_dim 256 in the yaml (the head_dim-256 arms); the head_dim-128 arms override to 16q/4kv head_dim 128 (`GEOM128` in arms.sh) | GatedDeltaNet 16 heads x 128 | 128 x 8192 tokens per step, 28,610 steps = 30.0 B tokens, lr 2.4e-3, WSD |
| `qwen35_xs.yaml`, `qwen35_l.yaml` | the head_dim-256 scale ladder around qwen35_m | 0.569 B / 5.289 B | 6q/2kv and 16q/4kv at head_dim 256 | GatedDeltaNet (the large cell widens V) | as qwen35_m |
| `qwen35_m_mla.yaml` | the GDN : MLA twin | 1.665 B | Multi-head Latent Attention, 16 heads, [rope 64 &#124; nope 64] q/k, v 128, latent 384, MHA on the wire | GatedDeltaNet | as qwen35_m |
| `qwen35_m_mamba2.yaml` | the Mamba2 : GQA twin | 1.666 B | 16q/4kv head_dim 128 | Mamba2 (d_inner 4096, headdim 64, d_state 128); FFN 7104 for parameter parity | as qwen35_m |
| `qwen35_m_kda.yaml` | the KDA : GQA twin | 1.665 B | 16q/4kv head_dim 128 | Kimi Delta Attention (16 heads x 128, expand_v 1); FFN 8064 for parity | as qwen35_m |
| `qwen3_m.yaml` | Qwen3-1.7B exactly (pure attention, no output gate, full rotary) with the 32k vocabulary | 1.475 B | 28 layers of 16q/8kv head_dim 128 | none | 30.0 B tokens, lr 2.4e-3 |

Head dim matters beyond fidelity: the fp8 attention kernel's block-scale contract is [128, 64] (Q/dO rows, K/V keys per
scale) at head_dim 128 and [64, 32] at head_dim 256, so the two head dims are two kernel numerics; every arm of a cell
shares the head dim.

## Knobs

Set in every yaml:

- `qk_norm: true` on every attention layer (per-head RMSNorm on q and k before RoPE; the four sub-vector norms in MLA);
  the recurrent mixers use their native stabilization.
- `fused_ce: true` (liger fused linear cross-entropy: no `[B, T, V]` logits), bf16, `compile: false`, `fp8: false`, and
  selective activation checkpointing (`gradient_checkpointing: true`, `ac_mode: selective`).
- WSD schedule (linear warmup, constant, linear decay), z-loss 1e-4, AdamW (0.9, 0.95), weight decay 0.1, clip 1.0.
- `hf_export_steps` snapshots at token-budget milestones (1.25 / 2.5 / 5 / 10 / 20 B tokens) in `qwen35_m.yaml`; the
  other yamls export at the end only (the launcher overrides both, see below).

Added by the launcher (`COMMON` in `scripts/campaign/arms.sh`; a direct `--config` run does not get them):
`--gradient_checkpointing false` at 8k (recompute is wasteful with flash-class attention), `--model.fused_rmsnorm true`,
`--model.fused_swiglu false`, `--fp8 true` and compiled MLPs (`--compile_mlp true`). The 64k context extension
(`stage2_arm.sh`) turns `--gradient_checkpointing true` back on, with the yaml's `ac_mode: selective`. The launcher also
sets `--hf_export_steps` itself, so the yaml's milestones apply to direct runs only.

## Precision is not in the yaml

`--fp8 true --fp8_scope ffn` (the Standard arm) or `ffn_qkvo` (the fp8 arms), `--model.dm_mode stale|matched`,
`--model.layer_mixers` with `dm` / `mla_dm` in place of `gqa` / `mla` (or `--model.mixer dm` for pure GQA), and
`--model.attn_impl sdpa|flash|te_fp8` for the other attention arms come from `scripts/campaign/arms.sh`, as do the
two variants of the head_dim-128 cell (`--model.nope true`, `--optim.name muon`). The fp8 arms reach the
Delta-Matching kernel through the `dm` mixer (`mla_dm` for the MLA arms); `attn_impl: dm` (the plain kernel route) is
not used by the launcher. `tests/configs/tiny_*.yaml` are the small twins the test suite trains for three steps.
