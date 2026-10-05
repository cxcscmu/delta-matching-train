"""Model wrapping (DDP / FSDP2), optimizer, and LR schedule.

FSDP2: one `fully_shard` per decoder block plus a root wrap,
MixedPrecisionPolicy(param=bf16, reduce=fp32), and reshard_after_forward=True on every
block except the last (which is about to be used in backward, so keeping it gathered
saves an all-gather). No DeviceMesh => 1-D data-parallel FSDP over the default process
group.
"""
from __future__ import annotations

import os
import re

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from transformers import get_constant_schedule_with_warmup, get_wsd_schedule

from .dist import get_world_size, get_local_rank


# ----------------------------- activation checkpointing -----------------------
# Ops whose outputs are SAVED under selective AC (everything else, i.e. norm, RoPE, SiLU and
# adds, is recomputed in backward). Keeping the GEMMs *and the flash-attention output* avoids the
# only expensive recomputes: with flash attn the O(L²) score matrix is never stored anyway,
# so recomputing the whole block (full AC) needlessly re-runs the attention + all matmuls.
def _sac_save_ops():
    a = torch.ops.aten
    ops = [a.mm.default, a.addmm.default, a.bmm.default]
    for name in ("_scaled_dot_product_flash_attention", "_scaled_dot_product_cudnn_attention",
                 "_scaled_dot_product_efficient_attention"):
        op = getattr(a, name, None)
        if op is not None:
            ops.append(op.default)
    # Save the Delta-Matching fp8-attention FORWARD, exactly like the aten SDPA variants above. Without this the
    # custom op is PREFER_RECOMPUTE, so under ac_mode=selective every checkpointed recompute re-runs it (two forwards
    # per layer per step instead of one). torch.ops.delta_matching.fwd_fp8 is a TORCH_LIBRARY dispatcher op of the
    # delta_matching package, so _CachingTorchDispatchMode sees it and MUST_SAVE stashes its (O, Lse) outputs; the
    # recompute pass then fetches them instead of re-launching the kernel. SEMANTICS under non-reentrant AC: the
    # recompute still re-runs the autograd Function, so the producer Triton kernels (NOT dispatcher ops) re-run and
    # ctx.save_for_backward gets Q8/K8/V8/kTt from the RECOMPUTED quantizers while O/Lse come from the stash. Both are
    # bitwise-reproducible on identical inputs (fwd_fp8 and the quantizers are deterministic), so training numerics
    # are EXACTLY unchanged. Covers BOTH routes: attn_impl='dm' and the dm mixer (same dispatcher op in both graphs).
    # ADDITIVE + INERT: arms that never load the kernel have no delta_matching namespace entry, the lookup raises
    # AttributeError, and the save set is the plain one. Opt out with DM_SAC_SAVE_DM_FWD=0.
    if os.environ.get("DM_SAC_SAVE_DM_FWD", "1") != "0":
        try:
            ops.append(torch.ops.delta_matching.fwd_fp8.default)
        except (AttributeError, RuntimeError):
            pass  # .so not loaded => this arm has no delta-matching op in its graph; set unchanged
    # OPT-IN (default OFF): the FA3 wheel's forward (torch.ops.flash_attn_3._flash_attn_forward, used by
    # attn_impl='flash' + DM_FLASH_IMPL=fa3) has the identical recompute pathology; DM_SAC_SAVE_FA3WHEEL=1 saves it
    # (the head_dim-256 clean arm exports it).
    if os.environ.get("DM_SAC_SAVE_FA3WHEEL", "0") == "1":
        try:
            ops.append(torch.ops.flash_attn_3._flash_attn_forward.default)
        except (AttributeError, RuntimeError):
            pass
    return set(ops)


def _selective_context():
    from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy
    save = _sac_save_ops()

    def policy(ctx, op, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE if op in save else CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy)


# Stage 2 (context extension), OPT-IN (default OFF): DM_AC_SAVE_ATTN=1 keeps ac_mode=full's whole-block recompute but
# MUST_SAVEs the attention forward's outputs (the aten SDPA variants, the delta_matching fwd_fp8 op, the FA3 wheel
# forward), so the one expensive-to-recompute / cheap-to-store tensor of the block (O + LSE, ~0.27 GB per attention layer
# per 64k microbatch) is fetched from the stash in the backward instead of re-running the O(N^2) kernel. Everything else
# (GEMMs, GDN, norms, the compiled MLP) recomputes exactly as under full. WHY: at 65,536 tokens the attention forward is
# ~7 % of the step and full recompute runs it twice. NUMERICS: the Delta-Matching forward is deterministic, so stash-fetch
# vs recompute is bitwise for the fp8 arms; cuDNN's forward is fetched instead of re-run (its recompute was never
# guaranteed bitwise). Realized as the selective-AC context with an attention-only save set.
def _attn_only_save_ops():
    a = torch.ops.aten
    ops = []
    for name in ("_scaled_dot_product_flash_attention", "_scaled_dot_product_cudnn_attention",
                 "_scaled_dot_product_efficient_attention"):
        op = getattr(a, name, None)
        if op is not None:
            ops.append(op.default)
    for ns, name in (("delta_matching", "fwd_fp8"), ("flash_attn_3", "_flash_attn_forward")):
        try:
            ops.append(getattr(getattr(torch.ops, ns), name).default)
        except (AttributeError, RuntimeError):
            pass  # that kernel library is not loaded in this arm
    return set(ops)


def _attn_only_context():
    from torch.utils.checkpoint import create_selective_checkpoint_contexts, CheckpointPolicy
    save = _attn_only_save_ops()

    def policy(ctx, op, *args, **kwargs):
        return CheckpointPolicy.MUST_SAVE if op in save else CheckpointPolicy.PREFER_RECOMPUTE

    return create_selective_checkpoint_contexts(policy)


def apply_activation_checkpointing(model: nn.Module, mode: str = "full") -> None:
    """Replace each decoder block in-place with a (non-reentrant) checkpoint wrapper.
    Must run before sharding so the FSDP unit wraps the checkpointed block.
    `mode="full"` recomputes the whole block; `mode="selective"` saves matmul/attention
    outputs and recomputes only the cheap ops (lower recompute cost, more memory than full)."""
    if hasattr(model, "config"):
        setattr(model.config, "use_cache", False)  # no KV cache during training
    kw = {"preserve_rng_state": True}
    if mode == "selective":
        kw["context_fn"] = _selective_context
    elif mode == "full":
        if os.environ.get("DM_AC_SAVE_ATTN", "0") == "1":
            kw["context_fn"] = _attn_only_context
            print("[wrap] DM_AC_SAVE_ATTN=1: full recompute except the attention forward "
                  f"(saved ops: {sorted(str(o) for o in _attn_only_save_ops())})", flush=True)
    else:
        raise ValueError(f"ac mode must be 'full' or 'selective', got {mode!r}")
    layers = model.model.layers
    # OPT-IN (default OFF = the whole-block wrapper below):
    # DM_AC_SCOPE=attn checkpoints ONLY the attention half of each decoder block
    # (x + mixer(input_layernorm(x))) and runs the MLP half OUTSIDE any checkpoint. WHY: a
    # torch.compile'd region is opaque to selective AC (torch's _CachingTorchDispatchMode sets
    # ignore_compile_internals, so the MUST_SAVE policy never sees the aten.mm/_scaled_mm inside
    # the compiled MLP), so the whole compiled MLP forward (three fp8 GEMMs plus every
    # amax/cast pass) re-runs inside every checkpoint recompute (measured at 1.7 B: 1152
    # instead of 576 forward GEMM calls per step). Moving the MLP out of the
    # checkpointed region lets its AOT graph keep its own saved activations, so the forward
    # GEMMs and casts run exactly once. Costs the MLP's saved activations back (~0.7 GB per
    # layer per microbatch at d2560/ff8192).
    # Realized as an INSTANCE forward override (no wrapper module, so parameter FQNs, the
    # FSDP2 per-block sharding and the DCP state-dict keys are untouched; same idiom as
    # apply_compile_mlp). Numerics: the recompute pass replays deterministic kernels, so
    # saved-vs-recomputed is bitwise for the MLP.
    scope = os.environ.get("DM_AC_SCOPE", "block")
    if scope == "attn":
        n = 0
        for layer in layers:
            if hasattr(layer, "mixer") and hasattr(layer, "mlp") and hasattr(layer, "input_layernorm"):
                _install_attn_half_checkpoint(layer, kw)
                n += 1
        print(f"[wrap] DM_AC_SCOPE=attn: {n} block(s) checkpoint the attention half only "
              f"(mode={mode}); the MLP half runs outside the checkpoint (no recompute)")
        return
    if scope != "block":
        raise ValueError(f"DM_AC_SCOPE must be 'block' (default) or 'attn', got {scope!r}")
    # The te_fp8 arm must be checkpointed by TE's OWN checkpoint. torch's
    # non-reentrant recompute runs the block forward outside the fp8 autocast region (the trainer
    # enters it once per micro-step around the whole model forward), so the TE core is asked to run
    # in bf16 during the recompute: modeling/te_attn.py::_require_region refuses (and if it did not,
    # the recomputed activations would differ from the saved ones and the gradients would be wrong).
    # transformer_engine.pytorch.distributed.checkpoint(use_reentrant=False) captures
    # FP8GlobalStateManager's enabled flag and recipe at forward time and re-enters that autocast in
    # its recompute_fn, together with TE's activation-recompute contexts, which tell the modules to
    # reuse the forward's fp8 metadata instead of updating the amax history a second time.
    if any(getattr(getattr(l, "mixer", None), "_te_core", None) is not None for l in layers):
        for layer in layers:
            _install_te_block_checkpoint(layer, kw)
        print(f"[wrap] te_fp8: {len(layers)} block(s) checkpointed by transformer_engine.pytorch.distributed.checkpoint "
              f"(use_reentrant=False, mode={mode}): the fp8 autocast region is restored in the recompute", flush=True)
        return
    for i in range(len(layers)):
        layers[i] = checkpoint_wrapper(layers[i], **kw)


def _install_te_block_checkpoint(layer: nn.Module, kw: dict) -> None:
    """Checkpoint one decoder block with TE's checkpoint (see the note in apply_activation_checkpointing).
    Instance forward override, as _install_attn_half_checkpoint: parameter FQNs, the FSDP2 per-block
    sharding and the DCP state-dict keys are untouched."""
    import types
    from transformer_engine.pytorch.distributed import checkpoint as te_checkpoint
    inner = layer.forward                      # bound class forward, captured before the override
    ctx_fn = kw.get("context_fn")

    def forward(self, x, cos, sin, attention_mask=None):
        assert attention_mask is None, "te_fp8 blocks use packed windows (no padding mask)"
        extra = {"context_fn": ctx_fn} if ctx_fn is not None else {}
        return te_checkpoint(self._te_block_inner, x, cos, sin, use_reentrant=False, **extra)

    layer._te_block_inner = inner
    layer.forward = types.MethodType(forward, layer)


def _install_attn_half_checkpoint(layer: nn.Module, kw: dict) -> None:
    """Replace `layer.forward` (instance attribute) with a version that runs
    x + mixer(input_layernorm(x)) under a non-reentrant torch.utils.checkpoint (with the
    selective policy when given) and the MLP half eagerly (DM_AC_SCOPE=attn). Mirrors
    DecoderBlock.forward (modeling/block.py) op for op, including the resid_drop placement."""
    import types
    from torch.utils.checkpoint import checkpoint
    ckpt_kw = {"use_reentrant": False, "preserve_rng_state": kw.get("preserve_rng_state", True)}
    if kw.get("context_fn") is not None:
        ckpt_kw["context_fn"] = kw["context_fn"]

    def forward(self, x, cos, sin, attention_mask=None):
        if hasattr(self, "_fp8glue_enter"):
            self._fp8glue_enter()          # no-op unless the fp8 cast glue runs delayed scaling

        def attn_half(x, cos, sin, attention_mask):
            return x + self.resid_drop(self.mixer(self.input_layernorm(x), cos, sin, attention_mask))
        x = checkpoint(attn_half, x, cos, sin, attention_mask, **ckpt_kw)
        return x + self.resid_drop(self.mlp(self.post_attention_layernorm(x)))

    layer.forward = types.MethodType(forward, layer)


# ----------------------------- torch.compile (regional) -----------------------
def apply_compile(model: nn.Module, mode: str = "default") -> None:
    """Regional (per-block) torch.compile: replace each decoder block in place with a
    compiled version (the torchtitan idiom). Compiling per block (rather than the whole
    model) keeps compile units small, composes with FSDP2's all-gather hooks, and avoids
    recompiles on the (variable) loss path. The fused/unfused CE head and embeddings stay
    eager. Default mode (graph breaks allowed) for robustness with activation checkpointing.
    `mode="max-autotune-no-cudagraphs"` autotunes the GEMMs (cudagraphs off: they clash with
    FSDP2's all-gather). Applied AFTER activation checkpointing and BEFORE sharding."""
    kw = {} if mode in ("default", "", None) else {"mode": mode}
    layers = model.model.layers
    for i in range(len(layers)):
        layers[i] = torch.compile(layers[i], **kw)


# ----------------------------- FP8 (torchao float8) ---------------------------
def _is_tensorwise(recipe) -> bool:
    return str(getattr(recipe, "value", recipe)) == "tensorwise"   # a recipe name or torchao's Float8LinearRecipeName


def fp8_fsdp_all_gather_enabled(recipe) -> bool:
    """Whether apply_fp8(..., fsdp=True) turns on torchao's FSDP2 fp8 weight all-gather: DM_FP8_FSDP_AG=1 (the
    default; 0 opts out) and the tensorwise recipe (torchao supports the fp8 all-gather for tensorwise weight scaling
    only). The trainer's once-per-step weight-scale precompute (DM_FP8_PRECOMPUTE) runs only when this holds."""
    return os.environ.get("DM_FP8_FSDP_AG", "1") == "1" and _is_tensorwise(recipe)


def apply_fp8(model: nn.Module, recipe: str = "tensorwise", scope: str = "layers", *, fsdp: bool = False) -> int:
    """Swap nn.Linear layers under `model.layers.*` to torchao Float8Linear (FP8 compute on H100
    tensor cores, ~2× the bf16 GEMM ceiling). `scope` picks which: "layers" = every Linear under
    model.layers (fp8 GEMMs need both dims divisible by 16, which GatedDeltaNet's num_heads-wide
    projections can break: use "safe" there), "safe" = the fp8-legal ones, "ffn" = the SwiGLU MLP,
    "ffn_qkvo" = the MLP plus the full-attention q/k/v/o projections. The tied `lm_head` (vocab
    not /16, and routed through fused-CE) and the embedding stay bf16. Apply BEFORE torch.compile +
    FSDP2 (torchtitan order); `fsdp=True` (build_fsdp2_model) allows the fp8 weight all-gather
    below. Returns #converted. Eager fp8 is slower than bf16: compile the converted Linears
    (`--compile_mlp true`: apply_compile_mlp and apply_compile_fp8_linears)."""
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig
    config = Float8LinearConfig.from_recipe_name(recipe)
    # DM_FP8_FSDP_AG=1 (the default, the campaign's setting; 0 opts out) turns on torchao's FSDP2 fp8 weight
    # all-gather, under FSDP2 with the tensorwise recipe only (DDP has no all-gather, and torchao asserts
    # tensorwise weight scaling). Effects: (a) the weight all-gather moves e4m3
    # codes instead of bf16 (half the weight-AG bytes for every converted Linear); (b) the
    # per-microbatch weight amax+cast disappears from the compute stream (measured at 1.7 B:
    # 13.8 ms amax + 18.1 ms cast per step) because fsdp_pre_all_gather casts
    # the SHARD once per all-gather and the matmul sees an already-cast Float8TrainingTensor.
    # Weight-code semantics change from "cast the gathered bf16 weight per matmul" to "cast
    # the shard at all-gather time with a tensor-global scale": with DM_FP8_PRECOMPUTE=1
    # (base_trainer.py, also the default) the scale comes from one batched foreach-amax + ONE
    # all-reduce per optimizer step (weights are static within a step); without it torchao
    # all-reduces an amax inside EVERY pre-all-gather hook (slower). Always run both together:
    # the all-gather alone regresses.
    if fsdp and fp8_fsdp_all_gather_enabled(recipe):
        import dataclasses
        config = dataclasses.replace(config, enable_fsdp_float8_all_gather=True)
        print("[fp8] FSDP2 fp8 weight all-gather ON (DM_FP8_FSDP_AG=1; "
              "paired with DM_FP8_PRECOMPUTE=1 for the once-per-step scale)")
    elif fsdp and os.environ.get("DM_FP8_FSDP_AG", "1") == "1":
        print(f"[fp8] recipe {recipe}: FSDP2 fp8 weight all-gather OFF (torchao supports it for tensorwise scaling only)")
    n = [0]
    skipped: list = []

    FFN = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")

    def _filter(mod: nn.Module, fqn: str) -> bool:
        if not (isinstance(mod, nn.Linear) and fqn.startswith("model.layers")):
            return False
        if getattr(mod, "_fusedprod_owned", False):
            # a mixer may mark a Linear whose GEMM it runs itself (no such mixer ships in this repository: the dm
            # mixer hands v_proj and o_proj to torchao); the marker is honoured so a port can exclude one cleanly
            skipped.append((fqn + " [mixer-owned]", mod.in_features, mod.out_features))
            return False
        if scope == "ffn":
            # SwiGLU MLP only: keeps an fp8-FFN arm from also quantizing attention (which the
            # dm mixer's fp8 kernel owns) and confounding the two mechanisms.
            keep = fqn.endswith(FFN)
        elif scope == "ffn_qkvo":
            # FFN plus the FULL-ATTENTION module's
            # q/k/v/o projections, and NOTHING else. The GDN (linear-attention) mixers are
            # deliberately excluded: fla nests their linears under ".gdn." (and fla's OWN
            # submodules are also named q_proj/k_proj/..., so the exclusion must be by that
            # nesting, not by projection name). This isolates "fp8 in and around the full
            # attention layer" from "fp8 in the recurrent path", which stays a separate ablation.
            keep = fqn.endswith(FFN) or (
                ".mixer." in fqn and ".gdn." not in fqn and ".kda." not in fqn   # .kda.: the KDA mixer (2026-09-22), excluded like GDN
                and fqn.endswith(("q_proj", "k_proj", "v_proj", "o_proj", "kv_a_proj")))   # kv_a_proj: the MLA latent projection (2026-09-18)
        elif scope == "safe":
            # EVERY layer Linear whose dims are fp8-legal. scope="layers" fails on a GatedDeltaNet
            # with 8 heads: fla's GatedDeltaNet carries a Linear with out_features = num_heads, and
            # fp8 GEMM requires both dims divisible by 16 ("RuntimeError: mat2 shape (1024x8) must
            # be divisible by 16"). Skipping just those tiny projections converts ~89% of linear
            # FLOPs instead of the MLP's ~57%.
            keep = (mod.in_features % 16 == 0) and (mod.out_features % 16 == 0)
            if not keep:
                skipped.append((fqn, mod.in_features, mod.out_features))
        else:
            keep = True
        n[0] += int(keep)
        return keep

    convert_to_float8_training(model, config=config, module_filter_fn=_filter)
    if skipped:
        uniq = sorted({(a.split(".")[-1], b, c) for a, b, c in skipped})
        print(f"[fp8] scope={scope} skipped {len(skipped)} Linear(s) "
              f"(fp8-illegal or mixer-owned): {uniq[:6]}")
    return n[0]


def apply_compile_mlp(model: nn.Module, mode: str = "default") -> int:
    """torch.compile ONLY the SwiGLU MLP submodules, leaving everything else eager.

    WHY THIS EXISTS. torchao fp8 needs torch.compile to be worth anything: measured on the real
    M-tier MLP shape (16384 tokens, 2048->6144, fwd+bwd) EAGER fp8 is 19.89 ms against bf16's
    6.50 ms (**3x SLOWER**), while COMPILED fp8 is 4.879 ms, i.e. 1.33x FASTER than eager
    bf16 and 1.30x faster than compiled bf16. Without compile, quantizing the FFN is a large
    net loss. But whole-model compile is unavailable here: the fla GatedDeltaNet kernels
    graph-break (which is why every config runs `compile: false`).
    The MLP contains no fla kernels, so compiling it in isolation captures the fp8 win without
    ever asking Dynamo to trace a linear-attention kernel. Applied per-module so a graph break
    anywhere else cannot disturb it.
    Must run AFTER apply_fp8 (so Float8Linear is what gets traced) and BEFORE fully_shard,
    matching torchtitan's ordering.

    IMPORTANT: compile the FORWARD METHOD IN PLACE rather than replacing the submodule.
    `layer.mlp = torch.compile(mlp)` wraps it in an OptimizedModule whose child is `_orig_mod`,
    which renames every parameter FQN under it and breaks downstream traversal
    ("AssertionError: Expected '_orig_mod', got 'gate_proj'" from the FSDP2/AC FQN walk).
    Compiling `mlp.forward` leaves the module tree, parameter names, AC wrapping and FSDP
    sharding completely untouched.
    """
    n = 0
    skipped = 0
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and getattr(mlp, "_fp8glue", None) is not None:
            # the fp8-glue MLP is Triton kernels + torch._scaled_mm inside an autograd
            # Function; there is nothing for Inductor to fuse, so
            # --compile_mlp true is a no-op for it (the flag keeps working on every other MLP).
            skipped += 1
            continue
        if mlp is not None and not getattr(mlp, "_compiled_fwd", False):
            mlp.forward = torch.compile(mlp.forward, mode=mode)
            mlp._compiled_fwd = True
            n += 1
    if skipped:
        print(f"[wrap] fp8glue: {skipped} MLP(s) left uncompiled (ported path; compile is a no-op)")
    return n


def apply_compile_fp8_linears(model: nn.Module, mode: str = "default") -> int:
    """Compile every Float8Linear that is NOT already inside a compiled region.

    torchao's fp8 Linear is only a win under torch.compile (eager fp8 measured 3x SLOWER than
    bf16 on the real MLP shape). `apply_compile_mlp` covers the SwiGLU MLP, but with
    `fp8_scope=safe` the GDN and attention projections are ALSO Float8Linear and, left eager,
    they turn a win into a large loss (measured: scope=safe + compile_mlp was 8.70 s/it against
    bf16's 4.90). Compiling each such Linear INDIVIDUALLY captures the fp8 fusion without ever
    asking Dynamo to trace the surrounding fla GatedDeltaNet kernels, which graph-break.
    """
    from torchao.float8.float8_linear import Float8Linear
    # Every Float8Linear compiled here shares ONE dynamo cache (the code object
    # Float8Linear.forward), one entry per distinct weight shape (the guard is on the parameter's shape). Past
    # torch._dynamo.config.recompile_limit (8) dynamo silently falls back to EAGER for the remaining shapes; without
    # activation checkpointing that only costs speed, but under checkpointing the recompute's visible op sequence then
    # differs from the forward's (an eager mm is seen by the selective policy, a compiled one is not), which raises
    # "aten.mm.default encountered during backward, but not found in storage" (selective) or "a different number of
    # tensors was saved" (full). The GDN:MLA hybrid's five projection shapes plus the MLP shapes cross the limit (the
    # GDN:GQA hybrid's four sit below it). A dynamo guard on the global thread count (num_threads changes between the
    # forward and the recompute) doubles the entries: every shape recompiles once inside the recompute. The limit is
    # raised PER COMPILED CALLABLE (torch.compile(..., recompile_limit=)): the global config value is not enough because
    # torch.utils.checkpoint patches the dynamo config around the recompute (the global 64 read back as 8 there, measured).
    # DM_DYNAMO_RECOMPILE_LIMIT overrides the 64.
    import torch._dynamo
    lim = int(os.environ.get("DM_DYNAMO_RECOMPILE_LIMIT", "64"))
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name) and getattr(torch._dynamo.config, name) < lim:
            setattr(torch._dynamo.config, name, lim)
    import inspect
    per_callable = {"recompile_limit": lim} if "recompile_limit" in inspect.signature(torch.compile).parameters else {}
    n = 0
    for _, mod in model.named_modules():
        if isinstance(mod, Float8Linear) and not getattr(mod, "_compiled_fwd", False):
            mod.forward = torch.compile(mod.forward, mode=mode, **per_callable)
            mod._compiled_fwd = True
            n += 1
    return n


# ----------------------------- DDP --------------------------------------------
def build_ddp_model(model: nn.Module, device: torch.device, *, activation_checkpointing: bool = False,
                    ac_mode: str = "full", compile: bool = False, compile_mode: str = "default",
                    fp8: bool = False, fp8_recipe: str = "tensorwise", fp8_scope: str = "layers",
                    find_unused_parameters: bool = False) -> nn.Module:
    if fp8:
        apply_fp8(model, fp8_recipe, fp8_scope)       # no fp8 all-gather: DDP replicates the weights
    if activation_checkpointing:
        apply_activation_checkpointing(model, ac_mode)
    if compile:
        apply_compile(model, compile_mode)
    model = model.to(device)
    if get_world_size() > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[get_local_rank()] if device.type == "cuda" else None,
            output_device=get_local_rank() if device.type == "cuda" else None,
            find_unused_parameters=find_unused_parameters,
            gradient_as_bucket_view=True,
        )
    return model


# ----------------------------- FSDP2 ------------------------------------------
def build_fsdp2_model(model: nn.Module, device: torch.device, *, activation_checkpointing: bool = False,
                      ac_mode: str = "full", compile: bool = False, compile_mode: str = "default",
                      fp8: bool = False, fp8_recipe: str = "tensorwise", fp8_scope: str = "layers",
                      compile_mlp: bool = False, param_dtype: torch.dtype = torch.bfloat16,
                      reduce_dtype: torch.dtype = torch.float32) -> nn.Module:
    model = model.to(device)
    if fp8:
        apply_fp8(model, fp8_recipe, fp8_scope, fsdp=True)
        # DM_FP8GLUE=1 (the default; 0 opts out) wires the fp8 cast-glue path (modeling/mlp_fp8glue.py) into the
        # blocks whose MLP linears torchao just converted. The glue implements tensorwise scaling only, so under
        # another recipe the MLPs keep torchao's own casts.
        if os.environ.get("DM_FP8GLUE", "1") == "1":
            if _is_tensorwise(fp8_recipe):
                from ..modeling.mlp_fp8glue import install_fp8glue
                install_fp8glue(model)
            else:
                print(f"[fp8glue] not installed: the fp8 cast glue implements tensorwise scaling only (recipe {fp8_recipe})")
    if activation_checkpointing:
        apply_activation_checkpointing(model, ac_mode)
    if compile:
        apply_compile(model, compile_mode)
    elif compile_mlp:
        n = apply_compile_mlp(model, compile_mode)
        m = apply_compile_fp8_linears(model, compile_mode) if fp8 else 0
        print(f"[wrap] torch.compile: {n} MLP submodule(s) + {m} standalone Float8Linear(s) "
              f"(fla layers left eager; dynamo recompile_limit {getattr(torch._dynamo.config, 'recompile_limit', '?')})")
    mp = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    layers = model.model.layers
    n = len(layers)
    for i, layer in enumerate(layers):
        fully_shard(layer, mp_policy=mp, reshard_after_forward=(i < n - 1))
    fully_shard(model, mp_policy=mp, reshard_after_forward=True)
    return model


# ----------------------------- optimizer --------------------------------------
def build_optimizer(model: nn.Module, opt_cfg) -> torch.optim.Optimizer:
    """AdamW with a two-group weight-decay split: params whose name matches any
    `opt_cfg.no_wd_regex` (norms, embeddings, biases) get weight_decay=0; the rest
    get `opt_cfg.weight_decay`. Only requires_grad params are included; tied weights
    appear once (named_parameters dedups). `opt_cfg.name == "muon"` (opt-in) returns
    parallel/muon.py's MuonWithAdamW instead (Muon on the 2-D hidden weights, AdamW on the rest)."""
    if getattr(opt_cfg, "name", "adamw") == "muon":
        from .muon import MuonWithAdamW, muon_param_groups
        groups, names = muon_param_groups(model, opt_cfg)
        optim = MuonWithAdamW(groups, lr=opt_cfg.lr, betas=tuple(opt_cfg.betas), eps=opt_cfg.eps,
                              weight_decay=opt_cfg.weight_decay, momentum=opt_cfg.muon_momentum,
                              nesterov=opt_cfg.muon_nesterov, ns_steps=opt_cfg.muon_ns_steps,
                              adjust_lr_fn=opt_cfg.muon_adjust_lr)
        optim._group_names = names
        n = {k: len(v) for k, v in names.items()}
        print(f"[optim] MUON: {n['muon']} 2-D tensors on Muon (momentum {opt_cfg.muon_momentum}, nesterov "
              f"{opt_cfg.muon_nesterov}, ns {opt_cfg.muon_ns_steps}, lr scale {opt_cfg.muon_adjust_lr}); AdamW on "
              f"{n['decay']} decay + {n['no_decay']} no-decay tensors (e.g. {', '.join(names['no_decay'][:3])})", flush=True)
        return optim
    patterns = [re.compile(p) for p in opt_cfg.no_wd_regex]
    decay, no_decay = [], []
    decay_names, no_decay_names = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # No weight decay on: any 1-D param (norms, biases, and SSM scalars like
        # Mamba2/GDN A_log, dt_bias, D: the standard ndim<=1 rule), or a name match
        # (embeddings). This keeps recurrent-mixer params correct without enumerating them.
        if p.ndim <= 1 or any(pat.search(name) for pat in patterns):
            no_decay.append(p); no_decay_names.append(name)
        else:
            decay.append(p); decay_names.append(name)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": opt_cfg.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        raise ValueError("No trainable parameters found for the optimizer.")
    optim = torch.optim.AdamW(groups, lr=opt_cfg.lr, betas=tuple(opt_cfg.betas), eps=opt_cfg.eps)
    optim._group_names = {"decay": decay_names, "no_decay": no_decay_names}  # for debugging/logging
    return optim


def build_lr_scheduler(optimizer: torch.optim.Optimizer, opt_cfg, num_steps: int):
    warmup = int(num_steps * opt_cfg.warmup_ratio)
    if opt_cfg.lr_scheduler_type == "constant_with_warmup":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup)
    decay = int(num_steps * opt_cfg.decay_ratio)
    return get_wsd_schedule(
        optimizer,
        num_warmup_steps=warmup,
        num_decay_steps=decay,
        num_training_steps=num_steps,
        warmup_type="linear",
        decay_type=getattr(opt_cfg, "decay_type", "cosine"),
    )
