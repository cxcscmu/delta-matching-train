#!/bin/bash
# scripts/campaign/arms.sh: the arm table of the paper's experiments, shared by train_arm.sh (full setup), stage2_arm.sh
# and the experiment drivers in scripts/experiments/ (run name only). Source after env.sh.
#   dm_arm_setup <arm> <seed>     sets RN (with ${TAG}), CFG, MODE, ARGS[] and exports the per-arm environment; checks that
#                                 the delta_matching kernel package imports and carries the build the arm needs (returns
#                                 2 on a fatal problem). NAME_ONLY=1 skips the checks and the exports (only RN is needed).
#   dm_run_name <arm> <seed>      prints the run name (a subshell around dm_arm_setup with NAME_ONLY=1).
#   dm_arms                       prints every arm, one per line.
# Numerics of the fp8 arms (dm_mode): stale = the naive fp8 backward (the delta from the forward's output), matched =
# Delta-Matching; the consistent-dO arm is the stale arm with the delta computed from the dequantized dO codes.
#
#   the head_dim-128 cell (1.665 B GatedDeltaNet : GQA 3:1, qwen35_m.yaml with 16 q / 4 kv heads of 128):
#     clean_hd128          Standard BF16/FP32 Mix-precision (cuDNN bf16 attention, fp8 FFN)
#     dm_stale_hd128       Naive FP8 (with Stale Delta)
#     dm_matched_hd128     Delta-Matching
#     dm_consistent_hd128  Naive FP8 (Consistent dO)
#     te_fp8               cuDNN / Transformer Engine FP8 attention (needs the Transformer Engine environment)
#   the same cell without positional encoding (--model.nope true) and trained with Muon (--optim.name muon):
#     nope_clean_hd128 | nope_dm_stale_hd128 | nope_dm_matched_hd128
#     muon_clean_hd128 | muon_dm_stale_hd128 | muon_dm_matched_hd128
#   head_dim 256 at three sizes (full fp8 pipeline; the Standard arm runs the FlashAttention-3 bf16 kernel):
#     xs_clean_hd256 | xs_dm_stale_hd256 | xs_dm_matched_hd256       0.569 B (qwen35_xs.yaml)
#     clean_hd256    | dm_stale_hd256    | dm_matched_hd256          1.665 B (qwen35_m.yaml)
#     l_clean_hd256  | l_dm_stale_hd256  | l_dm_matched_hd256        5.289 B (qwen35_l.yaml)
#   other architectures at 1.665 B / 1.475 B, head_dim 128:
#     mla_{clean,dm_stale,dm_matched}_hd128      GatedDeltaNet : MLA 3:1 (qwen35_m_mla.yaml)
#     kda_{clean,dm_stale,dm_matched}_hd128      KDA : GQA 3:1 (qwen35_m_kda.yaml)
#     mamba2_{clean,dm_stale,dm_matched}_hd128   Mamba2 : GQA 3:1 (qwen35_m_mamba2.yaml)
#     gqa_{clean,dm_stale,dm_matched}_hd128      pure GQA, the Qwen3-1.7B geometry (qwen3_m.yaml)
# shellcheck disable=SC2034   # RN, CFG, MODE and ARGS are read by the scripts that source this file
LM="[gated_deltanet, gated_deltanet, gated_deltanet, gqa]"
LM_DM="[gated_deltanet, gated_deltanet, gated_deltanet, dm]"
COMMON=(--fp8 true --compile_mlp true --model.fused_swiglu false --model.fused_rmsnorm true --gradient_checkpointing false --compile false)
GEOM128=(--model.n_heads 16 --model.n_kv_heads 4 --model.head_dim 128)
GEOM_MLA=(--model.n_heads 16 --model.n_kv_heads 16 --model.head_dim 128)   # MLA: MHA on the wire
CFG_GDN=src/pretrain/configs/qwen35_m.yaml; CFG_MLA=src/pretrain/configs/qwen35_m_mla.yaml; CFG_M2=src/pretrain/configs/qwen35_m_mamba2.yaml
CFG_KDA=src/pretrain/configs/qwen35_m_kda.yaml; CFG_XS=src/pretrain/configs/qwen35_xs.yaml; CFG_L=src/pretrain/configs/qwen35_l.yaml
CFG_Q3M=src/pretrain/configs/qwen3_m.yaml; GEOM_Q3M=(--model.n_heads 16 --model.n_kv_heads 8 --model.head_dim 128)   # pure GQA (Qwen3-1.7B geometry, 28 attention layers)
LM_MLA="[gated_deltanet, gated_deltanet, gated_deltanet, mla]"; LM_MLA_DM="[gated_deltanet, gated_deltanet, gated_deltanet, mla_dm]"
LM_M2="[mamba2, mamba2, mamba2, gqa]"; LM_M2_DM="[mamba2, mamba2, mamba2, dm]"
LM_KDA="[kda, kda, kda, gqa]"; LM_KDA_DM="[kda, kda, kda, dm]"
DM_ARMS="clean_hd128 dm_stale_hd128 dm_matched_hd128 dm_consistent_hd128 te_fp8
nope_clean_hd128 nope_dm_stale_hd128 nope_dm_matched_hd128 muon_clean_hd128 muon_dm_stale_hd128 muon_dm_matched_hd128
xs_clean_hd256 xs_dm_stale_hd256 xs_dm_matched_hd256 clean_hd256 dm_stale_hd256 dm_matched_hd256 l_clean_hd256 l_dm_stale_hd256 l_dm_matched_hd256
mla_clean_hd128 mla_dm_stale_hd128 mla_dm_matched_hd128 kda_clean_hd128 kda_dm_stale_hd128 kda_dm_matched_hd128
mamba2_clean_hd128 mamba2_dm_stale_hd128 mamba2_dm_matched_hd128 gqa_clean_hd128 gqa_dm_stale_hd128 gqa_dm_matched_hd128"
dm_arms () { local a; for a in $DM_ARMS; do echo "$a"; done; }
_dm_small_gpu_micro_batch () {   # the MLA and KDA yamls' micro-batch of 4 sequences per GPU needs more than 80 GB (the
  # paper's MLA and KDA runs had 141 GB GPUs); below 100 GB per GPU these arms run 2 sequences per GPU (45 to 47 GB peak
  # on an 80 GB H100): the same global batch, data order and schedule with twice the gradient accumulation
  # (EXTRA_ARGS="--per_device_batch_size <n>" wins)
  local mib
  mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | sort -n | head -n 1)
  case "$mib" in ''|*[!0-9]*) return 0 ;; esac
  if [ "$mib" -lt 100000 ]; then
    ARGS+=(--per_device_batch_size 2)
    echo "[arms] $BASE: $mib MiB per GPU: --per_device_batch_size 2 (the yaml's 4 needs more than 80 GB)" >&2
  fi
  return 0
}
_dm_env () {  # the runtime switches of the fp8 arms: the code defaults, exported explicitly so a launch log shows them
  export DM_SHARED_X8=1
  [ "$1" = stale ] && export DM_STALE_DOQ2S=1   # the stale arm's dO quantizer flag (kept for the launch contract; inert on this route)
  return 0
}
dm_check_kernel () {   # dm_check_kernel <need_hd128 0|1>: the delta_matching package imports and has the build the arm needs
  "$DM_PY" - "$1" <<'PY' || return 2
import sys
try:
    import delta_matching
    from delta_matching.core import _ops
except ImportError as e:
    sys.exit(f"FATAL: the delta_matching kernel package is not importable ({e}): install the delta-matching kernel package, "
             "see docs/SETUP.md section 3 (git clone --recursive; pip install --no-build-isolation .)")
ops = _ops()
need_hd128 = sys.argv[1] == "1"
hd128, gt, bf16dp = int(ops.build_hd128()), int(ops.build_gt()), int(getattr(ops, "build_bf16dp", lambda: 0)())
if gt != 1:
    sys.exit("FATAL: the delta_matching build lacks the global-memory twins (build_gt() != 1)")
if need_hd128 and hd128 != 1:
    sys.exit("FATAL: head_dim 128 needs a Delta-Matching-128 build of delta_matching (build_hd128() == 1)")
if bf16dp:
    sys.exit("FATAL: this delta_matching is a DM_BF16_DP=1 build (build_bf16dp() == 1), a study variant that refuses the "
             "stale and matched modes; rebuild the package with the default flags")
print(f"[arms] delta_matching {delta_matching.__version__} {delta_matching.kernel_version()}: build_hd128={hd128} build_gt={gt}")
PY
}
_size256 () {  # _size256 <arm>: the config and run-name prefix of a head_dim-256 arm (xs_ = 0.569 B, none = 1.665 B, l_ = 5.289 B)
  case "$1" in
    xs_*) CFG=$CFG_XS; P=Q35XS ;;
    l_*)  CFG=$CFG_L;  P=Q35L ;;
    *)    CFG=$CFG_GDN; P=Q35M ;;
  esac
}
dm_arm_setup () {
  local ARM=$1 SEED=${2:-11785} N=${NAME_ONLY:-0} BASE=$1 SFX="" P="" VAR=()
  case "$ARM" in   # the NoPE and Muon variants of the head_dim-128 cell: the base arm plus one flag and a run-name suffix
    nope_clean_hd128|nope_dm_stale_hd128|nope_dm_matched_hd128) BASE=${ARM#nope_}; SFX=_nope; VAR=(--model.nope true) ;;
    muon_clean_hd128|muon_dm_stale_hd128|muon_dm_matched_hd128) BASE=${ARM#muon_}; SFX=_muon; VAR=(--optim.name muon) ;;
  esac
  CFG=$CFG_GDN
  case "$BASE" in
    clean_hd128)
      [ "$N" = 1 ] || export DM_SDPA_BACKEND=cudnn
      RN="Q35M_clean_sdpa_ffn8_hd128_${SEED}"; MODE=clean
      ARGS=(--model.layer_mixers "$LM" --model.attn_impl sdpa --model.fused_rope true --fp8_scope ffn "${COMMON[@]}" "${GEOM128[@]}") ;;
    dm_stale_hd128|dm_matched_hd128|dm_consistent_hd128)
      case "$BASE" in
        dm_stale_hd128)      MODE=stale;   RN="Q35M_dm_stale_ffn8_qkvo8_hd128_${SEED}" ;;
        dm_matched_hd128)    MODE=matched; RN="Q35M_dm_matched_ffn8_qkvo8_hd128_${SEED}" ;;
        dm_consistent_hd128) MODE=stale;   RN="Q35M_dm_consistent_ffn8_qkvo8_hd128_${SEED}" ;;
      esac
      if [ "$N" != 1 ]; then
        dm_check_kernel 1 || return 2; _dm_env "$MODE"
        [ "$BASE" = dm_consistent_hd128 ] && export DM_DELTA_Q8=1   # the stale delta from the dO codes the dP matmul uses
      fi
      ARGS=(--model.layer_mixers "$LM_DM" --model.dm_mode "$MODE" --model.dm_fused_producers true --fp8_scope ffn_qkvo "${COMMON[@]}" "${GEOM128[@]}") ;;
    te_fp8)
      RN="Q35M_te_fp8_ffn8_qkvo8_hd128_${SEED}"; MODE=te
      ARGS=(--model.layer_mixers "$LM" --model.attn_impl te_fp8 --model.fused_rope true --fp8_scope ffn_qkvo "${COMMON[@]}" "${GEOM128[@]}") ;;
    # ---- head_dim 256 (the yaml's own geometry) at three sizes ----
    xs_clean_hd256|clean_hd256|l_clean_hd256)
      _size256 "$BASE"; [ "$N" = 1 ] || export DM_FLASH_IMPL=fa3 DM_SAC_SAVE_FA3WHEEL=1
      RN="${P}_clean_fa3_ffn8_${SEED}"; MODE=clean
      ARGS=(--model.layer_mixers "$LM" --model.attn_impl flash --model.fused_rope true --fp8_scope ffn "${COMMON[@]}") ;;
    xs_dm_stale_hd256|dm_stale_hd256|l_dm_stale_hd256|xs_dm_matched_hd256|dm_matched_hd256|l_dm_matched_hd256)
      _size256 "$BASE"
      case "$BASE" in *dm_stale_hd256) MODE=stale;; *) MODE=matched;; esac
      RN="${P}_dm_${MODE}_ffn8_qkvo8_${SEED}"
      if [ "$N" != 1 ]; then dm_check_kernel 0 || return 2; _dm_env "$MODE"; fi
      ARGS=(--model.layer_mixers "$LM_DM" --model.dm_mode "$MODE" --model.dm_fused_producers true --fp8_scope ffn_qkvo "${COMMON[@]}") ;;
    # ---- the 3:1 GatedDeltaNet : MLA hybrid (modeling/mla.py) ----
    mla_clean_hd128)
      CFG=$CFG_MLA; [ "$N" = 1 ] || export DM_SDPA_BACKEND=cudnn
      RN="Q35MLA_clean_sdpa_ffn8_hd128_${SEED}"; MODE=clean
      ARGS=(--model.layer_mixers "$LM_MLA" --model.attn_impl sdpa --fp8_scope ffn "${COMMON[@]}" "${GEOM_MLA[@]}")
      [ "$N" = 1 ] || _dm_small_gpu_micro_batch ;;
    mla_dm_stale_hd128|mla_dm_matched_hd128)
      CFG=$CFG_MLA
      if [ "$BASE" = mla_dm_stale_hd128 ]; then MODE=stale; else MODE=matched; fi
      RN="Q35MLA_dm_${MODE}_ffn8_qkvo8_hd128_${SEED}"
      if [ "$N" != 1 ]; then dm_check_kernel 1 || return 2; fi   # the plain kernel route: no fused producers, no DM_* switches
      ARGS=(--model.layer_mixers "$LM_MLA_DM" --model.dm_mode "$MODE" --fp8_scope ffn_qkvo "${COMMON[@]}" "${GEOM_MLA[@]}")
      [ "$N" = 1 ] || _dm_small_gpu_micro_batch ;;
    # ---- the 3:1 Mamba2 : GQA hybrid (modeling/mamba2.py) ----
    mamba2_clean_hd128)
      CFG=$CFG_M2; [ "$N" = 1 ] || export DM_SDPA_BACKEND=cudnn
      RN="Q35M2_clean_sdpa_ffn8_hd128_${SEED}"; MODE=clean
      ARGS=(--model.layer_mixers "$LM_M2" --model.attn_impl sdpa --model.fused_rope true --fp8_scope ffn "${COMMON[@]}" "${GEOM128[@]}") ;;
    mamba2_dm_stale_hd128|mamba2_dm_matched_hd128)
      CFG=$CFG_M2
      if [ "$BASE" = mamba2_dm_stale_hd128 ]; then MODE=stale; else MODE=matched; fi
      RN="Q35M2_dm_${MODE}_ffn8_qkvo8_hd128_${SEED}"
      if [ "$N" != 1 ]; then dm_check_kernel 1 || return 2; _dm_env "$MODE"; fi
      ARGS=(--model.layer_mixers "$LM_M2_DM" --model.dm_mode "$MODE" --model.dm_fused_producers true --fp8_scope ffn_qkvo "${COMMON[@]}" "${GEOM128[@]}") ;;
    # ---- the 3:1 KDA (Kimi Delta Attention) : GQA hybrid (modeling/kda.py) ----
    kda_clean_hd128)
      CFG=$CFG_KDA; [ "$N" = 1 ] || export DM_SDPA_BACKEND=cudnn
      RN="Q35KDA_clean_sdpa_ffn8_hd128_${SEED}"; MODE=clean
      ARGS=(--model.layer_mixers "$LM_KDA" --model.attn_impl sdpa --model.fused_rope true --fp8_scope ffn "${COMMON[@]}" "${GEOM128[@]}")
      [ "$N" = 1 ] || _dm_small_gpu_micro_batch ;;
    kda_dm_stale_hd128|kda_dm_matched_hd128)
      CFG=$CFG_KDA
      if [ "$BASE" = kda_dm_stale_hd128 ]; then MODE=stale; else MODE=matched; fi
      RN="Q35KDA_dm_${MODE}_ffn8_qkvo8_hd128_${SEED}"
      if [ "$N" != 1 ]; then dm_check_kernel 1 || return 2; _dm_env "$MODE"; fi
      ARGS=(--model.layer_mixers "$LM_KDA_DM" --model.dm_mode "$MODE" --model.dm_fused_producers true --fp8_scope ffn_qkvo "${COMMON[@]}" "${GEOM128[@]}")
      [ "$N" = 1 ] || _dm_small_gpu_micro_batch ;;
    # ---- pure GQA (the Qwen3-1.7B geometry, 28 attention layers) ----
    gqa_clean_hd128)
      CFG=$CFG_Q3M; [ "$N" = 1 ] || export DM_SDPA_BACKEND=cudnn
      RN="Q3M_clean_sdpa_ffn8_hd128_${SEED}"; MODE=clean
      ARGS=(--model.mixer gqa --model.attn_impl sdpa --model.fused_rope true --fp8_scope ffn "${COMMON[@]}" "${GEOM_Q3M[@]}") ;;
    gqa_dm_stale_hd128|gqa_dm_matched_hd128)   # Delta-Matching-128 with n_rep 2
      CFG=$CFG_Q3M
      if [ "$BASE" = gqa_dm_stale_hd128 ]; then MODE=stale; else MODE=matched; fi
      RN="Q3M_dm_${MODE}_ffn8_qkvo8_hd128_${SEED}"
      if [ "$N" != 1 ]; then dm_check_kernel 1 || return 2; _dm_env "$MODE"; fi
      ARGS=(--model.mixer dm --model.dm_mode "$MODE" --model.dm_fused_producers true --fp8_scope ffn_qkvo "${COMMON[@]}" "${GEOM_Q3M[@]}") ;;
    *) echo "unknown arm $ARM (the arms: scripts/campaign/arms.sh; or source scripts/campaign/env.sh and arms.sh, then dm_arms)" >&2; return 2 ;;
  esac
  ARGS+=(${VAR[@]+"${VAR[@]}"})
  RN="${RN}${SFX}${TAG:-}"
  return 0
}
dm_run_name () { ( NAME_ONLY=1 dm_arm_setup "$1" "${2:-11785}" >/dev/null 2>&1 && echo "$RN" ); }
