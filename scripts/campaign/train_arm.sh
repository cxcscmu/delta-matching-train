#!/bin/bash
# scripts/campaign/train_arm.sh <arm> [seed]
# One arm on one node (DM_NPROC GPUs, 8 by default): the run name, per-arm environment and trainer arguments come from
# scripts/campaign/arms.sh (the arm list is at its top), every path from scripts/campaign/env.sh. The experiment drivers in
# scripts/experiments/ call this script once per arm and seed.
#   te_fp8 needs a Python environment with Transformer Engine (docs/SETUP.md section 2): DM_TE_PY / DM_TE_TORCHRUN.
# env: SMOKE=1 (3-step smoke under <run>_smoke3, wandb off, dirs removed) | SMOKE=<n> (n-step probe);
#      STOP_AT_STEP=<n> (stop at n, WSD horizon unchanged); NUM_STEPS (28610); EXTRA_ARGS="--flag value ...";
#      SAVE_TOTAL_LIMIT (2); DM_SAVE_INTERVAL (1500); WANDB_JT (stage1); DM_WANDB=0 (no wandb);
#      TAG=<suffix> appended to the run name; DM_LAUNCH=torchrun|env (env = the ranks are started by this script, so a
#      SIGTERM reaches the trainer directly and it saves a checkpoint before exiting 75);
#      DM_STOP_BEFORE=<epoch seconds> (the trainer's time budget: it saves and exits 75 before that time).
# Resume: relaunching the same arm and seed resumes from $DM_WORKSPACE/qwen35_m/ckpts/<run>; a finished run (the
# TRAIN_COMPLETE marker in its export) exits 0 at once. Exit 75 = stopped early with a checkpoint (resume later).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/env.sh"; source "$HERE/arms.sh"
ARM=${1:?arm}; SEED=${2:-11785}; SMOKE=${SMOKE:-0}
REPO=$DM_REPO; W=$DM_WORKSPACE; SW=$W/qwen35_m; PY=$DM_PY; TR=$DM_TORCHRUN
DATA=$DM_DATA_ROOT/nemotron_cc_v2d1_hq_dqa
[ -f "$DATA/nemotron_rs_train.bin" ] || { echo "FATAL: training data missing at $DATA (docs/SETUP.md section 4)" >&2; exit 2; }
[ -f "$DM_TOKENIZER_DIR/tokenizer.model" ] || { echo "FATAL: tokenizer missing at $DM_TOKENIZER_DIR" >&2; exit 2; }
DATA_ARGS=(--data.paths "$DATA/nemotron_rs_train.bin" --data.doc_lengths "$DATA/nemotron_rs_train.doc_lengths.npy")
if [ "${DM_WANDB:-1}" = 1 ]; then export WANDB_MODE=online; else export WANDB_MODE=disabled; fi   # DM_WANDB=0: no wandb at all
# The fp8 recipe's switches are the code defaults (parallel/wrap.py, trainer/base_trainer.py, modeling/mlp_fp8glue.py);
# exported explicitly as well so a launch log shows the configuration every arm runs.
export DM_FP8_FSDP_AG=1 DM_FP8_PRECOMPUTE=1                                     # fp8 weight all-gather + one scale precompute per step, always both
export DM_FP8GLUE=1 DM_FP8GLUE_SCALING=dynamic DM_FP8GLUE_SAVE_HT=0             # the fp8 cast glue, on every arm
unset DM_STALE_DOQ2S DM_SHARED_X8 DM_FLASH_IMPL DM_SAC_SAVE_FA3WHEEL DM_SDPA_BACKEND DM_DELTA_Q8   # arms.sh sets the arm's own
cd "$REPO" || exit 2; mkdir -p "$SW/ckpts" "$SW/hf" "$W/logs"
dm_arm_setup "$ARM" "$SEED" || exit 2
if [ "$ARM" = te_fp8 ]; then TR=${DM_TE_TORCHRUN:?te_fp8 needs DM_TE_TORCHRUN}; PY=${DM_TE_PY:?te_fp8 needs DM_TE_PY}; fi
"$PY" -c "import pretrain.train" || { echo "FATAL: tree not importable with $PY (PYTHONPATH=$PYTHONPATH)" >&2; exit 2; }
NS=${NUM_STEPS:-28610}
STOPARG=(); [ -n "${STOP_AT_STEP:-}" ] && STOPARG=(--stop_at_step "$STOP_AT_STEP")
if [ "$SMOKE" != 0 ]; then
  SN=$([ "$SMOKE" = 1 ] && echo 3 || echo "$SMOKE")
  if [ -n "${STOP_AT_STEP:-}" ]; then STOPARG=(--stop_at_step "$SN"); else NS=$SN; fi
  RN="${RN}_smoke${SN}"; export WANDB_MODE=disabled
fi
DONE_MARK=$SW/hf/$RN/TRAIN_COMPLETE
if [ "$SMOKE" = 0 ] && [ -f "$DONE_MARK" ]; then echo "[$(hostname -s) $(date +%T)] $RN already complete ($DONE_MARK); nothing to do"; exit 0; fi
LAUNCH=${DM_LAUNCH:-torchrun}
echo "[$(hostname -s) $(date '+%F %T')] LAUNCH $RN smoke=$SMOKE stop_at=${STOP_AT_STEP:-none} extra=${EXTRA_ARGS:-none} launch=$LAUNCH save_interval=${DM_SAVE_INTERVAL:-1500} deadline=${DM_STOP_BEFORE:-none} py=$PY data=$DATA args=${ARGS[*]}"
# The trainer takes the last occurrence of a flag. --hf_export_steps comes before EXTRA_ARGS so that EXTRA_ARGS can set it
# (stage2_arm.sh's periodic exports); the flags after EXTRA_ARGS have their own knobs (NUM_STEPS, DM_LR, ...).
# shellcheck disable=SC2206   # EXTRA_ARGS is split into words on purpose
TRAIN_ARGS=(-m pretrain.train --config "${DM_TRAIN_CONFIG:-$CFG}"   # CFG from arms.sh (qwen35_m / _mla / _mamba2 / _kda / qwen3_m); DM_TRAIN_CONFIG: tests drive this launcher with tests/configs/tiny_hybrid.yaml
  --tokenizer_dir "$DM_TOKENIZER_DIR" "${DATA_ARGS[@]}" "${ARGS[@]}" --hf_export_steps "[]" ${EXTRA_ARGS:-} --num_steps "$NS" "${STOPARG[@]}"
  --save_interval "${DM_SAVE_INTERVAL:-1500}" --save_total_limit "${SAVE_TOTAL_LIMIT:-2}"
  --hf_save_dir "$SW/hf/$RN" --ckpt_root_dir "$SW/ckpts" --log_root_dir "$W/logs"
  --optim.lr "${DM_LR:-2.4e-3}" --seed "$SEED" --run_name "$RN" --wandb "$([ "$WANDB_MODE" = disabled ] && echo false || echo true)" --wandb_job_type "${WANDB_JT:-stage1}")
# SIGTERM handling: the trainer saves a complete checkpoint at the next step and exits 75. Under `env` launch the ranks
# are our children and get the signal directly; under torchrun the signal is forwarded to torchrun's children (the
# ranks), never to torchrun itself (it would kill the ranks after 30 s, before a 20 GB save can finish).
# >>> signal forwarding (tests/test_launcher_signals.py runs this block on its own, so keep it self-contained)
# The handler runs builtins only: no $(...) and no external command. A job scheduler may deliver SIGTERM twice within
# milliseconds (once to every process of the job, then again through a wrapper's own trap); under bash 5.2 a handler
# caught inside a command substitution by the second signal re-enters the parser and aborts the shell ("bad jump",
# exit 134), which would tear the ranks down with it.
# The ranks are signalled first and the message printed after; a repeated signal is only counted.
# Under torchrun (TR_PID set) the ranks are torchrun's children, read from /proc when the signal arrives (builtins
# only), so ranks spawned after the launch are reached too. Only when torchrun has no child yet is torchrun itself
# signalled: it then stops before any rank has trained.
DM_HOST=$(hostname -s); PIDS=(); NSIG=0; TR_PID=""
forward_term () {
  NSIG=$((NSIG + 1)); [ "$NSIG" -gt 1 ] && return 0
  local p f k who="trainer ranks"; local -a T=("${PIDS[@]}")
  if [ -n "$TR_PID" ]; then
    T=()
    for f in /proc/"$TR_PID"/task/*/children; do k=(); [ -r "$f" ] && read -r -a k < "$f"; T+=(${k[@]+"${k[@]}"}); done
    [ ${#T[@]} -gt 0 ] || { T=("$TR_PID"); who="torchrun (no rank started yet)"; }
  fi
  for p in "${T[@]}"; do kill -TERM "$p" 2>/dev/null; done
  printf '[%s %(%T)T] SIGTERM -> %s\n' "$DM_HOST" -1 "$who"
}
# wait_ranks: the ranks' exit statuses -> RC (a 75 from any rank wins, otherwise the last failure). The previous
# `while kill -0 "$p"; do wait "$p"; done` never collected a rank that had exited before its turn in the loop (kill -0
# fails on it), so a rank failing while rank 0 still ran was reported as 0. `wait` returns the stored status of a rank
# that has already exited; it is repeated only when a trapped signal interrupted it (it then returns 128+N early).
wait_ranks () {
  local p r s n; RC=0
  for p in "${PIDS[@]}"; do
    r=0
    while :; do
      n=$NSIG; wait "$p"; s=$?
      [ "$s" -eq 127 ] && break                     # already collected: keep what we have, never report 127
      r=$s; [ "$NSIG" != "$n" ] && [ "$s" -gt 128 ] && continue
      break
    done
    [ $r -ne 0 ] && { [ $RC -eq 0 ] || [ $r -eq 75 ] && RC=$r; [ $RC -eq 75 ] || RC=$r; }
  done
}
# wait_torchrun: torchrun's exit status -> RC, the same way. The previous `sleep 5; while kill -0 ...; do wait ...; done`
# reported 0 for a torchrun that had exited during the sleep, and a SIGTERM during the sleep killed this shell.
wait_torchrun () {
  local s n
  while :; do
    n=$NSIG; wait "$TR_PID"; s=$?
    [ "$NSIG" != "$n" ] && [ "$s" -gt 128 ] && continue
    break
  done
  RC=$s
}
# <<< signal forwarding
if [ "$LAUNCH" = env ]; then
  export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} MASTER_PORT=${DM_MASTER_PORT:-29517} WORLD_SIZE=$DM_NPROC LOCAL_WORLD_SIZE=$DM_NPROC OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
  for r in $(seq 0 $((DM_NPROC - 1))); do
    # Per-rank Triton/TileLang caches. With one shared cache on a network FS the 8 ranks autotune and
    # compile the same kernel concurrently and one rank reads a metadata file another is still writing
    # (triton/runtime/autotuner.py check_disk_cache -> FileNotFoundError); that rank dies and the other
    # seven deadlock in NCCL. Seen on the GDN kernel prepare_wy_repr_bwd_kernel.
    RANK=$r LOCAL_RANK=$r \
      TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$W/triton_cache}/r$r" \
      TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-$W/tilelang_cache}/r$r" \
      "$PY" "${TRAIN_ARGS[@]}" > "$W/logs/${RN}.rank${r}.log" 2>&1 & PIDS+=($!)
  done
  trap forward_term TERM INT
  wait_ranks
  echo "--- rank 0 tail ---"; tail -5 "$W/logs/${RN}.rank0.log"
else
  # torchrun reports a rank's exit 75 as its own failure (1): the stop is read from the trainer log, which every attempt
  # appends to, so only the part written by this launch counts. grep reads it through a process substitution: in a
  # `tail | grep -q` pipeline under pipefail, tail dies of SIGPIPE when more than a pipe buffer follows the match
  TLOG=$W/logs/${RN}.log; LOG0=0; [ -f "$TLOG" ] && LOG0=$(wc -c < "$TLOG")
  "$TR" --standalone --nproc_per_node="$DM_NPROC" "${TRAIN_ARGS[@]}" & TR_PID=$!
  trap forward_term TERM INT
  wait_torchrun
  if [ $RC -ne 0 ] && grep -aq "Stopping early" <(tail -c +$((LOG0 + 1)) "$TLOG" 2>/dev/null) && [ ! -f "$SW/hf/$RN/config.json" ]; then RC=75; fi
  # stopped by the signal before a rank saved: report a death by signal (128 + 15), not torchrun's 1, so a caller can tell
  # a stop from a configuration failure
  if [ $RC -ne 0 ] && [ $RC -ne 75 ] && [ "$NSIG" -gt 0 ]; then RC=143; fi
fi
trap - TERM INT
echo "[$(hostname -s) $(date +%T)] $RN done (exit $RC)"
if [ $RC -eq 75 ]; then
  echo "[$(hostname -s) $(date +%T)] $RN stopped early with a complete checkpoint: relaunch to resume"
  exit 75
fi
if [ -f "$SW/hf/$RN/config.json" ]; then
  "$PY" - "$SW/hf/$RN" <<'PYEOF'
import json, sys, shutil
p = f"{sys.argv[1]}/config.json"; shutil.copy(p, f"{sys.argv[1]}/config.train.json"); c = json.load(open(p))
c["attn_impl"] = "sdpa"                                   # every arm is scored through the common clean kernel
if c.get("mixer") == "dm": c["mixer"] = "gqa"
if c.get("layer_mixers"): c["layer_mixers"] = [{"dm": "gqa", "mla_dm": "mla"}.get(m, m) for m in c["layer_mixers"]]
if c.get("mixer") == "mla_dm": c["mixer"] = "mla"
for k in ("dm_mode", "dm_fused_producers"): c.pop(k, None)   # the fp8-only keys; config.train.json keeps them
json.dump(c, open(p, "w"), indent=2, sort_keys=True); print("export patched -> clean-kernel eval")
PYEOF
  VAL=$DATA/nemotron_rs_val.bin
  unset DM_FP8GLUE DM_FP8GLUE_SCALING DM_FP8GLUE_SAVE_HT DM_SHARED_X8 DM_STALE_DOQ2S DM_DELTA_Q8 DM_STOP_BEFORE
  VALCE_GPU=${CUDA_VISIBLE_DEVICES:-0}; VALCE_GPU=${VALCE_GPU%%,*}   # the first GPU this launcher was given
  # --wandb: eval/* go to this run's W&B page (memmap_ppl logs only when asked), exactly when the trainer ran with wandb
  VALCE_WANDB=(); [ "$WANDB_MODE" = disabled ] || VALCE_WANDB=(--wandb)
  RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29797 CUDA_VISIBLE_DEVICES=$VALCE_GPU \
  "$PY" -m evaluation.ppl.memmap_ppl --model_name_or_path "$SW/hf/$RN" --data "$VAL" --dtype uint16 --seq_len 8192 --max_windows 128 --batch_size 8 "${VALCE_WANDB[@]}" > "$SW/hf/$RN/valce.txt" 2>&1 || echo "(val ce failed)"
  grep -a "memmap-ppl RESULT" "$SW/hf/$RN/valce.txt" | tail -1
  if [ "$SMOKE" != 0 ]; then echo "SMOKE $RN: export+valce tail OK; removing smoke dirs"; rm -rf "$SW/hf/$RN" "$SW/ckpts/$RN"
  else
    date '+%F %T' > "$DONE_MARK"; echo "[$(hostname -s) $(date +%T)] TRAIN COMPLETE marker written: $DONE_MARK"
  fi
fi
exit $RC
