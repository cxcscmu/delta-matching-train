#!/bin/bash
# scripts/campaign/eval_arm.sh <run_name> [windows]
# The full evaluation protocol of docs/benchmark.md for one exported arm:
#   1. val CE on the first <windows> (default 2000) 8192-token windows of the held-out stream (one GPU; per-window artifact)
#   2. RULER 4k + 8k (13 tasks, 500 examples each; the in-repo engine)
#   3. commonsense-9 (lm-eval loglikelihood)          4. extraction (swde / fda / squad_completion; engine)
#   5. MQAR (recall_synthetic)                        6. MMLU (offline, from the HF datasets cache)
#   7. TriviaQA 5-shot exact match (engine; docs/benchmark.md section 7)
# Output: $DM_WORKSPACE/qwen35_m/eval/<run_name>/ with valce_n<W>.txt, <run>_n<W>_perwindow.npy, <cfg>/<cfg>_results.json
# and a <cfg>_results.complete marker per benchmark. Verify by artifact (the markers + zero tracebacks in the log), never
# by the "done" lines. Every step is skipped when its marker already exists, so the script can be re-run to fill gaps
# (a benchmark interrupted mid-way restarts from scratch, RULER per task). When all seven are complete the EVAL_COMPLETE
# marker is written and the script exits 0; otherwise it exits 1 naming what is missing. DM_EVAL_NO_CACHE=1 disables the
# harness request cache (recommended when jobs are stopped often: nothing half-written can trip a restart).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/env.sh"
RN=${1:?run name}; NW=${2:-2000}
REPO=$DM_REPO; W=$DM_WORKSPACE; PY=$DM_PY; TR=$DM_TORCHRUN
HFDIR=$W/qwen35_m/hf/$RN; OUT=$W/qwen35_m/eval/$RN; LOG=$W/logs/eval_${RN}.log
VAL=$DM_DATA_ROOT/nemotron_cc_v2d1_hq_dqa/nemotron_rs_val.bin; RD=$DM_DATA_ROOT/ruler-500-llama2
say () { echo "[$(hostname -s) $(date +%T)] $*" | tee -a "$LOG"; }
# SIGTERM / SIGINT stop the evaluation: the running benchmark's torchrun and its ranks are signalled, and the script exits
# (143 / 130) once they are gone instead of starting the next benchmark. The handler runs builtins only (the ranks are
# read from /proc): a second signal, as a scheduler may deliver, is only counted.
EPID=""; STOP=0; SIGRC=143
on_signal () {
  STOP=$((STOP + 1)); [ "$STOP" -gt 1 ] && return 0
  SIGRC=$1; local p f k; local -a T=()
  if [ -n "$EPID" ]; then
    for f in /proc/"$EPID"/task/*/children; do k=(); [ -r "$f" ] && read -r -a k < "$f"; T+=(${k[@]+"${k[@]}"}); done
    for p in ${T[@]+"${T[@]}"} "$EPID"; do kill -TERM "$p" 2>/dev/null; done
  fi
}
trap 'on_signal 143' TERM; trap 'on_signal 130' INT
stop_if_signalled () { [ "$STOP" -eq 0 ] || { say "stopped by a signal (a benchmark interrupted mid-way restarts from scratch next time)"; exit "$SIGRC"; }; }
[ -f "$HFDIR/config.json" ] || { echo "FATAL: no export at $HFDIR" >&2; exit 2; }
[ -s "$VAL" ] || { echo "FATAL: val stream missing at $VAL" >&2; exit 2; }
mkdir -p "$OUT" "$W/logs"; cd "$REPO" || exit 2
[ -f "$OUT/EVAL_COMPLETE" ] && { echo "[$(hostname -s) $(date +%T)] $RN eval already complete ($OUT/EVAL_COMPLETE)"; exit 0; }
"$PY" -c "import json, sys; c = json.load(open(sys.argv[1])); m = [c.get('mixer')] + list(c.get('layer_mixers') or []); assert c.get('attn_impl') == 'sdpa' and not {'dm', 'mla_dm'} & set(m), (c.get('attn_impl'), c.get('mixer'), c.get('layer_mixers')); print('export config OK', c.get('layer_mixers') or c.get('mixer'))" "$HFDIR/config.json" \
  || { echo "FATAL: export config not patched for the clean-kernel eval (attn_impl sdpa, no dm / mla_dm mixer; train_arm.sh does this at the end of a run)" >&2; exit 2; }
export WANDB_MODE=disabled TORCH_HOME=$W/torch_home
say "EVAL START $RN (7 benchmarks; windows=$NW, export=$HFDIR, data=$DM_DATA_ROOT)"
# 1. val CE: complete only when BOTH the RESULT line and the per-window artifact (NW windows) exist; the scorer prints
# the RESULT before it writes the artifact, so a kill in between must re-run the scorer.
valce_ok () { grep -aq "memmap-ppl RESULT" "$OUT/valce_n${NW}.txt" 2>/dev/null && "$PY" -c "import numpy as np,sys; a=np.load(sys.argv[1]); sys.exit(0 if a.shape==(2,int(sys.argv[2])) and np.isfinite(a[1]).all() else 1)" "$OUT/${RN}_n${NW}_perwindow.npy" "$NW" 2>/dev/null; }
if valce_ok; then say "val CE already done"; else
  say "1/7 val CE @ $NW windows"; rm -f "$OUT/valce_n${NW}.txt" "$OUT/${RN}_n${NW}_perwindow.npy"
  RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29800 + ${DM_EVAL_GPU:-0})) CUDA_VISIBLE_DEVICES=${DM_EVAL_GPU:-0} \
  "$PY" -m evaluation.ppl.memmap_ppl --model_name_or_path "$HFDIR" --data "$VAL" --dtype uint16 --seq_len 8192 \
      --max_windows "$NW" --batch_size 4 --output_dir "$OUT" --run_name "${RN}_n${NW}" > "$OUT/valce_n${NW}.txt" 2>&1 || say "  (val CE rc=$?)"
fi
grep -a "memmap-ppl RESULT" "$OUT/valce_n${NW}.txt" 2>/dev/null | tail -1 | tee -a "$LOG"
stop_if_signalled
mqar_mark () {  # the recall evaluator writes its own recall_synthetic_results.complete; this fallback also marks MQAR complete
  # when only its results JSON exists (as older evaluators left it) and carries a finite mean_accuracy
  local d=$OUT/recall_synthetic
  [ -f "$d/recall_synthetic_results.complete" ] && return 0
  "$PY" -c "import json,math,sys; r=json.load(open(sys.argv[1])); v=float(r['mean_accuracy']); sys.exit(0 if math.isfinite(v) else 1)" "$d/recall_synthetic_results.json" 2>/dev/null \
    && date '+%F %T' > "$d/recall_synthetic_results.complete" && say "  recall_synthetic marked complete (mean_accuracy present)"
}
run_cfg () {  # run_cfg <config name> <yaml path> <batch size> [engine]
  local name=$1 yaml=$2 bs=$3 engine=${4:-0} erc n
  stop_if_signalled
  if [ -f "$OUT/$name/${name}_results.complete" ]; then say "$name already done"; return 0; fi
  say "$name (bs=$bs engine=$engine)"
  if [ "$engine" = 1 ]; then export EVAL_ENGINE=mini_eval_dev ENGINE_MAX_SEQS=24; else unset EVAL_ENGINE ENGINE_MAX_SEQS; fi
  local extra=(); [ "${DM_EVAL_NO_CACHE:-0}" = 1 ] && extra=(--no_cache_requests)
  "$TR" --standalone --nproc_per_node="$DM_NPROC" -m evaluation.meta_eval --model_name_or_path "$HFDIR" \
     --eval_config "$yaml" --output_dir "$OUT" --run_name "$RN" --eval_batch_size "$bs" "${extra[@]}" >> "$OUT/${name}.txt" 2>&1 & EPID=$!
  while :; do n=$STOP; wait "$EPID"; erc=$?; [ "$STOP" != "$n" ] && [ "$erc" -gt 128 ] && continue; break; done   # re-wait only after a trapped signal
  EPID=""; [ "$erc" -ne 0 ] && say "  ($name rc=$erc)"
  stop_if_signalled
  unset EVAL_ENGINE ENGINE_MAX_SEQS
  [ -f "$OUT/$name/${name}_results.complete" ] && say "  $name complete" || say "  $name NOT complete (see $OUT/${name}.txt)"
}
# 2. RULER 4k + 8k: the config with the data_dir pointed at DM_DATA_ROOT
if [ -d "$RD/8192/data" ]; then
  sed "s#^data_dir:.*#data_dir: $RD#" src/evaluation/configs/ruler_llama2_8k.yaml > "$OUT/ruler_llama2_8k.yaml"
  run_cfg ruler_llama2_8k "$OUT/ruler_llama2_8k.yaml" 16 1
else say "RULER data not found at $RD: skipped"; fi
# 3. commonsense-9   4. extraction (engine)   5. MQAR   6. MMLU (offline cache)
run_cfg commonsense src/evaluation/configs/commonsense.yaml 32
run_cfg recall_extraction src/evaluation/configs/recall_extraction.yaml 32 1
run_cfg triviaqa src/evaluation/configs/triviaqa.yaml 32 1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
run_cfg recall_synthetic src/evaluation/configs/recall_synthetic.yaml 16   # the YAML's batch_size (8) wins over this value: the protocol's MQAR numbers ran at 8
mqar_mark
if [ -d "$HF_HOME/datasets/cais___mmlu" ]; then run_cfg mmlu src/evaluation/configs/mmlu.yaml 16; else say "SKIP mmlu: cais___mmlu not in $HF_HOME/datasets"; fi
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
MISSING=(); valce_ok || MISSING+=(valce)
for c in ruler_llama2_8k commonsense recall_extraction triviaqa recall_synthetic mmlu; do [ -f "$OUT/$c/${c}_results.complete" ] || MISSING+=("$c"); done
grep -c "Traceback" "$OUT"/*.txt 2>/dev/null | grep -v ":0$" | sed 's/^/  tracebacks in /' | tee -a "$LOG"
if [ ${#MISSING[@]} -eq 0 ]; then date '+%F %T' > "$OUT/EVAL_COMPLETE"; say "EVAL COMPLETE $RN: all seven benchmarks have their markers ($OUT/EVAL_COMPLETE)"; exit 0
else say "EVAL NOT COMPLETE $RN: missing ${MISSING[*]} (re-run this script to fill them)"; exit 1; fi
