#!/bin/bash
# scripts/campaign/eval_arm_64k.sh <run_name> [windows64k=256]
# The stage-2 (64k, YaRN) evaluation protocol for one exported arm (the context-extension results), reusing
# eval_arm.sh's conventions (artifacts + .complete markers, re-runnable, verify by artifact):
#   1. val CE at 65,536 tokens on the held-out ProLong-64K windows (prolong64k_val: 256 x 64k, disjoint from the
#      training pool; per-window artifact for paired contrasts)          2. val CE at 8,192 on the Nemotron held-out
#      stream (2000 windows, the stage-1 protocol; the regression check)   3. RULER 4k / 8k / 16k / 32k / 64k
#   4. commonsense-9   5. extraction   6. TriviaQA   7. MQAR (the stage-1 grid to 16k: the synthetic task needs seq_len < vocab_size = 32,000)
#   8. MMLU (offline cache)
# Output: $DM_WORKSPACE/qwen35_m/eval/<run_name>/ (valce64k_n<W>.txt + <run>_64k_n<W>_perwindow.npy, valce_n2000.txt +
# <run>_n2000_perwindow.npy, <cfg>/<cfg>_results.json + .complete). EVAL64K_COMPLETE when all eight are present.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); source "$HERE/env.sh"
RN=${1:?run_name}; NW64=${2:-256}; NW=2000
SW=$DM_WORKSPACE/qwen35_m; HFDIR=$SW/hf/$RN; OUT=$SW/eval/$RN; PY=$DM_PY
# Reduced (smoke) configs, S2_RULER_CFG / S2_MQAR_CFG, write a directory of their own: their markers carry the protocol's
# names (ruler_llama2_64k, recall_synthetic), so in the run's directory they would make a later full evaluation skip
# those benchmarks.
if [ -n "${S2_RULER_CFG:-}${S2_MQAR_CFG:-}" ]; then OUT=$SW/eval/${RN}_evalsmoke; fi
LOG=$OUT/eval64k.log
VAL8=$DM_DATA_ROOT/nemotron_cc_v2d1_hq_dqa/nemotron_rs_val.bin; VAL64=${S2_DATA:-$DM_DATA_ROOT/prolong_64k_llama2}/prolong64k_val.bin
RD=$DM_DATA_ROOT/ruler-500-llama2
mkdir -p "$OUT"; say () { echo "[$(hostname -s) $(date +%T)] $*" | tee -a "$LOG"; }
# SIGTERM / SIGINT stop the evaluation, as in eval_arm.sh: the running benchmark's torchrun and its ranks are signalled and
# the script exits (143 / 130) once they are gone. Builtins only in the handler; a second signal is only counted.
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
[ -f "$HFDIR/model.safetensors" ] || { say "FATAL: no export at $HFDIR"; exit 2; }
[ -f "$VAL64" ] || { say "FATAL: 64k val missing at $VAL64"; exit 2; }
[ -f "$OUT/EVAL64K_COMPLETE" ] && { say "$RN 64k eval already complete"; exit 0; }
cd "$DM_REPO" || exit 2; export PYTHONPATH=$DM_REPO/src${PYTHONPATH:+:$PYTHONPATH}
# W&B off, as in eval_arm.sh: otherwise RULER and the lm-eval benchmarks open W&B runs of their own and MQAR logs into
# the training run; the results are the artifacts in $OUT
export WANDB_MODE=disabled
"$PY" -c "import json, sys; c = json.load(open(sys.argv[1])); m = [c.get('mixer')] + list(c.get('layer_mixers') or []); assert c.get('attn_impl') == 'sdpa' and not {'dm', 'mla_dm'} & set(m), (c.get('attn_impl'), c.get('mixer'), c.get('layer_mixers'))" "$HFDIR/config.json" \
  || { say "FATAL: export config not patched for the clean-kernel eval (attn_impl sdpa, no dm / mla_dm mixer; train_arm.sh does this at the end of a run)"; exit 2; }
"$PY" -c "import json,sys; c=json.load(open('$HFDIR/config.json')); rs=c.get('rope_scaling') or c.get('rope_parameters'); ok=bool(rs) and str(rs.get('type', rs.get('rope_type',''))).lower()=='yarn' and int(c.get('max_seq_len',0))>=65536; print('export: rope_scaling', rs, 'max_seq_len', c.get('max_seq_len')); sys.exit(0 if ok else 3)" | tee -a "$LOG" || { say "FATAL: the export is not a YaRN 64k model"; exit 2; }
valce_ok () {   # $1 = tag (64k|8k), $2 = windows requested, $3 = perwindow stem
  # Two checks, both load-bearing. (a) Window count: the scorer needs seq_len + 1 tokens per window, so a stream of
  # exactly W x seq_len tokens yields W - 1 windows (the 64k val set: 256 x 65,536 tokens -> 255 windows); the artifact
  # must carry the count the RESULT line reports, and that count must be at least the request minus one (an exact-256
  # check would leave every 64k eval unmarked). (b) Finiteness, as in eval_arm.sh: a diverged run writes a
  # shape-correct per-window array full of NaN/inf, which would otherwise pass as "done", get VALCE_COMPLETE /
  # EVAL64K_COMPLETE, and only surface in the paired analysis built from exactly this artifact.
  local txt=$OUT/valce${1}_n${2}.txt w
  w=$(grep -a "memmap-ppl RESULT" "$txt" 2>/dev/null | tail -n 1 | grep -oE "windows=[0-9]+" | cut -d= -f2)
  [ -n "$w" ] && [ "$w" -ge $(( $2 - 1 )) ] && "$PY" -c "import numpy as np,sys; a=np.load(sys.argv[1]); sys.exit(0 if a.ndim==2 and a.shape[0]==2 and a.shape[1]==int(sys.argv[2]) and np.isfinite(a[1]).all() else 1)" "$OUT/${3}_n${2}_perwindow.npy" "$w" 2>/dev/null; }
if valce_ok 64k "$NW64" "${RN}_64k"; then say "val CE 64k already done"; else
  say "1/8 val CE @ $NW64 windows of 65,536"; rm -f "$OUT/valce64k_n${NW64}.txt" "$OUT/${RN}_64k_n${NW64}_perwindow.npy"
  RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29820 + ${DM_VALCE_GPU:-0})) CUDA_VISIBLE_DEVICES=${DM_VALCE_GPU:-0} "$PY" -m evaluation.ppl.memmap_ppl --model_name_or_path "$HFDIR" --data "$VAL64" --dtype uint16 --seq_len 65536 \
      --max_windows "$NW64" --batch_size 1 --output_dir "$OUT" --run_name "${RN}_64k_n${NW64}" > "$OUT/valce64k_n${NW64}.txt" 2>&1 || say "  (val CE 64k rc=$?)"
fi
grep -a "memmap-ppl RESULT" "$OUT/valce64k_n${NW64}.txt" 2>/dev/null | tail -1 | tee -a "$LOG"
stop_if_signalled
if valce_ok "" "$NW" "$RN"; then say "val CE 8k already done"; else
  say "2/8 val CE @ $NW windows of 8,192 (Nemotron held-out)"; rm -f "$OUT/valce_n${NW}.txt" "$OUT/${RN}_n${NW}_perwindow.npy"
  RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=$((29820 + ${DM_VALCE_GPU:-0})) CUDA_VISIBLE_DEVICES=${DM_VALCE_GPU:-0} "$PY" -m evaluation.ppl.memmap_ppl --model_name_or_path "$HFDIR" --data "$VAL8" --dtype uint16 --seq_len 8192 \
      --max_windows "$NW" --batch_size 4 --output_dir "$OUT" --run_name "${RN}_n${NW}" > "$OUT/valce_n${NW}.txt" 2>&1 || say "  (val CE 8k rc=$?)"
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
run_cfg () {  # run_cfg <config name> <yaml path> <batch size> [engine]   (the form of eval_arm.sh)
  local name=$1 yaml=$2 bs=$3 engine=${4:-0} erc n
  stop_if_signalled
  if [ -f "$OUT/$name/${name}_results.complete" ]; then say "$name already done"; return 0; fi
  say "$name (bs=$bs engine=$engine)"
  if [ "$engine" = 1 ]; then export EVAL_ENGINE=mini_eval_dev ENGINE_MAX_SEQS=${DM_ENGINE_MAX_SEQS:-8}; else unset EVAL_ENGINE ENGINE_MAX_SEQS; fi
  local extra=(); [ "${DM_EVAL_NO_CACHE:-0}" = 1 ] && extra=(--no_cache_requests)
  "$DM_TORCHRUN" --standalone --nproc_per_node="$DM_NPROC" -m evaluation.meta_eval --model_name_or_path "$HFDIR" \
     --eval_config "$yaml" --output_dir "$OUT" --run_name "$RN" --eval_batch_size "$bs" "${extra[@]}" >> "$OUT/${name}.txt" 2>&1 & EPID=$!
  while :; do n=$STOP; wait "$EPID"; erc=$?; [ "$STOP" != "$n" ] && [ "$erc" -gt 128 ] && continue; break; done   # re-wait only after a trapped signal
  EPID=""; [ "$erc" -ne 0 ] && say "  ($name rc=$erc)"
  stop_if_signalled
  unset EVAL_ENGINE ENGINE_MAX_SEQS
  [ -f "$OUT/$name/${name}_results.complete" ] && say "  $name complete" || say "  $name NOT complete (see $OUT/${name}.txt)"
}
RULER_CFG=${S2_RULER_CFG:-src/evaluation/configs/ruler_llama2_64k.yaml}; MQAR_CFG=${S2_MQAR_CFG:-src/evaluation/configs/recall_synthetic.yaml}   # smoke: reduced copies (output under <run>_evalsmoke)
# the evaluator names its results by the YAML file name, so every config runs from a copy under its canonical name in $OUT
sed "s#^data_dir:.*#data_dir: $RD#" "$RULER_CFG" > "$OUT/ruler_llama2_64k.yaml"; cp "$MQAR_CFG" "$OUT/recall_synthetic.yaml"
run_cfg ruler_llama2_64k "$OUT/ruler_llama2_64k.yaml" 4 1
run_cfg commonsense src/evaluation/configs/commonsense.yaml 32
run_cfg recall_extraction src/evaluation/configs/recall_extraction.yaml 32 1
run_cfg triviaqa src/evaluation/configs/triviaqa.yaml 32 1
run_cfg recall_synthetic "$OUT/recall_synthetic.yaml" 16   # the YAML's batch_size (8) wins over this value
mqar_mark
if [ -d "$HF_HOME/datasets/cais___mmlu" ]; then run_cfg mmlu src/evaluation/configs/mmlu.yaml 16; else say "SKIP mmlu: cais___mmlu not in $HF_HOME/datasets"; fi
MISSING=(); valce_ok 64k "$NW64" "${RN}_64k" || MISSING+=(valce64k); valce_ok "" "$NW" "$RN" || MISSING+=(valce8k)
for c in ruler_llama2_64k commonsense recall_extraction triviaqa recall_synthetic mmlu; do [ -f "$OUT/$c/${c}_results.complete" ] || MISSING+=("$c"); done
say "tracebacks in the benchmark logs: $(cat "$OUT"/*.txt 2>/dev/null | grep -aci traceback)"
if [ ${#MISSING[@]} -eq 0 ]; then date '+%F %T' > "$OUT/EVAL64K_COMPLETE"; say "EVAL64K COMPLETE $RN ($OUT/EVAL64K_COMPLETE)"; exit 0
else say "EVAL64K INCOMPLETE $RN: missing ${MISSING[*]}"; exit 1; fi
