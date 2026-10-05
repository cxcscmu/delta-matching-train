"""Aggregate the full benchmark suite for a set of runs into report-ready rows.

WHY A SCRIPT AND NOT HAND-COPYING. Every arm has 7 benchmark results, commonsense alone has 9 subtasks,
and the report requires mean +/- one standard error ACROSS SEEDS for each (SE = s.d. / sqrt(n), docs/benchmark.md,
"Means, uncertainty, units"). Transcribing that by hand across ~20 runs is where a digit gets transposed into a
headline table. This reads the artifacts the evals wrote and prints rows in the report's column order.

  usage: collect_results.py <run_name> [more...]      (the eval tree: $DM_WORKSPACE/qwen35_m/eval, or DM_EVAL_ROOT)
"""

import json, math, os, re, sys, statistics as st

BUCKET = os.environ.get("DM_EVAL_ROOT", os.path.join(os.environ.get("DM_WORKSPACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "workspace")), "qwen35_m", "eval"))
SHORT = {"lambada_openai":"lambada","hellaswag":"hella","piqa":"piqa",
         "arc_easy":"arc-e","sciq":"sciq","openbookqa":"obqa",
         "winogrande":"wino","copa":"copa","arc_challenge":"arc-c"}
CS9 = ["lambada_openai","hellaswag","piqa","arc_easy","sciq","openbookqa","winogrande","copa","arc_challenge"]

def cat(path):
    try:
        return open(path).read()
    except OSError:
        return None

def jload(path):
    s = cat(path)
    if not s: return None
    try: return json.loads(s)
    except Exception: return None

def metrics(run):
    m = {}
    v = cat(f"{BUCKET}/{run}/valce_n2000.txt")
    if v:
        hit = [l for l in v.splitlines() if "memmap-ppl RESULT" in l]
        f = re.search(r"nll/tok=([0-9.]+|nan|-?inf)", hit[-1]) if hit else None   # nan / inf: a diverged run
        if f: m["valCE"] = float(f.group(1))
    d = jload(f"{BUCKET}/{run}/ruler_llama2_8k/ruler_llama2_8k_results.json")
    if d:
        r = d.get("results", d)
        m["ruler4k"], m["ruler8k"] = r.get("avg/4096"), r.get("avg/8192")
    for cfg, key in (("recall_extraction","extract"), ("mmlu","MMLU")):
        d = jload(f"{BUCKET}/{run}/{cfg}/{cfg}_results.json")
        if d:
            r = d.get("results", d)
            m[key] = r.get("avg")
    # MQAR does NOT use the "avg" key the other configs use: recall_synthetic writes a
    # kv-pairs x seq-len grid and reports `mean_accuracy`. Reading .get("avg") returns None and the
    # column silently prints "-", which reads as "not run" rather than "read with the wrong key".
    d = jload(f"{BUCKET}/{run}/recall_synthetic/recall_synthetic_results.json")
    if d:
        r = d.get("results", d)
        m["MQAR"] = r.get("mean_accuracy", r.get("avg"))
    d = jload(f"{BUCKET}/{run}/triviaqa/triviaqa_results.json")
    if d:
        r = d.get("results", d); m["triviaqa"] = r.get("avg")
    d = jload(f"{BUCKET}/{run}/commonsense/commonsense_results.json")
    if d:
        r = d.get("results", d); vb = r.get("verbose", {})
        m["cs_avg"] = r.get("avg")
        for t in CS9:
            if t in vb: m["cs_"+t] = vb[t]
    return m

def fmt(vals, nd=4, scale=1.0):
    """mean +/- one standard error (s.d. / sqrt(n)); a single value is printed bare (no error bar)."""
    vals = [v*scale for v in vals if v is not None]
    if not vals: return "-"
    bad = sum(not math.isfinite(v) for v in vals)
    if bad: return f"nan ({bad} of {len(vals)} non-finite)"
    if len(vals) == 1: return f"{vals[0]:.{nd}f}"
    return f"{st.mean(vals):.{nd}f} ± {st.stdev(vals) / math.sqrt(len(vals)):.{nd}f}"

if __name__ == "__main__":
    runs = sys.argv[1:]
    if not runs or any(r.startswith("-") for r in runs):
        print(__doc__); sys.exit(0 if runs[:1] in (["-h"], ["--help"]) else 2)
    per = {r: metrics(r) for r in runs}
    for r in runs:
        m = per[r]
        print(f"{r:44s} valCE={m.get('valCE')} 4k={m.get('ruler4k')} 8k={m.get('ruler8k')} "
              f"cs={m.get('cs_avg')} ext={m.get('extract')} mqar={m.get('MQAR')} mmlu={m.get('MMLU')} tqa={m.get('triviaqa')}")
    print("\n--- AGGREGATE over the runs given (treated as seeds of one arm): mean ± 1 SE ---")
    def col(k): return [per[r].get(k) for r in runs]
    print(f"n={len(runs)}  valCE={fmt(col('valCE'))}  RULER4k={fmt(col('ruler4k'),2)}  "
          f"RULER8k={fmt(col('ruler8k'),2)}  extract={fmt(col('extract'))}  "
          f"MQAR%={fmt(col('MQAR'),2,100)}  MMLU%={fmt(col('MMLU'),2,100)}  cs_avg%={fmt(col('cs_avg'),2,100)}  TriviaQA%={fmt(col('triviaqa'),2,100)}")
    print("commonsense-9 %: " + " | ".join(f"{SHORT[t]}={fmt(col('cs_'+t),1,100)}" for t in CS9))
