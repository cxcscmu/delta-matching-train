"""The MMLU accuracy (%) overall and by group (humanities, STEM, social sciences, other) from a harness results JSON.

  usage: mmlu_groups.py <harness results json> <label>
         e.g. mmlu_groups.py $DM_WORKSPACE/qwen35_m/eval/<run>/mmlu/full_results/results.json <run>
"""
import json, sys
if len(sys.argv) != 3:
    print(__doc__); sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
d = json.load(open(sys.argv[1]))
out = {}
def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items(): walk(v, path + "/" + k)
    elif isinstance(o, (int, float)) and path.endswith(("acc,none", "/acc")) and "stderr" not in path:
        parts = path.split("/")
        name = [p for p in parts if p.startswith("mmlu")]
        if name and name[-1] in ("mmlu", "mmlu_humanities", "mmlu_stem", "mmlu_social_sciences", "mmlu_other"):
            out[name[-1]] = round(float(o) * 100, 2)
walk(d)
print(sys.argv[2], " ".join(f"{k}={out[k]}" for k in ("mmlu", "mmlu_humanities", "mmlu_stem", "mmlu_social_sciences", "mmlu_other") if k in out) or f"keys seen: {list(d.get('results', d).keys())[:6]}")
