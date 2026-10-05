#!/usr/bin/env python3
"""Download the evaluation data of the protocol (docs/benchmark.md) into the data root:

    python data/prepare_eval.py --root $DM_DATA_ROOT            # both parts
    python data/prepare_eval.py --root $DM_DATA_ROOT --only ruler

  ruler     <root>/ruler-500-llama2/: RULER regenerated with the Llama-2 tokenizer (13 tasks x 500 examples at 2K to
            128K; docs/benchmark.md section 3), the Hugging Face dataset tturing/ruler-500-llama2 at a pinned revision
            (7.95 GB), checked against data/checksums.json.
  datasets  <root>/hf_cache/ (the HF_HOME of the evaluation): the datasets of the lm-eval benchmarks the protocol runs
            (the nine commonsense tasks, SWDE / FDA / SQuAD completion, TriviaQA, MMLU), downloaded by lm-eval's own
            task definitions, so the versions are the ones the pinned lm-eval reads. MMLU and MQAR later run offline
            from this cache (scripts/campaign/eval_arm.sh). MQAR needs no data (it is generated).

Each dataset stays under its own license (see its dataset card). Needs network access; run it once per data root."""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verify import check_files, load_checksums  # noqa: E402

RULER_REPO, RULER_REVISION, RULER_DIR = "tturing/ruler-500-llama2", "09a858d8bb8d38fbe32920857fb884b303fdf978", "ruler-500-llama2"
TASKS = ["lambada_openai", "hellaswag", "piqa", "arc_easy", "arc_challenge", "sciq", "openbookqa", "winogrande", "copa",
         "swde", "fda", "squad_completion", "triviaqa", "mmlu"]


def ruler(root):
    from huggingface_hub import snapshot_download
    out = os.path.join(root, RULER_DIR)
    snapshot_download(RULER_REPO, repo_type="dataset", revision=RULER_REVISION, local_dir=out)
    print(f"[ruler] {RULER_REPO}@{RULER_REVISION[:12]} -> {out}")
    return check_files(root, {k: v for k, v in load_checksums().items() if k.startswith(RULER_DIR + "/")}, hashes=True)


def datasets(root):
    os.environ["HF_HOME"] = os.path.join(root, "hf_cache")          # before datasets / lm_eval are imported
    os.environ.pop("HF_HUB_OFFLINE", None); os.environ.pop("HF_DATASETS_OFFLINE", None)
    from lm_eval.tasks import TaskManager, get_task_dict
    tm = TaskManager(include_path=os.path.join(os.path.dirname(HERE), "src", "evaluation", "lm_eval_tasks"))
    ok = True
    for t in TASKS:   # one at a time, so a failure names its task
        try:
            get_task_dict([t], tm)
            print(f"[datasets] {t}: cached under {os.environ['HF_HOME']}")
        except Exception as e:   # noqa: BLE001
            ok = False
            print(f"[datasets] {t}: FAILED {type(e).__name__}: {e}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the data root (DM_DATA_ROOT)")
    ap.add_argument("--only", choices=["ruler", "datasets"], default=None)
    a = ap.parse_args()
    ok = True
    if a.only in (None, "ruler"):
        ok &= ruler(a.root)
    if a.only in (None, "datasets"):
        ok &= datasets(a.root)
    print("[prepare_eval] " + ("OK" if ok else "FAILED (see above)"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
