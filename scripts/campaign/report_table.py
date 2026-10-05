#!/usr/bin/env python3
"""report_table.py: the experiments' results as mean +/- error over seeds, for every cell, arm and metric.

    python scripts/campaign/report_table.py                      # every cell, every metric
    python scripts/campaign/report_table.py --cell HD128         # one cell (the list: --help)
    python scripts/campaign/report_table.py --per-seed           # also print the raw per-seed values
    python scripts/campaign/report_table.py --markdown           # GitHub-flavoured tables

Runs are recognised by the names scripts/campaign/arms.sh gives them, <arm part>_<seed>[<TAG>]: the arm from the name
(clean, stale = _dm_stale_, match = _dm_matched_, consistent = _dm_consistent_, te_fp8 = _te_fp8_), the seed from the
first all-digit field, the cell from the prefix and the tag (_nope, _muon, _s2_64k). Any other tag keeps a run out of
the tables. Stage-1 cells read eval_arm.sh's artifacts; the stage-2 cell reads eval_arm_64k.sh's.

The error bar is the standard error of the mean over seeds. With n = 2 that is

    sd  = |a - b| / sqrt(2)          (sample sd, ddof=1)
    sem = sd / sqrt(2) = |a - b| / 2

so for two seeds the SEM and the half-range coincide exactly. With n = 1 the value is printed bare and flagged (n=1):
a single seed has no error bar. n >= 3 uses the ordinary sd/sqrt(n).

IMPORTANT for reading the output: at n = 2 the SEM is a two-point spread, not a confidence interval. A difference smaller
than the error bars is not a result. A diverged run's non-finite val CE prints as nan.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

W = os.environ.get("DM_WORKSPACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "workspace"))
EVAL = os.path.join(W, "qwen35_m", "eval")

# metric key -> (label, source, json key, decimals); source = a benchmark directory, or a val-CE text file
METRICS_S1 = [
    ("valce",             "val CE",     "valce_n2000.txt",    "",              4),
    ("ruler",             "RULER",      "ruler_llama2_8k",    "avg/full",      2),
    ("commonsense",       "CSense",     "commonsense",        "avg",           4),
    ("recall_extraction", "Extract",    "recall_extraction",  "avg",           4),
    ("triviaqa",          "TriviaQA",   "triviaqa",           "avg",           4),
    ("mmlu",              "MMLU",       "mmlu",               "avg",           4),
    ("mqar",              "MQAR",       "recall_synthetic",   "mean_accuracy", 4),
]
METRICS_S2 = [
    ("valce64k",          "val CE 64k", "valce64k_n256.txt",  "",              4),
    ("valce",             "val CE 8k",  "valce_n2000.txt",    "",              4),
    ("ruler",             "RULER",      "ruler_llama2_64k",   "avg/full",      2),
    ("commonsense",       "CSense",     "commonsense",        "avg",           4),
    ("recall_extraction", "Extract",    "recall_extraction",  "avg",           4),
    ("triviaqa",          "TriviaQA",   "triviaqa",           "avg",           4),
    ("mmlu",              "MMLU",       "mmlu",               "avg",           4),
    ("mqar",              "MQAR",       "recall_synthetic",   "mean_accuracy", 4),
]

# cell key -> the model it stands for, in print order. Run names come from scripts/campaign/arms.sh.
CELLS = {"HD128": "GDN : GQA, head_dim 128, 1.665 B", "NOPE": "the head_dim-128 cell without positional encoding",
         "MUON": "the head_dim-128 cell trained with Muon", "HD256_XS": "GDN : GQA, head_dim 256, 0.569 B",
         "HD256_M": "GDN : GQA, head_dim 256, 1.665 B", "HD256_L": "GDN : GQA, head_dim 256, 5.289 B",
         "MLA": "GDN : MLA, 1.665 B", "KDA": "KDA : GQA, 1.665 B", "M2": "Mamba2 : GQA, 1.665 B",
         "GQA": "pure GQA, 1.475 B", "S2": "stage 2: the head_dim-128 cell extended to 64k"}
ARMS = ("clean", "match", "stale", "consistent", "te_fp8")   # print order
# <arm part>_<seed><TAG>: the seed is the first all-digit field (ffn8, hd128 and qkvo8 are not all digits)
_NAME = re.compile(r"^(?P<stem>.+?)_(?P<seed>\d+)(?P<tag>(?:_.*)?)$")


def split_name(rn: str):
    """(arm part, seed, tag) of a run name, or None when it carries no seed field."""
    m = _NAME.match(rn)
    return (m["stem"], m["seed"], m["tag"]) if m else None


def cell_of(rn: str) -> str | None:
    """The cell of a run, or None for a run outside the experiments' cells."""
    parts = split_name(rn)
    if parts is None:
        return None
    stem, _seed, tag = parts
    hd128_main = stem.startswith("Q35M_") and stem.endswith("_hd128")
    if tag in ("_nope", "_muon", "_s2_64k"):
        return {"_nope": "NOPE", "_muon": "MUON", "_s2_64k": "S2"}[tag] if hd128_main else None
    if tag:
        return None
    for prefix, cell in (("Q35M2_", "M2"), ("Q35MLA_", "MLA"), ("Q35KDA_", "KDA"), ("Q3M_", "GQA"),
                         ("Q35XS_", "HD256_XS"), ("Q35L_", "HD256_L")):
        if stem.startswith(prefix):
            return cell
    if stem.startswith("Q35M_"):
        return "HD128" if stem.endswith("_hd128") else "HD256_M"
    return None


def arm_of(rn: str) -> str | None:
    for key, arm in (("_dm_matched_", "match"), ("_dm_stale_", "stale"), ("_dm_consistent_", "consistent"),
                     ("_te_fp8_", "te_fp8"), ("_clean_", "clean")):
        if key in rn:
            return arm
    return None


def seed_of(rn: str) -> str:
    parts = split_name(rn)
    return parts[1] if parts else "?"


def read_run(rn: str, metrics) -> dict:
    d = os.path.join(EVAL, rn)
    out = {}
    for key, _lbl, src, jkey, _dp in metrics:
        if src.endswith(".txt"):
            try:
                txt = open(f"{d}/{src}", errors="ignore").read()
                out[key] = float(re.findall(r"nll/tok=([0-9.]+|nan|-?inf)", txt)[-1])   # nan / inf: a diverged run
            except Exception:
                out[key] = None
            continue
        try:
            r = json.load(open(f"{d}/{src}/{src}_results.json"))
            out[key] = r.get("results", r).get(jkey)
        except Exception:
            out[key] = None
    out["complete"] = os.path.exists(f"{d}/EVAL_COMPLETE") or os.path.exists(f"{d}/EVAL64K_COMPLETE")
    return out


def agg(vals):
    """(mean, sem, n). sem is None when n < 2. For n == 2, sem == |a-b|/2 == half-range."""
    v = [x for x in vals if isinstance(x, (int, float))]
    if not v:
        return None, None, 0
    n = len(v)
    m = sum(v) / n
    if n < 2:
        return m, None, 1
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / (n - 1))
    return m, sd / math.sqrt(n), n


def fmt(m, sem, n, dp):
    if m is None:
        return "-"
    if n == 1:
        return f"{m:.{dp}f} (n=1)"
    return f"{m:.{dp}f} ± {sem:.{dp}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", choices=list(CELLS), default=None)
    ap.add_argument("--per-seed", action="store_true")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args()

    runs = {}
    for d in sorted(glob.glob(f"{EVAL}/Q3*/")):
        rn = os.path.basename(d.rstrip("/"))
        cell = cell_of(rn)
        if cell is None or arm_of(rn) is None:
            continue
        runs[rn] = read_run(rn, METRICS_S2 if cell == "S2" else METRICS_S1)

    cells = [a.cell] if a.cell else list(CELLS)
    sep = " | " if a.markdown else "  "
    for cell in cells:
        metrics = METRICS_S2 if cell == "S2" else METRICS_S1
        arms = [x for x in ARMS if any(cell_of(r) == cell and arm_of(r) == x for r in runs)]
        if not arms:
            continue
        print(f"\n### {cell} ({CELLS[cell]}): mean ± SEM over seeds (n=2: SEM == half-range)\n" if a.markdown
              else f"\n=== {cell} ({CELLS[cell]}): mean +/- SEM over seeds (n=2: SEM == half-range) ===")
        labels = [lbl for _k, lbl, _s, _j, _d in metrics]
        if a.markdown:
            print("| arm | n | " + " | ".join(labels) + " |")
            print("|---" * (len(labels) + 2) + "|")
        else:
            print(f"{'arm':10}{sep}{'n':>2}{sep}" + sep.join(f"{lb:>16}" for lb in labels))
        for arm in arms:
            rs = [r for r in runs if cell_of(r) == cell and arm_of(r) == arm]
            cellvals = []
            for key, _lbl, _s, _j, dp in metrics:
                m, s, n = agg([runs[r][key] for r in rs])
                cellvals.append(fmt(m, s, n, dp))
            if a.markdown:
                print(f"| {arm} | {len(rs)} | " + " | ".join(cellvals) + " |")
            else:
                print(f"{arm:10}{sep}{len(rs):>2}{sep}" + sep.join(f"{c:>16}" for c in cellvals))
        if a.per_seed:
            print("  per-seed:" if not a.markdown else "\nper-seed:\n")
            for arm in arms:
                for r in sorted([x for x in runs if cell_of(x) == cell and arm_of(x) == arm],
                                key=lambda x: (seed_of(x), x)):   # as a string: 11785 before 2
                    v = runs[r]
                    s = "  ".join(f"{lbl}={v[k]:.4f}" if isinstance(v.get(k), (int, float)) else f"{lbl}=-"
                                 for k, lbl, _s, _j, _d in metrics)
                    print(f"    {arm:10} seed {seed_of(r):5} {s}   {'complete' if v['complete'] else 'partial'}")
    print("\nError bar = SEM over seeds. At n=2 it equals half the seed spread, so a difference smaller",
          "\nthan the bars is not separable; treat it as noise rather than a result.")


if __name__ == "__main__":
    main()
