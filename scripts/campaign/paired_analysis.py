"""Paired per-window val-CE analysis: runs contrasted with a baseline run on the same windows.

WHY PAIRED. The marginal SE printed by memmap_ppl is ~0.0068 at n=2000 -- that is the spread of the val set itself,
not the resolution of an A/B contrast. Every arm is scored on the SAME windows, and cross-arm per-window NLL correlates
at r ~ 0.9996, so the difference vector d_i = nll_A[i] - nll_B[i] has sd ~40x smaller than either arm's own sd. The
paired SE = sd(d)/sqrt(n) is the number that governs whether a delta is real (docs/benchmark.md, "Means, uncertainty,
units").

Each .npy is np.stack([window_ids_sorted, nll_sorted]) written by evaluation/ppl/memmap_ppl.py:
<run>_n<W>_perwindow.npy for the 8k val set (eval_arm.sh / eval_arm_64k.sh, W = 2000) and <run>_64k_n<W>_perwindow.npy
for the 64k one (eval_arm_64k.sh, W = 256 requested, 255 windows scored). Windows are aligned by id; a window whose
difference is not finite (a diverged run) is dropped and counted. paired_pair.py shares load() / paired() / row().

    python scripts/campaign/paired_analysis.py <baseline run> <run> [<run> ...]              # each run - baseline, 8k
    python scripts/campaign/paired_analysis.py --val 64k <baseline run> <run> [<run> ...]    # the 64k val set
"""
import argparse
import glob
import math
import os
import sys

import numpy as np

# The per-window artifacts: a flat directory (DM_PERWIN_DIR) or the eval tree $DM_WORKSPACE/qwen35_m/eval/<run>/
D = os.environ.get("DM_PERWIN_DIR", os.path.join(os.environ.get("DM_WORKSPACE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "workspace")), "qwen35_m", "eval"))
VALSETS = {"8k": ("", 2000), "64k": ("_64k", 256)}   # val set -> (infix in the artifact name, window count requested)


def load(val="8k", windows=None, root=None):
    """{run name: (window ids, NLL)} for every per-window artifact of one val set under root (flat or the eval tree)."""
    infix, default_w = VALSETS[val]
    tail = f"{infix}_n{windows or default_w}_perwindow.npy"
    root = root or D
    out = {}
    for p in sorted(set(glob.glob(os.path.join(root, "*" + tail)) + glob.glob(os.path.join(root, "*", "*" + tail)))):
        name = os.path.basename(p)[: -len(tail)]
        if val == "8k" and name.endswith("_64k"):
            continue   # a 64k artifact with the same window count in its name
        if os.path.basename(os.path.dirname(p)).endswith("_evalsmoke"):
            continue   # eval_arm_64k.sh's reduced-config output: same file names as the run's own artifacts
        a = np.load(p)
        out[name] = (a[0].astype(np.int64), a[1].astype(np.float64))
    return out


def paired(arms, A, B):
    """Paired difference A - B on the common window ids; windows with a non-finite difference are dropped and counted."""
    ia, na = arms[A]
    ib, nb = arms[B]
    if not np.array_equal(ia, ib):
        _common, xa, xb = np.intersect1d(ia, ib, return_indices=True)
        na, nb = na[xa], nb[xb]
    d = na - nb
    ok = np.isfinite(d)
    dropped = int(d.size - ok.sum())
    d, na, nb = d[ok], na[ok], nb[ok]
    n = int(d.size)
    nan = float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = float(d.mean()) if n else nan
        se = float(d.std(ddof=1)) / math.sqrt(n) if n > 1 else nan
        r = float(np.corrcoef(na, nb)[0, 1]) if n > 1 else nan
    t = mean / se if se > 0 else nan
    # sign test: fraction of windows where A is worse than B
    frac = float((d > 0).mean()) if n else nan
    return dict(n=n, dropped=dropped, mean=mean, se=se, t=t, r=r, frac_worse=frac,
                lo=mean - 1.96 * se, hi=mean + 1.96 * se)


def row(label, s):
    star = "***" if abs(s["t"]) > 3 else ("*" if abs(s["t"]) > 2 else "")
    print(f"  {label:34s} {s['mean']:+9.5f}  +/-{1.96*s['se']:.5f}  "
          f"SE={s['se']:.5f}  t={s['t']:+8.2f} {star:3s} "
          f"worse in {100*s['frac_worse']:5.1f}% of windows  r={s['r']:.4f}  n={s['n']} dropped={s['dropped']}")


def finite_mean(nll):
    """(mean, marginal SE, finite windows, non-finite windows) of one run's per-window NLL."""
    f = np.isfinite(nll)
    k = int(f.sum())
    mean = float(nll[f].mean()) if k else float("nan")
    se = float(nll[f].std(ddof=1)) / math.sqrt(k) if k > 1 else float("nan")
    return mean, se, k, int(nll.size - k)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="the baseline run first, then the runs to contrast with it (run - baseline)")
    ap.add_argument("--val", choices=sorted(VALSETS), default="8k", help="the val set (default 8k)")
    ap.add_argument("--windows", type=int, default=None,
                    help="the window count in the artifact name (default 2000 at 8k, 256 at 64k)")
    ap.add_argument("--dir", default=D, help="the artifacts: a flat directory or the eval tree (default: DM_PERWIN_DIR, "
                                             "else $DM_WORKSPACE/qwen35_m/eval)")
    a = ap.parse_args(argv)
    arms = load(a.val, a.windows, a.dir)
    missing = [r for r in a.runs if r not in arms]
    if missing:
        print(f"MISSING {a.val} per-window artifact for: " + ", ".join(missing))
        print("have: " + ("\n      ".join(sorted(arms)) or "(none under " + a.dir + ")"))
        return 2
    base = a.runs[0]
    print(f"{a.val} val set: {len(a.runs)} runs, baseline {base}\n")
    print("ABSOLUTE val CE (mean NLL over the finite windows; marginal SE)")
    for k in a.runs:
        mean, se, fin, bad = finite_mean(arms[k][1])
        print(f"  {k:34s} {mean:.5f}   (marginal SE {se:.5f}, {fin} windows" + (f", {bad} non-finite" if bad else "") + ")")
    if len(a.runs) > 1:
        print("\nPAIRED run - baseline (negative = the run is better; +/- = 1.96 SE for display, SE = 1 SE)")
        for k in a.runs[1:]:
            row(f"{k} - {base}", paired(arms, k, base))
    return 0


if __name__ == "__main__":
    sys.exit(main())
