"""Paired per-window val-CE contrast between any two runs, by run name.

Companion to paired_analysis.py (several runs against one baseline); both align window ids and drop and count
non-finite differences. Each .npy is np.stack([window_ids_sorted, nll_sorted]): ROW 0 = window ids, ROW 1 = NLL.
By default every val set that has an artifact for both runs is reported: the 8k one (<run>_n2000_perwindow.npy) and,
for stage-2 runs, the 64k one (<run>_64k_n256_perwindow.npy).

  usage: python3 scripts/campaign/paired_pair.py [--val 8k|64k] [--windows N] [--dir DIR] <run_A> <run_B> [<run_A2> <run_B2> ...]
         (A - B, so a positive mean means A is WORSE than B)
"""
import argparse
import sys

from paired_analysis import D, VALSETS, finite_mean, load, paired, row


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run names in pairs: <run_A> <run_B> ...")
    ap.add_argument("--val", choices=sorted(VALSETS), default=None,
                    help="one val set only (default: every val set both runs have an artifact for)")
    ap.add_argument("--windows", type=int, default=None,
                    help="with --val: the window count in the artifact name (default 2000 at 8k, 256 at 64k)")
    ap.add_argument("--dir", default=D, help="the artifacts: a flat directory or the eval tree (default: DM_PERWIN_DIR, "
                                             "else $DM_WORKSPACE/qwen35_m/eval)")
    a = ap.parse_args(argv)
    if len(a.runs) % 2:
        ap.error("run names come in pairs: <run_A> <run_B> [<run_A2> <run_B2> ...]")
    if a.windows and not a.val:
        ap.error("--windows needs --val")
    sets = [a.val] if a.val else ["8k", "64k"]
    loaded = {v: load(v, a.windows, a.dir) for v in sets}
    rc = 0
    for i in range(0, len(a.runs), 2):
        A, B = a.runs[i], a.runs[i + 1]
        found = False
        for v in sets:
            arms = loaded[v]
            if A in arms and B in arms:
                found = True
                print(f"  [{v}] mean NLL  {A} = {finite_mean(arms[A][1])[0]:.4f}   {B} = {finite_mean(arms[B][1])[0]:.4f}")
                row(f"{A} - {B}", paired(arms, A, B))
        if not found:
            rc = 2
            print(f"MISSING: no val set ({', '.join(sets)}) has a per-window artifact for both {A} and {B}")
            have = sorted(set().union(*(loaded[v] for v in sets)))
            print("have: " + ("\n      ".join(have) or "(none under " + a.dir + ")"))
        print()
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
