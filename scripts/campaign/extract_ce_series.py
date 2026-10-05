"""Extract the per-step train CE series from a trainer log into a plot-ready TSV.

Trainer logs append across relaunches and resumes, so one file can hold several SEGMENTS
(scratch restarts and checkpoint resumes, which REPLAY the steps after the checkpoint). A plot
that ignores this stitches together trajectories that never coexisted. Segments are detected by
the step number going backwards; every row carries its segment id, and the header names each
segment's range so the plotting side can choose (for the final trajectory: keep, for each step,
the row from the LAST segment containing it; for a death story: plot segments separately).

nan/inf CE rows are KEPT verbatim: for the stale arms the non-finite stretch IS the result.

usage: extract_ce_series.py <trainer_log> <out_tsv> [<comment>]
"""
import re
import sys
import time

if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
    print(__doc__); sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)
src, out = sys.argv[1], sys.argv[2]
comment = sys.argv[3] if len(sys.argv) > 3 else ""
t = open(src, errors="ignore").read()
rows = re.findall(
    r"\[step (\d+)\] \{'train/ce_loss': ([^,]+),.*?'train/learning_rate': ([0-9.e+-]+|nan)",
    t, re.S)
if not rows:  # older logs may order keys differently; fall back to ce only
    rows = [(s, c, "") for s, c in re.findall(r"\[step (\d+)\] \{'train/ce_loss': ([^,]+),", t)]

seg, prev = 0, -1
recs = []
for s, c, *lr in rows:
    s = int(s)
    if s < prev:
        seg += 1
    prev = s
    recs.append((seg, s, c.strip(), (lr[0].strip() if lr and lr[0] else "")))

segs = {}
for g, s, c, l in recs:
    a, b = segs.get(g, (s, s))
    segs[g] = (min(a, s), max(b, s))

with open(out, "w") as f:
    f.write(f"# source: {src}\n# extracted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    if comment:
        f.write(f"# {comment}\n")
    f.write(f"# segments: {', '.join(f'{g}:[{a}..{b}]' for g, (a, b) in sorted(segs.items()))}\n")
    f.write("# columns: segment\tstep\ttrain_ce\tlearning_rate\n")
    for g, s, c, l in recs:
        f.write(f"{g}\t{s}\t{c}\t{l}\n")
print(f"{out}: {len(recs)} rows, {len(segs)} segment(s): "
      + ", ".join(f"{g}:[{a}..{b}]" for g, (a, b) in sorted(segs.items())))
