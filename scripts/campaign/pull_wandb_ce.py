"""Recover per-step train CE for runs whose node-local logs are gone, from wandb.

Two wandb facts shape this: the trainer logs metrics with
step=global_step, so the wandb step axis IS the training step; and relaunches create NEW wandb
runs under the SAME display name, so one training run can be several wandb runs. All matching
runs are pulled, ordered by createdAt, and emitted as segments in that order.

usage: pull_wandb_ce.py <run_display_name> <out_tsv> [<comment>]
"""
import os
import sys
import time

if len(sys.argv) < 3 or sys.argv[1] in ("-h", "--help"):
    print(__doc__); sys.exit(0 if sys.argv[1:2] in (["-h"], ["--help"]) else 2)

import wandb

name, out = sys.argv[1], sys.argv[2]
comment = sys.argv[3] if len(sys.argv) > 3 else ""
api = wandb.Api(timeout=60)
# WANDB_ENTITY / WANDB_PROJECT name the wandb project the trainer logged to (entity: the account's default when unset)
project = os.environ.get("WANDB_PROJECT", "delta-matching")
path = f"{os.environ['WANDB_ENTITY']}/{project}" if os.environ.get("WANDB_ENTITY") else project
runs = [r for r in api.runs(path, filters={"display_name": name})]
runs.sort(key=lambda r: r.created_at)
print(f"{name}: {len(runs)} wandb run(s): "
      + ", ".join(f"{r.id}@{r.created_at}" for r in runs))
recs = []
for seg, r in enumerate(runs):
    n = 0
    for row in r.scan_history(page_size=2000):
        step = row.get("_step")
        ce = row.get("train/ce_loss")
        if step is None or ce is None:
            continue
        recs.append((seg, int(step), ce))
        n += 1
    print(f"  seg{seg} ({r.id}): {n} rows")
segs = {}
for g, s, c in recs:
    a, b = segs.get(g, (s, s))
    segs[g] = (min(a, s), max(b, s))
with open(out, "w") as f:
    f.write(f"# source: wandb {path} display_name={name} "
            f"({len(runs)} run(s): {', '.join(r.id for r in runs)})\n")
    f.write(f"# extracted: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    if comment:
        f.write(f"# {comment}\n")
    f.write(f"# segments: {', '.join(f'{g}:[{a}..{b}]' for g, (a, b) in sorted(segs.items()))}\n")
    f.write("# columns: segment\tstep\ttrain_ce\n")
    for g, s, c in recs:
        f.write(f"{g}\t{s}\t{c}\n")
print(f"{out}: {len(recs)} rows, {len(segs)} segment(s)")
