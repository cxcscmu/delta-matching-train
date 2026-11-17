"""scripts/slurm/job.sh and submit.sh for chains of multi-node jobs (README there), under the fake Slurm of
test_slurm_wrappers.py: after a failure without progress the successor is kept off the hosts that
scripts/campaign/train_arm.sh blamed the failure on in the chain's `badnodes` during this job (its lines with this job's
id since this run of the job began; each host once; the successor's own exclusions kept), and when there are none off
every node of this job (several nodes: scontrol show hostnames $SLURM_JOB_NODELIST; one node: this node, as before);
DM_JIT_CACHE_DIR names the chain's compile-cache directory for the launchers of the other nodes (a derived variable:
never inherited, never passed to the successor); a new submission clears `badnodes`; -N is replayed by every link."""
import os
import shlex
import subprocess
import time

import pytest

from test_slurm_wrappers import (CHAIN, OPTS, FakeSlurm, SUBMIT, _dry, _dry_chain, _dry_opts, _env, _fake_slurm, _link,
                                 _sbatch_words)

# `scontrol show hostnames <list>` (the comma-separated list, one name per line; FAKE_SHOW_FAIL=1: it fails) in front of
# the fake Slurm's scontrol, which records every other call (update)
SHOW = r"""#!/bin/bash
if [ "${1:-}" = show ] && [ "${2:-}" = hostnames ]; then
  [ "${FAKE_SHOW_FAIL:-0}" = 1 ] && exit 1
  tr ',' '\n' <<< "${3:-}"; exit 0
fi
exec %s "$@"
"""
BLAME = 'echo "$(date +%s) $SLURM_JOB_ID {host}" >> "$DM_CHAIN_DIR/badnodes"'   # what train_arm.sh appends


def _chain_dir(tmp_path):
    return tmp_path / "ws" / "logs" / "slurm" / "chains" / CHAIN


def _path(tmp_path, fake):
    d = tmp_path / "show"
    d.mkdir(exist_ok=True)
    (d / "scontrol").write_text(SHOW % shlex.quote(str(fake.bin / "scontrol")))
    (d / "scontrol").chmod(0o755)
    return f"{d}:{fake.bin}:{os.environ['PATH']}"


def _failed_link(tmp_path, blamed=(), nodes="", given="", progress=0, **extra):
    """job 99 on SLURMD_NODENAME node7 (of `nodes`, when given: a multi-node job), its command failing (exit 1) after
    appending a badnodes line for each host of `blamed`; no retry in place."""
    fake = _fake_slurm(tmp_path)
    if nodes:
        extra.update(SLURM_JOB_NUM_NODES=len(nodes.split(",")), SLURM_JOB_NODELIST=nodes)
    return _link(tmp_path, (1,), (progress,), fake=fake, body="\n".join(BLAME.format(host=h) for h in blamed),
                 DM_INPLACE_RETRIES=0, DM_SBATCH_OPTS=f"{OPTS} {given}".strip(), PATH=_path(tmp_path, fake), **extra)


def _updates(r):
    return [c["argv"] for c in r.fake.calls("scontrol")]


@pytest.mark.parametrize("blamed, nodes, given, want", [
    (["node8"], "node7,node8", "", "node8"),                        # the node train_arm.sh blamed, only it
    (["node8", "node8"], "node7,node8", "-x node1", "node1,node8"),  # once; the successor's own exclusions kept
    (["node7", "node8"], "node7,node8", "", "node7,node8"),          # both blamed (their failure times tied)
    ([], "node7,node8", "", "node7,node8"),                          # none blamed: every node of the job
    (["node7"], "", "", "node7"),                                     # one node: train_arm.sh blames this host
    ([], "", "", "node7"),                                            # one node, none blamed: this node (as before)
])
def test_link_keeps_the_successor_off_the_blamed_nodes(tmp_path, blamed, nodes, given, want):
    r = _failed_link(tmp_path, blamed, nodes, given)
    assert r.rc == 1 and r.fake.job(4242)["state"] == "PENDING", r.out
    assert _updates(r) == [["update", "JobId=4242", f"ExcNodeList={want}"]] and r.fake.job(4242)["exc"] == want, r.out
    added = [h for h in want.split(",") if h not in given]
    said = f"excluded node {added[0]} " if len(added) == 1 else f"excluded nodes {' '.join(added)} "
    assert said + "from the successor 4242" in r.out, r.out
    assert ("train_arm.sh blamed the failure on it" in r.out) == bool(blamed), r.out


def test_link_counts_only_this_runs_badnodes(tmp_path):
    """Lines of another job, and of this job id before this run of the job began (a requeue reuses the id), do not
    count: nothing blamed in this run, so every node of the job is excluded."""
    d = _chain_dir(tmp_path)
    d.mkdir(parents=True)
    (d / "badnodes").write_text(f"1000 99 node9\n{int(time.time())} 41 node3\nnot a line\n")
    r = _failed_link(tmp_path, nodes="node7,node8")
    assert r.rc == 1 and _updates(r) == [["update", "JobId=4242", "ExcNodeList=node7,node8"]], r.out


def test_link_already_excluded_nodes_are_not_updated(tmp_path):
    r = _failed_link(tmp_path, ["node8"], "node7,node8", "--exclude=node8")
    assert r.rc == 1 and _updates(r) == [] and r.fake.job(4242)["exc"] == "node8", r.out


def test_link_excludes_nothing_after_a_failure_with_progress(tmp_path):
    r = _failed_link(tmp_path, ["node8"], "node7,node8", progress=1)
    assert r.rc == 1 and _updates(r) == [] and "excluded" not in r.out, r.out


def test_link_that_cannot_list_its_nodes_excludes_this_node(tmp_path):
    r = _failed_link(tmp_path, nodes="node7,node8", FAKE_SHOW_FAIL=1)
    assert r.rc == 1 and _updates(r) == [["update", "JobId=4242", "ExcNodeList=node7"]], r.out
    assert "WARN: cannot list the nodes of this job" in r.out, r.out


SHOW_JIT = 'echo "JITDIR=${DM_JIT_CACHE_DIR:-unset} DERIVED=${DM_DERIVED:-none}"'


def test_link_names_the_compile_cache_dir_for_the_other_nodes(tmp_path):
    """DM_JIT_CACHE_DIR = the chain's node-local cache directory (job.sh restored it on this host): a derived variable,
    so the successor (and a submit.sh run from inside the link) derives its own."""
    r = _link(tmp_path, (75,), (1,), body=SHOW_JIT)
    assert r.rc == 75 and f"JITDIR={tmp_path / 'jit' / CHAIN} " in r.out, r.out
    derived = r.out.split("DERIVED=", 1)[1].splitlines()[0].split()
    assert "DM_JIT_CACHE_DIR" in derived and "TRITON_CACHE_DIR" in derived, derived
    assert "DM_JIT_CACHE_DIR" not in r.fake.job(4242)["env"], r.fake.job(4242)["env"]


@pytest.mark.parametrize("extra", [{"DM_JIT_CACHE": 0}, {"DM_RESUBMIT": 0}])
def test_link_without_compile_caches_names_no_cache_dir(tmp_path, extra):
    """Caches off, or a single job: no DM_JIT_CACHE_DIR, also when the submitting shell had one."""
    r = _link(tmp_path, (75,), (1,), body=SHOW_JIT, DM_JIT_CACHE_DIR="/stale/from/a/shell", **extra)
    assert r.rc == 75 and "JITDIR=unset" in r.out, r.out


def test_submit_clears_badnodes_and_replays_the_node_count(tmp_path):
    """A new submission starts with no blamed nodes (as with its failure records); -N 2 goes to every link."""
    fake = FakeSlurm(tmp_path)
    args = ["-N", "2", "-t", "4:00:00", "--", "scripts/campaign/train_arm.sh", "x"]
    dry = _dry(tmp_path, *args)
    assert dry.returncode == 0, dry.stderr
    assert "-N" in _dry_opts(dry.stdout) and "--nodes=1" not in _sbatch_words(dry.stdout), dry.stdout
    d = tmp_path / "ws" / "logs" / "slurm" / "chains" / _dry_chain(dry.stdout)[0]
    d.mkdir(parents=True)
    (d / "badnodes").write_text("1 41 node8\n")
    (d / "progress").write_text("1 41 ckpt RUN 100\n")
    p = subprocess.run(["bash", str(SUBMIT), *args], capture_output=True, text=True, timeout=60, cwd=tmp_path,
                       env=_env(tmp_path, PATH=f"{fake.bin}:{os.environ['PATH']}"))
    assert p.returncode == 0, p.stdout + p.stderr
    assert not (d / "badnodes").exists() and (d / "progress").exists(), list(d.iterdir())
    (job,) = fake.jobs()
    assert "-N" in job["env"]["DM_SBATCH_OPTS"].split() and "2" in job["env"]["DM_SBATCH_OPTS"].split(), job["env"]
