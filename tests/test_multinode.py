"""Multi-node training on the CPU: scripts/campaign/train_arm.sh with several nodes (NNODES > 1), the node launcher
scripts/campaign/node_ranks.sh and the shared collector scripts/campaign/ranks.sh, under a stand-in srun
(tests/_helpers.py FAKE_SRUN) that runs node_ranks.sh once per "node" on this host (SLURM_NODEID / SLURM_PROCID /
SLURMD_NODENAME fake-n0, fake-n1 set), ends the step when a task fails (kill-on-bad-exit: a catchable SIGTERM to every
task, SIGKILL after its KillWait), SIGKILLs its tasks and exits 137 when it gets a SIGTERM itself (what srun does on this
cluster), and exits with the first failing task's status. The stand-in ranks (FAKE_RANK per_rank / allreduce_stop)
emulate the trainer's MAX-all-reduced stop through a shared flag file: a stop that reaches one rank stops every rank of
every node at the same step.
  * the launch: one srun step (its options in one place; GPUs only when the job has some; DM_SRUN_EXTRA), global ranks
    node * DM_NPROC + r, MASTER_ADDR = the job's first node, the per-node status files and their summary;
  * a stop (SIGTERM to train_arm.sh alone, or to its whole process group as job.sh sends it) goes to the batch host's
    node launcher only, never to srun, and every node's ranks exit 75; a stop that comes before the launchers reported
    starts no rank (the stop file); a SIGUSR1, SIGHUP or SIGUSR2 sent to the srun step's tasks (`scancel --signal=<sig>
    <job>`: the node launchers; with --full train_arm.sh too) is a stop as well;
  * a rank failure on node 1: its peers are killed after DM_RANK_GRACE, node 1 exits non-zero, node 0 is ended by srun,
    the launch reports node 1's failure and blames node 1 alone in the chain's badnodes; of two nodes failing on their
    own the first one is blamed, of two failing together (within 2 s) both; a node whose launcher never reports (killed)
    is blamed when no node reported a failure time; a node killed after another one failed is not;
  * srun still running DM_SRUN_GRACE s after a stop (or DM_EXIT_GRACE s after a node's ranks ended) gets its step killed:
    srun found by its pid (also when setsid forked), or the step cancelled by id when srun ignores its SIGTERM;
  * compile caches restored from the chain's jit.tar on a node that does not have them, removed there at the end (and
    the job's /tmp scratch), never on the batch host; a stop during that restore starts no rank; a cache root other
    nodes may share gets a level per node; the status directories of old launches pruned;
  * configuration errors (a bad node count, multi-node without DM_LAUNCH=env, outside a Slurm job, without srun): 2;
  * one node: a failure in a chain names this host in badnodes; nothing after a success or a stop.
The one-node path itself is tested by test_driver_launch.py and test_launcher_signals.py (unchanged)."""
import os
import random
import re
import shutil
import signal
import subprocess
import tarfile
import time

import pytest

from _helpers import REPO, fake_launch_env, fake_multinode_env, read_calls

RN = "Q35M_clean_sdpa_ffn8_hd128_11785"            # clean_hd128, seed 11785
LAUNCHER = REPO / "scripts" / "campaign" / "train_arm.sh"
NODE_RANKS = REPO / "scripts" / "campaign" / "node_ranks.sh"


def _start(tmp_path, nnodes=2, job_id=4242, fork_setsid=False, **extra):
    """train_arm.sh clean_hd128 11785 as job `job_id` of `nnodes` fake nodes, in a session of its own."""
    env = fake_multinode_env(tmp_path, nnodes=nnodes, job_id=job_id, fork_setsid=fork_setsid, DM_NPROC=2, **extra)
    for k, v in extra.items():
        if v is None:
            env.pop(k, None)
    return subprocess.Popen(["bash", str(LAUNCHER), "clean_hd128", "11785"], cwd=str(REPO), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)


def _cleanup(tmp_path, p):
    """Whatever a failed test left: train_arm.sh's process group, the stand-in srun and its tasks."""
    for sig in (signal.SIGUSR1, signal.SIGKILL):   # USR1: the stand-in srun SIGKILLs its tasks
        try:
            pid = int((tmp_path / "fake" / "srun.pid").read_text())
            if "srun.py" in open(f"/proc/{pid}/cmdline").read():
                os.kill(pid, sig)
                time.sleep(0.2)
        except (OSError, ValueError):
            pass
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _finish(tmp_path, p, timeout=90):
    try:
        out, _ = p.communicate(timeout=timeout)
    finally:
        if p.poll() is None:
            _cleanup(tmp_path, p)
            p.communicate(timeout=30)
    return p.returncode, out


def _run(tmp_path, timeout=90, **extra):
    return _finish(tmp_path, _start(tmp_path, **extra), timeout)


def _wait(tmp_path, p, what, cond, timeout=60):
    t0 = time.time()
    while not cond(read_calls(tmp_path)):
        if p.poll() is not None or time.time() - t0 > timeout:
            _cleanup(tmp_path, p)
            pytest.fail(f"never: {what}\n{p.communicate(timeout=30)[0]}\n" + "\n".join(read_calls(tmp_path)))
        time.sleep(0.05)


def _ready(n):
    return lambda calls: sum(c.endswith(" ready") for c in calls) >= n


def _ranks(tmp_path):
    """{global rank: {mode, local, host, master, world, jit, tmpdir}} from the stand-in ranks' lines."""
    out = {}
    for c in read_calls(tmp_path):
        m = re.fullmatch(r"rank (\d+) mode (\S+) local (\S+) host (\S+) master (\S+) world (\S+) jit (\d) tmpdir (\d) ready", c)
        if m:
            out[int(m.group(1))] = dict(zip(("mode", "local", "host", "master", "world", "jit", "tmpdir"), m.groups()[1:]))
    return out


def _caches(tmp_path):
    """{global rank: (TRITON_CACHE_DIR, TILELANG_CACHE_DIR)} as each stand-in rank got them."""
    return {int(m.group(1)): (m.group(2), m.group(3)) for m in
            (re.fullmatch(r"rank (\d+) caches (\S+) (\S+)", c) for c in read_calls(tmp_path)) if m}


def _flag(tmp_path, name):
    """Make the stand-in ranks of mode failon:<name>:<rc> fail now."""
    (tmp_path / "fake" / name).write_text("")


def _status_dir(tmp_path):
    (d,) = [p for p in (tmp_path / "logs" / "nodes" / RN).iterdir() if p.name.startswith("4242.")]
    return d


def _reports(tmp_path):
    d = _status_dir(tmp_path)
    return {p.name: p.read_text().strip() for p in d.iterdir() if re.fullmatch(r"node\d+", p.name)}


def _chain(tmp_path):
    d = tmp_path / "chain"
    d.mkdir(exist_ok=True)
    return d


def _badnodes(tmp_path):
    f = tmp_path / "chain" / "badnodes"
    return f.read_text().splitlines() if f.exists() else []


def _hf(tmp_path):
    return tmp_path / "qwen35_m" / "hf" / RN


# ------------------------------------------------------------------------------------------------------------ the launch
def test_every_node_trains_and_the_run_completes(tmp_path):
    rc, out = _run(tmp_path, FAKE_MODE="per_rank", FAKE_RANK_MODES="ok", DM_CHAIN_DIR=_chain(tmp_path),
                   DM_MASTER_PORT=31000, TMPDIR=tmp_path / "scratch")   # TMPDIR: created by the launchers
    assert rc == 0 and (_hf(tmp_path) / "TRAIN_COMPLETE").exists(), out
    assert re.search(r"LAUNCH \S+ .* launch=env nodes=2 world=4 save_interval=", out), out
    ranks = _ranks(tmp_path)
    assert sorted(ranks) == [0, 1, 2, 3], ranks
    for r, d in ranks.items():   # global rank = node * DM_NPROC + local rank; rank 0 on the job's first node
        assert d == {"mode": "ok", "local": str(r % 2), "host": f"fake-n{r // 2}", "master": "fake-n0:31000",
                     "world": "4/2", "jit": "0", "tmpdir": "1"}, (r, d)
    (srun,) = [c.split() for c in read_calls(tmp_path) if c.startswith("srun --")]
    assert srun[1:11] == ["--nodes=2", "--ntasks=2", "--ntasks-per-node=1", "--kill-on-bad-exit=1", "--mpi=none",
                          "--cpu-bind=none", "--cpus-per-task=4", "--mem=0", "--export=ALL", "bash"], srun
    assert srun[11].endswith("scripts/campaign/node_ranks.sh") and srun[12] == RN, srun
    assert _reports(tmp_path) == {"node0": "0 fake-n0 -", "node1": "0 fake-n1 -"}, _reports(tmp_path)
    assert "node 0 (fake-n0): status 0" in out and "node 1 (fake-n1): status 0" in out, out
    log = (tmp_path / "logs" / f"{RN}.rank2.log").read_text()
    assert re.search(rf"^===== launch \S+ \S+ job 4242 host fake-n1 git fake: rank 2 of 4 of {RN} =====$", log, re.M), log
    assert _badnodes(tmp_path) == [] and "srun exit 0 tasks 0 0" in read_calls(tmp_path), out


@pytest.mark.parametrize("extra, want, nowant", [
    ({"SLURM_GPUS_ON_NODE": 8}, ["--gpus-per-node=2"], []),                     # DM_NPROC GPUs per node: the job has some
    ({"SLURM_GPUS_ON_NODE": 0, "DM_SRUN_EXTRA": "--label --network=x"}, ["--label", "--network=x"], ["--gpus-per-node"]),
    ({"SLURM_JOB_NUM_NODES": 3, "SLURM_JOB_NODELIST": "fake-n0,fake-n1,fake-n2", "DM_NNODES": 2},
     ["--nodelist=fake-n0,fake-n1"], ["--gpus-per-node"]),                       # fewer nodes than the job's: the first ones
])
def test_srun_options(tmp_path, extra, want, nowant):
    rc, out = _run(tmp_path, FAKE_MODE="per_rank", FAKE_RANK_MODES="ok", **extra)
    assert rc == 0, out
    (srun,) = [c.split() for c in read_calls(tmp_path) if c.startswith("srun --")]
    opts = srun[1:srun.index("bash")]
    assert all(w in opts for w in want) and not [w for w in opts for n in nowant if w.startswith(n)], opts
    if "DM_SRUN_EXTRA" in extra:
        assert opts[-2:] == want, opts   # DM_SRUN_EXTRA comes last


@pytest.mark.parametrize("extra", [{"SLURM_JOB_NUM_NODES": 1}, {"DM_NNODES": 1}])
def test_one_node_of_a_job_is_the_one_node_launch(tmp_path, extra):
    """One node (the job's, or DM_NNODES=1 in a bigger job): today's launch, no srun."""
    rc, out = _run(tmp_path, FAKE_MODE="complete", **extra)
    assert rc == 0 and "nodes=1 world=2" in out, out
    assert not [c for c in read_calls(tmp_path) if c.startswith("srun")] and len(
        [c for c in read_calls(tmp_path) if c.startswith("rank ")]) == 2, read_calls(tmp_path)


# -------------------------------------------------------------------------------------------------------------- stops
@pytest.mark.parametrize("batch", ["fake-n0", "fake-n1"])
@pytest.mark.parametrize("to", ["shell", "group"])
def test_a_stop_reaches_every_node_through_the_batch_hosts_launcher(tmp_path, to, batch):
    """The time limit's SIGTERM reaches train_arm.sh only (job.sh passes it on: to its whole process group). It goes
    to the node launcher of this host (normally node 0) and nowhere else, never to srun (in a session of its own):
    those ranks raise the trainer's stop flag, every rank of every node sees it at the same step, rank 0 saves, every
    rank exits 75; every launcher exits 0 (75 in its status file), srun 0, train_arm.sh 75; nothing is killed."""
    p = _start(tmp_path, FAKE_MODE="allreduce_stop", SLURMD_NODENAME=batch, DM_CHAIN_DIR=_chain(tmp_path))
    _wait(tmp_path, p, "4 ranks ready", _ready(4))
    (os.killpg if to == "group" else os.kill)(p.pid, signal.SIGTERM)
    rc, out = _finish(tmp_path, p)
    assert rc == 75 and "stopped early with a complete checkpoint" in out, out
    assert out.count("SIGTERM -> the node launcher on this host") == 1, out
    fwd = [l for l in out.splitlines() if "SIGTERM -> trainer ranks" in l]
    assert len(fwd) == 1 and fwd[0].startswith(f"[{batch} "), out   # only the batch host's launcher was signalled
    assert _reports(tmp_path) == {"node0": "75 fake-n0 -", "node1": "75 fake-n1 -"}, _reports(tmp_path)
    calls = read_calls(tmp_path)
    assert "srun exit 0 tasks 0 0" in calls and not [c for c in calls if "SIGTERM" in c and c.startswith("srun")], calls
    assert "SIGKILL" not in out and _badnodes(tmp_path) == [], out
    assert (tmp_path / "qwen35_m" / "ckpts" / RN / "checkpoint-5" / ".complete").exists()
    assert ("WARN: the job's first node is fake-n0, not this host (fake-n1)" in out) == (batch == "fake-n1"), out


def test_a_stop_before_the_launchers_reported_starts_no_rank(tmp_path):
    """The stop comes while srun is still starting the tasks (no launcher pid yet): the stop file tells every launcher
    not to start its ranks (status 143 recorded, exit 0); the launch ends with 143, nothing is blamed."""
    p = _start(tmp_path, FAKE_MODE="allreduce_stop", FAKE_SRUN_DELAY=3, DM_CHAIN_DIR=_chain(tmp_path))
    _wait(tmp_path, p, "srun started", lambda calls: any(c.startswith("srun --") for c in calls))
    os.kill(p.pid, signal.SIGTERM)
    rc, out = _finish(tmp_path, p)
    assert rc == 143 and "SIGTERM before the node launcher on this host reported" in out, out
    assert not [c for c in read_calls(tmp_path) if c.startswith("rank ")], read_calls(tmp_path)
    assert (_status_dir(tmp_path) / "stop").exists(), out
    assert _reports(tmp_path) == {"node0": "143 fake-n0 -", "node1": "143 fake-n1 -"}, _reports(tmp_path)
    assert out.count("a stop came before the ranks started: none started") == 2 and _badnodes(tmp_path) == [], out


@pytest.mark.parametrize("sig, full", [("SIGUSR1", False), ("SIGUSR1", True), ("SIGHUP", False), ("SIGUSR2", False),
                                       ("SIGTERM", True)])
def test_a_signal_sent_to_the_srun_step_is_a_stop(tmp_path, sig, full):
    """`scancel --signal=<sig> <job>` sends the signal to the tasks of the srun step, i.e. each node's launcher, not its
    ranks (proctrack/cgroup signals only slurmstepd's children unless the signal is SIGKILL: seen on this cluster, a
    SIGHUP the stand-in ranks do not handle left them running); with --full also to every process of the batch step
    (here: train_arm.sh's process group). The launchers and train_arm.sh take job.sh's stop signals (TERM INT HUP USR1
    USR2) as a stop and pass it on as SIGTERM: every rank saves, every node reports 75, srun is never signalled, the
    launch exits 75 and no node is blamed. Before, the launchers (and with --full train_arm.sh) died of it: exit 138, no
    report, every node blamed, and the ranks, which got no stop, were killed with the step. SIGTERM to both: the time
    limit (job.sh passes it to its process group, Slurm sends it to every task of the step) at once."""
    p = _start(tmp_path, FAKE_MODE="allreduce_stop", DM_CHAIN_DIR=_chain(tmp_path))
    _wait(tmp_path, p, "4 ranks ready", _ready(4))
    launchers = [int(f.read_text().split()[0]) for f in _status_dir(tmp_path).glob("node*.pid")]
    assert len(launchers) == 2, launchers
    if full:
        os.killpg(p.pid, getattr(signal, sig))
    for pid in launchers:
        os.kill(pid, getattr(signal, sig))
    rc, out = _finish(tmp_path, p)
    calls = read_calls(tmp_path)
    assert out.count("SIGTERM -> trainer ranks") == 2 and "no report" not in out, out   # both launchers passed it on
    assert rc == 75 and _reports(tmp_path) == {"node0": "75 fake-n0 -", "node1": "75 fake-n1 -"}, (_reports(tmp_path), out)
    assert "srun exit 0 tasks 0 0" in calls and _badnodes(tmp_path) == [], out
    assert out.count("SIGTERM -> the node launcher on this host") == (1 if full else 0), out
    assert (tmp_path / "qwen35_m" / "ckpts" / RN / "checkpoint-5" / ".complete").exists()


@pytest.mark.parametrize("how", ["srun", "forking setsid", "wedged srun"])
def test_srun_still_running_after_a_stop_gets_its_step_killed(tmp_path, how):
    """Ranks that never stop (blocked, ignoring SIGTERM) and launchers that would wait long (DM_TERM_GRACE=60):
    DM_SRUN_GRACE s after the stop train_arm.sh kills the step: a SIGTERM to srun (which SIGKILLs its tasks), srun found
    by its pid also when setsid forked, and scancel --signal=KILL <job>.<step> (enough for an srun that ignores its
    SIGTERM). The launch reports 143, no rank saved."""
    extra = {"FAKE_SRUN_IGNORE_TERM": 1} if how == "wedged srun" else {"FAKE_SCANCEL_NOOP": 1}
    p = _start(tmp_path, fork_setsid=(how == "forking setsid"), FAKE_MODE="per_rank", FAKE_RANK_MODES="hang",
               DM_TERM_GRACE=60, DM_SRUN_GRACE=2, **extra)
    _wait(tmp_path, p, "4 ranks ready", _ready(4))
    t0 = time.time()
    os.kill(p.pid, signal.SIGTERM)
    rc, out = _finish(tmp_path, p)
    calls = read_calls(tmp_path)
    assert rc == 143 and time.time() - t0 < 30, out
    assert "srun still running 2 s after the stop signal (DM_SRUN_GRACE): killing its step" in out, out
    assert "scancel --signal=KILL 4242.7" in calls, calls
    if how == "wedged srun":
        assert "srun: the step was cancelled: SIGKILL to every task" in calls, calls
    else:
        assert "srun: SIGTERM: forcing job termination" in calls, calls
    assert any(c.startswith("setsid (forks)") for c in calls) == (how == "forking setsid"), calls
    assert "node 0 (fake-n0): no report" in out and "node 1 (fake-n1): no report" in out, out


def test_a_node_stuck_after_another_ended_gets_the_step_killed(tmp_path):
    """Node 0's ranks finished (the export is whole) while every rank of node 1 hangs in its teardown: no rank of node 1
    exited, so its own DM_EXIT_GRACE never starts; train_arm.sh kills the step DM_EXIT_GRACE s after node 0 reported,
    and the run is complete."""
    rc, out = _run(tmp_path, FAKE_MODE="per_rank", FAKE_RANK_MODES="ok ok hang hang", DM_EXIT_GRACE=2)
    assert rc == 0 and (_hf(tmp_path) / "TRAIN_COMPLETE").exists(), out
    assert "2 s after node 0 reported that its ranks ended (status 0; DM_EXIT_GRACE): killing its step" in out, out
    assert "node 1 (fake-n1): no report" in out and "the trainer exited 143 after writing a whole export" in out, out


# ----------------------------------------------------------------------------------------------- failures and blame
def test_a_rank_failure_on_node_1_is_blamed_on_node_1(tmp_path):
    """Rank 3 (node 1) crashes while the others hang in a collective: node 1's launcher kills its other rank after
    DM_RANK_GRACE, prints rank 3's last 30 lines and exits 1; srun ends node 0's task (a catchable SIGTERM: its launcher
    passes it on and kills its blocked ranks DM_TERM_GRACE s later, status 143 without a failure time); the launch
    reports node 1's failure and badnodes names node 1's host only."""
    t0 = time.time()
    rc, out = _run(tmp_path, FAKE_MODE="per_rank", FAKE_RANK_MODES="hang hang hang fail", DM_RANK_GRACE=1,
                   DM_TERM_GRACE=1, DM_CHAIN_DIR=_chain(tmp_path))
    assert rc == 1 and time.time() - t0 < 60, out
    assert "rank 3 failed (exit 1): the other ranks get 1 s (DM_RANK_GRACE) to exit" in out, out
    assert "1 rank(s) still running 1 s after rank 3 failed (exit 1): SIGKILL" in out, out
    tail = out.split("--- rank 3 failed (exit 1)", 1)[1]
    assert "rank 3 traceback line 40" in tail and "rank 3 traceback line 11\n" in tail, tail
    assert "rank 3 traceback line 10\n" not in tail, tail
    assert "--- rank 0 failed" not in out and "--- rank 2 failed" not in out, out   # killed ranks: not their tails
    rep = _reports(tmp_path)
    assert rep["node0"] == "143 fake-n0 -" and re.fullmatch(r"1 fake-n1 \d+ 3", rep["node1"]), rep
    calls = read_calls(tmp_path)
    assert "srun: task 1 (fake-n1) exited 1: terminating the step" in calls and "srun exit 1 tasks 143 1" in calls, calls
    assert re.search(r"node 1 \(fake-n1\): status 1, first failure at \S+, failed global rank\(s\) 3", out), out
    (line,) = _badnodes(tmp_path)
    assert re.fullmatch(r"\d+ 4242 fake-n1", line), line


def test_of_two_nodes_failing_on_their_own_the_first_one_is_blamed(tmp_path):
    """Rank 3 (node 1) fails first (exit 7), rank 1 (node 0) 4 s later on its own (exit 9), no stop in between (each
    node's other rank hangs; DM_RANK_GRACE keeps node 1's launcher up meanwhile): both nodes report a failure time, the
    launch reports the first failure (7), and badnodes names node 1 alone."""
    p = _start(tmp_path, FAKE_MODE="per_rank", FAKE_RANK_MODES="hang failon:late:9 hang failon:first:7", DM_RANK_GRACE=8,
               DM_TERM_GRACE=1, DM_CHAIN_DIR=_chain(tmp_path))
    _wait(tmp_path, p, "4 ranks ready", _ready(4))
    _flag(tmp_path, "first")
    time.sleep(4)
    _flag(tmp_path, "late")
    rc, out = _finish(tmp_path, p)
    rep = _reports(tmp_path)
    m0, m1 = re.fullmatch(r"9 fake-n0 (\d+) 1", rep["node0"]), re.fullmatch(r"7 fake-n1 (\d+) 3", rep["node1"])
    assert m0 and m1 and int(m0[1]) - int(m1[1]) >= 3, (rep, out)
    assert rc == 7 and "2 nodes done: status 7 (srun exit 7)" in out, out
    (line,) = _badnodes(tmp_path)
    assert re.fullmatch(r"\d+ 4242 fake-n1", line), line


def _rank_children(pid):
    """The stand-in ranks a launcher started (its children running fake/rank.sh)."""
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as f:
            kids = [int(c) for c in f.read().split()]
    except OSError:
        return []
    out = []
    for k in kids:
        try:
            with open(f"/proc/{k}/cmdline", "rb") as f:
                if b"rank.sh" in f.read():
                    out.append(k)
        except OSError:
            pass
    return out


@pytest.mark.skipif(not os.path.exists(f"/proc/self/task/{os.getpid()}/children"),
                    reason="the kernel does not expose /proc/<pid>/task/<tid>/children")
@pytest.mark.parametrize("phase", ["free", "node 1 polls after the stop"])
def test_nodes_failing_together_are_all_blamed(tmp_path, phase):
    """Ranks 0 and 1 (node 0) and rank 2 (node 1) fail at the same moment (the NCCL watchdog's timeout around rank 3,
    which hangs on node 1: the culprit). Whichever launcher polls first, both nodes report failure times within 2 s of
    each other: both are blamed. "node 1 polls after the stop": node 1's launcher is frozen (SIGSTOP) right after it
    reaped rank 2 and thawed once srun's kill-on-bad-exit SIGTERM (node 0 failed) has reached it, so it collects rank 2
    only after that stop, as when node 0's launcher polls first: rank 2 was gone before the stop, a failure before it,
    and keeps a failure time (before, node 1 had none and node 0, healthy, was blamed alone)."""
    p = _start(tmp_path, FAKE_MODE="per_rank", FAKE_RANK_MODES="failon:now:134 failon:now:134 failon:now:134 hang",
               DM_RANK_GRACE=3, DM_TERM_GRACE=3, FAKE_SRUN_KILL_WAIT=8, DM_CHAIN_DIR=_chain(tmp_path))
    _wait(tmp_path, p, "4 ranks ready", _ready(4))
    launcher1 = int((_status_dir(tmp_path) / "node1.pid").read_text().split()[0])
    ranks1 = _rank_children(launcher1)
    assert len(ranks1) == 2, ranks1
    _flag(tmp_path, "now")
    if phase != "free":
        t0 = time.time()
        while all(os.path.exists(f"/proc/{r}") for r in ranks1):   # bash reaps an exiting child at once
            if time.time() - t0 > 30:
                _cleanup(tmp_path, p)
                pytest.fail("rank 2 never exited")
            time.sleep(0.002)
        os.kill(launcher1, signal.SIGSTOP)
        try:
            _wait(tmp_path, p, "srun ends node 1's task",
                  lambda calls: "srun: task 0 (fake-n0) exited 134: terminating the step" in calls)
        finally:
            try:
                os.kill(launcher1, signal.SIGCONT)
            except ProcessLookupError:
                pass
    rc, out = _finish(tmp_path, p)
    rep = _reports(tmp_path)
    assert re.fullmatch(r"134 fake-n0 \d+ 0 1", rep["node0"]) and re.fullmatch(r"134 fake-n1 \d+ 2", rep["node1"]), (rep, out)
    assert rc == 134 and sorted(l.split()[2] for l in _badnodes(tmp_path)) == ["fake-n0", "fake-n1"], out


@pytest.mark.parametrize("node0, node1, rc, blamed", [
    ("134 fake-n0 1000 0 1", "134 fake-n1 1001 2", 134, ["fake-n0", "fake-n1"]),   # 1 s apart: together
    ("9 fake-n0 1002 1", "7 fake-n1 1000 3", 7, ["fake-n0", "fake-n1"]),           # 2 s: still together; the first's status
    ("9 fake-n0 1003 1", "7 fake-n1 1000 3", 7, ["fake-n1"]),                      # 3 s: node 1 failed first
    ("143 fake-n0 -", "1 fake-n1 1000 3", 1, ["fake-n1"]),                         # node 0 only stopped after it
])
def test_failure_times_within_two_seconds_blame_every_such_node(tmp_path, node0, node1, rc, blamed):
    """The aggregation of given node reports (the stand-in srun writes them, FAKE_SRUN_REPORT_<k>): a node's failure
    time is when its launcher saw the failure (once a second) or got the stop; times within 2 s of the first count as
    the same moment."""
    got, out = _run(tmp_path, FAKE_SRUN_REPORT_0=node0, FAKE_SRUN_REPORT_1=node1, DM_CHAIN_DIR=_chain(tmp_path))
    assert got == rc and f"2 nodes done: status {rc} " in out, out
    assert sorted(l.split()[2] for l in _badnodes(tmp_path)) == blamed, out


def test_a_node_whose_launcher_never_reports_is_blamed(tmp_path):
    """Node 1's launcher is SIGKILLed (the OOM killer): it never reports; srun ends node 0's task, whose ranks are
    killed after the stop (143, no failure time). No node reported a failure time: the node that never reported is
    blamed."""
    rc, out = _run(tmp_path, FAKE_MODE="per_rank", FAKE_RANK_MODES="hang", FAKE_SRUN_KILL="1:2", FAKE_SRUN_KILL_WAIT=10,
                   DM_TERM_GRACE=1, DM_CHAIN_DIR=_chain(tmp_path))
    assert rc == 143, out
    assert _reports(tmp_path) == {"node0": "143 fake-n0 -"}, _reports(tmp_path)
    assert "node 1 (fake-n1): no report (its launcher was killed, or the node was lost)" in out, out
    (line,) = _badnodes(tmp_path)
    assert re.fullmatch(r"\d+ 4242 fake-n1", line), line


def test_a_node_killed_after_another_failed_is_not_blamed(tmp_path):
    """Node 1 fails (a failure time), node 0's launcher waits longer (DM_TERM_GRACE=30) than srun's KillWait and is
    SIGKILLed by srun without a report: node 1's failure is the launch's status and node 1 alone is blamed."""
    rc, out = _run(tmp_path, FAKE_MODE="per_rank", FAKE_RANK_MODES="hang hang hang fail", DM_RANK_GRACE=1,
                   DM_TERM_GRACE=30, FAKE_SRUN_KILL_WAIT=1, DM_CHAIN_DIR=_chain(tmp_path))
    assert rc == 1 and "node 0 (fake-n0): no report" in out, out
    assert list(_reports(tmp_path)) == ["node1"], _reports(tmp_path)
    (line,) = _badnodes(tmp_path)
    assert re.fullmatch(r"\d+ 4242 fake-n1", line), line


@pytest.mark.parametrize("mode, rc, blamed", [("fail", 1, True), ("stop", 75, False), ("complete", 0, False)])
def test_one_node_failure_in_a_chain_names_this_host(tmp_path, mode, rc, blamed):
    """One node: a failure (no stop) appends "<epoch> <job id> <this host>" to the chain's badnodes; not a success or a
    stop."""
    env = fake_launch_env(tmp_path, FAKE_MODE=mode, SLURM_JOB_ID=77, SLURMD_NODENAME="node7", DM_CHAIN_DIR=_chain(tmp_path))
    p = subprocess.run(["bash", str(LAUNCHER), "clean_hd128", "11785"], cwd=str(REPO), env=env, capture_output=True,
                       text=True, timeout=120)
    assert p.returncode == rc, p.stdout + p.stderr
    lines = _badnodes(tmp_path)
    if blamed:
        assert len(lines) == 1 and re.fullmatch(r"\d+ 77 node7", lines[0]), lines
    else:
        assert lines == [], lines


# ------------------------------------------------------------------------------------------- caches, scratch, pruning
@pytest.mark.parametrize("archive", ["good", "corrupt"])
def test_the_compile_caches_are_restored_on_a_node_without_them(tmp_path, archive):
    """job.sh restored the chain's caches on the batch host (DM_JIT_CACHE_DIR exists there); on node 1 (node-local
    paths: FAKE_SRUN_LOCAL_*) the launcher creates the directory, extracts the chain's jit.tar into it (a broken archive:
    a WARN, empty caches), and removes it at its end, with the job's scratch /tmp/dm-<job id>; the batch host's stay."""
    jid = random.randint(10 ** 8, 10 ** 9)
    scratch = f"/tmp/dm-{jid}"
    jit = tmp_path / "jit" / "dm-chain"
    jit.mkdir(parents=True)                       # the batch host's, restored by job.sh (here: empty)
    chain = _chain(tmp_path)
    if archive == "good":
        src = tmp_path / "jitsrc"
        for f in ("triton/r0/k.bin", "triton/r1/k.bin"):
            (src / f).parent.mkdir(parents=True, exist_ok=True)
            (src / f).write_text("kernel")
        with tarfile.open(chain / "jit.tar", "w") as t:
            t.add(src, arcname=".")
    else:
        (chain / "jit.tar").write_text("not a tar file")
    os.makedirs(scratch)                          # the site file's, on the batch host
    node1 = tmp_path / "nodes"
    try:
        p = _start(tmp_path, job_id=jid, FAKE_MODE="allreduce_stop", DM_CHAIN_DIR=chain, DM_JIT_CACHE_DIR=jit,
                   TRITON_CACHE_DIR=jit / "triton", TILELANG_CACHE_DIR=jit / "tilelang", TMPDIR=scratch,
                   FAKE_SRUN_LOCAL_ROOT=node1, FAKE_SRUN_LOCAL_VARS="DM_JIT_CACHE_DIR TRITON_CACHE_DIR TILELANG_CACHE_DIR")
        _wait(tmp_path, p, "4 ranks ready", _ready(4))
        (tmp_path / "stop.flag").write_text("")   # the trainer's stop (a deadline): every rank saves and exits 75
        rc, out = _finish(tmp_path, p)
        ranks = _ranks(tmp_path)
        assert rc == 75, out
        assert {r: (d["jit"], d["tmpdir"]) for r, d in ranks.items()} == \
            {0: ("0", "1"), 1: ("0", "1"), 2: ("1" if archive == "good" else "0", "1"),
             3: ("1" if archive == "good" else "0", "1")}, ranks
        n1 = f"{node1}/n1{jit}"   # node 1's cache directory, which its launcher created: node-local, jit.tar's layout
        assert _caches(tmp_path) == {0: (f"{jit}/triton/r0", f"{jit}/tilelang/r0"), 1: (f"{jit}/triton/r1", f"{jit}/tilelang/r1"),
                                     2: (f"{n1}/triton/r0", f"{n1}/tilelang/r0"), 3: (f"{n1}/triton/r1", f"{n1}/tilelang/r1")}, out
        restored = [l for l in out.splitlines() if "compile caches restored from" in l]
        if archive == "good":
            assert len(restored) == 1 and restored[0].startswith("[fake-n1 ") and f"{node1}/n1{jit}" in restored[0], out
        else:
            assert not restored and "WARN: restoring the compile caches" in out and "[fake-n1 " in out, out
        assert not os.path.exists(f"{node1}/n1{jit}"), "node 1's copy must go when its launcher ends"
        assert jit.is_dir(), "the batch host's caches are job.sh's"
        assert not os.path.exists(scratch), "node 1 (not the batch host) removes the job's /tmp scratch"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


@pytest.mark.parametrize("where", ["shared", "tmp"])
def test_a_compile_cache_root_other_nodes_may_share_gets_a_level_per_node(tmp_path, where):
    """TRITON_CACHE_DIR / TILELANG_CACHE_DIR on a file system every node sees (here: under tmp_path; env.sh's default is
    in the workspace): the batch host's ranks keep <root>/r<local rank>, node 1's get <root>/n1/r<local rank>, so no two
    ranks share a cache directory (one would read a metadata file another is still writing). A root under /tmp is
    node-local: <root>/r<local rank> on every node."""
    if where == "shared":
        if str(tmp_path).startswith(("/tmp/", "/dev/shm/")):
            pytest.skip("tmp_path is under /tmp or /dev/shm, which the launcher takes for node-local")
        root = tmp_path / "shared"
    else:
        root = f"/tmp/dm-test-cache-{random.randint(10 ** 8, 10 ** 9)}"   # never created: the stand-in ranks write nothing
    rc, out = _run(tmp_path, FAKE_MODE="per_rank", FAKE_RANK_MODES="ok", TRITON_CACHE_DIR=f"{root}/triton",
                   TILELANG_CACHE_DIR=f"{root}/tilelang")
    assert rc == 0, out
    sub = "n1/" if where == "shared" else ""
    assert _caches(tmp_path) == {0: (f"{root}/triton/r0", f"{root}/tilelang/r0"), 1: (f"{root}/triton/r1", f"{root}/tilelang/r1"),
                                 2: (f"{root}/triton/{sub}r0", f"{root}/tilelang/{sub}r0"),
                                 3: (f"{root}/triton/{sub}r1", f"{root}/tilelang/{sub}r1")}, out
    assert (f"TRITON_CACHE_DIR={root}/triton may be shared with the other nodes" in out) == (where == "shared"), out


def _node_launcher(tmp_path, node, **extra):
    """node_ranks.sh alone as node `node` (1 = not the batch host fake-n0) of job extra["SLURM_JOB_ID"], 2 ranks per
    node of the stand-in interpreter (FAKE_MODE per_rank ok), in a session of its own; extra PATH: put in front of the
    stand-ins' PATH. Returns (Popen, status dir)."""
    status = tmp_path / "status"
    status.mkdir()
    (tmp_path / "logs").mkdir()
    path = extra.pop("PATH", None)
    env = fake_launch_env(tmp_path, FAKE_MODE="per_rank", FAKE_RANK_MODES="ok", SLURM_NODEID=node,
                          SLURMD_NODENAME=f"fake-n{node}", DM_BATCH_HOST="fake-n0", DM_NODE_STATUS=status, DM_NPROC=2,
                          WORLD_SIZE=4, **extra)
    if path:
        env["PATH"] = f"{path}:{env['PATH']}"
    p = subprocess.Popen(["bash", str(NODE_RANKS), RN, str(tmp_path / "fake" / "python"), "-m", "pretrain.train",
                          "--run_name", RN, "--hf_save_dir", str(_hf(tmp_path)), "--ckpt_root_dir", str(tmp_path / "ck")],
                         cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    return p, status


@pytest.mark.parametrize("node", [0, 1])
def test_only_a_node_other_than_the_batch_host_removes_the_scratch_and_caches(tmp_path, node):
    """The job's /tmp/dm-<job id> (TMPDIR, the site file's) and the chain's compile-cache directory are job.sh's on the
    batch host: its node launcher leaves both (job.sh, the val CE and the next launch still use them). On another node
    the launcher removes the scratch and the cache directory it created."""
    jid = random.randint(10 ** 8, 10 ** 9)
    scratch, jit = f"/tmp/dm-{jid}", tmp_path / "jit" / "dm-chain"
    os.makedirs(scratch)
    if node == 0:
        jit.mkdir(parents=True)   # job.sh restored it on the batch host
    try:
        p, status = _node_launcher(tmp_path, node, SLURM_JOB_ID=jid, TMPDIR=scratch, DM_JIT_CACHE_DIR=jit)
        out, _ = p.communicate(timeout=120)
        assert p.returncode == 0 and (status / f"node{node}").read_text().split() == ["0", f"fake-n{node}", "-"], out
        assert os.path.isdir(scratch) == (node == 0) and jit.is_dir() == (node == 0), out
        assert sorted(_ranks(tmp_path)) == [2 * node, 2 * node + 1] and (_hf(tmp_path) / "config.json").exists() == (node == 0)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


SLOW_TAR = '#!/bin/bash\necho "slow tar $*" >> "$FAKE_CALLS"\nexec sleep 30\n'


def test_a_stop_during_the_cache_restore_starts_no_rank(tmp_path):
    """Slurm's SIGTERM (the time limit: every task of the step) reaches node 1's launcher while it restores the
    chain's jit.tar (a slow stand-in tar): the restore is cut short, no rank starts, "143 <host> -" is recorded and the
    launcher exits 0, leaves the stop file for the launchers that have not started their ranks yet, and removes the cache
    directory it created and the job's /tmp scratch."""
    jid = random.randint(10 ** 8, 10 ** 9)
    scratch, jit, chain = f"/tmp/dm-{jid}", tmp_path / "node1" / "dm-jit" / "dm-chain", _chain(tmp_path)
    (chain / "jit.tar").write_text("x")
    slow = tmp_path / "slowbin"
    slow.mkdir()
    (slow / "tar").write_text(SLOW_TAR)
    (slow / "tar").chmod(0o755)
    try:
        p, status = _node_launcher(tmp_path, 1, SLURM_JOB_ID=jid, TMPDIR=scratch, DM_JIT_CACHE_DIR=jit, DM_CHAIN_DIR=chain,
                                   PATH=slow)
        t0 = time.time()
        while not any(c.startswith("slow tar") for c in read_calls(tmp_path)):
            if p.poll() is not None or time.time() - t0 > 60:
                os.killpg(p.pid, signal.SIGKILL)
                pytest.fail(f"the restore never started:\n{p.communicate(timeout=30)[0]}")
            time.sleep(0.05)
        os.killpg(p.pid, signal.SIGTERM)
        out, _ = p.communicate(timeout=60)
        assert p.returncode == 0 and (status / "node1").read_text().strip() == "143 fake-n1 -", out
        assert "cut short by the stop" in out and "a stop came before the ranks started: none started" in out, out
        assert not [c for c in read_calls(tmp_path) if c.startswith("rank ")], read_calls(tmp_path)
        assert (status / "stop").exists() and not jit.exists() and not os.path.exists(scratch), out
        assert time.time() - t0 < 25, "the restore was waited for instead of stopped"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_old_status_dirs_are_pruned(tmp_path):
    """The newest 10 launches' status directories of the run are kept; nothing else there is touched."""
    base = tmp_path / "logs" / "nodes" / RN
    for k in range(12):
        d = base / f"4000.{1000 + k}.{k}"
        d.mkdir(parents=True)
        os.utime(d, (1000 + k, 1000 + k))
    (base / "notes").mkdir()
    rc, out = _run(tmp_path, FAKE_MODE="per_rank", FAKE_RANK_MODES="ok")
    assert rc == 0, out
    names = sorted(p.name for p in base.iterdir())
    launches = [n for n in names if re.fullmatch(r"\d+\.\d+\.\d+", n)]
    assert len(launches) == 10 and "notes" in names and any(n.startswith("4242.") for n in launches), names
    assert not {"4000.1000.0", "4000.1001.1", "4000.1002.2"} & set(names), names


# ---------------------------------------------------------------------------------------------- configuration errors
@pytest.mark.parametrize("extra, why", [
    ({"DM_NNODES": "two"}, "DM_NNODES='two' is not a whole number"),
    ({"DM_NNODES": "0"}, "DM_NNODES=0: at least one node"),
    ({"SLURM_JOB_NUM_NODES": "x2"}, "SLURM_JOB_NUM_NODES='x2' is not a whole number"),
    ({"DM_LAUNCH": "torchrun"}, "multi-node needs DM_LAUNCH=env"),
    ({"DM_LAUNCH": None}, "multi-node needs DM_LAUNCH=env"),
    ({"DM_NNODES": "3"}, "DM_NNODES=3 but job 4242 has 2 node(s)"),
    ({"SLURM_JOB_ID": None, "DM_NNODES": "2"}, "runs inside a Slurm job"),
    ({"PATH": "/usr/bin:/bin"}, "needs srun on PATH"),
])
def test_multinode_configuration_errors(tmp_path, extra, why):
    if extra.get("PATH") and shutil.which("srun", path=extra["PATH"]):
        pytest.skip("srun is on the plain PATH here")
    rc, out = _run(tmp_path, FAKE_MODE="per_rank", FAKE_RANK_MODES="ok", **extra)
    assert rc == 2 and why in out, out
    assert not [c for c in read_calls(tmp_path) if c.startswith(("srun", "rank "))], read_calls(tmp_path)


# ------------------------------------------------------------------------------------------- the node launcher alone
# a stand-in interpreter for node_ranks.sh run directly: every rank logs each SIGTERM it gets and exits 75 at the first;
# rank 0 sends a SIGTERM to its launcher as soon as its own handler is in place, while the launcher is still starting
# the other ranks; a rank that is never signalled runs out its time and logs "noterm"
STOP_DURING_START = r"""#!/bin/bash
trap 'echo "TERM $RANK" >> "$LOG"; exit 75' TERM
echo "start $RANK" >> "$LOG"
[ "$RANK" = 0 ] && kill -TERM "$PPID"
for i in $(seq 1 200); do sleep 0.05; done
echo "noterm $RANK" >> "$LOG"
exit 0
"""


def test_a_stop_while_the_ranks_start_reaches_every_rank(tmp_path):
    """A stop that reaches a node launcher while it is still starting its ranks (forward_term signals the ranks started
    so far) is passed to the ranks started after it too: no rank trains on."""
    py = tmp_path / "py.sh"
    py.write_text(STOP_DURING_START)
    py.chmod(0o755)
    status = tmp_path / "status"
    status.mkdir()
    (tmp_path / "logs").mkdir()
    env = fake_launch_env(tmp_path, SLURM_NODEID=0, SLURMD_NODENAME="fake-n0", DM_BATCH_HOST="fake-n0", SLURM_JOB_ID=4242,
                          DM_NODE_STATUS=status, DM_NPROC=6, WORLD_SIZE=6, LOG=tmp_path / "ranks.log", DM_TERM_GRACE=30)
    p = subprocess.run(["bash", str(REPO / "scripts" / "campaign" / "node_ranks.sh"), RN, str(py), "-m", "pretrain.train"],
                       cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120, start_new_session=True)
    log = (tmp_path / "ranks.log").read_text().splitlines()
    assert not [l for l in log if l.startswith("noterm")], log
    assert p.returncode == 0 and (status / "node0").read_text().split()[:3] == ["75", "fake-n0", "-"], p.stdout + p.stderr
    assert p.stdout.count("SIGTERM -> trainer ranks") + p.stdout.count("SIGTERM -> no rank") == 1, p.stdout
    assert [l for l in log if l.startswith("TERM ")].count("TERM 0") == 1, log   # rank 0: once (a repeated signal: counted)
