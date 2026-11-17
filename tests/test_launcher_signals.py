"""The launchers' SIGTERM forwarding and rank-status collection (scripts/campaign/ranks.sh, sourced by train_arm.sh and
node_ranks.sh), without a scheduler or a GPU.

The block between the `# >>> signal forwarding` / `# <<< signal forwarding` markers of ranks.sh is run on its own
around stand-in ranks (bash loops that behave like the trainer: on SIGTERM they "save" briefly and exit 75). What must
hold when a job scheduler stops the job and SIGTERM arrives twice within milliseconds (scripts/slurm/job.sh, or a
scheduler, signals every process of the job, then a wrapper script's own trap forwards it again):
  * the shell never aborts (the old handler ran $(hostname -s) $(date) inside the trap, and under bash 5.2 the second
    signal re-entered the parser: "trap: line 2: unexpected EOF ..." / "parse_and_execute: bad jump", exit 134);
  * the ranks are signalled exactly once (the old handler forwarded on every signal) and the launcher reports their 75;
  * a rank that exits before its turn in the collection loop is still collected;
  * under torchrun (a stand-in that starts the ranks) the signal reaches torchrun's children, not torchrun, unless no
    rank has started yet.
And what the collector (wait_ranks) adds, so that one dead rank never leaves the others blocked in a collective until
the 30-minute process-group timeout (or past Slurm's KillWait, which ends a chain without a word):
  * a rank that fails (an exit other than 0 and 75) with no stop requested: the others are SIGKILLed DM_RANK_GRACE
    seconds later; the launcher reports that first failure, never the 137 of its own kill;
  * under Slurm (SLURM_JOB_ID set) ranks still running DM_TERM_GRACE seconds after a forwarded SIGTERM are SIGKILLed
    (143 unless a rank saved and exited 75); outside Slurm they are left to finish their save;
  * a rank still running DM_EXIT_GRACE seconds after another one exited cleanly (0 or 75: the run's last collective is
    over) is stuck in its teardown: SIGKILLed (143, or the 75 that was saved);
  * a 75 from any rank wins;
  * a rank that failed before a stop signal but is collected after it (the collector polls once a second) is a failure
    before the stop: its failure time is kept (T_FAIL: the node's blame in a multi-node launch) and the others get
    DM_RANK_GRACE; unless it died of the stop signal itself (143).
The container's bash is 5.2 while a login node may have 5.1, and only 5.2 reproduces the abort: set DM_TEST_BASH to a
second interpreter (e.g. one extracted from the training image) to run every case under it as well.
"""
import os
import re
import signal
import subprocess
import time

import pytest

from _helpers import REPO

LAUNCHER = REPO / "scripts/campaign/ranks.sh"
BASHES = ["/bin/bash"] + ([os.environ["DM_TEST_BASH"]] if os.environ.get("DM_TEST_BASH") else [])

RANK = r"""
case "$1" in
  save) trap 'sleep 0.5; exit 75' TERM; : > "$READY_DIR/$$"; while :; do sleep 0.02; done ;;   # the trainer: checkpoint, exit 75
  exit:*) : > "$READY_DIR/$$"; IFS=: read -r _ code delay <<< "$1"; sleep "$delay"; exit "$code" ;;
  hang) trap 'echo "rank $$: SIGTERM pending (blocked in a collective)"' TERM; : > "$READY_DIR/$$"; while :; do sleep 0.05; done ;;
  kill:*) : > "$READY_DIR/$$"; IFS=: read -r _ delay <<< "$1"; sleep "$delay"; kill -KILL $$ ;;   # the host's OOM killer
  # exits <code> after <delay> s; a helper then sends SIGTERM to the launcher as soon as the launcher has reaped this
  # rank (`kill -0` fails), i.e. well before its next once-a-second poll collects it
  failsig:*) : > "$READY_DIR/$$"; IFS=: read -r _ code delay <<< "$1"; sleep "$delay"
     ( me=$$; while kill -0 "$me" 2>/dev/null; do sleep 0.01; done; kill -TERM "$PPID" ) & exit "$code" ;;
esac
"""


def _block() -> str:
    m = re.search(r"^# >>> signal forwarding.*?^# <<< signal forwarding[^\n]*\n", LAUNCHER.read_text(), re.S | re.M)
    assert m, "ranks.sh lost its signal-forwarding markers (this test runs that block on its own)"
    return m.group(0)


@pytest.fixture
def harness(tmp_path):
    (tmp_path / "rank.sh").write_text(RANK)
    h = tmp_path / "harness.sh"
    h.write_text(_block() + f"""
export READY_DIR=$(mktemp -d)
for m in "$@"; do /bin/bash {tmp_path / 'rank.sh'} "$m" & PIDS+=($!); done
trap forward_term TERM INT
until [ "$(ls "$READY_DIR" | wc -l)" -ge $# ]; do sleep 0.01; done; rm -rf "$READY_DIR"   # every rank has its trap
echo READY
wait_ranks
echo "T_FAIL=${{T_FAIL:-none}}"
echo "RC=$RC"; exit $RC
""")
    return h


def _launch(bash, harness, modes, **env):
    """Start the harness around stand-in ranks; env: extra variables (DM_RANK_GRACE, SLURM_JOB_ID, ...). The test's
    own SLURM_* never leak in: under Slurm the term grace would apply to every case."""
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "DM_"))}
    e.update({k: str(v) for k, v in env.items()})
    p = subprocess.Popen([bash, str(harness), *modes], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True, env=e)   # its own process group, so a signal can reach the ranks too
    assert p.stdout.readline().strip() == "READY"   # the trap is installed: signalling now is deterministic
    return p


def _finish(p, timeout=30):
    """Wait for the launcher, then kill whatever is left of its process group BEFORE reading its output: a stand-in
    rank that outlives the launcher (a broken forward, e.g. torchrun signalled instead of its ranks) holds the pipe
    open, and reading first would hang the test instead of failing it."""
    try:
        p.wait(timeout=timeout)
    finally:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return p.returncode, p.stdout.read()


# A stand-in torchrun (DM_LAUNCH=torchrun): it starts the ranks after DELAY seconds, reports it if it is signalled itself,
# and exits as torchrun does (1 when a rank failed, 0 otherwise). The delay is a `read -t` on a private FIFO, not a
# `sleep`: until the ranks start torchrun must have no child process at all.
TORCHRUN = r"""
trap 'echo TORCHRUN_SIGNALLED; exit 143' TERM
F=$(mktemp -u); mkfifo "$F"; exec 3<>"$F"; rm -f "$F"
: > "$READY_DIR/torchrun"
read -t "$DELAY" -u 3 || true
for m in "$@"; do /bin/bash "$RANK_SH" "$m" & PIDS+=($!); done
rc=0; for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
exit $rc
"""


@pytest.fixture
def torchrun_harness(tmp_path):
    (tmp_path / "rank.sh").write_text(RANK)
    (tmp_path / "torchrun.sh").write_text(TORCHRUN)
    h = tmp_path / "harness_torchrun.sh"
    h.write_text(_block() + f"""
export READY_DIR=$(mktemp -d) RANK_SH={tmp_path / 'rank.sh'}
WANT=$1; shift
/bin/bash {tmp_path / 'torchrun.sh'} "$@" & TR_PID=$!
trap forward_term TERM INT
until [ "$(ls "$READY_DIR" | wc -l)" -ge "$WANT" ]; do sleep 0.01; done   # torchrun (and WANT - 1 ranks) are up
echo READY
wait_torchrun
rm -rf "$READY_DIR"; echo "RC=$RC"; exit $RC
""")
    return h


def _has_proc_children():
    return os.path.exists(f"/proc/self/task/{os.getpid()}/children")


@pytest.mark.skipif(not _has_proc_children(), reason="the kernel does not expose /proc/<pid>/task/<tid>/children")
@pytest.mark.parametrize("bash", BASHES)
def test_torchrun_launch_forwards_to_the_ranks_not_torchrun(bash, torchrun_harness):
    """Under torchrun the SIGTERM goes to torchrun's children (read from /proc when it arrives), never to torchrun itself
    (it would kill the ranks 30 s later, before a checkpoint can finish); a second signal is only counted. The ranks
    save and exit 75, torchrun reports its usual 1 (train_arm.sh turns that into 75 from the trainer log)."""
    for i in range(4):
        env = dict(os.environ, DELAY="0")
        p = subprocess.Popen([bash, str(torchrun_harness), "5", *["save"] * 4], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, start_new_session=True, env=env)
        assert p.stdout.readline().strip() == "READY"
        os.kill(p.pid, signal.SIGTERM)
        time.sleep(0.001 * (1 + i))
        os.kill(p.pid, signal.SIGTERM)
        rc, out = _finish(p)
        assert "bad jump" not in out and "unexpected EOF" not in out, f"{bash} iteration {i}: the trap aborted the shell:\n{out}"
        assert out.count("SIGTERM -> trainer ranks") == 1 and "TORCHRUN_SIGNALLED" not in out, f"{bash} iteration {i}:\n{out}"
        assert rc == 1 and "RC=1" in out, f"{bash} iteration {i}: exit {rc}, expected torchrun's 1 after the ranks' 75:\n{out}"


@pytest.mark.skipif(not _has_proc_children(), reason="the kernel does not expose /proc/<pid>/task/<tid>/children")
@pytest.mark.parametrize("bash", BASHES)
def test_torchrun_launch_before_any_rank_signals_torchrun(bash, torchrun_harness):
    """A SIGTERM that arrives before torchrun has started a rank goes to torchrun itself, which stops before any rank
    has trained; the launcher reports the signal."""
    env = dict(os.environ, DELAY="5")
    p = subprocess.Popen([bash, str(torchrun_harness), "1", "save", "save"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, start_new_session=True, env=env)
    assert p.stdout.readline().strip() == "READY"
    os.kill(p.pid, signal.SIGTERM)
    rc, out = _finish(p)
    assert "SIGTERM -> torchrun (no rank started yet)" in out and "TORCHRUN_SIGNALLED" in out, out
    assert rc == 143 and "RC=143" in out, f"exit {rc}:\n{out}"


@pytest.mark.parametrize("bash", BASHES)
def test_double_sigterm_forwards_once_and_reports_75(bash, harness):
    """Twelve stops, alternating whether the first SIGTERM hits the whole process group (as a job scheduler does)
    or the launcher alone (so the ranks can only have heard it from forward_term), second one 1-9 ms later."""
    for i in range(12):
        p = _launch(bash, harness, ["save"] * 8)
        os.kill(p.pid, signal.SIGCONT)
        if i % 2:
            os.killpg(p.pid, signal.SIGTERM)
        else:
            os.kill(p.pid, signal.SIGTERM)
        time.sleep(0.001 * (1 + i % 9))
        os.kill(p.pid, signal.SIGTERM)
        rc, out = _finish(p)
        assert "bad jump" not in out and "unexpected EOF" not in out, f"{bash} iteration {i}: the trap aborted the shell:\n{out}"
        assert rc == 75 and "RC=75" in out, f"{bash} iteration {i}: exit {rc}, expected the ranks' 75:\n{out}"
        assert out.count("SIGTERM -> trainer ranks") == 1, f"{bash} iteration {i}: forwarded more than once:\n{out}"


@pytest.mark.parametrize("bash", BASHES)
def test_rank_that_exits_before_its_turn_is_collected(bash, harness):
    """Rank 0 runs longest; rank 3 fails at once. The old `while kill -0; do wait; done` loop never waited on rank 3
    (already gone by its turn) and reported 0 for the node."""
    rc, out = _finish(_launch(bash, harness, ["exit:0:0.6", "exit:0:0.1", "exit:0:0.1", "exit:1:0", "exit:0:0.1"]))
    assert rc == 1 and "RC=1" in out, f"{bash}: exit {rc}:\n{out}"


@pytest.mark.parametrize("bash", BASHES)
def test_clean_ranks_report_zero(bash, harness):
    rc, out = _finish(_launch(bash, harness, ["exit:0:0.1"] * 8))
    assert rc == 0 and "RC=0" in out and "SIGTERM" not in out, f"{bash}: exit {rc}:\n{out}"


# ------------------------------------------------------------------ the collector: failures, grace periods, kills
@pytest.mark.parametrize("bash", BASHES)
def test_a_failed_rank_gets_the_others_killed_after_the_grace(bash, harness):
    """One rank crashes (an OOM, an Xid) while the others are blocked in a collective (their SIGTERM handler cannot
    run): they are SIGKILLed DM_RANK_GRACE seconds later and the launcher reports the crash, not the kills' 137,
    instead of waiting for the 30-minute process-group timeout."""
    t0 = time.time()
    rc, out = _finish(_launch(bash, harness, ["hang", "exit:1:0.2", "hang", "hang"], DM_RANK_GRACE=2))
    assert rc == 1 and "RC=1" in out, f"{bash}: exit {rc}:\n{out}"
    assert "rank 1 failed (exit 1)" in out and "3 rank(s) still running 2 s after rank 1 failed (exit 1): SIGKILL" in out, out
    assert time.time() - t0 < 15, f"took {time.time() - t0:.1f} s:\n{out}"


@pytest.mark.parametrize("bash", BASHES)
def test_the_first_failure_is_reported(bash, harness):
    """Two ranks fail 1.5 s apart, a third hangs until the grace kill: the status is the first failure's."""
    rc, out = _finish(_launch(bash, harness, ["exit:3:0.1", "exit:4:1.6", "hang"], DM_RANK_GRACE=3))
    assert rc == 3 and "RC=3" in out and "rank 0 failed (exit 3)" in out and "after rank 0 failed (exit 3): SIGKILL" in out, out


@pytest.mark.parametrize("bash", BASHES)
@pytest.mark.parametrize("code", [3, 143])
def test_a_failure_collected_after_a_stop_that_came_later_is_a_failure_before_it(bash, harness, code):
    """Rank 1 exits <code>; the stop signal reaches the launcher right after it reaped that rank, before its next
    once-a-second poll collects it. The rank was gone when the stop came: a failure before the stop (its time is kept,
    T_FAIL = the stop's at the latest: in a multi-node launch it decides which node is blamed) and its blocked peers are
    SIGKILLed DM_RANK_GRACE seconds later, not DM_TERM_GRACE. A rank that died of the stop signal itself (143: the stop
    can reach the ranks at the same moment, as job.sh's `kill -TERM 0` does on one node) stays a death by the stop: no
    failure time, DM_TERM_GRACE."""
    t0 = time.time()
    p = _launch(bash, harness, ["hang", f"failsig:{code}:0.5", "hang"], SLURM_JOB_ID=12345, DM_RANK_GRACE=2,
                DM_TERM_GRACE=4 if code == 143 else 20)
    rc, out = _finish(p, timeout=60)
    assert rc == code and f"RC={code}" in out and out.count("SIGTERM -> trainer ranks") == 1, f"{bash}: exit {rc}:\n{out}"
    if code == 143:
        assert "T_FAIL=none" in out and "failed (exit" not in out, out
        assert "2 rank(s) still running 4 s after the forwarded SIGTERM: SIGKILL" in out, out
        return
    assert "rank 1 failed (exit 3) before the stop signal: the other ranks get 2 s (DM_RANK_GRACE) to exit" in out, out
    assert "2 rank(s) still running 2 s after rank 1 failed (exit 3): SIGKILL" in out and "T_FAIL=none" not in out, out
    assert time.time() - t0 < 15, f"took {time.time() - t0:.1f} s (DM_TERM_GRACE instead of DM_RANK_GRACE?):\n{out}"


@pytest.mark.parametrize("bash", BASHES)
def test_an_external_sigkill_is_a_failure(bash, harness):
    """A rank killed by someone else (the host's OOM killer: 137) is a failure like any other; only the kills of the
    collector itself are not counted."""
    rc, out = _finish(_launch(bash, harness, ["kill:0.2", "hang"], DM_RANK_GRACE=1))
    assert rc == 137 and "RC=137" in out and "rank 0 failed (exit 137)" in out and "1 rank(s) still running" in out, out


@pytest.mark.parametrize("bash", BASHES)
@pytest.mark.parametrize("modes", [["exit:1:0", "exit:75:1.5"], ["exit:75:0", "exit:1:0.5"]])
def test_a_75_wins_over_a_failure(bash, harness, modes):
    """A rank that exited 75 saved a complete checkpoint: the node reports 75 whatever the order of the exits."""
    rc, out = _finish(_launch(bash, harness, modes, DM_RANK_GRACE=30))
    assert rc == 75 and "RC=75" in out and "SIGKILL" not in out, f"{bash} {modes}: exit {rc}:\n{out}"


@pytest.mark.parametrize("bash", BASHES)
def test_under_slurm_ranks_that_cannot_stop_are_killed_after_the_term_grace(bash, harness):
    """The time limit's SIGTERM reaches ranks blocked in a collective (a peer died): their handler cannot run. Under
    Slurm the launcher SIGKILLs them DM_TERM_GRACE seconds after it forwarded the signal and returns 143, so job.sh can
    still record the link and hand over before KillWait."""
    p = _launch(bash, harness, ["hang", "hang"], SLURM_JOB_ID=12345, DM_TERM_GRACE=2)
    t0 = time.time(); os.kill(p.pid, signal.SIGTERM)
    rc, out = _finish(p)
    assert rc == 143 and "RC=143" in out, f"{bash}: exit {rc}:\n{out}"
    assert out.count("SIGTERM -> trainer ranks") == 1 and "2 rank(s) still running 2 s after the forwarded SIGTERM: SIGKILL" in out, out
    assert time.time() - t0 < 15, out


@pytest.mark.parametrize("bash", BASHES)
def test_under_slurm_a_rank_that_saved_still_wins_over_the_term_grace(bash, harness):
    """One rank saves and exits 75, another stays blocked: it is killed after the grace and the node reports 75."""
    p = _launch(bash, harness, ["save", "hang"], SLURM_JOB_ID=12345, DM_TERM_GRACE=2)
    os.kill(p.pid, signal.SIGTERM)
    rc, out = _finish(p)
    assert rc == 75 and "RC=75" in out and "1 rank(s) still running 2 s after the forwarded SIGTERM: SIGKILL" in out, out


@pytest.mark.parametrize("bash", BASHES)
def test_outside_slurm_the_ranks_are_left_to_finish_their_save(bash, harness):
    """Without SLURM_JOB_ID (a hand launch, Ctrl-C) there is no KillWait to beat: no kill after the SIGTERM."""
    p = _launch(bash, harness, ["hang"], DM_TERM_GRACE=1)
    os.kill(p.pid, signal.SIGTERM)
    time.sleep(3.5)
    still = p.poll() is None
    rc, out = _finish_now(p)
    assert still and "SIGKILL" not in out, f"{bash}: the launcher ended or killed the rank outside Slurm:\n{out}"


@pytest.mark.parametrize("bash", BASHES)
@pytest.mark.parametrize("first,rc", [("exit:0:0.1", 143), ("exit:75:0.1", 75)])
def test_a_rank_stuck_after_the_others_exited_is_killed_after_the_exit_grace(bash, harness, first, rc):
    """One rank finished (the export or the checkpoint is complete: both end with a barrier), another hangs in its
    teardown (destroy_process_group): it is SIGKILLed DM_EXIT_GRACE seconds later instead of holding the job until its
    time limit. The status is 143 (ranks had to be killed), or the 75 of a rank that saved."""
    t0 = time.time()
    got, out = _finish(_launch(bash, harness, [first, "hang"], DM_EXIT_GRACE=2))
    assert got == rc and f"RC={rc}" in out, f"{bash}: exit {got}:\n{out}"
    assert f"1 rank(s) still running 2 s after rank 0 exited {first.split(':')[1]} (DM_EXIT_GRACE" in out, out
    assert time.time() - t0 < 15, f"took {time.time() - t0:.1f} s:\n{out}"


@pytest.mark.parametrize("bash", BASHES)
def test_ranks_that_follow_a_clean_exit_within_the_grace_are_not_killed(bash, harness):
    """The master leaves last (it closes W&B after the others): within DM_EXIT_GRACE nothing is killed."""
    rc, out = _finish(_launch(bash, harness, ["exit:0:2.5", "exit:0:0.1", "exit:0:0.1"], DM_EXIT_GRACE=10))
    assert rc == 0 and "RC=0" in out and "SIGKILL" not in out, f"{bash}: exit {rc}:\n{out}"


def _finish_now(p):
    """Kill the launcher's whole process group, then read what it printed."""
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait(timeout=30)
    return p.returncode, p.stdout.read()


@pytest.fixture
def unkillable_harness(tmp_path):
    """The harness with rank 0 immune to the collector's SIGKILL (a `kill` function shadows the builtin): a process stuck
    in the kernel."""
    (tmp_path / "rank.sh").write_text(RANK)
    h = tmp_path / "harness_unkillable.sh"
    h.write_text(_block() + f"""
kill () {{ if [ "${{1:-}}" = -KILL ] && [ "${{2:-}}" = "${{PIDS[0]:-}}" ]; then return 0; fi; builtin kill "$@"; }}
export READY_DIR=$(mktemp -d)
for m in "$@"; do /bin/bash {tmp_path / 'rank.sh'} "$m" & PIDS+=($!); done
trap forward_term TERM INT
until [ "$(ls "$READY_DIR" | wc -l)" -ge $# ]; do sleep 0.01; done; rm -rf "$READY_DIR"
echo READY
wait_ranks
echo "RC=$RC RANK_RC=${{RANK_RC[*]}}"; exit $RC
""")
    return h


def test_a_rank_that_survives_its_sigkill_is_left_behind(unkillable_harness):
    """A rank still alive 30 s after the collector's SIGKILL (stuck in the kernel) no longer holds the launcher: it
    returns the failure, so the driver and job.sh can go on (and job.sh can steer the chain away from the node)."""
    t0 = time.time()
    rc, out = _finish(_launch("/bin/bash", unkillable_harness, ["hang", "exit:1:0.1"], DM_RANK_GRACE=1), timeout=90)
    assert rc == 1 and "RC=1 RANK_RC=unkillable 1" in out and "still alive 30 s after their SIGKILL" in out, out
    assert 30 < time.time() - t0 < 60, out
