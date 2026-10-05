"""The launcher's SIGTERM forwarding and rank-status collection (scripts/campaign/train_arm.sh), without a scheduler or a GPU.

The block between the `# >>> signal forwarding` / `# <<< signal forwarding` markers of train_arm.sh is run on its own
around stand-in ranks (bash loops that behave like the trainer: on SIGTERM they "save" briefly and exit 75). What must
hold when a job scheduler stops the job and SIGTERM arrives twice within milliseconds (the scheduler signals every
process of the job, then a wrapper script's own trap forwards it again):
  * the shell never aborts (the old handler ran $(hostname -s) $(date) inside the trap, and under bash 5.2 the second
    signal re-entered the parser: "trap: line 2: unexpected EOF ..." / "parse_and_execute: bad jump", exit 134);
  * the ranks are signalled exactly once (the old handler forwarded on every signal) and the launcher reports their 75;
  * a rank that exits before its turn in the collection loop is still collected;
  * under torchrun (a stand-in that starts the ranks) the signal reaches torchrun's children, not torchrun, unless no
    rank has started yet.
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

LAUNCHER = REPO / "scripts/campaign/train_arm.sh"
BASHES = ["/bin/bash"] + ([os.environ["DM_TEST_BASH"]] if os.environ.get("DM_TEST_BASH") else [])

RANK = r"""
case "$1" in
  save) trap 'sleep 0.5; exit 75' TERM; : > "$READY_DIR/$$"; while :; do sleep 0.02; done ;;   # the trainer: checkpoint, exit 75
  exit:*) : > "$READY_DIR/$$"; IFS=: read -r _ code delay <<< "$1"; sleep "$delay"; exit "$code" ;;
esac
"""


def _block() -> str:
    m = re.search(r"^# >>> signal forwarding.*?^# <<< signal forwarding[^\n]*\n", LAUNCHER.read_text(), re.S | re.M)
    assert m, "train_arm.sh lost its signal-forwarding markers (this test runs that block on its own)"
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
echo "RC=$RC"; exit $RC
""")
    return h


def _launch(bash, harness, modes):
    p = subprocess.Popen([bash, str(harness), *modes], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)   # its own process group, so a signal can reach the ranks too
    assert p.stdout.readline().strip() == "READY"   # the trap is installed: signalling now is deterministic
    return p


def _finish(p):
    """Wait for the launcher, then kill whatever is left of its process group BEFORE reading its output: a stand-in
    rank that outlives the launcher (a broken forward, e.g. torchrun signalled instead of its ranks) holds the pipe
    open, and reading first would hang the test instead of failing it."""
    try:
        p.wait(timeout=30)
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
