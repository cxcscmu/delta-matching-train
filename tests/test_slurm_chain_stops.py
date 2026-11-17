"""How a scripts/slurm/job.sh link tells and handles the ways a job ends, under a fake Slurm that does to the queue what
slurmctld does (test_slurm_wrappers.py): a scancel of the running link (its squeue end time becomes the moment of the
cancel; the start + time limit does not move) stops the chain, also when no successor was queued yet; the time limit
keeps it; a requeue (squeue's RestartCnt above the link's SLURM_RESTART_COUNT while the job is COMPLETING, as Slurm
24.05.5 shows it, or the job PENDING again) changes nothing, and a stop of a requeued job's rerun is a stop; a
successor that Slurm already started while the link
completes continues the chain (and is cancelled by a stop); a link waits for a predecessor that is still COMPLETING; an
attempt recorded before a SIGKILL is not counted twice; the first job of a newer submission starts afresh and is never
cancelled by an older link; a signal during the site file is a stop; a command that cannot be run stops the chain at
once; no compile-cache save after a signal; and a whole chain, from submit.sh through links started from the
environments the fake sbatch recorded."""
import os, signal, subprocess, threading, time
import pytest
from test_slurm_wrappers import (CHAIN, JOB, SUBMIT, FakeSlurm, _cmd, _env, _fake_slurm, _iso, _job_env, _job_sigterm,
                                 _link)

USER = os.environ.get("USER", "")


def _chain_dir(tmp_path):
    return tmp_path / "ws" / "logs" / "slurm" / "chains" / CHAIN


def _links(tmp_path, *lines):
    d = _chain_dir(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "links", "a") as f:
        f.write("".join(l + "\n" for l in lines))


def _progress(tmp_path, *lines):
    d = _chain_dir(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "progress", "a") as f:
        f.write("".join(l + "\n" for l in lines))


def _events(r):
    return [(rec[2], r.kv(rec)) for rec in r.records()]


def _records(tmp_path):
    f = _chain_dir(tmp_path) / "links"
    return [l.split() for l in f.read_text().splitlines()] if f.exists() else []


# what slurmctld does to the queue before the batch shell gets its SIGTERM
def _scancel_of(jid=99, name=None):
    """scancel <jid> (or scancel -n <name>): pending jobs CANCELLED; the running one CANCELLED + COMPLETING, its end
    time the moment of the cancel."""
    def f(fake):
        now = _iso(int(time.time()))

        def g(s):
            for j in s["jobs"]:
                if j["id"] == jid or (name and j["name"] == name):
                    if j["state"] == "PENDING":
                        j["state"] = "CANCELLED"
                    elif j["state"] == "RUNNING":
                        j.update(state="CANCELLED", completing=True, end=now)
        fake.update(g)
    return f


def _job_state(jid, state):
    """The time limit (TIMEOUT) or a requeue (PENDING): + COMPLETING, end time now."""
    def f(fake):
        now = _iso(int(time.time()))
        fake.update(lambda s: [j.update(state=state, completing=True, end=now) for j in s["jobs"] if j["id"] == jid])
    return f


def _requeued(jid):
    """`scontrol requeue` as Slurm 24.05.5 shows it while the batch script handles the SIGTERM (verified with a probe
    job): the job COMPLETING with its restart count already raised, never PENDING meanwhile; end time now."""
    def f(fake):
        now = _iso(int(time.time()))
        fake.update(lambda s: [j.update(completing=True, end=now, restarts=int(j.get("restarts", 0)) + 1)
                               for j in s["jobs"] if j["id"] == jid])
    return f


# --------------------------------------------------------------------------------------------- what a signal means
def test_scancel_of_the_running_link_stops_the_chain(tmp_path):
    """squeue's end time of a cancelled job is the cancel time: the stop is judged by start + time limit (2 h away)."""
    rc, out, err, fake = _job_sigterm(tmp_path, before_signal=_scancel_of(99))
    assert rc == 75 and "CMD saved" in out and "stopped by SIGTERM" in out and "the chain is stopped" in out, out
    assert fake.job(4242)["state"] == "CANCELLED" and "before the end of the job: the time limit" not in out, out
    assert [c["argv"] for c in fake.calls("scancel")] == [["4242"]], fake.calls("scancel")


def test_scancel_n_after_a_refused_start_queues_nothing(tmp_path):
    """`scancel -n <chain>` while the link has no successor (sbatch refused at its start, e.g. the QOS submit limit):
    nothing is queued at its end, so the chain does not outlive the scancel."""
    fake = FakeSlurm(tmp_path, running={"id": 99, "name": CHAIN, "end": _iso(int(time.time()) + 7200)}, sbatch=["qos"] * 3)
    script = tmp_path / "cmd.sh"
    ready = tmp_path / "ready"
    script.write_text(f"#!/bin/bash\ntrap 'exit 75' TERM\ntouch {ready}\nsleep 60 & wait\n")
    script.chmod(0o755)
    p = subprocess.Popen(["bash", str(JOB), str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=_job_env(tmp_path, fake, SLURM_JOB_END_TIME=int(time.time()) + 7200), start_new_session=True)
    t0 = time.time()
    while not ready.exists():
        assert p.poll() is None and time.time() - t0 < 30
        time.sleep(0.05)
    _scancel_of(99, name=CHAIN)(fake)
    os.kill(p.pid, signal.SIGTERM)
    out = p.communicate(timeout=60)[0]
    assert p.returncode == 75 and "the chain is stopped" in out and "queued successor" not in out, out
    assert len(fake.calls("sbatch")) == 3 and fake.jobs(state="PENDING") == [], out


def test_time_limit_is_the_time_limit_whatever_squeue_ends(tmp_path):
    """At the time limit slurmctld also sets the end time to now: start + limit says it is the limit. The link saves no
    compile caches then (Slurm kills it KillWait after the signal), although they changed."""
    rc, out, err, fake = _job_sigterm(tmp_path, end_in=20, before_signal=_job_state(99, "TIMEOUT"), DM_STOP_BEFORE=1,
                                      pre='mkdir -p "$TRITON_CACHE_DIR" && echo k > "$TRITON_CACHE_DIR/new.bin"')
    assert rc == 75 and "the time limit" in out and "the successor 4242 continues the chain" in out, out
    assert fake.job(4242)["state"] == "PENDING" and fake.calls("scancel") == [], out
    assert "compile caches not saved: the job is ending (SIGTERM)" in out, out
    assert not (_chain_dir(tmp_path) / "jit.tar").exists()


@pytest.mark.parametrize("how", ["restart_count", "pending"])
@pytest.mark.parametrize("save", [True, False])
def test_requeue_changes_nothing(tmp_path, save, how):
    """Preemption or `scontrol requeue`: Slurm starts the job again with its id. While this link handles the SIGTERM the
    job is COMPLETING with its restart count raised (what Slurm 24.05.5 shows), or already PENDING again. Nothing is
    cancelled or queued, and an attempt without progress is not a failure."""
    pre = "" if save else "export DM_PROGRESS_FILE=/dev/null"
    rq = _requeued(99) if how == "restart_count" else _job_state(99, "PENDING")
    rc, out, err, fake = _job_sigterm(tmp_path, before_signal=rq, pre=pre)
    assert rc == 75 and "Slurm requeued this job" in out and "the chain is stopped" not in out, out
    assert fake.calls("scancel") == [] and fake.job(4242)["state"] == "PENDING" and len(fake.calls("sbatch")) == 1, out
    recs = _records(tmp_path)
    att = [dict(w.split("=", 1) for w in r[3:]) for r in recs if r[2] == "attempt"]
    assert att == [{"n": "1", "rc": "75", "progress": "1" if save else "0", "fails": "0"}], recs
    assert recs[-1][2] == "end" and "action=requeued" in recs[-1], recs


def test_shrunk_time_limit_shortens_the_links_budget(tmp_path):
    """`scontrol update TimeLimit=` moved the end 15 s ahead (SLURM_JOB_END_TIME still says 2 h): at the time limit no
    compile caches are saved and the end's sbatch is not retried (90 s before the moved end has passed; with the old end
    it would be, 3 s later)."""
    end = int(time.time()) + 15
    fake = FakeSlurm(tmp_path, running={"id": 99, "name": CHAIN, "end": _iso(end)}, sbatch=["qos"] * 2,
                     sbatch_default="fail")
    script = tmp_path / "cmd.sh"
    ready = tmp_path / "ready"
    script.write_text("#!/bin/bash\n"
                      "trap 'mkdir -p \"$TRITON_CACHE_DIR\"; echo k > \"$TRITON_CACHE_DIR/new.bin\"; "
                      "echo \"$(date +%s) $SLURM_JOB_ID ckpt RUN 7\" >> \"$DM_PROGRESS_FILE\"; exit 75' TERM\n"
                      f"touch {ready}\nsleep 60 & wait\n")
    script.chmod(0o755)
    p = subprocess.Popen(["bash", str(JOB), str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=_job_env(tmp_path, fake, SLURM_JOB_END_TIME=int(time.time()) + 7200,
                                      DM_RESUBMIT_BACKOFF="3"), start_new_session=True)
    t0 = time.time()
    while not ready.exists():
        assert p.poll() is None and time.time() - t0 < 60
        time.sleep(0.05)
    os.kill(p.pid, signal.SIGTERM)
    out = p.communicate(timeout=60)[0]
    assert p.returncode == 76 and "the time limit" in out and "chain is broken" in out, out
    assert len(fake.calls("sbatch")) == 2 + 1 and "compile caches saved" not in out, out   # 2 at the start, 1 at the end


# ------------------------------------------------------------------- a predecessor that is still COMPLETING
def test_link_waits_for_its_completing_predecessor(tmp_path):
    """Singleton released this job when its predecessor timed out, but the predecessor still completes: its job.sh
    writes its last records. The link judges only after it has ended (else it would judge it as killed)."""
    fake = _fake_slurm(tmp_path)
    pred = fake.add_job(state="TIMEOUT", completing=True, complete_at=time.time() + 3)
    _links(tmp_path, f"1000 {pred} start k=0 restart=0 node=n1 fails=0", f"1001 {pred} succ id=99 how=queued",
           f"1002 {pred} run n=1")
    _progress(tmp_path, f"1500 {pred} ckpt RUN 100")
    late = (f"{int(time.time())} {pred} attempt n=1 rc=143 progress=1 fails=0\n"
            f"{int(time.time())} {pred} end k=0 rc=143 action=continue fails=0 succ=99\n")
    timer = threading.Timer(2, lambda: open(_chain_dir(tmp_path) / "links", "a").write(late))
    timer.start()
    t0 = time.time()
    r = _link(tmp_path, (75,), (1,), fake=fake, DM_RESUBMIT_COUNT=1)
    timer.join()
    assert r.rc == 75 and f"waiting for job(s) {pred} of the chain to end" in r.out and time.time() - t0 >= 3, r.out
    assert "judge" not in [e for e, _ in _events(r)], _events(r)
    assert r.out.index("waiting for job(s)") < r.out.index("CMD n=1"), r.out


def test_link_wait_for_a_predecessor_is_bounded(tmp_path):
    fake = _fake_slurm(tmp_path)
    pred = fake.add_job(state="CANCELLED", completing=True)   # an unkillable rank: COMPLETING for good
    t0 = time.time()
    r = _link(tmp_path, (75,), (1,), fake=fake, DM_PRED_WAIT=3)
    assert r.rc == 75 and f"job(s) {pred} of the chain still running or completing after" in r.out, r.out
    assert "starting anyway" in r.out and 3 <= time.time() - t0 < 30 and "CMD n=1" in r.out, r.out


@pytest.mark.parametrize("how", ["no signal", "time limit"])
def test_successor_already_started_continues_the_chain(tmp_path, how):
    """Slurm started the successor while this link completes (its state final at the time limit): the successor is not
    'cancelled', nothing is cancelled, and this node is not excluded (it is too late to)."""
    if how == "no signal":
        r = _link(tmp_path, (1,), (0,), body="fake_set 4242 state=RUNNING", DM_INPLACE_RETRIES=0)
        rc, out, fake = r.rc, r.out, r.fake
        assert rc == 1, out
    else:
        rc, out, err, fake = _job_sigterm(tmp_path, end_in=20, pre="fake_set 4242 state=RUNNING",
                                          before_signal=_job_state(99, "TIMEOUT"), DM_STOP_BEFORE=1)
        assert rc == 75 and "the time limit" in out, out
    assert "the successor 4242 is RUNNING already" in out and "the successor 4242 continues the chain" in out, out
    assert fake.calls("scancel") == [] and fake.calls("scontrol") == [] and fake.job(4242)["state"] == "RUNNING", out


def test_stop_cancels_a_successor_that_slurm_already_started(tmp_path):
    """`scancel <running link>`: singleton released the successor at once; the stop cancels it although it runs."""
    rc, out, err, fake = _job_sigterm(tmp_path, pre="fake_set 4242 state=RUNNING", before_signal=_scancel_of(99))
    assert rc == 75 and "the chain is stopped" in out and fake.job(4242)["state"] == "CANCELLED", out
    assert fake.job(99)["state"] == "CANCELLED", fake.state()["jobs"]   # (by the user; job.sh never cancels itself)
    assert [c["argv"] for c in fake.calls("scancel")] == [["4242"]]


# ----------------------------------------------------------------------------------------------- failure accounting
def test_judge_does_not_count_a_recorded_attempt_twice(tmp_path):
    """Job 41 recorded its failed attempt (fails=1) and was killed before its end record: one failure, not two."""
    _links(tmp_path, "1000 41 start k=0 restart=0 node=n1 fails=0", "1001 41 succ id=99 how=queued",
           "1002 41 run n=1", "1003 41 attempt n=1 rc=1 progress=0 fails=1")
    r = _link(tmp_path, (1,), (0,), DM_RESUBMIT_COUNT=1, DM_INPLACE_RETRIES=0)
    ev = _events(r)
    assert ("judge", {"of": "41", "progress": "-", "fails": "1"}) in ev, ev
    assert r.rc == 1 and ev[-1][1]["action"] == "continue" and ev[-1][1]["fails"] == "2", ev   # 2 <= DM_MAX_FAILS


def test_judge_counts_a_cut_off_retry_from_its_own_start(tmp_path):
    """Job 41's first attempt progressed (and failed); its retry in place was killed without progress: one failure (the
    progress line of the first attempt does not count for the retry)."""
    _links(tmp_path, "1000 41 start k=0 restart=0 node=n1 fails=1", "1002 41 run n=1",
           "1600 41 attempt n=1 rc=1 progress=1 fails=0", "1700 41 run n=2")
    _progress(tmp_path, "1500 41 ckpt RUN 100")
    r = _link(tmp_path, (75,), (1,), DM_RESUBMIT_COUNT=1)
    assert ("judge", {"of": "41", "progress": "0", "fails": "1"}) in _events(r), _events(r)


def test_killed_after_its_attempt_record_counts_once(tmp_path):
    """End to end: a link SIGKILLed (KillWait) while its end checks a slow controller, after its attempt record; the next
    link counts that failure once: two failed attempts are not a give-up with DM_MAX_FAILS=2."""
    fake = _fake_slurm(tmp_path)
    slow = tmp_path / "slowbin"
    slow.mkdir()
    (slow / "squeue").write_text(f'#!/bin/bash\ncase "$*" in *%T*) sleep 30 ;; esac\nexec {fake.bin}/squeue "$@"\n')
    (slow / "squeue").chmod(0o755)
    env = _job_env(tmp_path, fake, DM_INPLACE_RETRIES=0)
    env["PATH"] = f"{slow}:{env['PATH']}"
    p = subprocess.Popen(["bash", str(JOB), str(_cmd(tmp_path, (1,), (0,)))], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, env=env, start_new_session=True)
    t0 = time.time()
    while not any(r[2] == "attempt" for r in _records(tmp_path)):
        assert time.time() - t0 < 30 and p.poll() is None
        time.sleep(0.05)
    time.sleep(0.5)
    os.killpg(p.pid, signal.SIGKILL)
    p.communicate(timeout=30)
    assert not any(r[2] == "end" for r in _records(tmp_path)), _records(tmp_path)
    nxt = tmp_path / "next"
    nxt.mkdir()
    r = _link(tmp_path, (1,), (0,), DM_RESUBMIT_COUNT=1, DM_INPLACE_RETRIES=0, SLURM_JOB_ID=100,
              fake=FakeSlurm(nxt, running={"id": 100, "name": CHAIN}))
    assert r.rc == 1 and "giving up" not in r.out, r.out
    assert ("judge", {"of": "99", "progress": "-", "fails": "1"}) in _events(r), _events(r)
    assert _events(r)[-1][1]["fails"] == "2", _events(r)


# ------------------------------------------------------------------------------------- a submission while it ran
def test_first_job_of_a_submission_starts_afresh(tmp_path):
    """The user submitted the chain again (job 99) while an older link (41) still ran; 41's last records came after the
    submission. 99 starts at 0 failures and judges none of them; a requeue of 99 still judges its own first run."""
    _links(tmp_path, "100 99 submit fails=0 k=0 by=u", "101 41 attempt n=3 rc=1 progress=0 fails=3",
           "102 41 end k=7 rc=1 action=giveup fails=3")
    r = _link(tmp_path, (75,), (1,))
    assert r.rc == 75 and "giving up" not in r.out and "first job of a submission" in r.out, r.out
    ev = _events(r)
    assert ("start", "0") in [(e, kv.get("fails")) for e, kv in ev] and "judge" not in [e for e, _ in ev], ev
    # its requeue: the first run (no attempt record: killed, no progress line since it began) is judged
    (_chain_dir(tmp_path) / "progress").unlink()
    (_chain_dir(tmp_path) / "links").write_text("100 99 submit fails=0 k=0 by=u\n"
                                                "102 41 end k=7 rc=1 action=giveup fails=3\n"
                                                "103 99 start k=0 restart=0 node=n1 fails=0\n104 99 run n=1\n")
    r = _link(tmp_path, (75,), (1,), SLURM_RESTART_COUNT=1)
    assert ("judge", {"of": "99", "progress": "0", "fails": "1"}) in _events(r), _events(r)


@pytest.mark.parametrize("how", ["exit 2", "scancel"])
def test_a_newer_submission_is_never_cancelled_by_an_older_link(tmp_path, how):
    """While the link ran, the user submitted the chain again (job 4243, a `submit` record): the link's stop cancels its
    own successor only; the new submission continues the chain."""
    def resubmit(fake):
        new = fake.add_job()
        _links(tmp_path, f"{int(time.time())} {new} submit fails=0 k=0 by=u")
    if how == "exit 2":
        fake = _fake_slurm(tmp_path)
        body = ('new=$(sbatch --parsable --job-name="$SLURM_JOB_NAME" x); '
                'echo "$(date +%s) $new submit fails=0 k=0 by=u" >> "$DM_CHAIN_DIR/links"')
        r = _link(tmp_path, (2,), fake=fake, body=body)
        rc, out = r.rc, r.out
    else:
        rc, out, err, fake = _job_sigterm(tmp_path, before_signal=lambda f: (_scancel_of(99, name=CHAIN)(f), resubmit(f)))
    assert "not cancelling job 4243: a newer submission of the chain" in out, out
    assert fake.job(4242)["state"] == "CANCELLED" and fake.job(4243)["state"] == "PENDING", fake.state()["jobs"]


# ------------------------------------------------------------------------------------------------- the link's setup
@pytest.mark.parametrize("to", ["shell", "group"])
def test_signal_during_the_site_file_is_a_stop(tmp_path, to):
    """The handler is set before the site file (conda activate takes seconds): a scancel there is a stop, not a killed
    batch shell (no records, no cleanup). A group signal also stops the site file's command (it then fails)."""
    fake = _fake_slurm(tmp_path)
    mark = tmp_path / "site-started"
    site = tmp_path / "slow_site.sh"
    site.write_text(f"touch {mark}\nsleep 3\n")
    p = subprocess.Popen(["bash", str(JOB), str(_cmd(tmp_path, (75,), (1,)))], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, start_new_session=True,
                         env=_job_env(tmp_path, fake, DM_SITE_ENV=site, SLURM_JOB_END_TIME=int(time.time()) + 7200))
    t0 = time.time()
    while not mark.exists():
        assert p.poll() is None and time.time() - t0 < 30
        time.sleep(0.05)
    (os.killpg if to == "group" else os.kill)(p.pid, signal.SIGTERM)
    out = p.communicate(timeout=60)[0]
    assert p.returncode == 143 and "CMD" not in out and "SIGTERM before the command started" in out, out
    assert "the chain is stopped" in out and fake.calls("sbatch") == [], out
    if to == "group":
        assert "WARN: sourcing DM_SITE_ENV" in out and "after a SIGTERM" in out, out
    assert [r[2] for r in _records(tmp_path)] == ["start", "end"], _records(tmp_path)


@pytest.mark.parametrize("cmd", ["noexec", "missing", "nosuchcommand"])
def test_a_command_that_cannot_be_run_stops_the_chain(tmp_path, cmd):
    """bash would exit 126 / 127 in every link (an in-place retry, a queue wait, again): exit 2 before anything."""
    fake = _fake_slurm(tmp_path)
    script = _cmd(tmp_path, (75,))
    target = {"noexec": str(script), "missing": str(tmp_path / "missing.sh"), "nosuchcommand": "nosuchcommand_xyz"}[cmd]
    if cmd == "noexec":
        script.chmod(0o644)
    p = subprocess.run(["bash", str(JOB), target], capture_output=True, text=True, timeout=60, env=_job_env(tmp_path, fake))
    assert p.returncode == 2 and "cannot be run" in p.stderr and fake.calls("sbatch") == [], p.stdout + p.stderr


# --------------------------------------------------------------------------------------------- a chain, end to end
TRAINER = r'''#!/bin/bash
n=$(( $(cat "$CNT" 2>/dev/null || echo 0) + 1 )); echo $n > "$CNT"
echo "TRAINER n=$n k=$DM_RESUBMIT_COUNT stop=${DM_STOP_BEFORE:-none} derived=${DM_DERIVED:-none} triton=$TRITON_CACHE_DIR"
mkdir -p "$TRITON_CACHE_DIR" && echo "k$n" > "$TRITON_CACHE_DIR/kern$n.bin"
echo "$(date +%s) $SLURM_JOB_ID ckpt RUN $((n * 100))" >> "$DM_PROGRESS_FILE"
[ -n "${STOP_AT:-}" ] && [ "$n" -ge "$STOP_AT" ] && { trap 'exit 75' TERM; touch "$CNT.ready"; sleep 60 & wait; }
[ "$n" -ge "$DMT_NLINKS" ] && exit 0
exit 75
'''


def _start(fake, job, node, **extra):
    """Run a queued job as slurmd would: its recorded argv, environment and working directory (running, then done)."""
    env = dict(job["env"])
    end = int(time.time()) + 4 * 3600
    env.update(SLURM_JOB_ID=str(job["id"]), SLURM_JOB_NAME=job["name"], SLURM_SUBMIT_DIR=job["cwd"],
               SLURMD_NODENAME=node, SLURM_JOB_END_TIME=str(end), **extra)
    fake.update(lambda s: [j.update(state="RUNNING", end=_iso(end), start=_iso(end - 4 * 3600))
                           for j in s["jobs"] if j["id"] == job["id"]])
    script = job["argv"][[i for i, a in enumerate(job["argv"]) if a.endswith("/scripts/slurm/job.sh")][0]:]
    return subprocess.Popen(["bash", *script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env,
                            cwd=job["cwd"], start_new_session=True)


def _finish(fake, jid, p, timeout=120):
    out = p.communicate(timeout=timeout)[0]
    fake.update(lambda s: [j.update(state="COMPLETED") for j in s["jobs"] if j["id"] == jid and j["state"] == "RUNNING"])
    return p.returncode, out


def _submit_chain(tmp_path, **extra):
    fake = FakeSlurm(tmp_path, first_id=int(time.time()) % 10 ** 6 * 100)
    trainer = tmp_path / "trainer.sh"
    trainer.write_text(TRAINER)
    trainer.chmod(0o755)
    env = _env(tmp_path, PATH=f"{fake.bin}:{os.environ['PATH']}", CNT=tmp_path / "cnt", DMT_NLINKS=3,
               DM_JIT_ROOT=tmp_path / "jit", DM_RESUBMIT_BACKOFF="0 0", USER=os.environ["USER"], **extra)
    p = subprocess.run(["bash", str(SUBMIT), "-t", "4:00:00", "--", str(trainer), "arg"], capture_output=True, text=True,
                       timeout=60, env=env, cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    return fake


def _pending(fake):
    return sorted((j for j in fake.state()["jobs"] if j["state"] == "PENDING"), key=lambda j: j["id"])


def test_a_chain_runs_from_submit_to_completion(tmp_path):
    """submit.sh, then every pending job started from what the fake sbatch recorded: link k gets DM_RESUBMIT_COUNT=k, no
    deadline or derived variables of its predecessor, the compile caches it saved; the last one completes the chain."""
    fake = _submit_chain(tmp_path)
    outs = []
    for step in range(6):
        pend = _pending(fake)
        if not pend:
            break
        rc, out = _finish(fake, pend[0]["id"], _start(fake, pend[0], f"node{step % 2}"))
        outs.append((rc, out))
    assert [rc for rc, _ in outs] == [75, 75, 0], [o for _, o in outs]
    lines = [next(l for l in o.splitlines() if l.startswith("TRAINER")) for _, o in outs]
    assert [l.split()[2] for l in lines] == ["k=0", "k=1", "k=2"] and all("derived=" in l for l in lines), lines
    assert "compile caches restored" in outs[1][1] and "compile caches restored" in outs[2][1], outs[1][1]
    for j in fake.state()["jobs"][1:]:   # the successors' environments: none of their predecessor's own variables
        assert not {"DM_STOP_BEFORE", "DM_JOB_END", "DM_DERIVED", "DM_VALCE_PORT", "DM_NPROC"} & set(j["env"]), j["env"]
    assert _pending(fake) == [], fake.state()["jobs"]


def test_a_chain_stops_when_its_running_link_is_cancelled(tmp_path):
    """`scancel <running link id>` (not -n): the link cancels the successor it queued, and nothing runs after it."""
    fake = _submit_chain(tmp_path, STOP_AT=2)
    rc, out = _finish(fake, _pending(fake)[0]["id"], _start(fake, _pending(fake)[0], "node0"))
    assert rc == 75, out
    job = _pending(fake)[0]
    p = _start(fake, job, "node1")
    t0 = time.time()
    while not (tmp_path / "cnt.ready").exists():
        assert p.poll() is None and time.time() - t0 < 60, p.communicate()[0]
        time.sleep(0.05)
    _scancel_of(job["id"])(fake)
    os.kill(p.pid, signal.SIGTERM)
    rc, out = _finish(fake, job["id"], p)
    assert rc == 75 and "the chain is stopped" in out and _pending(fake) == [], out


def test_a_stop_of_a_requeued_jobs_rerun_is_a_stop(tmp_path):
    """The rerun of a requeued job (SLURM_RESTART_COUNT=1, squeue's RestartCnt 1): a scancel of it is a stop by hand,
    not another requeue."""
    rc, out, err, fake = _job_sigterm(tmp_path, running={"restarts": 1}, before_signal=_scancel_of(99),
                                      SLURM_RESTART_COUNT=1)
    assert [j.get("restarts") for j in fake.state()["jobs"] if j["id"] == 99] == [1]   # what squeue -O RestartCnt printed
    assert rc == 75 and "the chain is stopped" in out and "Slurm requeued this job" not in out, out
