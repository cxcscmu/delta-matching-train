"""The life of a chain of scripts/slurm/job.sh links (README there), under the fake Slurm of test_slurm_wrappers.py:
the successor queued at the start of a link (its environment; a pending job of the chain adopted instead of a duplicate,
also after an sbatch that timed out after queueing its job), the end of a link (exit 0 / 2 / 3 / 4 and a stop by hand
cancel the chain's other jobs, by id; the time limit, 75 and a failure keep the successor; a successor someone cancelled
stops the chain; a missing one is queued at the end or the chain is reported broken), the failure accounting (`links`:
a predecessor that left no record is judged at the next start; an attempt without a progress line is a failure; a
progress line resets the count; more than DM_MAX_FAILS in a row gives up with exit 4), in-place retries, the node
exclusion after a failure, signals before the command started, a single job outside the chain, and the compile-cache
archive with the EXIT cleanup. test_slurm_chain_stops.py: scancel / time limit / requeue, a COMPLETING predecessor,
newer submissions."""
import os, random, shutil, signal, subprocess, tarfile, time
import pytest
from test_slurm_wrappers import (CHAIN, JOB, OPTS, FakeSlurm, _cmd, _fake_slurm, _iso, _job_env, _job_sigterm, _link)

USER = os.environ.get("USER", "")


def _chain_dir(tmp_path):
    return tmp_path / "ws" / "logs" / "slurm" / "chains" / CHAIN


def _links(tmp_path, *lines):
    """Pre-existing `links` records (epoch, job id, event, key=value...)."""
    d = _chain_dir(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    (d / "links").write_text("".join(l + "\n" for l in lines))


def _progress(tmp_path, *lines):
    d = _chain_dir(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "progress", "a") as f:
        f.write("".join(l + "\n" for l in lines))


def _events(r):
    return [(rec[2], r.kv(rec)) for rec in r.records()]


# ----------------------------------------------------------------------------------------------- the start of a link
def test_link_records_and_queues_its_successor_with_a_clean_environment(tmp_path):
    """The successor: DM_RESUBMIT_COUNT+1, the chain's state, none of this job's deadline / end time / derived ports
    (a user's port goes on), queued from the submit directory before the command runs; the command gets the chain's
    variables; `links` gets start / succ / attempt / end records, each failure count included."""
    sub = tmp_path / "sub"
    sub.mkdir()
    r = _link(tmp_path, (75,), (1,), DM_RESUBMIT_COUNT=3, DM_MASTER_PORT=30000, SLURM_SUBMIT_DIR=sub)
    assert r.rc == 75, r.out
    d = _chain_dir(tmp_path)
    assert f"k=3 chain={CHAIN} dir={d} pf={d}/progress" in r.out, r.out
    (succ,) = r.fake.jobs(state="PENDING")
    env = succ["env"]
    assert succ["id"] == 4242 and succ["name"] == CHAIN and succ["cwd"] == str(sub), succ
    assert env["DM_RESUBMIT_COUNT"] == "4" and env["DM_CHAIN_DIR"] == str(d) and env["DM_MASTER_PORT"] == "30000", env
    assert not {"DM_STOP_BEFORE", "DM_JOB_END", "DM_VALCE_PORT", "DM_DERIVED"} & set(env), env
    assert not env.get("TRITON_CACHE_DIR", "").startswith(str(tmp_path / "jit")), env   # not this link's caches
    assert "deadline=none" not in r.out and "port=30000/30001" in r.out, r.out   # the command's own are untouched
    ev = _events(r)
    assert [e for e, _ in ev] == ["start", "succ", "run", "attempt", "end"], ev
    assert ev[0][1] == {"k": "3", "restart": "0", "node": "node7", "fails": "0"}, ev
    assert ev[1][1] == {"id": "4242", "how": "queued"}, ev
    assert ev[2][1] == {"n": "1"}, ev
    assert ev[3][1] == {"n": "1", "rc": "75", "progress": "1", "fails": "0"}, ev
    assert ev[4][1] == {"k": "3", "rc": "75", "action": "continue", "fails": "0", "succ": "4242"}, ev
    assert r.fake.calls("scancel") == [] and "the successor 4242 continues the chain" in r.out, r.out


def test_link_derived_ports_are_not_inherited(tmp_path):
    r = _link(tmp_path, (75,), (1,), DM_EPHEMERAL_PORTS="9000 65000")
    assert r.rc == 75 and "port=7898/7899" in r.out, r.out
    assert not {"DM_MASTER_PORT", "DM_VALCE_PORT"} & set(r.fake.job(4242)["env"]), r.fake.job(4242)["env"]


def test_link_adopts_a_pending_job_of_its_chain(tmp_path):
    """A pending job of the chain (a requeued link's successor, a second submission) is the successor: no duplicate.
    Jobs of other chains or users are not."""
    fake = _fake_slurm(tmp_path)
    other = fake.add_job(name="dm-other-chain")
    stranger = fake.add_job(user="someone-else")
    mine = fake.add_job()
    r = _link(tmp_path, (75,), (1,), fake=fake)
    assert r.rc == 75 and fake.calls("sbatch") == [], r.out
    assert f"pending job {mine} of the chain is the successor" in r.out, r.out
    assert {"id": str(mine), "how": "adopted"} in [kv for e, kv in _events(r) if e == "succ"], _events(r)
    assert [j["state"] for j in map(fake.job, (other, stranger, mine))] == ["PENDING"] * 3


def test_link_adopts_the_job_of_an_sbatch_that_timed_out(tmp_path):
    """sbatch queued the job but its reply timed out (exit 1): the retry finds the job pending and takes it."""
    fake = _fake_slurm(tmp_path, sbatch=["ambiguous"])
    r = _link(tmp_path, (75,), (1,), fake=fake)
    assert r.rc == 75 and len(fake.calls("sbatch")) == 1, r.out
    assert len(fake.jobs(name=CHAIN, state="PENDING")) == 1, fake.state()["jobs"]
    assert "WARN: sbatch failed; retry 1" in r.out and "pending job 4242 of the chain is the successor" in r.out, r.out


def test_link_submits_when_squeue_fails(tmp_path):
    """Whether a pending job exists is unknown: submit (singleton keeps a duplicate from running beside a link)."""
    fake = _fake_slurm(tmp_path, squeue_fail=["PENDING"])
    r = _link(tmp_path, (75,), (1,), fake=fake)
    assert r.rc == 75 and len(fake.calls("sbatch")) == 1 and "queued successor 4242" in r.out, r.out


def test_link_queues_at_its_end_when_the_start_failed(tmp_path):
    """sbatch refused (a full submit quota) at the start: the command still runs, and the successor is queued at the
    end."""
    fake = _fake_slurm(tmp_path, sbatch=["qos"] * 3)
    r = _link(tmp_path, (75,), (1,), fake=fake)
    assert r.rc == 75 and len(fake.calls("sbatch")) == 4, r.out
    assert "no successor could be queued at the start of this job: trying again at its end" in r.out, r.out
    assert r.out.index("CMD n=1") < r.out.index("queued successor 4242"), r.out
    assert fake.job(4242)["state"] == "PENDING" and [e for e, _ in _events(r)][-2:] == ["succ", "end"], _events(r)


def test_link_end_retries_stop_90_s_before_the_end(tmp_path):
    """At the end of a link, sbatch is retried only until 90 s before the job's end (KillWait follows)."""
    fake = _fake_slurm(tmp_path, end=_iso(int(time.time()) + 60), sbatch_default="fail")
    r = _link(tmp_path, (75,), (1,), fake=fake, SLURM_JOB_END_TIME=int(time.time()) + 60, DM_STOP_BEFORE=1)
    assert r.rc == 76 and len(fake.calls("sbatch")) == 3 + 1, r.out   # 3 at the start, 1 at the end
    assert "the chain is broken: resume with" in r.out and [e for e, _ in _events(r)][-1] == "end", r.out
    assert _events(r)[-1][1]["action"] == "broken", _events(r)


def test_link_without_successor_at_the_last_link(tmp_path):
    r = _link(tmp_path, (75,), (1,), DM_RESUBMIT_COUNT=400)   # the default DM_MAX_RESUBMITS
    assert r.rc == 75 and r.fake.calls("sbatch") == [] and "DM_MAX_RESUBMITS=400" in r.out, r.out
    assert r.fake.calls("scancel") == [] and _events(r)[-1][1]["action"] == "max", _events(r)


# --------------------------------------------------------------------------------------------------- the end of a link
def test_link_exit_0_cancels_the_pending_jobs(tmp_path):
    fake = _fake_slurm(tmp_path)
    other = fake.add_job(name="dm-other-chain")
    r = _link(tmp_path, (0,), fake=fake)
    assert r.rc == 0 and "everything is done; the chain is complete" in r.out, r.out
    (succ,) = [j for j in fake.jobs(name=CHAIN) if j["id"] != 99]
    assert succ["state"] == "CANCELLED" and fake.job(other)["state"] == "PENDING", fake.state()["jobs"]
    assert fake.job(99)["state"] == "RUNNING"
    assert [c["argv"] for c in fake.calls("scancel")] == [[str(succ["id"])]]   # by id: never this job
    assert _events(r)[-1][1]["action"] == "done", _events(r)


@pytest.mark.parametrize("rc, why", [(2, "a configuration error"), (3, "the run belongs to another live job"),
                                     (4, "it gave up")])
def test_link_exit_2_3_4_stops_the_chain(tmp_path, rc, why):
    r = _link(tmp_path, (rc,))
    assert r.rc == rc and f"exit {rc} ({why}): the chain is stopped" in r.out, r.out
    assert r.out.count("CMD n=") == 1 and r.fake.job(4242)["state"] == "CANCELLED", r.out   # never retried
    assert [e for e, _ in _events(r)] == ["start", "succ", "run", "end"], _events(r)
    assert _events(r)[-1][1]["action"] == "stop", _events(r)


@pytest.mark.parametrize("squeue_fail", [[], ["%T"]])   # by id; squeue failing: by name, pending only
def test_link_reports_a_failed_cancel(tmp_path, squeue_fail):
    fake = _fake_slurm(tmp_path, squeue_fail=squeue_fail)
    fake.update(lambda s: s.update(scancel_fail=True))
    r = _link(tmp_path, (0,), fake=fake)
    assert r.rc == 0 and "WARN: cancelling the chain's jobs failed" in r.out, r.out
    want = "scancel 4242" if not squeue_fail else f"scancel -u {USER} -n {CHAIN}"
    assert f"cancel them by hand: {want}" in r.out, r.out


@pytest.mark.parametrize("how, state", [("scancel -u $USER -n $SLURM_JOB_NAME -t PENDING", "CANCELLED"),
                                        ("fake_purge 4242", "no longer known to Slurm")])
def test_link_successor_cancelled_by_hand_stops_the_chain(tmp_path, how, state):
    """Someone cancelled the queued successor while the link ran (scancel, and maybe Slurm has forgotten it since):
    nothing replaces it."""
    r = _link(tmp_path, (75,), (1,), body=how)
    assert r.rc == 75 and f"successor 4242 was cancelled ({state}): chain stopped" in r.out, r.out
    assert len(r.fake.calls("sbatch")) == 1 and _events(r)[-1][1]["action"] == "stop", (r.out, _events(r))


def test_link_assumes_the_successor_is_queued_when_squeue_fails(tmp_path):
    fake = _fake_slurm(tmp_path, squeue_fail=["%T"])
    r = _link(tmp_path, (75,), (1,), fake=fake)
    assert r.rc == 75 and "cannot check the successor 4242 (squeue failed): assuming it is still queued" in r.out, r.out
    assert len(fake.calls("sbatch")) == 1 and _events(r)[-1][1]["action"] == "continue", _events(r)


def test_link_time_limit_shrunk_after_the_start(tmp_path):
    """`scontrol update TimeLimit=` moved the end: the end is read again from squeue when the signal comes, so its
    SIGTERM is the time limit (the successor stays), although SLURM_JOB_END_TIME still says 2 h."""
    rc, out, err, fake = _job_sigterm(tmp_path, "shell", end_in=7200, fake_end=_iso(int(time.time()) + 10))
    assert rc == 75 and "CMD saved" in out and "the time limit" in out, (rc, out, err)
    assert fake.job(4242)["state"] == "PENDING" and fake.calls("scancel") == [], out


def test_link_sigterm_with_an_unknown_end_is_the_time_limit(tmp_path):
    rc, out, err, fake = _job_sigterm(tmp_path, "shell", fake_end="Unknown", SLURM_JOB_END_TIME="")
    assert rc == 75 and "WARN: no end time" in out and "taken as the time limit" in out, (rc, out, err)
    assert fake.job(4242)["state"] == "PENDING" and fake.calls("scancel") == [], out


def test_link_signal_before_the_command_starts(tmp_path):
    """A scancel during the link's setup (here: while sbatch queues the successor): the command is not started, the
    chain stops (its new successor is cancelled)."""
    fake = _fake_slurm(tmp_path, sbatch=["slow:3"])
    env = _job_env(tmp_path, fake, SLURM_JOB_END_TIME=int(time.time()) + 7200)
    script = _cmd(tmp_path, (75,))
    p = subprocess.Popen(["bash", str(JOB), str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=env, start_new_session=True)
    t0 = time.time()
    while not fake.calls("sbatch"):
        assert p.poll() is None and time.time() - t0 < 30, "sbatch never ran"
        time.sleep(0.05)
    os.kill(p.pid, signal.SIGTERM)
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 143 and "CMD" not in out, out
    assert "SIGTERM before the command started: not running it" in out and "the chain is stopped" in out, out
    assert fake.job(4242)["state"] == "CANCELLED" and len(fake.calls("sbatch")) == 1, out


# --------------------------------------------------------------------------------------------- failures and progress
def test_link_retries_a_failure_in_place(tmp_path):
    """A crash with time left before the deadline: the command runs again at once in the same job (no queue wait)."""
    r = _link(tmp_path, (1, 75), (0, 1))
    assert r.rc == 75 and r.out.count("CMD n=") == 2, r.out
    assert "exit 1 (progress: 0): retrying in place in 0 s (retry 1 of DM_INPLACE_RETRIES=1" in r.out, r.out
    att = [kv for e, kv in _events(r) if e == "attempt"]
    assert att == [{"n": "1", "rc": "1", "progress": "0", "fails": "1"}, {"n": "2", "rc": "75", "progress": "1", "fails": "0"}]
    assert len(r.fake.calls("sbatch")) == 1 and r.fake.calls("scontrol") == [], r.out   # no node exclusion after a 75


@pytest.mark.parametrize("extra", [{"DM_RETRY_MIN_LEFT": 10 ** 10}, {"DM_INPLACE_RETRIES": 0}, {"DM_RESUBMIT": 0},
                                   {"end": ""}])
def test_link_does_not_retry_in_place(tmp_path, extra):
    """Not with less than DM_RETRY_MIN_LEFT before the deadline, no retries allowed, a single job, or no deadline."""
    extra = dict(extra)
    fake = _fake_slurm(tmp_path, end=extra.pop("end", "2030-01-01T12:00:00"))
    r = _link(tmp_path, (1, 75), (0, 1), fake=fake, **extra)
    assert r.rc == 1 and r.out.count("CMD n=") == 1 and "retrying in place" not in r.out, r.out


def test_link_counts_failures_and_gives_up(tmp_path):
    """Attempts without progress add up over the links (`links`: the last failure count); one more than DM_MAX_FAILS
    gives up: exit 4, the pending jobs cancelled, the way to resume printed."""
    _links(tmp_path, "100 41 start k=0 fails=0", "101 41 attempt n=1 rc=1 progress=0 fails=1",
           "102 41 end k=0 rc=1 action=continue fails=2 succ=99")
    r = _link(tmp_path, (1,), (0,), DM_RESUBMIT_COUNT=1)
    assert r.rc == 4 and r.out.count("CMD n=") == 1, r.out   # 3 > 2: no retry in place either
    assert "giving up after 3 attempts without progress (last exit 1); resume with" in r.out, r.out
    assert f"{_chain_dir(tmp_path)}/submit.sh" in r.out and r.fake.job(4242)["state"] == "CANCELLED", r.out
    assert _events(r)[-1][1] == {"k": "1", "rc": "1", "action": "giveup", "fails": "3"}, _events(r)


def test_link_failure_with_progress_continues(tmp_path):
    """A crash after a new checkpoint is not a failure without progress: the count goes back to 0."""
    _links(tmp_path, "102 41 end k=0 rc=1 action=continue fails=2 succ=99")
    r = _link(tmp_path, (1,), (1,), DM_INPLACE_RETRIES=0)
    assert r.rc == 1 and _events(r)[-1][1]["action"] == "continue" and _events(r)[-1][1]["fails"] == "0", _events(r)
    assert r.fake.calls("scontrol") == [] and r.fake.job(4242)["state"] == "PENDING", r.out


def test_link_progress_resets_the_failure_count(tmp_path):
    _links(tmp_path, "102 41 end k=0 rc=1 action=continue fails=2 succ=99")
    r = _link(tmp_path, (75,), (1,))
    ev = [(e, kv["fails"]) for e, kv in _events(r)[1:] if "fails" in kv]
    assert r.rc == 75 and ev == [("start", "2"), ("attempt", "0"), ("end", "0")], _events(r)


def test_link_75_without_progress_counts(tmp_path):
    """An exit 75 that saved nothing new is a link without progress too (no endless chain of empty links)."""
    _links(tmp_path, "102 41 end k=0 rc=75 action=continue fails=2 succ=99")
    r = _link(tmp_path, (75,), (0,))
    assert r.rc == 4 and "giving up after 3 attempts without progress (last exit 75)" in r.out, r.out


@pytest.mark.parametrize("progress, fails", [(True, "0"), (False, "2")])
def test_link_judges_a_predecessor_that_left_no_record(tmp_path, progress, fails):
    """Job 41 started (fails=1 before it) and never wrote its end (SIGKILL after KillWait, a lost node): it progressed
    iff `progress` has a line of job 41 since its start."""
    _links(tmp_path, "1000 41 start k=0 restart=0 node=n1 fails=1", "1001 41 succ id=99 how=queued")
    _progress(tmp_path, "999 41 ckpt RUN 50", "1000 40 ckpt RUN 100")   # before its start / another job's
    if progress:
        _progress(tmp_path, "1500 41 ckpt RUN 200")
    r = _link(tmp_path, (75,), (1,), DM_RESUBMIT_COUNT=1)
    assert r.rc == 75, r.out
    ev = _events(r)
    assert ev[2] == ("judge", {"of": "41", "progress": "1" if progress else "0", "fails": fails}), ev
    assert ev[3][0] == "start" and ev[3][1]["fails"] == fails, ev
    assert f"the chain's previous link (job 41) ended without a record" in r.out, r.out


def test_link_gives_up_at_the_start_after_killed_links(tmp_path):
    """The third link in a row that died without progress: the next one gives up before running anything."""
    fake = _fake_slurm(tmp_path)
    pending = fake.add_job()
    _links(tmp_path, "1000 41 start k=2 restart=0 node=n1 fails=2")
    r = _link(tmp_path, (0,), fake=fake, DM_RESUBMIT_COUNT=3)
    assert r.rc == 4 and "CMD" not in r.out and fake.calls("sbatch") == [], r.out
    assert "giving up: 3 links in a row without progress" in r.out and fake.job(pending)["state"] == "CANCELLED", r.out
    assert [e for e, _ in _events(r)][-2:] == ["judge", "end"] and _events(r)[-1][1]["action"] == "giveup", _events(r)


def test_link_requeued_judges_itself_and_keeps_its_successor(tmp_path):
    """A requeue restarts job 99 (same id): it judges its own first run and adopts the successor that run queued."""
    fake = _fake_slurm(tmp_path)
    succ = fake.add_job()
    _links(tmp_path, f"1000 99 start k=3 restart=0 node=n1 fails=0", f"1001 99 succ id={succ} how=queued")
    r = _link(tmp_path, (75,), (1,), fake=fake, DM_RESUBMIT_COUNT=3, SLURM_RESTART_COUNT=1)
    assert r.rc == 75 and fake.calls("sbatch") == [], r.out
    ev = _events(r)
    assert ev[2] == ("judge", {"of": "99", "progress": "0", "fails": "1"}) and ev[3][1]["restart"] == "1", ev
    assert ev[-1][1] == {"k": "3", "rc": "75", "action": "continue", "fails": "0", "succ": str(succ)}, ev


@pytest.mark.parametrize("given, want", [("", "node7"), ("-x node1", "node1,node7"), ("--exclude=node7", None)])
def test_link_excludes_its_node_after_a_failure_without_progress(tmp_path, given, want):
    """The successor of a crash without progress avoids this node (its own exclusions kept)."""
    r = _link(tmp_path, (1,), (0,), DM_INPLACE_RETRIES=0, DM_SBATCH_OPTS=f"{OPTS} {given}".strip())
    assert r.rc == 1 and r.fake.job(4242)["state"] == "PENDING", r.out
    if want is None:   # already excluded
        assert r.fake.calls("scontrol") == [] and r.fake.job(4242)["exc"] == "node7", r.out
    else:
        assert [c["argv"] for c in r.fake.calls("scontrol")] == [["update", "JobId=4242", f"ExcNodeList={want}"]]
        assert r.fake.job(4242)["exc"] == want and f"excluded node node7 from the successor 4242" in r.out, r.out


def test_link_unwritable_state_directory_costs_only_the_accounting(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        r = _link(tmp_path, (75,), (1,), DM_CHAIN_DIR=ro / CHAIN)
    finally:
        ro.chmod(0o755)
    assert r.rc == 75 and "CMD n=1" in r.out and "cannot be written" in r.out, r.out
    assert r.fake.job(4242)["state"] == "PENDING", r.out


def test_link_rederives_a_state_directory_of_another_chain(tmp_path):
    r = _link(tmp_path, (75,), (1,), DM_CHAIN_DIR=tmp_path / "ws" / "logs" / "slurm" / "chains" / "dm-other")
    assert r.rc == 75 and f"dir={_chain_dir(tmp_path)} " in r.out and "is not the state directory" in r.out, r.out


# ----------------------------------------------------------------------------------------------------- compile caches
def _jit(tmp_path):
    return tmp_path / "jit" / CHAIN


def _tar(tmp_path, files):
    """The chain's jit.tar holding `files`, written an hour ago (by an earlier link)."""
    src = tmp_path / "jitsrc"
    for name, data in files.items():
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_text(data)
    old = time.time() - 3600
    for p in [src, *src.rglob("*")]:
        os.utime(p, (old, old))
    d = _chain_dir(tmp_path)
    d.mkdir(parents=True, exist_ok=True)
    with tarfile.open(d / "jit.tar", "w") as t:
        t.add(src, arcname=".")
    return d / "jit.tar"


def _members(tar):
    with tarfile.open(tar) as t:
        return sorted(m.name for m in t.getmembers() if m.isfile())


SHOW = 'for f in triton/r0/k.bin tilelang/t.bin; do [ -f "$TRITON_CACHE_DIR/../$f" ] && echo "HAVE $f"; done'


def test_jit_caches_restored_and_saved_when_changed(tmp_path):
    tar = _tar(tmp_path, {"triton/r0/k.bin": "k", "tilelang/t.bin": "t"})
    (_chain_dir(tmp_path) / "jit.tar.tmp.41").write_text("left by a killed link")
    r = _link(tmp_path, (75,), (1,), body=SHOW + '\necho new > "$TRITON_CACHE_DIR/new.bin"')
    assert r.rc == 75 and "HAVE triton/r0/k.bin" in r.out and "HAVE tilelang/t.bin" in r.out, r.out
    j = _jit(tmp_path)
    assert f"triton={j}/triton tilelang={j}/tilelang inductor={j}/inductor" in r.out, r.out
    assert "compile caches saved to" in r.out, r.out
    assert _members(tar) == ["./tilelang/t.bin", "./triton/new.bin", "./triton/r0/k.bin"], _members(tar)
    assert sorted(p.name for p in _chain_dir(tmp_path).iterdir() if p.name.startswith("jit")) == ["jit.tar"]
    assert not j.exists(), "the node-local copy must go at the end of the job"


def test_jit_caches_kept_when_unchanged(tmp_path):
    tar = _tar(tmp_path, {"triton/r0/k.bin": "k"})
    before = (tar.stat().st_mtime_ns, tar.stat().st_ino)
    r = _link(tmp_path, (75,), (1,), body=SHOW)
    assert r.rc == 75 and "HAVE triton/r0/k.bin" in r.out and "compile caches unchanged: jit.tar kept" in r.out, r.out
    assert (tar.stat().st_mtime_ns, tar.stat().st_ino) == before


@pytest.mark.parametrize("rcs, extra, why", [((0,), {}, None), ((75,), {"DM_JIT_MAX_MB": 0}, "DM_JIT_MAX_MB=0"),
                                             ((75,), {"SLURM_JOB_END_TIME": "now+100", "DM_STOP_BEFORE": 1}, "s left")])
def test_jit_caches_not_saved(tmp_path, rcs, extra, why):
    """Not when the chain ends, not beyond DM_JIT_MAX_MB, not with less than 120 s left."""
    extra = dict(extra)
    if extra.get("SLURM_JOB_END_TIME") == "now+100":
        extra["SLURM_JOB_END_TIME"] = int(time.time()) + 100
    r = _link(tmp_path, rcs, (1,), body='mkdir -p "$TRITON_CACHE_DIR" && echo new > "$TRITON_CACHE_DIR/new.bin"', **extra)
    assert r.rc == rcs[0] and not (_chain_dir(tmp_path) / "jit.tar").exists(), r.out
    if why:
        assert f"compile caches not saved" in r.out and why in r.out, r.out


def test_jit_corrupt_archive_starts_cold(tmp_path):
    d = _chain_dir(tmp_path)
    d.mkdir(parents=True)
    (d / "jit.tar").write_text("not a tar file")
    r = _link(tmp_path, (75,), (1,), body='ls -A "$(dirname "$TRITON_CACHE_DIR")" | tr "\\n" " " | sed "s/^/LS /"; echo')
    assert r.rc == 75 and "restoring the compile caches" in r.out and "starting with empty caches" in r.out, r.out
    assert "LS .stamp" in r.out, r.out


def test_jit_cache_off(tmp_path):
    r = _link(tmp_path, (75,), (1,), DM_JIT_CACHE=0, TRITON_CACHE_DIR="/x/triton")
    assert r.rc == 75 and "triton=/x/triton" in r.out and not (tmp_path / "jit").exists(), r.out


@pytest.mark.parametrize("which", ["own", "other", "elsewhere"])
def test_exit_removes_only_the_jobs_own_scratch(tmp_path, which):
    """The site file's node-local scratch /tmp/dm-$SLURM_JOB_ID goes with the job; no other directory."""
    jid = random.randint(10 ** 8, 10 ** 9)
    own, other = f"/tmp/dm-{jid}", f"/tmp/dm-{jid + 1}"
    scratch = {"own": own, "other": other, "elsewhere": str(tmp_path / "scratch")}[which]
    os.makedirs(scratch)
    try:
        fake = FakeSlurm(tmp_path, running={"id": jid, "name": CHAIN})
        r = _link(tmp_path, (0,), fake=fake, SLURM_JOB_ID=jid, TMPDIR=scratch)
        assert r.rc == 0, r.out
        assert os.path.exists(scratch) == (which != "own"), (which, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_link_signal_during_the_in_place_pause(tmp_path):
    """A scancel while the link waits to retry a failure in place: no second attempt, the chain stops."""
    fake = _fake_slurm(tmp_path)
    script = _cmd(tmp_path, (1,))
    p = subprocess.Popen(["bash", str(JOB), str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=_job_env(tmp_path, fake, DM_RETRY_PAUSE=30), start_new_session=True)
    head = []
    for line in p.stdout:   # until the link waits to retry
        head.append(line)
        if "retrying in place in 30 s" in line:
            break
    assert any("retrying in place in 30 s" in l for l in head), "".join(head)
    t0 = time.time()
    os.kill(p.pid, signal.SIGTERM)
    out = "".join(head) + p.communicate(timeout=20)[0]
    assert time.time() - t0 < 20 and p.returncode == 1 and out.count("CMD n=") == 1, out   # not after the 30 s
    assert "stopped by SIGTERM" in out and fake.job(4242)["state"] == "CANCELLED", out


@pytest.mark.parametrize("extra", [{"DM_RESUBMIT": 0}, {"SLURM_JOB_END_TIME": "now+700"}])
def test_single_job_is_outside_the_chain(tmp_path, extra):
    """DM_RESUBMIT=0, or a time limit too short for a chain: a job of its own. Nothing queued or cancelled (other jobs of
    the same name stay pending), no records, no compile caches, and the command sees no chain (the experiment driver
    would count its failures in runfails/ and exit 75 for a retry that never comes)."""
    extra = {k: (int(time.time()) + 700 if v == "now+700" else v) for k, v in extra.items()}
    fake = _fake_slurm(tmp_path)
    other = fake.add_job()
    r = _link(tmp_path, (1,), fake=fake, DM_CHAIN_DIR=_chain_dir(tmp_path), DM_PROGRESS_FILE="/x", DM_CHAIN="x",
              **extra)
    assert r.rc == 1 and r.out.count("CMD n=") == 1 and fake.calls("sbatch") == [] and fake.calls("scancel") == [], r.out
    assert "chain=unset dir=unset pf=unset" in r.out and f"triton={tmp_path / 'jit'}" not in r.out, r.out
    assert fake.job(other)["state"] == "PENDING" and r.records() == [], r.records()


def test_job_outside_slurm_runs_the_command_once(tmp_path):
    """By hand (no SLURM_JOB_ID): the command, its exit status, nothing queued or written."""
    fake = _fake_slurm(tmp_path)
    script = _cmd(tmp_path, (75,))
    env = _job_env(tmp_path, fake)
    env.pop("SLURM_JOB_ID")
    p = subprocess.run(["bash", str(JOB), str(script)], capture_output=True, text=True, timeout=60, env=env)
    assert p.returncode == 75 and p.stdout.count("CMD n=") == 1 and "chain=unset" in p.stdout, p.stdout + p.stderr
    assert fake.calls() == [] and not (tmp_path / "ws").exists(), fake.calls()


def test_job_without_user_name_falls_back_to_id(tmp_path):
    r = _link(tmp_path, (0,), USER="")
    assert r.rc == 0 and len(r.fake.calls("sbatch")) == 1 and r.fake.job(4242)["state"] == "CANCELLED", r.out


def test_link_stop_while_its_end_queues_the_successor(tmp_path):
    """A scancel while the end of the link queues the successor (the start could not): the new job would outlive the
    user's `scancel -n`, so the link cancels it."""
    fake = _fake_slurm(tmp_path, sbatch=["qos"] * 3 + ["slow:3"])
    script = _cmd(tmp_path, (75,), (1,))
    p = subprocess.Popen(["bash", str(JOB), str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=_job_env(tmp_path, fake, SLURM_JOB_END_TIME=int(time.time()) + 7200),
                         start_new_session=True)
    t0 = time.time()
    while len(fake.calls("sbatch")) < 4:
        assert p.poll() is None and time.time() - t0 < 30, "no sbatch at the end"
        time.sleep(0.05)
    os.kill(p.pid, signal.SIGTERM)
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 75 and "stopped by SIGTERM" in out and "the chain is stopped" in out, out
    assert fake.job(4242)["state"] == "CANCELLED", out
    ends = [l.split()[5] for l in (_chain_dir(tmp_path) / "links").read_text().splitlines() if l.split()[2] == "end"]
    assert ends == ["action=stop"], ends
