"""The Slurm wrappers of scripts/slurm (README there). submit.sh: its sbatch defaults, the options that override them (every
spelling sbatch 24.05's getopt accepts: abbreviations, attached values, values starting with '-', grouped flags, SBATCH_*
variables), the chain's name (a hash of the command as the job runs it, the workspace's physical path and the knobs),
--dependency=singleton, the first job's own options, its refusals, the chain's state directory and its submit.sh recipe
(none for a single job), a new chain started afresh from inside a link, the WARN about the same work under another name
(dry mode and a fake sbatch). job.sh: the trainer deadline from the job's end time (none when the end is unknown, and
neither a deadline nor a successor when the time limit is too short), DM_JOB_END, the successor queued at the start of
the job (exact arguments, from the submit directory, retried, a broken chain reported), malformed numeric knobs, the site
file, and Slurm's SIGTERM (job.sh outlives it, lets the command save, and keeps the successor only at the time limit).
The chain's life (failure accounting, in-place retries, adoption of pending jobs, cancellations, compile caches) is
tested in test_slurm_chain.py, how a link tells and handles the ways a job ends in test_slurm_chain_stops.py.

No Slurm needed: FakeSlurm puts fake `sbatch` / `squeue` / `scancel` / `scontrol` executables on PATH that share one
queue file (jobs with their id, name, user, state, the COMPLETING flag, start / end time, time limit, excluded nodes,
argv, working directory and environment) and record every call. Like slurmctld, its scancel of a RUNNING job makes it
CANCELLED + COMPLETING with the end time of the cancel (squeue's %e is then no guide to the time limit), and squeue's
-t COMPLETING matches the flag whatever the base state (a requeued job is PENDING + COMPLETING)."""
import hashlib, json, os, pwd, re, shlex, signal, socket, stat, subprocess, sys, textwrap, time
import fcntl
import pytest
from _helpers import REPO, clean_env

SUBMIT = REPO / "scripts" / "slurm" / "submit.sh"
JOB = REPO / "scripts" / "slurm" / "job.sh"
CHAIN = "dm-test-chain"            # SLURM_JOB_NAME of the job.sh runs
ME = pwd.getpwuid(os.getuid()).pw_name   # the owner of the fake's jobs
OPTS = "-p batch -t 4:00:00 --dependency=singleton"   # their DM_SBATCH_OPTS (as submit.sh records them)
END = "2030-01-01T12:00:00"
LIMIT = 4 * 3600                   # the fake running job's time limit: its start is END - LIMIT


def _env(tmp_path, **extra):
    env = clean_env()
    for k in list(env):
        if k.startswith(("SLURM_", "SBATCH_", "SQUEUE_", "SCANCEL_")):
            env.pop(k)
    env.update({"DM_WORKSPACE": str(tmp_path / "ws")})
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _dry(tmp_path, *args, cwd=None, **extra):
    return subprocess.run(["bash", str(SUBMIT), *args], capture_output=True, text=True, timeout=60, cwd=cwd,
                          env=_env(tmp_path, DM_SUBMIT_DRY=1, **extra))


def _sbatch_words(out):
    line = next(l for l in out.splitlines() if l.startswith("DRY sbatch"))
    return shlex.split(line)[3:]   # after "DRY sbatch --parsable" (the arguments are printed %q-quoted)


def _dry_opts(out):
    return re.search(r"DM_SBATCH_OPTS=(.*)", out).group(1).split()


def _dry_chain(out):
    m = re.search(r"^DRY chain=(\S+) dir=(\S+)$", out, re.M)
    return m.group(1), m.group(2)


def _epoch(iso):
    return int(subprocess.run(["date", "-d", iso, "+%s"], capture_output=True, text=True).stdout)


def _iso(epoch):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(epoch))


# ------------------------------------------------------------------------------------------------------------ fake Slurm
_FAKE = r'''
"""Fake sbatch / squeue / scancel / scontrol sharing one queue (state.json); every call goes to calls.jsonl first."""
import fcntl, json, os, pwd, sys, time
D, TOOL, ARGS = sys.argv[1], sys.argv[2], sys.argv[3:]
ME = pwd.getpwuid(os.getuid()).pw_name   # Slurm records the submitting uid, whatever USER says
lock = open(os.path.join(D, "lock"), "a")
fcntl.flock(lock, fcntl.LOCK_EX)
SP = os.path.join(D, "state.json")
with open(SP) as f:
    S = json.load(f)
with open(os.path.join(D, "calls.jsonl"), "a") as f:
    f.write(json.dumps({"tool": TOOL, "argv": ARGS, "cwd": os.getcwd(), "t": time.time()}) + "\n")
ACTIVE = ("PENDING", "RUNNING", "SUSPENDED", "CONFIGURING", "COMPLETING")
SHORT = {"PD": "PENDING", "R": "RUNNING", "S": "SUSPENDED", "CF": "CONFIGURING", "CG": "COMPLETING", "CA": "CANCELLED",
         "CD": "COMPLETED", "F": "FAILED", "TO": "TIMEOUT"}
for j in S["jobs"]:   # a COMPLETING job whose processes are gone by now
    if j.get("complete_at") is not None and time.time() >= j["complete_at"]:
        j["completing"], j["complete_at"] = False, None


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


def shown(j):   # squeue's %T: COMPLETING (a flag) hides the base state
    return "COMPLETING" if j.get("completing") else j["state"]


def matches(j, want):   # squeue / scancel -t: the base state, or the COMPLETING flag
    return j["state"] in want or ("COMPLETING" in want and bool(j.get("completing")))


def done(rc=0, out="", err=""):
    tmp = SP + ".tmp"
    with open(tmp, "w") as f:
        json.dump(S, f)
    os.replace(tmp, SP)
    sys.stdout.write(out)
    sys.stderr.write(err)
    sys.exit(rc)


def parse(args, valued, flags=()):
    """{key: value} for the options (-x v, -xv, --long v, --long=v) and the other words."""
    out, pos, i = {}, [], 0
    while i < len(args):
        a = args[i]
        i += 1
        if a in flags:
            out[flags[a]] = True
        elif a.startswith("--") and "=" in a and a.split("=", 1)[0] in valued:
            out[valued[a.split("=", 1)[0]]] = a.split("=", 1)[1]
        elif a in valued:
            out[valued[a]] = args[i] if i < len(args) else ""
            i += 1
        elif not a.startswith("--") and a[:2] in valued and len(a) > 2:
            out[valued[a[:2]]] = a[2:]
        else:
            pos.append(a)
    return out, pos


def states(spec):
    return {SHORT.get(s.upper(), s.upper()) for s in spec.split(",")} if spec else None


if TOOL == "sbatch":
    mode = S["sbatch_modes"].pop(0) if S["sbatch_modes"] else S["sbatch_default"]
    if mode == "fail":
        done(1, err="sbatch: error: Batch job submission failed: Socket timed out on send/recv operation\n")
    if mode == "qos":
        done(1, err="sbatch: error: QOSMaxSubmitJobPerUserLimit\nsbatch: error: Batch job submission failed: Job "
                    "violates accounting/QOS policy (job submit limit, user's size and/or time limits)\n")
    script = next((i for i, a in enumerate(ARGS) if a.endswith("/scripts/slurm/job.sh")), len(ARGS))
    o, _ = parse(ARGS[:script], {"-J": "name", "--job-name": "name", "-x": "exc", "--exclude": "exc",
                                 "-d": "dep", "--dependency": "dep"}, {"-H": "hold", "--hold": "hold"})
    jid = S["next_id"]
    S["next_id"] += 1
    S["jobs"].append({"id": jid, "name": o.get("name", "job.sh"), "user": ME,
                      "state": "PENDING", "end": "Unknown", "start": "N/A", "limit": "4:00:00", "exc": o.get("exc", ""),
                      "argv": ARGS, "cwd": os.getcwd(), "env": dict(os.environ)})
    if mode == "ambiguous":   # queued, but the reply timed out
        done(1, err="sbatch: error: Batch job submission failed: Socket timed out on send/recv operation\n")
    if mode == "junk":
        done(0, out="sbatch: something else\n")
    if mode.startswith("slow:"):   # queued, then a slow reply (the queue is unlocked meanwhile)
        tmp = SP + ".tmp"
        with open(tmp, "w") as f:
            json.dump(S, f)
        os.replace(tmp, SP)
        lock.close()
        time.sleep(float(mode[5:]))
        print(jid)
        sys.exit(0)
    done(0, out=f"{jid}\n" if "--parsable" in ARGS else f"Submitted batch job {jid}\n")

if TOOL == "squeue":
    joined = " ".join(ARGS)
    if any(s in joined for s in S["squeue_fail"]):
        done(1, err="squeue: error: Socket timed out on send/recv operation\n")
    o, pos = parse(ARGS, {"-u": "user", "--user": "user", "-n": "name", "--name": "name", "-t": "states",
                          "--states": "states", "-o": "fmt", "--format": "fmt", "-j": "jobs", "--jobs": "jobs",
                          "-O": "longfmt", "--Format": "longfmt"},
                   {"-h": "noheader", "--noheader": "noheader"})
    jobs = S["jobs"]
    want = states(o.get("states"))
    if "jobs" in o:
        ids = {int(x) for x in o["jobs"].split(",") if x.strip().isdigit()}
        jobs = [j for j in jobs if j["id"] in ids]
        if not jobs:
            done(1, err="slurm_load_jobs error: Invalid job id specified\n")
    elif want is None:
        want = set(ACTIVE)
    if want is not None and "ALL" not in want:
        jobs = [j for j in jobs if matches(j, want)]
    if "user" in o:
        jobs = [j for j in jobs if j["user"] in o["user"].split(",")]
    if "name" in o:
        jobs = [j for j in jobs if j["name"] in o["name"].split(",")]
    if "longfmt" in o:   # -O: the fields this fake knows (RestartCnt: the job's restart count), padded like squeue
        known = {"restartcnt": lambda j: str(j.get("restarts", 0)), "jobid": lambda j: str(j["id"]),
                 "state": lambda j: shown(j)}
        fields = [f.split(":")[0].lower() for f in o["longfmt"].split(",")]
        if any(f not in known for f in fields):
            done(1, err=f"squeue: error: Invalid job format specification: {o['longfmt']}\n")
        done(0, out="".join("".join(known[f](j).ljust(20) for f in fields) + "\n" for j in jobs))
    fmt = o.get("fmt", "%i %j %T")
    lines = [] if o.get("noheader") else ["JOBID NAME STATE"]
    for j in jobs:
        line = fmt
        for k, v in (("%i", str(j["id"])), ("%j", j["name"]), ("%T", shown(j)), ("%e", j["end"]),
                     ("%S", j.get("start", "N/A")), ("%l", j.get("limit", "4:00:00")),
                     ("%x", j["exc"] or "(null)"), ("%u", j["user"])):
            line = line.replace(k, v)
        lines.append(line)
    done(0, out="".join(l + "\n" for l in lines))

if TOOL == "scancel":
    if S["scancel_fail"]:
        done(1, err="scancel: error: Socket timed out on send/recv operation\n")
    o, pos = parse(ARGS, {"-u": "user", "--user": "user", "-n": "name", "--name": "name", "-t": "state",
                          "--state": "state", "-s": "signal", "--signal": "signal", "-p": "partition"},
                   {"-b": "batch", "--batch": "batch", "-f": "full"})
    ids = {int(x) for x in pos if x.isdigit()}
    if not ids and not any(k in o for k in ("user", "name", "state", "partition")):
        done(1, err="scancel: error: No job identification provided\n")
    want = states(o.get("state"))
    for j in S["jobs"]:
        if (not ids or j["id"] in ids) and (j["state"] in ACTIVE) and (want is None or matches(j, want)) \
                and ("user" not in o or j["user"] == o["user"]) and ("name" not in o or j["name"] == o["name"]):
            if j["state"] in ("RUNNING", "SUSPENDED", "CONFIGURING"):   # slurmctld: end_time = now, COMPLETING
                j["completing"], j["end"] = True, iso(time.time())
            j["state"] = "CANCELLED"
    done(0)

if TOOL == "set":   # fake_set <id> key=value...: change a job (state, completing=0/1, end, start, limit, complete_at)
    job = next((j for j in S["jobs"] if str(j["id"]) == ARGS[0]), None)
    if job is None:
        done(1, err=f"fake_set: no job {ARGS[0]}\n")
    for a in ARGS[1:]:
        k, v = a.split("=", 1)
        job[k] = (v == "1") if k == "completing" else (float(v) if k == "complete_at" else v)
    done(0)

if TOOL == "scontrol":
    if S["scontrol_fail"]:
        done(1, err="scontrol: error: Socket timed out on send/recv operation\n")
    if ARGS[:1] == ["update"]:
        kv = dict(a.split("=", 1) for a in ARGS[1:] if "=" in a)
        kv = {k.lower(): v for k, v in kv.items()}
        job = next((j for j in S["jobs"] if str(j["id"]) == kv.get("jobid")), None)
        if job is None:
            done(1, err="scontrol: error: Invalid job id specified\n")
        if "excnodelist" in kv:
            job["exc"] = kv["excnodelist"]
    done(0)
if TOOL == "purge":   # Slurm forgets a job (MinJobAge after its end)
    S["jobs"] = [j for j in S["jobs"] if str(j["id"]) not in ARGS]
    done(0)
done(2, err=f"fake slurm: unknown tool {TOOL}\n")
'''


class FakeSlurm:
    """The queue (state.json) and the calls (calls.jsonl) of the fake Slurm tools in `bin`. `running`: this job.sh's own
    job (id, name, end: what squeue prints for %e; its start (%S) is end - LIMIT and its limit (%l) 4:00:00 unless given;
    an end that is not an ISO time gives start N/A, limit UNLIMITED; restarts: squeue's RestartCnt, default 0); `sbatch`: the modes of the next sbatch calls (ok,
    fail = nothing queued, ambiguous = queued but reported as a timeout, qos = refused by a submit limit, junk = no job
    id printed, slow:<s> = queued, the reply after s seconds), then `sbatch_default`; `squeue_fail`: squeue fails when
    its arguments contain any of these strings ("" = always). In bin: `fake_purge <id>...` makes Slurm forget jobs,
    `fake_set <id> key=value...` changes one (state, completing=0/1, end, start, limit, complete_at=<epoch>: the
    COMPLETING flag clears then)."""

    def __init__(self, tmp_path, running=None, sbatch=(), sbatch_default="ok", squeue_fail=(), first_id=4242):
        self.d = tmp_path / "fake"
        self.bin = tmp_path / "bin"
        self.d.mkdir(exist_ok=True)
        self.bin.mkdir(exist_ok=True)
        (self.d / "calls.jsonl").unlink(missing_ok=True)
        jobs = []
        if running:
            end = running.get("end", END)
            iso_end = bool(re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", end or ""))
            jobs.append({"id": int(running.get("id", 99)), "name": running.get("name", CHAIN),
                         "user": ME, "state": running.get("state", "RUNNING"), "completing": False, "end": end,
                         "start": running.get("start", _iso(_epoch(end) - LIMIT) if iso_end else "N/A"),
                         "limit": running.get("limit", "4:00:00" if iso_end else "UNLIMITED"),
                         "restarts": int(running.get("restarts", 0)), "exc": "", "argv": [], "cwd": "", "env": {}})
        self._write({"next_id": first_id, "jobs": jobs, "sbatch_modes": list(sbatch), "sbatch_default": sbatch_default,
                     "squeue_fail": list(squeue_fail), "scancel_fail": False, "scontrol_fail": False})
        (self.d / "fake_slurm.py").write_text(_FAKE)
        for tool in ("sbatch", "squeue", "scancel", "scontrol", "purge", "set"):
            f = self.bin / (tool if tool not in ("purge", "set") else "fake_" + tool)
            f.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} -I -S {shlex.quote(str(self.d / "fake_slurm.py"))} '
                         f'{shlex.quote(str(self.d))} {tool} "$@"\n')
            f.chmod(f.stat().st_mode | stat.S_IEXEC)

    def _write(self, s):
        tmp = self.d / "state.json.tmp"
        tmp.write_text(json.dumps(s))
        os.replace(tmp, self.d / "state.json")

    def state(self):
        return json.loads((self.d / "state.json").read_text())

    def update(self, fn):
        with open(self.d / "lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            s = self.state()
            fn(s)
            self._write(s)

    def add_job(self, name=CHAIN, state="PENDING", **kw):
        out = {}

        def add(s):
            out["id"] = s["next_id"]
            s["next_id"] += 1
            s["jobs"].append({"id": out["id"], "name": name, "user": ME, "state": state, "completing": False,
                              "end": "Unknown", "start": "N/A", "limit": "4:00:00", "exc": "", "argv": [], "cwd": "",
                              "env": {}, **kw})
        self.update(add)
        return out["id"]

    def jobs(self, **match):
        return [j for j in self.state()["jobs"] if all(j.get(k) == v for k, v in match.items())]

    def job(self, jid):
        return next(j for j in self.state()["jobs"] if j["id"] == jid)

    def calls(self, tool=None):
        f = self.d / "calls.jsonl"
        out = [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
        return [c for c in out if tool is None or c["tool"] == tool]

    def sbatch_argvs(self):
        return [c["argv"] for c in self.calls("sbatch")]


def _fake_slurm(tmp_path, end=END, squeue_rc=0, sbatch_fails=0, **kw):
    """A fake whose running job (99, CHAIN) ends at `end`; squeue fails throughout when squeue_rc; the first
    `sbatch_fails` sbatch calls fail with nothing queued (FakeSlurm's own arguments win)."""
    kw = {"sbatch": ["fail"] * sbatch_fails, "squeue_fail": [""] if squeue_rc else [], **kw}
    return FakeSlurm(tmp_path, running={"id": 99, "name": CHAIN, "end": end}, **kw)


# ------------------------------------------------------------------------------------------------------- job.sh runner
_CMD = """\
#!/bin/bash
# attempt n of the command: exit rcs[n] (the last one repeats) after writing a progress line when progs[n] is 1
n=$(( $(cat {cnt} 2>/dev/null || echo 0) + 1 )); echo $n > {cnt}
echo "CMD n=$n args=$* deadline=${{DM_STOP_BEFORE:-none}} end=${{DM_JOB_END:-none}} site=${{SITE_SOURCED:-no}} cwd=$PWD nproc=${{DM_NPROC:-unset}} launch=${{DM_LAUNCH:-unset}} port=${{DM_MASTER_PORT:-unset}}/${{DM_VALCE_PORT:-unset}} k=${{DM_RESUBMIT_COUNT:-unset}} chain=${{DM_CHAIN:-unset}} dir=${{DM_CHAIN_DIR:-unset}} pf=${{DM_PROGRESS_FILE:-unset}} triton=${{TRITON_CACHE_DIR:-unset}} tilelang=${{TILELANG_CACHE_DIR:-unset}} inductor=${{TORCHINDUCTOR_CACHE_DIR:-unset}}"
rcs=({rcs}); progs=({progs})
rc=${{rcs[$((n - 1))]:-${{rcs[-1]}}}}; p=${{progs[$((n - 1))]:-${{progs[-1]}}}}
[ "$p" = 1 ] && echo "$(date +%s) ${{SLURM_JOB_ID:-0}} ckpt RUN $((n * 100))" >> "$DM_PROGRESS_FILE"
{body}
exit $rc
"""


class Link:
    """One job.sh run: p (the CompletedProcess), out (stdout + stderr), fake, chain_dir, and the chain's records."""

    def __init__(self, p, fake, chain_dir):
        self.p, self.fake, self.chain_dir = p, fake, chain_dir
        self.rc = p.returncode
        self.out = p.stdout + p.stderr

    def records(self, event=None):
        f = self.chain_dir / "links"
        recs = [l.split() for l in f.read_text().splitlines()] if f.exists() else []
        return [r for r in recs if event is None or r[2] == event]

    def kv(self, rec):
        return dict(w.split("=", 1) for w in rec[3:] if "=" in w)


def _cmd(tmp_path, rcs=(0,), progs=(0,), body=""):
    script = tmp_path / "cmd.sh"
    (tmp_path / "cmd.count").unlink(missing_ok=True)
    script.write_text(_CMD.format(cnt=tmp_path / "cmd.count", rcs=" ".join(map(str, rcs)),
                                  progs=" ".join(map(str, progs)), body=body))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _job_env(tmp_path, fake, **extra):
    site = tmp_path / "site.sh"
    if not site.exists():
        site.write_text("export SITE_SOURCED=yes\n")
    env = dict(PATH=f"{fake.bin}:{os.environ['PATH']}", SLURM_JOB_ID="99", SLURM_JOB_NAME=CHAIN, SLURMD_NODENAME="node7",
               DM_SITE_ENV=site, DM_SBATCH_OPTS=OPTS, DM_JIT_ROOT=tmp_path / "jit", DM_RETRY_PAUSE=0,
               DM_RESUBMIT_BACKOFF="0 0")
    env.update(extra)
    return _env(tmp_path, **env)


def _link(tmp_path, rcs=(0,), progs=(0,), args=("a1", "a2"), fake=None, body="", **extra):
    """Run job.sh <cmd.sh> <args> as job 99 of the chain CHAIN (DM_WORKSPACE=tmp/ws) under `fake` (default: a fake whose
    running job ends at END), the command exiting rcs[n] in its attempt n."""
    fake = fake or _fake_slurm(tmp_path)
    script = _cmd(tmp_path, rcs, progs, body)
    p = subprocess.run(["bash", str(JOB), str(script), *args], capture_output=True, text=True, timeout=120,
                       env=_job_env(tmp_path, fake, **extra))
    return Link(p, fake, tmp_path / "ws" / "logs" / "slurm" / "chains" / CHAIN)


def _job(tmp_path, cmd_rc, args=("a1", "a2"), end=END, squeue_rc=0, sbatch_fails=0, **extra):
    """job.sh with a command exiting cmd_rc: (the process, the sbatch calls' arguments, their working directories)."""
    fake = _fake_slurm(tmp_path, end, squeue_rc, sbatch_fails)
    r = _link(tmp_path, (cmd_rc,), args=args, fake=fake, **extra)
    return r.p, fake.sbatch_argvs(), [c["cwd"] for c in fake.calls("sbatch")]


# ----------------------------------------------------------------------------------------------------------- submit.sh
def test_submit_defaults_take_the_whole_node(tmp_path):
    p = _dry(tmp_path, "-t", "4:00:00", "--", "scripts/campaign/train_arm.sh", "dm_matched_hd128", "11785",
             DM_SLURM_ACCOUNT="acct", DM_SLURM_PARTITION="batch")
    assert p.returncode == 0, p.stderr
    w = _sbatch_words(p.stdout)
    for opt in ("-t", "--account=acct", "--partition=batch", "--nodes=1", "--gpus-per-node=8", "--mem=0", "--exclusive",
                "--dependency=singleton"):
        assert opt in w, (opt, w)
    assert any(re.fullmatch(r"--job-name=dm-train_arm-dm_matched_hd128-[0-9a-f]{8}", o) for o in w), w
    assert any(o.startswith("--output=") and o.endswith("/logs/slurm/%x-%j.log") for o in w), w
    assert w[-4:] == [str(JOB), "scripts/campaign/train_arm.sh", "dm_matched_hd128", "11785"], w


def test_submit_given_options_win(tmp_path):
    p = _dry(tmp_path, "-A", "x", "-p", "batch_short", "--gpus-per-node=2", "-J", "mine", "-o", "/tmp/o.log",
             "-t", "1:00:00", "--", "scripts/experiments/stage1_hd128.sh", DM_SLURM_ACCOUNT="acct",
             DM_SLURM_PARTITION="batch")
    assert p.returncode == 0, p.stderr
    w = _sbatch_words(p.stdout)
    assert "--account=acct" not in w and "--partition=batch" not in w, w
    assert "--gpus-per-node=8" not in w and "--exclusive" not in w and "--mem=0" not in w, w   # a partial node stays shared
    assert not any(o.startswith("--job-name=") or o.startswith("--output=") for o in w), w
    assert w[:4] == ["-A", "x", "-p", "batch_short"], w
    assert _dry_chain(p.stdout)[0] == "mine", p.stdout   # -J names the chain


@pytest.mark.parametrize("given, default", [
    ("-pbatch_short", "--partition="), ("-Aother", "--account="), ("-Jmine", "--job-name="), ("-o/tmp/o.log", "--output="),
    ("-N1", "--nodes="), ("-G2", "--gpus-per-node="), ("--gpus-per-task=1", "--gpus-per-node="),
    ("--tres-per-task=gres/gpu:1", "--gpus-per-node=")])
def test_submit_recognises_attached_and_other_spellings(tmp_path, given, default):
    p = _dry(tmp_path, given, "-t4:00:00", "--", "scripts/campaign/train_arm.sh", "x", DM_SLURM_ACCOUNT="acct",
             DM_SLURM_PARTITION="b")
    assert p.returncode == 0, p.stderr
    w = _sbatch_words(p.stdout)
    assert given in w and not any(o.startswith(default) for o in w), w


def test_submit_records_options_and_site_env(tmp_path):
    site = tmp_path / "ws" / "site_env.sh"
    site.parent.mkdir(parents=True)
    site.write_text("export FOO=1\n")
    p = _dry(tmp_path, "-t", "1:00:00", "--", "echo", "hi")
    assert p.returncode == 0, p.stderr
    assert f"DM_SITE_ENV={site}" in p.stdout                       # the workspace's site_env.sh by default
    opts = _dry_opts(p.stdout)
    assert opts == _sbatch_words(p.stdout)[:len(opts)]             # every link replays exactly these options
    assert opts[-1] == "--dependency=singleton", opts              # Slurm runs one job of the chain at a time


def test_submit_makes_a_relative_site_file_absolute(tmp_path):
    p = _dry(tmp_path, "-t", "1:00:00", "--", "echo", cwd=tmp_path, DM_SITE_ENV="site.sh")
    assert p.returncode == 0 and f"DM_SITE_ENV={tmp_path}/site.sh" in p.stdout, p.stdout + p.stderr


def test_submit_job_name_is_a_safe_file_name(tmp_path):
    """The name becomes the log file's name (%x) and the chain's directory: no '/' (a directory that does not exist) and
    no space (it would split the recorded options)."""
    p = _dry(tmp_path, "-t", "1:00:00", "--", "bash", "scripts/campaign/eval_arm.sh", "RUN")
    assert p.returncode == 0, p.stdout + p.stderr
    assert any(re.fullmatch(r"--job-name=dm-bash-scripts_campaign_eval_arm\.sh-[0-9a-f]{8}", o)
               for o in _sbatch_words(p.stdout)), p.stdout
    p = _dry(tmp_path, "-t", "1:00:00", "--", "scripts/campaign/train_arm.sh", "a b")
    assert p.returncode == 0, p.stdout + p.stderr
    assert any(re.fullmatch(r"--job-name=dm-train_arm-a_b-[0-9a-f]{8}", o) for o in _sbatch_words(p.stdout)), p.stdout


def _name(tmp_path, *cmd, **extra):
    p = _dry(tmp_path, "-t", "4:00:00", "--", *cmd, **extra)
    assert p.returncode == 0, p.stdout + p.stderr
    return _dry_chain(p.stdout)[0]


def test_submit_chain_name_hashes_the_command_and_its_knobs(tmp_path):
    """Both seeds of an arm (an argument or SEEDS) and every knob that changes the work are different chains (singleton
    would otherwise serialise them, and they would adopt each other's jobs); the same command and knobs are the same
    chain; how a chain runs (DM_RESUBMIT, the save interval...) is not part of its name."""
    cmd = ("scripts/campaign/train_arm.sh", "dm_matched_hd128", "11785")
    base = _name(tmp_path, *cmd)
    ws = os.path.realpath(tmp_path / "ws")
    h8 = hashlib.sha1(("\n".join(cmd) + "\n" + f"DM_WORKSPACE={ws}\n").encode()).hexdigest()[:8]
    assert base == f"dm-train_arm-dm_matched_hd128-{h8}", base   # sha1 of the command, its arguments, K=V lines (sorted)
    assert _name(tmp_path, *cmd) == base
    assert _name(tmp_path, *cmd, DM_RESUBMIT=1, DM_SAVE_INTERVAL=5, DM_STOP_MARGIN=300, SEEDS="") == base
    names = {_name(tmp_path, *cmd[:2], "11786")}
    for knob in ("SEEDS", "TAG", "SMOKE", "STOP_AT_STEP", "NUM_STEPS", "EXTRA_ARGS", "EVAL", "DM_LR", "DM_TRAIN_CONFIG",
                 "DM_DATA_ROOT", "S2_INIT_RUN", "S2_ANYTHING"):
        names.add(_name(tmp_path, *cmd, **{knob: "1"}))
    names.add(_name(tmp_path, *cmd, SEEDS="2"))
    names.add(_name(tmp_path, *cmd, DM_WORKSPACE=tmp_path / "other"))
    assert base not in names and len(names) == 15, names
    two = _name(tmp_path, *cmd, SEEDS="1", TAG="_x")
    assert _name(tmp_path, *cmd, TAG="_x", SEEDS="1") == two   # sorted: the order of the environment does not matter


def test_submit_chain_name_is_the_same_however_the_work_is_spelled(tmp_path):
    """The command as the job runs it (from the repository root) and the workspace's physical path: ./x, the absolute
    path, a link to the workspace, a sourced site file (DM_WORKSPACE, DM_DATA_ROOT = its data) name one chain; another
    workspace or data root, another chain."""
    rel = ("scripts/campaign/train_arm.sh", "arm", "1")
    base = _name(tmp_path, *rel)
    assert _name(tmp_path, "./" + rel[0], *rel[1:]) == base
    assert _name(tmp_path, str(REPO / rel[0]), *rel[1:]) == base
    assert _name(tmp_path, "scripts/../scripts/campaign/train_arm.sh", *rel[1:]) == base
    (tmp_path / "ws").mkdir()
    os.symlink(tmp_path / "ws", tmp_path / "wslink")
    assert _name(tmp_path, *rel, DM_WORKSPACE=tmp_path / "wslink") == base
    assert _name(tmp_path, *rel, DM_DATA_ROOT=tmp_path / "ws" / "data") == base
    assert _name(tmp_path, *rel, DM_DATA_ROOT=tmp_path / "wslink" / "data") == base
    assert _name(tmp_path, *rel, DM_DATA_ROOT=tmp_path / "other") != base
    assert _name(tmp_path, *rel, DM_WORKSPACE=tmp_path / "ws2") != base
    env = _env(tmp_path, DM_SUBMIT_DRY=1)
    env.pop("DM_WORKSPACE")                      # the repository's workspace: its physical path is hashed
    p = subprocess.run(["bash", str(SUBMIT), "-t", "4:00:00", "--", *rel], capture_output=True, text=True, env=env)
    q = subprocess.run(["bash", str(SUBMIT), "-t", "4:00:00", "--", *rel], capture_output=True, text=True,
                       env={**env, "DM_WORKSPACE": os.path.realpath(REPO / "workspace")})
    assert p.returncode == q.returncode == 0 and _dry_chain(p.stdout)[0] == _dry_chain(q.stdout)[0], (p.stdout, q.stdout)


def test_submit_dry_prints_the_chain_and_writes_nothing(tmp_path):
    p = _dry(tmp_path, "-t", "4:00:00", "--", "scripts/campaign/train_arm.sh", "x")
    assert p.returncode == 0, p.stderr
    name, d = _dry_chain(p.stdout)
    assert d == f"{tmp_path}/ws/logs/slurm/chains/{name}", (name, d)
    assert f"--job-name={name}" in _sbatch_words(p.stdout)
    assert not (tmp_path / "ws").exists()


@pytest.mark.parametrize("given", [["-d", "afterok:12"], ["-dafterok:12"], ["--dependency", "afterok:12"],
                                   ["--dependency=afterok:12"], ["--dep=afterok:12"]])
def test_submit_dependency_is_for_the_first_job_only(tmp_path, given):
    """A dependency on another job holds the first link only (a later link's sbatch would name a job long gone), merged
    with the chain's singleton."""
    p = _dry(tmp_path, *given, "-t", "4:00:00", "--", "scripts/campaign/train_arm.sh", "x")
    assert p.returncode == 0, p.stderr
    w, opts = _sbatch_words(p.stdout), _dry_opts(p.stdout)
    assert [o for o in w if o.startswith("--dep")] == ["--dependency=afterok:12,singleton"], w
    assert [o for o in opts if o.startswith("--dep")] == ["--dependency=singleton"], opts
    assert not any("afterok" in o for o in opts), opts


@pytest.mark.parametrize("given, first", [(["-b", "now+1hour"], ["--begin=now+1hour"]),
                                          (["-bnow+1hour"], ["--begin=now+1hour"]),
                                          (["--begin=now+1hour"], ["--begin=now+1hour"]), (["-H"], ["--hold"]),
                                          (["--hold"], ["--hold"]), (["-H", "-d", "singleton"], ["--hold"])])
def test_submit_begin_and_hold_are_for_the_first_job_only(tmp_path, given, first):
    p = _dry(tmp_path, *given, "-t", "4:00:00", "--", "scripts/campaign/train_arm.sh", "x")
    assert p.returncode == 0, p.stderr
    w, opts = _sbatch_words(p.stdout), _dry_opts(p.stdout)
    assert w[len(opts):-3] == first and w[:len(opts)] == opts, (w, opts)
    assert not any(o.startswith(("--begin", "--hold", "-b", "-H")) for o in opts), opts
    assert [o for o in w if o.startswith("--dep")] == ["--dependency=singleton"], w


@pytest.mark.parametrize("args, extra, why", [
    (["-t", "4:00 :00", "--", "echo"], {}, "contains a space"), ([], {}, "usage"),
    (["--", "echo"], {"DM_WORKSPACE": "{tmp}/a b"}, "contains a space"),
    (["--array=0-4", "--", "echo"], {}, "--array"), (["-a", "0-4", "--", "echo"], {}, "--array"),
    (["-a0-4", "--", "echo"], {}, "--array"), (["--wait", "--", "echo"], {}, "--wait"), (["-W", "--", "echo"], {}, "--wait"),
    (["-t", "1:00:00", "--signal=B:USR1@600", "--", "echo"], {}, "--signal"),
    (["-t", "1:00:00", "--signal", "USR1@60", "--", "echo"], {}, "--signal"),
    (["-t", "1:00:00", "--export=NONE", "--", "echo"], {}, "--export"),
    (["-t", "1:00:00", "--export", "ALL,DM_RESUBMIT_COUNT=0", "--", "echo"], {}, "--export"),
    (["-t", "1:00:00", "--", "echo"], {"SBATCH_SIGNAL": "B:USR1@600"}, "SBATCH_SIGNAL"),
    (["-p", "batch", "--", "echo"], {}, "time limit"), (["--time-min=2:00:00", "--", "echo"], {}, "time limit"),
    (["-t", "1:00:00", "-J", "a,b", "--", "echo"], {}, "job name"), (["-t", "1:00:00", "-J..", "--", "echo"], {}, "job name"),
    (["-t", "1:00:00", "-d", "afterok:1?afterok:2", "--", "echo"], {}, "OR dependency"),
    (["-t", "1:00:00", "-d", "afterok:1", "-d", "afterok:2", "--", "echo"], {}, "one -d"),
    (["-t", "1:00:00", "-d", "--", "echo"], {}, "needs a value"),
    (["-t", "1:00:00", "--", "echo"], {"DM_MAX_FAILS": "two"}, "DM_MAX_FAILS"),
    (["-t", "1:00:00", "--", "echo"], {"DM_PRED_WAIT": "10m"}, "DM_PRED_WAIT"),
    (["-t", "1:00:00", "--", "echo"], {"DM_RESUBMIT": "yes"}, "DM_RESUBMIT"),
    (["--ti=4:00:00", "--", "echo"], {}, "ambiguous"),              # sbatch's getopt: --time or --time-min
    (["-t", "1:00:00", "batch", "--", "echo"], {}, "not an sbatch option"),   # sbatch would take it as its script
    (["-t", "1:00:00", ":", "--", "echo"], {}, "not an sbatch option"),
    (["-t", "1:00:00", "--wrap=hostname", "--", "echo"], {}, "--wrap"),
    (["-t", "1:00:00", "--test-only", "--", "echo"], {}, "--test-only"),
    (["-t", "1:00:00", "-HW", "--", "echo"], {}, "--wait"),         # grouped flags
    (["-t", "1:00:00", "--sig=USR1@60", "--", "echo"], {}, "--signal"),
    (["-t", "1:00:00", "--", "echo"], {"DRY": "1"}, "DRY=1"),       # the driver's preview would complete the chain
    (["-t", "1:00:00", "--", "campaign/train_arm.sh"], {}, "not an executable file"),   # relative to the repository
    (["-t", "1:00:00", "--", "scripts/campaign/env.sh"], {}, "not an executable file"),  # no x bit
    (["-t", "1:00:00", "--", "{tmp}/missing.sh"], {}, "not an executable file")])
def test_submit_refuses(tmp_path, args, extra, why):
    p = _dry(tmp_path, *[a.format(tmp=tmp_path) for a in args], **{k: v.format(tmp=tmp_path) for k, v in extra.items()})
    assert p.returncode == 2, (p.returncode, p.stdout, p.stderr)
    if why != "usage":
        assert why in p.stderr, p.stderr


@pytest.mark.parametrize("args, extra", [(["--", "echo"], {"DM_RESUBMIT": "0"}), (["-t", "1:00:00", "--export=ALL", "--", "echo"], {}),
                                         (["--time=1:00:00", "--export-file=f", "--", "echo"], {}),
                                         (["--", "echo"], {"SBATCH_TIMELIMIT": "4:00:00"}),
                                         (["-t", "1:00:00", "--", "echo"], {"DRY": "0"}),
                                         (["-t", "1:00:00", "--", "bash", "scripts/campaign/env.sh"], {})])
def test_submit_accepts(tmp_path, args, extra):
    """A single job (DM_RESUBMIT=0) may take the partition's default time limit, and SBATCH_TIMELIMIT is a time limit;
    --export=ALL is what a chain does; a script without the x bit runs through bash."""
    p = _dry(tmp_path, *args, **extra)
    assert p.returncode == 0, p.stderr


@pytest.mark.parametrize("given, gone, first", [
    (["--part=batch_short"], ["--partition=b"], []), (["--acco=x"], ["--account=a"], []),
    (["--gpus-per-n=4"], ["--gpus-per-node=8", "--mem=0", "--exclusive"], []),
    (["--output", "-b.log"], ["--output="], []),             # a value that starts with '-' is the option's value
    (["-o", "-b.log"], ["--output="], []), (["--comment", "-abc"], [], []),
    (["-Hp", "batch_short"], ["--partition=b"], ["--hold"]),  # grouped: -H (first job only), then -p and its value
    (["--be=now+1hour"], [], ["--begin=now+1hour"]), (["--ho"], [], ["--hold"])])
def test_submit_reads_options_as_sbatch_does(tmp_path, given, gone, first):
    """Every spelling sbatch's getopt accepts (abbreviations, a value starting with '-', grouped flags) counts as the
    option given: no default added after it (sbatch would take the last one)."""
    p = _dry(tmp_path, *given, "-t", "4:00:00", "--", "scripts/campaign/train_arm.sh", "x", DM_SLURM_ACCOUNT="a",
             DM_SLURM_PARTITION="b")
    assert p.returncode == 0, p.stderr
    w, opts = _sbatch_words(p.stdout), _dry_opts(p.stdout)
    assert not [o for o in w for g in gone if o.startswith(g)], (gone, w)
    assert w[len(opts):-3] == first, (w, opts)
    assert not any(o.startswith(("--begin", "--hold", "--be", "--ho")) for o in opts) and "-Hp" not in opts, opts
    assert "-b.log" not in " ".join(w[len(opts):-3]), w


def test_submit_sbatch_variables_count_as_given(tmp_path):
    p = _dry(tmp_path, "-t", "4:00:00", "--", "scripts/campaign/train_arm.sh", "x", DM_SLURM_PARTITION="b",
             DM_SLURM_ACCOUNT="a", SBATCH_PARTITION="batch_short", SBATCH_ACCOUNT="mine", SBATCH_GPUS_PER_NODE="4",
             SBATCH_OUTPUT="/tmp/o.log")
    assert p.returncode == 0, p.stderr
    w = _sbatch_words(p.stdout)
    assert not any(o.startswith(("--partition=", "--account=", "--gpus-per-node=", "--exclusive", "--mem=", "--output="))
                   for o in w), w


def test_submit_from_inside_a_link_starts_a_new_chain_afresh(tmp_path):
    """A shell inside a link (or a chain's own command) has that link's index, deadline and the variables its job.sh
    derived (DM_DERIVED): the new chain starts at link 0 without them; variables the user set themselves stay."""
    fake = FakeSlurm(tmp_path)
    p = _submit(tmp_path, "-t", "4:00:00", "--", "scripts/campaign/train_arm.sh", "x", fake=fake, cwd=tmp_path,
                DM_RESUBMIT_COUNT=57, DM_MASTER_PORT=23456, DM_VALCE_PORT=23457, DM_NPROC=4, DM_STOP_BEFORE=5,
                TRITON_CACHE_DIR="/tmp/dm-jit-1/old/triton", DM_DERIVED="DM_VALCE_PORT DM_NPROC TRITON_CACHE_DIR")
    assert p.returncode == 0, p.stderr
    (job,) = fake.jobs()
    env = job["env"]
    assert env["DM_RESUBMIT_COUNT"] == "0" and env["DM_MASTER_PORT"] == "23456", env   # the user's own port stays
    assert not {"DM_VALCE_PORT", "DM_NPROC", "TRITON_CACHE_DIR", "DM_DERIVED", "DM_STOP_BEFORE"} & set(env), env


def test_submit_warns_about_the_same_work_under_another_name(tmp_path):
    """An older submit.sh's chain (dm-<cmd>-<arg1>, no hash) or the same command with other knobs is named: two chains
    of one run would train it at once."""
    fake = FakeSlurm(tmp_path)
    args = ["-t", "4:00:00", "--", "scripts/campaign/train_arm.sh", "x"]
    name = _dry_chain(_dry(tmp_path, *args).stdout)[0]
    legacy = fake.add_job(name="dm-train_arm-x", state="RUNNING")
    other = fake.add_job(name="dm-train_arm-x-0123abcd")
    unrelated = fake.add_job(name="dm-train_arm-xy-0123abcd")
    p = _submit(tmp_path, *args, fake=fake, cwd=tmp_path)
    assert p.returncode == 0, p.stderr
    assert f"{legacy} (dm-train_arm-x)" in p.stderr and f"{other} (dm-train_arm-x-0123abcd)" in p.stderr, p.stderr
    assert str(unrelated) not in p.stderr and f"the chain {name} already has jobs" not in p.stderr, p.stderr


def test_submit_single_job_leaves_the_chain_state_alone(tmp_path):
    """DM_RESUBMIT=0 is no chain: the state of a chain of the same name (its failure records, runfails/) is kept and no
    recipe is written."""
    fake = FakeSlurm(tmp_path)
    args = ["-t", "1:00:00", "--", "scripts/campaign/train_arm.sh", "x"]
    d = tmp_path / "ws" / "logs" / "slurm" / "chains" / _dry_chain(_dry(tmp_path, *args).stdout)[0]
    (d / "runfails").mkdir(parents=True)
    (d / "links").write_text("1 41 start k=3 fails=2\n")
    p = _submit(tmp_path, *args, fake=fake, cwd=tmp_path, DM_RESUBMIT="0")
    assert p.returncode == 0 and "a single job (DM_RESUBMIT=0)" in p.stdout, p.stdout + p.stderr
    assert (d / "links").read_text() == "1 41 start k=3 fails=2\n" and (d / "runfails").exists()
    assert not (d / "submit.sh").exists() and len(fake.jobs()) == 1


def test_submit_refuses_an_sbatch_reply_without_a_job_id(tmp_path):
    fake = FakeSlurm(tmp_path, sbatch=["junk"])
    p = _submit(tmp_path, "-t", "4:00:00", "--", "echo", fake=fake, cwd=tmp_path)
    assert p.returncode == 1 and "not a job id" in p.stderr, p.stdout + p.stderr
    assert "submit" not in "".join(f.read_text() for f in (tmp_path / "ws").rglob("links")), "no submit record"


def test_submit_records_the_repository(tmp_path):
    p = _dry(tmp_path, "-t", "1:00:00", "--", "echo", "hi")
    assert p.returncode == 0 and f"DM_SLURM_REPO={REPO}" in p.stdout, p.stdout


def _submit(tmp_path, *args, fake, cwd, **extra):
    return subprocess.run(["bash", str(SUBMIT), *args], capture_output=True, text=True, timeout=60, cwd=cwd,
                          env=_env(tmp_path, PATH=f"{fake.bin}:{os.environ['PATH']}", **extra))


def test_submit_starts_the_chain_state_and_writes_its_recipe(tmp_path):
    """A submission: the chain's directory with fresh failure records (`links` with one submit record, no runfails/;
    the progress lines and the compile caches stay), a submit.sh that submits the same chain again (same name, same
    options, without the first job's own options, wherever it is run from and whatever the shell's knobs are), no stale
    deadline / end time / chain of the submitting shell in the job, and the ways to stop it and find its state."""
    fake = FakeSlurm(tmp_path)
    sub = tmp_path / "submitted from"
    sub.mkdir()
    args = ["-A", "acct", "-t", "4:00:00", "-d", "afterok:7", "-H", "--", "scripts/campaign/train_arm.sh", "arm one", "*"]
    knobs = dict(SEEDS="11785", TAG="_x y", S2_INIT_RUN="r$1", DM_SAVE_INTERVAL=400, DM_SLURM_PARTITION="batch")
    name = _dry_chain(_dry(tmp_path, *args, cwd=sub, **knobs).stdout)[0]
    d = tmp_path / "ws" / "logs" / "slurm" / "chains" / name
    (d / "runfails").mkdir(parents=True)
    (d / "runfails" / "RUN.train").write_text("5 2\n")
    (d / "links").write_text("1 41 start k=3 fails=2\n")
    (d / "progress").write_text("1 41 ckpt RUN 100\n")
    (d / "jit.tar").write_text("tar")
    p = _submit(tmp_path, *args, fake=fake, cwd=sub, DM_STOP_BEFORE=1, DM_JOB_END=2, DM_CHAIN="old",
                DM_PROGRESS_FILE="/x", **knobs)
    assert p.returncode == 0, p.stdout + p.stderr
    (job,) = fake.jobs()
    assert job["name"] == name and job["cwd"] == str(sub), job
    assert "--dependency=afterok:7,singleton" in job["argv"] and "--hold" in job["argv"], job["argv"]
    env = job["env"]
    assert env["DM_CHAIN_DIR"] == str(d) and env["DM_RESUBMIT_COUNT"] == "0", env
    assert not {"DM_STOP_BEFORE", "DM_JOB_END", "DM_CHAIN", "DM_PROGRESS_FILE"} & set(env), env
    recs = [l.split() for l in (d / "links").read_text().splitlines()]
    assert len(recs) == 1 and recs[0][1:4] == [str(job["id"]), "submit", "fails=0"], recs
    assert not (d / "runfails").exists() and (d / "progress").exists() and (d / "jit.tar").exists()
    for want in (f"submitted job {job['id']}", f"scancel -u {os.environ['USER']} -n {name}", str(d), "logs/slurm/"):
        assert want in p.stdout, (want, p.stdout)
    recipe = d / "submit.sh"
    assert stat.S_IMODE(recipe.stat().st_mode) == 0o755, oct(recipe.stat().st_mode)
    text = recipe.read_text()
    (line,) = [l for l in text.splitlines() if l.startswith("exec env ")]
    assert "afterok" not in line and "-H" not in shlex.split(line), text   # the first job's own options are not repeated
    assert "# Not repeated (they applied to the first job only): -d afterok:7 -H" in text, text
    # the recipe, run from elsewhere by a shell with other knobs: the same chain, the same options; no -d / -H
    q = subprocess.run([str(recipe)], capture_output=True, text=True, timeout=60, cwd=tmp_path,
                       env=_env(tmp_path, DM_SUBMIT_DRY=1, SEEDS="1", TAG="", S2_OTHER="1", DM_SLURM_PARTITION="x"))
    assert q.returncode == 0, q.stdout + q.stderr
    assert _dry_chain(q.stdout)[0] == name, (name, q.stdout)
    w = _sbatch_words(q.stdout)
    assert w == _dry_opts(q.stdout) + [str(JOB), "scripts/campaign/train_arm.sh", "arm one", "*"], w
    assert "--partition=batch" in w and "--hold" not in w and not any("afterok" in o for o in w), w
    assert set(_dry_opts(q.stdout)) == set(job["env"]["DM_SBATCH_OPTS"].split()), (q.stdout, job["env"]["DM_SBATCH_OPTS"])


def test_submit_warns_when_the_chain_already_has_jobs(tmp_path):
    fake = FakeSlurm(tmp_path)
    args = ["-t", "4:00:00", "--", "scripts/campaign/train_arm.sh", "x"]
    name = _dry_chain(_dry(tmp_path, *args).stdout)[0]
    old = fake.add_job(name=name, state="RUNNING")
    p = _submit(tmp_path, *args, fake=fake, cwd=tmp_path)
    assert p.returncode == 0 and f"WARN: the chain {name} already has jobs in the queue ({old}" in p.stderr, p.stderr
    assert len(fake.jobs(name=name)) == 2


def test_submit_reports_a_failed_sbatch(tmp_path):
    fake = FakeSlurm(tmp_path, sbatch=["fail"])
    p = _submit(tmp_path, "-t", "4:00:00", "--", "echo", fake=fake, cwd=tmp_path)
    assert p.returncode == 1 and "FATAL: sbatch failed" in p.stderr and fake.jobs() == [], p.stderr
    links = list((tmp_path / "ws" / "logs" / "slurm" / "chains").glob("*/links"))
    assert links == [], links


# ------------------------------------------------------------------------------------------------------------ job.sh
def test_job_sets_the_deadline_and_runs_from_the_repo(tmp_path):
    p, calls, _ = _job(tmp_path, 0, DM_STOP_MARGIN=600)
    assert p.returncode == 0, p.stdout + p.stderr
    end = _epoch(END)
    assert f"deadline={end - 600}" in p.stdout and f"end={end}" in p.stdout, p.stdout
    assert "site=yes" in p.stdout and f"cwd={REPO}" in p.stdout and "args=a1 a2" in p.stdout, p.stdout


def test_job_default_margin_is_600_s_and_exports_the_end(tmp_path):
    end = int(time.time()) + 7200
    r = _link(tmp_path, (0,), SLURM_JOB_END_TIME=end)
    assert r.rc == 0 and f"deadline={end - 600} end={end}" in r.out, r.out


@pytest.mark.parametrize("extra, want", [({}, "launch=env"), ({"DM_LAUNCH": "torchrun"}, "launch=torchrun")])
def test_job_launches_the_ranks_itself_by_default(tmp_path, extra, want):
    """Slurm's SIGTERM reaches torchrun too, which kills its ranks after 30 s: chains default to DM_LAUNCH=env."""
    p, _, _ = _job(tmp_path, 0, **extra)
    assert p.returncode == 0 and want in p.stdout, p.stdout


@pytest.mark.parametrize("eph, rfile, want, warns", [
    ("9000 65000", None, "7898/7899", ()),         # this cluster's compute nodes: the band 7700-8999 just below the range
    ("32768 60999", None, "31666/31667", ()),      # Linux's default: 31468-32767
    ("1024 50000", None, "50199/50200", ()),       # no room below: the band just above, 50001-51300
    ("500 1000", None, "1222/1223", ()),           # above, but never below 1024: 1024-2323
    ("1024 65535", None, "20198/20199", ()),       # no room on either side: 20000-31999
    ("", "9000\t65000\n", "7898/7899", ()),      # the kernel's file (tab-separated)
    ("", "32768 60999\n", "31666/31667", ()),
    ("junk", "", "7898/7899", ("is not '<lo> <hi>'", "cannot read")),        # unknown: 9000 65000
    ("70000 80000", "", "7898/7899", ("is not '<lo> <hi>'", "cannot read")),  # not ports
    ("0 0", "32768 60999\n", "31666/31667", ("is not '<lo> <hi>'",)),        # a bad override: the kernel's file
])
def test_job_gives_each_job_its_own_rendezvous_ports(tmp_path, eph, rfile, want, warns):
    """Two partial-node jobs on one node must not share train_arm.sh's fixed env-launch / val-CE ports, and neither may
    lie in the host's ephemeral port range, where any outgoing connection can hold it (EADDRINUSE at the rendezvous on a
    node shared with other jobs): job 99 gets the 99th pair of a band of 1300 ports outside the range (DM_EPHEMERAL_PORTS,
    else the kernel's file, else 9000 65000)."""
    rf = tmp_path / "port_range"
    if rfile is None:
        rf = tmp_path / "no_such_file"
    else:
        rf.write_text(rfile)
    p, _, _ = _job(tmp_path, 0, DM_EPHEMERAL_PORTS=eph, DM_PORT_RANGE_FILE=rf)   # SLURM_JOB_ID=99
    assert p.returncode == 0 and f"port={want}" in p.stdout, p.stdout
    for w in ("is not '<lo> <hi>'", "cannot read"):
        assert (w in p.stdout) == (w in warns), (w, p.stdout)


@pytest.mark.parametrize("given, want, warn", [
    ({"DM_MASTER_PORT": 8000}, "8000/8001", False),
    ({"DM_MASTER_PORT": 30000}, "30000/30001", True),                     # kept, with a warning: in the range
    ({"DM_MASTER_PORT": 65535, "DM_VALCE_PORT": 4000}, "65535/4000", False),   # above the range
    ({"DM_MASTER_PORT": "08000"}, "08000/8001", False),                    # no octal
])
def test_job_keeps_the_users_own_ports(tmp_path, given, want, warn):
    p, _, _ = _job(tmp_path, 0, DM_EPHEMERAL_PORTS="9000 65000", **given)
    assert p.returncode == 0 and f"port={want}" in p.stdout, p.stdout
    assert ("lies in this host's ephemeral port range 9000-65000" in p.stdout) == warn, p.stdout


@pytest.mark.parametrize("given", [{"DM_MASTER_PORT": 65535}, {"DM_MASTER_PORT": 0}, {"DM_MASTER_PORT": 70000}])
def test_job_refuses_a_port_that_cannot_be_one(tmp_path, given):
    """65535 leaves no port for the val CE (DM_VALCE_PORT unset): a configuration error, the chain stops (exit 2)."""
    p, argvs, _ = _job(tmp_path, 0, DM_EPHEMERAL_PORTS="9000 65000", **given)
    assert p.returncode == 2 and "is not a usable port" in p.stderr and "CMD n=" not in p.stdout, (p.stdout, p.stderr)


def _free_quad():
    """Four consecutive ports free on this host (each binds), the first one even."""
    for q in range(41000, 48000, 4):
        socks = []
        try:
            for k in range(4):
                s = socket.socket(); socks.append(s); s.bind(("127.0.0.1", q + k))
            return q
        except OSError:
            continue
        finally:
            for s in socks:
                s.close()
    pytest.skip("no four consecutive free ports in 41000-47999")


@pytest.mark.parametrize("busy, how", [(0, "listener"), (1, "listener"), (0, "connection"), (1, "connection")])
def test_job_skips_a_port_pair_in_use(tmp_path, busy, how):
    """A socket of this host on the derived port (0) or on the val CE's next one (1), a listener or the local end of a
    connection (which no connect probe sees: /proc/net/tcp does): the pair is moved on by 2."""
    q = _free_quad()
    lo = q - 198 + 1300                                                 # job 99: the band's start + 198 is q
    socks = []
    try:
        if how == "listener":
            s = socket.socket(); socks.append(s)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", q + busy)); s.listen()
        else:
            srv = socket.socket(); socks.append(srv)
            srv.bind(("127.0.0.1", 0)); srv.listen()
            c = socket.socket(); socks.append(c)
            c.bind(("127.0.0.1", q + busy)); c.connect(srv.getsockname())
            a, _ = srv.accept(); socks.append(a)
        p, _, _ = _job(tmp_path, 0, DM_EPHEMERAL_PORTS=f"{lo} 65535")
    finally:
        for s in socks:
            s.close()
    assert p.returncode == 0 and f"port={q + 2}/{q + 3}" in p.stdout, p.stdout


def test_job_derives_the_ports_again_for_an_in_place_retry(tmp_path):
    """The first attempt fails and its port stays in use (here: a listener the attempt left behind): the in-place retry
    gets the next pair, not the same one again."""
    q = _free_quad()
    lo = q - 198 + 1300
    pidf = tmp_path / "holder.pid"
    body = (f'[ "$n" = 1 ] && {{ python3 -c "import os, socket, time; s = socket.socket(); '
            f's.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind((\'127.0.0.1\', $DM_MASTER_PORT)); s.listen(); '
            f'open(\'{pidf}\', \'w\').write(str(os.getpid())); time.sleep(60)" >/dev/null 2>&1 & '
            f'for i in $(seq 50); do [ -s {pidf} ] && break; sleep 0.1; done; }}')
    try:
        r = _link(tmp_path, (1, 0), (0, 1), body=body, DM_EPHEMERAL_PORTS=f"{lo} 65535")
    finally:
        if pidf.exists() and pidf.read_text().strip():
            try:
                os.kill(int(pidf.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
    assert r.rc == 0, r.out
    assert f"CMD n=1 " in r.out and f"port={q}/{q + 1}" in r.out, r.out
    assert re.search(rf"CMD n=2 .* port={q + 2}/{q + 3} ", r.out), r.out
    assert f"rendezvous ports for attempt 2: {q + 2}/{q + 3}" in r.out, r.out


def test_job_prefers_slurm_job_end_time(tmp_path):
    end = int(time.time()) + 7200
    p, _, _ = _job(tmp_path, 0, SLURM_JOB_END_TIME=end, DM_STOP_MARGIN=600)   # squeue would say 2030
    assert p.returncode == 0 and f"deadline={end - 600}" in p.stdout, p.stdout


@pytest.mark.parametrize("end, squeue_rc", [("", 1), ("", 0), ("Unknown", 0), ("NONE", 0), ("Tomorr 04:00", 0),
                                            ("2030-01-01T12:00:00", 1)])
def test_job_sets_no_deadline_when_the_end_is_unknown(tmp_path, end, squeue_rc):
    """Never a wrong deadline: `date -d ''` is last midnight, which would stop every job of a chain after one step."""
    p, _, _ = _job(tmp_path, 0, end=end, squeue_rc=squeue_rc)
    assert p.returncode == 0 and "deadline=none end=none" in p.stdout and "WARN: no end time" in p.stdout, p.stdout


@pytest.mark.parametrize("left", [0, 900, 1100])   # seconds to the job's end, with the default 600 s margin
def test_job_with_too_short_a_time_limit_sets_no_deadline_and_never_resubmits(tmp_path, left):
    p, calls, _ = _job(tmp_path, 75, SLURM_JOB_END_TIME=int(time.time()) + left)
    assert p.returncode == 75 and "deadline=none" in p.stdout and calls == [], (p.stdout, calls)
    assert "no trainer deadline and no resubmission" in p.stdout, p.stdout


def test_job_queues_its_successor_at_the_start(tmp_path):
    p, calls, _ = _job(tmp_path, 75, DM_RESUBMIT_COUNT=3)
    assert p.returncode == 75, p.stdout + p.stderr
    assert calls == [["--parsable", *OPTS.split(), f"--job-name={CHAIN}", str(JOB), str(tmp_path / "cmd.sh"), "a1", "a2"]], calls
    assert "queued successor 4242 (chain step 4; it starts when this job ends)" in p.stdout, p.stdout
    assert p.stdout.index("queued successor") < p.stdout.index("CMD n=1"), p.stdout   # before the command runs


def test_job_resubmits_exact_arguments_from_the_submit_directory(tmp_path):
    sub = tmp_path / "submitted from"
    sub.mkdir()
    args = ["a b", "*", "it's", "$HOME"]
    p, calls, dirs = _job(tmp_path, 75, args=args, SLURM_SUBMIT_DIR=sub)
    assert p.returncode == 75, p.stdout + p.stderr
    assert calls == [["--parsable", *OPTS.split(), f"--job-name={CHAIN}", str(JOB), str(tmp_path / "cmd.sh"), *args]], calls
    assert dirs == [str(sub)], dirs   # relative paths of the options and the site file mean the same in every job


def test_submit_options_round_trip_through_a_resubmission(tmp_path):
    p = _dry(tmp_path, "-pbatch_short", "-t", "4:00:00", "--", "scripts/campaign/train_arm.sh", "dm matched",
             DM_SLURM_ACCOUNT="acct")
    assert p.returncode == 0, p.stderr
    opts = re.search(r"DM_SBATCH_OPTS=(.*)", p.stdout).group(1)
    submitted = _sbatch_words(p.stdout)
    p, calls, _ = _job(tmp_path, 75, args=["dm matched"], DM_SBATCH_OPTS=opts)
    assert p.returncode == 75, p.stdout + p.stderr
    assert calls == [["--parsable", *submitted[:-3], f"--job-name={CHAIN}", str(JOB), str(tmp_path / "cmd.sh"),
                      "dm matched"]], (calls, submitted)


def test_job_retries_sbatch(tmp_path):
    p, calls, _ = _job(tmp_path, 75, sbatch_fails=2, DM_RESUBMIT_BACKOFF="0 0 0")
    assert p.returncode == 75 and len(calls) == 3 and "queued successor 4242" in p.stdout, (p.stdout, calls)
    assert p.stdout.count("WARN: sbatch failed; retry") == 2, p.stdout


def test_job_reports_a_broken_chain(tmp_path):
    """No successor at the start (3 attempts) nor at the end (3 more): exit 76, and the way to resume."""
    p, calls, _ = _job(tmp_path, 75, sbatch_fails=99, DM_RESUBMIT_BACKOFF="0 0")
    assert p.returncode == 76 and len(calls) == 6, (p.returncode, calls, p.stdout)
    assert "no successor could be queued; the chain is broken: resume with" in p.stdout and "/submit.sh" in p.stdout


@pytest.mark.parametrize("extra, want", [({"SLURM_GPUS_ON_NODE": 4}, "nproc=4"), ({}, "nproc=unset"),
                                         ({"SLURM_GPUS_ON_NODE": 4, "DM_NPROC": 2}, "nproc=2"),
                                         ({"SLURM_GPUS_ON_NODE": 4, "DM_NPROC": 8}, "nproc=4")])   # never more than the GPUs
def test_job_rank_count_follows_the_allocation(tmp_path, extra, want):
    p, _, _ = _job(tmp_path, 0, **extra)
    assert p.returncode == 0 and want in p.stdout, p.stdout


@pytest.mark.parametrize("rc, extra", [(75, {"DM_RESUBMIT": 0}), (75, {"DM_RESUBMIT_COUNT": 5, "DM_MAX_RESUBMITS": 5}),
                                       (1, {"DM_RESUBMIT": 0}), (75, {"SLURM_JOB_NAME": "a,b"}),
                                       (75, {"DM_SBATCH_OPTS": "-p batch -t 4:00:00"})])
def test_job_does_not_resubmit_otherwise(tmp_path, rc, extra):
    """A single job; the last link; a job name that is not a chain's; options without singleton (a chain of an older
    submit.sh: its successors could run beside it, so none is queued)."""
    p, calls, _ = _job(tmp_path, rc, **extra)
    assert p.returncode == rc and calls == [] and p.stdout.count("CMD n=") == 1, (p.returncode, calls, p.stdout)


@pytest.mark.parametrize("var, value", [("DM_STOP_MARGIN", "20m"), ("DM_MAX_RESUBMITS", "none"),
                                        ("DM_RESUBMIT_COUNT", "-1"), ("DM_MAX_FAILS", "x"), ("DM_INPLACE_RETRIES", "1.5"),
                                        ("DM_RETRY_MIN_LEFT", "30m"), ("DM_RETRY_PAUSE", "-1"), ("DM_JIT_MAX_MB", "8G"),
                                        ("DM_RESUBMIT", "yes"), ("DM_JIT_CACHE", "2"), ("DM_RESUBMIT_BACKOFF", "1,2"),
                                        ("DM_STOP_BEFORE", "soon"), ("DM_PRED_WAIT", "5m")])
def test_job_refuses_malformed_numbers(tmp_path, var, value):
    p, calls, _ = _job(tmp_path, 75, **{var: value})
    assert p.returncode == 2 and var in p.stderr and "CMD" not in p.stdout and calls == [], (p.returncode, p.stderr)


def test_job_reads_numbers_with_a_leading_zero_as_decimal(tmp_path):
    r = _link(tmp_path, (1,), DM_MAX_FAILS="08", DM_RETRY_PAUSE="00", DM_INPLACE_RETRIES="09")
    assert r.rc == 4 and r.out.count("CMD n=") == 9, r.out   # 08 is 8, not an error: 9 failures in a row, then give up


def test_job_site_file(tmp_path):
    site = tmp_path / "site_nounset.sh"   # written for an interactive shell: an unset variable must not kill the job
    site.write_text('export LD_LIBRARY_PATH=/compat:$DM_UNSET_EVERYWHERE\nexport SITE_SOURCED=yes\n')
    p, _, _ = _job(tmp_path, 0, DM_SITE_ENV=site)
    assert p.returncode == 0 and "site=yes" in p.stdout, p.stdout + p.stderr
    site.write_text("return 3\n")         # a failing site file stops the job before the command
    p, calls, _ = _job(tmp_path, 0, DM_SITE_ENV=site)
    assert p.returncode == 2 and "FATAL: sourcing DM_SITE_ENV" in p.stderr and "CMD" not in p.stdout, p.stdout + p.stderr
    assert calls == []


def test_job_runs_from_the_repo_when_slurm_runs_a_copy(tmp_path):
    """sbatch runs a COPY of job.sh from its spool directory: the repository comes from DM_SLURM_REPO (submit.sh), the
    command runs from it, and a successor names the repository's job.sh, not the copy."""
    import shutil
    spool = tmp_path / "spool"
    spool.mkdir()
    copy = spool / "slurm_script"
    shutil.copy(JOB, copy)
    fake = _fake_slurm(tmp_path)
    env = _env(tmp_path, PATH=f"{fake.bin}:{os.environ['PATH']}", SLURM_JOB_ID="99", SLURM_JOB_NAME=CHAIN,
               DM_SBATCH_OPTS="-p batch --dependency=singleton", DM_SLURM_REPO=REPO, DM_JIT_ROOT=tmp_path / "jit")
    p = subprocess.run(["bash", str(copy), "bash", "-c", "echo CWD=$PWD; exit 75"], capture_output=True, text=True,
                       timeout=60, env=env, cwd=str(spool))
    assert p.returncode == 75 and f"CWD={REPO}" in p.stdout, p.stdout + p.stderr
    calls = fake.sbatch_argvs()
    assert calls[0][:5] == ["--parsable", "-p", "batch", "--dependency=singleton", f"--job-name={CHAIN}"], calls
    assert calls[0][5] == str(JOB), calls
    env.pop("DM_SLURM_REPO")
    p = subprocess.run(["bash", str(copy), "true"], capture_output=True, text=True, timeout=60, env=env, cwd=str(spool))
    assert p.returncode == 2 and "DM_SLURM_REPO" in p.stderr, (p.returncode, p.stderr)


def _job_sigterm(tmp_path, to="shell", sig=signal.SIGTERM, end_in=7200, fake_end=None, running=None, pre="",
                 before_signal=None, **extra):
    """job.sh running a command that saves (1 s) and exits 75 on SIGTERM (and writes a progress line, as the trainer
    does after a checkpoint). The signal goes to the batch shell only (to="shell": what Slurm does at the time limit and
    on scancel, so job.sh must pass it on) or to every process of the job (to="group": a scheduler that signals them
    all; the command then gets it twice). The job ends end_in s from now (SLURM_JOB_END_TIME); squeue says
    fake_end (default: the same), `running` overrides the fake's running job; `pre`: shell lines the command runs
    first; `before_signal(fake)` runs just before the signal (what slurmctld does to the queue first)."""
    end = int(time.time()) + end_in
    fake = FakeSlurm(tmp_path, running={"id": 99, "name": CHAIN, "end": fake_end or _iso(end), **(running or {})})
    ready = tmp_path / "ready"
    script = tmp_path / "cmd.sh"
    script.write_text("#!/bin/bash\n"
                      "trap 'if [ -z \"$S\" ]; then S=1; echo \"CMD got TERM: saving\"; sleep 1; "
                      "echo \"$(date +%s) $SLURM_JOB_ID ckpt RUN 7\" >> \"$DM_PROGRESS_FILE\"; echo \"CMD saved\"; exit 75; fi' TERM\n"
                      f"{pre}\ntouch {ready}\nsleep 60 & wait\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    env = _job_env(tmp_path, fake, **{"SLURM_JOB_END_TIME": end, **extra})
    p = subprocess.Popen(["bash", str(JOB), str(script)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         env=env, start_new_session=True)
    t0 = time.time()
    while not ready.exists():
        assert p.poll() is None and time.time() - t0 < 30, "the command never started"
        time.sleep(0.05)
    if before_signal:
        before_signal(fake)
    if to == "group":
        os.killpg(p.pid, sig)
    else:
        os.kill(p.pid, sig)
    out, err = p.communicate(timeout=60)
    return p.returncode, out, err, fake


@pytest.mark.parametrize("to", ["shell", "group"])
def test_job_outlives_an_early_sigterm_and_stops_the_chain(tmp_path, to):
    rc, out, err, fake = _job_sigterm(tmp_path, to)
    assert rc == 75 and "CMD saved" in out and "exit 75 after a SIGTERM" in out, (rc, out, err)
    assert out.count("CMD got TERM") == 1, out
    assert "before the time limit: the chain is stopped" in out, out           # a scancel ends the chain
    (succ,) = fake.jobs(name=CHAIN, state="CANCELLED")                         # the successor queued at the start
    assert fake.jobs(id=99)[0]["state"] == "RUNNING", fake.state()             # never this job itself
    assert [c["argv"] for c in fake.calls("scancel")] == [[str(succ["id"])]]   # by id


@pytest.mark.parametrize("to", ["shell", "group"])
def test_job_keeps_its_successor_after_the_time_limit_sigterm(tmp_path, to):
    rc, out, err, fake = _job_sigterm(tmp_path, to, end_in=30, DM_STOP_BEFORE=1)
    assert rc == 75 and "CMD saved" in out and "queued successor 4242" in out, (rc, out, err)
    assert fake.job(4242)["state"] == "PENDING" and fake.calls("scancel") == [], (fake.state(), out)
    assert "the successor 4242 continues the chain" in out, out


@pytest.mark.parametrize("sig", [signal.SIGUSR1, signal.SIGINT, signal.SIGHUP, signal.SIGUSR2])
def test_job_handles_other_signals_like_sigterm(tmp_path, sig):
    """--signal=B:USR1@N (refused by submit.sh) or a signal sent by hand must not kill the batch shell and with it the
    job: the command is told to save (SIGTERM), and the stop is judged like a SIGTERM's."""
    rc, out, err, fake = _job_sigterm(tmp_path, "shell", sig)
    name = signal.Signals(sig).name
    assert rc == 75 and "CMD saved" in out and f"exit 75 after a {name}" in out, (rc, out, err)
    assert f"stopped by {name}" in out and fake.job(4242)["state"] == "CANCELLED", out
