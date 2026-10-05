"""The experiment drivers (scripts/experiments/*.sh and their shared _run.sh), CPU only: every experiment runs exactly
the arms and seeds its script lists (DRY=1 prints the plan), an arm outside it is refused, SEEDS overrides the seeds,
SMOKE=1 plans smokes only, and the driver's control flow (train, then evaluate; skip the evaluation after a failed
training; stop on an early stop with exit 75; report what is missing) holds with stand-in launchers."""
import os
import re
import shutil
import subprocess

import pytest

from _helpers import REPO, dry_env

EXP = REPO / "scripts" / "experiments"
# experiment -> (stage, {arm: seeds}) as the paper runs them
PLANS = {
    "stage1_hd128": (1, {a: ("11785", "2") for a in ("clean_hd128", "dm_stale_hd128", "dm_matched_hd128",
                                                     "dm_consistent_hd128", "te_fp8")}),
    "stage1_hd256_scale": (1, {f"{p}{a}": ("11785", "2") for p in ("xs_", "", "l_")
                               for a in ("clean_hd256", "dm_stale_hd256", "dm_matched_hd256")}),
    "stage1_architectures": (1, {**{f"{f}_{a}_hd128": ("11785", "2") for f in ("mla", "kda", "mamba2")
                                    for a in ("clean", "dm_stale", "dm_matched")},
                                 **{f"gqa_{a}_hd128": ("11785", "1") for a in ("clean", "dm_stale", "dm_matched")}}),
    "stage1_nope": (1, {f"nope_{a}_hd128": ("11785", "2") for a in ("clean", "dm_stale", "dm_matched")}),
    "stage1_muon": (1, {f"muon_{a}_hd128": ("11785",) for a in ("clean", "dm_stale", "dm_matched")}),
    "stage2_context": (2, {a: ("11785", "2") for a in ("clean_hd128", "dm_matched_hd128", "dm_stale_hd128", "te_fp8")}),
}
PLAN_RE = re.compile(r"^  (\S+) seed (\S+) -> (\S+): bash (\S+) \S+ \S+; bash (\S+) (\S+)$")


def _run(script, args, tmp_path, **extra):
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True,
                          env=dry_env(tmp_path, **extra), timeout=300)


def _plan(out):
    return [m.groups() for m in map(PLAN_RE.match, out.splitlines()) if m]


def test_every_experiment_script_is_tested():
    assert {p.stem for p in EXP.glob("*.sh") if not p.name.startswith("_")} == set(PLANS)


@pytest.mark.parametrize("exp", sorted(PLANS))
def test_dry_plan(exp, tmp_path):
    """DRY=1: one line per (arm, seed), in the script's arm order, the right launcher and evaluator, run names from
    arms.sh (with _s2_64k for stage 2)."""
    stage, arms = PLANS[exp]
    p = _run(EXP / f"{exp}.sh", [], tmp_path, DRY=1)
    assert p.returncode == 0, p.stderr[-2000:]
    plan = _plan(p.stdout)
    assert [(a, s) for a, s, *_ in plan] == [(a, s) for a in arms for s in arms[a]], plan
    for arm, seed, rn, train, ev, ev_rn in plan:
        assert os.path.basename(train) == ("stage2_arm.sh" if stage == 2 else "train_arm.sh"), train
        assert os.path.basename(ev) == ("eval_arm_64k.sh" if stage == 2 else "eval_arm.sh"), ev
        assert ev_rn == rn and rn.endswith("_s2_64k") is (stage == 2), (rn, ev_rn)
        name = subprocess.run(["bash", "-c", 'source scripts/campaign/env.sh; source scripts/campaign/arms.sh; '
                               'TAG="$3" dm_run_name "$1" "$2"', "x", arm, seed, "_s2_64k" if stage == 2 else ""],
                              cwd=str(REPO), capture_output=True, text=True, env=dry_env(tmp_path)).stdout.strip()
        assert rn == name, (arm, seed, rn, name)


def test_arm_subset_seed_override_and_refusal(tmp_path):
    p = _run(EXP / "stage1_architectures.sh", ["gqa_clean_hd128", "kda_dm_matched_hd128"], tmp_path, DRY=1)
    assert p.returncode == 0, p.stderr[-2000:]
    assert [(a, s) for a, s, *_ in _plan(p.stdout)] == [("gqa_clean_hd128", "11785"), ("gqa_clean_hd128", "1"),
                                                         ("kda_dm_matched_hd128", "11785"), ("kda_dm_matched_hd128", "2")]
    p = _run(EXP / "stage1_architectures.sh", ["gqa_clean_hd128"], tmp_path, DRY=1, SEEDS="7 8 9")
    assert [s for _, s, *_ in _plan(p.stdout)] == ["7", "8", "9"], p.stdout
    for exp, arm in (("stage1_nope", "clean_hd128"), ("stage2_context", "dm_consistent_hd128"), ("stage1_hd128", "no_such_arm")):
        p = _run(EXP / f"{exp}.sh", [arm], tmp_path, DRY=1)
        assert p.returncode == 2 and f"{arm} is not an arm of {exp}" in p.stderr, (exp, arm, p.stderr[-1000:])


def test_smoke_plan(tmp_path):
    p = _run(EXP / "stage1_muon.sh", [], tmp_path, DRY=1, SMOKE=1)
    lines = [l for l in p.stdout.splitlines() if l.startswith("  ")]
    assert p.returncode == 0 and len(lines) == 3 and all("SMOKE=1 bash" in l and "eval_arm" not in l for l in lines), p.stdout


def _stub_tree(tmp_path, train_rc):
    """A copy of scripts/ whose train / eval launchers are stand-ins that log each call; train_rc maps arm -> exit code."""
    root = tmp_path / "repo"
    shutil.copytree(REPO / "scripts", root / "scripts")
    log = tmp_path / "calls.log"
    rcs = " ".join(f"{a}={rc}" for a, rc in train_rc.items())
    for name, kind in (("train_arm.sh", "train"), ("stage2_arm.sh", "train"), ("eval_arm.sh", "eval"), ("eval_arm_64k.sh", "eval")):
        (root / "scripts" / "campaign" / name).write_text(f'''#!/bin/bash
echo "{kind} {name} $*" >> "{log}"
if [ "{kind}" = train ]; then for kv in {rcs}; do [ "${{kv%%=*}}" = "$1" ] && exit "${{kv#*=}}"; done; fi
exit 0
''')
    return root, log


def test_driver_flow(tmp_path):
    """Train then evaluate each run; a failed training skips its evaluation and is reported (exit 1); the rest still run."""
    root, log = _stub_tree(tmp_path, {"muon_dm_stale_hd128": 3})
    p = _run(root / "scripts" / "experiments" / "stage1_muon.sh", [], tmp_path)
    calls = log.read_text().splitlines()
    assert calls == ["train train_arm.sh muon_clean_hd128 11785", "eval eval_arm.sh Q35M_clean_sdpa_ffn8_hd128_11785_muon",
                     "train train_arm.sh muon_dm_stale_hd128 11785",
                     "train train_arm.sh muon_dm_matched_hd128 11785", "eval eval_arm.sh Q35M_dm_matched_ffn8_qkvo8_hd128_11785_muon"], calls
    assert p.returncode == 1 and "Q35M_dm_stale_ffn8_qkvo8_hd128_11785_muon:train" in p.stdout, p.stdout[-1500:]


def test_driver_stops_on_early_stop_and_skips_eval_in_smoke(tmp_path):
    root, log = _stub_tree(tmp_path, {"dm_matched_hd128": 75})
    p = _run(root / "scripts" / "experiments" / "stage2_context.sh", [], tmp_path)
    calls = log.read_text().splitlines()
    assert p.returncode == 75 and calls == ["train stage2_arm.sh clean_hd128 11785",
                                            "eval eval_arm_64k.sh Q35M_clean_sdpa_ffn8_hd128_11785_s2_64k",
                                            "train stage2_arm.sh clean_hd128 2",
                                            "eval eval_arm_64k.sh Q35M_clean_sdpa_ffn8_hd128_2_s2_64k",
                                            "train stage2_arm.sh dm_matched_hd128 11785"], calls
    log.unlink()
    p = _run(root / "scripts" / "experiments" / "stage1_nope.sh", [], tmp_path, SMOKE=1)
    calls = log.read_text().splitlines()
    assert p.returncode == 0 and len(calls) == 6 and all(c.startswith("train train_arm.sh nope_") for c in calls), calls
