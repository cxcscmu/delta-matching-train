"""The paired per-window comparison of two arms (scripts/campaign/paired_analysis.py, paired_pair.py: the primitive every
arm contrast of docs/SETUP.md section 6 goes through), CPU only, on synthetic per-window artifacts
(`<run>_n2000_perwindow.npy` = np.stack([window ids, NLL]) as evaluation/ppl/memmap_ppl.py writes them): the paired mean
and standard error of a known difference, the alignment of the two arms by window id, non-finite windows dropped and
counted, the 8k / 64k artifact sets kept apart, and the two command lines."""
import importlib.util
import math
import subprocess

import numpy as np
import pytest

from _helpers import REPO, clean_env, python

CAMPAIGN = REPO / "scripts" / "campaign"
N = 2000


def _module(monkeypatch, d):
    monkeypatch.setenv("DM_PERWIN_DIR", str(d))
    spec = importlib.util.spec_from_file_location("paired_analysis_under_test", CAMPAIGN / "paired_analysis.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _save(d, run, ids, nll, n=N):
    np.save(d / f"{run}_n{n}_perwindow.npy", np.stack([np.asarray(ids, dtype=np.float64), np.asarray(nll, dtype=np.float64)]))


@pytest.fixture
def arms(tmp_path):
    """B: a per-window NLL; A = B + a known per-window difference (mean +0.02, s.d. 0.005)."""
    rng = np.random.default_rng(0)
    ids = np.arange(N)
    b = rng.normal(2.5, 0.3, N)
    diff = 0.02 + rng.normal(0.0, 0.005, N)
    _save(tmp_path, "A", ids, b + diff)
    _save(tmp_path, "B", ids, b)
    return tmp_path, ids, b, diff


def test_paired_mean_and_se(arms, monkeypatch):
    d, ids, b, diff = arms
    m = _module(monkeypatch, d)
    loaded = m.load()
    assert set(loaded) == {"A", "B"} and loaded["A"][0].dtype == np.int64, sorted(loaded)
    s = m.paired(loaded, "A", "B")
    assert s["n"] == N and s["dropped"] == 0, s
    assert math.isclose(s["mean"], diff.mean(), rel_tol=1e-9), (s["mean"], diff.mean())
    assert math.isclose(s["se"], diff.std(ddof=1) / math.sqrt(N), rel_tol=1e-9), s["se"]
    assert math.isclose(s["t"], s["mean"] / s["se"], rel_tol=1e-9) and s["t"] > 50, s
    assert math.isclose(s["hi"] - s["lo"], 2 * 1.96 * s["se"], rel_tol=1e-9), s
    assert s["frac_worse"] > 0.99 and s["r"] > 0.99, s
    r = m.paired(loaded, "B", "A")
    assert math.isclose(r["mean"], -s["mean"], rel_tol=1e-12) and math.isclose(r["se"], s["se"], rel_tol=1e-12), r


def test_paired_aligns_by_window_id(arms, monkeypatch):
    """The artifacts pair windows by id, not by position: an arm stored in another order, or scored on extra windows,
    gives the same contrast on the common windows; arms without a common window give no number at all."""
    d, ids, b, diff = arms
    rng = np.random.default_rng(1)
    perm = rng.permutation(N)
    _save(d, "A_shuffled", ids[perm], (b + diff)[perm])
    extra = np.arange(N, N + 300)
    _save(d, "B_more", np.concatenate([ids, extra]), np.concatenate([b, rng.normal(2.5, 0.3, extra.size)]))
    _save(d, "C_disjoint", extra, rng.normal(2.5, 0.3, extra.size))
    m = _module(monkeypatch, d)
    loaded = m.load(root=str(d))
    ref = m.paired(loaded, "A", "B")
    for a, bb in (("A_shuffled", "B"), ("A", "B_more"), ("A_shuffled", "B_more")):
        s = m.paired(loaded, a, bb)
        assert s["n"] == N and math.isclose(s["mean"], ref["mean"], rel_tol=1e-9) and math.isclose(s["se"], ref["se"], rel_tol=1e-9), (a, bb, s, ref)
    s = m.paired(loaded, "A", "C_disjoint")
    assert s["n"] == 0 and math.isnan(s["mean"]) and math.isnan(s["se"]), s


def test_paired_drops_and_counts_non_finite_windows(arms, monkeypatch):
    """A diverged run's NaN / inf windows are dropped from the contrast and counted, never averaged in."""
    d, ids, b, diff = arms
    a = b + diff
    bad = [5, 17, 1999]
    a[bad[0]], a[bad[1]], a[bad[2]] = np.nan, np.inf, np.nan
    _save(d, "A_nan", ids, a)
    m = _module(monkeypatch, d)
    s = m.paired(m.load(), "A_nan", "B")
    keep = np.setdiff1d(ids, bad)
    assert s["n"] == N - 3 and s["dropped"] == 3, s
    assert math.isclose(s["mean"], diff[keep].mean(), rel_tol=1e-9) and math.isclose(s["se"], diff[keep].std(ddof=1) / math.sqrt(N - 3), rel_tol=1e-9), s


def test_8k_and_64k_artifacts_stay_apart(arms, monkeypatch):
    d, ids, b, diff = arms
    np.save(d / "A_64k_n256_perwindow.npy", np.stack([np.arange(255.0), np.full(255, 3.0)]))   # stage 2: 255 windows scored
    np.save(d / "B_64k_n256_perwindow.npy", np.stack([np.arange(255.0), np.full(255, 2.9)]))
    _save(d, "A_64k", ids, b)                                   # a 64k artifact whose window count matches the 8k one
    m = _module(monkeypatch, d)
    assert set(m.load("8k")) == {"A", "B"}
    long = m.load("64k")
    assert set(long) == {"A", "B"} and long["A"][0].size == 255
    assert math.isclose(m.paired(long, "A", "B")["mean"], 0.1, rel_tol=1e-9)


def test_command_lines(arms):
    d, ids, b, diff = arms
    env = dict(clean_env(), DM_PERWIN_DIR=str(d))
    p = subprocess.run([python(), str(CAMPAIGN / "paired_pair.py"), "A", "B"], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0 and "[8k] mean NLL" in p.stdout and f"{diff.mean():+9.5f}" in p.stdout \
        and f"n={N} dropped=0" in p.stdout, p.stdout + p.stderr
    p = subprocess.run([python(), str(CAMPAIGN / "paired_pair.py"), "A", "no_such_run"], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 2 and "MISSING" in p.stdout and "no_such_run" in p.stdout, p.stdout + p.stderr
    p = subprocess.run([python(), str(CAMPAIGN / "paired_analysis.py"), "B", "A"], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == 0 and "A - B" in p.stdout and f"{diff.mean():+9.5f}" in p.stdout, p.stdout + p.stderr
