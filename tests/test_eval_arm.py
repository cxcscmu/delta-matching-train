"""scripts/campaign/eval_arm.sh's guards, CPU only (no model is loaded, no benchmark runs): an export that was not
patched for the clean-kernel eval (a dm or mla_dm core, or an fp8 attn_impl, still in config.json) is refused with exit 2
before any benchmark starts, and a run whose EVAL_COMPLETE marker exists exits 0 at once (docs/SETUP.md sections 6, 8)."""
import json
import shutil
import subprocess

import pytest

from _helpers import REPO, NEMO, dry_env

RN = "Q35M_test_arm_11785"
GDN = ["gated_deltanet"] * 3


def _setup(tmp_path, config):
    """A workspace with a fake export (config.json only) and a data root with a non-empty val stream."""
    hf = tmp_path / "qwen35_m" / "hf" / RN
    hf.mkdir(parents=True)
    (hf / "config.json").write_text(json.dumps(config))
    d = tmp_path / "data" / NEMO
    d.mkdir(parents=True)
    (d / "nemotron_rs_val.bin").write_bytes(b"\x01\x00" * 64)
    # a guard that let an export through would start the benchmarks: make every launch fail at once instead
    return dry_env(tmp_path, DM_NPROC=1, DM_TORCHRUN=shutil.which("false") or "/bin/false")


def _eval(env, windows="4"):
    return subprocess.run(["bash", str(REPO / "scripts" / "campaign" / "eval_arm.sh"), RN, windows], capture_output=True,
                          text=True, env=env, timeout=120)


@pytest.mark.parametrize("label,config", [
    ("dm", {"attn_impl": "sdpa", "layer_mixers": GDN + ["dm"], "mixer": "gqa", "dm_mode": "matched"}),
    ("mla_dm", {"attn_impl": "sdpa", "layer_mixers": GDN + ["mla_dm"], "mixer": "mla", "dm_mode": "matched"}),
    ("pure_dm", {"attn_impl": "sdpa", "layer_mixers": None, "mixer": "dm", "dm_mode": "stale"}),
    ("te_fp8", {"attn_impl": "te_fp8", "layer_mixers": GDN + ["gqa"], "mixer": "gqa"}),
])
def test_eval_arm_refuses_an_unpatched_export(label, config, tmp_path):
    p = _eval(_setup(tmp_path, config))
    out = p.stdout + p.stderr
    assert p.returncode == 2 and "not patched" in out, f"{label}: an unpatched export was accepted (rc={p.returncode}):\n{out[-1500:]}"
    assert "EVAL START" not in out and not (tmp_path / "qwen35_m" / "eval" / RN / "valce_n4.txt").exists(), out[-1500:]


def test_eval_arm_skips_a_completed_run(tmp_path):
    env = _setup(tmp_path, {"attn_impl": "sdpa", "layer_mixers": GDN + ["gqa"], "mixer": "gqa"})
    out_dir = tmp_path / "qwen35_m" / "eval" / RN
    out_dir.mkdir(parents=True)
    (out_dir / "EVAL_COMPLETE").write_text("done\n")
    p = _eval(env)
    out = p.stdout + p.stderr
    assert p.returncode == 0 and "already complete" in out, f"rc={p.returncode}\n{out[-1500:]}"
    assert "EVAL START" not in out and sorted(q.name for q in out_dir.iterdir()) == ["EVAL_COMPLETE"], out[-1500:]
