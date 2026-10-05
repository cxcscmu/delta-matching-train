"""scripts/campaign/train_arm.sh end to end on one GPU with the tiny configs (DM_TRAIN_CONFIG): the arm table, the
`env` launch (ranks started by the script), the completion marker and the patched export, the idempotent relaunch,
SIGTERM forwarding to the trainer (exit 75, no export), and the resume that follows. This is the chain a job that a
scheduler may stop relies on."""
import json, os, signal, subprocess, time
import pytest
from _helpers import REPO, base_env, require_kernel, data_root, tokenizer_dir, python, read_metrics, dm_route, free_port

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.data]
ARM, SEED = "clean_hd128", "11785"


def _env(run_root, gpu, tag, **extra):
    env = base_env(gpu)
    env.update({"DM_WORKSPACE": str(run_root), "DM_DATA_ROOT": str(data_root()), "DM_TOKENIZER_DIR": str(tokenizer_dir()), "DM_PY": python(),
                "DM_TRAIN_CONFIG": str(REPO / "tests/configs/tiny_hybrid.yaml"), "DM_LAUNCH": "env", "DM_NPROC": "1",
                "DM_WANDB": "0", "TAG": tag, "DM_SAVE_INTERVAL": "3", "DM_MASTER_PORT": str(free_port())})
    env.update(extra)
    return env


def _run(run_root, gpu, tag, **extra):
    p = subprocess.run(["bash", "scripts/campaign/train_arm.sh", ARM, SEED], cwd=str(REPO), env=_env(run_root, gpu, tag, **extra),
                       capture_output=True, text=True, timeout=1800)
    return p.returncode, p.stdout + "\n" + p.stderr


def test_launcher_env_launch_completes_and_is_idempotent(run_root, root, tok_dir, test_gpu):
    tag = f"_t{os.getpid()}"
    rc, log = _run(run_root, test_gpu, tag, NUM_STEPS="6")
    assert rc == 0, log[-4000:]
    rn = "Q35M_clean_sdpa_ffn8_hd128_11785" + tag
    hf = run_root / "qwen35_m" / "hf" / rn
    assert (hf / "TRAIN_COMPLETE").exists(), log[-3000:]
    cfg = json.load(open(hf / "config.json"))
    assert cfg["attn_impl"] == "sdpa" and (hf / "config.train.json").exists(), "export not patched for the clean-kernel eval"
    assert json.load(open(hf / "config.train.json"))["layer_mixers"][-1] == "gqa", "the clean arm did not train the gqa core"
    assert "memmap-ppl RESULT" in log, "the 128-window val CE tail is missing"
    assert (run_root / "logs" / f"{rn}.rank0.log").exists() and (run_root / "logs" / f"{rn}.metrics.jsonl").exists()
    assert "dm" not in dm_route((run_root / "logs" / f"{rn}.rank0.log").read_text()), "the clean arm built the dm mixer"
    assert sorted(int(p.name.split("-")[-1]) for p in (run_root / "qwen35_m" / "ckpts" / rn).glob("checkpoint-*")) == [3, 6], "the launcher keeps SAVE_TOTAL_LIMIT=2 checkpoints: 3 and the final 6"
    rc, log = _run(run_root, test_gpu, tag, NUM_STEPS="6")
    assert rc == 0 and "already complete" in log, log[-2000:]


def test_launcher_forwards_sigterm_then_resumes(run_root, root, tok_dir, test_gpu):
    tag = f"_s{os.getpid()}"
    rn = "Q35M_clean_sdpa_ffn8_hd128_11785" + tag
    out = open(run_root / f"launcher{tag}.log", "w")
    p = subprocess.Popen(["bash", "scripts/campaign/train_arm.sh", ARM, SEED], cwd=str(REPO), env=_env(run_root, test_gpu, tag, NUM_STEPS="400", DM_SAVE_INTERVAL="1000"),
                         stdout=out, stderr=subprocess.STDOUT)
    t0 = time.time(); seen = False
    while time.time() - t0 < 900 and p.poll() is None:
        if any(r.get("event") == "metrics" and r.get("step", -1) >= 3 for r in read_metrics(run_root, rn)):
            seen = True; break
        time.sleep(2)
    assert seen, f"trainer never reached step 3; launcher log tail:\n{(run_root / f'launcher{tag}.log').read_text()[-3000:]}"
    os.kill(p.pid, signal.SIGTERM)
    rc = p.wait(timeout=600); out.close()
    log = (run_root / f"launcher{tag}.log").read_text()
    assert rc == 75 and "stopped early with a complete checkpoint" in log, f"exit {rc}; tail:\n{log[-3000:]}"
    ck = sorted(int(q.name.split("-")[-1]) for q in (run_root / "qwen35_m" / "ckpts" / rn).glob("checkpoint-*") if (q / ".complete").exists())
    assert ck and ck[-1] >= 3, ck
    assert not (run_root / "qwen35_m" / "hf" / rn / "TRAIN_COMPLETE").exists()
    rc, log = _run(run_root, test_gpu, tag, NUM_STEPS="400", STOP_AT_STEP=str(ck[-1] + 2))
    assert rc == 0 and (run_root / "qwen35_m" / "hf" / rn / "TRAIN_COMPLETE").exists(), log[-3000:]
    assert f"checkpoint-{ck[-1]}" in (run_root / "logs" / f"{rn}.rank0.log").read_text()


# arm -> (tiny config, the attention mixer the run trains (config.train.json, before the export patch), dm_mode)
ARMS = {
    "dm_stale_hd128": ("tiny_hybrid.yaml", "dm", "stale"),
    "dm_matched_hd128": ("tiny_hybrid.yaml", "dm", "matched"),
    "dm_consistent_hd128": ("tiny_hybrid.yaml", "dm", "stale"),
    "nope_dm_matched_hd128": ("tiny_hybrid.yaml", "dm", "matched"),
    "muon_clean_hd128": ("tiny_hybrid.yaml", "gqa", None),
    "mla_clean_hd128": ("tiny_mla.yaml", "mla", None),
    "mla_dm_matched_hd128": ("tiny_mla.yaml", "mla_dm", "matched"),
    "mamba2_dm_matched_hd128": ("tiny_mamba2.yaml", "dm", "matched"),
    "kda_clean_hd128": ("tiny_kda.yaml", "gqa", None),
    "kda_dm_matched_hd128": ("tiny_kda.yaml", "dm", "matched"),
}
PATCHED = {"dm": "gqa", "mla_dm": "mla"}   # train_arm.sh's export patch: every arm is scored through the bf16 core


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_launcher_arms(arm, run_root, root, tok_dir, test_gpu):
    """The fp8 arms, the NoPE and Muon variants and the MLA / Mamba2 / KDA hybrid arms through the launcher with their
    tiny configs: the arm table (the attention core the run really trains, read from config.train.json and the rank-0
    log banners), the fp8 pipeline (for mla_dm with the MLA latent projection in fp8), the consistent-dO delta, the NoPE
    and Muon switches, the export patch (mla_dm -> mla, dm -> gqa, the fp8-only keys dropped), the completion marker
    and the val-CE tail."""
    cfg_name, mixer, mode = ARMS[arm]
    if mode is not None:
        require_kernel()
    tag = f"_n{os.getpid()}"
    env = _env(run_root, test_gpu, tag, NUM_STEPS="3", DM_TRAIN_CONFIG=str(REPO / "tests/configs" / cfg_name))
    p = subprocess.run(["bash", "scripts/campaign/train_arm.sh", arm, SEED], cwd=str(REPO), env=env, capture_output=True, text=True, timeout=1800)
    log = p.stdout + "\n" + p.stderr
    assert p.returncode == 0, f"{arm}: exit {p.returncode}; tail:\n{log[-4000:]}"
    rn = subprocess.run(["bash", "-c", f"source scripts/campaign/env.sh; source scripts/campaign/arms.sh; TAG={tag} dm_run_name {arm} {SEED}"],
                        cwd=str(REPO), env=env, capture_output=True, text=True).stdout.strip()
    hf = run_root / "qwen35_m" / "hf" / rn
    assert rn and (hf / "TRAIN_COMPLETE").exists(), f"{arm}: no completion marker for {rn!r}; tail:\n{log[-3000:]}"
    trained = json.load(open(hf / "config.train.json"))
    assert trained["layer_mixers"][-1] == mixer, f"{arm}: trained {trained['layer_mixers']}, expected the {mixer} core"
    if mode is not None:
        assert trained.get("dm_mode") == mode, f"{arm}: config.train.json dm_mode {trained.get('dm_mode')!r}, expected {mode}"
    c = json.load(open(hf / "config.json"))
    assert c["layer_mixers"][-1] == PATCHED.get(mixer, mixer) and c["attn_impl"] == "sdpa" and "dm_mode" not in c, c
    assert "memmap-ppl RESULT" in log, "no val CE tail"
    route = dm_route((run_root / "logs" / f"{rn}.rank0.log").read_text())
    if mixer == "dm":
        assert route.get("dm") == mode, f"{arm}: the [dm] DMAttention banner says {route}, expected dm_mode={mode}"
    elif mixer == "mla_dm":
        assert route.get("mla_dm") == mode and route.get("mla_core") == f"dm:{mode}", f"{arm}: MLA banners {route}"
    elif mixer == "mla":
        assert route.get("mla_core") == "sdpa" and "mla_dm" not in route, f"{arm}: MLA banners {route}"
    else:
        assert "dm" not in route, f"{arm}: a clean arm built the dm mixer: {route}"
    rank0 = (run_root / "logs" / f"{rn}.rank0.log").read_text()
    assert ("DM_DELTA_Q8=1" in rank0) is (arm == "dm_consistent_hd128"), f"{arm}: the delta-from-codes line"
    assert bool(trained.get("nope")) is arm.startswith("nope_"), f"{arm}: config.train.json nope={trained.get('nope')!r}"
    assert ("[optim] MUON:" in rank0) is arm.startswith("muon_"), f"{arm}: the Muon optimizer banner"
