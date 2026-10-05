"""Shared helpers of the test suite: where things are, the environment of the subprocesses, the tiny arms.

Locations follow scripts/campaign/env.sh (DM_WORKSPACE, DM_DATA_ROOT, DM_TOKENIZER_DIR, HF_HOME), with one difference:
DM_PY defaults to the interpreter running pytest (env.sh: the repo's .venv, then PATH)."""
import importlib.util, os, pathlib, re, socket, subprocess, sys, time
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "src"
NEMO = "nemotron_cc_v2d1_hq_dqa"
LN_VOCAB = 10.373491  # ln(32000): the cross-entropy of a uniform model over the 32k vocabulary


def workspace() -> pathlib.Path:
    return pathlib.Path(os.environ.get("DM_WORKSPACE", str(REPO / "workspace")))


def data_root() -> pathlib.Path:
    """DM_DATA_ROOT, else <workspace>/data (the data root data/ builds, data/README.md)."""
    r = os.environ.get("DM_DATA_ROOT")
    return pathlib.Path(r) if r else workspace() / "data"


def hf_home() -> pathlib.Path:
    """HF_HOME, else <data root>/hf_cache (the evaluation datasets data/prepare_eval.py downloads)."""
    return pathlib.Path(os.environ["HF_HOME"]) if os.environ.get("HF_HOME") else data_root() / "hf_cache"


def tokenizer_dir() -> pathlib.Path | None:
    """DM_TOKENIZER_DIR, else <data root>/tokenizer/llama2_32k; None when it holds no tokenizer.model."""
    d = pathlib.Path(os.environ.get("DM_TOKENIZER_DIR") or data_root() / "tokenizer" / "llama2_32k")
    return d if (d / "tokenizer.model").exists() else None


def require_kernel():
    """The fp8 arms need the delta_matching kernel package (docs/SETUP.md section 3); a missing package is a failure of
    the environment, not a reason to skip."""
    if importlib.util.find_spec("delta_matching") is None:
        pytest.fail("the delta_matching kernel package is not importable: install the delta-matching kernel package "
                    "(docs/SETUP.md section 3) or put its checkout on PYTHONPATH")


_BF16DP: dict = {}


def kernel_bf16dp() -> bool:
    """Whether the delta_matching library the tests load (DM_SO, else the package's own) is the bf16-dP variant, i.e. a
    DM_BF16_DP=1 build (build_bf16dp() == 1). Read once per session, in a subprocess with the tests' environment (the
    library the subprocess tests load); a missing package fails as in require_kernel."""
    if "flag" not in _BF16DP:
        require_kernel()
        p = subprocess.run([python(), "-c", "from delta_matching.core import _ops; print('FLAG', int(_ops().build_bf16dp()))"],
                           capture_output=True, text=True, env=base_env(), timeout=300)
        if p.returncode != 0 or "FLAG" not in p.stdout:
            pytest.fail(f"loading the delta_matching library failed:\n{p.stderr[-2000:]}")
        _BF16DP["flag"] = p.stdout.split("FLAG", 1)[1].split()[0] == "1"
    return _BF16DP["flag"]


def require_standard_kernel(what: str):
    """Skip a test of the stale / matched kernel paths when the loaded library is the bf16-dP variant (DM_BF16_DP=1):
    that build serves dm_mode bf16_dp only and refuses the other modes by design (the refusal itself is checked by
    test_kernels.py::test_dm_mixer_checks_the_bf16dp_build_flag). A missing package still fails (require_kernel)."""
    if kernel_bf16dp():
        pytest.skip(f"{what}: the loaded delta_matching library is the bf16-dP variant (a DM_BF16_DP=1 build, "
                    "build_bf16dp() == 1), which serves dm_mode bf16_dp only and refuses the stale / matched paths by "
                    "design; run this test against a DM_BF16_DP=0 build (unset DM_SO)")


def python() -> str:
    return os.environ.get("DM_PY", sys.executable)


def torchrun() -> str:
    return os.environ.get("DM_TORCHRUN", str(pathlib.Path(python()).parent / "torchrun"))


def free_port() -> int:
    """A TCP port nobody listens on right now (for a subprocess's torch.distributed rendezvous)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# The behaviour knobs of the launchers, the experiment drivers, the trainer, the kernels and the evaluators that a
# user's shell may carry (docs/SETUP.md, scripts/campaign/*.sh, scripts/experiments/*.sh). The subprocesses of the tests
# never inherit them: a test sets the ones it needs. Only the kernel library's location passes through (DM_SO names the library the
# delta_matching package loads, as PYTHONPATH names the code).
_SCRUB = ("EXTRA_ARGS", "TAG", "SMOKE", "STOP_AT_STEP", "NUM_STEPS", "SAVE_TOTAL_LIMIT", "FORCE", "NAME_ONLY", "WANDB_JT",
          "ARM", "SEED", "SEEDS", "DRY", "EVAL", "EVAL_ENGINE", "ENGINE_MAX_SEQS",
          "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
_SCRUB_PREFIXES = ("DM_", "S2_")
_KEEP = ("DM_SO", "DM_JIT_DIR", "DM_FA_SRC")


def clean_env() -> dict:
    """The caller's environment minus the knobs above."""
    return {k: v for k, v in os.environ.items()
            if k in _KEEP or not (k in _SCRUB or k.startswith(_SCRUB_PREFIXES))}


def base_env(gpu: str | None = None) -> dict:
    """The environment for a subprocess that imports the repo (trainer, eval): the scrubbed environment, PYTHONPATH,
    caches, the test workspace, no wandb."""
    env = clean_env()
    env["PYTHONPATH"] = str(SRC) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["DM_WORKSPACE"] = str(workspace())
    env.setdefault("HF_HOME", str(hf_home()))
    env.setdefault("TRITON_CACHE_DIR", str(workspace() / "triton_cache"))
    env.setdefault("TILELANG_CACHE_DIR", str(workspace() / "tilelang_cache"))
    env["WANDB_MODE"] = "disabled"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def memmap_ppl_env(gpu) -> dict:
    """base_env for evaluation.ppl.memmap_ppl, which initializes a single-process torch.distributed group."""
    env = base_env(gpu)
    env.update({"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(free_port())})
    return env


def dry_env(tmp_path, **extra) -> dict:
    """The environment of the dry launcher checks (no GPU, no data): the scrubbed environment with every GPU hidden
    (CUDA_VISIBLE_DEVICES empty: the checks never need one, and a pre-flight that imports the model stack must not touch
    a busy GPU), and a workspace and a data root under tmp_path."""
    env = clean_env()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env.update({"DM_WORKSPACE": str(tmp_path), "DM_DATA_ROOT": str(tmp_path / "data"), "DM_PY": python()})
    env.update({k: str(v) for k, v in extra.items()})
    return env


def fake_data_root(tmp_path) -> pathlib.Path:
    """<tmp_path>/data holding the files the launchers check for before they start anything: the stage-1 training
    stream and the tokenizer (empty files)."""
    d = tmp_path / "data"
    (d / NEMO).mkdir(parents=True, exist_ok=True)
    (d / NEMO / "nemotron_rs_train.bin").write_bytes(b"")
    (d / "tokenizer" / "llama2_32k").mkdir(parents=True, exist_ok=True)
    (d / "tokenizer" / "llama2_32k" / "tokenizer.model").write_bytes(b"")
    return d


def dm_env(stale: bool) -> dict:
    """The Delta-Matching runtime switches scripts/campaign/train_arm.sh and arms.sh export for the fp8 arms. Every one
    of them is also the code default; the launcher exports them so a run's environment is explicit in its log."""
    e = {"DM_SHARED_X8": "1", "DM_FP8_FSDP_AG": "1", "DM_FP8_PRECOMPUTE": "1",
         "DM_FP8GLUE": "1", "DM_FP8GLUE_SCALING": "dynamic", "DM_FP8GLUE_SAVE_HT": "0"}
    if stale:
        e["DM_STALE_DOQ2S"] = "1"
    return e


# ---------------------------------------------------------------- the GDN:GQA arms on the tiny config
# The trainer arguments and switches of scripts/campaign/arms.sh + train_arm.sh, minus the M-tier geometry (the tiny
# yaml keeps its own) and the data and output paths (tiny_cmd passes them).
CE_RE = re.compile(r"ce_loss'?\s*[:=]\s*([0-9.]+|nan|inf)")
GLUE = {"DM_FP8_FSDP_AG": "1", "DM_FP8_PRECOMPUTE": "1", "DM_FP8GLUE": "1", "DM_FP8GLUE_SCALING": "dynamic",
        "DM_FP8GLUE_SAVE_HT": "0"}
COMMON = ["--fp8", "true", "--compile_mlp", "true", "--model.fused_swiglu", "false", "--model.fused_rmsnorm", "true",
          "--compile", "false", "--logging_steps", "1"]
DM_LM = "[gated_deltanet, gated_deltanet, gated_deltanet, dm]"
DM_MODES = {"dm_stale_hd128": "stale", "dm_matched_hd128": "matched", "dm_consistent_hd128": "stale",
            "dm_stale_hd256": "stale", "dm_matched_hd256": "matched"}


def arm_args(arm: str) -> tuple[list, dict]:
    """(trainer arguments, environment) of a GDN:GQA arm on tests/configs/tiny_hybrid.yaml. The head_dim-256 arms add
    --model.head_dim 256 (the tiny yaml is head_dim 128); the consistent-dO arm is the stale arm plus DM_DELTA_Q8=1, as
    arms.sh does."""
    if arm == "clean_hd128":
        return COMMON + ["--model.attn_impl", "sdpa", "--model.fused_rope", "true", "--fp8_scope", "ffn"], \
            {**GLUE, "DM_SDPA_BACKEND": "cudnn"}
    mode = DM_MODES[arm]
    args = COMMON + ["--model.layer_mixers", DM_LM, "--model.dm_mode", mode, "--model.dm_fused_producers", "true",
                     "--fp8_scope", "ffn_qkvo"]
    if arm.endswith("_hd256"):
        args += ["--model.head_dim", "256"]
    env = {**GLUE, **dm_env(stale=(mode != "matched"))}
    if arm == "dm_consistent_hd128":
        env["DM_DELTA_Q8"] = "1"
    return args, env


def dm_route(log: str) -> dict:
    """What the attention banners of a run's log say: "dm" = the dm_mode of the `[dm] DMAttention` banner, "mla_dm" =
    the delta of the `[mla_dm]` line, "mla_core" = the core of the `[mla]` banner; a key is absent with its banner."""
    out = {}
    for key, pat in (("dm", r"\[dm\] DMAttention:.*?\bdm_mode=(\w+)"), ("mla_dm", r"\[mla_dm\] .*?\bdelta=(\w+)"),
                     ("mla_core", r"\[mla\] \w+:.*?\bcore=([\w:]+)")):
        m = re.search(pat, log)
        if m:
            out[key] = m.group(1)
    return out


def tiny_cmd(name, run_root, root, tok_dir, steps=3, extra_args=(), config=None):
    """The trainer command line for tests/configs/tiny_hybrid.yaml (single process, one GPU); `config` picks another
    tiny yaml of tests/configs (tiny_mla.yaml, tiny_mamba2.yaml, tiny_kda.yaml)."""
    val = root / NEMO / "nemotron_rs_val.bin"
    dl = root / NEMO / "nemotron_rs_val.doc_lengths.npy"
    hf = run_root / "hf" / name
    cmd = [python(), "-m", "pretrain.train", "--config", str(config or REPO / "tests" / "configs" / "tiny_hybrid.yaml"),
           "--run_name", name, "--num_steps", str(steps), "--wandb", "false",
           "--data.paths", str(val), "--data.doc_lengths", str(dl), "--tokenizer_dir", str(tok_dir),
           "--hf_save_dir", str(hf), "--ckpt_root_dir", str(run_root / "ckpts"), "--log_root_dir", str(run_root / "logs")]
    return cmd + list(extra_args), hf


def train_tiny(name, run_root, root, tok_dir, gpu, extra_args=(), extra_env=None, steps=3, config=None):
    """Run the trainer on tests/configs/tiny_hybrid.yaml (or `config`) for a few steps; returns (rc, log text, export dir, seconds)."""
    cmd, hf = tiny_cmd(name, run_root, root, tok_dir, steps, extra_args, config=config)
    env = base_env(gpu)
    env.update(extra_env or {})
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=1800)
    log = p.stdout + "\n" + p.stderr
    (run_root / f"{name}.console.log").write_text(log)
    return p.returncode, log, hf, time.time() - t0


def read_metrics(run_root, name):
    """The trainer's local metrics log as a list of dicts (empty if absent); a line still being written is skipped."""
    import json
    p = run_root / "logs" / f"{name}.metrics.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out
