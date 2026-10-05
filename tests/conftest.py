"""pytest configuration: markers, skips, and the session fixtures shared by the smoke tests (tests/README.md).

    pytest tests -m "not gpu"                 # any machine: environment, CPU logic, dry launcher checks, data checks
    pytest tests                              # a GPU node: everything (the smoke tests take a few minutes)
    DM_TEST_GPU=3 pytest tests/test_train_smoke.py    # pick the GPU the subprocess smokes run on (default 0)
"""
import os, shutil, sys, time
from pathlib import Path
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import SRC, NEMO, data_root, tokenizer_dir, workspace, train_tiny  # noqa: E402

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU (Hopper for the fp8 kernels)")
    config.addinivalue_line("markers", "slow: launches training or evaluation subprocesses (minutes)")
    config.addinivalue_line("markers", "data: needs the data root (DM_DATA_ROOT, else workspace/data; data/README.md)")


def _cuda_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    has_cuda = _cuda_ok()
    root = data_root()
    has_data = (root / NEMO / "nemotron_rs_val.bin").exists()
    for it in items:
        if it.get_closest_marker("gpu") and not has_cuda:
            it.add_marker(pytest.mark.skip(reason="no CUDA device visible"))
        if it.get_closest_marker("data") and not has_data and not os.environ.get("DM_DATA_ROOT"):
            it.add_marker(pytest.mark.skip(reason=f"no data root found (looked at {root}); set DM_DATA_ROOT"))


@pytest.fixture(scope="session")
def root():
    """The data root. Nothing configured and nothing found: skip; DM_DATA_ROOT set but incomplete: fail."""
    r = data_root()
    if not (r / NEMO / "nemotron_rs_val.bin").exists():
        msg = f"data root {r} has no {NEMO}/nemotron_rs_val.bin (docs/SETUP.md section 4)"
        if os.environ.get("DM_DATA_ROOT"):
            pytest.fail(msg)
        pytest.skip(f"no data root found: {msg}; set DM_DATA_ROOT")
    return r


@pytest.fixture(scope="session")
def tok_dir():
    """The Llama-2 tokenizer. Nothing configured and nothing found: skip; DM_TOKENIZER_DIR or DM_DATA_ROOT set but
    without a tokenizer: fail."""
    d = tokenizer_dir()
    if d is None:
        msg = ("no tokenizer found: DM_TOKENIZER_DIR, <data root>/tokenizer/llama2_32k or the Llama-2 snapshot in the HF "
               "hub cache (HF_HOME) must hold tokenizer.model")
        if os.environ.get("DM_TOKENIZER_DIR") or os.environ.get("DM_DATA_ROOT"):
            pytest.fail(msg)
        pytest.skip(msg)
    return d


@pytest.fixture(scope="session")
def test_gpu():
    return os.environ.get("DM_TEST_GPU", "0")


@pytest.fixture(scope="session")
def run_root():
    """Scratch for the smoke runs under $DM_WORKSPACE/tests_tmp (default <repo>/workspace, which is git-ignored; point
    DM_WORKSPACE at node-local disk); removed at the end of the session unless DM_KEEP_TEST_RUNS=1."""
    d = workspace() / "tests_tmp" / f"{time.strftime('%m%d_%H%M%S')}_{os.getpid()}"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    if os.environ.get("DM_KEEP_TEST_RUNS", "0") != "1":
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def clean_export(run_root, root, tok_dir, test_gpu):
    """A 3-step tiny clean run's HF export (bf16 attention, no fp8), shared by the eval smoke tests and the HF
    conversion test. DM_TEST_EVAL_CONFIG=<a tiny yaml of tests/configs> (default tiny_hybrid.yaml, the GDN hybrid) runs
    them on another hybrid's export, e.g. DM_TEST_EVAL_CONFIG=tiny_kda.yaml pytest tests/test_eval_smoke.py."""
    cfg = os.environ.get("DM_TEST_EVAL_CONFIG")
    rc, log, hf, dt = train_tiny("tiny_clean_for_eval", run_root, root, tok_dir, test_gpu,
                                 extra_args=["--model.attn_impl", "sdpa"],
                                 config=(Path(__file__).resolve().parent / "configs" / cfg) if cfg else None)
    assert rc == 0, f"tiny clean training failed (rc={rc}); console tail:\n{log[-3000:]}"
    assert (hf / "config.json").exists(), f"no HF export at {hf}"
    return hf
