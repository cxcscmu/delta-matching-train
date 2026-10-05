"""Environment checks: Python, torch + CUDA 13, Hopper, the pinned packages, nvcc, the repo's imports, the kernel package.
Run anywhere with `-m "not gpu"`; on a GPU node without the filter."""
import importlib, os, re, shutil, subprocess, sys
import pytest
from _helpers import require_kernel, python, base_env

# package -> required version prefix (None = import only). The pins are the environment the study's runs were produced
# with (env/requirements-cu130-hopper.lock.txt; fla is pinned to 0.5.2, docs/benchmark.md "Software" has which runs used
# 0.5.0); a different version is a failure to look at, not necessarily a bug.
PINS = {
    "torch": "2.12", "triton": "3.7", "transformers": "5.9", "datasets": "5.", "lm_eval": "0.4.12", "torchao": "0.17",
    "fla": "0.5.2", "flash_attn": "2.8", "numpy": "2.", "wandb": None, "safetensors": None, "sentencepiece": None,
    "einops": None, "yaml": None, "portalocker": None, "liger_kernel": None, "causal_conv1d": None,
    "mamba_ssm": None, "huggingface_hub": None, "tokenizers": None, "tqdm": None, "psutil": None,
}


def test_python_version():
    assert sys.version_info >= (3, 12), f"Python 3.12 expected, got {sys.version}"


def test_torch_is_cuda13_build():
    import torch
    assert torch.__version__.startswith("2.12"), torch.__version__
    assert torch.version.cuda and torch.version.cuda.startswith("13."), f"torch built for CUDA {torch.version.cuda}, need 13.x"
    assert torch.backends.cudnn.version() >= 91000, torch.backends.cudnn.version()


@pytest.mark.parametrize("pkg", sorted(PINS))
def test_package_importable_and_pinned(pkg):
    mod = importlib.import_module(pkg)
    want = PINS[pkg]
    if want:
        have = getattr(mod, "__version__", None) or importlib.metadata.version(pkg.replace("_", "-") if pkg == "lm_eval" else pkg)
        assert str(have).startswith(want), f"{pkg} {have}, expected {want}*"


def test_fa3_hopper_wheel_importable():
    import flash_attn_interface
    assert hasattr(flash_attn_interface, "flash_attn_func")


def test_torchao_float8_api():
    from torchao.float8 import convert_to_float8_training, Float8LinearConfig  # noqa: F401


def test_nvcc_is_cuda13():
    cands = [os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "nvcc"), "/usr/local/cuda-13.0/bin/nvcc", shutil.which("nvcc") or ""]
    nvcc = next((c for c in cands if c and os.path.exists(c)), None)
    assert nvcc, "nvcc not found (CUDA_HOME, /usr/local/cuda-13.0 or PATH)"
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"release (\d+)\.(\d+)", out)
    assert m and m.group(1) == "13", f"nvcc release {m.group(0) if m else out[-200:]}, need 13.x"


def test_repo_modules_import():
    for m in ("pretrain.train", "pretrain.modeling", "pretrain.modeling.dm_attention", "pretrain.modeling.attention",
              "pretrain.modeling.mla", "pretrain.parallel.wrap", "pretrain.trainer", "pretrain.hf",
              "evaluation.meta_eval", "evaluation.ppl.memmap_ppl", "inference.engine",
              "kernel.dm_port.orchestrator", "kernel.dm_port.diag", "kernel.dm_port.fused_producers.tier3a_ropeq",
              "kernel.fp8glue"):
        importlib.import_module(m)


def test_delta_matching_package_importable():
    """The fp8 attention kernel is the separate delta_matching package (docs/SETUP.md section 3): its public API imports
    without a GPU; the compiled library is checked by the gpu tests below."""
    require_kernel()
    import delta_matching
    from delta_matching import dm_attn_func, dm_attn_func_bhnd, blocks, kernel_version  # noqa: F401
    assert delta_matching.__version__
    assert blocks(128) == (128, 64) and blocks(256) == (64, 32), (blocks(128), blocks(256))
    kv = kernel_version()                    # the package reports its version, code sha and per-head-dim numerics
    assert kv and isinstance(kv, (str, dict)), kv


def _probe_package(expr):
    """Evaluate `expr` over the loaded library's ops `o` (the build-flag ops the training code checks) in a subprocess."""
    code = f"from delta_matching.core import _ops; o = _ops(); print('PROBE', {expr})"
    p = subprocess.run([python(), "-c", code], capture_output=True, text=True, env=base_env(), timeout=300)
    assert p.returncode == 0, f"loading the delta_matching library failed:\n{p.stderr[-2000:]}"
    return p.stdout.strip().split("PROBE", 1)[1].strip()


@pytest.mark.gpu
def test_delta_matching_library_is_the_hd128_build():
    """The campaign's head_dim-128 arms need a library built with the head_dim-128 instantiations and the global-memory
    twins (the package's shipped flag set); a library built without them fails at model build, so it fails here first."""
    require_kernel()
    assert _probe_package("(int(o.build_hd128()), int(o.build_gt()))") == "(1, 1)"


@pytest.mark.gpu
def test_delta_matching_library_reports_bf16dp_flag():
    """build_bf16dp() tells whether the library keeps the dP matmul in bf16 (DM_BF16_DP=1: the bf16_dp arm's build);
    the dm mixer asserts it against dm_mode, so the flag must at least be readable (0 or 1)."""
    require_kernel()
    assert _probe_package("int(o.build_bf16dp())") in ("0", "1")


@pytest.mark.gpu
def test_gpu_is_hopper():
    import torch
    cap = torch.cuda.get_device_capability(0)
    assert cap == (9, 0), f"the fp8 kernels are sm_90a builds; this GPU is sm_{cap[0]}{cap[1]} ({torch.cuda.get_device_name(0)})"
