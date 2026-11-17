"""The fp8 attention kernel bundled with this repository (kernels/delta-matching, kernels/README.md): the vendored source
is the one kernels/PROVENANCE.json describes, carries the KDA-evolved flag set, and is what the environment runs. No GPU
is needed: the installed package's build stamp (kernel_version()["code_sha"], the hash of the csrc/ it was built from)
is compared with the bundled source. DM_EXPECT_BUNDLED_KERNEL=0 turns the installed-build check off for an environment
that runs another kernel on purpose (e.g. the original release, DM_KERNEL_SRC=... scripts/install_kernel.sh)."""
import importlib.util, json, os, re, subprocess, sys
import pytest
from _helpers import REPO, python, base_env, require_kernel
from pretrain.trainer.base_trainer import kernel_py_sha

BUNDLE = REPO / "kernels" / "delta-matching"
PROV = json.loads((REPO / "kernels" / "PROVENANCE.json").read_text())
# the flags the KDA-evolved kernel changed against the original release (kernels/PROVENANCE.json)
KDA_FLAGS = {"FA3FP11_HD128_P2ST": 3, "FA3FP11_HD128_BN256": 0, "FA3FP11_HD128_BM192": 1, "FA3FP11_HD128_FWDST": 3,
             "FA3FP11_HD128_FWDLAZY": 1, "FA3FP11_HD128_FWDDFOLD": 1}


def _bundle_buildcfg():
    """The bundle's delta_matching/_buildcfg.py, loaded by path without writing bytecode into the vendored tree."""
    spec = importlib.util.spec_from_file_location("_dm_bundle_buildcfg", BUNDLE / "delta_matching" / "_buildcfg.py")
    mod = importlib.util.module_from_spec(spec)
    keep, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = keep
    return mod


def test_bundle_matches_provenance():
    """kernels/delta-matching is the source PROVENANCE.json names: its csrc/ hashes to the recorded code_sha (an edit of
    the vendored kernel must come with a new PROVENANCE.json entry, kernels/README.md "Updating the bundle")."""
    cfg = _bundle_buildcfg()
    assert os.path.samefile(cfg.CSRC, BUNDLE / "csrc"), cfg.CSRC
    assert cfg.csrc_sha() == PROV["code_sha"], \
        f"kernels/delta-matching/csrc hashes to {cfg.csrc_sha()}, PROVENANCE.json records {PROV['code_sha']}"
    assert PROV["code_sha"] != PROV["base"]["code_sha"], "the bundle must not be the original release"
    py = kernel_py_sha(BUNDLE / "delta_matching")      # the Python half (producers, split), which code_sha does not cover
    assert py == PROV["py_sha"], f"kernels/delta-matching/delta_matching/*.py hash to {py}, PROVENANCE.json records {PROV['py_sha']}"
    for f in ("setup.py", "pyproject.toml", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "delta_matching/__init__.py",
              "tests/test_api.py", "tests/test_dm_attn.py"):
        assert (BUNDLE / f).is_file(), f"kernels/delta-matching/{f} is missing"


def test_bundle_carries_the_kda_flag_set():
    """The certified flag set of the bundled build (delta_matching/_buildcfg.py) is the KDA one, and csrc/dm_flagset.h
    (which stops a build with any other value) agrees with it."""
    flags = _bundle_buildcfg().FLAGS
    bad = {k: (flags.get(k), v) for k, v in KDA_FLAGS.items() if flags.get(k) != v}
    assert not bad, f"bundled _buildcfg.FLAGS differ from the KDA set (have, want): {bad}"
    guard = (BUNDLE / "csrc" / "dm_flagset.h").read_text()
    for k, v in KDA_FLAGS.items():
        assert re.search(rf"\({k}\)\s*!=\s*{v}\b", guard), f"csrc/dm_flagset.h does not pin {k} = {v}"


def test_bundle_pins_the_flash_attention_commits():
    """scripts/install_kernel.sh fetches flash-attention and CUTLASS at the commits the package pins."""
    init = (BUNDLE / "delta_matching" / "__init__.py").read_text()
    fa = PROV["build_dependencies"]["flash_attention_commit"]; cut = PROV["build_dependencies"]["cutlass_commit"]
    assert f'FA_SRC_COMMIT = "{fa}"' in init and f'CUTLASS_COMMIT = "{cut}"' in init


def test_installed_kernel_is_the_bundle():
    """The delta_matching package the training code imports was built from kernels/delta-matching: its build stamp
    reports the bundle's code_sha, its Python files are the bundle's (py_sha), its compiled library is loaded from the
    installed package, and it is the installed package, not the source tree (which has no library)."""
    if os.environ.get("DM_EXPECT_BUNDLED_KERNEL", "1") == "0":
        pytest.skip("DM_EXPECT_BUNDLED_KERNEL=0: this environment runs another kernel build on purpose")
    if os.environ.get("DM_SO"):
        pytest.skip(f"DM_SO={os.environ['DM_SO']} overrides the package's library; its build cannot be read from the stamp")
    require_kernel()
    code = ("import os, delta_matching as d; kv = d.kernel_version(); "
            "print('KV', kv.get('code_sha'), os.path.dirname(os.path.abspath(d.__file__)), kv.get('library'))")
    p = subprocess.run([python(), "-c", code], capture_output=True, text=True, env=base_env(), cwd="/", timeout=300)
    assert p.returncode == 0 and "KV" in p.stdout, f"importing delta_matching failed:\n{p.stderr[-2000:]}"
    sha, where, lib = p.stdout.split("KV", 1)[1].split()[:3]
    assert not os.path.realpath(where).startswith(os.path.realpath(BUNDLE)), \
        f"delta_matching is imported from the source tree {where}; install it with scripts/install_kernel.sh"
    assert os.path.realpath(lib).startswith(os.path.realpath(where) + os.sep), \
        f"the compiled library {lib} is not the installed package's ({where})"
    assert kernel_py_sha(where) == PROV["py_sha"], (
        f"the installed delta_matching's Python files ({where}) hash to {kernel_py_sha(where)}, not the bundle's "
        f"{PROV['py_sha']}: run scripts/install_kernel.sh")
    hint = {PROV["base"]["code_sha"]: " (the ORIGINAL release)"}.get(sha, "")
    assert sha == PROV["code_sha"], (
        f"the installed delta_matching was built from code_sha {sha}{hint}, not the bundled kernel {PROV['code_sha']}: "
        "run scripts/install_kernel.sh (or set DM_EXPECT_BUNDLED_KERNEL=0 to run another kernel on purpose)")


def _install_kernel(tmp_path, **extra):
    """scripts/install_kernel.sh in this environment, with a workspace under tmp_path (the refusals below stop before
    anything is built or installed)."""
    env = base_env()
    env.pop("DM_FA_SRC", None)   # a caller's complete checkout would turn a refusal into a real build and install
    env.update({"DM_PY": python(), "DM_WORKSPACE": str(tmp_path / "ws")})
    env.update({k: str(v) for k, v in extra.items()})
    p = subprocess.run(["bash", str(REPO / "scripts" / "install_kernel.sh")], capture_output=True, text=True, env=env,
                       timeout=600)
    if p.returncode == 2 and ("nvcc" in p.stderr or "torch" in p.stderr) and "flash-attention" not in p.stderr:
        pytest.skip(f"no CUDA 13 build environment here: {p.stderr.strip()[-300:]}")
    return p


def test_install_kernel_refuses_an_incomplete_flash_attention_dir(tmp_path):
    """An existing flash-attention directory without the headers (an interrupted fetch, a clone without submodules) is
    refused before any download, and nothing is left behind."""
    fa = tmp_path / "ws" / "build" / "flash-attention"
    (fa / "csrc").mkdir(parents=True)
    p = _install_kernel(tmp_path)
    assert p.returncode == 2 and "exists but has no hopper/" in p.stderr, (p.returncode, p.stdout[-1500:], p.stderr[-1500:])
    assert sorted(x.name for x in fa.parent.iterdir()) == ["flash-attention"]


def test_install_kernel_refuses_unpinned_flash_attention(tmp_path):
    """flash-attention / CUTLASS headers at other commits than the package pins are refused (code_sha covers csrc/
    only, so nothing later could tell) unless DM_FA_ALLOW_UNPINNED=1."""
    fa = tmp_path / "fa"
    for d in ("hopper", "csrc/cutlass/include"):
        (fa / d).mkdir(parents=True)
        (fa / d / "x.h").write_text("// x\n")
    git = ["git", "-C", str(fa), "-c", "user.name=t", "-c", "user.email=t@t"]
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "x"]):
        subprocess.run(git + cmd, check=True, capture_output=True)
    p = _install_kernel(tmp_path, DM_FA_SRC=fa)
    assert p.returncode == 2 and "the kernel pins" in p.stderr and "DM_FA_ALLOW_UNPINNED=1" in p.stderr, \
        (p.returncode, p.stdout[-1500:], p.stderr[-1500:])
