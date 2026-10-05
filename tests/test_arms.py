"""The arm table (scripts/campaign/arms.sh), CPU only: every arm `dm_arms` lists is in the arms.sh header and in the
table below, and `NAME_ONLY=1 dm_arm_setup <arm>` gives the run name, config, delta mode and trainer arguments the
release documents (the arms.sh header, docs/SETUP.md's run-name table). The arguments are then parsed by the trainer's
own config loader against the arm's yaml, so what is checked is the model the launcher would really train: the
attention core of every layer, the recurrent layers, the geometry, dm_mode, the fp8 scope, and the NoPE / Muon flags.
A clean / fp8 swap, a wrong delta mode, a typo in an argument or a renamed run fails here without a GPU."""
import re
import os
import subprocess

import pytest

from _helpers import REPO, dry_env

ARMS_SH = REPO / "scripts" / "campaign" / "arms.sh"
SEED = "11785"
# arm -> (run name at seed 11785, config, attention mixer, recurrent mixer (None = pure attention), dm_mode (None = no
#         dm core), fp8_scope, attn_impl (None = not used by the core), (n_heads, n_kv_heads, head_dim))
G128, MLA128, Q3M = (16, 4, 128), (16, 16, 128), (16, 8, 128)
XS256, M256, L256 = (6, 2, 256), (8, 2, 256), (16, 4, 256)
ARMS = {
    "clean_hd128": ("Q35M_clean_sdpa_ffn8_hd128_11785", "qwen35_m.yaml", "gqa", "gated_deltanet", None, "ffn", "sdpa", G128),
    "dm_stale_hd128": ("Q35M_dm_stale_ffn8_qkvo8_hd128_11785", "qwen35_m.yaml", "dm", "gated_deltanet", "stale", "ffn_qkvo", None, G128),
    "dm_matched_hd128": ("Q35M_dm_matched_ffn8_qkvo8_hd128_11785", "qwen35_m.yaml", "dm", "gated_deltanet", "matched", "ffn_qkvo", None, G128),
    "dm_consistent_hd128": ("Q35M_dm_consistent_ffn8_qkvo8_hd128_11785", "qwen35_m.yaml", "dm", "gated_deltanet", "stale", "ffn_qkvo", None, G128),
    "te_fp8": ("Q35M_te_fp8_ffn8_qkvo8_hd128_11785", "qwen35_m.yaml", "gqa", "gated_deltanet", None, "ffn_qkvo", "te_fp8", G128),
    "nope_clean_hd128": ("Q35M_clean_sdpa_ffn8_hd128_11785_nope", "qwen35_m.yaml", "gqa", "gated_deltanet", None, "ffn", "sdpa", G128),
    "nope_dm_stale_hd128": ("Q35M_dm_stale_ffn8_qkvo8_hd128_11785_nope", "qwen35_m.yaml", "dm", "gated_deltanet", "stale", "ffn_qkvo", None, G128),
    "nope_dm_matched_hd128": ("Q35M_dm_matched_ffn8_qkvo8_hd128_11785_nope", "qwen35_m.yaml", "dm", "gated_deltanet", "matched", "ffn_qkvo", None, G128),
    "muon_clean_hd128": ("Q35M_clean_sdpa_ffn8_hd128_11785_muon", "qwen35_m.yaml", "gqa", "gated_deltanet", None, "ffn", "sdpa", G128),
    "muon_dm_stale_hd128": ("Q35M_dm_stale_ffn8_qkvo8_hd128_11785_muon", "qwen35_m.yaml", "dm", "gated_deltanet", "stale", "ffn_qkvo", None, G128),
    "muon_dm_matched_hd128": ("Q35M_dm_matched_ffn8_qkvo8_hd128_11785_muon", "qwen35_m.yaml", "dm", "gated_deltanet", "matched", "ffn_qkvo", None, G128),
    "xs_clean_hd256": ("Q35XS_clean_fa3_ffn8_11785", "qwen35_xs.yaml", "gqa", "gated_deltanet", None, "ffn", "flash", XS256),
    "xs_dm_stale_hd256": ("Q35XS_dm_stale_ffn8_qkvo8_11785", "qwen35_xs.yaml", "dm", "gated_deltanet", "stale", "ffn_qkvo", None, XS256),
    "xs_dm_matched_hd256": ("Q35XS_dm_matched_ffn8_qkvo8_11785", "qwen35_xs.yaml", "dm", "gated_deltanet", "matched", "ffn_qkvo", None, XS256),
    "clean_hd256": ("Q35M_clean_fa3_ffn8_11785", "qwen35_m.yaml", "gqa", "gated_deltanet", None, "ffn", "flash", M256),
    "dm_stale_hd256": ("Q35M_dm_stale_ffn8_qkvo8_11785", "qwen35_m.yaml", "dm", "gated_deltanet", "stale", "ffn_qkvo", None, M256),
    "dm_matched_hd256": ("Q35M_dm_matched_ffn8_qkvo8_11785", "qwen35_m.yaml", "dm", "gated_deltanet", "matched", "ffn_qkvo", None, M256),
    "l_clean_hd256": ("Q35L_clean_fa3_ffn8_11785", "qwen35_l.yaml", "gqa", "gated_deltanet", None, "ffn", "flash", L256),
    "l_dm_stale_hd256": ("Q35L_dm_stale_ffn8_qkvo8_11785", "qwen35_l.yaml", "dm", "gated_deltanet", "stale", "ffn_qkvo", None, L256),
    "l_dm_matched_hd256": ("Q35L_dm_matched_ffn8_qkvo8_11785", "qwen35_l.yaml", "dm", "gated_deltanet", "matched", "ffn_qkvo", None, L256),
    "mla_clean_hd128": ("Q35MLA_clean_sdpa_ffn8_hd128_11785", "qwen35_m_mla.yaml", "mla", "gated_deltanet", None, "ffn", "sdpa", MLA128),
    "mla_dm_stale_hd128": ("Q35MLA_dm_stale_ffn8_qkvo8_hd128_11785", "qwen35_m_mla.yaml", "mla_dm", "gated_deltanet", "stale", "ffn_qkvo", None, MLA128),
    "mla_dm_matched_hd128": ("Q35MLA_dm_matched_ffn8_qkvo8_hd128_11785", "qwen35_m_mla.yaml", "mla_dm", "gated_deltanet", "matched", "ffn_qkvo", None, MLA128),
    "mamba2_clean_hd128": ("Q35M2_clean_sdpa_ffn8_hd128_11785", "qwen35_m_mamba2.yaml", "gqa", "mamba2", None, "ffn", "sdpa", G128),
    "mamba2_dm_stale_hd128": ("Q35M2_dm_stale_ffn8_qkvo8_hd128_11785", "qwen35_m_mamba2.yaml", "dm", "mamba2", "stale", "ffn_qkvo", None, G128),
    "mamba2_dm_matched_hd128": ("Q35M2_dm_matched_ffn8_qkvo8_hd128_11785", "qwen35_m_mamba2.yaml", "dm", "mamba2", "matched", "ffn_qkvo", None, G128),
    "kda_clean_hd128": ("Q35KDA_clean_sdpa_ffn8_hd128_11785", "qwen35_m_kda.yaml", "gqa", "kda", None, "ffn", "sdpa", G128),
    "kda_dm_stale_hd128": ("Q35KDA_dm_stale_ffn8_qkvo8_hd128_11785", "qwen35_m_kda.yaml", "dm", "kda", "stale", "ffn_qkvo", None, G128),
    "kda_dm_matched_hd128": ("Q35KDA_dm_matched_ffn8_qkvo8_hd128_11785", "qwen35_m_kda.yaml", "dm", "kda", "matched", "ffn_qkvo", None, G128),
    "gqa_clean_hd128": ("Q3M_clean_sdpa_ffn8_hd128_11785", "qwen3_m.yaml", "gqa", None, None, "ffn", "sdpa", Q3M),
    "gqa_dm_stale_hd128": ("Q3M_dm_stale_ffn8_qkvo8_hd128_11785", "qwen3_m.yaml", "dm", None, "stale", "ffn_qkvo", None, Q3M),
    "gqa_dm_matched_hd128": ("Q3M_dm_matched_ffn8_qkvo8_hd128_11785", "qwen3_m.yaml", "dm", None, "matched", "ffn_qkvo", None, Q3M),
}
MODE = {None: "clean", "te_fp8": "te"}   # arms.sh MODE of the arms without a dm core, by attn_impl
RECURRENT = {"gated_deltanet", "mamba2", "kda"}

SETUP_SCRIPT = r'''
source scripts/campaign/env.sh; source scripts/campaign/arms.sh
for a in "$@"; do
  if NAME_ONLY=1 dm_arm_setup "$a" 11785; then
    echo "@@ARM $a"; echo "RN=$RN"; echo "CFG=$CFG"; echo "MODE=$MODE"; for x in "${ARGS[@]}"; do echo "ARG=$x"; done
  else
    echo "@@FAIL $a"
  fi
done
echo "@@EXPORTED $(env | grep -cE '^(DM_SDPA_BACKEND|DM_FLASH_IMPL|DM_SAC_SAVE_FA3WHEEL|DM_SHARED_X8|DM_STALE_DOQ2S|DM_DELTA_Q8)=')"
'''


def _bash(script, args, tmp_path, **extra):
    return subprocess.run(["bash", "-c", script, "arms", *args], cwd=str(REPO), capture_output=True, text=True,
                          env=dry_env(tmp_path, **extra), timeout=120)


def _listed_arms(tmp_path):
    """The arms `dm_arms` lists (arms.sh's DM_ARMS)."""
    p = _bash("source scripts/campaign/env.sh; source scripts/campaign/arms.sh; dm_arms", [], tmp_path)
    assert p.returncode == 0, p.stderr[-2000:]
    return p.stdout.split()


def _header_words():
    """The words of the arms.sh header comment, brace forms expanded (mla_{clean,dm_stale}_hd128 -> mla_clean_hd128 ...)."""
    head = []
    for line in ARMS_SH.read_text().splitlines()[1:]:
        if not line.startswith("#"):
            break
        head.append(line)
    text = "\n".join(head)
    words = set(re.findall(r"[A-Za-z0-9_]+", text))
    for pre, alts, post in re.findall(r"(\w*)\{([\w,]+)\}(\w*)", text):
        words |= {pre + a + post for a in alts.split(",")}
    return words


@pytest.fixture(scope="module")
def setups(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("arms")
    p = _bash(SETUP_SCRIPT, sorted(ARMS), tmp)
    assert p.returncode == 0, p.stderr[-2000:]
    out, cur = {}, None
    for line in p.stdout.splitlines():
        if line.startswith("@@ARM "):
            cur = out.setdefault(line.split()[1], {"ARGS": []})
        elif line.startswith("@@FAIL "):
            out[line.split()[1]] = None; cur = None
        elif line.startswith("@@EXPORTED "):
            out["@@exported"] = int(line.split()[1])
        elif cur is not None and line.startswith("ARG="):
            cur["ARGS"].append(line[4:])
        elif cur is not None and "=" in line:
            k, v = line.split("=", 1); cur[k] = v
    return out


def test_every_arm_is_in_the_header_and_the_table(tmp_path):
    arms = _listed_arms(tmp_path)
    assert len(arms) == len(set(arms)) == 32, arms
    missing_header = sorted(set(arms) - _header_words())
    assert not missing_header, f"arms listed by dm_arms but missing from the arms.sh header: {missing_header}"
    assert set(arms) == set(ARMS), f"dm_arms vs this test's table: only in arms.sh {sorted(set(arms) - set(ARMS))}, only here {sorted(set(ARMS) - set(arms))}"


def test_name_only_exports_nothing(setups):
    """NAME_ONLY=1 (the experiment drivers' run-name lookup) sets RN and the arguments only: no per-arm environment."""
    assert setups.get("@@exported") == 0, setups.get("@@exported")


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_arm_setup(arm, setups):
    rn, cfg, attn, recur, mode, scope, impl, (nh, nkv, hd) = ARMS[arm]
    s = setups.get(arm)
    assert s, f"NAME_ONLY=1 dm_arm_setup {arm} failed"
    assert s["RN"] == rn, (s["RN"], rn)
    assert s["CFG"] == f"src/pretrain/configs/{cfg}", s["CFG"]
    assert s["MODE"] == (mode or MODE.get(impl, "clean")), s["MODE"]
    from pretrain.config import load_config
    c = load_config(["--config", str(REPO / s["CFG"])] + s["ARGS"])
    m = c.model
    kinds = [m.mixer_for_layer(i) for i in range(m.n_layers)]
    assert {k for k in kinds if k not in RECURRENT} == {attn}, f"{arm}: attention layers {sorted(set(kinds))}, expected {attn}"
    assert {k for k in kinds if k in RECURRENT} == ({recur} if recur else set()), f"{arm}: recurrent layers {sorted(set(kinds))}"
    if recur:
        assert kinds.count(attn) * 4 == m.n_layers, f"{arm}: not the 3:1 pattern: {kinds}"
    assert (m.n_heads, m.n_kv_heads, m.head_dim) == (nh, nkv, hd), (m.n_heads, m.n_kv_heads, m.head_dim)
    assert c.fp8 is True and c.fp8_scope == scope, (c.fp8, c.fp8_scope)
    if mode is not None:
        assert m.dm_mode == mode, m.dm_mode
    if attn == "dm":
        assert m.dm_fused_producers is True, f"{arm}: the dm mixer runs the fused-producer route only"
    if impl is not None:
        assert m.attn_impl == impl, m.attn_impl
    assert bool(getattr(m, "nope", False)) is arm.startswith("nope_"), f"{arm}: nope={getattr(m, 'nope', None)}"
    assert (c.optim.name == "muon") is arm.startswith("muon_"), f"{arm}: optim.name={c.optim.name}"


def test_run_name_helper_and_tag(tmp_path):
    """dm_run_name: the seed, the TAG suffix, and an unknown arm (no name, non-zero status)."""
    script = 'source scripts/campaign/env.sh; source scripts/campaign/arms.sh; dm_run_name "$1" "$2"'
    p = _bash(script, ["dm_matched_hd128", "2"], tmp_path)
    assert p.returncode == 0 and p.stdout.strip() == "Q35M_dm_matched_ffn8_qkvo8_hd128_2", p.stdout + p.stderr
    p = _bash(script, ["mla_clean_hd128", "2"], tmp_path, TAG="_nope")
    assert p.returncode == 0 and p.stdout.strip() == "Q35MLA_clean_sdpa_ffn8_hd128_2_nope", p.stdout + p.stderr
    p = _bash(script, ["no_such_arm", "2"], tmp_path)
    assert p.returncode != 0 and p.stdout.strip() == "", p.stdout + p.stderr


def test_setup_doc_run_names(setups):
    """docs/SETUP.md's run-name table agrees with arms.sh (`<seed>` = the seed, `...` = any infix)."""
    text = (REPO / "docs" / "SETUP.md").read_text()
    seen = set()
    for row in text.splitlines():
        if not row.startswith("|"):
            continue
        ticks = re.findall(r"`([^`]+)`", row)
        names = [t for t in ticks if t.startswith("Q3") and t.endswith("<seed>")]
        arms = [a.strip() for t in ticks for a in t.split("/") if a.strip() in ARMS]
        if not names or not arms:
            continue
        pat = re.escape(names[-1]).replace(re.escape("<seed>"), SEED).replace(re.escape("..."), ".*")
        for a in arms:
            assert re.fullmatch(pat, setups[a]["RN"]), f"docs/SETUP.md says {names[-1]} for {a}, arms.sh gives {setups[a]['RN']}"
            seen.add(a)
    need = {"clean_hd128", "dm_matched_hd128", "dm_stale_hd128", "dm_consistent_hd128", "te_fp8"}
    assert need <= seen, f"docs/SETUP.md's run-name table lacks {sorted(need - seen)}"


def test_configs_load_and_a_typo_fails():
    """Every shipped yaml and every tiny test yaml loads under the trainer's strict config loader, and a misspelled key
    fails at load instead of training the default (a stale arm launched with --model.dm_mdoe stale would run matched)."""
    from pretrain.config import load_config
    yamls = sorted((REPO / "src" / "pretrain" / "configs").glob("*.yaml")) + sorted((REPO / "tests" / "configs").glob("tiny_*.yaml"))
    assert len(yamls) >= 11, yamls
    for f in yamls:
        load_config(["--config", str(f)])
    with pytest.raises(ValueError, match=r"model\.dm_mdoe \(did you mean model\.dm_mode\?\)"):
        load_config(["--config", str(REPO / "tests" / "configs" / "tiny_hybrid.yaml"), "--model.dm_mdoe", "stale"])


@pytest.mark.parametrize("mib", ["81559", "143771", None])
def test_small_gpu_micro_batch(tmp_path, mib):
    """The MLA and KDA arms add --per_device_batch_size 2 below 100 GB per GPU (their yamls' micro-batch of 4 needs more
    than 80 GB); a larger GPU, or no answer from nvidia-smi, keeps the yaml's value, and the other arms never change."""
    fake = tmp_path / "bin"
    fake.mkdir()
    body = f"printf '{mib}\\n{mib}\\n'" if mib else "exit 9"
    (fake / "nvidia-smi").write_text(f"#!/bin/bash\n{body}\n")
    (fake / "nvidia-smi").chmod(0o755)
    for arm in ("mla_clean_hd128", "kda_clean_hd128", "mamba2_clean_hd128", "clean_hd128"):
        p = _bash('source scripts/campaign/env.sh; source scripts/campaign/arms.sh; dm_arm_setup "$1" 11785 && echo "ARGS ${ARGS[*]}"',
                  [arm], tmp_path, PATH=f"{fake}:{os.environ['PATH']}")
        assert p.returncode == 0, (arm, p.stderr[-1500:])
        want = mib == "81559" and arm.startswith(("mla_", "kda_"))
        assert ("--per_device_batch_size 2" in p.stdout) is want, (arm, mib, p.stdout[-800:], p.stderr[-800:])

