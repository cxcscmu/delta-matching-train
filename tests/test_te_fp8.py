"""The te_fp8 arm's guards (Transformer Engine's cuDNN fp8 fused attention, modeling/te_attn.py), CPU only: the core
refuses head_dim 256 (cuDNN has no sm90 fp8 training path there) before touching Transformer Engine; without Transformer
Engine the mixer build fails with the documented message (never a silent fallback to another core); and the launcher
refuses the arm without its separate TE environment (DM_TE_PY / DM_TE_TORCHRUN)."""
import subprocess
import sys

import pytest

from _helpers import REPO, dry_env, fake_data_root


def _cfg(head_dim):
    from pretrain.modeling.config import ModelConfig
    return ModelConfig(vocab_size=32000, d_model=256, n_layers=4, n_heads=4, n_kv_heads=1, head_dim=head_dim, max_seq_len=512,
                       layer_mixers=["gated_deltanet"] * 3 + ["gqa"], intermediate_size=512, gdn_head_dim=64, gdn_num_heads=4,
                       attn_output_gate=True, partial_rotary_factor=0.25, qk_norm=True, attn_impl="te_fp8")


@pytest.fixture
def no_te(monkeypatch):
    """Transformer Engine absent, whatever this environment has: `import transformer_engine` raises ImportError, the
    route's import cache is empty, and the NVTE_* variables the route sets are restored afterwards."""
    from pretrain.modeling import te_attn
    for k, v in te_attn._NVTE_DEFAULTS.items():   # the route's own values: its setdefault then changes nothing
        monkeypatch.setenv(k, v)
    for mod in ("transformer_engine", "transformer_engine.pytorch", "transformer_engine.common", "transformer_engine.common.recipe"):
        monkeypatch.setitem(sys.modules, mod, None)
    monkeypatch.setitem(te_attn._S, "te", None)
    monkeypatch.setitem(te_attn._S, "core_cls", None)
    monkeypatch.setitem(te_attn._S, "recipe", None)


def test_te_fp8_refuses_head_dim_256(no_te):
    from pretrain.modeling.mixers import build_mixer
    with pytest.raises(ValueError, match="head_dim-128-only"):
        build_mixer("gqa", _cfg(256), 3)


def test_te_fp8_without_transformer_engine_fails_with_the_documented_message(no_te):
    from pretrain.modeling.mixers import build_mixer
    with pytest.raises(RuntimeError, match="needs NVIDIA Transformer Engine") as e:
        build_mixer("gqa", _cfg(128), 3)
    assert "DM_TE_PY" in str(e.value) and "attn_impl='sdpa'" in str(e.value), str(e.value)


def test_train_arm_refuses_te_fp8_without_the_te_environment(tmp_path):
    """train_arm.sh te_fp8 needs DM_TE_TORCHRUN and DM_TE_PY (the TE environment of docs/SETUP.md section 2): without them
    it stops before the trainer starts (a fake data root and tokenizer get it past the earlier checks)."""
    fake_data_root(tmp_path)
    env = dry_env(tmp_path, DM_WANDB=0, DM_LAUNCH="env", DM_NPROC=1)
    p = subprocess.run(["bash", "scripts/campaign/train_arm.sh", "te_fp8", "11785"], cwd=str(REPO), env=env,
                       capture_output=True, text=True, timeout=120)
    out = p.stdout + p.stderr
    assert p.returncode != 0 and "DM_TE_TORCHRUN" in out, f"rc={p.returncode}\n{out[-1500:]}"
    assert "LAUNCH" not in out and not list((tmp_path / "logs").glob("*.rank0.log")), f"the trainer was launched:\n{out[-1500:]}"
