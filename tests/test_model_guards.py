"""Model-build guards and small model-level contracts, CPU only (no GPU, no kernel library): the dm mixer refuses the
configurations it cannot run before it loads the kernel, the bf16 MLA mixer refuses an fp8 attn_impl instead of silently
running SDPA, the trainer refuses --backend ddp with a Delta-Matching route, the HF wrapper's logits_to_keep keeps exactly
the requested positions, and the analytic parameter count equals the live model where it has a closed form (None on the
recurrent hybrids)."""
import sys

import pytest
import torch
import yaml

from _helpers import REPO

CONFIGS = REPO / "src" / "pretrain" / "configs"


def _cfg(**over):
    from pretrain.modeling.config import ModelConfig
    base = dict(vocab_size=32000, d_model=256, n_layers=4, n_heads=4, n_kv_heads=1, head_dim=128, max_seq_len=512,
                layer_mixers=["gated_deltanet"] * 3 + ["dm"], intermediate_size=512, gdn_head_dim=64, gdn_num_heads=4,
                attn_output_gate=True, partial_rotary_factor=0.25, qk_norm=True, attn_impl="sdpa", dm_mode="matched")
    base.update(over)
    return ModelConfig(**base)


YARN4 = {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 128, "beta_fast": 32, "beta_slow": 1}


@pytest.mark.parametrize("label,over,msg", [
    ("bf16_dp at head_dim 256", dict(head_dim=256, dm_mode="bf16_dp"), "bf16_dp runs at head_dim 128 only"),
    ("YaRN at head_dim 256", dict(head_dim=256, rope_scaling=YARN4), "YaRN temperature at head_dim 128 only"),
    ("head_dim 64", dict(head_dim=64), "head_dim 128 or 256"),
    ("the plain-kernel route", dict(dm_fused_producers=False), "fused-producer route only"),
    ("fused RoPE", dict(partial_rotary_factor=1.0, fused_rope=True), "fused_rope must be off"),
])
def test_dm_mixer_refuses_before_loading_the_kernel(label, over, msg, monkeypatch):
    monkeypatch.delenv("DM_MODE_LAYERS", raising=False)
    monkeypatch.setitem(sys.modules, "delta_matching", None)   # the kernel package unimportable: a guard after the load fails here
    from pretrain.modeling.mixers import build_mixer
    with pytest.raises(ValueError, match=msg):
        build_mixer("dm", _cfg(**over), 3)


@pytest.mark.parametrize("impl", ["dm", "te_fp8"])
def test_bf16_mla_refuses_an_fp8_attn_impl(impl):
    """The mla mixer runs a bf16 core; the fp8 MLA core is the mixer mla_dm (never a silent SDPA fallback)."""
    from pretrain.modeling.mixers import build_mixer
    cfg = _cfg(n_kv_heads=4, kv_lora_rank=96, partial_rotary_factor=0.5, layer_mixers=["gated_deltanet"] * 3 + ["mla"],
               attn_impl=impl)
    with pytest.raises(ValueError, match="bf16 core"):
        build_mixer("mla", cfg, 3)


DM_LAYERS = "[gated_deltanet, gated_deltanet, gated_deltanet, dm]"


@pytest.mark.parametrize("config,extra,refused", [
    ("tests/configs/tiny_hybrid.yaml", ["--backend", "ddp", "--model.layer_mixers", DM_LAYERS], True),
    ("tests/configs/tiny_mla.yaml", ["--backend", "ddp", "--model.layer_mixers", "[gated_deltanet, gated_deltanet, gated_deltanet, mla_dm]"], True),
    ("tests/configs/tiny_hybrid.yaml", ["--backend", "ddp", "--model.attn_impl", "dm"], True),
    ("src/pretrain/configs/qwen3_m.yaml", ["--backend", "ddp", "--model.mixer", "dm"], True),
    ("tests/configs/tiny_hybrid.yaml", ["--backend", "ddp"], False),
    ("tests/configs/tiny_mla.yaml", ["--backend", "ddp"], False),
    ("tests/configs/tiny_hybrid.yaml", ["--model.layer_mixers", DM_LAYERS], False),
    ("tests/configs/tiny_hybrid.yaml", ["--model.attn_impl", "dm"], False),
])
def test_ddp_refuses_the_delta_matching_route(config, extra, refused, monkeypatch):
    """--backend ddp keeps fp32 parameters (autocast only), so the Delta-Matching routes (a dm or mla_dm layer, or
    attn_impl dm) would get fp32 activations: the trainer refuses the combination at startup, naming --backend fsdp2,
    and lets every clean ddp run and every fsdp2 run (the shipped configs' backend) through."""
    monkeypatch.setenv("WORLD_SIZE", "1")
    from pretrain.config import load_config
    from pretrain.train import _refuse_ddp_with_delta_matching
    c = load_config(["--config", str(REPO / config)] + extra)
    if refused:
        with pytest.raises(ValueError, match="--backend fsdp2"):
            _refuse_ddp_with_delta_matching(c)
    else:
        _refuse_ddp_with_delta_matching(c)


def test_hf_wrapper_logits_to_keep(monkeypatch):
    """DMLMForCausalLM honors transformers' logits_to_keep: k > 0 = the last k positions, 0 or None = all, a 1-D tensor =
    those positions; a negative k, or k together with labels (the loss needs every position), raises."""
    monkeypatch.delenv("DM_SDPA_BACKEND", raising=False)
    from pretrain.modeling.config import ModelConfig
    from pretrain.hf.configuration_dmlm import DMLMConfig
    from pretrain.hf.modeling_dmlm import DMLMForCausalLM
    torch.manual_seed(0)
    mc = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=32, max_seq_len=64,
                     mixer="gqa", attn_impl="sdpa", intermediate_size=128)
    m = DMLMForCausalLM(DMLMConfig.from_model_config(mc)).eval()
    ids = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        full = m(input_ids=ids).logits
        assert full.shape == (2, 16, 256)
        for k in (1, 5):
            got = m(input_ids=ids, logits_to_keep=k).logits
            assert got.shape == (2, k, 256) and torch.allclose(got, full[:, -k:], atol=1e-6, rtol=0), k
        for keep_all in (0, None):
            assert torch.allclose(m(input_ids=ids, logits_to_keep=keep_all).logits, full, atol=1e-6, rtol=0), keep_all
        idx = torch.tensor([0, 3, 7])
        assert torch.allclose(m(input_ids=ids, logits_to_keep=idx).logits, full[:, idx], atol=1e-6, rtol=0)
        with pytest.raises(ValueError):
            m(input_ids=ids, logits_to_keep=-1)
        with pytest.raises(ValueError):
            m(input_ids=ids, labels=ids, logits_to_keep=2)
        assert m(input_ids=ids, labels=ids).loss is not None


def _live_params(mc):
    from pretrain.modeling.model import PretrainLM
    with torch.device("meta"):
        m = PretrainLM(mc)
    return sum(p.numel() for p in {id(p): p for p in m.parameters()}.values())


def test_analytic_parameter_count():
    from pretrain.modeling.config import ModelConfig
    for name in ("qwen3_m",):
        mc = ModelConfig(**yaml.safe_load(open(CONFIGS / f"{name}.yaml"))["model"])
        assert mc.num_params_analytic() == _live_params(mc), name
    for over in (dict(layer_mixers=None, mixer="mla", n_kv_heads=4, kv_lora_rank=96, partial_rotary_factor=0.5),
                 dict(layer_mixers=["gqa", "mla"], n_kv_heads=4, kv_lora_rank=96, partial_rotary_factor=0.5)):
        mc = _cfg(**over)
        assert mc.num_params_analytic() == _live_params(mc), over
    for name in ("qwen35_m", "qwen35_m_mla", "qwen35_m_mamba2", "qwen35_m_kda"):
        mc = ModelConfig(**yaml.safe_load(open(CONFIGS / f"{name}.yaml"))["model"])
        assert mc.num_params_analytic() is None, f"{name}: the recurrent layers have no closed form"
