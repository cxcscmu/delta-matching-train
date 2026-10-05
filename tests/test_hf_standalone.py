"""The HF remote-code conversion (scripts/hf; HF_CONVERSION.md). CPU: make_standalone.py refuses what it cannot ship (an
export still carrying the training-only dm / mla_dm cores or an fp8 attn_impl, i.e. not patched by train_arm.sh, and a
config that is not a dmlm model) before writing anything, and verify_standalone.py refuses a foreign config. GPU: the
tiny clean export converts to a standalone directory that stock transformers loads with trust_remote_code, and
verify_standalone.py finds it bitwise equal to this repository's loader (logits, loss, greedy decoding, generate())."""
import json
import subprocess

import pytest
import torch

from _helpers import REPO, base_env, clean_env, python

HF = REPO / "scripts" / "hf"
GDN = ["gated_deltanet"] * 3


def _fake_export(tmp_path, config):
    d = tmp_path / "export"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(config))
    (d / "model.safetensors").write_bytes(b"")
    return d


@pytest.mark.parametrize("label,config,msg", [
    ("dm", {"model_type": "dmlm", "attn_impl": "sdpa", "mixer": "gqa", "layer_mixers": GDN + ["dm"]}, "not in the shipped model code"),
    ("mla_dm", {"model_type": "dmlm", "attn_impl": "sdpa", "mixer": "mla", "layer_mixers": GDN + ["mla_dm"]}, "not in the shipped model code"),
    ("pure_dm", {"model_type": "dmlm", "attn_impl": "sdpa", "mixer": "dm", "layer_mixers": None}, "not in the shipped model code"),
    ("te_fp8", {"model_type": "dmlm", "attn_impl": "te_fp8", "mixer": "gqa", "layer_mixers": GDN + ["gqa"]}, "not patched"),
    ("foreign", {"model_type": "llama", "attn_impl": "sdpa"}, "was not written by this repository"),
])
def test_make_standalone_refuses(label, config, msg, tmp_path):
    export, out = _fake_export(tmp_path, config), tmp_path / "standalone"
    p = subprocess.run([python(), str(HF / "make_standalone.py"), str(export), "--out", str(out)], capture_output=True,
                       text=True, env=clean_env(), timeout=300)
    assert p.returncode != 0 and msg in p.stderr, f"{label}: rc={p.returncode}\n{p.stdout[-1000:]}\n{p.stderr[-1500:]}"
    assert not out.exists() or not any(out.iterdir()), f"{label}: a refused export left files in {out}"


def test_make_standalone_builds_on_cpu(tok_dir, tmp_path):
    """A tiny GQA export (this repository's exporter, with the campaign tokenizer) converts without a GPU: the published
    config.json names the shipped code, carries the tokenizer's bos / eos ids and drops the training-only keys, and the
    weights, tokenizer, code and card are all there."""
    from transformers import AutoTokenizer
    from pretrain.hf.export import save_hf_checkpoint
    from pretrain.modeling import build_model
    from pretrain.modeling.config import ModelConfig
    torch.manual_seed(0)
    mc = ModelConfig(vocab_size=32000, d_model=64, n_layers=2, n_heads=2, n_kv_heads=1, head_dim=32, max_seq_len=64,
                     mixer="gqa", attn_impl="sdpa", intermediate_size=128, dm_mode="stale")
    export, out = tmp_path / "export", tmp_path / "standalone"
    save_hf_checkpoint(build_model(mc).state_dict(), mc, str(export), tokenizer=AutoTokenizer.from_pretrained(str(tok_dir)))
    p = subprocess.run([python(), str(HF / "make_standalone.py"), str(export), "--out", str(out)], capture_output=True,
                       text=True, env=clean_env(), timeout=300)
    assert p.returncode == 0, f"rc={p.returncode}\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}"
    c = json.loads((out / "config.json").read_text())
    assert c["architectures"] == ["DMLMForCausalLM"] and c["auto_map"]["AutoConfig"] == "configuration_dmlm.DMLMConfig", c
    assert (c.get("bos_token_id"), c.get("eos_token_id")) == (1, 2) and "dm_mode" not in c, c
    for f in ("model.safetensors", "modeling_dmlm.py", "configuration_dmlm.py", "README.md", "tokenizer.json"):
        assert (out / f).exists(), f"{f} missing from the standalone directory"


def test_verify_standalone_refuses_a_foreign_config(tmp_path):
    export = _fake_export(tmp_path, {"model_type": "llama"})
    p = subprocess.run([python(), str(HF / "verify_standalone.py"), "--export", str(export), "--standalone", str(export),
                        "--windows", "1", "--val", "random"], capture_output=True, text=True, env=clean_env(), timeout=300)
    assert p.returncode != 0 and "was not written by this repository" in p.stderr, f"rc={p.returncode}\n{p.stderr[-1500:]}"


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.data
def test_standalone_conversion_round_trip(clean_export, test_gpu, tmp_path):
    out = tmp_path / "standalone"
    env = base_env(test_gpu)
    env["HF_MODULES_CACHE"] = str(tmp_path / "hf_modules")   # trust_remote_code copies the model code here
    p = subprocess.run([python(), str(HF / "make_standalone.py"), str(clean_export), "--out", str(out)], capture_output=True,
                       text=True, env=env, timeout=600)
    assert p.returncode == 0, f"make_standalone failed rc={p.returncode}\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}"
    c = json.loads((out / "config.json").read_text())
    assert c["architectures"] == ["DMLMForCausalLM"] and c["auto_map"]["AutoModelForCausalLM"] == "modeling_dmlm.DMLMForCausalLM", c
    assert c.get("eos_token_id") == 2 and "dm_mode" not in c, c
    for f in ("model.safetensors", "modeling_dmlm.py", "configuration_dmlm.py", "README.md", "tokenizer.json"):
        assert (out / f).exists(), f"{f} missing from the standalone directory"
    assert not (out / "TRAIN_COMPLETE").exists()
    p = subprocess.run([python(), str(HF / "verify_standalone.py"), "--export", str(clean_export), "--standalone", str(out),
                        "--windows", "1", "--val", "random"], capture_output=True, text=True, env=env, timeout=1800)
    assert p.returncode == 0 and p.stdout.strip().endswith("PASS"), f"verify_standalone rc={p.returncode}\n{p.stdout[-3000:]}\n{p.stderr[-1500:]}"
