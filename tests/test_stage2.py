"""Stage 2 context extension (scripts/campaign/stage2_arm.sh, scripts/experiments/stage2_context.sh): the pieces that
stage 2 adds on top of stage 1.
1. The fused fp8 mixer under YaRN: the temperature (mscale^2) folded into the pre-RoPE q must reproduce the bf16 mixer
   with the same weights under the same YaRN, and must NOT reproduce a no-temperature control (so a missing fold fails).
2. A tiny warm start (init_from_hf) from a stage-1 export at 2048 into 8192 with YaRN factor 4, for the clean and the
   matched fp8 arms, with gradient checkpointing; the export carries rope_scaling and the new max_seq_len and scores
   through the val CE scorer at 8192.
3. The stage-2 launcher (dry, no GPU): it refuses to start without the warm-start export, runs only the four arms of
   the stage-2 experiment, and composes the paper's recipe exactly (28,000 steps, a 100-step re-warm and a 2,000-step
   decay after the trainer's truncation of the ratios, 8e-4, chunk_subset [0, 0.85], an export every 2,000 steps, the
   Standard arm's stage-1 export of the same seed as the warm start).
4. The stage-2 memory lever: the fused fp8 backward split per KV-head group or per query head is bitwise the single call."""
import json, math, os, re, subprocess
import pytest
from _helpers import (REPO, NEMO, CE_RE, arm_args, dm_env, dm_route, require_kernel, base_env, memmap_ppl_env, dry_env,
                      python, train_tiny)

YARN4 = '{"type":"yarn","factor":4.0,"original_max_position_embeddings":2048,"beta_fast":32,"beta_slow":1}'


def _plain_kernel_env():
    """dm_env without the fp8 all-gather lever and the cast glue: the standalone mixer tests run the plain-kernel route."""
    env = base_env(os.environ.get("DM_TEST_GPU", "0")); env.update(dm_env(stale=False))
    env.update({"DM_FP8GLUE": "0", "DM_FP8_FSDP_AG": "0", "DM_FP8_PRECOMPUTE": "0"})
    return env


@pytest.mark.gpu
def test_fused_fp8_mixer_yarn_temperature_matches_bf16_mixer():
    """Same weights (the seeded initialization), same YaRN (factor 4): qk-norm leaves q and k at unit RMS (logit std
    about 1), where the temperature (logits x mscale^2 = 1.29) moves the bf16 mixer's output by about 30 % (guarded at
    3 %). The fused fp8 output must sit within the fp8 tolerance of the bf16 mixer with the temperature, and closer to it
    than to the bf16 mixer without the temperature (rope_scaling.mscale = 1.0 keeps the interpolation, drops the fold)."""
    require_kernel()
    code = r'''
import json, torch
from pretrain.modeling.config import ModelConfig
from pretrain.modeling.mixers import build_mixer
from pretrain.modeling.layers import RotaryEmbedding
torch.manual_seed(0); dev = "cuda"; T = 1024
def cfg(mixer, rope_scaling):
    return ModelConfig(vocab_size=32000, d_model=256, n_layers=4, n_heads=4, n_kv_heads=1, head_dim=128, max_seq_len=T,
                       layer_mixers=["gated_deltanet"] * 3 + [mixer], intermediate_size=512, gdn_head_dim=64, gdn_num_heads=4,
                       attn_output_gate=True, partial_rotary_factor=0.25, qk_norm=True, attn_impl="sdpa", rope_theta=1e7,
                       rope_scaling=rope_scaling, dm_mode="matched", dm_fused_producers=(mixer == "dm"))
yarn = {"type": "yarn", "factor": 4.0, "original_max_position_embeddings": T // 4, "beta_fast": 32, "beta_slow": 1}
notemp = dict(yarn, mscale=1.0)
ref = build_mixer("gqa", cfg("gqa", yarn), 3).to(dev, torch.bfloat16)
sd = {k: v.clone() for k, v in ref.state_dict().items()}
ctl = build_mixer("gqa", cfg("gqa", notemp), 3).to(dev, torch.bfloat16); ctl.load_state_dict(sd)
fp8 = build_mixer("dm", cfg("dm", yarn), 3).to(dev, torch.bfloat16); missing, unexpected = fp8.load_state_dict(sd, strict=False)
assert not missing and not unexpected, (missing, unexpected)
import torch.nn as nn
from pretrain.parallel.wrap import apply_fp8
host = nn.Module(); host.model = nn.Module(); host.model.layers = nn.ModuleList([nn.Module()]); host.model.layers[0].mixer = fp8   # the trainer's fqn: model.layers.N.mixer.q_proj
n_conv = apply_fp8(host, "tensorwise", "ffn_qkvo"); assert n_conv >= 2 and type(fp8.q_proj).__name__ == "Float8Linear", (n_conv, type(fp8.q_proj).__name__)
x = (torch.randn(1, T, 256, device=dev) * 1.0).to(torch.bfloat16)
c = cfg("gqa", yarn); rot = RotaryEmbedding(c.head_dim, c.max_seq_len, c.rope_theta, rope_scaling=c.rope_scaling, rotary_dim=c.rotary_dim).to(dev)
cos, sin = rot(torch.arange(T, device=dev)[None]); cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
with torch.no_grad():
    o_ref = ref(x, cos, sin).float(); o_ctl = ctl(x, cos, sin).float(); o_fp8 = fp8(x, cos, sin).float()
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
print(json.dumps({"fp8_vs_ref": rel(o_fp8, o_ref), "fp8_vs_notemp": rel(o_fp8, o_ctl), "ref_vs_notemp": rel(o_ref, o_ctl), "mscale": float(fp8.attn_mscale)}))
'''
    p = subprocess.run([python(), "-c", code], capture_output=True, text=True, env=_plain_kernel_env(), timeout=900)
    assert p.returncode == 0, f"rc={p.returncode}\n{p.stderr[-3000:]}"
    r = json.loads([l for l in p.stdout.splitlines() if l.startswith("{")][-1])
    print(f"YaRN fold (fused dm mixer): {r}")   # shown by pytest -rP
    assert abs(r["mscale"] - (0.1 * math.log(4.0) + 1.0)) < 1e-6, r
    assert r["ref_vs_notemp"] > 0.03, f"the temperature must matter at this logit scale: {r}"
    assert r["fp8_vs_ref"] < 0.10, f"fused fp8 mixer under YaRN off the bf16 mixer: {r}"
    assert r["fp8_vs_ref"] < r["fp8_vs_notemp"], f"the fold is missing on the fused path (closer to the no-temperature control): {r}"


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.data
@pytest.mark.parametrize("arm", ["clean_hd128", "dm_matched_hd128"])
def test_tiny_stage2_warm_start_yarn_8192(arm, run_root, root, tok_dir, test_gpu):
    """Stage 1: three steps of the tiny clean hybrid at 2048, exported. Stage 2: warm start from that export into
    max_seq_len 8192 with YaRN factor 4 and full gradient checkpointing, three steps, for the arm; the export must carry
    the YaRN config and the new length, and the val CE scorer must score it at 8192."""
    if arm != "clean_hd128":
        require_kernel()
    a1, e1 = arm_args("clean_hd128")
    rc, log, hf1, dt = train_tiny("s2_stage1_clean", run_root, root, tok_dir, test_gpu, extra_args=a1 + ["--gradient_checkpointing", "false"], extra_env=e1)
    assert rc == 0 and (hf1 / "model.safetensors").exists(), f"stage 1 failed rc={rc}\n{log[-3000:]}"
    a2, e2 = arm_args(arm)
    s2 = a2 + ["--init_from_hf", str(hf1), "--model.max_seq_len", "8192", "--model.rope_scaling", YARN4,
               "--gradient_checkpointing", "true", "--data.chunk_subset", "[0.0,0.5]"]
    rc, log, hf2, dt = train_tiny(f"s2_{arm}", run_root, root, tok_dir, test_gpu, extra_args=s2, extra_env=e2)
    assert rc == 0, f"{arm}: stage 2 exit {rc} after {dt:.0f}s\n{log[-4000:]}"
    assert "warm-start: loaded" in log, f"{arm}: no warm-start line\n{log[-2500:]}"
    route = dm_route(log)
    if arm == "clean_hd128":
        assert "dm" not in route, f"{arm}: the clean arm built the dm mixer: {route}"
    else:
        assert route.get("dm") == "matched", f"{arm}: the [dm] DMAttention banner says {route}, expected dm_mode=matched"
    ces = [float(x) for x in CE_RE.findall(log)]
    assert len(ces) >= 3 and all(math.isfinite(c) for c in ces), f"{arm}: losses {ces}"
    c = json.loads((hf2 / "config.json").read_text())
    rs = c.get("rope_scaling") or c.get("rope_parameters") or {}   # transformers 5 serializes rope_scaling as rope_parameters
    assert str(rs.get("type", rs.get("rope_type", ""))).lower() == "yarn" and float(rs.get("factor", 0)) == 4.0, f"{arm}: export rope_scaling {rs}"
    assert int(c.get("max_seq_len", 0)) == 8192, f"{arm}: export max_seq_len {c.get('max_seq_len')}"
    # the launcher's export patch (scripts/campaign/train_arm.sh): every arm is scored through the common clean kernel
    c["attn_impl"] = "sdpa"
    if c.get("mixer") == "dm": c["mixer"] = "gqa"
    if c.get("layer_mixers"): c["layer_mixers"] = [{"dm": "gqa", "mla_dm": "mla"}.get(m, m) for m in c["layer_mixers"]]
    for k in ("dm_mode", "dm_fused_producers"): c.pop(k, None)
    (hf2 / "config.json").write_text(json.dumps(c, indent=2, sort_keys=True))
    out = run_root / "ppl" / arm; out.mkdir(parents=True, exist_ok=True)
    val = root / NEMO / "nemotron_rs_val.bin"
    cmd = [python(), "-m", "evaluation.ppl.memmap_ppl", "--model_name_or_path", str(hf2), "--data", str(val), "--dtype", "uint16",
           "--seq_len", "8192", "--max_windows", "2", "--batch_size", "1", "--output_dir", str(out), "--run_name", f"s2_{arm}_n2"]
    p = subprocess.run(cmd, cwd=str(REPO), env=memmap_ppl_env(test_gpu), capture_output=True, text=True, timeout=900)
    m = re.search(r"nll/tok=([0-9.]+)", p.stdout + p.stderr)
    assert p.returncode == 0 and m and math.isfinite(float(m.group(1))), (f"{arm}: val CE at 8192 failed rc={p.returncode}; patched config mixers={c.get('layer_mixers')} "
        f"dm keys={[k for k in c if k.startswith('dm_')]} at {hf2}\n--- scorer head ---\n{(p.stdout + p.stderr)[:1800]}\n--- scorer tail ---\n{(p.stdout + p.stderr)[-1800:]}")


def test_stage2_launcher_refuses_without_warm_start(tmp_path):
    """Dry: with an empty workspace the launcher must stop at the warm-start check (exit 2) before touching the trainer."""
    p = subprocess.run(["bash", str(REPO / "scripts" / "campaign" / "stage2_arm.sh"), "dm_matched_hd128", "11785"],
                       capture_output=True, text=True, env=dry_env(tmp_path), timeout=120)
    assert p.returncode == 2 and "warm-start export missing" in (p.stdout + p.stderr), f"rc={p.returncode}\n{(p.stdout + p.stderr)[-1500:]}"


@pytest.mark.parametrize("arm", ["mla_clean_hd128", "dm_consistent_hd128", "gqa_dm_matched_hd128", "no_such_arm"])
def test_stage2_launcher_runs_only_the_stage2_arms(arm, tmp_path):
    """Dry: an arm outside the stage-2 experiment is refused (exit 2) before any other check."""
    p = subprocess.run(["bash", str(REPO / "scripts" / "campaign" / "stage2_arm.sh"), arm, "11785"],
                       capture_output=True, text=True, env=dry_env(tmp_path), timeout=120)
    assert p.returncode == 2 and "stage 2 runs the head_dim-128" in (p.stdout + p.stderr), f"rc={p.returncode}\n{(p.stdout + p.stderr)[-1500:]}"


def _fake_stage2_inputs(tmp_path, seed):
    """The files stage2_arm.sh checks for: the Standard arm's stage-1 export of the seed and the ProLong-64K corpus."""
    exp = tmp_path / "qwen35_m" / "hf" / f"Q35M_clean_sdpa_ffn8_hd128_{seed}"
    exp.mkdir(parents=True)
    (exp / "model.safetensors").write_bytes(b"")
    s2 = tmp_path / "data" / "prolong_64k_llama2"
    s2.mkdir(parents=True)
    for f in ("prolong64k_train.bin", "prolong64k_train.doc_lengths.npy"):
        (s2 / f).write_bytes(b"")
    return exp, s2


@pytest.mark.parametrize("arm,seed", [("clean_hd128", "11785"), ("dm_matched_hd128", "2"), ("dm_stale_hd128", "11785"), ("te_fp8", "2")])
def test_stage2_launcher_composes_the_recipe(arm, seed, tmp_path):
    """Dry (S2_DRY=1): the composed arguments are the paper's recipe, and the trainer's own config loader turns them into
    exactly 100 warm-up steps and 2,000 decay steps of 28,000 (the trainer truncates num_steps * ratio)."""
    exp, s2 = _fake_stage2_inputs(tmp_path, seed)
    p = subprocess.run(["bash", str(REPO / "scripts" / "campaign" / "stage2_arm.sh"), arm, seed], capture_output=True,
                       text=True, env=dry_env(tmp_path, S2_DRY=1), timeout=120)
    out = p.stdout + p.stderr
    assert p.returncode == 0 and "S2_DRY:" in out, f"rc={p.returncode}\n{out[-2000:]}"
    line = out.split("S2_DRY:", 1)[1].strip()
    kv = dict(re.findall(r"(TAG|NUM_STEPS|DM_LR)=(\S+)", line.split("EXTRA_ARGS=", 1)[0]))
    assert kv == {"TAG": "_s2_64k", "NUM_STEPS": "28000", "DM_LR": "8e-4"}, kv
    extra = line.split("EXTRA_ARGS=", 1)[1].split()
    arg = {extra[i]: extra[i + 1] for i in range(len(extra) - 1) if extra[i].startswith("--")}
    assert arg["--init_from_hf"] == str(exp), arg["--init_from_hf"]
    assert arg["--data.paths"] == str(s2 / "prolong64k_train.bin") and arg["--data.chunk_subset"] == "[0.0,0.85]", arg
    assert (arg["--model.max_seq_len"], arg["--global_batch_size"], arg["--per_device_batch_size"]) == ("65536", "16", "1"), arg
    assert arg["--gradient_checkpointing"] == "true" and arg["--optim.lr"] == "8e-4" and arg["--optim.decay_type"] == "linear", arg
    assert json.loads(arg["--model.rope_scaling"]) == {"type": "yarn", "factor": 8.0, "original_max_position_embeddings": 8192,
                                                       "beta_fast": 32, "beta_slow": 1}, arg["--model.rope_scaling"]
    assert arg["--hf_export_steps"] == "[" + ",".join(str(k) for k in range(2000, 28000, 2000)) + "]", arg["--hf_export_steps"]
    from pretrain.config import load_config
    c = load_config(["--config", str(REPO / "src" / "pretrain" / "configs" / "qwen35_m.yaml"), "--num_steps", "28000",
                     "--optim.warmup_ratio", arg["--optim.warmup_ratio"], "--optim.decay_ratio", arg["--optim.decay_ratio"]])
    assert int(c.num_steps * c.optim.warmup_ratio) == 100 and int(c.num_steps * c.optim.decay_ratio) == 2000, \
        (c.optim.warmup_ratio, c.optim.decay_ratio)


def test_stage2_launcher_schedule_knobs(tmp_path):
    """Dry: other step counts compose exactly too, and an impossible schedule is refused."""
    _fake_stage2_inputs(tmp_path, "11785")
    p = subprocess.run(["bash", str(REPO / "scripts" / "campaign" / "stage2_arm.sh"), "clean_hd128"], capture_output=True, text=True,
                       env=dry_env(tmp_path, S2_DRY=1, S2_STEPS=5000, S2_WARM_STEPS=7, S2_DECAY_STEPS=1000, S2_EXPORT_EVERY=0), timeout=120)
    out = p.stdout + p.stderr
    assert p.returncode == 0, out[-2000:]
    w = float(re.search(r"--optim.warmup_ratio (\S+)", out).group(1)); d = float(re.search(r"--optim.decay_ratio (\S+)", out).group(1))
    assert int(5000 * w) == 7 and int(5000 * d) == 1000 and "--hf_export_steps []" in out, out[-1500:]
    p = subprocess.run(["bash", str(REPO / "scripts" / "campaign" / "stage2_arm.sh"), "clean_hd128"], capture_output=True, text=True,
                       env=dry_env(tmp_path, S2_DRY=1, S2_STEPS=1000, S2_DECAY_STEPS=2000), timeout=120)
    assert p.returncode == 2 and "could not compose the schedule" in (p.stdout + p.stderr), (p.stdout + p.stderr)[-1500:]


@pytest.mark.gpu
@pytest.mark.parametrize("mode,streams", [("kv", 1), ("head", 1), ("head", 3), ("kv", 2)])
def test_fp8_backward_per_head_group_is_bitwise(mode, streams, tmp_path):
    """The stage-2 memory lever: the fused fp8 backward called per KV-head group or per query head (forced by
    DM_BWD_HEAD_GROUPS) must return bitwise the gradients of the single call (heads are independent in the op).
    streams > 1 (DM_BWD_STREAMS) adds the stage-2 pipeline lever: the groups on several CUDA streams with the
    matched delta's pass 0 per group; the gradients must still be bitwise the single call. The grouped run must
    announce the split (`BACKWARD PER HEAD GROUP`) and the single call must not, so the comparison is never vacuous."""
    require_kernel()
    code = r'''
import sys, torch
from pretrain.modeling.config import ModelConfig
from pretrain.modeling.mixers import build_mixer
from pretrain.modeling.layers import RotaryEmbedding
torch.manual_seed(0); dev = "cuda"; T = 2048
cfg = ModelConfig(vocab_size=32000, d_model=256, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=128, max_seq_len=T,
                  layer_mixers=["gated_deltanet"] * 3 + ["dm"], intermediate_size=512, gdn_head_dim=64, gdn_num_heads=4,
                  attn_output_gate=True, partial_rotary_factor=0.25, qk_norm=True, attn_impl="sdpa", rope_theta=1e7,
                  dm_mode="matched", dm_fused_producers=True)
m = build_mixer("dm", cfg, 3).to(dev, torch.bfloat16)
import torch.nn as nn
from pretrain.parallel.wrap import apply_fp8
host = nn.Module(); host.model = nn.Module(); host.model.layers = nn.ModuleList([nn.Module()]); host.model.layers[0].mixer = m
assert apply_fp8(host, "tensorwise", "ffn_qkvo") >= 2 and type(m.q_proj).__name__ == "Float8Linear"
x = torch.randn(1, T, 256, device=dev).to(torch.bfloat16).requires_grad_(True)
rot = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, rope_scaling=None, rotary_dim=cfg.rotary_dim).to(dev)
cos, sin = rot(torch.arange(T, device=dev)[None]); cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
o = m(x, cos, sin); (o.float() * torch.randn_like(o, dtype=torch.float32)).sum().backward()
torch.save({"o": o.detach().cpu(), "x": x.grad.cpu(), **{n: p.grad.cpu() for n, p in m.named_parameters() if p.grad is not None}}, sys.argv[1])
print("saved", sys.argv[1])
'''
    outs = {}
    for groups in ("off", mode):
        env = _plain_kernel_env()
        env["DM_BWD_HEAD_GROUPS"] = groups
        env["DM_BWD_STREAMS"] = "1" if groups == "off" else str(streams)
        f = tmp_path / f"grads_{groups}.pt"
        p = subprocess.run([python(), "-c", code, str(f)], capture_output=True, text=True, env=env, timeout=900)
        assert p.returncode == 0, f"{groups}: rc={p.returncode}\n{p.stderr[-3000:]}"
        split = "BACKWARD PER HEAD GROUP" in p.stdout
        assert split == (groups != "off"), f"DM_BWD_HEAD_GROUPS={groups}: split announced={split}\n{p.stdout[-2000:]}"
        outs[groups] = __import__("torch").load(f)
    a, b = outs["off"], outs[mode]
    assert set(a) == set(b) and len(a) >= 5, sorted(a)
    diffs = {k: (a[k].float() - b[k].float()).abs().max().item() for k in a}
    bad = {k: v for k, v in diffs.items() if v != 0.0}
    assert not bad, f"per-{mode} backward is not bitwise the single call: max abs diffs {bad}"
