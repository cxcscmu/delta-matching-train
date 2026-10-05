"""The hybrids' mixers: MLA (modeling/mla.py; bf16 core and the Delta-Matching fp8 core), the Mamba2 mixer and KDA (modeling/kda.py wrapping fla.layers.kda)
with its exposed geometry and dedicated fused kernels. Module-level numerics, the HF export round trip (the eval
path builds the same mixers from the saved config), and parameter parity of the M-tier configs."""
import json, math, sys
import pytest
import torch
from _helpers import REPO, SRC, require_kernel, LN_VOCAB

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _mc(**over):
    from pretrain.modeling.config import ModelConfig
    base = dict(vocab_size=32000, d_model=256, n_layers=4, n_heads=4, n_kv_heads=4, head_dim=128, kv_lora_rank=96,
                max_seq_len=512, layer_mixers=["gated_deltanet", "gated_deltanet", "gated_deltanet", "mla"],
                intermediate_size=512, gdn_head_dim=64, gdn_num_heads=4, attn_output_gate=True, partial_rotary_factor=0.5,
                qk_norm=True, attn_impl="sdpa", rope_theta=1e7, norm_eps=1e-6, tie_embeddings=True)
    base.update(over)
    return ModelConfig(**base)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


# ---------------------------------------------------------------- parameter parity (CPU, meta device)
def test_mtier_configs_are_parameter_matched():
    import yaml
    from pretrain.modeling.config import ModelConfig
    from pretrain.modeling.model import PretrainLM
    def count(mc):
        with torch.device("meta"):
            m = PretrainLM(mc)
        seen, tot = set(), 0
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p)); tot += p.numel()
        return tot
    base = yaml.safe_load(open(REPO / "src/pretrain/configs/qwen35_m.yaml"))["model"]
    t_gdn = count(ModelConfig(**dict(base, n_heads=16, n_kv_heads=4, head_dim=128)))
    for name in ("qwen35_m_mla", "qwen35_m_mamba2", "qwen35_m_kda"):
        mc = ModelConfig(**yaml.safe_load(open(REPO / f"src/pretrain/configs/{name}.yaml"))["model"])
        t = count(mc)
        assert abs(t - t_gdn) / t_gdn < 2e-3, f"{name}: {t:,} vs the GDN:GQA twin {t_gdn:,} ({(t - t_gdn) / t_gdn * 100:+.3f}%)"
    # the analytic count of the MLA attention layer equals the live module
    from pretrain.modeling.mla import MLAAttention
    mc = _mc()
    live = sum(p.numel() for p in MLAAttention(mc, 3).parameters())
    d, dc, rd, dn, H, D = mc.d_model, mc.kv_lora_rank, mc.rotary_dim, mc.head_dim - mc.rotary_dim, mc.n_heads, mc.head_dim
    want = d * 2 * H * D + d * (dc + rd) + dc + dc * H * dn + dc * H * D + H * D * d + 2 * (rd + dn)
    assert live == want, (live, want)


# ---------------------------------------------------------------- MLA numerics (GPU)
@pytest.mark.gpu
def test_mla_clean_forward_backward():
    from pretrain.modeling import build_model
    torch.manual_seed(0)
    model = build_model(_mc()).cuda()
    ids = torch.randint(0, 32000, (2, 512), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(ids, labels=ids)["loss"].float()
    assert math.isfinite(loss.item()) and abs(loss.item() - LN_VOCAB) < 1.0, loss.item()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    mla = model.model.layers[3].mixer
    assert type(mla).__name__ == "MLAAttention" and mla.kv_lora_rank == 96 and mla.rotary_dim == 64


@pytest.mark.gpu
def test_mla_fp8_core_matches_bf16_core_and_modes_share_forward_and_dv():
    """mla_dm (Delta-Matching, stale and matched) against the same MLA layer with the bf16 core, identical weights."""
    require_kernel()
    from pretrain.modeling.mla import MLAAttention, MLADMAttention
    from pretrain.modeling.layers import RotaryEmbedding
    torch.manual_seed(0)
    mc = _mc(max_seq_len=1024)
    ref = MLAAttention(mc, 3).cuda().bfloat16()
    outs = {}
    for mode in ("stale", "matched"):
        m = MLADMAttention(_mc(max_seq_len=1024, dm_mode=mode), 3).cuda().bfloat16()
        m.load_state_dict(ref.state_dict())
        outs[mode] = m
    x = (torch.randn(1, 1024, 256, device="cuda") * 0.5).bfloat16()
    pos = torch.arange(1024, device="cuda")[None]
    cos, sin = RotaryEmbedding(mc.head_dim, mc.max_seq_len, mc.rope_theta, rotary_dim=mc.rotary_dim)(pos)
    cos, sin = cos.bfloat16(), sin.bfloat16()
    g = torch.randn(1, 1024, 256, device="cuda").bfloat16()
    def run(m):
        xx = x.clone().requires_grad_()
        y = m(xx, cos, sin); y.backward(g)
        return y.detach(), xx.grad.detach(), {n: p.grad.detach().clone() for n, p in m.named_parameters()}
    y_ref, dx_ref, _ = run(ref)
    res = {mode: run(m) for mode, m in outs.items()}
    for mode, (y, dx, _) in res.items():
        assert torch.isfinite(y.float()).all() and torch.isfinite(dx.float()).all()
        assert _rel(y, y_ref) < 8e-2, f"{mode}: output rel err {_rel(y, y_ref)}"
        assert _rel(dx, dx_ref) < 2e-1, f"{mode}: input-grad rel err {_rel(dx, dx_ref)}"
    ys, dxs, gs = res["stale"]; ym, dxm, gm = res["matched"]
    assert torch.equal(ys, ym), "forward must be bitwise identical across delta modes"
    assert torch.equal(gs["v_proj.weight"], gm["v_proj.weight"]), "dV (hence W_UV's grad) must be identical across delta modes"
    assert not torch.equal(gs["k_proj.weight"], gm["k_proj.weight"]), "dK must differ between the stale and the matched delta"


@pytest.mark.gpu
def test_mla_hf_export_round_trip(tmp_path):
    """save_hf_checkpoint -> DMLMForCausalLM.from_pretrained rebuilds the MLA layers (kv_lora_rank round-trips) and matches."""
    from pretrain.modeling import build_model
    from pretrain.hf.export import save_hf_checkpoint
    from pretrain.hf.modeling_dmlm import DMLMForCausalLM, register_auto
    register_auto()
    torch.manual_seed(0)
    mc = _mc()
    model = build_model(mc).cuda().eval()
    out = tmp_path / "hf_mla"
    save_hf_checkpoint(model.state_dict(), mc, str(out))
    cfg = json.load(open(out / "config.json"))
    assert cfg.get("kv_lora_rank") == 96 and cfg["layer_mixers"][3] == "mla", cfg
    hf = DMLMForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16).cuda().eval()
    ids = torch.randint(0, 32000, (1, 256), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = model(ids)["logits"].float()
        b = hf(input_ids=ids).logits.float()
    assert _rel(a, b) < 5e-2, _rel(a, b)


# ---------------------------------------------------------------- Mamba2 (GPU)
@pytest.mark.gpu
def test_mamba2_hybrid_forward_backward_uses_fused_kernels_and_round_trips(tmp_path):
    from pretrain.modeling import build_model
    from pretrain.hf.export import save_hf_checkpoint
    from pretrain.hf.modeling_dmlm import DMLMForCausalLM, register_auto
    register_auto()
    torch.manual_seed(0)
    mc = _mc(n_kv_heads=1, kv_lora_rank=None, layer_mixers=["mamba2", "mamba2", "mamba2", "gqa"], partial_rotary_factor=0.25,
             mamba2_headdim=32, mamba2_d_state=64, mamba2_chunk_size=128)
    model = build_model(mc).cuda()
    mm = model.model.layers[0].mixer.mamba
    assert mm.headdim == 32 and mm.d_state == 64 and mm.chunk_size == 128 and mm.use_mem_eff_path and mm.d_inner == 512, (mm.headdim, mm.d_state, mm.d_inner)
    ids = torch.randint(0, 32000, (2, 512), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(ids, labels=ids)["loss"].float()
    assert math.isfinite(loss.item()) and abs(loss.item() - LN_VOCAB) < 1.0, loss.item()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    out = tmp_path / "hf_m2"
    save_hf_checkpoint(model.state_dict(), mc, str(out))
    cfg = json.load(open(out / "config.json"))
    assert cfg.get("mamba2_headdim") == 32 and cfg.get("mamba2_d_state") == 64 and cfg.get("mamba2_chunk_size") == 128, cfg
    hf = DMLMForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16).cuda().eval()
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = model(ids[:1, :256])["logits"].float(); b = hf(input_ids=ids[:1, :256]).logits.float()
    assert _rel(a, b) < 5e-2, _rel(a, b)


# ---------------------------------------------------------------- KDA (GPU)
@pytest.mark.gpu
def test_kda_hybrid_forward_backward_and_round_trips(tmp_path):
    """The KDA:GQA hybrid on fla's chunk kernels: finite loss near ln(vocab) at init, finite grads, the kda_* fields in the
    export, and the HF reload reproducing the logits (the eval path builds the same mixer from the saved config)."""
    from pretrain.modeling import build_model
    from pretrain.hf.export import save_hf_checkpoint
    from pretrain.hf.modeling_dmlm import DMLMForCausalLM, register_auto
    register_auto()
    torch.manual_seed(0)
    mc = _mc(n_kv_heads=1, kv_lora_rank=None, layer_mixers=["kda", "kda", "kda", "gqa"], partial_rotary_factor=0.25,
             kda_expand_v=1.0, kda_conv_size=4)
    model = build_model(mc).cuda()
    k = model.model.layers[0].mixer.kda
    assert type(model.model.layers[0].mixer).__name__ == "KDAMixer" and k.num_heads == 4 and k.head_k_dim == 64 and k.head_v_dim == 64 \
        and k.use_short_conv and k.conv_size == 4 and k.mode == "chunk", (k.num_heads, k.head_k_dim, k.head_v_dim)
    ids = torch.randint(0, 32000, (2, 512), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(ids, labels=ids)["loss"].float()
    assert math.isfinite(loss.item()) and abs(loss.item() - LN_VOCAB) < 1.0, loss.item()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.model.layers[0].mixer.kda.f_proj[0].weight.grad.abs().sum() > 0, "the KDA gate projection got no gradient"
    out = tmp_path / "hf_kda"
    save_hf_checkpoint(model.state_dict(), mc, str(out))
    cfg = json.load(open(out / "config.json"))
    assert cfg["layer_mixers"][:3] == ["kda", "kda", "kda"] and cfg.get("kda_expand_v") == 1.0 and cfg.get("kda_conv_size") == 4, cfg
    hf = DMLMForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16).cuda().eval()
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        a = model(ids[:1, :256])["logits"].float(); b = hf(input_ids=ids[:1, :256]).logits.float()
    assert _rel(a, b) < 5e-2, _rel(a, b)


@pytest.mark.gpu
def test_kda_engine_prefill_decode_matches_hf_forward(tmp_path):
    """EngineKDA (src/inference/dm_layers.py): the engine's prefill and twelve one-token decodes over the slot pools
    reproduce the HF forward's logits on a tiny kda+gqa model, i.e. the decode path (fused_recurrent_kda with the gathered
    state) is faithful to the training forward (chunk_kda)."""
    from transformers import AutoModelForCausalLM
    from pretrain.modeling import build_model
    from pretrain.hf.export import save_hf_checkpoint
    from pretrain.hf.modeling_dmlm import register_auto
    from inference.engine import EngineModel
    from inference.dm_cache import DMStatePool
    from inference.context import EngineContext, PHASE_PREFILL, PHASE_DECODE
    register_auto()
    torch.manual_seed(0)
    mc = _mc(n_kv_heads=1, kv_lora_rank=None, layer_mixers=["kda", "kda", "kda", "gqa"], partial_rotary_factor=0.25, vocab_size=512)
    model = build_model(mc).cuda().bfloat16().eval()
    out = tmp_path / "hf_kda_eng"
    save_hf_checkpoint({k: v for k, v in model.state_dict().items()}, mc, str(out))
    hf = AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16).cuda().eval()
    L, M = 40, 12
    seq = torch.randint(1, 512, (1, L + M), device="cuda")
    with torch.no_grad():
        ref = hf(seq).logits.float()
    em = EngineModel(hf)
    assert type(em.engine.layers[0].mixer).__name__ == "EngineKDA", type(em.engine.layers[0].mixer).__name__
    pool = DMStatePool(em.engine.layers, 1, L + M + 2, torch.device("cuda"), kv_dtype=torch.bfloat16, state_dtype=em.state_dtype)
    ctx = EngineContext(pool=pool, phase=PHASE_PREFILL, prefill_slot=0)
    with torch.no_grad():
        pre = em.engine.forward(seq[:, :L], ctx, logits_last_only=False).float()
    pool.seqlens[0] = L
    top1_pref = (pre.argmax(-1) == ref[:, :L].argmax(-1)).float().mean().item()
    assert torch.isfinite(pre).all() and top1_pref > 0.98, ("prefill", top1_pref, (pre - ref[:, :L]).abs().max().item())
    hits = 0
    for j in range(M):
        ctx = EngineContext(pool=pool, phase=PHASE_DECODE, slot_ids=torch.tensor([0], device="cuda", dtype=torch.int32))
        with torch.no_grad():
            got = em.engine.forward(seq[:, L + j:L + j + 1], ctx, logits_last_only=True).float()
        pool.seqlens[0] += 1
        hits += int(got[:, -1].argmax(-1).item() == ref[:, L + j].argmax(-1).item())
    assert hits >= M - 1, ("decode top-1 agreement", hits, M)
