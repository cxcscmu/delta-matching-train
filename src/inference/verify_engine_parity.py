"""Teacher-forced parity of the continuous-batching engine against the HF forward, on a GPU.

    PYTHONPATH=src python -m inference.verify_engine_parity [case]

Feeds the SAME sequence (prompt + a fixed continuation) to both the engine (prefill + slot-cached decode) and HF
.forward, and compares the logits PER POSITION (no free-running cascade). The decode-path positions are the real
cache-correctness test. A tiny random-weight model is built, exported to a temporary directory (removed afterwards)
and loaded through this repository's HF loader. Exit status 0 on PASS, 1 on FAIL, 2 on a bad argument.

case (default gqa):
  gqa | mla | mamba2 | gated_deltanet (or gdn) | kda   three layers of one mixer
  nope                                                  GQA with nope=True: the engine must apply the identity rotation
  hybrid_gdn | hybrid_mamba2 | hybrid_kda              the 3:1 hybrid [recurrent x3, gqa]
  hybrid_gdn_mla                                        the 3:1 hybrid [gated_deltanet x3, mla]
"""
import os
import shutil
import sys
import tempfile

CASES = ("gqa", "mla", "mamba2", "gated_deltanet", "gdn", "kda", "nope",
         "hybrid_gdn", "hybrid_mamba2", "hybrid_kda", "hybrid_gdn_mla")
CLEAR_MARGIN = 0.05   # logit units: a top-1 lead smaller than this counts as a near tie


def _model_config(case: str):
    from pretrain.modeling.config import ModelConfig
    common = dict(vocab_size=256, d_model=256, n_heads=4, n_kv_heads=2, head_dim=64,
                  max_seq_len=512, attn_impl="flash", z_loss_weight=0.0, qk_norm=True)
    if case in ("mla", "hybrid_gdn_mla"):
        # MLA's own asserts (modeling/mla.py): MHA on the wire (n_kv_heads == n_heads), 0 < rotary_dim <
        # head_dim (partial_rotary_factor 0.5, as the campaign config uses), kv_lora_rank a multiple of 16.
        # attn_impl sdpa mirrors the patched export config (eval_arm.sh asserts it). The load in main() passes
        # attn_implementation="flash_attention_2", as the evaluation's loader does by default, and the HF wrapper maps
        # it to attn_impl flash, so the reference runs the bf16 MLA core on flash-attn, the kernel the engine uses too.
        # Checked BEFORE the generic hybrid_ branch: "hybrid_gdn_mla".split("_", 1)[1] is "gdn_mla", which that branch
        # would mis-parse.
        # head_dim 128: MLA runs on the Delta-Matching kernels in the fp8 arms and config.py asserts head_dim in
        # (128, 256), so the harness's default 64 cannot be used for this cell.
        cm = {**common, "n_kv_heads": common["n_heads"], "attn_impl": "sdpa", "head_dim": 128,
              "kv_lora_rank": 64, "partial_rotary_factor": 0.5}
        if case == "hybrid_gdn_mla":    # the campaign's real 3:1 GatedDeltaNet : MLA shape
            return ModelConfig(n_layers=4, mixer="mla",
                               layer_mixers=["gated_deltanet", "gated_deltanet", "gated_deltanet", "mla"], **cm)
        return ModelConfig(n_layers=3, mixer="mla", **cm)
    if case == "nope":
        # NoPE with a partial rotary slice, as in the campaign's NoPE cell (partial_rotary_factor 0.25): an engine that
        # ignored `nope` would rotate the leading rotary_dim channels that the trained model leaves unrotated.
        return ModelConfig(n_layers=3, mixer="gqa", nope=True, partial_rotary_factor=0.25, **common)
    if case.startswith("hybrid_"):   # 3:1 hybrid: [recurrent,recurrent,recurrent,gqa]
        base = case.split("_", 1)[1]
        recur = "gated_deltanet" if base == "gdn" else base
        return ModelConfig(n_layers=4, mixer="gqa", layer_mixers=[recur, recur, recur, "gqa"], **common)
    return ModelConfig(n_layers=3, mixer={"gdn": "gated_deltanet"}.get(case, case), **common)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    case = sys.argv[1] if len(sys.argv) > 1 else "gqa"
    if case not in CASES or len(sys.argv) > 2:
        print(f"unknown or extra argument: {sys.argv[1:]}\n{__doc__}", file=sys.stderr)
        sys.exit(2)
    os.environ.setdefault("ENGINE_PROGRESS", "0")   # read when the scheduler is imported
    import torch
    from pretrain.modeling.model import build_model
    from pretrain.hf import save_hf_checkpoint, register_auto
    from transformers import AutoModelForCausalLM
    from inference import EngineModel
    from inference.context import EngineContext, PHASE_PREFILL, PHASE_DECODE

    register_auto()
    dev = "cuda"
    mc = _model_config(case)
    torch.manual_seed(0)
    m = build_model(mc).to(dev).to(torch.bfloat16).eval()
    out_dir = tempfile.mkdtemp(prefix=f"engine_tf_{case}_")
    try:
        save_hf_checkpoint(m.state_dict(), mc, out_dir)
        hf = AutoModelForCausalLM.from_pretrained(
            out_dir, dtype=torch.bfloat16, attn_implementation="flash_attention_2").to(dev).eval()
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    Lp, G = 19, 40
    torch.manual_seed(1)
    prompt = torch.randint(1, 256, (Lp,), device=dev)
    # fixed continuation (a real HF greedy roll-out, computed before the engine rebinds the mixers)
    seq = prompt[None].clone()
    for _ in range(G):
        with torch.no_grad():
            seq = torch.cat([seq, hf(seq).logits[:, -1].argmax(-1, keepdim=True)], 1)
    with torch.no_grad():
        hf_logits = hf(seq).logits[0].float()        # (Lp+G, V); position p predicts p+1
    full_len = seq.shape[1]

    # === engine teacher-forced (same `seq`) ===
    eng = EngineModel(hf)
    pool = eng._make_pool(1, full_len + 1, dev)
    slot = pool.alloc()
    ctx = EngineContext(pool=pool, phase=PHASE_PREFILL, prefill_slot=slot)
    pre = eng.engine.forward(prompt[None], ctx, logits_last_only=False)[0].float()   # (Lp, V)
    pool.seqlens[slot] = Lp
    eng_rows = [pre]
    for t in range(Lp, full_len):
        tok = seq[:, t:t + 1]
        ctxd = EngineContext(pool=pool, phase=PHASE_DECODE,
                             slot_ids=torch.tensor([slot], device=dev, dtype=torch.int32))
        lg = eng.engine.forward(tok, ctxd, logits_last_only=True)[0].float()          # (1, V) predicts t+1
        pool.seqlens[slot] += 1
        eng_rows.append(lg)
    eng_logits = torch.cat(eng_rows, 0)              # (Lp+G, V)

    # compare positions 0..len-2 (each predicts the next token)
    P = full_len - 1
    pre_d = (hf_logits[:Lp - 1] - eng_logits[:Lp - 1]).abs().max().item()
    dec_d = (hf_logits[Lp - 1:P] - eng_logits[Lp - 1:P]).abs().max().item()
    am = (hf_logits[:P].argmax(-1) == eng_logits[:P].argmax(-1)).float().mean().item()
    dec_am = (hf_logits[Lp - 1:P].argmax(-1) == eng_logits[Lp - 1:P].argmax(-1)).float().mean().item()
    # A tiny random-weight model has near-uniform logits, so a numerically equivalent decode path (e.g. Mamba2's step
    # kernel against the chunked scan of the prefill) can flip the argmax where the top two logits nearly tie. The
    # decisions that are clear (top-1 minus top-2 above CLEAR_MARGIN in the HF logits) must all agree; the logit gap
    # bounds the rest. A wrong cache or rotation moves the logits by far more (the NoPE bug: max|diff| 5.9).
    top2 = hf_logits[Lp - 1:P].topk(2, dim=-1).values
    clear = (top2[:, 0] - top2[:, 1]) > CLEAR_MARGIN
    agree = hf_logits[Lp - 1:P].argmax(-1) == eng_logits[Lp - 1:P].argmax(-1)
    clear_am = agree[clear].float().mean().item() if bool(clear.any()) else 1.0
    print(f"[{case}] teacher-forced | prefill max|Δ|={pre_d:.2e} | DECODE-path max|Δ|={dec_d:.2e} | "
          f"argmax match all={am*100:.1f}% decode={dec_am*100:.1f}% decode-clear={clear_am*100:.1f}% "
          f"({int(clear.sum())} of {clear.numel()} positions)", flush=True)
    ok = pre_d < 0.2 and dec_d < 0.2 and clear_am == 1.0
    print(f"[{case}] TEACHER-FORCED PARITY:", "PASS" if ok else "FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
