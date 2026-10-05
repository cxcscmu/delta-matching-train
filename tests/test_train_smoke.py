"""Training smoke: the trainer runs three optimizer steps of the tiny hybrid on one GPU for the GDN:GQA arms of
scripts/campaign/arms.sh (the clean bf16-attention arm; the stale, the matched and the consistent-dO Delta-Matching arms
with the full fp8 pipeline: fp8 FFN, fp8 q/k/v/o with fused producers, the fp8 cast glue, compiled MLP; at head_dim 128
and at head_dim 256), with the same trainer arguments and environment switches as scripts/campaign/train_arm.sh, and
exports an HF checkpoint at the end. Every fp8 run must announce the delta it was asked for (`dm_mode=<mode>` in the
`[dm] DMAttention` banner, and the delta-from-codes line for the consistent-dO arm), so a flag that never reaches the
model fails here. On a DM_BF16_DP=1 build of the kernel library (a study variant that refuses these modes) the smokes
skip."""
import math, re
import pytest
from _helpers import (CE_RE, DM_MODES, LN_VOCAB, arm_args, dm_route, require_kernel, require_standard_kernel,
                      train_tiny)

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.data]


def _smoke(arm, run_root, root, tok_dir, test_gpu):
    args, env = arm_args(arm)
    rc, log, hf, dt = train_tiny(f"tiny_{arm}", run_root, root, tok_dir, test_gpu,
                                 extra_args=args + ["--gradient_checkpointing", "false"], extra_env=env)
    assert rc == 0, f"{arm}: trainer exit {rc} after {dt:.0f}s; console tail:\n{log[-4000:]}"
    ces = [float(x) for x in CE_RE.findall(log)]
    assert len(ces) >= 3, f"{arm}: expected a ce_loss line per step, found {ces}; tail:\n{log[-2000:]}"
    assert all(math.isfinite(c) for c in ces), f"{arm}: non-finite loss {ces}"
    assert abs(ces[0] - LN_VOCAB) < 1.0, f"{arm}: step-0 CE {ces[0]} is far from ln(32000) = {LN_VOCAB:.3f}"
    for f in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):   # the export bakes the fast tokenizer
        assert (hf / f).exists(), f"{arm}: export file {f} missing in {hf}"
    route = dm_route(log)
    if arm in DM_MODES:
        assert route.get("dm") == DM_MODES[arm], \
            f"{arm}: the [dm] DMAttention banner says {route}, expected dm_mode={DM_MODES[arm]}; tail:\n{log[-2000:]}"
        hd = re.search(r"\[dm\] DMAttention:.*?\bhead_dim=(\d+)", log)
        want = "256" if arm.endswith("_hd256") else "128"
        assert hd and hd.group(1) == want, f"{arm}: the dm mixer ran at head_dim {hd and hd.group(1)}, expected {want}"
        from_codes = "DM_DELTA_Q8=1" in log
        assert from_codes is (arm == "dm_consistent_hd128"), \
            f"{arm}: the delta-from-codes line is {'present' if from_codes else 'absent'}; tail:\n{log[-2000:]}"
    else:
        assert "dm" not in route, f"{arm}: the clean arm built the dm mixer: {route}"


@pytest.mark.parametrize("arm", ["clean_hd128", "dm_stale_hd128", "dm_matched_hd128", "dm_consistent_hd128",
                                 "dm_stale_hd256", "dm_matched_hd256"])
def test_tiny_arm_trains_three_steps_and_exports(arm, run_root, root, tok_dir, test_gpu):
    if arm != "clean_hd128":
        require_kernel()
        require_standard_kernel(f"the {arm} smoke")
    _smoke(arm, run_root, root, tok_dir, test_gpu)
