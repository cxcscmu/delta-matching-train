"""dm_port: the training-side port of the Delta-Matching fp8 attention kernel (the `delta_matching` package).

    orchestrator.py        the fused-producer orchestrator of the `dm` mixer (fwd_fp8 / delta_fp8 / bwd_fp8_t on the
                           producers' codes, scales and twins; the stale / matched / bf16_dp delta modes)
    diag.py                the optional side-channel diagnostics of the stale-vs-matched study (DM_DIAG*, default off)
    fused_producers/       the Triton producers that feed the kernel from the mixer's own tensors: tier3a_ropeq (q RoPE
                           + Q codes in one pass), m13_rope_bwd (the bitwise q/k RoPE backward), tier1_vproj (the V
                           quantizer)
"""
