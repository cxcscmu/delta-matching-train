"""fused_producers: the Triton producers of the `dm` mixer's fp8 attention boundary (one module each; the file names
are historical):
    tier3a_ropeq.py    ropeq_q: fused partial RoPE + softmax-scale bake + block e4m3 quantize of Q (codes + scales)
    m13_rope_bwd.py    rope_bwd_q: the one-pass q-RoPE backward; k_rope_fused: the k RoPE as one kernel each way
    tier1_vproj.py     v8_quantize: bf16 V -> e4m3 codes + per-block scales
Each producer runs the same fp32 operations in the same order as the eager RoPE (or its autograd backward) followed by
the block quantizer of delta_matching/quant.py, so its codes, scales and bf16 outputs are the same bits; only the data
movement changes. m13_rope_bwd.py carries a GPU self-test against the eager reference
(python -m kernel.dm_port.fused_producers.m13_rope_bwd).
"""
