"""The "cast glue" ports of Transformer Engine's fp8 mechanisms (P1-P4 below), re-implemented in Triton inside this
repository's stack. TE is NOT imported anywhere here.

Kernels (all deterministic; every amax is an order-free max):
  cast_transpose.py  P4: bf16 -> fp8 (e4m3|e5m2) ROWWISE + COLUMNWISE twin in one read, amax side output
  swiglu_cast.py     P1: fwd g,u -> h8 (+twin); bwd dh,g,u -> dg8/du8 (+twins); h never materialized
  rmsnorm_cast.py    P2: Liger-identical RMSNorm forward (same 1D reduction tree) + amax epilogue; adapted from
                     Liger-Kernel (BSD 2-Clause, license text in the file)
  scaling.py         P3: dynamic (torchao-bitwise) and DELAYED (TE-class amax history) scale rules
  ops.py             autograd Functions that save the P4 codes and twins: fp8 MLP, shared-x8 q/k/v projection,
                     generic fp8 linear

The trainer enables the glue by default (env DM_FP8GLUE=1, DM_FP8GLUE_SCALING=dynamic|delayed; 0 opts out);
see src/pretrain/modeling/mlp_fp8glue.py.
"""
from .common import E4M3_MAX, E5M2_MAX, Fp8Pack, fp8_max_of  # noqa: F401
FP8GLUE_VERSION = "fp8glue-v1"
