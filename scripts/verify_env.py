#!/usr/bin/env python3
"""Check the kernel libraries and packages of a delta-matching-train environment (needs a GPU; Hopper for the fp8 kernels).

    python scripts/verify_env.py        (or scripts/verify_env.py; exit 0 when every check passes)
"""
import os, sys, warnings
if sys.argv[1:2] in (["-h"], ["--help"]):
    print(__doc__); sys.exit(0)
warnings.filterwarnings("ignore")
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")   # <repo>/src, from any directory
os.environ.setdefault("PYTHONPATH", SRC)
sys.path.insert(0, SRC)
import torch
print(f"python      {sys.version.split()[0]}")
print(f"torch       {torch.__version__}  (cuda {torch.version.cuda})")
print(f"gpu         {torch.cuda.get_device_name(0)} sm{''.join(map(str,torch.cuda.get_device_capability(0)))} x{torch.cuda.device_count()}")
print()
ok=[]; bad=[]
def check(name, fn):
    try:
        r=fn(); ok.append(name); print(f"  PASS  {name}" + (f"  [{r}]" if r else ""))
    except Exception as e:
        bad.append((name,e)); print(f"  FAIL  {name}: {type(e).__name__}: {e}")

print("== kernel libraries ==")
def _fa2():
    import flash_attn, flash_attn_2_cuda  # noqa: F401  (import check)
    q=torch.randn(2,512,8,64,device="cuda",dtype=torch.bfloat16)
    from flash_attn import flash_attn_func, flash_attn_with_kvcache  # noqa: F401  (import check)
    o=flash_attn_func(q,q,q,causal=True); torch.cuda.synchronize()
    return f"flash_attn {flash_attn.__version__}, fwd {tuple(o.shape)}"
check("flash-attn (FA2 CUDA ext)", _fa2)

def _dm():
    # the fp8 attention kernel of the dm mixers: the separate delta-matching package (docs/SETUP.md section 3)
    import delta_matching
    from delta_matching.core import _ops
    o = _ops()
    bf16dp = int(getattr(o, "build_bf16dp", lambda: 0)())
    flags = f"build_gt={int(o.build_gt())} build_hd128={int(o.build_hd128())} build_bf16dp={bf16dp}"
    if bf16dp:   # a DM_BF16_DP=1 build serves only the trainer's dm_mode bf16_dp (no experiment arm uses it); dm_attn_func* refuse it
        return f"delta_matching {delta_matching.kernel_version()}: {flags} (a bf16_dp build: the forward check is skipped)"
    q=torch.randn(1,4,1024,128,device="cuda",dtype=torch.bfloat16)
    out=delta_matching.dm_attn_func_bhnd(q,q,q,causal=True,delta_mode="matched"); torch.cuda.synchronize()
    return f"delta_matching {delta_matching.kernel_version()}: {flags}, fwd {tuple(out.shape)}"
check("delta-matching (the fp8 attention kernel)", _dm)

def _cc1d():
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states  # noqa: F401  (import check)
    import causal_conv1d_cuda  # noqa: F401  (import check)
    x=torch.randn(2,64,128,device="cuda",dtype=torch.bfloat16)
    w=torch.randn(64,4,device="cuda",dtype=torch.bfloat16)
    y=causal_conv1d_fn(x,w,None,activation="silu"); torch.cuda.synchronize()
    import causal_conv1d as m
    return f"causal_conv1d {m.__version__}, out {tuple(y.shape)}"
check("causal_conv1d (CUDA ext)", _cc1d)

def _cc1d_upd():
    from causal_conv1d import causal_conv1d_update  # noqa: F401  (import check)
    return "causal_conv1d_update"
check("causal_conv1d_update", _cc1d_upd)

def _mamba():
    import mamba_ssm
    from mamba_ssm import Mamba2
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined  # noqa: F401  (import check)
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update  # noqa: F401  (import check)
    m=Mamba2(d_model=256, d_state=64, d_conv=4, expand=2, headdim=64).cuda().to(torch.bfloat16)
    x=torch.randn(2,64,256,device="cuda",dtype=torch.bfloat16)
    y=m(x); torch.cuda.synchronize()
    return f"mamba_ssm {mamba_ssm.__version__}, Mamba2 fwd {tuple(y.shape)}"
check("mamba_ssm (Mamba2 fwd)", _mamba)

def _fa3():
    import flash_attn_3
    import flash_attn_3.flash_attn_interface as fi3
    outs=[]
    for d, dt in [(128, torch.bfloat16), (256, torch.bfloat16)]:
        q=torch.randn(2,512,4,d,device="cuda",dtype=torch.bfloat16)
        o=fi3.flash_attn_func(q,q,q,causal=True,softmax_scale=d**-0.5)
        o=o[0] if isinstance(o,tuple) else o
        outs.append(f"bf16/d{d}")
    torch.cuda.synchronize()
    return f"flash_attn_3 {getattr(flash_attn_3,'__version__','?')}: " + ", ".join(outs)
check("flash_attn_3 (FA3, hopper)", _fa3)

def _te():
    # NOTE: transformer_engine MUST be imported after torch, or TE may pull in an older system
    # NCCL (no ncclCommResume) and torch then fails to load. Export
    #   LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/nccl/lib:$LD_LIBRARY_PATH
    # to make the order irrelevant.
    import torch as _t  # noqa: F401  (ensure torch is already loaded)
    import transformer_engine
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format
    from transformer_engine.pytorch.fp8 import check_fp8_support
    ok, reason = check_fp8_support()
    r = DelayedScaling(fp8_dpa=True, fp8_mha=False, fp8_format=Format.HYBRID,
                       amax_history_len=16, amax_compute_algo="max")
    lin = te.Linear(128,128).cuda()
    with te.autocast(enabled=True, recipe=r):
        y = lin(torch.randn(8,128,device="cuda"))  # noqa: F841
    torch.cuda.synchronize()
    return f"TE {transformer_engine.__version__}, fp8_supported={ok}"
check("transformer_engine (te_fp8)", _te)

def _fla():
    import fla
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule  # noqa: F401  (import check)
    B,T,H,K,V=2,128,4,64,64
    q=torch.randn(B,T,H,K,device="cuda",dtype=torch.bfloat16)
    k=torch.nn.functional.normalize(torch.randn(B,T,H,K,device="cuda",dtype=torch.bfloat16),dim=-1)
    v=torch.randn(B,T,H,V,device="cuda",dtype=torch.bfloat16)
    g=torch.nn.functional.logsigmoid(torch.randn(B,T,H,device="cuda",dtype=torch.float32))
    beta=torch.rand(B,T,H,device="cuda",dtype=torch.bfloat16)
    o,_=chunk_gated_delta_rule(q,k,v,g=g,beta=beta); torch.cuda.synchronize()
    return f"fla {fla.__version__}, GDN out {tuple(o.shape)}"
check("flash-linear-attention (GDN)", _fla)

def _liger():
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss  # noqa: F401  (import check)
    import liger_kernel; return "liger fused linear CE"  # noqa: F401  (import check)
check("liger_kernel", _liger)

def _tao():
    import torchao
    from torchao.float8 import convert_to_float8_training  # noqa: F401  (import check)
    return f"torchao {torchao.__version__}"
check("torchao (fp8)", _tao)

def _tri():
    import triton; return f"triton {triton.__version__}"
check("triton", _tri)

def _hf():
    import transformers, datasets
    return f"transformers {transformers.__version__}, datasets {datasets.__version__}"
check("transformers / datasets", _hf)

print("\n== delta-matching-train packages ==")
for mod in ["pretrain","evaluation","inference","modeling","utils","kernel"]:
    check(f"import {mod}", lambda m=mod: __import__(m) and "")

print(f"\n{len(ok)} passed, {len(bad)} failed")
if bad:
    print("\nFAILURES:")
    for n,e in bad: print(f"  - {n}: {type(e).__name__}: {e}")
sys.exit(1 if bad else 0)
