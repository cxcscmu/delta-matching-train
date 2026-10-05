"""src.pretrain: a from-scratch LM pretraining engine (DDP + FSDP2).

The training engine of the Delta-Matching study: dense GQA attention (flash-attn / SDPA / the Delta-Matching fp8
kernel), Gated DeltaNet, Mamba2, KDA and MLA mixers through modeling.MIXER_REGISTRY. See docs/train_engine.md.
"""
__version__ = "0.1.0"
