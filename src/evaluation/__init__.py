"""The evaluation suite: meta_eval (the lm-eval harness, RULER and recall benchmarks), ppl (memmap_ppl, the
validation cross-entropy, and pg19_ppl), recall and ruler. Run from the repository root with PYTHONPATH=src, e.g.
`torchrun -m evaluation.meta_eval --eval_config src/evaluation/configs/<benchmark>.yaml ...`; docs/benchmark.md has
the protocol."""
