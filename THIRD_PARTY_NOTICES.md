# Third-party notices

This repository is licensed under the Apache License 2.0 (`LICENSE`, with the project notice in `NOTICE`). The files
below are adapted from other projects and remain under those projects' licenses; every one keeps (or names) the upstream
copyright notice in its header. Files under the Apache License 2.0 were modified for this repository; that license's
text is the one in `LICENSE`. Libraries this repository only imports (PyTorch, transformers,
flash-linear-attention, mamba_ssm, lm-evaluation-harness, Liger-Kernel, torchao, ...) are dependencies, not
redistributed code, and are not listed.

| component | files | copyright | license |
|---|---|---|---|
| NVIDIA training / evaluation framework (jetlm) | `src/utils/{__init__,config,debug,dist,misc}.py`, `src/modeling/{__init__,wrapper}.py`, `src/evaluation/meta_eval.py`, `src/evaluation/lm_eval_harness/{__init__,evaluator,wrapper}.py` | Copyright 2025 NVIDIA CORPORATION & AFFILIATES | Apache-2.0 |
| RULER (https://github.com/NVIDIA/RULER) | `src/evaluation/ruler/{scorer,data_constants,eval_constants}.py`, `src/evaluation/ruler/config_tasks.yaml` | Copyright (c) 2024, NVIDIA CORPORATION | Apache-2.0 |
| Hugging Face transformers | the YaRN helpers in `src/pretrain/modeling/layers.py` and `scripts/hf/remote_code/modeling_dmlm.py`; the top-k / top-p filters in `src/inference/sampler.py`; the Nemotron-H mixer and attention paths in `src/inference/layers.py` | Copyright 2024 The HuggingFace Inc. team (and Copyright (c) 2025, NVIDIA CORPORATION for Nemotron-H) | Apache-2.0 |
| mamba_ssm (https://github.com/state-spaces/mamba) | `EngineMamba2` in `src/inference/dm_layers.py` | Copyright (c) 2024, Tri Dao, Albert Gu | Apache-2.0 |
| zoology (https://github.com/HazyResearch/zoology) | `make_mqar` in `src/evaluation/recall/synthetic_kv.py` | Copyright 2021 The Meerkat Team | Apache-2.0 |
| flash-linear-attention (https://github.com/fla-org/flash-linear-attention) | `EngineGDN`, `EngineKDA` in `src/inference/dm_layers.py` | Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li | MIT |
| lm-evaluation-harness (https://github.com/EleutherAI/lm-evaluation-harness) | parts of `src/evaluation/lm_eval_harness/wrapper.py` | Copyright (c) 2020 EleutherAI | MIT |
| LiveCodeBench (https://github.com/LiveCodeBench/LiveCodeBench) | `src/evaluation/lm_eval_tasks/_lcb_testing_util.py` | the LiveCodeBench authors | MIT |
| human-eval (https://github.com/openai/human-eval) | `reliability_guard` in `src/evaluation/lm_eval_tasks/_lcb_testing_util.py` (via LiveCodeBench) | Copyright (c) OpenAI (https://openai.com) | MIT |
| Liger-Kernel (https://github.com/linkedin/Liger-Kernel) | the forward kernel of `src/kernel/fp8glue/rmsnorm_cast.py` | Copyright 2024 LinkedIn Corporation | BSD-2-Clause (full text in the file header) |

## MIT License (flash-linear-attention, lm-evaluation-harness, LiveCodeBench, human-eval)

The copyright line of each project is in the table above and in the file headers.

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
