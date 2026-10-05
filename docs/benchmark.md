# Evaluation protocol and references (confirmed against the code)

This document states the evaluation protocol behind every number in the paper. Every protocol statement below was checked
against the evaluator in this repository, the installed evaluation harness, the data files and real result artifacts. Code
pointers are `file:line` or `file::function` in this repository. Each run's full harness output records the evaluator
commit it ran (`git_hash`). **References** close each section.

The reported suite is validation cross-entropy, RULER-4K/8K, extraction recall, TriviaQA, MMLU, and CS9 with its nine-task
breakdown. The context-extension (stage-2) results add a 64K validation cross-entropy and RULER up to 64K. MQAR is still
computed by the evaluation chain (its result is required for the chain's completion marker: `mqar_mark` and the final
marker check of `scripts/campaign/eval_arm.sh`) but is no longer reported.

## 1. Shared evaluation stack

### Drivers and entry points

- Stage 1 (8K models): `scripts/campaign/eval_arm.sh <run>`. Stage 2 (64K context extension):
  `scripts/campaign/eval_arm_64k.sh <run>`.
- Validation cross-entropy runs `python -m evaluation.ppl.memmap_ppl` on one GPU; it reads token IDs directly and uses no
  tokenizer (step 1 of `eval_arm.sh`).
- Every other benchmark runs `torchrun --nproc_per_node=8 -m evaluation.meta_eval --model_name_or_path <export>
  --eval_config src/evaluation/configs/<benchmark>.yaml` (`run_cfg` in `eval_arm.sh` and `eval_arm_64k.sh`),
  data-parallel over 8 GPUs, with the configs in `src/evaluation/configs/`:
  `ruler_llama2_8k.yaml` (stage 2: `ruler_llama2_64k.yaml`), `recall_extraction.yaml`, `triviaqa.yaml`, `commonsense.yaml`,
  `mmlu.yaml`.
- A benchmark counts as done only by artifact: its results JSON, its `.complete` marker and zero tracebacks in its log. The
  chain then writes `EVAL_COMPLETE` (stage 2: `EVAL64K_COMPLETE`).
- Every run evaluates the full test or validation set of each task; no sample limit is set anywhere (`limit: None`).

### Exported model and inference path

- **Export.** Each checkpoint is exported to Hugging Face format (safetensors, all tensors bf16) with the tokenizer baked in:
  LlamaTokenizer, 32,000 entries, `<unk>`=0, `<s>`=1, `</s>`=2, no pad token (the loader sets pad = eos,
  `src/evaluation/ruler/load_model.py:103-104`). It encodes identically to `meta-llama/Llama-2-7b-hf` revision 01c7f73d.
- **Architecture is preserved.** The launcher's export patch (at the end of `train_arm.sh`) rewrites only the attention entries of
  the config: the fp8 training mixer `dm` becomes `gqa`, `mla_dm` becomes `mla`, `attn_impl` becomes `sdpa`, and the
  fp8-only `dm_mode` / `dm_fused_producers` keys are dropped. The recurrent layers (GatedDeltaNet, KDA, Mamba2) and the layer pattern are
  kept, e.g. `[gated_deltanet x3, gqa] x6` for the GDN:GQA cell; `config.train.json` keeps the training values.
  `eval_arm.sh` (and `eval_arm_64k.sh`) refuses an export whose attention is not `sdpa` or that still names `dm` or `mla_dm`.
- **Attention kernel.** Both model loaders request `attn_implementation = flash_attention_2` (the override variable
  `EVAL_ATTN_IMPL` is never set), which `src/pretrain/hf/modeling_dmlm.py:38-41` (`_ATTN_MAP`, :22) maps to
  `attn_impl: flash`. The attention layers therefore run FlashAttention-2 (flash_attn 2.8.3.post1) whatever the export's
  `sdpa` entry says; the generation engine also uses FlashAttention-2 with a bf16 KV cache. GatedDeltaNet layers run the
  fla kernels with an fp32 state.
- **One inference path for every arm.** FP8 (GEMMs and attention) is training-only. Every training method, the bf16 Standard
  arm, the Delta-Matching fp8 arms and the Transformer Engine arm, is evaluated from its bf16 export through the same bf16 path in
  one environment.
- **Two backends**, chosen per benchmark by the chain (`EVAL_ENGINE=mini_eval_dev`, set by `run_cfg` in `eval_arm.sh`):
  - the continuous-batching generation engine `inference.EngineModel` (the `EVAL_ENGINE` branch of `meta_eval.main`) for RULER, extraction and
    TriviaQA, with greedy decoding (argmax of fp32 logits);
  - the model's forward pass through the repository's own lm-eval model class (`src/evaluation/lm_eval_harness/wrapper.py`,
    driven by `evaluator.py` in the same directory; not the harness's built-in HF model) for CS9, MMLU and MQAR;
    validation cross-entropy uses the same forward pass.
- **Batch size per GPU:** CS9 32, MMLU 16, extraction 32, TriviaQA 32, RULER 16 (stage 2: 4, with 24 and 8 engine slots);
  MQAR 8 (the config's `batch_size` wins over the chain's `--eval_batch_size 16`).
- **No BOS token and no chat template** in any evaluation prompt (`add_prefix` False and `add_special_tokens` False,
  `LMEvalWrapper.__init__` and `tok_encode` in `src/evaluation/lm_eval_harness/wrapper.py`, `src/evaluation/ruler/ruler.py:286`),
  although every training
  document begins with BOS.
- **Likelihood scoring in bf16.** Log-softmax runs on bf16 logits and the per-answer sums are bf16
  (`LMEvalWrapper._loglikelihood_tokens`). Exactly tied answer choices therefore occur and go to the first option (`np.argmax`):
  WinoGrande 66 of 1,267, HellaSwag 286 of 10,042, MMLU 705 of 14,042 items. The rule is the same for every arm.
- **Harness.** lm-evaluation-harness 0.4.12, installed from PyPI and unmodified (all files match the wheel's recorded
  hashes), called through `simple_evaluate` with `bootstrap_iters=0`: the harness computes no standard errors.
- **Software.** torch 2.12.0+cu130, transformers 5.9.0, flash_attn 2.8.3.post1, flash-linear-attention (fla). The repository
  pins fla 0.5.2. The published runs: every 1.665 B run (the base environment, both clusters) trained on fla 0.5.0; the
  across-scale cells (0.569 B and 5.289 B) on 0.5.2, which also runs their GatedDeltaNet geometry with more value heads than key heads
  (0.5.0 refuses it); every published number was scored with 0.5.0. The two versions differ by about 2 bf16 ulps per
  GatedDeltaNet op (8e-3 relative on outputs and gradients, each version run-to-run deterministic); scoring an export
  under 0.5.2 instead of 0.5.0 moved val CE by 1e-6 nll/token (measured on the 5.289 B matched export).

**References.** lm-evaluation-harness: the [repository](https://github.com/EleutherAI/lm-evaluation-harness) (version
0.4.12) and the [reproducible-evaluation paper](https://arxiv.org/abs/2405.14782). The software reference does not pin
task definitions; sections 4 to 7 do.

### Means, uncertainty, units

Tables report the arithmetic mean and one standard error across initialization seeds:

$$
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i,\qquad
s=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2},\qquad
\mathrm{SE}=\frac{s}{\sqrt{n}}.
$$

Each $x_i$ is one seed's complete benchmark score. For two seeds, SE is exactly half their absolute difference. These are
not standard deviations or 95% confidence intervals. Use the seed count stated for each result. For CS9 and extraction, the
task aggregate is formed within each seed first, and its SE is taken across seeds; task SEs are never averaged. The error
bars describe seed-to-seed variation only: no test-sampling uncertainty is included, since the harness computes none. For
scale, a task of $n$ items at accuracy $p$ has a sampling SE of $\sqrt{p(1-p)/n}$, about 4.6 points for COPA ($n=100$)
and about 0.4 for MMLU ($n=14{,}042$).

Paired validation-loss contrasts use windows as the sampling unit. For identical scored windows, form
$d_w=\ell_{A,w}-\ell_{B,w}$ and report $\bar d\pm s_d/\sqrt{N}$ with $s_d$ the sample standard deviation (denominator
$N-1$) of the differences, plus the share of windows with $d_w>0$. $N=2{,}000$ at 8K and $N=255$ at 64K. Never substitute
the across-seed SE and never multiply by 1.96. Tools: `scripts/campaign/paired_pair.py` and
`scripts/campaign/paired_analysis.py` (they align window IDs; the printed row also shows 1.96 SE for display, the SE field is
1 SE; non-finite differences are dropped and counted; `--val 64k` reads the 64K artifacts).

**Units.** Validation CE is in nats per token (natural-log `F.cross_entropy`, `memmap_ppl.py:126-127`); it is distinct from the
training loss and is not perplexity ($\exp(\mathrm{CE})$). The result JSONs store accuracies as fractions in [0, 1] (RULER
already in [0, 100]); the paper's tables report percentages, multiplying both the mean and the SE by 100, except where a
caption states that extraction is kept as a fraction. A displayed SE of 0.0 may reflect rounding. Loss plots (a paper
convention, not part of the code) show the first seed with a logarithmic y-axis; tables summarize all seeds.

## 2. Validation cross-entropy

### 8K, the primary metric

- **Corpus.** `nemotron_rs_val.bin`, built from Nemotron-CC v2.1 `High-Quality` and `High-Quality-DQA`, tokenized with the
  Llama-2 tokenizer (`data/stage1_tokenize.py`; the recipe: `data/pretrain_stage1.md`). The held-out set is a per-document Bernoulli(0.02) draw with
  `RandomState(11785)`, and the held-out documents are shuffled (`data/stage1_split.py`). It holds
  807,013,275 tokens in 740,467 documents (counted from the file and its `doc_lengths` sidecar).
- **Packing.** Each document is `[BOS] + text + [EOS]` (`data/stage1_tokenize.py`), and documents are concatenated into one
  stream. All 740,467 held-out documents begin with token 1 and end with token 2.
- **Windows.** The first 2,000 contiguous windows of the stream, not a sample: window $w$ covers token IDs
  $[8192w,\ 8192w+8193)$, so each window uses 8,193 IDs and scores 8,192 predictions (2,000 of the 98,512 available windows,
  16,384,000 predictions; RESULT line `windows=2000 ... tokens=16384000`). Adjacent windows share one boundary token while
  their predicted positions are disjoint. There is no attention mask, `position_ids` are `arange(8192)` in every window, and
  windows cross document boundaries with no position or recurrent-state reset: the same naive packing as training.
- **Loss.** The forward pass runs in bf16; logits are cast to fp32 for the per-token cross-entropy and the per-batch sums,
  and the running totals are fp64. The reported value is the token-weighted mean, total NLL over total predicted tokens.
  One GPU, batch 4 (`src/evaluation/ppl/memmap_ppl.py:105-139`).
- **Per-window artifact** for paired contrasts: `<run>_n2000_perwindow.npy`, a float64 array of shape (2, 2000): row 0 the
  window IDs 0..1999, row 1 each window's mean NLL. Every method is scored on the identical windows.
- The 128-window evaluation written at the end of training (by `train_arm.sh`, `hf/<run>/valce.txt`) is a smoke check,
  not the reported 2,000-window protocol.

### 64K (context-extension results)

- **Corpus.** The 256 held-out windows of the stage-2 corpus, `prolong_64k_llama2/prolong64k_val.bin` (ProLong's
  long-context data re-tokenized with the Llama-2 tokenizer and re-packed into 65,536-token windows; drawn from the part
  of the window permutation that no stage-2 run trains on; `data/pretrain_stage2.md`). The file holds exactly
  256 x 65,536 = 16,777,216 tokens.
- **Windows.** Same code with a sequence length of 65,536 and batch 1 (step 1 of `eval_arm_64k.sh`). Each window needs 65,537
  IDs, so 255 windows are scored, 16,711,680 predictions. The artifact has shape (2, 255).
- The stage-2 chain also re-scores the 8K Nemotron protocol above (2,000 windows) on the extended models.

**References.**

- [Su et al., *Nemotron-CC*, ACL 2025](https://aclanthology.org/2025.acl-long.123/).
- The [Nemotron-CC v2.1 dataset card](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1), which describes the
  `High-Quality` and `High-Quality-DQA` subsets used here, and the
  [Nemotron 3 Nano technical report](https://research.nvidia.com/labs/nemotron/files/NVIDIA-Nemotron-3-Nano-Technical-Report.pdf)
  it asks to cite. The held-out split is this study's own.
- For the 64K results, ProLong: [Gao et al., *How to Train Long-Context Language Models (Effectively)*, 2024](https://arxiv.org/abs/2410.02660).

## 3. RULER-4K and RULER-8K

- **Data.** `ruler-500-llama2` (HF dataset `tturing/ruler-500-llama2` at a pinned revision; `data/prepare_eval.py`
  downloads it): our own generation with the official RULER generators, the `NousResearch/Llama-2-7b-hf` tokenizer, seed 42,
  500 examples per task and length (our tokenizer reproduces every recorded length). It is a **regeneration, not the
  published data**: the five essay-haystack NIAH tasks use Project Gutenberg novels instead of the Paul Graham essays,
  and `qa_2` at 4K comes from a patched generator (`qa_short.py`, not part of this repository, identical to the stock one
  wherever the stock one finishes). Scores are therefore not comparable to published RULER numbers.
- **Lengths.** 4,096 and 8,192 (`ruler_llama2_8k.yaml:8`). Stage 2 uses `ruler_llama2_64k.yaml` at 4,096, 8,192, 16,384,
  32,768 and 65,536; its file also carries `avg/4096` and `avg/8192`, so every RULER number must name its file.
- **Tasks.** 13 (`yaml:11-23`): `niah_single_1`, `niah_single_2`, `niah_single_3`, `niah_multikey_1`, `niah_multikey_2`,
  `niah_multikey_3`, `niah_multivalue`, `niah_multiquery`, `vt`, `cwe`, `fwe`, `qa_1`, `qa_2`, matching the
  [official task configuration](https://github.com/NVIDIA/RULER/blob/main/scripts/synthetic.yaml).
- **Context budget.** The nominal length includes the generation budget: each example's
  tokens(input + answer prefix) + tokens to generate is at most the nominal length (no violations), so prompt lengths vary by
  task. Nothing is truncated at evaluation (`ruler.py:294-295` only asserts prompt length at most 1.1 times the nominal).
- **Generation.** Continuous-batching engine, greedy (`yaml:24-29`: `do_sample` false, temperature 0; argmax of fp32 logits),
  no BOS, no chat template, prompt = input + answer prefix. Budgets, equal to the official ones: niah 128, vt 30, cwe 120,
  fwe 50, qa 32 tokens. The configured newline stops (`ruler.py:302-307`) never fire, because the Llama-2 tokenizer encodes
  a standalone `"\n"` as [29871, 13] while a generated newline is the bare token 13; generation ends at EOS (id 2; the
  engine's fallback EOS set also holds id 11, since the export config carries no `eos_token_id`: `_FALLBACK_EOS` and
  `_eos_set` in `src/inference/engine.py:31-40`) or at the budget, and the whole continuation is scored. This matches
  official RULER, which uses no stop words.
- **Scoring.** `src/evaluation/ruler/eval_constants.py` is byte-identical to the
  [official scorer constants](https://github.com/NVIDIA/RULER/blob/main/scripts/eval/synthetic/constants.py), and the
  prediction post-processing (`scorer.py:30-38`: strip, replace control characters with newlines, strip) matches the
  official one. Retrieval (NIAH), variable-tracking and word-extraction tasks use `string_match_all`, the fraction of the
  required references found as case-insensitive substrings; the QA tasks use `string_match_part`, one if any accepted
  reference is found. Do not describe RULER as whole-string exact match.
- **Aggregation.** Each task score is 100 times the mean over its 500 samples, rounded to 2 decimals; `avg/<length>` is the
  unweighted mean of the 13 task scores (`ruler.py:378-379,382`). With 500 samples in every task this equals sample weighting up
  to that rounding.
- **Failures.** A missing prediction or task marker aborts the whole evaluation (`ruler.py:352,360`); an empty prediction
  stays in the denominator, scores 0 and is counted in the task's `nulls`.
- **Results.** `ruler_llama2_8k/ruler_llama2_8k_results.json`: `avg/4096` and `avg/8192` (the reported numbers), `avg/full`
  (the mean over every length and task cell: not a RULER-4K/8K number), `eval_time/<length>`, and per-task entries at the
  top level as `"<length>/<task>": {score, nulls}`; the `verbose` field is empty.

**Reference.** [Hsieh et al., *RULER: What's the Real Context Size of Your Long-Context Language Models?*, 2024](https://arxiv.org/abs/2404.06654).

## 4. Extraction recall

The suite is SWDE, FDA and SQuAD completion from the Based evaluation suite, run as the installed harness 0.4.12 tasks
`swde`, `fda` and `squad_completion` (`recall_extraction.yaml:10-16`; no local task copies) on the generation engine.

| Task | Dataset (revision); split; items | Stop | Generation budget |
|---|---|---|---|
| `swde` | [`hazyresearch/based-swde-v2`](https://huggingface.co/datasets/hazyresearch/based-swde-v2) (27f0223028a5); validation; 1,111 | newline | 48 tokens |
| `fda` | [`hazyresearch/based-fda`](https://huggingface.co/datasets/hazyresearch/based-fda) (42569d301e12); validation; 1,102 | newline | 48 tokens |
| `squad_completion` | [`hazyresearch/based-squad`](https://huggingface.co/datasets/hazyresearch/based-squad) (d152c14c2d7d); validation; 2,984 | blank line (`"\n\n"`) | 256 tokens |

- **Decoding.** Zero-shot, greedy continuation, no BOS; EOS also stops every task (`LMEvalWrapper._prepare_batch_for_generation`);
  stop strings are applied to the decoded text (`LMEvalWrapper.generate_until`). The SQuAD budget of 256 and its `"\n\n"` stop are the harness 0.4.12
  defaults for this task, not a project override (the task code's 48-token fallback is never used in 0.4.12).
- **Truncation.** Prompts are left-truncated to 7,936 = 8,192 - 256 tokens (`LMEvalWrapper._get_encoded_requests_for_generation`); this never binds, the
  longest prompts being 2,291 (SWDE), 2,770 (FDA) and 961 (SQuAD) tokens.
- **Task version.** Our runs use task **version 0** (recorded in each run's full harness output), which does not strip
  whitespace from prompts or targets. The upstream version 1 does and is not comparable; do not mix the two.
- **Metric.** `contains`: one if the gold value (`doc["value"]`) occurs in the generated continuation as a case-insensitive
  literal substring (`re.search(re.escape(label), re.IGNORECASE)`), with no other normalization. This is not token F1 and not
  SQuAD exact match, and it tests only the one supplied value, not every SQuAD alias. SQuAD is lenient: its generations
  average 166 tokens, and the match searches the whole span.
- **Aggregate.** The unweighted mean of the three task means (`avg` in `LMHarnessEvaluator.evaluate`, `src/evaluation/lm_eval_harness/evaluator.py`), not sample-weighted;
  e.g. (0.83348 + 0.80399 + 0.48090) / 3 = 0.70612.
- **Results.** `recall_extraction/recall_extraction_results.json`: `avg`, and `verbose` with `swde`, `fda`,
  `squad_completion` (fractions, the harness key `contains,none`).

**References.**

- Suite and task adaptation: [Arora et al., *Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff*, 2024](https://arxiv.org/abs/2402.18668).
- Expanded SWDE formulation: [Lockard et al., *OpenCeres*, 2019](https://aclanthology.org/N19-1309/). This is not the original introduction of the SWDE corpus, which OpenCeres attributes to Hao et al. (2011).
- FDA source: [Arora et al., *Language Models Enable Simple Systems for Generating Structured Views of Heterogeneous Data Lakes*, 2023](https://arxiv.org/abs/2304.09433).
- SQuAD source: [Rajpurkar, Jia, and Liang, *Know What You Don't Know: Unanswerable Questions for SQuAD*, 2018](https://aclanthology.org/P18-2124/). The Based dataset card recommends this SQuAD 2.0 reference; the completion task still differs from standard SQuAD 2 evaluation.

## 5. CS9 and its nine-task breakdown

CS9 is our aggregate: the unweighted mean of nine zero-shot task accuracies, reported with the nine tasks. Each task runs as
the installed harness 0.4.12 task on its full split (test if the task defines one, else validation), and the reported
metric per task is set by the `metric:` keys of `commonsense.yaml:7-25`. Several tasks emit both `acc` and `acc_norm`, and
the choice matters (e.g. ARC-Easy 0.672 with `acc` vs 0.623 with `acc_norm` on one run).

| Reported task | Task ID; HF dataset | Split; items | Metric | Reference |
|---|---|---|---|---|
| LAMBADA | `lambada_openai`; `EleutherAI/lambada_openai` | test; 5,153 | `acc` | [Paperno et al., ACL 2016](https://aclanthology.org/P16-1144/) |
| HellaSwag | `hellaswag`; `Rowan/hellaswag` | validation; 10,042 | `acc_norm` | [Zellers et al., ACL 2019](https://aclanthology.org/P19-1472/) |
| PIQA | `piqa`; `baber/piqa` | validation; 1,838 | `acc_norm` | [Bisk et al., AAAI 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6239) |
| ARC-Easy | `arc_easy`; `allenai/ai2_arc` ARC-Easy | test; 2,376 | `acc_norm` | [Clark et al., 2018](https://arxiv.org/abs/1803.05457) |
| SciQ | `sciq`; `allenai/sciq` | test; 1,000 | `acc_norm` | [Welbl et al., W-NUT 2017](https://aclanthology.org/W17-4413/) |
| OpenBookQA | `openbookqa`; `allenai/openbookqa` main | test; 500 | `acc_norm` | [Mihaylov et al., EMNLP 2018](https://aclanthology.org/D18-1260/) |
| WinoGrande | `winogrande`; `allenai/winogrande` winogrande_xl | validation; 1,267 | `acc` | Sakaguchi et al., CACM 2021 (the [original conference paper](https://ojs.aaai.org/index.php/AAAI/article/view/6399) is AAAI 2020) |
| COPA | `copa`; `aps/super_glue` copa | validation; 100 | `acc` | [SuperGLUE, 2019](https://arxiv.org/abs/1905.00537); the original task: [Roemmele et al., 2011](https://asgordon.github.io/copa.html) |
| ARC-Challenge | `arc_challenge`; `allenai/ai2_arc` ARC-Challenge | test; 1,172 | `acc_norm` | [Clark et al., 2018](https://arxiv.org/abs/1803.05457) |

Cached dataset revisions: lambada 900124bf, hellaswag 218ec52e, ai2_arc 210d026f, piqa 142f6d73, sciq 2c94ad3e,
openbookqa 388097ea, winogrande 01e74176, copa 3de24cf8.

- **Zero-shot** for all nine (no `num_fewshot` in the config, so the harness uses 0).
- **`acc_norm`** picks the choice with the highest log-likelihood divided by the choice's character length (harness 0.4.12
  `lm_eval/api/task.py` L1494-L1511); it is not per-byte (`acc_bytes`, unused) or per-token normalization. The leading-space
  delimiter is included in the scored log-likelihood but not in the length.
- **LAMBADA** is not multiple choice: `lambada_openai` scores the target `" " + last word` through a likelihood request, and
  `acc` is one when every target token is the model's argmax (`task.py:1461-1466`; greedy check in `LMEvalWrapper._loglikelihood_tokens`),
  on the OpenAI-preprocessed data.
- **Prompts.** SciQ is open-book: its prompt includes the support passage (`"{{support.lstrip()}}\nQuestion: ..."`), with no
  local override. OpenBookQA is closed-book (question stem only). COPA uses the SuperGLUE release.
- **Length.** No truncation: `max_length: 4096` in `commonsense.yaml:3` is only an assertion in the model class
  (`LMEvalWrapper._loglikelihood_tokens`), and the longest CS9 input is 771 tokens (SciQ).
- **Aggregate.** `avg` is the plain mean of the nine selected metrics (`LMHarnessEvaluator.evaluate`), computed per seed from
  unrounded values, so COPA (100 items) weighs as much as HellaSwag (10,042).
- **Ties.** Exactly tied choices under bf16 scoring go to the first option: WinoGrande 66 / 1,267, HellaSwag 286 / 10,042.
- **Results.** `commonsense/commonsense_results.json`: `avg` and `verbose` (the nine tasks, fractions); per-sample
  log-likelihoods in `commonsense/full_results/` and `samples/`.

## 6. MMLU

- **Task.** The harness task `mmlu` (the `default` variant, group version 2), metric `acc` (`mmlu.yaml`), **zero-shot** (no
  shots key; the `dev` few-shot split is unused). `cais/mmlu` (revision c30699e8), test split, all 14,042 questions.
- **Prompt and scoring.** Each prompt begins with "The following are multiple choice questions (with answers) about
  <subject>.", and the answer is chosen by likelihood over the four labels " A" to " D". No truncation (longest input 1,206
  tokens; `max_length: 4096` never reached). Exact ties under bf16 scoring (705 of 14,042 questions) go to the first option.
- **Aggregate.** Question-weighted: the harness group configuration sets `weight_by_size: True`, and our `avg` is the harness
  group value passed through unchanged (`LMHarnessEvaluator.evaluate`); e.g. 0.3472 on one run, where the unweighted mean of the 57
  subjects would be 0.3597. The four groups (STEM 3,153, humanities 4,705, social sciences 3,077, other 3,107 questions) are
  passed through the same way; the paper reports the overall accuracy.
- **Results.** `mmlu/mmlu_results.json`: `avg` and the four groups. The stage-1 chain runs MMLU with the dataset cache
  offline and skips it if `cais___mmlu` is not cached.

**Reference.** [Hendrycks et al., *Measuring Massive Multitask Language Understanding*](https://arxiv.org/abs/2009.03300), 2020 preprint, ICLR 2021.

## 7. TriviaQA

The protocol is fixed by `src/evaluation/configs/triviaqa.yaml` on top of the installed harness 0.4.12 task `triviaqa`
(task version 3.0).

- **Data.** `mandarjoshi/trivia_qa`, configuration `rc.nocontext` (revision 0f7faf33a390), validation split, all 17,944
  questions; closed-book (no evidence documents).
- **Shots.** **5-shot** (`triviaqa.yaml:9`), the only few-shot benchmark in the suite. The examples are drawn at random for
  each question from the train split (138,384 questions) with the harness's `fewshot_seed` 1234, and each example's answer is
  its first alias.
- **Prompt.** `Question: <question>?\nAnswer:` followed by `" " + answer` for the examples, joined by blank lines, no BOS.
  Questions that already end in "?" keep both marks (15,898 of 17,944).
- **Generation.** Generation engine, greedy, at most 32 new tokens (`triviaqa.yaml:8`), stopping at a newline, a period, a
  comma or EOS. Prompts are left-truncated at 4,064 = 4,096 - 32 tokens; this never binds (longest prompt 432 tokens).
- **Scoring.** The `remove_whitespace` filter (strip both ends, then the first response) and `exact_match` with lowercasing
  and removal of ASCII punctuation, then string equality; a response scores one if it equals any of the question's aliases
  (about 15.6 per question). Articles are not removed, so this differs from the
  [original TriviaQA evaluator](https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py),
  which also reports F1. The paper reports exact match; no F1 is computed.
- **Results.** `triviaqa/triviaqa_results.json`: `avg` (= `verbose.triviaqa`), the harness key
  `exact_match,remove_whitespace` as a fraction; the paper reports it times 100.

**Reference.** [Joshi et al., *TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension*, ACL 2017](https://aclanthology.org/P17-1147/).

