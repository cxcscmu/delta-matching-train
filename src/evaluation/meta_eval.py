# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# Modified by the delta-matching-train authors, 2026.

import os
import json
import argparse

from utils import (
    is_master,
    dist_init,
    dist_close,
    dist_barrier,
    build_config,
    get_device,
    master_print,
    get_filename_without_ext,
    check_complete_marker,
)
from evaluation.lm_eval_harness import LMEvalWrapper, LMHarnessEvaluator
from evaluation.ruler import RULEREvaluator, load_model_for_eval
from evaluation.recall import RecallEvaluator
from modeling import HFModel
from evaluation.constants import DEFAULT_EVAL_VERSION_STR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3.5-9B-Base", help="Path to the model configuration or checkpoint.")
    parser.add_argument("--run_name", type=str, default="", help="Run name for wandb")
    parser.add_argument("--eval_config", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "mmlu.yaml"),
                        help="Path to the evaluation configuration file (default: src/evaluation/configs/mmlu.yaml).")
    parser.add_argument("--output_dir", type=str, default="results/eval/", help="Directory to save evaluation results.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--max_length", type=int, default=None, help="Maximum sequence length for evaluation.")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Maximum number of new tokens to generate during evaluation.")
    parser.add_argument("--prefill_chunk_size", type=int, default=None, help="Chunk size for prefill during evaluation.")
    parser.add_argument("--generation_num_chunks", type=int, default=1, help="Number of chunks for generation during evaluation.")
    parser.add_argument("--no_runtime_cache", action="store_true", help="Whether to disable runtime cache during evaluation.")
    parser.add_argument("--no_cache_requests", action="store_true", help="Disable lm-eval's request cache.")
    parser.add_argument("--max_test_num", type=int, default=None, help="Maximum number of tests to run during evaluation.")
    parser.add_argument("--ckpt_step", type=int, default=-1, help="Number of steps trained, -1 for last step")
    parser.add_argument("--version_str", type=str, default=DEFAULT_EVAL_VERSION_STR, help="version string for evaluation")
    parser.add_argument("--add_bos", action="store_true",
                        help="BOS-prompt variant (default off): every prompt starts with the tokenizer's BOS "
                             "(harness: the wrapper's add_prefix; RULER: prepended before left padding). For models trained "
                             "on BOS-anchored windows (--data.prepend_bos). The protocol (docs/benchmark.md) has no BOS.")

    # parse args (an unknown or mistyped flag is an error, not silently ignored)
    args = parser.parse_args()
    VERSION_STR = args.version_str

    # Model-generated code (tasks marked unsafe_code, e.g. LiveCodeBench) runs on this host, not in a sandbox. It runs
    # only with the explicit opt-in DM_ALLOW_CODE_EVAL=1, the only thing that sets HF_ALLOW_CODE_EVAL=1 (the switch of
    # the HF `code_eval` metric) and lm-eval's confirm_run_unsafe_code. Use an isolated node for such tasks.
    allow_code_eval = os.environ.get("DM_ALLOW_CODE_EVAL") == "1"
    if allow_code_eval:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    # setup dist env. For a model too large to fit on one GPU (e.g. Nemotron-H instruct
    # at 262k), EVAL_DEVICE_MAP=auto shards each rank's model across its GPUs via HF
    # device_map (pipeline parallel). This needs one process per data-parallel replica,
    # each started with CUDA_VISIBLE_DEVICES already set to its own GPU slice (e.g. 8 GPUs,
    # 2 replicas: rank 0 sees GPUs 0-3, rank 1 sees GPUs 4-7), set by the launcher BEFORE
    # python starts, e.g. `torchrun --nproc_per_node=2 --no-python <wrapper>` with a wrapper
    # that exports CUDA_VISIBLE_DEVICES from LOCAL_RANK and then execs
    # `python -m evaluation.meta_eval ...`: an in-python reassignment is too late (a CUDA
    # context gets created during import/init and freezes the device set). So each torchrun rank is one
    # data-parallel replica, world_size == dp_size, and the RULER data-split/gather logic
    # below is unchanged. We only bind the NCCL/current device to the first GPU of this
    # rank's slice. True tensor parallel is unavailable for NemotronH in transformers 5.3
    # (no tp_plan; Mamba2/MoE not shardable).
    device_map = os.environ.get("EVAL_DEVICE_MAP") or None
    _shard_device_index = 0 if device_map else None
    # NCCL collective timeout (s). DP replicas finish generate_until at different times; at
    # large budgets (131k) one rank can get a trace that runs the full budget and lag the
    # post-eval gather by >1h. Bump EVAL_DIST_TIMEOUT for those runs (default 1h).
    _dist_timeout = int(os.environ.get("EVAL_DIST_TIMEOUT", "3600"))
    dist_init(gpu=None, cudnn_benchmark=False, timeout=_dist_timeout, device_index=_shard_device_index)

    eval_config = build_config(args.eval_config)

    # Optional length cap for RULER (env EVAL_RULER_MAX_LEN). Used for Nemotron-H, whose
    # mamba_ssm Triton SSD kernel int32-overflows at seqlen >= ~256k (the 262144 bucket);
    # the pure-torch fallback OOMs (O(L*chunk^2)). So cap the HF/device_map sweep at 131072
    # (RULER-128k) and handle 262144 via a different engine (e.g. vLLM) if needed.
    _max_len = os.environ.get("EVAL_RULER_MAX_LEN")
    if _max_len and "lengths" in eval_config:
        _ml = int(_max_len)
        eval_config["lengths"] = [l for l in eval_config["lengths"] if l <= _ml]

    if "lengths" in eval_config:
        master_print(eval_config["lengths"])

    # Short-circuit BEFORE loading the model if this benchmark is already complete,
    # so resuming a suite doesn't load the full model on every rank just to skip
    # (which also avoids extra concurrent HF-cache resolutions that can transiently
    # race across ranks). Mirrors pg19_ppl.py's marker-first pattern.
    _task_abbrev = get_filename_without_ext(args.eval_config)
    _output_dir = os.path.join(args.output_dir, eval_config["name"])
    if is_master():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(_output_dir, exist_ok=True)
    _res_path = os.path.join(_output_dir, f"{_task_abbrev}_results.json")
    dist_barrier()
    if check_complete_marker(_res_path, version_str=VERSION_STR, clean_dir=False):
        master_print('-' * 10 + f"[{_task_abbrev} already evaluated]" + '-' * 10)
        dist_close()
        return

    model, tokenizer = load_model_for_eval(model_name_or_path=args.model_name_or_path, device_map=device_map)

    # Reasoning vs non-reasoning mode. For most chat models we DISABLE thinking by
    # default so lm_eval gets direct answers (not <think>...</think>). For reasoning
    # benchmarks (GPQA/AIME/LiveCodeBench on the Nemotron instruct model) the config
    # sets enable_thinking: true: then we leave the template default intact, the wrapper
    # passes enable_thinking=True explicitly into apply_chat_template, and (strip_reasoning)
    # scores only the text after the last </think>.
    enable_thinking = eval_config.get("enable_thinking", False)
    strip_reasoning = eval_config.get("strip_reasoning", enable_thinking)
    if eval_config.get("apply_chat_template", False) and tokenizer.chat_template and not enable_thinking:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "enable_thinking is defined and enable_thinking is false",
            "enable_thinking is not defined or enable_thinking is false",
        )

    # A device_map-dispatched model is already on its GPUs (do NOT .cuda() it: that
    # collapses every shard onto one device). Otherwise move the single replica to GPU.
    model = model.eval() if device_map else model.eval().cuda()
    master_print(getattr(model, "hf_device_map", None) if device_map else model.device)
    # The recall eval scores model(input_ids).logits via the plain HF .forward path, so it
    # must NOT be wrapped in the Nemotron-H continuous-batching EngineModel (which rebinds
    # the mixers in place and invalidates the forward/loglikelihood path: only .generate
    # remains valid). Keep EVAL_ENGINE out of the recall path; always use HFModel for it.
    if os.environ.get("EVAL_ENGINE", "").lower() == "mini_eval_dev" and eval_config["type"] != "recall":
        # Continuous-batching engine (inference). Reuses the HF weight modules and
        # rebinds the Mamba/attention mixers in place (so the plain HF forward/loglikelihood
        # path is no longer valid: only .generate, which is all generate_until/ruler use).
        from inference import EngineModel
        # For dmlm models the engine only pools mixers it has decode kernels for, and EngineDMLM
        # rebinds each mixer's __class__ AS IT WALKS the layers: so on an unsupported mixer it
        # raises with the model already PARTIALLY rebound, which invalidates the plain HF path
        # too. A try/except around the construction would therefore fall back onto a corrupted
        # model. Validate every dmlm layer up front instead, and only enter the engine when all of
        # them are supported. (mixer_kind knows only the dmlm mixers: other models, e.g.
        # Nemotron-H with its own engine path, go to the engine unchecked.)
        unsupported = ""
        if getattr(model.config, "model_type", None) == "dmlm":
            from inference.dm_cache import mixer_kind
            try:
                for blk in model.model.layers:
                    mixer_kind(blk.mixer)
            except ValueError as e:
                unsupported = str(e)
            except AttributeError:   # unexpected backbone shape: let the engine decide as before
                unsupported = ""
        if unsupported:
            # HFModel is the reference path (same weights, no continuous batching, cache-free
            # generation), so degrading runs the benchmark instead of failing the eval. It left-pads
            # batches, which the recurrent dmlm mixers (Mamba2 / GDN / KDA) cannot mask: for such
            # models its generation paths refuse --eval_batch_size > 1 (see
            # evaluation/ruler/load_model.py::padded_generation_error).
            master_print(f"[engine] {unsupported}; falling back to the plain HF path (cache-free, slower)")
            model = HFModel(model)
        else:
            master_print("[engine] EVAL_ENGINE=mini_eval_dev -> inference.EngineModel (continuous batching)")
            model = EngineModel(model)
    else:
        model = HFModel(model)
    
    if args.add_bos:
        assert tokenizer.bos_token_id is not None, "--add_bos needs a tokenizer with a BOS token"
        master_print(f"[meta_eval] add_bos ON: every prompt starts with BOS id {tokenizer.bos_token_id}")
    # W&B run name (also the seed of the evaluators' run ids): --run_name, else the model directory's own name, so
    # different exports under one parent directory never share a run.
    wandb_log_name = args.run_name or os.path.basename(os.path.normpath(args.model_name_or_path))
    task_abbrev = get_filename_without_ext(args.eval_config)
    output_dir = os.path.join(args.output_dir, eval_config["name"])
    if is_master():
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    res_path = os.path.join(output_dir, f"{task_abbrev}_results.json")
    dist_barrier()
    
    if not check_complete_marker(res_path, version_str=VERSION_STR, clean_dir=False):
        if eval_config["type"] == "lm_eval_harness":
            max_seq_len = args.max_length if args.max_length is not None else eval_config.get("max_length", None)
            assert max_seq_len is not None, "max_length must be specified either in the config or as an argument."
            max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else eval_config.get("max_new_tokens", 256)

            eval_wrapper = LMEvalWrapper(model, tokenizer, max_seq_len=max_seq_len,
                                        max_new_tokens=max_new_tokens,
                                        device=get_device(model),
                                        prefill_chunk_size=args.prefill_chunk_size,
                                        generation_num_chunks=args.generation_num_chunks,
                                        batch_size=args.eval_batch_size,
                                        use_runtime_cache=not args.no_runtime_cache,
                                        add_prefix=args.add_bos,
                                        enable_thinking=enable_thinking,
                                        strip_reasoning=strip_reasoning)

            evaluator = LMHarnessEvaluator(eval_wrapper,
                                        eval_config["tasks"],
                                        subtasks=eval_config.get("subtasks", None),
                                        cache_requests=not args.no_cache_requests,
                                        max_test_num=args.max_test_num,
                                        apply_chat_template=eval_config.get("apply_chat_template", False),
                                        fewshot_as_multiturn=eval_config.get("fewshot_as_multiturn", False),
                                        system_instruction=eval_config.get("system_instruction", None),
                                        keys_to_remove_in_save=eval_config.get("keys_to_remove_in_save", None),
                                        include_path=eval_config.get("include_path", None),
                                        wandb_log_name=f"{wandb_log_name}_full",
                                        version_str=VERSION_STR,
                                        num_fewshot=eval_config.get("num_fewshot", None),
                                        allow_code_eval=allow_code_eval)

        elif eval_config["type"] == "ruler":
            evaluator = RULEREvaluator(model, tokenizer, eval_config,
                                    batch_size=args.eval_batch_size,
                                    prefill_chunk_size=args.prefill_chunk_size,
                                    res_path=res_path,
                                    wandb_log_name=wandb_log_name,
                                    version_str=VERSION_STR,
                                    add_bos=args.add_bos)
        elif eval_config["type"] == "recall":
            if args.add_bos:   # token-space sequences, no tokenizer path to prefix: refuse rather than mislabel the result
                raise ValueError("--add_bos is not supported for the recall evaluators (token-space MQAR / phonebook)")
            # Synthetic-KV / MQAR + phonebook in-context recall. Plain HF
            # .forward path (model wrapped as HFModel above; EngineModel excluded for
            # recall). model_dir is passed so the optional wandb hook can resume the
            # train run via {model_dir}/wandb_run_id.txt (mirrors memmap_ppl).
            evaluator = RecallEvaluator(model, tokenizer, eval_config,
                                    batch_size=args.eval_batch_size,
                                    res_path=res_path,
                                    wandb_log_name=wandb_log_name,
                                    version_str=VERSION_STR,
                                    model_dir=args.model_name_or_path)
        else:
            raise ValueError(f"Unsupported evaluation type: {eval_config['type']}. Supported types are 'lm_eval_harness', 'ruler', and 'recall'.")

        results = evaluator.evaluate(full_save_dir=os.path.join(output_dir, "full_results"))
        if is_master():
            master_print(f"Evaluation results:{json.dumps(results, indent=4)}")
            with open(res_path, "w") as f:
                json.dump(results, f, indent=4)
            # writes the completion marker first, then the optional W&B logging
            evaluator.finish_and_log(json_file_path=res_path, task_abbrev=task_abbrev, train_step=args.ckpt_step)
    else:
        master_print('-'*10+f"[{task_abbrev} already evaluated for {wandb_log_name}]"+'-'*10)
    dist_close()


if __name__ == "__main__":
    main()
