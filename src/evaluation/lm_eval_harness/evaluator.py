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

import os, json, hashlib, wandb
import lm_eval
import pandas as pd

from time import time
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.distributed as dist

from evaluation.lm_eval_harness.wrapper import LMEvalWrapper
from utils import (
    is_master,
    val2tuple,
    dist_print, 
    to_uppercase_only_letters,
    create_complete_marker
)
from evaluation.constants import (
    DEFAULT_WANDB_RESULT_LOG_SUFFIX,
    DEFAULT_VERSION_STR,
)


pd.set_option("future.no_silent_downcasting", True)


def _wandb_init(**kwargs):
    """wandb.init for the optional result logging; None when W&B is disabled (WANDB_MODE=disabled or
    WANDB_DISABLED=true) or cannot start (for example no API key): the evaluation then goes on without it."""
    if (os.environ.get("WANDB_MODE", "").lower() == "disabled"
            or os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}):
        return None
    try:
        return wandb.init(**kwargs)
    except Exception as e:
        print(f"[lm-eval] W&B logging skipped: wandb.init failed ({type(e).__name__}: {e})", flush=True)
        return None


def prepare_for_json(obj):
    if isinstance(obj, dict):
        if any(not isinstance(key, str) for key in obj.keys()):
            return {str(k): prepare_for_json(v) for k, v in obj.items()}
        return {k: prepare_for_json(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple, set)):
        return [prepare_for_json(item) for item in obj]
    
    return obj


def save_to_local(results: dict, save_dir: str, keys_to_remove: list[str] = None) -> None:
    dist_print(f"Saving full results to {save_dir}")
    if is_master():
        os.makedirs(save_dir, exist_ok=True)
    
    keys_to_remove = keys_to_remove or []
    
    if "samples" in results:
        samples = results.pop("samples")
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4, default=str)
        sample_save_dir = os.path.join(save_dir, "samples")
        os.makedirs(sample_save_dir, exist_ok=True)
        for key, value in samples.items():
            with open(os.path.join(sample_save_dir, f"{key}.json"), "w") as f:
                value = prepare_for_json(value)
                for inst in value:
                    for k in keys_to_remove:
                        if k in inst:
                            del inst[k]
                json.dump(value, f, indent=4, default=str)


class LMHarnessEvaluator:
    def __init__(self, 
        wrapper: LMEvalWrapper, 
        tasks: list[tuple[str, str]], 
        subtasks: list[tuple[str, str]] = None,
        cache_requests: bool = True,
        max_test_num: Optional[int] = None,
        apply_chat_template: bool = False,
        fewshot_as_multiturn: bool = False,
        system_instruction: Optional[str] = None,
        keys_to_remove_in_save: Optional[str] = None,
        include_path: Optional[str] = None,
        wandb_log_name: Optional[str] = None,
        version_str: Optional[str] = DEFAULT_VERSION_STR,
        num_fewshot: Optional[int] = None,
        allow_code_eval: bool = False,
    ) -> None:

        self.wrapper = wrapper
        self.tasks = tasks
        self.subtasks = subtasks
        self.runtime_cache = {}
        self.cache_requests = cache_requests
        self.max_test_num = max_test_num
        self.apply_chat_template = apply_chat_template
        self.fewshot_as_multiturn = fewshot_as_multiturn
        self.system_instruction = system_instruction
        self.keys_to_remove_in_save = keys_to_remove_in_save
        # Directory of custom lm_eval task YAMLs (gpqa/aime/livecodebench reasoning tasks)
        # resolved via TaskManager(include_path=...) so simple_evaluate can find them.
        self.include_path = include_path
        # An explicit few-shot count for every task of the config (None = the task's own default), e.g. the 5-shot
        # TriviaQA protocol (src/evaluation/configs/triviaqa.yaml); the harness draws the shots from the train split.
        self.num_fewshot = num_fewshot
        # Tasks marked unsafe_code execute model-generated code on this host. lm-eval runs them only when this is True,
        # which meta_eval sets from the explicit opt-in DM_ALLOW_CODE_EVAL=1 (default off).
        self.allow_code_eval = allow_code_eval
        dist_print(f"Tasks:\n{json.dumps(self.tasks, indent=2)}" + (f"\nnum_fewshot: {num_fewshot}" if num_fewshot is not None else ""))
        if self.subtasks is not None:
            dist_print(f"Sub tasks:\n{json.dumps(self.subtasks, indent=2)}")
        self.version_str = version_str
        self.run_name = wandb_log_name or "lm_eval_harness"
        self.run_id = hashlib.sha1((self.run_name+self.version_str).encode("utf-8")).hexdigest()

    def _get_task_metric_score(self, task_name: str, metric_name: str, results: dict) -> float:
        try:
            eval_dict = results[task_name]
        except KeyError:
            raise ValueError(f"Task {task_name} not found in evaluation results, eval_dict keys: {results.keys()}")
        if metric_name not in eval_dict:
            raise ValueError(f"Metric {metric_name} not found in evaluation results for task {task_name}, eval_dict: {eval_dict}")
        return eval_dict[metric_name]

    def evaluate(self, 
                 model: Optional[nn.Module] = None, 
                 full_save_dir: Optional[str] = None, 
                 verbose: bool = True
        ) -> dict[str, Any]:
        
        if model is not None:
            self.wrapper.model = model
        
        is_training = self.wrapper.model.training
        self.wrapper.model.eval()

        dist_print("Starting evaluation with lm-eval...")

        verbose_dict = {}
        subtask_dict = {}
        all_metrics = []

        with torch.no_grad():
            st = time()
            torch.cuda.synchronize()
            task_manager = None
            if self.include_path:
                from lm_eval.tasks import TaskManager
                task_manager = TaskManager(include_path=self.include_path)
            # `return_runtime_cache` + a seeded `**runtime_cache` are features of our PATCHED lm_eval
            # fork (reuse prefill/KV across eval requests); vanilla lm_eval (0.4.x, the installed env)
            # lacks them and errors on the unexpected kwarg. Detect support and degrade gracefully:
            # runtime cache only accelerates repeated generation; MC loglikelihood (commonsense) is fine
            # without it.
            import inspect
            _supports_rtc = "return_runtime_cache" in inspect.signature(lm_eval.simple_evaluate).parameters
            _rtc_kwargs = {}
            if _supports_rtc:
                _rtc_kwargs["return_runtime_cache"] = self.wrapper.use_runtime_cache
                _rtc_kwargs.update(self.runtime_cache)
            try:
                out = lm_eval.simple_evaluate(
                    model=self.wrapper,
                    model_args={"tokenizer": self.wrapper.tokenizer.name_or_path},
                    tasks=[t["name"] for t in self.tasks],
                    verbosity="ERROR",
                    bootstrap_iters=0,
                    cache_requests=self.cache_requests,
                    limit=self.max_test_num,
                    num_fewshot=self.num_fewshot,
                    log_samples=verbose,
                    confirm_run_unsafe_code=self.allow_code_eval,
                    apply_chat_template=self.apply_chat_template,
                    fewshot_as_multiturn=self.fewshot_as_multiturn,
                    system_instruction=self.system_instruction,
                    task_manager=task_manager,
                    **_rtc_kwargs,
                )
            except ValueError as e:
                # lm-eval refuses a task marked unsafe_code unless confirm_run_unsafe_code is set, and the HF code_eval
                # metric refuses to run without HF_ALLOW_CODE_EVAL=1: name our opt-in, which controls both
                if not self.allow_code_eval and ("marked as unsafe" in str(e) or "HF_ALLOW_CODE_EVAL" in str(e)):
                    raise ValueError(
                        "this config has a task that executes model-generated code on this host (not in a sandbox). "
                        "It runs only with the explicit opt-in DM_ALLOW_CODE_EVAL=1; use an isolated node. "
                        f"(lm-eval: {e})") from e
                raise
            torch.cuda.synchronize()
            et = time()
            dist_print(f"lm-eval took {et - st:.2f}s")
            if _supports_rtc and self.wrapper.use_runtime_cache:
                results, self.runtime_cache = out
            else:
                results = out

        if is_master():
            full_results = results
            results = results["results"]
            for task in self.tasks:
                task_name = task["name"]
                metric_name = task["metric"]
                display_name = task.get("display_name", task_name)
                verbose_dict[display_name] = self._get_task_metric_score(task_name, metric_name, results)
                all_metrics.append(verbose_dict[display_name])
                secondary_metric_names = task.get("secondary_metrics", None)
                if secondary_metric_names is not None:
                    for secondary_metric_name in val2tuple(secondary_metric_names):
                        secondary_score = self._get_task_metric_score(task_name, secondary_metric_name, results)
                        verbose_dict[f"{display_name}@{secondary_metric_name}"] = secondary_score
            
            if self.subtasks is not None:
                for task in self.subtasks:
                    task_name = task["name"]
                    metric_name = task["metric"]
                    display_name = task.get("display_name", task_name)
                    subtask_dict[display_name] = self._get_task_metric_score(task_name, metric_name, results)
                    secondary_metric_names = task.get("secondary_metrics", None)
                    if secondary_metric_names is not None:
                        for secondary_metric_name in val2tuple(secondary_metric_names):
                            secondary_score = self._get_task_metric_score(task_name, secondary_metric_name, results)
                            subtask_dict[f"{display_name}@{secondary_metric_name}"] = secondary_score

            if verbose and full_save_dir is not None:
                    save_to_local(full_results, full_save_dir, keys_to_remove=self.keys_to_remove_in_save)

            out = {
                "avg": sum(all_metrics) / len(all_metrics),
                "verbose": verbose_dict,
                "subtasks": subtask_dict,
                "run_id": self.run_id,
            }
        else:
            out = None

        out = [out]
        dist.broadcast_object_list(out, src=0)
        out = out[0]
        self.res_to_log = out

        self.wrapper.model.train(is_training)

        return out
    
    def finish_and_log(self, json_file_path: str ='', 
                       wandb_hash_suffix: str=DEFAULT_WANDB_RESULT_LOG_SUFFIX,
                       task_abbrev: str ="BENCHMARK",
                       train_step: int = -1):
        task_abbrev = to_uppercase_only_letters(task_abbrev)
        if not is_master():
            return
        # The completion marker comes first: W&B logging is optional and never decides whether a finished benchmark
        # counts as done.
        if json_file_path:
            create_complete_marker(json_file_path, version_str=self.version_str)
        # W&B env: entity WANDB_ENTITY (default: the account's), project WANDB_PROJECT (default delta-matching).
        wandb_log_run = _wandb_init(name=self.run_name, id=self.run_id,
                                    resume=os.environ.get("WANDB_RESUME", "allow"),
                                    job_type=os.environ.get("WANDB_RUNTYPE", "eval"),
                                    entity=os.environ.get("WANDB_ENTITY"),
                                    project=os.environ.get("WANDB_PROJECT", "delta-matching"),
                                    dir=os.environ.get("WANDB_DIR", os.environ.get("DM_WORKSPACE", ".")))
        if wandb_log_run is None:
            return
        try:
            if train_step > 0:
                wandb_log_run.log({f"{task_abbrev}_avg": self.res_to_log["avg"], "step": train_step})
            else:
                # Standalone eval (not across training steps): log once so the metric is a
                # single value (renders as a bar in W&B) rather than a flat multi-step line.
                wandb_log_run.log({f"{task_abbrev}_avg": self.res_to_log["avg"]})
            wandb_log_run.summary.update(self.res_to_log)
            if json_file_path:
                wandb_log_run.save(json_file_path)
            wandb_log_run.finish()
        except Exception as e:
            print(f"[lm-eval] W&B logging failed: {type(e).__name__}: {e}", flush=True)
