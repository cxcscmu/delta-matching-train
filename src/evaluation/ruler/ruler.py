import os
import shutil
import yaml, hashlib
import json, wandb
import math, time
import numpy as np
from tqdm import tqdm
from typing import Optional
from copy import deepcopy

import torch.nn as nn
import torch.distributed as dist

from .data_constants import TASKS as TASKS_DATA
from .eval_constants import TASKS as TASKS_EVAL
from .scorer import run_evaluation_per_task
from .load_model import padded_generation_error
from utils import (
    is_master,
    get_dist_rank,
    get_dist_size,
    get_device,
    chunk_list_interleaved,
    locked_atomic_write_json,
    locked_read_json,
    log_and_flush,
    master_print,
    to_uppercase_only_letters,
    create_complete_marker
)
from utils.ckpt import get_complete_marker_path
from evaluation.constants import (
    DEFAULT_WANDB_RESULT_LOG_SUFFIX,
    DEFAULT_VERSION_STR,
)


def encode_left_padded_with_bos(tokenizer, prompts):
    """The BOS-prompt variant of the batch encoding below (meta_eval --add_bos): each prompt becomes
    [bos] + its tokens (add_special_tokens=False, as the default path), then the batch is left-padded with pad_token_id,
    the same layout tokenizer(..., padding_side="left") gives. Returns {"input_ids", "attention_mask"} as long tensors."""
    import torch
    ids = [[tokenizer.bos_token_id] + tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    n = max(len(x) for x in ids)
    pad = tokenizer.pad_token_id
    return {"input_ids": torch.tensor([[pad] * (n - len(x)) + x for x in ids], dtype=torch.long),
            "attention_mask": torch.tensor([[0] * (n - len(x)) + [1] * len(x) for x in ids], dtype=torch.long)}


def _wandb_init(**kwargs):
    """wandb.init for the optional progress / result logging; None when W&B is disabled (WANDB_MODE=disabled or
    WANDB_DISABLED=true) or cannot start (for example no API key): the evaluation then goes on without it."""
    if (os.environ.get("WANDB_MODE", "").lower() == "disabled"
            or os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}):
        return None
    try:
        return wandb.init(**kwargs)
    except Exception as e:
        print(f"[ruler] W&B logging skipped: wandb.init failed ({type(e).__name__}: {e})", flush=True)
        return None


def _fsync_dir(path: str) -> None:
    """Make the directory entries of files just created in `path` durable (best effort)."""
    try:
        fd = os.open(path, os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


class RULEREvaluator:
    def __init__(self,
                 model: nn.Module,
                 tokenizer,
                 eval_config: dict,
                 res_path: str = 'tmp_res.json',
                 wandb_log_name: str = 'ruler_benchmark_result',
                 batch_size: int = 1,
                 prefill_chunk_size: Optional[int] = None,
                 version_str: str = DEFAULT_VERSION_STR,
                 add_bos: bool = False):
        # The plain HF path left-pads batches, which the recurrent dmlm mixers cannot mask: refuse before any work.
        err = padded_generation_error(model, batch_size)
        if err:
            raise ValueError(err)
        self.model = model
        self.tokenizer = tokenizer
        self.add_bos = add_bos   # BOS-prompt variant (meta_eval --add_bos); default = the protocol's no-BOS prompts
        self.batch_size = batch_size
        self.tasks = eval_config["tasks"]
        self.lengths = eval_config["lengths"]
        self.data_dir = eval_config["data_dir"]
        self.prefill_chunk_size = prefill_chunk_size
        self.verbose = eval_config.get("verbose", False)
        self.apply_chat_template = eval_config.get("apply_chat_template", False)
        # Optional cap on samples per (task,length): used by smoke configs to validate
        # the pipeline / long-context fit quickly. None = use all samples in the file.
        self.max_samples = eval_config.get("max_samples", None)

        curr_folder = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(curr_folder, "config_tasks.yaml"), "r") as f:
            self.tasks_customized: dict = yaml.safe_load(f)

        for _, config in self.tasks_customized.items():
            config.update(TASKS_DATA[config['task']])
            config.update(TASKS_EVAL[config['task']])

        self.generation_config = eval_config["generation_config"]
        self.stop = self.generation_config.pop('stop')
        run_name = f"TMP_{wandb_log_name}"
        self.wandb_report_name = f"EVAL_{wandb_log_name}"

        self.version_str = version_str
        self.run_id = hashlib.sha1((run_name+self.version_str).encode("utf-8")).hexdigest()
        self.wandb_run = None
        if is_master():
            print(run_name, self.run_id)
            self.wandb_run = _wandb_init(name=run_name, id=self.run_id, resume="allow",
                                         job_type=os.environ.get("WANDB_RUNTYPE", "eval") + "_tmp",
                                         entity=os.environ.get("WANDB_ENTITY"),
                                         project=os.environ.get("WANDB_PROJECT", "delta-matching"),
                                         dir=os.environ.get("WANDB_DIR", os.environ.get("DM_WORKSPACE", ".")))

        # This config's own artifacts: its results JSON (<cfg>_results.json, with the lock file and completion marker
        # beside it) and its predictions directory <full_save_dir>/<cfg>/ (see evaluate). A fresh run deletes exactly
        # these, never other files of the output directory (other configs' results and predictions).
        self.res_path = res_path
        stem = os.path.splitext(os.path.basename(res_path))[0]
        assert stem, f"res_path must name a results file, got {res_path!r}"
        self.cfg_name = stem[:-len("_results")] if stem.endswith("_results") and stem != "_results" else stem

    def _load_or_reset_results(self, pred_dir: str) -> None:
        """Resume this config's results JSON when it belongs to the same run (same run_id), else start fresh."""
        dist.barrier()
        if is_master():
            prev = locked_read_json(path=self.res_path) if os.path.isfile(self.res_path) else None
            if prev is not None and prev.get('run_id') == self.run_id:
                master_print("[STATUS] Found previous run, loading...")
            else:
                if prev is None:
                    master_print("[STATUS] No previous runs, running new...")
                else:
                    master_print("[STATUS] Found previous run, but run_id mismatch, clear and rerunning...")
                for path in (self.res_path, self.res_path + ".lock", get_complete_marker_path(self.res_path)):
                    if os.path.isfile(path):
                        os.remove(path)
                shutil.rmtree(pred_dir, ignore_errors=True)
                if os.path.exists(pred_dir):   # a failed clear would let this run reuse another run's predictions
                    raise RuntimeError(f"could not clear the old RULER predictions at {pred_dir}; remove it by hand and rerun")
                locked_atomic_write_json(self.res_path, {
                    "avg/full": [],
                    **{f"avg/{l}": [] for l in self.lengths},
                    **{f"eval_time/{l}": [] for l in self.lengths},
                    "verbose": {},
                    'run_id': self.run_id,
                })
        dist.barrier()
        self.full_results = locked_read_json(path=self.res_path)

    def pack_batch(self, data: list, batch_size: int):
        batched_data = []
        batch = []
        for data_point in data:
            if len(batch) >= batch_size:
                batched_data.append(batch)
                batch = []

            batch.append(data_point)

        if len(batch):
            batched_data.append(batch)
            
        return batched_data

    def load_data(self, task: dict, length: int):
        task_name, task_subset = task["name"], task["subset"]
        if task_name not in self.tasks_customized:
            raise ValueError(f'{task} is not found in config_tasks.yaml')
        
        task_file = os.path.join(self.data_dir, str(length), "data", task_name, f'{task_subset}.jsonl')
        
        with open(task_file, 'r') as f:
            data = [json.loads(line) for line in f.readlines()]

        if self.max_samples is not None:
            data = data[:self.max_samples]

        for i, d in enumerate(data):
            d["idx"] = i
            d["task_key"] = f"{length}/{task_name}"
            ap = d.get("answer_prefix")
            if ap:
                # binary-search RULER format (e.g. tturing/ruler-500-llama2): `input` EXCLUDES the
                # trailing answer prefix; the model must be fed input + answer_prefix (dataset README).
                # Exact concat reproduces the original prompt; no strip (the prefix carries its own
                # leading space) and no ':' hack (the prefix already ends with the answer cue).
                d["input"] = d["input"] + ap
            else:
                # classic RULER format: the answer prefix was baked into `input` (NIAH needed a ':').
                d["input"] = d["input"].strip()
                if task_name in ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
                                 "niah_multiquery", "niah_multivalue",
                                 "niah_single_1", "niah_single_2", "niah_single_3"]:
                    d["input"] = d["input"] + ":"
        return data

    def evaluate(
        self, full_save_dir: Optional[str] = None, **kwargs):
        """Generate and score every (length, task). Predictions and per-task markers go to <full_save_dir>/<cfg>/
        (full_save_dir defaults to the full_results directory beside the results JSON, as meta_eval passes it)."""
        if full_save_dir is None:
            full_save_dir = os.path.join(os.path.dirname(self.res_path), "full_results")
        pred_dir = os.path.join(full_save_dir, self.cfg_name)
        self._load_or_reset_results(pred_dir)
        if is_master():
            master_print(f"tasks {self.tasks}")

        self.model.eval()

        device = get_device(self.model)

        for length in self.lengths:
            if isinstance(self.full_results[f"avg/{length}"], list):
                auto_batch_size = max(1, self.batch_size * self.lengths[0] // length)
                self.full_results[f"avg/{length}"] = []
                self.full_results[f"eval_time/{length}"] = []
                if is_master():
                    os.makedirs(os.path.join(pred_dir, str(length)), exist_ok=True)
                    print("auto_batch_size", auto_batch_size)

                for task in self.tasks:
                    task_name = task["name"]
                    task_key = f"{length}/{task_name}"
                    complete_marker_file = os.path.join(pred_dir, f'{task_key}.complete')
                    pred_file = os.path.join(pred_dir, f'{task_key}.jsonl')
                    if os.path.isfile(complete_marker_file):
                        master_print(f"<< {task_key} already processed, skipping {task_key} >>")
                    else:
                        if is_master():
                            print(f"Evaluating task: {task_name}, length: {length}")
                        
                        data = self.load_data(task, length)
                        task_outputs = [
                            {
                                "idx": d["idx"], 
                                "pred": None,
                                "task_key": task_key,
                                "input": d["input"], 
                                "outputs": d["outputs"],
                                "others": d.get('others', {}),
                                "truncation_list": d.get('truncation', -1),
                                "length_list": d.get('length', -1),
                            } for d in data
                        ]

                        pad_to_size = math.ceil(len(data) / get_dist_size()) * get_dist_size()
                        padding_data = [deepcopy(data[0]) for _ in range(pad_to_size - len(data))]
                        for d in padding_data:
                            d["idx"] = -1
                        padded_data = data + padding_data

                        batched_data = self.pack_batch(padded_data, auto_batch_size)
                    
                        batched_data_on_rank = chunk_list_interleaved(batched_data, get_dist_size())[get_dist_rank()]
                                    
                        all_texts = []
                        start_time = time.perf_counter()
                        for batch in tqdm(batched_data_on_rank, desc="Generating", disable=not is_master()):
                            prompts = [d["input"] for d in batch]
                            if self.apply_chat_template:
                                prompts = [self.tokenizer.apply_chat_template(
                                    [{"role": "user", "content": p}],
                                    tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False,
                                ) for p in prompts]
                            if self.add_bos:
                                inputs = encode_left_padded_with_bos(self.tokenizer, prompts)
                            else:
                                inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
                            for k in inputs:
                                inputs[k] = inputs[k].to(device)
                            # The RULER prompts were sized with one tokenizer (the published data: Llama-2);
                            # another tokenizer can make them longer (up to ~+4% measured on variable_tracking
                            # and niah_multikey). Allow 10% headroom (that + any chat-template overhead) so the
                            # harness does not stop on a slightly longer tokenization; the per-length labels stay
                            # approximately correct.
                            length_tolerance = length * 1.10
                            assert inputs["input_ids"].shape[1] <= length_tolerance, f"Input length {inputs['input_ids'].shape[1]} exceeds the set length {length} (tolerance: {length_tolerance})"
                            generated_ids = self.model.generate(
                                **inputs,
                                **self.generation_config,
                                prefill_chunk_size=self.prefill_chunk_size,
                                max_new_tokens=self.tasks_customized[task_name]["tokens_to_generate"],
                                pad_token_id=self.tokenizer.pad_token_id,
                                stop_token_list=[
                                    [self.tokenizer.eos_token_id], 
                                    self.tokenizer.encode("\n", add_special_tokens=False),
                                    self.tokenizer.encode(".\n", add_special_tokens=False),
                                    self.tokenizer.encode("\n\n", add_special_tokens=False),
                                    self.tokenizer.encode(".\n\n", add_special_tokens=False)],
                            )
                            generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
                            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        
                            for text, d in zip(generated_texts, batch):
                                if self.stop is not None:
                                    for s in self.stop:
                                        text = text.split(s)[0]

                                all_texts.append((text, d["idx"]))
                        end_time = time.perf_counter()
                        time_elapsed = end_time - start_time
                        all_texts_gather = [[] for _ in range(get_dist_size())]
                        dist.all_gather_object(all_texts_gather, all_texts)
                        for _texts in all_texts_gather:
                            for text, idx in _texts:
                                if idx >= 0:
                                    task_outputs[idx]["pred"] = text
                                    assert task_outputs[idx]["idx"] == idx
                        
                        if is_master():
                            # predictions first, then the task marker; both made durable (fsync of the two files
                            # and their directory, not a node-wide os.sync)
                            with open(pred_file, 'w') as f:
                                for out in task_outputs:
                                    f.write(json.dumps(out) + "\n")
                                f.write(json.dumps({'time_elapsed': time_elapsed}) + "\n")
                                f.flush()
                                os.fsync(f.fileno())
                            fd = os.open(complete_marker_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                            try:
                                os.fsync(fd)
                            finally:
                                os.close(fd)
                            _fsync_dir(os.path.dirname(complete_marker_file))

                # load every task's predictions back (all ranks) and score them
                dist.barrier()
                outputs = {}
                for task in self.tasks:
                    task_name = task["name"]
                    task_key = f"{length}/{task_name}"
                    complete_marker_file = os.path.join(pred_dir, f'{task_key}.complete')
                    pred_file = os.path.join(pred_dir, f'{task_key}.jsonl')
                    assert os.path.isfile(complete_marker_file), f"Missing prediction for {task_key}"
                    with open(pred_file, "r") as f:
                        outputs[task_key] = [json.loads(line) for line in f]
                    self.full_results[f"eval_time/{length}"].append((outputs[task_key].pop())['time_elapsed'])
                    
                for task_key, output in outputs.items():
                    _, task_name = task_key.split("/")
                    for d in output:
                        assert d["pred"] is not None, f"Missing prediction for {d}"

                    eval_results, subm_results = {}, {}
                    task_score, task_nulls, predicts, indices = run_evaluation_per_task(
                        outputs=outputs[task_key],
                        task_config=self.tasks_customized[task_name],
                    )
                    eval_results[task_key] = {
                        'score': task_score,
                        'nulls': task_nulls,
                    }
                    subm_results[task_key] = {
                        'predicts': predicts,
                        'indices': indices,
                    }
                    
                    if self.verbose:
                        self.full_results.update(eval_results)
                    self.full_results["avg/full"].append(task_score)
                    self.full_results[f"avg/{length}"].append(task_score)
                
                dist.barrier()
                self.full_results[f"avg/{length}"] = np.mean(self.full_results[f"avg/{length}"])
                
                if is_master():
                    self.full_results[f"eval_time/{length}"] = sum(self.full_results[f"eval_time/{length}"])
                    if self.wandb_run is not None:
                        try:
                            log_and_flush(
                                wandb_run=self.wandb_run,
                                length=length,
                                log_dict={"RULER_by_length(%)": self.full_results[f"avg/{length}"],
                                          "n(length)": length,
                                          "eval_time(s)": self.full_results[f"eval_time/{length}"]},
                            )
                        except Exception as e:
                            print(f"[ruler] W&B logging failed: {type(e).__name__}: {e}", flush=True)
                    locked_atomic_write_json(self.res_path, self.full_results)
            else:
                master_print(f"<< Skip Length={length} >>")
        self.full_results["avg/full"] = np.mean(self.full_results["avg/full"])
        if is_master() and self.wandb_run is not None:
            try:
                self.wandb_run.log({"RULER_avg": self.full_results["avg/full"]}, step=length+1)
                self.wandb_run.summary.update(self.full_results)
            except Exception as e:
                print(f"[ruler] W&B logging failed: {type(e).__name__}: {e}", flush=True)
        return self.full_results

    def finish_and_log(self, json_file_path: str = '',
                       wandb_hash_suffix: str = DEFAULT_WANDB_RESULT_LOG_SUFFIX,
                       task_abbrev: str = "RULER",
                       train_step: int = -1):
        task_abbrev = to_uppercase_only_letters(task_abbrev)
        if not is_master():
            return
        # The completion marker comes first: W&B logging is optional and never decides whether a finished benchmark
        # counts as done.
        if json_file_path:
            create_complete_marker(json_file_path, version_str=self.version_str)
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception as e:
                print(f"[ruler] W&B logging failed: {type(e).__name__}: {e}", flush=True)
        finish_log_id = hashlib.sha1((self.wandb_report_name+wandb_hash_suffix+str(self.version_str)).encode("utf-8")).hexdigest()
        wandb_log_run = _wandb_init(name=self.wandb_report_name, id=finish_log_id, resume="allow",
                                    job_type=os.environ.get("WANDB_RUNTYPE", "eval"),
                                    entity=os.environ.get("WANDB_ENTITY"),
                                    project=os.environ.get("WANDB_PROJECT", "delta-matching"),
                                    dir=os.environ.get("WANDB_DIR", os.environ.get("DM_WORKSPACE", ".")))
        if wandb_log_run is None:
            return
        try:
            for length in self.lengths:
                wandb_log_run.log(
                    {f"{task_abbrev}_by_length(%)": self.full_results[f"avg/{length}"],
                     "eval_time(s)": self.full_results[f"eval_time/{length}"]},
                    step=length
                )
            if train_step > 0:
                wandb_log_run.log({f"{task_abbrev}_avg": self.full_results["avg/full"], "step": train_step})
            else:
                # log once (single value / bar in W&B), not a flat multi-step line
                wandb_log_run.log({f"{task_abbrev}_avg": self.full_results["avg/full"]})
            wandb_log_run.summary.update(self.full_results)
            if json_file_path:
                wandb_log_run.save(json_file_path)
            wandb_log_run.finish()
        except Exception as e:
            print(f"[ruler] W&B logging failed: {type(e).__name__}: {e}", flush=True)
