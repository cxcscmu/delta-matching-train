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
#
# Portions of this file (the tok_encode body and its comments, the loglikelihood and loglikelihood_rolling
# docstrings, and comments of the generation loop) are adapted from lm-evaluation-harness
# (https://github.com/EleutherAI/lm-evaluation-harness, lm_eval/models/huggingface.py and lm_eval/api/model.py),
# distributed under the MIT License:
#
# MIT License
#
# Copyright (c) 2020 EleutherAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
import pandas as pd

from tqdm import tqdm
from typing import Any, Optional
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn.functional as F

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.utils import get_rolling_token_windows, make_disjoint_window

from transformers import PreTrainedTokenizer

from modeling import HFModel
from utils import (
    chunk_list_by_size, 
    get_device, 
    is_master,
    get_dist_size,
    get_dist_rank,
)


pd.set_option("future.no_silent_downcasting", True)


class LMEvalWrapper(LM):
    def __init__(
        self,
        model: HFModel,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int,
        batch_size: int = 1,
        max_new_tokens: Optional[int] = None,
        device: torch.device = "cuda",
        use_runtime_cache: bool = False,
        prefill_chunk_size: Optional[int] = None, # chunk_prefill
        generation_num_chunks: int = 1, # chunk_generation (like continuous batching)
        add_prefix: bool = False,
        disable_tqdm: bool = False,
        enable_thinking: bool = False,   # reasoning models: render chat template with thinking ON
        strip_reasoning: bool = False,   # reasoning models: score only text after the last </think>
    ) -> None:
        super().__init__()

        self.model = model
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.strip_reasoning = strip_reasoning
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.generation_num_chunks = generation_num_chunks
        self.batch_size = batch_size
        self.prefill_chunk_size = prefill_chunk_size

        self._world_size = get_dist_size()
        self._rank = get_dist_rank()
        # NOTE: lm_eval's LM base class exposes `device` as a getter-only @property, so we must NOT
        # assign self.device (raises AttributeError). Keep the ctor hint privately; the `device`
        # property below returns the model's actual device (what the rest of this class uses).
        self._device = device

        self.add_prefix = add_prefix

        self.disable_tqdm = disable_tqdm

        self.use_runtime_cache = use_runtime_cache
        self.loglikelihood_cache = None
        self.generation_cache = None
        # Smallest per-rank instance count of any task (from lm-eval's all_gather; None until one is seen). The
        # work-stealing path runs only when it is > 0, i.e. every rank calls generate_until (see generate_until).
        self._min_rank_instances: Optional[int] = None

        if use_runtime_cache and is_master():
            print("Using runtime cache for LM evaluation")

    def tok_encode(self, string: str, left_truncate_len: Optional[int] = None, 
                   add_special_tokens: Optional[bool] = None, add_prefix: Optional[bool] = None) -> list[int]:
        add_prefix = add_prefix if add_prefix is not None else self.add_prefix
        # default for None - empty dict, use predefined tokenizer param
        # used for all models except for CausalLM or predefined value
        special_tokens_kwargs = {}

        # by default for CausalLM - false or self.add_prefix is set
        if add_special_tokens is None:
            special_tokens_kwargs = {"add_special_tokens": False}
        else:
            special_tokens_kwargs = {"add_special_tokens": add_special_tokens}

        encoding = self.tokenizer.encode(string, **special_tokens_kwargs)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            if add_prefix:
                encoding = encoding[-left_truncate_len+1:]
            else:
                encoding = encoding[-left_truncate_len:]
        
        if add_prefix:
            encoding = [self.tokenizer.bos_token_id] + encoding

        if left_truncate_len:
            assert len(encoding) <= left_truncate_len, (len(encoding), left_truncate_len)
        return encoding

    def tok_decode(self, tokens: list[int], skip_special_tokens: bool = True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    @property
    def prefix_token_id(self) -> int:
        # The token every loglikelihood_rolling window is conditioned on: BOS, falling back to EOS (lm-eval HFLM's rule).
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def device(self):
        # Override lm_eval.api.model.LM's getter-only `device` property: return the model's actual
        # device. The rest of this class uses get_device(self.model) directly; this keeps any lm_eval
        # internals that query `lm.device` consistent with it.
        return get_device(self.model)

    def all_gather(self, tensor):
        # lm_eval (evaluator.py) all-gathers a scalar per-rank instance count and expects a
        # [world_size] tensor back (it does .tolist() then max()/[rank]). lm_eval's base LM.all_gather
        # is a NO-OP (returns the input): OK at world_size==1, but at DP>1 it yields the unchanged
        # scalar -> `TypeError: 'int' object is not iterable`. Do a real NCCL all_gather.
        import torch.distributed as dist
        t = tensor if tensor.dim() > 0 else tensor.reshape(1)
        if self._world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
            return t
        out = [torch.empty_like(t) for _ in range(self._world_size)]
        dist.all_gather(out, t.contiguous())
        gathered = torch.cat(out)
        # lm_eval all-gathers exactly one tensor per task: every rank's instance count for it. Keep the smallest count
        # seen: 0 means some rank owns no documents of a task and may never call generate_until.
        n_min = int(gathered.min())
        self._min_rank_instances = n_min if self._min_rank_instances is None else min(self._min_rank_instances, n_min)
        return gathered

    def gather_object(self, obj, dst=0):
        # lm_eval gathers each rank's sample/metric list then `itertools.chain.from_iterable`s them on
        # RANK 0, so this must return the PER-RANK list of objects. Base LM.gather_object returns [obj]
        # (this rank only) -> silently drops other ranks' results at DP>1. all_gather_object is fine
        # (lm_eval only reads the result on RANK 0).
        import torch.distributed as dist
        if self._world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
            return [obj]
        gathered = [None] * self._world_size
        dist.all_gather_object(gathered, obj)
        return gathered

    def _loglikelihood_tokens(
        self, requests: list[tuple[tuple[str, str], list[int], list[int]]]
    ) -> list[tuple[float, bool]]:
        total_requests_num = len(requests)
        sorted_indices, requests = zip(*sorted(enumerate(requests), key=lambda x: len(x[1][1] + x[1][2]), reverse=True))

        batched_requests = chunk_list_by_size(requests, self.batch_size)

        device = get_device(self.model)
        response1 = []
        response2 = []
        for batch in tqdm(
            batched_requests,
            total=len(batched_requests),
            disable=self.disable_tqdm or (self.rank != 0),
            desc="Running requests",
        ):
            _, batched_context_enc, batched_continuation_enc = zip(*batch)
            batched_all_enc = [
                context_enc + continuation_enc[:-1]
                for context_enc, continuation_enc in zip(batched_context_enc, batched_continuation_enc)
            ]
            assert all(len(enc) <= self.max_seq_len for enc in batched_all_enc)
            max_len = max(len(enc) for enc in batched_all_enc)
            batched_all_enc = [enc + [self.tokenizer.pad_token_id] * (max_len - len(enc)) for enc in batched_all_enc]
            batched_inp = torch.tensor(batched_all_enc, dtype=torch.long, device=device)
                        
            logits = self.model(batched_inp, attention_mask=None)["logits"]
            batched_logprob = F.log_softmax(logits, dim=-1)          
            for logprob, context_enc, continuation_enc in zip(
                batched_logprob, batched_context_enc, batched_continuation_enc
            ):
                logprob = logprob[len(context_enc) - 1 : len(context_enc) + len(continuation_enc) - 1]
                greedy_tokens = logprob.argmax(dim=-1)
                max_equal = greedy_tokens.tolist() == continuation_enc
                logprob = [logprob[i][continuation_enc[i]] for i in range(len(continuation_enc))]
                                
                response1.append(float(sum(logprob)))
                response2.append(bool(max_equal))

        # pack
        response = list(zip(response1, response2))[:total_requests_num]
        _response = sorted(zip(sorted_indices, response), key=lambda x: x[0])
        response = [x[1] for x in _response]
        return response

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute log-likelihood of generating a continuation from a context.
        Downstream tasks should attempt to use loglikelihood instead of other
        LM calls whenever possible.

        :param requests: list[Instance]
            A list of Instance objects, with property `args` which returns a tuple (context, continuation).
            `context: str`
                Context string. Implementations of LM must be able to handle an
                empty context string.
            `continuation: str`
                The continuation over which log likelihood will be calculated. If
                there is a word boundary, the space should be in the continuation.
                For example, context="hello" continuation=" world" is correct.

        :return: list[tuple[float, bool]]
            A list of pairs (logprob, isgreedy)
            `logprob: float`
                The log probability of `continuation`.
            `isgreedy`:
                Whether `continuation` would be generated by greedy sampling from `context`.
        """
        new_reqs = []
        if self.use_runtime_cache and self.loglikelihood_cache is not None:
            new_reqs = self.loglikelihood_cache
        else:
            for context, continuation in [req.args for req in requests]:
                continuation = str(continuation)
                                
                assert len(context) > 0
                rstrip_context = context.rstrip()
                n_spaces = len(context) - len(rstrip_context)
                if n_spaces > 0:
                    continuation = context[-n_spaces:] + continuation
                    context = context[:-n_spaces]
                    assert context == rstrip_context

                context_enc = [self.tokenizer.bos_token_id] if self.add_prefix else []
                context_enc += self.tokenizer.encode(context, add_special_tokens=False)
                full_enc = [self.tokenizer.bos_token_id] if self.add_prefix else []
                full_enc += self.tokenizer.encode(context + continuation, add_special_tokens=False)
                continuation_enc = full_enc[len(context_enc) :]

                new_reqs.append(((context, continuation), context_enc, continuation_enc))

        if self.use_runtime_cache and self.loglikelihood_cache is None:
            self.loglikelihood_cache = new_reqs

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        """Compute full log-likelihood of a string, with no truncation, for perplexity computation
        - We will use the full max context length of the model.
        - For inputs that exceed the max context length, we divide the tokenized string into chunks of up to
        the max context length.
        - IMPORTANT: Each document's loglikelihood/perplexity is computed *separately*, unlike other implementations
          which may simply concatenate multiple documents together.
        - IMPORTANT: We maximize the amount of context for each prediction. Specifically, for inputs that we break into
          multiple chunks, the last input will still a full-sized context.
          Example:
            Input tokens: [ 0 1 2 3 4 5 6 7 8 9 ]
            Prefix: BOS/EOS
            Max context length: 4
            Resulting input/prediction pairs:

                INPUT:  BOS   0   1   2
                PRED:     0   1   2   3

                INPUT:    3   4   5   6
                PRED:     4   5   6   7

                INPUT:    5   6   7   8
                PRED:             8   9

          Observe that:
            1. Each token is predicted exactly once
            2. For the last pair, we provide the full context, but only score the last two tokens

        :param requests: list[Instance]
            A list of Instance objects with property `args` which returns a tuple (context,).
            string: str
                String for which we are computing overall loglikelihood
        :return: list[tuple[float]]
            A list of tuples (logprob,)
            logprob: float
                The log probability of `context` conditioned on the BOS/EOS token
                (`self.prefix_token_id`: BOS, falling back to EOS).
        """
        loglikelihoods = []

        for (string,) in [req.args for req in requests]:
            rolling_token_windows = list(
                map(
                    make_disjoint_window,
                    get_rolling_token_windows(
                        token_list=self.tokenizer.encode(string, add_special_tokens=False),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_seq_len,
                        context_len=1,
                    ),
                )
            )
            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            string_nll = self._loglikelihood_tokens(rolling_token_windows)
            string_nll = [x[0] for x in string_nll]
            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def _get_encoded_requests_for_generation(self, requests: list[Instance]) -> list[int, tuple[list[int], dict]]:
        context_encs = [self.tok_encode(req.args[0], left_truncate_len=self.max_seq_len-self.max_new_tokens) for req in requests]
        prompt_lens = [len(enc) for enc in context_encs]
        kwargs = [req.args[1] for req in requests]
        for kwarg in kwargs:
            if "max_gen_toks" in kwarg:
                kwarg["max_new_tokens"] = kwarg.pop("max_gen_toks")
                
        ids = list(range(len(requests)))
        return list(zip(ids, prompt_lens, context_encs, kwargs))

    def _prepare_batch_for_generation(self, batch: list[tuple[int, int, list[int], dict]]) -> dict[str, Any]:
        cids, prompt_lens, context_encs, kwargs_origin = zip(*batch)

        pad_to_len = max(map(len, context_encs))
        attn_masks = [[0] * (pad_to_len - len(enc)) +  [1] * len(enc) for enc in context_encs]
        attn_masks = torch.tensor(attn_masks, dtype=torch.long)
        context_encs = [[self.tokenizer.pad_token_id] * (pad_to_len - len(enc)) + enc for enc in context_encs]
        context_encs = torch.tensor(context_encs, dtype=torch.long)

        # we assume all gen kwargs in the batch are the same
        # this is safe to assume because the `grouper` object ensures it.
        kwargs_origin = kwargs_origin[0]
        # unpack our keyword arguments.
        kwargs = deepcopy(kwargs_origin)  # edge case for repeats > 1
        # add EOS token to stop sequences
        if "max_new_tokens" not in kwargs.keys():
            assert self.max_new_tokens is not None, "max_new_tokens must be specified either in the config or as an argument."
            kwargs["max_new_tokens"] = self.max_new_tokens
        else:
            if self.max_new_tokens is not None:
                assert kwargs["max_new_tokens"] <= self.max_new_tokens, (f"max_new_tokens in kwargs ({kwargs['max_new_tokens']}) is greater than max_new_tokens in config ({self.max_new_tokens})")
            assert kwargs["max_new_tokens"] <= self.max_seq_len, (f"max_new_tokens in kwargs ({kwargs['max_new_tokens']}) is greater than max_seq_len in config ({self.max_seq_len})")

        max_ctx_len = self.max_seq_len - kwargs["max_new_tokens"]
        assert context_encs.shape[1] <= max_ctx_len, (context_encs.shape[1], max_ctx_len)
        assert all(l <= max_ctx_len for l in prompt_lens), (max_ctx_len, prompt_lens)
        
        eos = self.tok_decode(self.tokenizer.eos_token_id, skip_special_tokens=False)
        until = handle_stop_sequences(kwargs.pop("until", None), eos=eos)
        until_tokens = [self.tok_encode(x, add_prefix=False) for x in until]
        
        device = get_device(self.model)
        context_encs = context_encs.to(device)
        attn_masks = attn_masks.to(device)

        return {
            "cids": cids,
            "prompt_lens": prompt_lens,
            "context_encs": context_encs,
            "attn_masks": attn_masks,
            "until": until,
            "until_tokens": until_tokens,
            "kwargs": kwargs,
        }

    def _make_batched_requests(self, requests: tuple[list[int], dict], batch_size: int) -> list[dict[str, Any]]:
        groups = defaultdict(list)
        
        for cid, prompt_len, context_enc, kwargs in requests:
            key = json.dumps(kwargs, sort_keys=True)
            groups[key].append((cid, prompt_len, context_enc, kwargs))
        
        batched_requests = []
        for group in groups.values():
            batched_requests.extend(chunk_list_by_size(group, batch_size))
        
        batched_requests = [self._prepare_batch_for_generation(batch) for batch in batched_requests]

        return batched_requests

    def _generate_until_stealing(self, requests: list[Instance]) -> list[str]:
        """Cross-rank WORK-STEALING generate_until for high-variance, high-repeat evals (AIME).

        lm_eval shards DOCS round-robin, so a rank owning hard docs grinds all their long repeats
        while other ranks idle (the AIME straggler tail). Here every rank gathers the GLOBAL set
        of encoded prompts, then a shared torch.distributed Store atomic counter hands out
        instances dynamically: a rank that empties its slots STEALS the next global instance, so
        all ranks finish together. Results are routed back to the lm_eval-assigned owner so the
        normal gather/scoring is unchanged. No NCCL collective runs during generation (the counter
        is a TCP Store), only a prompt all-gather before and a result all-gather after. Sampled
        output is not token-identical to the static path (different rank/RNG): fine for sampled
        reasoning evals, same as continuous batching already is. Gated by EVAL_WORK_STEAL=1, and used
        only when every rank owns requests (generate_until checks it): the all-gathers need every rank.
        """
        import torch.distributed as dist
        rank, world = self._rank, self._world_size

        # 1. Encode THIS rank's owned shard + pull the (uniform) gen kwargs.
        enc = self._get_encoded_requests_for_generation(requests)  # (cid, plen, ids, kwargs)
        if not enc:
            # Unreachable through generate_until (it steals only when every rank owns requests). The gen kwargs come
            # from this rank's own requests, so a rank without any could not generate stolen instances correctly.
            raise RuntimeError("work stealing needs generate_until requests on every rank")
        k0 = deepcopy(enc[0][3])
        max_new = int(k0.get("max_new_tokens", self.max_new_tokens))
        do_sample = bool(k0.get("do_sample", False))
        temperature = float(k0.get("temperature", 1.0))
        top_k = int(k0.get("top_k", 0) or 0)
        top_p = k0.get("top_p", None)
        eos = self.tok_decode(self.tokenizer.eos_token_id, skip_special_tokens=False)
        until = handle_stop_sequences(k0.get("until", None), eos=eos)
        until_tokens = [self.tok_encode(x, add_prefix=False) for x in until]

        # 2. Gather the GLOBAL prompt pool (deterministic global ids = rank-major concat order).
        my_prompts = [ids for (_c, _p, ids, _k) in enc]
        gathered = [None] * world
        dist.all_gather_object(gathered, my_prompts)
        global_prompts, my_localidx_to_gid = {}, {}
        gid = 0
        for r in range(world):
            for local_i, ids in enumerate(gathered[r] or []):
                global_prompts[gid] = ids
                if r == rank:
                    my_localidx_to_gid[local_i] = gid
                gid += 1
        N = gid

        # 3. Distributed atomic work counter (Store.add is atomic across ranks; no NCCL per step).
        self._ws_call_id = getattr(self, "_ws_call_id", 0) + 1
        key = f"mini_eval_dev_ws_ctr_{self._ws_call_id}"
        store = dist.distributed_c10d._get_default_store()
        def claim_fn():
            g = store.add(key, 1) - 1   # atomic; first call -> 0
            return g if g < N else None

        max_prompt_len = max((len(p) for p in global_prompts.values()), default=1)
        if is_master():
            print(f"[work-steal] N={N} global instances across world={world} | "
                  f"max_new={max_new} max_prompt={max_prompt_len} do_sample={do_sample} "
                  f"temp={temperature} top_p={top_p}", flush=True)

        # 4. Steal-driven generation: returns done Requests (rid=gid) THIS rank processed.
        done = self.model.generate_stealing(
            global_prompts, claim_fn, max_new_tokens=max_new, max_prompt_len=max_prompt_len,
            do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p,
            stop_token_list=until_tokens)

        # 5. Decode + stop-split + </think>-strip each processed instance -> {gid: answer}.
        local_ans = {}
        for req in done:
            s = self.tok_decode(req.gen)
            for term in until:
                if len(term) > 0:
                    s = s.split(term)[0]
            if self.strip_reasoning and "</think>" in s:
                s = s.rsplit("</think>", 1)[-1].lstrip()
            local_ans[req.rid] = s

        # 6. Route results to owners: all-gather {gid:answer}, merge.
        all_ans = [None] * world
        dist.all_gather_object(all_ans, local_ans)
        merged = {}
        for d in all_ans:
            if d:
                merged.update(d)

        # 7. Return THIS rank's owned answers in original request order.
        res = [""] * len(requests)
        for local_i in range(len(requests)):
            res[local_i] = merged.get(my_localidx_to_gid.get(local_i), "")
        return res

    def generate_until(self, requests: list[Instance], disable_tqdm: bool = False) -> list[str]:
        original_len = len(requests)
        if (os.environ.get("EVAL_WORK_STEAL") == "1" and self._world_size > 1
                and hasattr(self.model, "generate_stealing")):
            # lm_eval calls generate_until only on ranks that own requests, while work stealing all-gathers across
            # every rank. Steal only when lm_eval's per-task counts (recorded in all_gather) show that every rank
            # owns documents of every task; otherwise the static path below, which needs no collective.
            if self._min_rank_instances:
                return self._generate_until_stealing(requests)
            if is_master():
                print("[work-steal] not every rank owns requests (or the per-rank counts are unknown): "
                      "using the static path", flush=True)
        # The static path batches left-padded prompts. On the plain HF path a dmlm model with recurrent mixers cannot
        # mask that padding, so batch sizes > 1 are refused there (the engine is not affected).
        from evaluation.ruler.load_model import padded_generation_error
        err = padded_generation_error(self.model, self.batch_size)
        if err:
            raise ValueError(err)
        requests = self._get_encoded_requests_for_generation(requests)
        requests = sorted(requests, key=lambda x: x[1], reverse=True)
        
        if self.use_runtime_cache and self.generation_cache is not None:
            batched_requests = self.generation_cache
        else:
            batched_requests = self._make_batched_requests(requests, self.batch_size)
            if self.use_runtime_cache and self.generation_cache is None:
                self.generation_cache = batched_requests
        
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )

        res = []

        # ENV-GATED per-sequence generation-length logger. Pure no-op unless
        # EVAL_GENLEN_DIR is set. Never raises into the eval (warn once, continue).
        _genlen_dir = os.environ.get("EVAL_GENLEN_DIR")
        _genlen_path = None
        if _genlen_dir:
            try:
                os.makedirs(_genlen_dir, exist_ok=True)
                _genlen_path = os.path.join(_genlen_dir, f"genlen_rank{self.rank}.jsonl")
            except Exception as e:
                print(f"[genlen-logger] disabled, could not create {_genlen_dir!r}: {e}")
                _genlen_path = None

        assert self.max_new_tokens % self.generation_num_chunks == 0
        max_new_tokens_chunk = self.max_new_tokens // self.generation_num_chunks
        for chunk_id in range(self.generation_num_chunks):
            remained_batch = []
            if is_master():
                print(f"generate for chunk {chunk_id}/{self.generation_num_chunks} num batched_requests: {len(batched_requests)}")
            for batch in batched_requests:
                kwargs_chunk = deepcopy(batch["kwargs"])
                kwargs_chunk["max_new_tokens"] = min(max_new_tokens_chunk, batch["kwargs"]["max_new_tokens"])
                out = self.model.generate(
                    input_ids=batch["context_encs"],
                    attention_mask=batch["attn_masks"],
                    stop_token_list=batch["until_tokens"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict=True,
                    return_unfinished=True,
                    prefill_chunk_size=self.prefill_chunk_size,
                    **kwargs_chunk,
                )
                batch_full_tokens = out["sequences"]
                unfinished = out["unfinished_sequences"]
                
                starts = torch.argmax((batch_full_tokens != self.tokenizer.pad_token_id).long(), dim=-1)
                full_tokens_list = batch_full_tokens.tolist()
                num_finished = 0
                                
                for cid, prompt_len, full_tokens, start, ufi in zip(
                    batch["cids"], batch["prompt_lens"], full_tokens_list, starts, unfinished):
                    # discard context + left-padding toks if using causal decoder-only LM
                    num_new_tokens = len(full_tokens)-batch["context_encs"].shape[1]
                    if (ufi == 0) or (num_new_tokens >= batch["kwargs"]["max_new_tokens"]):
                        cont_toks = full_tokens[start+prompt_len:]
                        # REAL per-sequence generated length: generated tokens precede the
                        # right-padding, so the first pad marks the end. (num_new_tokens above is
                        # the batch-PADDED width = identical for every row in a batch -> wrong per
                        # sequence when batch_size>1; only used here for the finished-branch gate.)
                        try:
                            _genlen_new = cont_toks.index(self.tokenizer.pad_token_id)
                        except ValueError:
                            _genlen_new = len(cont_toks)   # no pad -> ran the full budget (cap)
                        s = self.tok_decode(cont_toks)

                        # use secondary stop seqs to cut off should-have-been-stopped content post-hoc
                        for term in batch["until"]:
                            if len(term) > 0:
                                s = s.split(term)[0]

                        # Capture whether the model closed its reasoning block BEFORE the
                        # strip below mutates s (genlen logger reuses this, no re-decode).
                        _genlen_terminated = "</think>" in s

                        # Reasoning models emit "<reasoning> </think> answer"; score only the
                        # text after the LAST </think> (matches NeMo parse_reasoning / the
                        # nano_v3 reasoning parser). If </think> is absent (budget exhausted,
                        # think never closed) the answer-extraction filter will fail -> 0, which
                        # is the intended outcome. Gated so non-reasoning evals are untouched.
                        if self.strip_reasoning and "</think>" in s:
                            s = s.rsplit("</think>", 1)[-1].lstrip()

                        res.append((cid, s))
                        num_finished += 1

                        # ENV-GATED genlen log line (no-op when _genlen_path is None).
                        if _genlen_path is not None:
                            try:
                                with open(_genlen_path, "a") as _gf:
                                    _gf.write(json.dumps({
                                        "cid": int(cid),
                                        "new_tokens": int(_genlen_new),               # real per-seq length
                                        "terminated": bool(_genlen_terminated),       # </think> closed (answer-producing)
                                        "hit_cap": bool(int(ufi) == 1),               # engine: ran to budget w/o stop
                                        "answer": s,
                                    }) + "\n")
                            except Exception as e:
                                print(f"[genlen-logger] disabled, write to {_genlen_path!r} failed: {e}")
                                _genlen_path = None
                    else:
                        kwargs_next_chunk = deepcopy(batch["kwargs"])
                        kwargs_next_chunk["max_new_tokens"] = batch["kwargs"]["max_new_tokens"] - max_new_tokens_chunk
                        kwargs_next_chunk["until"] = batch["until"]
                        remained_batch.append((cid, prompt_len, full_tokens[start:], kwargs_next_chunk))
                
                pbar.set_postfix({"size": batch["context_encs"].size()})
                pbar.update(num_finished)
                            
            batched_requests = self._make_batched_requests(remained_batch, self.batch_size)
        
        # reorder this group of results back to original unsorted form
        res = sorted(res, key=lambda x: x[0])
        res = [x[1] for x in res]
        
        assert len(res) == len(requests), (len(res), len(requests))

        original_res = res        
        assert len(original_res) == original_len        
        
        pbar.close()

        return original_res
    
    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        enable_thinking is forwarded to the template (reasoning models render an open
        <think> generation prompt when True; templates that ignore the kwarg are unaffected).
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
            enable_thinking=self.enable_thinking,
        )

        return chat_templated
