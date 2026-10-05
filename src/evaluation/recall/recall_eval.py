"""Synthetic-KV / MQAR + phonebook in-context recall evaluator for dmlm checkpoints.

Probes a *pretrained* LM's in-context associative recall: put N key->value pairs in the
context, re-query a key, the model must recall its value. Produces a degradation MATRIX over
a grid of ``num_kv_pairs x seq_len``, plus a simpler phonebook single-query variant.

Runs on the plain HF ``.forward`` path (NOT the Nemotron-H continuous-batching
EngineModel): score ``model(input_ids).logits`` argmax against the gold value at the
answer positions. Mirrors `evaluation/ppl/memmap_ppl.py` for DP-shard across ranks
(`chunk_list_interleaved`), all-reduce of correct/total, rank-0-only result write,
and the optional resume-train-run wandb hook; and `evaluation/ruler/ruler.py` for the
``(model, tokenizer, eval_config, batch_size, res_path, wandb_log_name, ...)``
constructor so it drops into `meta_eval.py` exactly like RULER.

The answer-position shift is the crux: `synthetic_kv.make_mqar` writes ``labels`` so
that a non-(-100) entry at index ``p`` is the value the model must produce AT ``p``,
i.e. predicted from ``logits[:, p-1]``. So we score
``logits[:, :-1].argmax(-1) == labels[:, 1:]`` at the non-(-100) positions: the
standard next-token shift, identical to `PretrainLM._loss` and memmap_ppl.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from utils import (
    is_master,
    get_dist_rank,
    get_dist_size,
    get_device,
    chunk_list_interleaved,
    master_print,
    create_complete_marker,
)
from evaluation.constants import DEFAULT_EVAL_VERSION_STR
from .synthetic_kv import make_mqar, make_phonebook


def _extract_logits(out):
    """Accept both the HF CausalLMOutput (.logits) and the raw PretrainLM dict
    ({'logits': ...}) so the evaluator works on an exported dmlm checkpoint AND on a
    bare in-process `build_model(...)` (used by the smoke test)."""
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, dict):
        return out["logits"]
    if isinstance(out, (tuple, list)):
        return out[0]
    return out  # assume it's already a logits tensor


def score_cell(
    model,
    input_ids: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device,
    rank: int,
    world: int,
):
    """Score one (pairs, ctx) cell. DP-shards the examples across ranks, runs the
    model under inference_mode in batches, counts argmax==gold at the answer
    positions with the standard next-token shift, and all-reduces (correct, total)
    across ranks. Returns (accuracy: float, correct: int, total: int)."""
    n = input_ids.shape[0]
    my_idx = chunk_list_interleaved(list(range(n)), world)[rank]

    correct = torch.zeros((), device=device, dtype=torch.float64)
    total = torch.zeros((), device=device, dtype=torch.float64)
    for i in range(0, len(my_idx), batch_size):
        sl = my_idx[i: i + batch_size]
        if not sl:
            continue
        inp = torch.from_numpy(input_ids[sl].astype(np.int64)).to(device)
        lab = torch.from_numpy(labels[sl].astype(np.int64)).to(device)
        with torch.inference_mode():
            out = model(inp)
        logits = _extract_logits(out)
        # standard next-token shift: logits[:, p-1] predicts token at position p.
        # labels[:, 1:] are the golds for positions 1..T-1, scored by logits[:, :-1].
        pred = logits[:, :-1, :].argmax(dim=-1)            # [b, T-1]
        gold = lab[:, 1:]                                   # [b, T-1]
        mask = gold != -100
        correct += (pred[mask] == gold[mask]).sum().double()
        total += mask.sum().double()

    if world > 1:
        dist.all_reduce(correct)
        dist.all_reduce(total)
    tot = total.item()
    acc = (correct / total).item() if tot > 0 else float("nan")
    return acc, int(correct.item()), int(tot)


class RecallEvaluator:
    """Constructor mirrors `RULEREvaluator` so `meta_eval.py` can wire it identically."""

    def __init__(
        self,
        model,
        tokenizer,
        eval_config: dict,
        res_path: str = "recall_results.json",
        wandb_log_name: str = "recall_benchmark_result",
        batch_size: int = 8,
        version_str: Optional[str] = DEFAULT_EVAL_VERSION_STR,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_config = eval_config
        self.res_path = res_path
        self.wandb_log_name = wandb_log_name
        self.batch_size = eval_config.get("batch_size", batch_size)
        self.version_str = version_str

        self.mode = eval_config.get("mode", "synthetic_kv")  # synthetic_kv | phonebook
        self.num_kv_pairs = list(eval_config.get("num_kv_pairs", [64, 128, 256, 512]))
        self.seq_lens = list(eval_config.get("seq_lens", [2048, 4096]))
        self.num_examples = int(eval_config.get("num_examples", 256))
        self.power_a = float(eval_config.get("power_a", 0.01))
        self.seed = int(eval_config.get("seed", 11785))
        self.random_non_queries = bool(eval_config.get("random_non_queries", True))

        # Use the model's ACTUAL vocab size; the yaml value is only a fallback.
        self.vocab_size = self._resolve_vocab_size(eval_config.get("vocab_size", 32000))
        # where to find wandb_run_id.txt (resume the train run); set by meta_eval
        self.model_dir = kwargs.get("model_dir") or eval_config.get("model_dir")

    def _resolve_vocab_size(self, fallback: int) -> int:
        m = self.model
        for attr in ("vocab_size",):
            v = getattr(m, attr, None)
            if isinstance(v, int) and v > 0:
                return v
        cfg = getattr(getattr(m, "model", m), "config", None) or getattr(m, "config", None)
        if cfg is not None:
            v = getattr(cfg, "vocab_size", None)
            if isinstance(v, int) and v > 0:
                return v
        return int(fallback)

    def _feasible(self, pairs: int, ctx: int):
        """Skip infeasible cells instead of crashing. Need room for the 2*pairs
        context tokens plus the queries region (>= 2*pairs more); for phonebook the
        query is a single trailing pair. Also require vocab > ctx for the disjoint
        key/value draw. Returns (ok: bool, reason: str)."""
        if self.mode == "phonebook":
            need = pairs * 2 + 2
        else:
            need = pairs * 2 * 2   # context + queries region
        if need > ctx:
            return False, f"2*context+queries={need} > seq_len={ctx}"
        if self.vocab_size <= ctx:
            return False, f"vocab_size={self.vocab_size} <= seq_len={ctx}"
        return True, ""

    def _gen(self, pairs: int, ctx: int):
        if self.mode == "phonebook":
            return make_phonebook(
                vocab_size=self.vocab_size, num_entries=pairs, seq_len=ctx,
                num_examples=self.num_examples, seed=self.seed,
                random_non_queries=self.random_non_queries)
        return make_mqar(
            vocab_size=self.vocab_size, num_kv_pairs=pairs, seq_len=ctx,
            num_examples=self.num_examples, seed=self.seed, power_a=self.power_a,
            random_non_queries=self.random_non_queries)

    def evaluate(self, full_save_dir: Optional[str] = None, **kwargs):
        self.model.eval()
        device = get_device(self.model)
        rank, world = get_dist_rank(), get_dist_size()

        master_print(
            f"[recall] mode={self.mode} vocab_size={self.vocab_size} "
            f"pairs={self.num_kv_pairs} ctx={self.seq_lens} "
            f"num_examples={self.num_examples} bs={self.batch_size} world={world}")

        # matrix[ str(pairs) ][ str(ctx) ] = accuracy (or None when skipped)
        matrix = {str(p): {str(c): None for c in self.seq_lens} for p in self.num_kv_pairs}
        skipped, accs = [], []
        for ctx in self.seq_lens:
            for pairs in self.num_kv_pairs:
                ok, reason = self._feasible(pairs, ctx)
                if not ok:
                    master_print(f"[recall] SKIP cell (pairs={pairs}, ctx={ctx}): {reason}")
                    skipped.append({"pairs": pairs, "ctx": ctx, "reason": reason})
                    continue
                input_ids, labels = self._gen(pairs, ctx)
                acc, correct, total = score_cell(
                    self.model, input_ids, labels, self.batch_size, device, rank, world)
                matrix[str(pairs)][str(ctx)] = acc
                accs.append(acc)
                master_print(
                    f"[recall] pairs={pairs:>5} ctx={ctx:>6}  acc={acc:.4f}  "
                    f"({correct}/{total})")

        mean_acc = float(np.mean(accs)) if accs else float("nan")
        results = {
            "mode": self.mode,
            "vocab_size": self.vocab_size,
            "num_kv_pairs": self.num_kv_pairs,
            "seq_lens": self.seq_lens,
            "num_examples": self.num_examples,
            "power_a": self.power_a,
            "seed": self.seed,
            "world_size": world,
            "accuracy_matrix": matrix,
            "mean_accuracy": mean_acc,
            "skipped_cells": skipped,
        }

        if is_master():
            self._print_matrix(matrix, mean_acc)
            try:
                os.makedirs(os.path.dirname(self.res_path) or ".", exist_ok=True)
                with open(self.res_path, "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:  # never crash eval on a result-write failure
                print(f"[recall] result write skipped: {e}", flush=True)
        self._last_results = results   # for the W&B logging in finish_and_log (after the completion marker)
        return results

    def _print_matrix(self, matrix, mean_acc):
        cols = self.seq_lens
        header = "pairs\\ctx".ljust(12) + "".join(f"{c:>10}" for c in cols)
        master_print("[recall] accuracy matrix (rows=num_kv_pairs, cols=seq_len):")
        master_print(header)
        for p in self.num_kv_pairs:
            row = f"{p:<12}"
            for c in cols:
                v = matrix[str(p)][str(c)]
                row += ("{:>10}".format("  --  ") if v is None else f"{v:>10.4f}")
            master_print(row)
        master_print(f"[recall] mean_accuracy = {mean_acc:.4f}")

    def _maybe_log_wandb(self, matrix, mean_acc):
        """Resume the train wandb run (if `{model_dir}/wandb_run_id.txt` exists) and
        log eval/recall_acc_p{pairs}_c{ctx} per cell + eval/recall_mean: lands in the
        same wandb run under eval/. Rank-0 only, best-effort (never raises); off when
        WANDB_MODE=disabled or WANDB_DISABLED=true.
        Mirrors `evaluation/ppl/memmap_ppl.py::_maybe_log_wandb`."""
        model_dir = self.model_dir
        if not model_dir:
            return
        if (os.environ.get("WANDB_MODE", "").lower() == "disabled"
                or os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}):
            return
        id_path = os.path.join(model_dir, "wandb_run_id.txt")
        if not os.path.exists(id_path):
            return
        try:
            with open(id_path) as fh:
                run_id = fh.read().strip()
            if not run_id:
                return
            import wandb
            entity = os.environ.get("WANDB_ENTITY")            # None = the account's default entity
            project = os.environ.get("WANDB_PROJECT", "delta-matching")
            wandb.init(entity=entity, project=project, id=run_id, resume="allow", dir=os.environ.get("WANDB_DIR", os.environ.get("DM_WORKSPACE", ".")))
            log = {"eval/recall_mean": mean_acc}
            tag = "pb" if self.mode == "phonebook" else "kv"
            for p in self.num_kv_pairs:
                for c in self.seq_lens:
                    v = matrix[str(p)][str(c)]
                    if v is not None:
                        log[f"eval/recall_{tag}_acc_p{p}_c{c}"] = v
            wandb.log(log)
            wandb.finish()
        except Exception as e:  # never crash eval on a wandb failure
            print(f"[recall] wandb logging skipped: {e}", flush=True)

    # Parity with RULEREvaluator's meta_eval call surface. evaluate() already wrote the json;
    # here the completion marker comes first (it is NOT optional: meta_eval's resume gate,
    # check_complete_marker on res_path, and scripts/campaign/eval_arm.sh both test for it:
    # <cfg>_results.complete beside <cfg>_results.json; without it a finished recall run looks
    # perpetually unfinished and is re-run on every attempt), then the optional W&B logging
    # to the resumed train run.
    def finish_and_log(self, json_file_path: str = "", task_abbrev: str = "RECALL",
                       train_step: int = -1, **kwargs):
        if not is_master():
            return
        if json_file_path and self.version_str is not None:
            create_complete_marker(json_file_path, version_str=self.version_str)
        results = getattr(self, "_last_results", None)
        if results is not None:
            self._maybe_log_wandb(results["accuracy_matrix"], results["mean_accuracy"])
