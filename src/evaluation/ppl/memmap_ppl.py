"""Perplexity over a flat token memmap (pre-tokenized), multi-node data-parallel.

A self-contained PPL eval for src/pretrain checkpoints (`model_type=dmlm`: GQA flash-attn /
GDN fla / Mamba2 / KDA / MLA): no tokenizer or dataset download; reads held-out
token ids directly. One torchrun rank = one DP replica; non-overlapping `seq_len` windows are
strided-sharded (`chunk_list_interleaved`) and NLL / token counts are all-reduced -> perplexity.

Run (single node, from the repository root):
  PYTHONPATH=src CUDA_VISIBLE_DEVICES=0,1,..  \
    torchrun --nproc_per_node=8 -m evaluation.ppl.memmap_ppl \
      --model_name_or_path <hf_ckpt> --data <val.bin> --dtype uint16 --seq_len 2048 \
      --max_windows 512 --output_dir <out>
Multi-node: add --nnodes/--node_rank/--master_addr/--master_port (or a rendezvous endpoint) to torchrun
(example in docs/eval_engine.md).
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from transformers import AutoModelForCausalLM

from utils import (
    dist_init, dist_close, dist_barrier, get_dist_rank, get_dist_size, is_master,
    chunk_list_interleaved, master_print,
)
from evaluation.ppl.load import _maybe_register_pretrain_models

DT = {"uint16": np.uint16, "uint32": np.uint32}


def _load_model(path):
    """Load the HF causal LM (tokenizer-free: memmap ppl reads token ids directly)."""
    from transformers import AutoConfig
    _maybe_register_pretrain_models()  # register model_type=dmlm if src/pretrain is importable
    attn_impl = os.environ.get("EVAL_ATTN_IMPL", "flash_attention_2")
    cfg = AutoConfig.from_pretrained(path)
    return AutoModelForCausalLM.from_pretrained(
        path, config=cfg, dtype=torch.bfloat16, attn_implementation=attn_impl).cuda().eval()


def _maybe_log_wandb(model_dir, ppl, nll):
    """If `{model_dir}/wandb_run_id.txt` exists, resume that (train) wandb run and log
    the eval metrics under eval/*. Rank-0 only, best-effort (never raises); off when
    WANDB_MODE=disabled or WANDB_DISABLED=true."""
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
        wandb.log({"eval/perplexity": ppl, "eval/nll_per_token": nll})
        wandb.finish()
    except Exception as e:  # never crash eval on a wandb failure
        print(f"[memmap-ppl] wandb logging skipped: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--data", required=True, help="flat token .bin (np.memmap) or .npy")
    ap.add_argument("--dtype", default="uint16", choices=list(DT))
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--max_windows", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--run_name", default="memmap_ppl")
    ap.add_argument("--wandb", action="store_true",
                    help="log eval/* to the train wandb run if wandb_run_id.txt exists in the model dir")
    ap.add_argument("--prepend_bos", type=int, default=None,
                    help="BOS-anchored windows (the --data.prepend_bos training format; the value is the BOS id): input "
                         "[bos] + t[0:L-1], labels t[0:L], so BOS predicts the window's first token. Default "
                         "off = the protocol's windows (input t[0:L], labels t[1:L+1]).")
    a = ap.parse_args()

    dist_init()
    rank, world = get_dist_rank(), get_dist_size()
    device = torch.device("cuda")

    model = _load_model(a.model_name_or_path)

    tokens = (np.load(a.data, mmap_mode="r") if a.data.endswith(".npy")
              else np.memmap(a.data, dtype=DT[a.dtype], mode="r"))
    n_windows = min(a.max_windows, (len(tokens) - 1) // a.seq_len)
    if n_windows <= 0:
        raise ValueError(f"data too short: {len(tokens)} tokens < seq_len {a.seq_len}+1")
    my_windows = chunk_list_interleaved(list(range(n_windows)), world)[rank]
    if a.prepend_bos is not None:
        master_print(f"[memmap-ppl] prepend_bos ON: input [{a.prepend_bos}] + t[0:L-1], labels t[0:L]")
    master_print(f"[memmap-ppl] {a.model_name_or_path}: {n_windows} windows x {a.seq_len} tok "
                 f"across {world} ranks (~{len(my_windows)}/rank)")

    nll_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_sum = torch.zeros((), device=device, dtype=torch.float64)
    # PER-WINDOW NLL. A bare point estimate of nll/tok cannot tell whether a small difference between
    # two models is real. Keeping the per-window vector costs nothing, makes the uncertainty
    # reportable, and lets two models scored on the same windows be compared PAIRED on window index:
    # per-window NLLs of two similar models are highly correlated (r ~ 0.9996 between the study's
    # arms), so the SE of a paired difference is tens of times smaller than the unpaired SE printed
    # below (~37x there).
    win_ids, win_nll = [], []
    for i in range(0, len(my_windows), a.batch_size):
        batch = my_windows[i: i + a.batch_size]
        ids = np.stack([np.asarray(tokens[w * a.seq_len: (w + 1) * a.seq_len + 1]).astype(np.int64)
                        for w in batch])
        ids = torch.from_numpy(ids).to(device)            # [b, seq_len+1]
        if a.prepend_bos is None:
            inp, lab = ids[:, :-1], ids[:, 1:]
        else:                                             # [bos] + t[0:L-1] -> t[0:L] (both [b, seq_len])
            inp = torch.cat([torch.full_like(ids[:, :1], a.prepend_bos), ids[:, :-2]], dim=1)
            lab = ids[:, :-1]
        with torch.inference_mode():
            logits = model(inp).logits.float()            # [b, seq_len, V]
        flat = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), lab.reshape(-1), reduction="none")
        nll = flat.sum()
        # per-window mean nll/token; windows are all exactly seq_len tokens so the token-weighted
        # mean over windows equals the global nll/tok, and a per-window SE is well defined
        win_nll.extend((flat.view(lab.shape).mean(dim=1)).double().tolist())
        win_ids.extend(int(w) for w in batch)
        nll_sum += nll.double()
        tok_sum += lab.numel()

    if world > 1:
        torch.distributed.all_reduce(nll_sum)
        torch.distributed.all_reduce(tok_sum)
    nll_per_tok = (nll_sum / tok_sum).item()
    ppl = math.exp(nll_per_tok)

    # gather the per-window vector so the master can report a standard error and persist it
    all_ids, all_nll = win_ids, win_nll
    if world > 1:
        gath = [None] * world
        torch.distributed.all_gather_object(gath, (win_ids, win_nll))
        all_ids = [i for part in gath for i in part[0]]
        all_nll = [v for part in gath for v in part[1]]
    sem = float("nan")
    try:
        if len(all_nll) > 1:
            arr = np.asarray(all_nll, dtype=np.float64)
            sem = float(arr.std(ddof=1) / math.sqrt(arr.size))
    except Exception as e:                                    # never let stats kill the eval
        master_print(f"[memmap-ppl] per-window SE unavailable: {type(e).__name__}: {e}")

    if is_master():
        res = {"model": a.model_name_or_path, "data": a.data, "seq_len": a.seq_len,
               "n_windows": n_windows, "world_size": world,
               "total_tokens": int(tok_sum.item()), "nll_per_token": nll_per_tok, "perplexity": ppl,
               "nll_sem": sem, "nll_ci95_halfwidth": 1.96 * sem if sem == sem else None,
               "available_windows": (len(tokens) - 1) // a.seq_len}
        print(f"[memmap-ppl RESULT] nll/tok={nll_per_tok:.4f} ppl={ppl:.3f} "
              f"(windows={n_windows} world={world} tokens={int(tok_sum.item())})", flush=True)
        perwindow = (os.path.join(a.output_dir, f"{a.run_name}_perwindow.npy") if a.output_dir
                     else "<run_name>_perwindow.npy, written with --output_dir")
        print(f"[memmap-ppl UNCERTAINTY] unpaired per-window SE={sem:.6f} 95%CI=+/-{1.96*sem:.6f} "
              f"| scored {n_windows} of {(len(tokens)-1)//a.seq_len} available windows "
              f"({100.0*n_windows/max((len(tokens)-1)//a.seq_len,1):.2f}%). "
              f"This SE ignores the pairing of windows: to compare two models scored on the same windows, "
              f"pair them on window index with the per-window artifact ({perwindow}), "
              f"whose paired SE is much smaller.", flush=True)
        if a.output_dir:
            os.makedirs(a.output_dir, exist_ok=True)
            with open(os.path.join(a.output_dir, f"{a.run_name}_results.json"), "w") as f:
                json.dump(res, f, indent=2)
            # per-window vector, indexed by window id, so ANY two arms scored on the same windows
            # can be compared PAIRED after the fact without re-running either
            try:
                order = np.argsort(np.asarray(all_ids))
                np.save(os.path.join(a.output_dir, f"{a.run_name}_perwindow.npy"),
                        np.stack([np.asarray(all_ids)[order],
                                  np.asarray(all_nll, dtype=np.float64)[order]]))
            except Exception as e:
                master_print(f"[memmap-ppl] per-window save failed: {type(e).__name__}: {e}")
        # --wandb: log eval/* to the same train wandb run (when wandb_run_id.txt is present), after the artifacts
        if a.wandb:
            _maybe_log_wandb(a.model_name_or_path, ppl, nll_per_tok)
    dist_barrier()
    dist_close()


if __name__ == "__main__":
    main()
