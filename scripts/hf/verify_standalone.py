#!/usr/bin/env python3
"""verify_standalone.py: check a standalone HF model against this repository's loader, on a GPU.

    python scripts/hf/verify_standalone.py --export <HF export dir> --standalone <dir | repo id> [--windows 4]

Each side runs in its own process. The reference loads the export with this repository's code (PYTHONPATH=src,
pretrain.hf.register_auto()); the standalone side loads `--standalone` (a local dir from make_standalone.py, or a Hub
repo id) with trust_remote_code, in an environment where `pretrain` is NOT importable, so a hidden dependency on this
repository's source fails loudly. It PASSES when:

  * the logits are bitwise identical on --windows held-out windows of 8,192 tokens (and so is their CE);
  * a training forward/backward (2 x 2,048 tokens, labels = inputs) gives the same loss with finite gradients;
  * greedy decoding matches: a plain-forward greedy loop on both sides (this repository's loader takes no attention mask for
    MLA, so the reference cannot use generate()), and the standalone's generate() with and without its cache (generate()
    stops at the eos id in the standalone's config, so it is compared with the loop cut after its first eos token).

Both sides load with transformers' default attention (SDPA), so like is compared with like; the study's evaluation
loaded models with attn_implementation="flash_attention_2" instead. Both models must come from this repository: a
config whose model_type is not dmlm is refused.

Exit status 0 on PASS, 1 on FAIL. Needs a CUDA GPU (the Mamba2, GatedDeltaNet and KDA layers run GPU kernels from
mamba_ssm and fla). --val points at a uint16 token file (default: the stage-1 held-out stream of the data root);
--val random uses random token ids when that file is not at hand.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
W = 8192
PROMPT = "The capital of France is"
NOT_OURS = "this export was not written by this repository; convert its config to model_type dmlm first"


def _check_model_type(config_path: str, what: str) -> None:
    if not os.path.isfile(config_path):
        sys.exit(f"{what}: no config.json there (not an HF model directory)")
    model_type = json.load(open(config_path)).get("model_type")
    if model_type != "dmlm":
        sys.exit(f"{what}: config.json has model_type {model_type!r}: {NOT_OURS}")


def _tokens(val: str, n_windows: int, seed: int = 0):
    import numpy as np
    import torch
    if val == "random":
        g = torch.Generator().manual_seed(seed)
        return torch.randint(3, 32000, ((n_windows + 1) * W,), generator=g)
    arr = np.memmap(val, dtype=np.uint16, mode="r")
    return torch.from_numpy(arr[:(n_windows + 1) * W].astype(np.int64))


def side(mode: str, path: str, val: str, n_windows: int, out: str) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if mode == "ref":
        from pretrain.hf import register_auto
        register_auto()
        kw = {}
    else:
        try:
            import pretrain  # noqa: F401
            raise SystemExit("`pretrain` is importable on the standalone side: the check would not be clean")
        except ImportError:
            pass
        if not os.path.isdir(path):   # a Hub repo id (main() checked a local dir already)
            from transformers.utils import cached_file
            _check_model_type(cached_file(path, "config.json"), path)
        kw = {"trust_remote_code": True}
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, **kw).cuda().eval()
    tok = AutoTokenizer.from_pretrained(path, **kw)
    toks = _tokens(val, n_windows).cuda()
    res = {"class": f"{type(model).__module__}.{type(model).__name__}", "sha": [], "ce": []}
    with torch.no_grad():
        for i in range(n_windows):
            ids = toks[i * W:(i + 1) * W][None]
            logits = model(input_ids=ids).logits
            res["sha"].append(hashlib.sha256(logits.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest())
            res["ce"].append(F.cross_entropy(logits[0, :-1].float(), ids[0, 1:]).item())
    model.train()
    ids = toks[n_windows * W:n_windows * W + 2 * 2048].view(2, 2048)
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    grads = [p.grad.float() for p in model.parameters() if p.grad is not None]
    res["loss"] = loss.item()
    res["grad_norm"] = torch.sqrt(sum((g * g).sum() for g in grads)).item()
    res["grads"] = [len(grads), sum(1 for p in model.parameters() if p.requires_grad)]
    res["finite"] = all(torch.isfinite(g).all().item() for g in grads)
    model.zero_grad(set_to_none=True)
    model.eval()
    enc = tok(PROMPT, return_tensors="pt").to("cuda")
    with torch.no_grad():
        ids = enc["input_ids"]
        for _ in range(24):   # greedy by plain forwards, no mask: the reference both sides can run
            ids = torch.cat([ids, model(input_ids=ids).logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        res["gen"] = ids[0].tolist()
        if mode == "remote":
            # generate() stops after the first eos token it produces (the standalone config carries the eos id)
            eos = model.generation_config.eos_token_id
            eos = set() if eos is None else {eos} if isinstance(eos, int) else set(eos)
            n0 = enc["input_ids"].shape[1]
            cut = next((i + 1 for i in range(n0, len(res["gen"])) if res["gen"][i] in eos), len(res["gen"]))
            res["gen_stop"] = res["gen"][:cut]
            res["gen_cache"] = model.generate(**enc, max_new_tokens=24, do_sample=False)[0].tolist()
            res["gen_nocache"] = model.generate(**enc, max_new_tokens=24, do_sample=False, use_cache=False)[0].tolist()
    res["text"] = tok.decode(res["gen"], skip_special_tokens=True)
    json.dump(res, open(out, "w"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--export", required=True, help="HF export dir (loaded with this repository's code)")
    ap.add_argument("--standalone", required=True, help="standalone dir or Hub repo id (loaded with trust_remote_code)")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--val", default=None, help="uint16 token file or 'random' "
                    "(default $DM_DATA_ROOT/nemotron_cc_v2d1_hq_dqa/nemotron_rs_val.bin)")
    ap.add_argument("--side", nargs=2, metavar=("MODE", "OUT"), help=argparse.SUPPRESS)
    a = ap.parse_args()
    val = a.val or os.path.join(os.environ.get("DM_DATA_ROOT", os.path.join(REPO, "workspace", "data")),
                                "nemotron_cc_v2d1_hq_dqa", "nemotron_rs_val.bin")
    if a.side:
        mode, out = a.side
        side(mode, a.export if mode == "ref" else a.standalone, val, a.windows, out)
        return
    _check_model_type(os.path.join(a.export, "config.json"), a.export)
    if os.path.isdir(a.standalone):
        _check_model_type(os.path.join(a.standalone, "config.json"), a.standalone)
    tmp = tempfile.mkdtemp(prefix="verify_hf_")
    results = {}
    try:
        for mode in ("ref", "remote"):
            env = dict(os.environ)
            # DM_SDPA_BACKEND pins an SDPA backend in this repository's attention only (the reference side); unset,
            # both sides run torch's default SDPA dispatch
            env.pop("DM_SDPA_BACKEND", None)
            if mode == "ref":
                env["PYTHONPATH"] = os.path.join(REPO, "src")
            else:
                env.pop("PYTHONPATH", None)
            out = os.path.join(tmp, f"{mode}.json")
            cmd = [sys.executable, os.path.abspath(__file__), "--export", a.export, "--standalone", a.standalone,
                   "--windows", str(a.windows), "--val", val, "--side", mode, out]
            r = subprocess.run(cmd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, cwd=REPO)
            if r.returncode != 0:
                print(f"{mode} side failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
                sys.exit(1)
            results[mode] = json.load(open(out))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ref, rem = results["ref"], results["remote"]
    rel = abs(ref["grad_norm"] - rem["grad_norm"]) / max(ref["grad_norm"], 1e-12)
    checks = {
        f"logits bitwise identical ({a.windows} x {W} tokens)": ref["sha"] == rem["sha"],
        "val CE identical": ref["ce"] == rem["ce"],
        "training loss identical": ref["loss"] == rem["loss"],
        "gradients finite, same count": rem["finite"] and ref["grads"] == rem["grads"],
        f"gradient norm within 1e-4 (rel diff {rel:.1e})": rel < 1e-4,
        "greedy decoding = reference": ref["gen"] == rem["gen"],
        "generate() with cache = greedy decoding (to the first eos)": rem["gen_cache"] == rem["gen_stop"],
        "generate() without cache = greedy decoding (to the first eos)": rem["gen_nocache"] == rem["gen_stop"],
    }
    print(f"reference : {ref['class']}\nstandalone: {rem['class']}")
    print("val CE    :", ", ".join(f"{x:.6f}" for x in rem["ce"]))
    print(f"generation: {rem['text']!r}")
    for k, v in checks.items():
        print(f"  [{'ok' if v else 'FAIL'}] {k}")
    passed = all(checks.values())
    print("PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
