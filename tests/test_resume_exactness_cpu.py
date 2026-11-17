"""A resumed run reproduces the uninterrupted run's LOSSES, not only its data stream: CPU (gloo, fp32), the real
`python -m pretrain.train` loop with the FSDP2 backend and DCP checkpoints, AdamW and Muon, 1 and 2 ranks, a WSD schedule
whose warmup and decay both straddle a resume point (step 4 in the warmup, step 18 in the decay). On CPU every kernel
is deterministic, so the check is bitwise: per-step ce_loss / grad_norm / lr from <run>.metrics.jsonl (the record after
the last resume wins) and the final weights (pytorch_model.bin). What the resume must restore: the weights, the AdamW
moments and step, Muon's momentum, the scheduler's position, global_step and the data position.
The GPU twin (tests/test_resume.py, the campaign arms) compares the first step after the resume (it depends only on the
restored weights and the batch) and the following steps against the noise floor of two uninterrupted runs.
About 2 minutes for the four cases (slow: it launches training subprocesses)."""
import json, os, signal, subprocess, sys

import numpy as np
import pytest
import torch

from _helpers import REPO, clean_env

pytestmark = pytest.mark.slow

CFG = """run_name: x
backend: fsdp2
global_batch_size: 4
per_device_batch_size: 2
num_steps: 24
hf_export_steps: []
fused_ce: false
bf16: false
save_interval: 1000000
save_total_limit: 0
logging_steps: 1
seed: 11785
model: {vocab_size: 256, d_model: 64, n_layers: 2, n_heads: 4, n_kv_heads: 2, max_seq_len: 32, mixer: gqa, attn_impl: sdpa,
        qk_norm: true, attn_output_gate: true, tie_embeddings: true, z_loss_weight: 1.0e-4}
optim: {lr: 3.0e-3, weight_decay: 0.1, max_grad_norm: 1.0, lr_scheduler_type: wsd, warmup_ratio: 0.25, decay_ratio: 0.3}
data: {paths: X, dtype: uint16, num_workers: 0, pin_memory: false, seed: 7}
"""


def _run(tmp, name, *args, nproc=1, timeout=600):
    env = clean_env()
    env.update(PYTHONPATH=str(REPO / "src"), CUDA_VISIBLE_DEVICES="", WANDB_MODE="disabled", OMP_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    launch = [sys.executable, "-m", "pretrain.train"] if nproc == 1 else \
        [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", "-m", "pretrain.train"]
    cmd = launch + ["--config", str(tmp / "c.yaml"), "--data.paths", str(tmp / "tok.bin"), "--ckpt_root_dir", str(tmp / "ck"),
                    "--log_root_dir", str(tmp / "lg"), "--run_name", name, *args]
    p = subprocess.Popen(cmd, cwd=str(tmp), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         start_new_session=True)
    try:
        out, _ = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        pytest.fail(f"{name}: no end within {timeout} s\n{out[-3000:]}")
    assert p.returncode == 0, out[-4000:]


def _metrics(tmp, name):
    out = {}
    for line in open(tmp / "lg" / f"{name}.metrics.jsonl"):
        e = json.loads(line)
        if e["event"] == "metrics":   # the record after the LAST resume wins (the documented canonical rule)
            out[e["step"]] = (e["train/ce_loss"], e["train/grad_norm"], e["train/learning_rate"])
    return out


@pytest.mark.parametrize("opt,nproc", [("adamw", 1), ("muon", 1), ("adamw", 2), ("muon", 2)])
def test_resumed_losses_equal_uninterrupted(tmp_path, opt, nproc):
    rng = np.random.default_rng(0)
    base = rng.integers(2, 256, size=4096)
    np.concatenate([np.roll(base, k) for k in range(16)]).astype(np.uint16).tofile(tmp_path / "tok.bin")
    (tmp_path / "c.yaml").write_text(CFG)
    a = f"A_{opt}_{nproc}"
    _run(tmp_path, a, "--optim.name", opt, nproc=nproc)                                   # uninterrupted 0..23
    for k in (4, 18):                                                                     # in the warmup / in the decay
        b = f"B_{opt}_{nproc}_{k}"
        _run(tmp_path, b, "--optim.name", opt, "--stop_at_step", str(k), nproc=nproc)    # same horizon: same WSD curve
        _run(tmp_path, b, "--optim.name", opt, nproc=nproc)                               # resumes from checkpoint-k
        ma, mb = _metrics(tmp_path, a), _metrics(tmp_path, b)
        assert sorted(mb) == list(range(24))
        bad = [(s, ma[s], mb[s]) for s in range(k, 24) if ma[s] != mb[s]]
        assert not bad, f"resume at {k}: the first diverging step {bad[0]}"
        wa = torch.load(tmp_path / "ck" / a / "checkpoint-24" / "pytorch_model.bin", weights_only=True)
        wb = torch.load(tmp_path / "ck" / b / "checkpoint-24" / "pytorch_model.bin", weights_only=True)
        assert wa.keys() == wb.keys() and all(torch.equal(wa[n], wb[n]) for n in wa), f"resume at {k}: final weights differ"
