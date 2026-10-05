"""Evaluation smoke on the tiny clean export: the memmap perplexity scorer (the val-CE protocol's primitive), the
lm-eval harness path (two cached commonsense tasks, plain HF loglikelihood) and the generation paths through the in-repo
continuous-batching engine (RULER, TriviaQA few-shot); the engine runs must show that the engine really served them."""
import json, subprocess
import numpy as np
import pytest
from _helpers import REPO, NEMO, python, torchrun, base_env, memmap_ppl_env

pytestmark = [pytest.mark.gpu, pytest.mark.slow, pytest.mark.data]
ENGINE_ON = "inference.EngineModel (continuous batching)"   # meta_eval's line when the engine serves the run
ENGINE_FALLBACK = "falling back to the plain HF path"         # meta_eval's line when a mixer keeps the engine out


def _run(cmd, env, timeout=1800):
    p = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + "\n" + p.stderr


def _assert_engine_ran(log):
    assert ENGINE_ON in log, f"the engine did not serve the run (no {ENGINE_ON!r} line):\n{log[-2500:]}"
    assert ENGINE_FALLBACK not in log, f"meta_eval fell back to the plain HF path:\n{log[-2500:]}"


def test_memmap_ppl(clean_export, root, run_root, test_gpu):
    out = run_root / "eval_ppl"; out.mkdir(exist_ok=True)
    rc, log = _run([python(), "-m", "evaluation.ppl.memmap_ppl", "--model_name_or_path", str(clean_export),
                    "--data", str(root / NEMO / "nemotron_rs_val.bin"), "--dtype", "uint16", "--seq_len", "2048",
                    "--max_windows", "4", "--batch_size", "2", "--output_dir", str(out), "--run_name", "tiny_n4"],
                   memmap_ppl_env(test_gpu))
    assert rc == 0, log[-3000:]
    line = [l for l in log.splitlines() if "memmap-ppl RESULT" in l]
    assert line, log[-2000:]
    nll = float(line[-1].split("nll/tok=")[1].split()[0])
    assert 5.0 < nll < 12.0, line[-1]
    npy = list(out.glob("*perwindow.npy"))
    assert npy, f"no per-window artifact in {out}"
    a = np.load(npy[0]); assert a.shape == (2, 4) and np.isfinite(a[1]).all(), a


def test_lm_eval_harness_commonsense(clean_export, run_root, test_gpu):
    """Loglikelihood tasks run on the plain HF path (eval_arm.sh never sets EVAL_ENGINE for them)."""
    out = run_root / "eval_cs"; out.mkdir(exist_ok=True)
    env = base_env(test_gpu)
    rc, log = _run([torchrun(), "--standalone", "--nproc_per_node=1", "-m", "evaluation.meta_eval",
                    "--model_name_or_path", str(clean_export), "--eval_config", str(REPO / "tests/configs/commonsense_tiny.yaml"),
                    "--output_dir", str(out), "--run_name", "tiny", "--eval_batch_size", "8", "--max_test_num", "16"], env)
    res = out / "commonsense_tiny" / "commonsense_tiny_results.json"
    assert rc == 0 and res.exists(), f"rc={rc}; {res} exists={res.exists()}; tail:\n{log[-3000:]}"
    assert (out / "commonsense_tiny" / "commonsense_tiny_results.complete").exists(), "no .complete marker"
    assert ENGINE_ON not in log, "a loglikelihood benchmark ran through the engine (only .generate is valid there)"
    d = json.load(open(res)); r = d.get("results", d)
    assert "avg" in r and 0.0 <= float(r["avg"]) <= 1.0, r
    assert {"copa", "piqa"} <= set(r.get("verbose", {})), r.get("verbose")


def test_ruler_engine(clean_export, root, run_root, test_gpu):
    out = run_root / "eval_ruler"; out.mkdir(exist_ok=True)
    cfg = (REPO / "tests/configs/ruler_tiny.yaml").read_text().replace("SET_BY_THE_TEST", str(root / "ruler-500-llama2"))
    (out / "ruler_tiny.yaml").write_text(cfg)
    env = base_env(test_gpu); env.update({"EVAL_ENGINE": "mini_eval_dev", "ENGINE_MAX_SEQS": "4"})
    rc, log = _run([torchrun(), "--standalone", "--nproc_per_node=1", "-m", "evaluation.meta_eval",
                    "--model_name_or_path", str(clean_export), "--eval_config", str(out / "ruler_tiny.yaml"),
                    "--output_dir", str(out), "--run_name", "tiny", "--eval_batch_size", "4"], env)
    res = out / "ruler_tiny" / "ruler_tiny_results.json"
    assert rc == 0 and res.exists(), f"rc={rc}; {res} exists={res.exists()}; tail:\n{log[-3000:]}"
    assert (out / "ruler_tiny" / "ruler_tiny_results.complete").exists(), "no .complete marker"
    _assert_engine_ran(log)
    d = json.load(open(res)); r = d.get("results", d)
    assert "avg/2048" in r, list(r)[:10]


def test_lm_eval_harness_triviaqa_fewshot(clean_export, run_root, test_gpu):
    """The generation path of the harness through the engine with an explicit few-shot count (the TriviaQA protocol)."""
    out = run_root / "eval_tqa"; out.mkdir(exist_ok=True)
    env = base_env(test_gpu); env.update({"EVAL_ENGINE": "mini_eval_dev", "ENGINE_MAX_SEQS": "8"})
    rc, log = _run([torchrun(), "--standalone", "--nproc_per_node=1", "-m", "evaluation.meta_eval",
                    "--model_name_or_path", str(clean_export), "--eval_config", str(REPO / "tests/configs/triviaqa_tiny.yaml"),
                    "--output_dir", str(out), "--run_name", "tiny", "--eval_batch_size", "8", "--max_test_num", "16"], env)
    res = out / "triviaqa_tiny" / "triviaqa_tiny_results.json"
    assert rc == 0 and res.exists(), f"rc={rc}; {res} exists={res.exists()}; tail:\n{log[-3000:]}"
    assert (out / "triviaqa_tiny" / "triviaqa_tiny_results.complete").exists(), "no .complete marker"
    assert "num_fewshot: 5" in log, "the few-shot count was not passed to the harness"
    _assert_engine_ran(log)
    d = json.load(open(res)); r = d.get("results", d)
    assert "avg" in r and 0.0 <= float(r["avg"]) <= 1.0 and "triviaqa" in r.get("verbose", {}), r
