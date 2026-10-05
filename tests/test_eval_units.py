"""CPU unit checks of evaluation paths whose bugs move numbers without failing: the PG19 growing-window scorer scores
every token exactly once whether or not the model honors logits_to_keep, the plain HF generation path refuses batched
generation for dmlm models with recurrent layers (their state would run through the left padding), RULER resumes and
clears per config, and meta_eval rejects an unknown flag instead of ignoring it."""
import subprocess
import types

import pytest
import torch
import torch.nn.functional as F

from _helpers import REPO, base_env, python

V = 11   # the toy vocabulary


class _ToyLM:
    """A causal toy LM: logits[t] = the running mean of fixed random embeddings of tokens[0..t]."""

    def __init__(self, honor_logits_to_keep):
        self.honor = honor_logits_to_keep
        self.emb = torch.randn(V, V, generator=torch.Generator().manual_seed(0), dtype=torch.float64)

    def __call__(self, ids, use_cache=False, logits_to_keep=0):
        e = self.emb[ids]
        logits = e.cumsum(1) / torch.arange(1, ids.size(1) + 1, dtype=torch.float64)[None, :, None]
        if self.honor and logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return types.SimpleNamespace(logits=logits)


@pytest.mark.parametrize("honor", [True, False])
def test_growing_window_scores_every_token_once(honor):
    """At current_len = k * stride the scorer sums the NLL of the last stride targets (stride - 1 at the first step): the
    records must equal the full forward's per-token NLL over exactly those targets."""
    from evaluation.ppl.growing_window import evaluate_books_batched_growing_window
    stride, l_max = 16, 64
    g = torch.Generator().manual_seed(1)
    books = [(0, torch.randint(0, V, (64,), generator=g)), (1, torch.randint(0, V, (48,), generator=g))]
    recs, oom = evaluate_books_batched_growing_window(_ToyLM(honor), books, stride=stride, l_max=l_max, batch_size=2)
    assert not oom
    for idx, ids in books:
        nll = F.cross_entropy(_ToyLM(False)(ids[None]).logits[0, :-1], ids[1:], reduction="none")   # nll[j]: target j + 1
        want = []
        for k in range(1, l_max // stride + 1):
            L = k * stride
            if ids.numel() < L:
                break
            lo = 1 if k == 1 else L - stride          # the scored targets are positions [lo, L)
            want.append((L, float(nll[lo - 1:L - 1].sum()), L - lo))
        got = [(r.current_len, r.sum_nll, r.n_scored) for r in recs if r.book_idx == idx]
        assert [(L, n) for L, _, n in got] == [(L, n) for L, _, n in want], (idx, got, want)
        for (L, s, _), (_, s_ref, _) in zip(got, want):
            assert abs(s - s_ref) < 1e-4 * max(1.0, abs(s_ref)), (idx, L, s, s_ref)


def _fake_hf(model_type, *mixers):
    m = torch.nn.Module()
    m.config = types.SimpleNamespace(model_type=model_type, vocab_size=10)
    m.model = torch.nn.Module()
    m.model.layers = torch.nn.ModuleList()
    for name in mixers:
        blk = torch.nn.Module()
        blk.mixer = type(name, (torch.nn.Module,), {})()
        m.model.layers.append(blk)
    m.p = torch.nn.Parameter(torch.zeros(1))
    return m


def test_plain_hf_generation_refuses_batches_for_recurrent_dmlm():
    from modeling import HFModel
    from evaluation.ruler.load_model import padded_generation_error
    hybrid = HFModel(_fake_hf("dmlm", "GatedDeltaNetMixer", "GQAAttention"))
    msg = padded_generation_error(hybrid, 4)
    assert msg and "EVAL_ENGINE=mini_eval_dev" in msg and "--eval_batch_size 1" in msg, msg
    assert padded_generation_error(hybrid, 1) is None
    for kinds in (("Mamba2Mixer", "GQAAttention"), ("KDAMixer", "GQAAttention")):
        assert padded_generation_error(HFModel(_fake_hf("dmlm", *kinds)), 4), kinds
    assert padded_generation_error(HFModel(_fake_hf("dmlm", "GQAAttention", "MLAAttention")), 4) is None
    assert padded_generation_error(HFModel(_fake_hf("dmlm", "EngineMamba2", "GQAAttention")), 4) is None   # engine-rebound
    assert padded_generation_error(HFModel(_fake_hf("nemotron_h", "Mamba2Mixer")), 4) is None


RULER_SCENARIOS = r'''
import glob, json, os, sys
os.environ["WANDB_MODE"] = "disabled"
import torch, torch.distributed as dist
from transformers import AutoTokenizer
work, tok_dir = sys.argv[1], sys.argv[2]
dist.init_process_group("gloo", init_method=f"file://{work}/pg", rank=0, world_size=1)
from evaluation.ruler.ruler import RULEREvaluator
tok = AutoTokenizer.from_pretrained(tok_dir)
tok.pad_token = tok.pad_token or tok.eos_token
data = os.path.join(work, "ruler")
for task in ("niah_single_1", "niah_single_2"):   # a tiny RULER data dir in the published layout
    d = os.path.join(data, "2048", "data", task); os.makedirs(d)
    with open(os.path.join(d, "validation.jsonl"), "w") as f:
        for i in range(4):
            text = f"The grass is green. One of the special magic numbers for key-{i} is: {1000 + 37 * i}. " * 3
            q = f"What is the special magic number for key-{i} mentioned in the provided text? The special magic number for key-{i} mentioned in the provided text is"
            f.write(json.dumps({"index": i, "input": text + q, "outputs": [str(1000 + 37 * i)], "length": 100}) + "\n")


class FakeGen(torch.nn.Module):
    """Deterministic generation: six prompt tokens picked by a hash of the prompt."""
    def __init__(self):
        super().__init__(); self.p = torch.nn.Parameter(torch.zeros(1)); self.calls = 0
    def generate(self, input_ids, attention_mask=None, max_new_tokens=8, **kw):
        self.calls += 1
        outs = []
        for row, m in zip(input_ids, attention_mask):
            ids = row[m.bool()].tolist()
            h = sum(ids[-50:]) % max(1, len(ids) - 8)
            outs.append(ids[h:h + 6])
        return torch.cat([input_ids, torch.tensor(outs, dtype=torch.long)], dim=1)


def run(cfg_name, version="v1", max_samples=4):
    d = os.path.join(work, "out", "shared"); os.makedirs(d, exist_ok=True)
    res = os.path.join(d, f"{cfg_name}_results.json")
    cfg = {"name": "shared", "type": "ruler", "data_dir": data, "lengths": [2048], "verbose": True, "max_samples": max_samples,
           "tasks": [{"name": "niah_single_1", "subset": "validation"}, {"name": "niah_single_2", "subset": "validation"}],
           "generation_config": {"do_sample": False, "temperature": 0, "top_k": 32, "top_p": 1.0, "stop": ""}}
    model = FakeGen()
    ev = RULEREvaluator(model, tok, cfg, res_path=res, wandb_log_name="sim", batch_size=2, version_str=version)
    results = ev.evaluate(full_save_dir=os.path.join(d, "full_results"))
    json.dump(results, open(res, "w"), indent=1, default=float)
    ev.finish_and_log(json_file_path=res, task_abbrev=cfg_name)
    return results, model.calls


def files(tag):
    return sorted(p for p in glob.glob(os.path.join(work, "out", "**"), recursive=True) if os.path.isfile(p) and tag in p)


ra, ca = run("ruler_a")
a_files = files("ruler_a")
assert ca > 0 and any(p.endswith("ruler_a_results.complete") for p in a_files) and len([p for p in a_files if p.endswith(".complete")]) == 3, a_files
rb, cb = run("ruler_b", max_samples=3)              # another config in the same output dir: runs fresh ...
assert files("ruler_a") == a_files, "a fresh run of another config deleted this one's artifacts"   # ... and leaves A alone
shared = os.path.join(work, "out", "shared")
os.remove(os.path.join(shared, "ruler_b_results.complete"))   # B interrupted after its predictions: same run_id resumes them
run_id = json.load(open(os.path.join(shared, "ruler_b_results.json")))["run_id"]
json.dump({"avg/full": [], "avg/2048": [], "eval_time/2048": [], "verbose": {}, "run_id": run_id}, open(os.path.join(shared, "ruler_b_results.json"), "w"))
rb2, cb2 = run("ruler_b", max_samples=3)
assert cb2 == 0 and rb2["avg/full"] == rb["avg/full"], (cb2, rb2["avg/full"], rb["avg/full"])
stale = os.path.join(shared, "full_results", "ruler_b", "2048", "STALE.jsonl"); open(stale, "w").write("x")
rb3, cb3 = run("ruler_b", version="v2", max_samples=3)       # a new version: B's own artifacts cleared and regenerated
assert cb3 == cb and not os.path.exists(stale) and files("ruler_a") == a_files, (cb3, cb, os.path.exists(stale))
print("SCENARIOS OK")
'''


def test_ruler_resumes_and_clears_per_config(tok_dir, tmp_path):
    """RULER (the engine-scored protocol benchmark) with a deterministic fake generator on a tiny data dir: a fresh run of
    one config never deletes another config's results or predictions in the same output directory, an interrupted run
    with the same run id resumes from its predictions without generating again (same scores), and a new version clears
    and regenerates only its own artifacts (eval_arm.sh relies on all three when it re-runs a benchmark)."""
    p = subprocess.run([python(), "-c", RULER_SCENARIOS, str(tmp_path), str(tok_dir)], cwd=str(REPO), env=base_env(""),
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0 and "SCENARIOS OK" in p.stdout, f"rc={p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-2500:]}"


def test_meta_eval_rejects_an_unknown_flag():
    p = subprocess.run([python(), "-m", "evaluation.meta_eval", "--model_name_or_path", "unused", "--eval_bsz", "4"],
                       cwd=str(REPO), env=base_env(""), capture_output=True, text=True, timeout=600)
    assert p.returncode == 2 and "unrecognized arguments" in p.stderr, f"rc={p.returncode}\n{p.stderr[-1500:]}"
