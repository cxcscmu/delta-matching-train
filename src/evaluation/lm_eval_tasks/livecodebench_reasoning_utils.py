"""LiveCodeBench as a reasoning generation task (NeMo aai recipe).

Grades model-generated Python by EXECUTING it against each problem's public+private
test cases, using the vendored LiveCodeBench grader (_lcb_testing_util.run_test). The
model reasons then answers; the harness wrapper (LMEvalWrapper, when the config sets
strip_reasoning) has already stripped through the last </think>; here we take the LAST
```python fenced block, run it against ALL tests (a sample passes iff EVERY test passes:
LCB semantics), and average over the N repeats -> pass@1[avg-of-N]. Mirrors NeMo-Skills
evaluator/livecodebench.py + code.py.

Problem windows (contest months, inclusive), one per task:
  livecodebench_reasoning       release_v5, 2024-07..2024-12 (process_docs; NVIDIA's
                                test_v5_2407_2412 split, 315 problems)
  livecodebench_reasoning_131k  release_v6, 2024-08..2025-05 (process_docs_v6; the model
                                card's v6 window, 454 problems)
LCB_START_DATE / LCB_END_DATE (YYYY-MM) override the window of either task (e.g. 2024-10 /
2025-02 for the 166-problem split).

Safety: this runs untrusted, model-generated code on the host. The tasks are marked
unsafe_code, so lm_eval runs them only with confirm_run_unsafe_code=True, which meta_eval
passes only with the explicit opt-in DM_ALLOW_CODE_EVAL=1. Each sample runs in a child
process under reliability_guard, per-test SIGALRM timeouts (LCB_TIMEOUT, default 6 s) and
an address-space limit of 4 GiB above what the child inherits (_cap_child_memory). None of
this is a sandbox: run these tasks on an isolated machine (ideally in a container or jail).
"""
import base64
import json
import multiprocessing
import os
import pickle
import re
import sys
import zlib
from datetime import date, datetime

# vendored LiveCodeBench grader lives next to this file
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lcb_testing_util import run_test  # noqa: E402

LCB_START = os.environ.get("LCB_START_DATE", "2024-07")  # livecodebench_reasoning (release_v5)
LCB_END = os.environ.get("LCB_END_DATE", "2024-12")
LCB_V6_START = os.environ.get("LCB_START_DATE", "2024-08")  # livecodebench_reasoning_131k (release_v6)
LCB_V6_END = os.environ.get("LCB_END_DATE", "2025-05")
TIMEOUT = int(os.environ.get("LCB_TIMEOUT", "6"))
CHILD_EXTRA_MEMORY_BYTES = 4 * 1024**3  # address space a candidate program may add (see _cap_child_memory)

_FORMAT_NO_STARTER = (
    "Write Python code to solve the problem. Please place the solution code in the "
    "following format:\n```python\n# Your solution code here\n```"
)
_FORMAT_STARTER = (
    "Please place the solution code in the following format:\n```python\n"
    "# Your solution code here\n```"
)


def _month_window(start_ym, end_ym):
    sy, sm = (int(x) for x in start_ym.split("-"))
    ey, em = (int(x) for x in end_ym.split("-"))
    start = date(sy, sm, 1)
    end = date(ey + em // 12, em % 12 + 1, 1)  # first day of month AFTER end_ym (exclusive)
    return start, end


def _decode_tests(doc):
    pub = json.loads(doc["public_test_cases"])
    raw = doc["private_test_cases"]
    try:
        priv = json.loads(raw)
    except Exception:
        # base64 + zlib + pickle blob, as upstream LiveCodeBench stores it. pickle.loads runs in the evaluation
        # process and trusts the dataset: load it only from the dataset revision the task YAML names.
        priv = json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))
    tests = pub + priv
    meta = json.loads(doc["metadata"]) if doc.get("metadata") else {}
    fn_name = meta.get("func_name")
    return json.dumps({
        "inputs": [t["input"] for t in tests],
        "outputs": [t["output"] for t in tests],
        "fn_name": fn_name,
    })


def process_docs(dataset):
    """livecodebench_reasoning: contests 2024-07..2024-12 unless LCB_START_DATE / LCB_END_DATE are set."""
    return _process_docs(dataset, LCB_START, LCB_END)


def process_docs_v6(dataset):
    """livecodebench_reasoning_131k: contests 2024-08..2025-05 unless LCB_START_DATE / LCB_END_DATE are set."""
    return _process_docs(dataset, LCB_V6_START, LCB_V6_END)


def _process_docs(dataset, start_ym, end_ym):
    start, end = _month_window(start_ym, end_ym)

    def _keep(doc):
        d = datetime.strptime(doc["contest_date"], "%Y-%m-%dT%H:%M:%S").date()
        return start <= d < end

    dataset = dataset.filter(_keep)

    def _proc(doc):
        starter = doc.get("starter_code") or ""
        if starter.strip():
            starter_block = (
                "\n\nSolve the problem starting with the provided function header.\n\n"
                "Function header:\n```\n" + starter + "\n```"
            )
            fmt = _FORMAT_STARTER
        else:
            starter_block = ""
            fmt = _FORMAT_NO_STARTER
        return {
            "_user": doc["question_content"] + starter_block + "\n\n" + fmt,
            "_input_output": _decode_tests(doc),
        }

    # drop the heavy raw test fields once decoded (keeps the in-memory dataset + saved
    # samples small; the decoded _input_output is consumed by process_results).
    drop = [c for c in ("public_test_cases", "private_test_cases", "metadata") if c in dataset.column_names]
    return dataset.map(_proc, remove_columns=drop)


def doc_to_text(doc):
    return doc["_user"]


def keep_all(resps, docs):
    # identity filter: keep all N (repeats) responses per doc for pass@1[avg-of-N]
    return resps


_FENCE = re.compile(r"```(?:python)?[ \t]*\r?\n(.*?)```", re.DOTALL)


def _extract_code(text):
    # LAST fenced code block (```python preferred, else ```); unclosed fence -> "" (fail),
    # matching NeMo preprocess_code.
    blocks = _FENCE.findall(text or "")
    return blocks[-1] if blocks else ""


def _cap_child_memory(extra_bytes=CHILD_EXTRA_MEMORY_BYTES):
    """Limit this child's address space (RLIMIT_AS) to its current size plus `extra_bytes`.

    The child is forked from the evaluation rank and inherits its mappings (a CUDA process reserves
    tens of GiB of virtual address space), so an absolute 4 GiB limit would leave no room for any
    allocation; relative to the inherited size, the candidate program gets 4 GiB. Linux only (reads
    /proc/self/statm); where the size cannot be read or the limit cannot be set, no limit is applied."""
    try:
        import resource
        with open("/proc/self/statm") as f:
            current = int(f.read().split()[0]) * resource.getpagesize()
        _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = current + extra_bytes
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))   # hard too: the candidate cannot raise it again
    except (ImportError, OSError, ValueError, IndexError):
        pass


def _temp_run(sample, code, timeout, result):
    try:
        _cap_child_memory()  # in the child, before run_test executes the candidate
        res, _meta = run_test(sample, test=code, debug=False, timeout=timeout)
        result.append(res)
    except Exception:
        pass


def check_correctness(input_output, code, timeout=TIMEOUT):
    """Run `code` against the problem's tests in a child process; return the per-test
    result list. All-pass iff every entry > 0 (LCB semantics)."""
    sample = {"input_output": input_output}
    n = len(json.loads(input_output)["inputs"])
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, code, timeout, result))
    p.start()
    p.join(timeout=(timeout + 1) * n + 5)
    if p.is_alive():
        p.kill()
    if not result:
        return [-1] * n  # global timeout / crash -> all fail
    return list(result[0])


def _sample_passes(code, input_output):
    if not code.strip():
        return False
    res = check_correctness(input_output, code, TIMEOUT)
    return len(res) > 0 and all((x is True) or (isinstance(x, (int, float)) and x > 0) for x in res)


def process_results(doc, results):
    cands = results[0]  # list of N post-</think> generations (kept by keep_all)
    if isinstance(cands, str):
        cands = [cands]
    io = doc["_input_output"]
    scores = [1.0 if _sample_passes(_extract_code(c), io) else 0.0 for c in cands]
    acc = sum(scores) / len(scores) if scores else 0.0
    return {"pass_at_1": acc}
