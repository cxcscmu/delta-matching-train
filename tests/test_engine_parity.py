"""The continuous-batching inference engine (src/inference) against the HF forward, per mixer family, on a GPU: the
teacher-forced harness src/inference/verify_engine_parity.py (prefill + slot-cached decode vs HF .forward over the same
sequence, logits compared per position) on tiny random-weight models. The engine serves every generation benchmark of
the evaluation (RULER, extraction, TriviaQA), so a decode path that drifts from the training forward biases them all.
Cases: the paper's GatedDeltaNet:GQA hybrid, pure GQA, the GatedDeltaNet:MLA and Mamba2:GQA hybrids, the KDA:GQA
hybrid, and NoPE (the engine must leave the rotary slice unrotated when the model does)."""
import subprocess

import pytest

from _helpers import REPO, base_env, python

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("case", ["hybrid_gdn", "gqa", "hybrid_gdn_mla", "hybrid_mamba2", "hybrid_kda", "nope"])
def test_engine_teacher_forced_parity(case, test_gpu):
    p = subprocess.run([python(), "-m", "inference.verify_engine_parity", case], cwd=str(REPO), env=base_env(test_gpu),
                       capture_output=True, text=True, timeout=900)
    out = p.stdout + p.stderr
    assert p.returncode == 0 and f"[{case}] TEACHER-FORCED PARITY: PASS" in out, f"{case}: rc={p.returncode}\n{out[-3000:]}"
