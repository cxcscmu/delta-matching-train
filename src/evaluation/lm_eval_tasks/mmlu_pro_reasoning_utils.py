"""Generative (thinking-ON) MMLU-Pro for reasoning (instruct) models such as Nemotron: mirrors
the gpqa_diamond_reasoning recipe but for 10 choices (A-J), boxed answer.

A reasoning model must run MMLU-Pro thinking-ON with a large budget: the standard greedy /
2048-token / 5-shot recipe scores near 0 because the think block never closes. Score the text
after the last </think> (the harness wrapper strips it when the config sets strip_reasoning);
extract the option letter; pass@1[avg-of-N] (the task uses repeats: 1).
"""

from __future__ import annotations

import re

_LETTERS = "ABCDEFGHIJ"

# Official NVIDIA aai/mcq-10choices-boxed prompt (NeMo-Skills), verbatim wording.
_PROMPT = (
    "Answer the following multiple choice question. The last line of your response should "
    "be in the following format: 'Answer: \\boxed{{A/B/C/D/E/F/G/H/I/J}}' (e.g. 'Answer: "
    "\\boxed{{A}}').\n\n{question}\n\n{options}"
)


def doc_to_text(doc):
    opts = doc["options"]
    block = "\n".join(f"{_LETTERS[i]}) {o}" for i, o in enumerate(opts) if i < len(_LETTERS))
    return _PROMPT.format(question=str(doc["question"]).strip(), options=block)


def keep_all(resps, docs):
    # Identity filter: pass ALL N (repeats) generations per doc to process_results.
    return resps


_BOXED_RE = re.compile(r"\\boxed\{\s*\(?([A-Ja-j])\)?\s*\}")
# Case-insensitive, as NeMo-Skills evaluator/mcq.py (a lowercase letter counts), so the scores reproduce the reference.
# The "answer is" fallback (not in the reference) needs the letter to end its word, so "the answer is definitely (C)"
# is not read as D and "the answer is Iodine" not as I.
_ANSWER_RE = re.compile(r"(?i)\bAnswer\b[\*\s:_]*\(?\[?\{?\s*([A-Ja-j])(?![A-Za-z0-9])")
_ANSWER_IS_RE = re.compile(r"(?i)answer is\s*\(?([A-Ja-j])(?![A-Za-z0-9])")
_STANDALONE_RE = re.compile(r"(?<![A-Za-z0-9])([A-J])(?![A-Za-z0-9])")


def _extract_letter(text):
    if not text:
        return None
    for rx in (_BOXED_RE, _ANSWER_RE, _ANSWER_IS_RE):
        m = rx.findall(text)
        if m:
            return m[-1].upper()
    m = _STANDALONE_RE.findall(text)
    if m:
        return m[-1].upper()
    return None


def process_results(doc, results):
    cands = results[0]  # list of N post-</think> generations (kept by keep_all)
    if isinstance(cands, str):
        cands = [cands]
    gold = str(doc["answer"]).strip().upper()
    scores = [1.0 if _extract_letter(c) == gold else 0.0 for c in cands]
    acc = sum(scores) / len(scores) if scores else 0.0
    return {"exact_match": acc}
