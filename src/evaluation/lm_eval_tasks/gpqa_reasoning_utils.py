"""GPQA-diamond as a reasoning generation task (NeMo aai mcq-4choices recipe).

The model reasons then answers; the harness wrapper (LMEvalWrapper, when the config sets
strip_reasoning) has already stripped everything through the last </think> before these
functions see the text. We extract the answer LETTER (\\boxed{X} / Answer: X / last
standalone A-D) and average correctness over the N sampled repeats (pass@1[avg-of-N]).
Mirrors NeMo-Skills evaluator/mcq.py.
"""
import random
import re

SHUFFLE_SEED = 42  # NeMo prepare.py shuffles distractors with seed 42


def _preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset):
    rng = random.Random(SHUFFLE_SEED)

    def _process(doc):
        correct = _preprocess(doc["Correct Answer"])
        choices = [
            _preprocess(doc["Incorrect Answer 1"]),
            _preprocess(doc["Incorrect Answer 2"]),
            _preprocess(doc["Incorrect Answer 3"]),
            correct,
        ]
        rng.shuffle(choices)
        correct_idx = choices.index(correct)
        return {
            "question": _preprocess(doc["Question"]),
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": chr(65 + correct_idx),  # bare letter A-D
        }

    return dataset.map(_process)


_PROMPT = (
    "Answer the following multiple choice question. The last line of your response "
    "should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').\n\n"
    "{question}\n\nA) {c1}\nB) {c2}\nC) {c3}\nD) {c4}"
)


def doc_to_text(doc):
    return _PROMPT.format(
        question=doc["question"], c1=doc["choice1"], c2=doc["choice2"],
        c3=doc["choice3"], c4=doc["choice4"],
    )


def keep_all(resps, docs):
    # Identity filter: pass ALL N (repeats) responses per doc through to process_results
    # (the default take_first would keep only one, defeating pass@1[avg-of-N]).
    return resps


_BOXED_RE = re.compile(r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}")
# Case-insensitive, as NeMo-Skills evaluator/mcq.py (its "(?i)...Answer...:...([A-Z])" also reads a lowercase letter), so
# the scores reproduce the reference; the letter must end its word. Like the reference, "Answer: a bit unclear" reads A.
_ANSWER_RE = re.compile(r"(?i)\bAnswer\b[\*\s:_]*\(?\[?\{?\s*([A-Da-d])(?![A-Za-z0-9])")
_STANDALONE_RE = re.compile(r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])")


def _extract_letter(text):
    if not text:
        return None
    m = _BOXED_RE.findall(text)
    if m:
        return m[-1].upper()
    m = _ANSWER_RE.findall(text)
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
