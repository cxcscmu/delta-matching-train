"""AIME-2025 as a reasoning generation task (NeMo aai math recipe).

30 problems (AIME I+II 2025), integer answers 0-999. The harness wrapper (LMEvalWrapper, when the
config sets strip_reasoning) has already stripped through the last </think>. We extract the LAST
\\boxed{...} and compare to the
gold answer (integer/normalized-string equality: AIME answers are integers, so no
math_verify dependency is needed), averaging over the N repeats (pass@1[avg-of-N]).
Mirrors NeMo-Skills evaluator/math.py (extract_from_boxed + equality).
"""
import re


def doc_to_text(doc):
    # NeMo aai math-oai prompt (boxed, step-by-step).
    return (
        "Solve the following math problem step by step. Put your answer inside "
        "\\boxed{}.\n\n" + doc["problem"] + "\n\nRemember to put your answer inside \\boxed{}."
    )


def keep_all(resps, docs):
    # Identity filter: pass ALL N (repeats) responses per doc to process_results.
    return resps


def _last_boxed(text):
    """Return the content of the LAST \\boxed{...} via balanced-brace scan (NeMo search_boxed). A \\boxed without its
    own brace group gives None, as in NeMo (never the next, unrelated {...} further on)."""
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    if not text[idx + len("\\boxed"):].lstrip().startswith("{"):
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None  # unbalanced


def _normalize(ans):
    if ans is None:
        return None
    ans = ans.strip()
    ans = re.sub(r"\\text\{.*?\}", "", ans)
    ans = ans.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    ans = ans.replace("$", "").replace(",", "").replace(" ", "").strip()
    ans = ans.lstrip("0") or "0"  # AIME answers 0-999; drop leading zeros (016 -> 16)
    return ans


def _equal(pred, gold):
    p, g = _normalize(pred), _normalize(gold)
    if p is None:
        return False
    if p == g:
        return True
    try:
        return int(p) == int(g)
    except (ValueError, TypeError):
        return False


def process_results(doc, results):
    cands = results[0]  # list of N post-</think> generations (kept by keep_all)
    if isinstance(cands, str):
        cands = [cands]
    gold = str(doc["expected_answer"])
    scores = [1.0 if _equal(_last_boxed(c), gold) else 0.0 for c in cands]
    acc = sum(scores) / len(scores) if scores else 0.0
    return {"exact_match": acc}
