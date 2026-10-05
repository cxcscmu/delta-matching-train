"""The evaluation configs and answer extractors, CPU only: every config under src/evaluation/configs parses with a
unique `name` (two configs of one name write into the same results directory; the ruler_llama2_64k pair shares it on
purpose, for eval_arm_64k.sh), eval_arm.sh's completion checks name the files meta_eval really writes, the utils package
exports its own submodules, and the reasoning tasks' answer extractors keep their fixed cases."""
import importlib.util
import re
from collections import Counter

import yaml

from _helpers import REPO

CONFIGS = REPO / "src" / "evaluation" / "configs"
TASKS = REPO / "src" / "evaluation" / "lm_eval_tasks"


def test_eval_configs_parse_with_unique_names():
    names = {}
    for p in sorted(CONFIGS.glob("*.yaml")):
        c = yaml.safe_load(p.read_text())
        assert isinstance(c, dict) and c.get("name") and c.get("type"), f"{p.name}: no name / type"
        names[p.name] = c["name"]
    dup = {n: sorted(f for f, m in names.items() if m == n) for n, k in Counter(names.values()).items() if k > 1}
    assert dup == {"ruler_llama2_64k": ["ruler_llama2_64k.yaml", "ruler_llama2_64k_smoke.yaml"]}, dup


def test_eval_arm_markers_match_what_meta_eval_writes():
    """meta_eval writes <output_dir>/<config name>/<yaml file stem>_results.complete; eval_arm.sh checks
    $OUT/<name>/<name>_results.complete, so each benchmark's config name and yaml stem must both be <name>."""
    text = (REPO / "scripts" / "campaign" / "eval_arm.sh").read_text()
    calls = re.findall(r"^\s*(?:if .*?then )?run_cfg (\w+) \"?(\S+?\.yaml)\"?", text, re.M)
    assert {n for n, _ in calls} == {"ruler_llama2_8k", "commonsense", "recall_extraction", "triviaqa", "recall_synthetic", "mmlu"}, calls
    for name, path in calls:
        src = CONFIGS / path.split("/")[-1]           # $OUT/ruler_llama2_8k.yaml is a copy of the config of that name
        assert src.stem == name and yaml.safe_load(src.read_text())["name"] == name, (name, path)


def test_utils_exports_its_own_submodules():
    import utils
    assert utils.io.__name__ == "utils.io" and utils.dist.__name__ == "utils.dist"


def _task_utils(name):
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", TASKS / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_reasoning_answer_extractors():
    gpqa, mmlu_pro, aime = (_task_utils(n) for n in ("gpqa_reasoning_utils", "mmlu_pro_reasoning_utils", "aime2025_reasoning_utils"))
    # case-insensitive like the NeMo-Skills reference (a lowercase letter counts; the article "a" reads A there too)
    assert gpqa._extract_letter("Answer: B") == "B" and gpqa._extract_letter("**answer:** c") == "C"
    assert gpqa._extract_letter("Answer: a bit unclear") == "A"
    assert gpqa._extract_letter("The answer is (B).") == "B"
    # the "answer is" fallback needs the letter to end its word
    assert mmlu_pro._extract_letter("the answer is e") == "E"
    assert mmlu_pro._extract_letter("So the answer is definitely (C).") == "C"
    assert mmlu_pro._extract_letter("the answer is Iodine") is None
    # only a \\boxed with its own brace group counts, as NeMo's search_boxed
    assert aime._last_boxed("so \\boxed 5 and {x}") is None and aime._last_boxed("\\boxed {7}") == "7"
    assert aime._last_boxed("first \\boxed{3}, then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
