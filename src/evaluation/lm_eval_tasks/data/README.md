# Task data

## aime2025.jsonl

The 30 problems of the 2025 American Invitational Mathematics Examination (AIME I and AIME II), used by the
`aime2025_reasoning` and `aime2025_reasoning_131k` tasks (`../aime2025_reasoning*.yaml`) for evaluation only.

- **Source.** The `aime25` test set of NVIDIA NeMo-Skills (`dataset/aime25` in the NeMo-Skills repository, Apache
  License 2.0).
- **Problems.** The problem statements are from the MAA AIME 2025 and are copyright the Mathematical Association of
  America (MAA).
- **Fields.** `id` (`aime25-0` to `aime25-29`), `problem` (LaTeX text), `expected_answer` (the integer answer, as a
  string). The upstream records also carry a `reference_solution` field (community solutions from the Art of Problem
  Solving wiki); this copy drops it because no code reads it. `id`, `problem` and `expected_answer` are unchanged.
- **Answers.** The 30 `expected_answer` values were checked against the official AIME 2025 I and II answer keys.
