"""PG19 book loader for the perplexity eval.

Two sources, in priority order:

1. **Local curated set** (`processed_data/PG-19-curated/books.jsonl`, relative to the working
   directory; not shipped with this repository): a book subset you build yourself, one JSON
   object per line, `{"idx": <0, 1, 2, ... in ascending order>, "text": <book text>}`, for example
   the first 100 PG19 test books that are at least `l_max` (131072) tokens long under the
   evaluated model's tokenizer. Use it when every curve point should have the same
   `n_books_scored`. Selected when `split == "curated"` (or `split == "auto"` and the file exists).

2. **HuggingFace `emozilla/pg19`** parquet mirror: full splits (test/val/train).
   The original `deepmind/pg19` dataset script is no longer supported by `datasets >= 3.x`.
   Selected when `split` is one of {test, validation, train}.
"""

import json
import os
from typing import Iterator


PG19_REPO = "emozilla/pg19"
PG19_CURATED_DIR = "processed_data/PG-19-curated"
PG19_CURATED_BOOKS = os.path.join(PG19_CURATED_DIR, "books.jsonl")


def _curated_available() -> bool:
    return os.path.isfile(PG19_CURATED_BOOKS)


def _iter_curated(num_books: int) -> Iterator[tuple[int, str]]:
    with open(PG19_CURATED_BOOKS) as f:
        for line in f:
            entry = json.loads(line)
            idx = int(entry["idx"])
            if idx >= num_books:
                break
            yield idx, entry["text"]


def _iter_hf(num_books: int, split: str) -> Iterator[tuple[int, str]]:
    from datasets import load_dataset

    ds = load_dataset(PG19_REPO, split=split)
    n = min(num_books, len(ds))
    for i in range(n):
        yield i, ds[i]["text"]


def iter_pg19_books(num_books: int = 100, split: str = "auto") -> Iterator[tuple[int, str]]:
    """Yield (book_idx, text) for the first `num_books` books.

    `split` values:
      - "curated"          → use processed_data/PG-19-curated/books.jsonl (raises if missing)
      - "auto"             → "curated" if available, else "test"
      - "test" / "validation" / "train" → HF emozilla/pg19 split
    """
    if split == "auto":
        split = "curated" if _curated_available() else "test"
    if split == "curated":
        if not _curated_available():
            raise FileNotFoundError(
                f"Curated PG19 set not found at {PG19_CURATED_BOOKS} (relative to the working directory; "
                "not shipped with this repository). Create it as JSONL with one {\"idx\", \"text\"} object per "
                "book (idx ascending from 0), or set `split: test` (HF emozilla/pg19) or `split: auto` in the config."
            )
        return _iter_curated(num_books)
    return _iter_hf(num_books, split)


def load_pg19_books(num_books: int = 100, split: str = "auto") -> list[tuple[int, str]]:
    return list(iter_pg19_books(num_books=num_books, split=split))
