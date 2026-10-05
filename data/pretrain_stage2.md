# Stage 2: the context-extension corpus (ProLong-64K: ProLong re-tokenized with Llama-2, 65,536-token windows)

Every stage-2 run of the paper (every arm of `scripts/experiments/stage2_context.sh`) continues a stage-1 model at
65,536 tokens on the same windows in the same order: 28,000 steps of 16 x 65,536 tokens (29.4 B tokens) drawn from the
training stream below, and the 64K validation CE is measured on the held-out windows. The scripts call it the
ProLong-64K corpus. This page is the recipe; `data/build_stage2.py` runs it.

## Result

`<root>/prolong_64k_llama2/` (`<root>` = `DM_DATA_ROOT`):

| file | content | size | sha256 |
|---|---|---|---|
| `prolong64k_train.bin` | the training stream: uint16 Llama-2 ids, 566,949 windows of 65,536 tokens back to back | 37,155,569,664 tokens, 74,311,139,328 bytes | `b9028c40...df5b31` |
| `prolong64k_train.doc_lengths.npy` | int64, one length per document, sums to the token count | 20,186,304 documents, 161,490,560 bytes | `27516c60...c8513b` |
| `prolong64k_val.bin` | the held-out validation windows | 256 x 65,536 = 16,777,216 tokens, 33,554,432 bytes | `dd7a3b6e...645cdc` |
| `prolong64k_val.doc_lengths.npy` | 256 x 65,536: each window is scored as one sequence | 2,176 bytes | `e65da5ed...a6d783` |

The full hashes are in `data/checksums.json`; `python data/verify.py --root <root> --part stage2` checks them.

## Recipe

1. **Source.** The Hugging Face dataset `princeton-nlp/prolong-data-512K` at revision
   `0bb2becdbb958cf29804da01ae615eb838170aa4`: the long-context data of ProLong (Gao et al., 2024), 31 B Llama-3 tokens (its
   card) in 12 directories of MDS shards (122.4 GB), each directory in 64 independent `procNN-64` streams packed to 524,288
   tokens. The 512K release is used rather than the 64K one (`princeton-nlp/prolong-data-64K`): it holds the books and
   code repositories in pieces of up to 524,288 Llama-3 tokens (the 64K release only in pieces of 65,536), so more of
   each long document stays together through the re-tokenization. The ten output domains are the domains of ProLong's 64K
   training stage (the 64K release's `dclm-baseline` is not one of them); the long ones take both length bins of the 512K
   release:

   | output domain | source directories | windows | window ids | share |
   |---|---|---|---|---|
   | `thestackv1_concat_by_repo-65536` (code, whole repositories) | `thestackv1_concat_by_repo-65536`, `-524288` | 134,498 | [0, 134,498) | 23.7 % |
   | `book-65536` (books, SlimPajama) | `book-65536`, `book-524288` | 115,656 | [134,498, 250,154) | 20.4 % |
   | `fineweb-edu` | `fineweb-edu` | 115,771 | [250,154, 365,925) | 20.4 % |
   | `fineweb-2023-50` | `fineweb-2023-50` | 116,312 | [365,925, 482,237) | 20.5 % |
   | `stackexchange` | `stackexchange` | 20,096 | [482,237, 502,333) | 3.5 % |
   | `dolmawiki` (Wikipedia, Dolma) | `dolmawiki` | 19,271 | [502,333, 521,604) | 3.4 % |
   | `tuluv2` | `tuluv2` | 5,071 | [521,604, 526,675) | 0.9 % |
   | `arxiv` | `arxiv` | 17,376 | [526,675, 544,051) | 3.1 % |
   | `openwebmath` | `openwebmath` | 19,061 | [544,051, 563,112) | 3.4 % |
   | `textbooks` | `textbooks` | 3,837 | [563,112, 566,949) | 0.7 % |

2. **Tokenizers.** Decoding: Meta's Llama-3 tokenizer (`meta-llama/Meta-Llama-3-8B`, revision
   `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920`, file `original/tokenizer.model`), which `build_stage2.py` puts at
   `<root>/tokenizer/llama3/tokenizer.model`. Encoding: the Llama-2 SentencePiece model of stage 1
   (`<root>/tokenizer/llama2_32k/tokenizer.model`), called directly through `sentencepiece` (stage 1 used transformers'
   fast tokenizer; the two can differ on rare inputs, so each stage keeps its own encoder).
3. **Repack** (`data/stage2_repack.py`), per `procNN-64` stream (a stream's documents never continue in another one):
   - *Reconstruct the source documents.* A document starts at a Llama-3 BOS (128000) or right after a Llama-3 EOS
     (128001); a document cut by the 524,288-token windows is rejoined. It ended genuinely iff its last token is the
     EOS. Special ids (>= 128000: BOS, EOS and the chat markers of `tuluv2`) are dropped; a document with nothing else
     is skipped.
   - *Re-encode.* Decode the whole document (the bytes of its ids, UTF-8 with replacement, as tiktoken does; never a
     packed blob, so no token merges across documents) and encode the text with no special tokens. The document is
     laid down as `[BOS=1] + ids + [EOS=2]` when it ended genuinely and `[BOS=1] + ids` otherwise: the books and code
     of the release are length-chunked with no EOS, so they get none, and the model is never taught to stop in the
     middle of a document.
   - *Pack.* datatools' `SingleBuffer`, as ProLong packs: documents are concatenated into 65,536-token windows, a
     document longer than the rest of a window continues in the next one, and each stream's trailing partial window is
     dropped. 20,200,975 source documents make 566,949 windows.
   - *Flatten.* The windows are written back to back, the streams of a domain in name order and the domains in the
     table's order, so window c of the corpus is tokens [c x 65,536, (c + 1) x 65,536). The document lengths come from
     each window's document spans: a new document starts at every span but a window's first, and at a window's first
     span only when the previous window ended on EOS. Result: 20,186,304 documents (mean 1,841 tokens, median 565,
     max 13,332,246); 45.4 % of the tokens are in documents longer than 65,536 tokens and 22.2 % in documents longer than
     524,288.
4. **Validation** (`data/stage2_val.py`). The training sampler draws windows in the order
   `numpy.random.Generator(PCG64(11785)).permutation(566949)` (`data.seed 11785`, `src/pretrain/data/dist_sampler.py`),
   and stage 2 keeps the first 85 % of that permutation (`data.chunk_subset [0, 0.85]`: 481,906 windows, of which
   28,000 x 16 = 448,000 are drawn, none twice). The validation windows are the first 256 of the permutation's last 2 %
   (positions 555,610 to 566,948), sorted and copied whole: a random sample of the domains (window ids 4,116 to
   562,976) that no stage-2 run trains on. They stay inside `prolong64k_train.bin` too; a run with another `data.seed`
   or a `chunk_subset` reaching past 0.98 would train on them.

No deduplication, filtering or mixture reweighting is applied: ProLong's 64K training script weights the domains
(code 0.30, books 0.30, fineweb-edu 0.10, fineweb-2023-50 0.10, stackexchange 0.04, dolmawiki 0.04, tuluv2, arxiv,
openwebmath, textbooks 0.03 each), while stage 2 draws windows uniformly, so the domains enter with the shares of the
table. Positions run 0 to 65,535 across each window (`position_mode: packed`, no intra-document attention mask), as in
stage 1.

## Commands

    export DM_DATA_ROOT=$PWD/workspace/data
    python data/fetch_tokenizer.py --root $DM_DATA_ROOT      # once (stage 1 needs it too)
    python data/build_stage2.py --root $DM_DATA_ROOT         # download, repack, val, verify

`--step download|repack|val|verify` runs one step; `--raw DIR` puts the release elsewhere (default
`<root>/raw/prolong-data-512K`); `--delete-raw` removes it after a verified build; `--workers N` sets the repack
processes (default: every CPU; the output does not depend on it). An interrupted repack continues where it stopped
(finished streams are kept under `prolong64k_train_shards/` until the end). Needs about 200 GB of free disk during the
build (122.4 GB for the release, 74.3 GB for the output) and 74.3 GB after it. On a 208-CPU node the build takes 36
minutes (17 to download, 18 to repack on 192 processes, 1 to carve and check).

The build is exact: with numpy 2.4.6 and sentencepiece 0.2.1 (`env/requirements-cu130-hopper.lock.txt`) it reproduces
the four files above bit for bit, which the last step checks.

## Terms

`princeton-nlp/prolong-data-512K` is public and carries no license of its own; its data come from the sources listed on
its dataset card (The Stack, the SlimPajama books, arXiv and StackExchange subsets, FineWeb and FineWeb-Edu, Dolma's
Wikipedia, Tulu v2, OpenWebMath, TextbookChapters), each under its own terms, which every user must check and follow.
`meta-llama/Meta-Llama-3-8B` is gated: accept the Llama 3 Community License on its page and log in (`hf auth login`)
before the download step. This repository ships no data.
