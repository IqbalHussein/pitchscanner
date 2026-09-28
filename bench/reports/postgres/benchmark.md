# Benchmark report

- Generated: `2026-09-28T18:08:35+00:00`
- Commit: `5aec1dc`
- Hardware: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39` / Python `3.12.3`
- Settings: `{"store": "postgres", "index_kind": "hnsw", "hnsw_ef_search": 80, "ivfflat_probes": 20, "candidate_multiplier": 5, "outcome_window_s": 15.0}`

Rerun with `make bench`. Every number quoted anywhere else in this repo has to come from this file.

## Dataset

| matches | events | frames | possessions | outcomes |
| --- | --- | --- | --- | --- |
| 4 | 8066 | 5018 | 1543 | 5018 |

## Query latency

Target: p95 under 150ms end to end, aggregate under 10ms.

| encoding | queries | p50 ms | p95 ms | p99 ms | encode | search | dedup | aggregate | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 1000 | 8.123 | 11.256 | 15.16 | 0.395 | 3.583 | 1.683 | 6.931 | PASS |
| E1 | 1000 | 7.974 | 11.947 | 14.612 | 0.381 | 3.526 | 1.602 | 7.083 | PASS |
| E2 | 1000 | 9.648 | 13.291 | 17.309 | 0.393 | 5.388 | 2.713 | 6.283 | PASS |

## Recall against exact search

Target: 0.95 or higher at the chosen settings.

Measured with `index_kind=hnsw`, `ef_search=80`, `probes=20`.

| encoding | k | filtered | recall | queries | verdict |
| --- | --- | --- | --- | --- | --- |
| E0 | 10 | False | 1.0 | 200 | PASS |
| E0 | 50 | False | 1.0 | 200 | PASS |
| E0 | 10 | True | 1.0 | 200 | PASS |
| E1 | 10 | False | 1.0 | 200 | PASS |
| E1 | 50 | False | 1.0 | 200 | PASS |
| E1 | 10 | True | 1.0 | 200 | PASS |
| E2 | 10 | False | 1.0 | 200 | PASS |
| E2 | 50 | False | 1.0 | 200 | PASS |
| E2 | 10 | True | 1.0 | 200 | PASS |

## Jitter sanity check

Every player nudged 1-2m; the original frame should still rank first.

| encoding | original in top 1 | original in top 5 | queries |
| --- | --- | --- | --- |
| E0 | 1.0 | 1.0 | 200 |
| E1 | 1.0 | 1.0 | 200 |
| E2 | 0.505 | 0.785 | 200 |

## Index size and build

| encoding | dim | metric | kind | vectors | bytes |
| --- | --- | --- | --- | --- | --- |
| E0 | 292 | cosine | hnsw | 5018 | 22282240 |
| E1 | 72 | cosine | hnsw | 5018 | 6930432 |
| E2 | 18 | l2 | hnsw | 5018 | 3588096 |

## FAISS, for comparison

Same HNSW parameters, in process, no filters and no network hop. This is the evidence for choosing pgvector, not an argument against FAISS: the search itself is faster, and the cost is a second process owning an in-memory index that knows nothing about the filter columns every query uses. Search time only - no encode, dedup or aggregation.

| encoding | vectors | build ms | search p50 ms | search p95 ms | recall@10 |
| --- | --- | --- | --- | --- | --- |
| E0 | 5018 | 678.7 | 0.1022 | 0.2941 | 1.0 |
| E1 | 5018 | 588.0 | 0.106 | 0.2163 | 0.998 |
| E2 | 5018 | 532.2 | 0.0856 | 0.1685 | 1.0 |

## Retrieval quality (blind, hand-labelled)

No labels found. Generate the pool with `make eval-pool`, label `bench/reports/eval-labels.jsonl`, then rerun `make bench`. Until then there are no precision numbers, and none should be quoted.
