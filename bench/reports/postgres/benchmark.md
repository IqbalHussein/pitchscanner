# Benchmark report

- Generated: `2026-09-28T17:57:18+00:00`
- Commit: `931c021`
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
| E0 | 1000 | 6.755 | 9.406 | 12.119 | 0.352 | 2.983 | 1.343 | 5.875 | PASS |
| E1 | 1000 | 6.523 | 8.933 | 12.547 | 0.346 | 2.748 | 1.329 | 5.461 | PASS |
| E2 | 1000 | 8.88 | 11.5 | 14.274 | 0.364 | 4.765 | 2.33 | 5.371 | PASS |

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
| E1 | 72 | cosine | hnsw | 5018 | 6922240 |
| E2 | 18 | l2 | hnsw | 5018 | 3596288 |

## Retrieval quality (blind, hand-labelled)

No labels found. Generate the pool with `make eval-pool`, label `bench/reports/eval-labels.jsonl`, then rerun `make bench`. Until then there are no precision numbers, and none should be quoted.
