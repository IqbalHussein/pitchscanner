# Benchmark report

- Generated: `2026-09-28T17:17:31+00:00`
- Commit: `a475a4e`
- Hardware: `Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39` / Python `3.12.3`
- Settings: `{"store": "memory", "index_kind": "hnsw", "hnsw_ef_search": 80, "ivfflat_probes": 20, "candidate_multiplier": 5, "outcome_window_s": 15.0}`

Rerun with `make bench`. Every number quoted anywhere else in this repo has to come from this file.

## Dataset

| matches | events | frames | possessions | outcomes |
| --- | --- | --- | --- | --- |
| 4 | 8066 | 5018 | 1543 | 5018 |

## Query latency

Target: p95 under 150ms end to end, aggregate under 10ms.

| encoding | queries | p50 ms | p95 ms | p99 ms | encode | search | dedup | aggregate | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 1000 | 9.31 | 19.736 | 28.399 | 0.566 | 16.682 | 0.262 | 3.286 | PASS |
| E1 | 1000 | 1.655 | 2.981 | 3.942 | 0.32 | 1.394 | 0.169 | 1.285 | PASS |
| E2 | 1000 | 1.332 | 2.125 | 2.737 | 0.197 | 0.82 | 0.087 | 0.981 | PASS |

## Recall against exact search

Target: 0.95 or higher at the chosen settings.

> These recall numbers were produced against the in-memory backend, which is exact by construction, so they are 1.0 by definition and mean nothing. Recall is only informative against `store=postgres` with an ANN index. Do not quote the numbers below from a memory run.

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
| E1 | 0.995 | 1.0 | 200 |
| E2 | 0.505 | 0.79 | 200 |

## Index size and build

| encoding | dim | metric | kind | vectors | bytes |
| --- | --- | --- | --- | --- | --- |
| E0 | 292 | cosine | exact | 5018 | 5861024 |
| E1 | 72 | cosine | exact | 5018 | 1445184 |
| E2 | 18 | l2 | exact | 5018 | 361296 |

## Retrieval quality (blind, hand-labelled)

No labels found. Generate the pool with `make eval-pool`, label `bench/reports/eval-labels.jsonl`, then rerun `make bench`. Until then there are no precision numbers, and none should be quoted.
