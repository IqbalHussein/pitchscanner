# Match Situation Search Engine

![Demo](docs/media/demo.gif)

> _Placeholder: record `docs/media/demo.gif` from a local run — sketch a situation,
> watch the results and the outcome statistics update._

Sketch a soccer situation — where the ball is, where the attackers and defenders
stand — and get back the real moments from StatsBomb's open positional data that
look most like it, together with what actually happened next: how often those
situations produced a shot, how much xG they were worth, how often possession
was lost, how often the ball reached the box. Every number comes with a
confidence interval and a comparison against the baseline for that part of the
pitch, so "this shape is dangerous" is a claim you can check rather than a vibe.
Filters are written in plain language ("Bayern at home, last fifteen minutes,
trailing"), parsed by rules first and only then, optionally, by a model.

## Contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [The three encodings](#the-three-encodings)
- [Benchmarks](#benchmarks)
- [Observability](#observability)
- [Design notes](#design-notes)
- [Limitations](#limitations)
- [Development](#development)
- [Attribution](#attribution)

## How it works

```mermaid
flowchart TB
    subgraph offline["Offline batch pipeline (make ingest)"]
        SB[("StatsBomb open data<br/>events + 360 freeze frames<br/>pinned commit")]
        SB --> PARSE["Parse and normalise<br/>orient left-to-right, roles, camera polygon"]
        PARSE --> ENC["Encode<br/>E0 grid · E1 polar · E2 tactical"]
        PARSE --> LAB["Label outcomes<br/>15s forward scan inside the possession"]
        PARSE --> CAT["Entity catalogue<br/>competitions, teams, players, aliases"]
        ENC --> DB[("Postgres + pgvector<br/>one vector table per encoding<br/>denormalised filter columns")]
        LAB --> DB
        CAT --> DB
        LAB --> BASE["Zone baselines<br/>18 zones x outcome"]
        BASE --> DB
    end

    subgraph online["Online query path (p95 target 150ms)"]
        SKETCH["Sketch on the SVG pitch<br/>+ plain-language filters"]
        SKETCH --> API["FastAPI /search"]
        API --> NLF["Filter parser<br/>rules, then optional LLM fallback"]
        API --> QENC["Encode the sketch<br/>and its mirror image"]
        NLF --> SEARCH
        QENC --> SEARCH["ANN over-fetch k x multiplier<br/>filter, dedup by possession,<br/>widen or fall back to exact"]
        DB --> SEARCH
        SEARCH --> AGG["Aggregate outcomes<br/>similarity-weighted, Wilson + bootstrap,<br/>vs zone baseline"]
        AGG --> RESP["Results, statistics, timings"]
    end
```

The offline half is a batch job: it is allowed to be slow, and it is where all
the judgement lives (what counts as a turnover, how a frame is oriented, which
frames are usable at all). The online half only encodes, searches, dedups and
aggregates — that is what makes a 150ms budget realistic.

## Quick start

One command, with Docker:

```bash
make up      # builds the images, starts db + api + web, ingests the fixtures
             # app: http://localhost:8080   API docs: http://localhost:8000/docs
make down
```

No Docker, no Postgres — the in-memory store is exact by construction and the
fixture dataset is small enough to serve from it:

```bash
make install   # venv + backend (dev extras) + frontend deps
make ingest    # fixtures into an in-memory snapshot under .data/
make serve     # API on :8000
make web       # frontend on :5173
```

To run against the real dataset instead of the fixtures, point it at Postgres
and let it fetch StatsBomb open data at the pinned commit:

```bash
docker compose up -d db
make migrate
make ingest-real
```

## The three encodings

A situation has no natural order — eleven players are a *set*, not a list — so
every encoding is built to be order-independent, and each one trades detail for
generality differently.

| id | what it is | dim | metric | good at | bad at |
| --- | --- | --- | --- | --- | --- |
| `E0` | 12x8 occupancy grid, 3 channels (attackers, defenders, unknown) + 4 ball features | 292 | cosine | literal shape matching; easy to explain and to debug visually | absolute positions, so a shape shifted 5m looks different; widest vectors, slowest search |
| `E1` | ball-relative polar bins, 4 rings x 8 sectors, per-ring smoothing | 72 | cosine | "who is near the ball, and where" — translation-invariant, the default | loses absolute pitch location unless the ball fixes it |
| `E2` | 18 named tactical features (compactness, defensive line height, numerical advantage near the ball, …) | 18 | L2 | cheap, interpretable, generalises across areas of the pitch | coarse: different shapes can share a feature vector, which the jitter check below shows |

`E1` is the default (`PITCHSCANNER_ENCODING=E1`). All three are built by the same
ingest run and stored side by side, so switching is a query parameter, not a
re-ingest.

Two details that matter more than the bin counts:

- **Unknown is not empty.** A 360 freeze frame only shows what the broadcast
  camera saw. Every encoding carries an explicit unknown mask derived from the
  camera polygon, so "no defender there" and "we cannot see there" are different
  inputs.
- **Mirroring happens at query time.** Situations are oriented left-to-right at
  ingest; the query is encoded twice, as drawn and mirrored across the halfway
  axis, so a sketch on the left touchline also finds its right-touchline twin
  without doubling the index.

## Benchmarks

Two committed reports, because they answer different questions:

- [`bench/reports/postgres/benchmark.md`](bench/reports/postgres/benchmark.md) —
  Postgres + pgvector with an HNSW index. **This is the one to read**: it is the
  production configuration, and the only place ANN recall means anything.
- [`bench/reports/benchmark.md`](bench/reports/benchmark.md) — the same harness
  against the in-memory store, which is the infrastructure-free path and the
  exact-search reference. Its recall column is 1.0 by definition and the report
  says so.

`make bench` regenerates whichever one the current `PITCHSCANNER_STORE` points at.
Nothing is quoted anywhere in this repo that is not in one of those files. Both
runs below are the **fixture** dataset (4 synthetic matches, 5,018 frames) on a
WSL2 laptop; repeat runs on the same machine vary by around 1.5x, which is why the
CI regression gate allows 2.5x.

Dataset:

| matches | events | frames | possessions | outcomes |
| --- | --- | --- | --- | --- |
| 4 | 8066 | 5018 | 1543 | 5018 |

Query latency on Postgres + pgvector/HNSW, 1000 queries per encoding,
milliseconds, stage columns are p95 for that stage:

| encoding | p50 | p95 | p99 | encode | search | dedup | aggregate | target |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 | 6.8 | 9.4 | 12.1 | 0.35 | 3.0 | 1.3 | 5.9 | PASS |
| E1 | 6.5 | 8.9 | 12.5 | 0.35 | 2.7 | 1.3 | 5.5 | PASS |
| E2 | 8.9 | 11.5 | 14.3 | 0.36 | 4.8 | 2.3 | 5.4 | PASS |

Targets: end-to-end p95 under 150ms, aggregation under 10ms, encoding under 1ms.
The aggregate column is the one worth watching — the first Postgres run had it at
30ms p95 because `_assemble` fetched the match and the frame inside its per-result
loop. See [`docs/design-notes.md`](docs/design-notes.md) §10.

**ANN recall against exact search**, 200 queries per configuration,
`ef_search=80`:

| encoding | k=10 | k=50 | k=10 filtered | target |
| --- | --- | --- | --- | --- |
| E0 | 1.000 | 1.000 | 1.000 | ≥ 0.95 |
| E1 | 1.000 | 1.000 | 1.000 | ≥ 0.95 |
| E2 | 1.000 | 1.000 | 1.000 | ≥ 0.95 |

Read that as "the index is losing nothing at this corpus size" rather than as a
claim about HNSW in general: 5,018 vectors is few enough that the graph walk is
close to exhaustive. The nightly job re-measures it and fails below 0.95.

Jitter sanity check — every player nudged 1–2m, the original frame should still
come back first:

| encoding | original ranked first | original in top 5 |
| --- | --- | --- |
| E0 | 1.000 | 1.000 |
| E1 | 1.000 | 1.000 |
| E2 | 0.505 | 0.785 |

That E2 row is the honest cost of an 18-dimensional summary, and it is the
reason E2 is offered as a lens rather than the default.

Index size (fixture dataset, 5018 vectors, HNSW):

| encoding | dim | metric | index bytes |
| --- | --- | --- | --- |
| E0 | 292 | cosine | 22.3 MB |
| E1 | 72 | cosine | 6.9 MB |
| E2 | 18 | L2 | 3.6 MB |

**Natural-language filters.** 145 hand-labelled phrases in
[`data/eval/filter-phrases.jsonl`](data/eval/filter-phrases.jsonl); the rule
parser scores **1.000 exact match** with per-field f1 of 1.000 across all nine
filter fields (`pytest tests/unit/test_filter_parsing.py -s` prints the table).
The LLM fallback is off by default and is only consulted for phrases the rules
leave unresolved.

**Retrieval quality.** `make eval-pool` builds a 200-query blind pool for hand
labelling. Until `bench/reports/eval-labels.jsonl` exists there are no
precision@10 or nDCG@10 numbers here, and the report refuses to invent any. The
metrics themselves are unit-tested against hand-computed values, including that
nDCG's ideal ranking comes from the whole pool rather than from the run being
scored.

**Tests.** 129 fast tests (unit, property-based, golden snapshots) plus 16
integration tests against a real Postgres with pgvector, plus 24 Playwright tests
across desktop and mobile viewports; 98% statement coverage on the core packages
(encoders, labeller, stats, geometry, filters) against an 85% gate.

## Observability

```bash
make up          # the app
make monitoring  # Prometheus, Grafana, and a blackbox probe of /readyz
                 # grafana: http://localhost:3000 (anonymous, local only)
```

![Grafana dashboard](docs/media/grafana.png)

> _Placeholder: screenshot `docs/media/grafana.png` from a local
> `make monitoring` run after some traffic._

Every request logs one structured JSON line carrying a request id (propagated
through a contextvar, so the id appears on log lines written deep in the search
service), and every response carries a `server-timing` header with the per-stage
breakdown. `GET /metrics` exposes request and per-stage latency histograms, cache
hits and misses, handled errors, rate-limit rejections, exact-scan fallbacks and
the vector count per encoding.

The dashboard in [`ops/grafana/dashboards/pitchscanner.json`](ops/grafana/dashboards/pitchscanner.json)
is provisioned automatically and shows eight panels: request percentiles with the
150ms budget drawn as a threshold line, per-stage p95 (so "which stage moved?" is
one glance, not an investigation), throughput by status, cache hit rate, errors
and rejections, exact-scan fallback rate, indexed vectors per encoding, and 24h
`/readyz` availability. [`ops/alerts.yml`](ops/alerts.yml) has three alerts and
deliberately no more: the service is down, the p95 has been over budget for ten
minutes, or the error rate is elevated.

## Design notes

Longer write-ups live in [`docs/`](docs/):

- [`docs/design-notes.md`](docs/design-notes.md) — the decisions worth arguing
  about: encoder trade-offs, possession dedup, the exact-search fallback,
  query-time mirroring, unknown-vs-empty, and how the statistics are computed.
- [`docs/blog-outline.md`](docs/blog-outline.md) — an outline for the write-up,
  starting from "why does a naive similarity search return the same play eight
  times?"

## Limitations

Read these before quoting anything from the demo.

- **Camera bias.** StatsBomb 360 freeze frames only contain players the
  broadcast camera saw. Off-screen players are unknown, not absent, and the
  visible area shrinks exactly when play is stretched — so wide, transitional
  situations are systematically less complete than settled ones near the ball.
- **Coverage skew.** The 360 data covers a handful of competitions (heavily
  tournament football), not a balanced sample of matches. `GET /v1/filters`
  reports exactly which competitions and seasons have frames in the index. Any
  cross-competition comparison inherits that skew.
- **The demo runs on fixtures.** The checked-in dataset is 4 synthetic matches
  generated by `scripts/make_fixture.py` with a fixed seed — enough to exercise
  every code path deterministically, far too small for a football conclusion.
  Statistics computed over it are arithmetic, not insight. Run `make ingest-real`
  for the real corpus.
- **Grid resolution.** Everything is a 120x80 pitch with 12x8 or 4x8 bins. That
  is roughly 10m cells: it cannot distinguish a defender's near shoulder from the
  far one, and it will call two distinct pressing shapes identical if they agree
  at that scale.
- **A frame is not a phase.** Each indexed vector is one instant. "What happened
  next" is a 15-second forward scan within the same possession and period, which
  is a choice, not a law: a shot at second 16 is invisible, and a possession that
  ends at a half boundary is truncated rather than extrapolated.
- **Similarity weighting is a heuristic.** Neighbours are weighted `1/(1+d)`, and
  the effective sample size (Kish) is reported alongside the raw count precisely
  because that weighting can make 40 neighbours behave like 12. Small samples are
  flagged `low_confidence` rather than hidden.
- **Outcome labels are event-derived.** "Turnover", "box entry" and "shot" are
  defined in `labeller.py` from StatsBomb event types; a foul won is not treated
  as a turnover, for instance. Reasonable people would draw some of those lines
  elsewhere.
- **No learned encoder.** `E3` — a trained embedding — is deliberately out of
  scope. All three encodings are hand-designed, which makes them explainable and
  keeps the quality ceiling where it is.

## Development

```bash
make help              # every target, with one-line descriptions
make test              # unit, property, golden, API
make test-integration  # needs Postgres + pgvector
make test-slow         # the nightly tier: deep property-test budget
make test-e2e          # builds the frontend, runs Playwright
make lint typecheck    # ruff + eslint, mypy (strict on core) + tsc
make bench             # rewrite bench/reports/
make perf-gate         # fresh quick benchmark vs the committed one
make openapi           # regenerate openapi.json and the TS types
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the same things
in cheapest-first order: lint and type-check, then the fast test tiers with the
coverage gate, then integration against a Postgres + pgvector service container,
then Docker image builds with Playwright against them, then the p95 regression
gate, and on merge to `main` it publishes images and deploys.

The slow tier runs nightly ([`.github/workflows/nightly.yml`](.github/workflows/nightly.yml)):
the property tests at a much larger Hypothesis budget, the full 1000-query
benchmark against a real pgvector index with a recall gate at 0.95, and an ingest
of the actual StatsBomb data at the pinned commit — the only job that notices if
upstream changes shape. The deep budget earns its place: it falsified the
mirroring property twice in ways the pull-request budget never reached, both
times by finding situations whose asymmetry lay entirely outside what the encoder
can see — the camera polygon for E0, the outermost ring for E1.

The frontend's TypeScript types are generated from the API's own OpenAPI schema,
and `frontend/src/lib/contract.ts` asserts at compile time that the hand-written
types the app uses actually match them — renaming a response field the app never
reads still fails the build.

## Attribution

Positional and event data from [StatsBomb Open Data](https://github.com/statsbomb/open-data),
used under its terms; StatsBomb requires attribution and this project is not
affiliated with or endorsed by StatsBomb. The fixture dataset in `data/fixtures/`
is synthetic and generated locally — it contains no StatsBomb data.
