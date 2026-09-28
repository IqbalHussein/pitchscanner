# Design notes

The decisions in here are the ones where a reasonable person would have chosen
differently, and where the reason is not visible in the code.

## 1. Why a naive similarity search is useless, and what fixes it

The first working version returned ten results that were the same play. Freeze
frames come every few seconds inside one possession, and consecutive frames of
the same attack are, by construction, each other's nearest neighbours. A top-10
by raw distance is therefore a top-10 of *one moment*, sampled ten times.

The fix is dedup by possession: at most one result per `(match_id,
possession_id)`, keeping the closest frame. That in turn breaks the ANN contract,
because you can no longer ask the index for `k` and stop — most of what comes
back is thrown away. So the search over-fetches `k * candidate_multiplier`
(default 5), dedups, and if it still has fewer than `k` survivors it widens the
multiplier by 4x up to `max_candidate_multiplier` (40) and tries again. The
widening is bounded on purpose: an unbounded retry loop is how a 150ms budget
becomes a 2-second p99.

Dedup is per possession rather than per match because two distinct attacks in the
same match are genuinely two observations; and it is not per event, because that
is the degenerate case we started with.

## 2. Why three encodings instead of one good one

A situation is a *set* of players plus a ball. Sets have no order, so anything
that puts player coordinates in a fixed-length vector by index is wrong the
moment a team lines up differently or a player is missing from the camera view.
All three encodings are order-independent by construction — each sums a kernel
over players rather than indexing them.

| | `E0` grid | `E1` polar | `E2` tactical |
| --- | --- | --- | --- |
| invariances | none beyond order | translation (ball-centred) | translation and, for most features, reflection |
| dim | 292 | 72 | 18 |
| fails when | the same shape sits 5m further up the pitch | absolute pitch location matters and the ball does not pin it | two different shapes agree on all 18 summaries |

`E1` is the default because "who is around the ball, and in which direction" is
what people actually mean when they sketch a situation, and because ball-relative
binning generalises across the pitch without the encoder having seen that area
before. `E0` is kept because it is the one you can *look at* — debugging a bad
result usually starts by rendering its grid. `E2` is kept because 18 named
numbers are explainable to someone who does not care about vector search, and
because it is a useful lens: "find me situations with this defensive line height
and this compactness" is a different question from "find me this shape".

The jitter check in the benchmark report is what keeps this honest. Nudging every
player 1–2m should not change the answer. `E0` and `E1` still rank the original
frame first ~100% and 99.5% of the time; `E2` manages 50.5%. That is not a bug in
`E2`, it is what 18 dimensions buy you, and it is why `E2` is not the default.

### Comparing situations with different numbers of players

Two frames rarely have the same number of visible players: the camera saw nine in
one and six in the other. Three things handle that without special-casing:

1. **Occupancy, not slots.** A player contributes a Gaussian bump to the field.
   Six players produce a smaller total mass than nine, they do not produce
   garbage in six of nine slots.
2. **Normalisation.** `E0` and `E1` are L2-normalised and compared by cosine, so
   overall mass differences matter less than the *pattern* of mass.
3. **An explicit unknown channel.** The difference between "six players, three
   off-camera" and "six players, and we could see the whole pitch" is recorded,
   not inferred.

## 3. Unknown is not empty

This is the single most consequential modelling decision in the project.

A 360 freeze frame ships a `visible_area` polygon: the region the broadcast
camera covered. Outside it, the data says nothing. If you encode absence as zero
occupancy, you teach the index that "no defender here" and "we cannot see here"
are the same input — and since the camera narrows exactly when play is stretched,
you systematically make transitional situations look sparsely defended.

So every encoding carries an unknown mask derived from the polygon
(`known_mask`), and `E2` exposes `known_fraction` as a feature. Sketches have no
camera, so a sketch marks a disk of radius `SKETCH_KNOWN_RADIUS = 18m` around
each placed token as known and everything else as unknown: the user asserted
things about what they drew, and asserted nothing about the rest.

One subtlety found by a property test: `known_fraction` was originally estimated
by sampling cell *corners* of the polygon, which is not symmetric under
reflection, so mirroring a situation changed a feature that is supposed to be
side-agnostic. It samples cell centres now.

## 4. Mirroring at query time, not at index time

Situations are oriented left-to-right at ingest (`orient_left_to_right`, which is
idempotent), so the attacking direction is a constant and not a feature. That
still leaves the touchline: a move down the left and the same move down the right
are the same tactical idea and different vectors.

Two options: store both orientations of every frame (index doubles, recall
budget halves, dedup gets a new class of near-duplicate), or encode the *query*
twice and merge. The second is strictly cheaper — encoding costs well under half
a millisecond, and searching twice is still well inside budget — and it keeps the
index honest:
every row in it is a real frame that really happened.

## 5. The exact-search fallback

ANN indexes degrade badly when a filter removes most of the corpus. HNSW walks a
graph built over *all* vectors; if only 300 of 5,000 pass the filter, the walk
spends its effort on rejected nodes and the survivors it returns are not the
nearest ones. Post-filtering k results gives you too few; pre-filtering is not
something an HNSW index can do.

So the store counts the matching rows first (`count_matching`, answered from the
denormalised filter columns on the frame row — no joins), and when that count is
at or below `exact_fallback_threshold` (2000), the search does a sequential scan
of the filtered subset instead, with `enable_indexscan` off. At that size a brute
force scan is faster *and* exact, so the narrow-filter case is the one where
quality is best rather than worst — the opposite of the naive behaviour.

Filters live in exactly one place, `store/filtering.py`, which has both
`matches()` for the in-memory store and `to_sql()` for Postgres. Two
implementations of filter semantics would silently diverge, and the property
tests that compare the two backends would be the only thing to notice.

## 6. Statistics: intervals, weights, and effective n

Reporting "62% shot rate" from 13 neighbours is the failure mode this section
exists to prevent.

- **Similarity weighting.** A neighbour at distance 0.02 should count for more
  than one at 0.4, so each contributes `w = 1/(1+d)`. That is a heuristic; it is
  chosen for being monotone, bounded and free of tuning parameters.
- **Effective sample size.** Weighting reduces information. `effective_n` is
  Kish's effective sample size, it is what the intervals are computed from, and it
  is a first-class field on the response next to `n` - not a line item inside
  `unweighted`, where nobody would look for it. The frontend shows both, because
  the gap between them is the interesting part.
- **Wilson intervals for rates**, not normal approximations: rates here are often
  near 0 (goals) where the normal interval happily returns a negative lower
  bound.
- **Bootstrap for means** (xG), percentile method, seeded (`bootstrap_seed =
  20260925`) so the same query returns the same interval — a confidence interval
  that flickers between reloads destroys trust faster than a wide one.
- **Zone baselines.** An absolute shot rate means little; the same rate in the
  opposition box versus at the halfway line means opposite things. Every outcome
  is also reported as a ratio against the baseline for its zone (the pitch is cut
  into 6 columns x 3 rows = 18 zones), computed over the whole corpus at ingest.
- **Low confidence is a flag, not a filter.** Below `low_confidence_n` (30
  effective) the response is marked `low_confidence` with a note, and the
  frontend shows it. Hiding small samples would be a lie by omission; the numbers
  are still the best available answer to the question asked.

## 7. Natural language filters: rules first, model second

"Bayern at home, last fifteen minutes, trailing" has to become structured
filters. The rule parser does entity matching with RapidFuzz `fuzz.ratio` over
longest-first n-grams against a catalogue built at ingest, plus an alias file,
plus patterns for time ranges, game state, stage and roles.

Three things that cost real debugging time:

- **`fuzz.ratio`, not `fuzz.WRatio`.** `WRatio` does partial matching, which
  cheerfully matched "rainy" to a team name. Full-string ratio with a threshold
  of 84 handles real typos ("Marocco", "Engalnd" — both around 85.7) without
  inventing matches.
- **Ambiguity produces a warning and no filter.** "United" and "Real" match
  several teams. Silently picking the most frequent one is worse than saying
  "ambiguous: did you mean …", because the user cannot see that a choice was
  made.
- **Order of consumption matters.** "the final ten minutes" matched the knockout
  `final` stage until stage matching was moved to run on the residual text, after
  time and game-state phrases have been consumed.

The LLM fallback (`filters/llm.py`, off by default) is only consulted for text
the rules leave unresolved. It uses strict tool use — `strict: true`,
`additionalProperties: false`, every field required — so the model returns a
schema-valid object or nothing; every id it returns is validated against the
catalogue before being applied; the catalogue block is sent with
`cache_control: ephemeral` because it is large and unchanging; the user's text is
wrapped in `<user_text>` tags so a phrase like "ignore your instructions" is data,
not instruction; and the parser never raises — a failed fallback degrades to the
rule result. Model: `claude-opus-5`.

## 8. Why the in-memory store exists

It is not a mock. It is (a) the exact-search ground truth that ANN recall is
measured against, (b) the reason the whole demo runs with no Docker and no
Postgres, and (c) a second implementation that the integration tier compares against
Postgres frame by frame - `test_filter_semantics_match_the_memory_store` over
nine filter shapes and `test_exact_search_agrees_with_the_memory_store` over
ranking and distances - which is how divergence in filter semantics or distance
metrics gets caught. The cost is one extra implementation of the `Store` protocol; the benefit
is that "is the ANN index actually returning the right answers?" is a question
with a committed numeric answer.

## 9. What the tests are for

The layers exist because they catch different things, and each one has caught
something real:

- **Property tests** (Hypothesis) found that `E2`'s `known_fraction` was not
  mirror-symmetric, by falsifying a property that was itself slightly wrong —
  `E2` is side-agnostic by design, so the test had to be split into "spatial
  encodings change under mirroring" and "the tactical encoding only moves its two
  ball-side features".
- **Golden snapshots** pin encoder output, so a refactor that changes a bin edge
  cannot pass quietly.
- **End-to-end tests** found that a missing URL parameter became a filter on
  competition id 0, because `Number('')` is `0` — a bug invisible from the
  backend, which was answering exactly the query it was sent.
- **The fixture generator's own tests** found that events drew a player's `id`
  and `name` independently, producing frames whose player identities did not
  match. That made surnames ambiguous and dropped filter accuracy to 0.903; with
  one draw per defender it is 1.000.
- **The integration tier found that it was not running.** Its `pg_dsn` fixture
  used `return dsn` when a DSN was supplied, and `return` inside a generator
  fixture ends it without producing a value — so the tier was green and tested
  nothing in exactly the configuration it exists for. Two real Postgres bugs were
  waiting behind it: `SET LOCAL hnsw.ef_search = %s` (SET takes no parameters, so
  the placeholder reached the server as `$1`), and the N+1 in section 10.

A lesson in there worth stating plainly: a passing test tier is not evidence
until you have watched it fail. Both of those bugs would have shipped.

## 10. What only the real backend tells you

The in-memory store made every stage measurable early, and it hid two things that
only appeared the first time the benchmark ran against Postgres.

**Recall was untested, not good.** Against an exact store, recall is 1.0 by
definition. Measured properly — pgvector, HNSW, `ef_search=80`, 200 queries per
configuration — it is also 1.0 at k=10 and k=50, filtered and unfiltered. That is
a real result and a small one: 5,018 vectors is few enough that HNSW's graph walk
is nearly exhaustive, so the honest reading is "the index is not losing anything
at this corpus size", not "HNSW recall is solved". The recall gate lives in the
nightly job because that is the only place with an index to be approximate about.

**Aggregation was an N+1.** The first Postgres run put the aggregate stage at
22–30ms p95 against a 10ms budget, while search itself took 3ms. `_assemble`
called `store.match()` and `store.frame()` inside the per-result loop: two round
trips per hit, twenty per query, and about 27ms of latency that no amount of
index tuning would have fixed. Batched into two keyed reads, the stage drops to
7.1ms p95 and the end-to-end p95 from 34.6ms to 11.9ms on E1 (the committed
report: `bench/reports/postgres/benchmark.md`; the absolute numbers move a little
run to run on a laptop, the ratio does not).

The in-memory store could never have shown this - a dict lookup in a loop is
free, which is exactly why the second implementation has to be the slow one when
it matters.

## 11. pgvector or FAISS

Benchmarked rather than argued about (`bench/faiss_compare.py`, optional
dependency). With the same HNSW parameters over the same vectors, FAISS searches
the 72-dimensional E1 index at 0.22ms p95 against pgvector's 3.5ms - roughly
sixteen times faster - at recall 0.998 against exact search, against pgvector's
1.000.

It still loses, for reasons that have nothing to do with speed:

- **Filters.** Every real query carries filters, and they are answered from
  columns on the frame row. A FAISS index holds vectors and integer ids; the
  filtered-subset path - count first, scan exactly when the subset is small -
  simply does not exist, and reimplementing it outside the database means keeping
  a second copy of the filter columns in sync.
- **One process, one truth.** pgvector's index lives next to the rows it indexes
  and is rebuilt by the same batch job in the same transaction. A FAISS index is
  a separate artefact with a separate lifecycle, which is a second thing to
  deploy, warm and invalidate.
- **The budget does not care.** Search is 3.5ms of an 11.9ms end-to-end p95
  against a 150ms target. Spending an operational component to reclaim 4ms is a
  bad trade, and would stay a bad trade at ten times the corpus.

The number to watch is corpus size. At 5,018 vectors both are effectively exact;
FAISS's advantage is constant-factor, not asymptotic. If the index grew by two
orders of magnitude and search came to dominate the budget, this is the first
decision to revisit - which is why the comparison is in the benchmark harness and
not in a footnote.
