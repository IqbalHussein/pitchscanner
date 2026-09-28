# Blog outline: "Why your similarity search returns the same play eight times"

Working title, and also the hook. One post, roughly 2,500 words, four figures.
Each section below names the figure it needs and the number it is allowed to
quote — every quoted number has to exist in `bench/reports/benchmark.md` before
the post goes out.

## 1. The opening: a search that works and is useless

- Screenshot: ten results, all the same attack, three seconds apart.
- The naive design: encode every freeze frame, put the vectors in pgvector, ask
  for the ten nearest. It works on the first try, which is the problem.
- Why it happens: consecutive frames of one possession are each other's nearest
  neighbours by construction. A top-10 by distance is one moment sampled ten
  times.
- The claim the rest of the post defends: for query-by-example over time-series
  data, *diversity constraints are part of the retrieval problem*, not a
  post-processing nicety.

**Figure 1.** Before/after of the same query, with dedup off and on.

## 2. Dedup breaks the ANN contract, and what you do about it

- One result per possession, keep the closest frame. Two lines of code.
- The consequence: you can no longer ask the index for `k`. Most candidates get
  discarded, so how many do you ask for?
- Over-fetch `k x 5`, dedup, and widen by 4x up to 40x if short. Why the ceiling
  matters: an unbounded retry loop turns a 150ms p95 into a 2-second p99.
- Generalises: this is the same shape as pagination over a filtered feed, or
  MMR-style diversification in document retrieval.

## 3. "How do you compare two formations with different numbers of players?"

This is the question that gets asked in every conversation about the project, and
it has a clean answer.

- Why the obvious encoding is wrong: eleven players are a **set**. Any
  fixed-length vector indexed by player breaks under substitution, formation
  change, or a camera that only saw six of them.
- Occupancy fields: each player contributes a Gaussian bump; the encoding is a
  sum, so it is order-independent for free and degrades smoothly when players are
  missing instead of producing garbage.
- Normalise and use cosine, so the *pattern* of mass matters more than the total.
- The three encodings as three answers to the same question, with the table from
  `docs/design-notes.md`: 292-d grid, 72-d ball-relative polar, 18-d named
  tactical features.
- The jitter test as the evaluation that decides between them: nudge every player
  1–2m, does the original frame still come back first? E0 1.000, E1 0.995, E2
  0.505. That last number is the whole argument against shipping the most
  interpretable encoding as the default.

**Figure 2.** One situation rendered three ways: the grid, the polar bins, and the
18 features as a bar chart.

## 4. Unknown is not empty

- StatsBomb 360 frames come with a `visible_area` polygon. Outside it, the data
  says nothing.
- The bug this prevents: encoding unseen as empty makes stretched, transitional
  play look sparsely defended — because that is exactly when the camera narrows.
  The bias correlates with the thing you are trying to measure, which is the
  worst kind.
- An explicit unknown channel, and `known_fraction` as a feature.
- Sketches have no camera: a drawn token asserts an 18m radius of knowledge and
  nothing beyond it.
- The property test that caught corner-sampling asymmetry in `known_fraction`,
  and why "this feature should be invariant under reflection" is a better test
  than any example.

**Figure 3.** A freeze frame with its visible-area polygon overlaid, and the same
frame's unknown channel.

## 5. Filters and ANN indexes do not get along

- What HNSW actually does with a filter that removes 94% of the corpus, and why
  post-filtering `k` results gives you three.
- Count first, then choose: below 2,000 matching rows, sequential scan of the
  filtered subset with `enable_indexscan` off. Brute force is both faster and
  exact at that size, so narrow filters become the *best* case rather than the
  worst.
- Denormalised filter columns on the frame row: joining against the candidate set
  from an index is how you spend your entire latency budget.

## 6. Numbers that are allowed to be uncertain

- The temptation: "this shape produces a shot 62% of the time." From 13
  neighbours.
- Similarity weighting (`1/(1+d)`), and the honesty tax it incurs: Kish effective
  sample size, reported next to the raw count.
- Wilson intervals rather than normal approximations, because goal rates live
  near zero where the normal interval returns negative lower bounds.
- Seeded bootstrap for xG means — an interval that changes on reload costs more
  trust than a wide one.
- Zone baselines: 18 zones, every rate also reported as a ratio to its zone, so
  "dangerous" is relative to somewhere.
- `low_confidence` as a visible flag rather than a filter. Show the number, show
  that it is thin.

**Figure 4.** The statistics panel: value, interval, baseline ratio, effective n,
and a low-confidence badge.

## 7. Plain-language filters without handing the problem to a model

- Rules first: fuzzy entity matching against a catalogue built at ingest, plus
  time, game state, stage and role patterns. 145 hand-labelled phrases, 1.000
  exact match, checked in CI.
- Three bugs worth the retelling: `WRatio` matching "rainy" to a team; "United"
  resolving silently to one of several clubs; "the final ten minutes" matching
  the knockout `final`.
- Ambiguity as a first-class outcome: warn, apply nothing.
- Where the model earns its place: only the residue. Strict tool use, every
  returned id validated against the catalogue, user text wrapped in tags,
  catalogue block cached, and a fallback that can never raise.
- The general lesson: a model is a good last resort and a bad first one, and the
  labelled phrase set is what let that be an empirical claim instead of taste.

## 8. Closing: what the benchmark file is for

- Every number in the README comes from a committed report, and CI fails the
  build on a p95 regression against it.
- The recall caveat as the example: the in-memory store is exact, so its recall
  numbers are 1.0 by definition and mean nothing. The report says so itself, in
  the section where the numbers are.
- The limitations section, in full, and why it is at the top of the README rather
  than at the bottom of a footnote: the demo runs on four synthetic matches, and
  a portfolio project that pretends otherwise is the failure mode.

## Repurposing

- **Talk version (20 min):** sections 1, 3 and 6 only, one figure each.
- **Short post:** section 3 alone stands up as "your vector encoding probably
  assumes an order that isn't there".
