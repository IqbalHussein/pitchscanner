"""The query flow: encode, over-fetch, filter, dedup, aggregate.

Timings for each stage are returned to the caller. They drive the benchmark
report and the latency readout in the UI, and they are the first thing to look
at when a query gets slow.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from pitchscanner.baselines import as_mapping
from pitchscanner.config import Settings, get_settings
from pitchscanner.encoders import get_encoder
from pitchscanner.geometry import Situation, mirror, zone_id, zone_label
from pitchscanner.models import (
    Filters,
    Outcome,
    SearchRequest,
    SearchResult,
    Stats,
    Timing,
)
from pitchscanner.stats import Neighbour, aggregate
from pitchscanner.store.base import Candidate, Store


@dataclass(slots=True)
class SearchOutcome:
    results: list[SearchResult]
    stats: Stats | None
    filters: Filters
    encoding: str
    candidates_examined: int
    relaxed: bool
    timing: Timing
    notes: list[str] = field(default_factory=list)


class SearchService:
    def __init__(self, store: Store, settings: Settings | None = None) -> None:
        self.store = store
        self.cfg = settings or get_settings()
        self._baseline_cache: dict[str, dict[int, dict[str, float]]] = {}

    # ----------------------------------------------------------------- helpers
    def _baselines(self, encoding_version: str) -> dict[int, dict[str, float]]:
        if encoding_version not in self._baseline_cache:
            self._baseline_cache[encoding_version] = as_mapping(
                self.store.baselines(encoding_version)
            )
        return self._baseline_cache[encoding_version]

    def _dedup_by_possession(
        self, candidates: list[Candidate]
    ) -> tuple[list[Candidate], int]:
        """Keep the closest frame per possession.

        Consecutive frames inside one possession are nearly identical vectors, so
        without this the top 50 is routinely six plays repeated. Dropped count is
        returned because it is what justifies the over-fetch multiplier.
        """

        metas = self.store.metas([c.event_id for c in candidates])
        best: dict[tuple[int, int], Candidate] = {}
        order: list[tuple[int, int]] = []
        dropped = 0
        for c in candidates:
            meta = metas.get(c.event_id)
            if meta is None:
                continue
            key = (meta.match_id, meta.possession_id)
            current = best.get(key)
            if current is None:
                best[key] = c
                order.append(key)
            else:
                dropped += 1
                if c.distance < current.distance:
                    best[key] = c
        kept = sorted((best[k] for k in order), key=lambda c: c.distance)
        return kept, dropped

    def _fetch(
        self,
        encoding_version: str,
        vector: np.ndarray,
        limit: int,
        metric: str,
        filters: Filters | None,
    ) -> tuple[list[Candidate], bool]:
        """ANN by default; exact scan when the filtered subset is small.

        A strict filter can leave so few rows that graph traversal never reaches
        them and recall collapses. Below the threshold an exact scan of the
        filtered subset is both correct and cheap, so it wins outright.
        """

        if filters is not None and not filters.is_empty():
            subset = self.store.count_matching(encoding_version, filters)
            if subset <= self.cfg.exact_fallback_threshold:
                return (
                    self.store.exact_search(
                        encoding_version, vector, limit, metric, filters
                    ),
                    True,
                )
        return self.store.ann_search(encoding_version, vector, limit, metric, filters), False

    # -------------------------------------------------------------------- main
    def search(self, request: SearchRequest, filters: Filters | None = None) -> SearchOutcome:
        cfg = self.cfg
        active = filters if filters is not None else request.filters
        encoder = get_encoder(request.options.encoding)
        notes: list[str] = []

        t0 = time.perf_counter()
        situation = request.to_situation()
        vector = encoder.encode(situation)
        # Mirroring doubles the candidate pool for free: a left-wing sketch also
        # searches the right wing, which matters a lot at this dataset size.
        mirrored_vector = encoder.encode(mirror(situation)) if request.options.mirror else None
        t_encode = (time.perf_counter() - t0) * 1000

        multiplier = cfg.candidate_multiplier
        kept: list[Candidate] = []
        examined = 0
        used_exact = False
        t_search = 0.0
        t_dedup = 0.0

        while True:
            limit = min(max(request.k * multiplier, request.k), 10_000)
            t1 = time.perf_counter()
            raw, exact = self._fetch(
                encoder.version, vector, limit, encoder.metric, active
            )
            if mirrored_vector is not None:
                extra, exact_m = self._fetch(
                    encoder.version, mirrored_vector, limit, encoder.metric, active
                )
                exact = exact and exact_m
                seen = {c.event_id for c in raw}
                raw = raw + [c for c in extra if c.event_id not in seen]
                raw.sort(key=lambda c: c.distance)
            t_search += (time.perf_counter() - t1) * 1000
            used_exact = used_exact or exact
            examined += len(raw)

            t2 = time.perf_counter()
            kept = raw
            if request.options.dedup:
                kept, _ = self._dedup_by_possession(raw)
            t_dedup += (time.perf_counter() - t2) * 1000

            if len(kept) >= request.k or multiplier >= cfg.max_candidate_multiplier:
                break
            # Not enough survivors: widen the net rather than return a short list.
            multiplier = min(multiplier * 4, cfg.max_candidate_multiplier)

        kept = kept[: request.k]
        if len(kept) < request.k:
            notes.append(
                f"Only {len(kept)} distinct possessions match this sketch and these filters."
            )

        t3 = time.perf_counter()
        results, stats = self._assemble(kept, encoder.version, situation)
        t_aggregate = (time.perf_counter() - t3) * 1000

        if stats is not None:
            stats.notes = notes + stats.notes

        return SearchOutcome(
            results=results,
            stats=stats,
            filters=active,
            encoding=encoder.id,
            candidates_examined=examined,
            relaxed=used_exact,
            timing=Timing(
                encode=round(t_encode, 3),
                search=round(t_search, 3),
                dedup=round(t_dedup, 3),
                aggregate=round(t_aggregate, 3),
                total=round((time.perf_counter() - t0) * 1000, 3),
            ),
            notes=notes,
        )

    def _assemble(
        self, candidates: list[Candidate], encoding_version: str, situation: Situation
    ) -> tuple[list[SearchResult], Stats | None]:
        event_ids = [c.event_id for c in candidates]
        metas = self.store.metas(event_ids)
        outcomes = self.store.outcomes(event_ids)
        # Four batched reads, not four per result. Fetching the match and the
        # frame inside the loop cost one round trip each per hit, which on
        # Postgres put this stage at ~27ms p95 against a 10ms budget while the
        # search itself took 3ms.
        matches = self.store.matches_by_id(
            sorted({m.match_id for m in metas.values()})
        )
        frames = self.store.frames_by_id(event_ids)

        results: list[SearchResult] = []
        for c in candidates:
            meta = metas.get(c.event_id)
            if meta is None:
                continue
            match = matches.get(meta.match_id)
            frame = frames.get(c.event_id)
            if match is None:
                continue
            o = outcomes.get(c.event_id)
            team = (
                match.home_team
                if meta.possession_team_id == match.home_team_id
                else match.away_team
            )
            results.append(
                SearchResult(
                    event_id=c.event_id,
                    distance=round(c.distance, 6),
                    similarity=round(1.0 / (1.0 + max(0.0, c.distance)), 6),
                    match=f"{match.home_team} vs {match.away_team}",
                    match_id=match.match_id,
                    competition=match.competition_name,
                    season=match.season_name,
                    minute=meta.minute,
                    period=meta.period,
                    possession_id=meta.possession_id,
                    team=team,
                    mirrored=bool(frame.mirrored) if frame else False,
                    outcome=Outcome(**o.model_dump(exclude={"event_id"})) if o else None,
                )
            )

        neighbours = [
            Neighbour(outcome=outcomes[c.event_id], distance=c.distance)
            for c in candidates
            if c.event_id in outcomes
        ]
        if not neighbours:
            return results, None

        zone = zone_id(situation.ball.x, situation.ball.y)
        baseline = self._baselines(encoding_version).get(zone)
        agg = aggregate(
            neighbours,
            baseline=baseline,
            resamples=self.cfg.bootstrap_resamples,
            seed=self.cfg.bootstrap_seed,
            low_confidence_n=self.cfg.low_confidence_n,
        )
        n_possessions = len(
            {(metas[c.event_id].match_id, metas[c.event_id].possession_id)
             for c in candidates if c.event_id in metas}
        )
        stats = Stats(
            n=agg.n,
            n_possessions=n_possessions,
            shot_rate=agg.shot_rate,
            xg_mean=agg.xg_mean,
            turnover_rate=agg.turnover_rate,
            box_entry_rate=agg.box_entry_rate,
            goal_rate=agg.goal_rate,
            unweighted=agg.unweighted,
            zone_id=zone,
            zone_label=zone_label(zone),
            baseline_n=int((baseline or {}).get("n", 0)),
            low_confidence=agg.low_confidence,
            notes=list(agg.notes),
        )
        return results, stats
