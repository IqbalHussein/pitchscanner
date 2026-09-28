"""The storage contract.

Two implementations: Postgres + pgvector for the real system, and an in-process
one used by unit tests, the fixture demo and the exact-search ground truth. The
protocol is what keeps the search service from caring which is behind it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypedDict

import numpy as np

from pitchscanner.models import (
    EventRecord,
    Filters,
    FrameRecord,
    MatchRecord,
    OutcomeRecord,
    ZoneBaseline,
)


@dataclass(frozen=True, slots=True)
class FrameMeta:
    """Denormalised filter columns, stored on the frame row.

    Filters and ANN interact badly, so every predicate the API supports has to
    be answerable without a join: the candidate set from the index is checked
    against one narrow row per frame.
    """

    event_id: str
    match_id: int
    possession_id: int
    competition_id: int
    season_id: int
    possession_team_id: int
    opponent_team_id: int
    player_id: int | None
    minute: int
    period: int
    stage: str
    game_state: str
    zone_id: int


class CoverageRow(TypedDict):
    """One competition/season pair that actually has 360 frames."""

    competition_id: int
    competition_name: str
    season_id: int
    season_name: str
    frames: int
    matches: int


class TeamRow(TypedDict):
    team_id: int
    team_name: str
    frames: int


@dataclass(frozen=True, slots=True)
class Candidate:
    event_id: str
    distance: float


@dataclass(slots=True)
class Counts:
    matches: int = 0
    events: int = 0
    frames: int = 0
    vectors: int = 0
    outcomes: int = 0
    possessions: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "matches": self.matches,
            "events": self.events,
            "frames": self.frames,
            "vectors": self.vectors,
            "outcomes": self.outcomes,
            "possessions": self.possessions,
        }


@dataclass(slots=True)
class CatalogueEntry:
    kind: str  # competition | season | team | player
    entity_id: int
    name: str
    aliases: list[str] = field(default_factory=list)
    parent_id: int | None = None  # season -> competition
    frames: int = 0


class Store(Protocol):
    """Everything the pipeline writes and the API reads."""

    # --- writes ------------------------------------------------------------
    def init_schema(self) -> None: ...
    def upsert_matches(self, matches: Iterable[MatchRecord]) -> int: ...
    def upsert_events(self, events: Iterable[EventRecord]) -> int: ...
    def upsert_frames(
        self, frames: Iterable[FrameRecord], metas: Iterable[FrameMeta]
    ) -> int: ...
    def upsert_outcomes(self, outcomes: Iterable[OutcomeRecord]) -> int: ...
    def upsert_vectors(
        self, encoding_version: str, dim: int, rows: Sequence[tuple[str, np.ndarray]]
    ) -> int: ...
    def upsert_baselines(self, baselines: Iterable[ZoneBaseline]) -> int: ...
    def upsert_catalogue(self, entries: Iterable[CatalogueEntry]) -> int: ...
    def build_index(self, encoding_version: str, kind: str) -> dict[str, object]: ...

    # --- reads -------------------------------------------------------------
    def counts(self) -> Counts: ...
    def match(self, match_id: int) -> MatchRecord | None: ...
    def frame(self, event_id: str) -> FrameRecord | None: ...
    def event(self, event_id: str) -> EventRecord | None: ...
    def meta(self, event_id: str) -> FrameMeta | None: ...
    def outcome(self, event_id: str) -> OutcomeRecord | None: ...
    def outcomes(self, event_ids: Sequence[str]) -> dict[str, OutcomeRecord]: ...
    def metas(self, event_ids: Sequence[str]) -> dict[str, FrameMeta]: ...
    def matches_by_id(self, match_ids: Sequence[int]) -> dict[int, MatchRecord]: ...
    def frames_by_id(self, event_ids: Sequence[str]) -> dict[str, FrameRecord]: ...
    def frames(self, limit: int | None = None) -> list[FrameRecord]: ...
    def possession_events(self, event_id: str) -> list[EventRecord]: ...
    def baselines(self, encoding_version: str) -> list[ZoneBaseline]: ...
    def catalogue(self) -> list[CatalogueEntry]: ...
    def coverage(self) -> list[CoverageRow]: ...
    def team_options(self) -> list[TeamRow]: ...

    # --- search ------------------------------------------------------------
    def ann_search(
        self,
        encoding_version: str,
        vector: np.ndarray,
        limit: int,
        metric: str,
        filters: Filters | None = None,
    ) -> list[Candidate]: ...
    def exact_search(
        self,
        encoding_version: str,
        vector: np.ndarray,
        limit: int,
        metric: str,
        filters: Filters | None = None,
    ) -> list[Candidate]: ...
    def count_matching(self, encoding_version: str, filters: Filters | None) -> int: ...
    def close(self) -> None: ...
