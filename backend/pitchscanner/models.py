"""Domain records and the typed API contract FastAPI turns into OpenAPI docs."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pitchscanner.geometry import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    PlayerPos,
    Point,
    Situation,
    zone_id,
)

GameState = Literal["any", "winning", "drawing", "losing"]
Role = Literal["any", "attacking", "defending"]
Stage = Literal["group", "knockout", "final", "regular"]
MAX_PER_SIDE = 11


# --------------------------------------------------------------------------
# storage records
# --------------------------------------------------------------------------
class MatchRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    match_id: int
    competition_id: int
    competition_name: str
    season_id: int
    season_name: str
    home_team_id: int
    home_team: str
    away_team_id: int
    away_team: str
    match_date: date | None = None
    stage: Stage = "regular"


class EventRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    match_id: int
    possession_id: int
    index: int
    team_id: int
    player_id: int | None
    player_name: str | None
    type: str
    x: float | None
    y: float | None
    ts: float
    period: int
    minute: int
    xg: float | None = None
    goal: bool = False
    possession_team_id: int | None = None


class FrameRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    match_id: int
    possession_id: int
    players: tuple[PlayerPos, ...]
    visible_area: tuple[Point, ...] | None
    ball_x: float
    ball_y: float
    mirrored: bool = False

    def to_situation(self) -> Situation:
        return Situation(
            ball=Point(self.ball_x, self.ball_y),
            players=self.players,
            visible_area=self.visible_area,
            mirrored=self.mirrored,
        )

    @property
    def zone_id(self) -> int:
        return zone_id(self.ball_x, self.ball_y)


class OutcomeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    shot: bool
    xg_next: float
    goal: bool
    turnover: bool
    box_entry: bool
    seconds_to_end: float


class ZoneBaseline(BaseModel):
    model_config = ConfigDict(frozen=True)

    zone_id: int
    encoding_version: str
    n: int
    shot_rate: float
    xg_mean: float
    goal_rate: float
    turnover_rate: float
    box_entry_rate: float


# --------------------------------------------------------------------------
# API models
# --------------------------------------------------------------------------
Coordinate = Annotated[float, Field(ge=0.0)]


class PointIn(BaseModel):
    x: Coordinate = Field(le=PITCH_LENGTH, description="0 at own goal line, 120 at the target goal")
    y: Coordinate = Field(le=PITCH_WIDTH)
    keeper: bool = False


class Filters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    competition_ids: list[int] = Field(default_factory=list)
    season_ids: list[int] = Field(default_factory=list)
    team_ids: list[int] = Field(default_factory=list)
    team_role: Role = "any"
    player_ids: list[int] = Field(default_factory=list)
    player_role: Role = "any"
    game_state: GameState = "any"
    minute_range: tuple[int, int] | None = None
    stages: list[Stage] = Field(default_factory=list)
    ball_zone_ids: list[int] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return self == Filters()

    @field_validator("minute_range")
    @classmethod
    def _ordered(cls, v: tuple[int, int] | None) -> tuple[int, int] | None:
        if v is not None and v[0] > v[1]:
            raise ValueError("minute_range must be (low, high)")
        return v


class SearchOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mirror: bool = True
    encoding: str = "E1"
    window_s: float = Field(default=15.0, ge=1.0, le=60.0)
    dedup: bool = True


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ball: PointIn
    attackers: list[PointIn] = Field(default_factory=list, max_length=MAX_PER_SIDE)
    defenders: list[PointIn] = Field(default_factory=list, max_length=MAX_PER_SIDE)
    k: int = Field(default=25, ge=1, le=100)
    filters: Filters = Field(default_factory=Filters)
    filter_text: str | None = Field(default=None, max_length=500)
    options: SearchOptions = Field(default_factory=SearchOptions)

    @model_validator(mode="after")
    def _needs_players(self) -> SearchRequest:
        if not self.attackers and not self.defenders:
            raise ValueError("place at least one player")
        return self

    def to_situation(self) -> Situation:
        players = tuple(
            [PlayerPos(p.x, p.y, True, p.keeper) for p in self.attackers]
            + [PlayerPos(p.x, p.y, False, p.keeper) for p in self.defenders]
        )
        return Situation(ball=Point(self.ball.x, self.ball.y), players=players)

    def cache_key(self) -> str:
        return self.model_dump_json()


class Interval(BaseModel):
    value: float
    ci95: tuple[float, float]
    vs_zone_baseline: float | None = None


class Outcome(BaseModel):
    shot: bool
    xg_next: float
    goal: bool
    turnover: bool
    box_entry: bool
    seconds_to_end: float


class SearchResult(BaseModel):
    event_id: str
    distance: float
    similarity: float
    match: str
    match_id: int
    competition: str
    season: str
    minute: int
    period: int
    possession_id: int
    team: str
    mirrored: bool
    outcome: Outcome | None = None


class Stats(BaseModel):
    n: int
    n_possessions: int
    shot_rate: Interval
    xg_mean: Interval
    turnover_rate: Interval
    box_entry_rate: Interval
    goal_rate: Interval
    unweighted: dict[str, float]
    zone_id: int
    zone_label: str
    baseline_n: int
    low_confidence: bool
    notes: list[str] = Field(default_factory=list)


class Chip(BaseModel):
    field: str
    label: str
    value: str
    ids: list[int] = Field(default_factory=list)


class ParsedFilters(BaseModel):
    filters: Filters
    chips: list[Chip]
    unmatched: list[str]
    warnings: list[str]
    source: Literal["rules", "llm", "none"] = "rules"


class Timing(BaseModel):
    encode: float
    search: float
    dedup: float
    aggregate: float
    total: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    stats: Stats | None
    parsed_filters: ParsedFilters | None = None
    filters_applied: Filters
    encoding: str
    candidates_examined: int
    relaxed: bool = False
    timing_ms: Timing


class SituationDetail(BaseModel):
    event_id: str
    match_id: int
    match: str
    competition: str
    season: str
    minute: int
    period: int
    possession_id: int
    team: str
    ball: PointIn
    attackers: list[PointIn]
    defenders: list[PointIn]
    visible_area: list[PointIn] | None
    outcome: Outcome | None


class SequenceEvent(BaseModel):
    event_id: str
    type: str
    team: str
    player: str | None
    x: float | None
    y: float | None
    second_offset: float
    xg: float | None = None
    goal: bool = False


class Sequence(BaseModel):
    event_id: str
    window_s: float
    events: list[SequenceEvent]


class SeasonCoverage(BaseModel):
    season_id: int
    season_name: str
    matches: int
    frames: int


class CompetitionCoverage(BaseModel):
    competition_id: int
    competition_name: str
    matches: int
    frames: int
    seasons: list[SeasonCoverage]


class TeamOption(BaseModel):
    team_id: int
    team_name: str
    frames: int


class FiltersResponse(BaseModel):
    competitions: list[CompetitionCoverage]
    teams: list[TeamOption]
    stages: list[Stage]
    game_states: list[GameState]
    encodings: list[str]
    total_frames: int
    total_matches: int
    total_events: int


class ParseRequest(BaseModel):
    text: str = Field(max_length=500)
