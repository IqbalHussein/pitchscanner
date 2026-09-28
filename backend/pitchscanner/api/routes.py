"""The REST surface. Typed in, typed out; FastAPI generates the OpenAPI docs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response

from pitchscanner.api import metrics
from pitchscanner.api.deps import AppState, get_state
from pitchscanner.encoders import available as available_encodings
from pitchscanner.geometry import PITCH_LENGTH, PITCH_WIDTH
from pitchscanner.labeller import DEFAULT_WINDOW_S
from pitchscanner.models import (
    CompetitionCoverage,
    FiltersResponse,
    Outcome,
    ParsedFilters,
    ParseRequest,
    PointIn,
    SearchRequest,
    SearchResponse,
    SeasonCoverage,
    Sequence,
    SequenceEvent,
    SituationDetail,
    TeamOption,
)

router = APIRouter()


@router.post("/v1/search", response_model=SearchResponse, summary="Search by sketch")
def search(request: SearchRequest, state: AppState = Depends(get_state)) -> SearchResponse:
    """Sketch in, the most similar real situations plus aggregated outcomes out."""

    if request.k > state.settings.max_k:
        raise HTTPException(422, f"k must be <= {state.settings.max_k}")
    if request.options.encoding.upper() not in available_encodings():
        raise HTTPException(422, f"unknown encoding; available: {available_encodings()}")

    key = request.cache_key()
    cached = state.cache.get(key)
    if cached is not None:
        metrics.CACHE_HITS.inc()
        assert isinstance(cached, SearchResponse)
        return cached
    metrics.CACHE_MISSES.inc()

    filters, parsed = state.filters.resolve(request.filters, request.filter_text)
    outcome = state.search.search(request, filters=filters)

    for stage, value in (
        ("encode", outcome.timing.encode),
        ("search", outcome.timing.search),
        ("dedup", outcome.timing.dedup),
        ("aggregate", outcome.timing.aggregate),
    ):
        metrics.STAGE_LATENCY.labels(stage=stage).observe(value / 1000.0)
    if outcome.relaxed:
        metrics.EXACT_FALLBACKS.inc()

    response = SearchResponse(
        results=outcome.results,
        stats=outcome.stats,
        parsed_filters=parsed,
        filters_applied=outcome.filters,
        encoding=outcome.encoding,
        candidates_examined=outcome.candidates_examined,
        relaxed=outcome.relaxed,
        timing_ms=outcome.timing,
    )
    state.cache.put(key, response)
    return response


@router.post(
    "/v1/filters/parse", response_model=ParsedFilters, summary="Parse a free-text filter"
)
def parse_filters(request: ParseRequest, state: AppState = Depends(get_state)) -> ParsedFilters:
    """Text in; structured filters, chips, unmatched words and warnings out."""

    return state.filters.parse(request.text)


@router.get("/v1/filters", response_model=FiltersResponse, summary="Available filters")
def filters(state: AppState = Depends(get_state)) -> FiltersResponse:
    """Only competitions and seasons that actually have 360 frames."""

    rows = state.store.coverage()
    by_competition: dict[int, CompetitionCoverage] = {}
    for row in rows:
        cid = int(row["competition_id"])
        comp = by_competition.get(cid)
        season = SeasonCoverage(
            season_id=int(row["season_id"]),
            season_name=str(row["season_name"]),
            matches=int(row["matches"]),
            frames=int(row["frames"]),
        )
        if comp is None:
            by_competition[cid] = CompetitionCoverage(
                competition_id=cid,
                competition_name=str(row["competition_name"]),
                matches=season.matches,
                frames=season.frames,
                seasons=[season],
            )
        else:
            comp.seasons.append(season)
            comp.matches += season.matches
            comp.frames += season.frames

    counts = state.store.counts()
    return FiltersResponse(
        competitions=sorted(by_competition.values(), key=lambda c: -c.frames),
        teams=[TeamOption(**row) for row in state.store.team_options()],
        stages=["group", "knockout", "final", "regular"],
        game_states=["any", "winning", "drawing", "losing"],
        encodings=list(available_encodings()),
        total_frames=counts.frames,
        total_matches=counts.matches,
        total_events=counts.events,
    )


@router.get(
    "/v1/situations/{event_id}", response_model=SituationDetail, summary="One frame"
)
def situation(event_id: str, state: AppState = Depends(get_state)) -> SituationDetail:
    frame = state.store.frame(event_id)
    meta = state.store.meta(event_id)
    if frame is None or meta is None:
        raise HTTPException(404, "no such situation")
    match = state.store.match(meta.match_id)
    if match is None:
        raise HTTPException(404, "no such match")
    outcome = state.store.outcome(event_id)
    team = match.home_team if meta.possession_team_id == match.home_team_id else match.away_team
    return SituationDetail(
        event_id=event_id,
        match_id=match.match_id,
        match=f"{match.home_team} vs {match.away_team}",
        competition=match.competition_name,
        season=match.season_name,
        minute=meta.minute,
        period=meta.period,
        possession_id=meta.possession_id,
        team=team,
        ball=PointIn(x=frame.ball_x, y=frame.ball_y),
        attackers=[PointIn(x=p.x, y=p.y, keeper=p.keeper) for p in frame.players if p.teammate],
        defenders=[
            PointIn(x=p.x, y=p.y, keeper=p.keeper) for p in frame.players if not p.teammate
        ],
        visible_area=(
            [PointIn(x=p.x, y=p.y) for p in frame.visible_area] if frame.visible_area else None
        ),
        outcome=Outcome(**outcome.model_dump(exclude={"event_id"})) if outcome else None,
    )


@router.get(
    "/v1/situations/{event_id}/sequence",
    response_model=Sequence,
    summary="What happened next",
)
def sequence(
    event_id: str, window_s: float = DEFAULT_WINDOW_S, state: AppState = Depends(get_state)
) -> Sequence:
    """The events that followed, for playback."""

    anchor = state.store.event(event_id)
    if anchor is None:
        raise HTTPException(404, "no such event")
    match = state.store.match(anchor.match_id)
    events = [
        e
        for e in state.store.possession_events(event_id)
        if e.index >= anchor.index and e.period == anchor.period and e.ts - anchor.ts <= window_s
    ]
    home = match.home_team if match else ""
    away = match.away_team if match else ""
    return Sequence(
        event_id=event_id,
        window_s=window_s,
        events=[
            SequenceEvent(
                event_id=e.event_id,
                type=e.type,
                team=home if match and e.team_id == match.home_team_id else away,
                player=e.player_name,
                x=e.x,
                y=e.y,
                second_offset=round(e.ts - anchor.ts, 2),
                xg=e.xg,
                goal=e.goal,
            )
            for e in events
        ],
    )


@router.get("/v1/meta", summary="Dataset counts and pitch dimensions")
def meta(state: AppState = Depends(get_state)) -> dict[str, object]:
    """Real counts, so the README and the resume can quote something committed."""

    counts = state.store.counts()
    return {
        "counts": counts.as_dict(),
        "pitch": {"length": PITCH_LENGTH, "width": PITCH_WIDTH},
        "encodings": list(available_encodings()),
        "default_encoding": state.settings.encoding,
        "index_kind": state.settings.index_kind,
        "outcome_window_s": state.settings.outcome_window_s,
        "low_confidence_n": state.settings.low_confidence_n,
        "attribution": "Data: StatsBomb Open Data, free for non-commercial use.",
    }


@router.get("/healthz", summary="Liveness")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz", summary="Readiness: database reachable and index loaded")
def readyz(response: Response, state: AppState = Depends(get_state)) -> dict[str, object]:
    try:
        counts = state.store.counts()
    except Exception as exc:
        response.status_code = 503
        metrics.ERRORS.labels(kind="readiness").inc()
        return {"status": "unavailable", "error": str(exc)}
    ready = counts.frames > 0
    if not ready:
        response.status_code = 503
    return {"status": "ready" if ready else "empty", "counts": counts.as_dict()}


@router.get("/metrics", summary="Prometheus metrics")
def prometheus() -> Response:
    payload, content_type = metrics.render()
    return Response(content=payload, media_type=content_type)
