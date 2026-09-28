"""Normalisation: one orientation for every frame, plus optional mirroring.

Both transforms are idempotent in the sense the property tests require:
normalising twice equals normalising once, and mirroring twice is the identity.
"""

from __future__ import annotations

from pitchscanner.geometry import Situation, mirror, orient_left_to_right, zone_id
from pitchscanner.models import EventRecord, FrameRecord, MatchRecord
from pitchscanner.store.base import FrameMeta


def normalise_frame(frame: FrameRecord, *, attacks_right: bool = True) -> FrameRecord:
    """Put the team in possession attacking towards x = 120."""

    situation = orient_left_to_right(frame.to_situation(), attacks_right=attacks_right)
    return _from_situation(frame, situation)


def mirror_frame(frame: FrameRecord) -> FrameRecord:
    """Top-to-bottom reflection, stored as a separate row with mirrored = True."""

    return _from_situation(frame, mirror(frame.to_situation()))


def _from_situation(frame: FrameRecord, situation: Situation) -> FrameRecord:
    return FrameRecord(
        event_id=frame.event_id,
        match_id=frame.match_id,
        possession_id=frame.possession_id,
        players=situation.players,
        visible_area=situation.visible_area,
        ball_x=situation.ball.x,
        ball_y=situation.ball.y,
        mirrored=situation.mirrored,
    )


def game_state_at(
    goals: list[tuple[int, float, int]], team_id: int, period: int, ts: float
) -> str:
    """Score state from the point of view of the team in possession."""

    for_team = sum(1 for p, t, tid in goals if tid == team_id and (p, t) <= (period, ts))
    against = sum(1 for p, t, tid in goals if tid != team_id and (p, t) <= (period, ts))
    if for_team > against:
        return "winning"
    if for_team < against:
        return "losing"
    return "drawing"


def frame_meta(
    frame: FrameRecord,
    event: EventRecord,
    match: MatchRecord,
    goals: list[tuple[int, float, int]],
) -> FrameMeta:
    """Build the denormalised filter row that sits next to the frame."""

    possession_team = event.possession_team_id or event.team_id
    opponent = (
        match.away_team_id if possession_team == match.home_team_id else match.home_team_id
    )
    return FrameMeta(
        event_id=frame.event_id,
        match_id=frame.match_id,
        possession_id=frame.possession_id,
        competition_id=match.competition_id,
        season_id=match.season_id,
        possession_team_id=possession_team,
        opponent_team_id=opponent,
        player_id=event.player_id,
        minute=event.minute,
        period=event.period,
        stage=match.stage,
        game_state=game_state_at(goals, possession_team, event.period, event.ts),
        zone_id=zone_id(frame.ball_x, frame.ball_y),
    )
