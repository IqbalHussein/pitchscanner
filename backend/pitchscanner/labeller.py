"""Outcome labelling. Runs once in batch so a query only aggregates labels.

For each frame, scan forward through the events of the same possession until the
possession ends or the window runs out, and record what happened.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pitchscanner.geometry import in_penalty_area
from pitchscanner.models import EventRecord, FrameRecord, OutcomeRecord

DEFAULT_WINDOW_S = 15.0

#: Event types that mean the ball changed hands. A foul won or a ball out of
#: play is *not* a turnover: possession returns to the same team, so counting
#: those would inflate the turnover rate of every set-piece situation.
TURNOVER_TYPES = frozenset(
    {
        "Dispossessed",
        "Miscontrol",
        "Interception",
        "Ball Recovery",
        "Duel",
        "Clearance",
        "Pass",  # only when the pass is incomplete; checked via `type_detail`
    }
)
NEUTRAL_TYPES = frozenset({"Foul Won", "Foul Committed", "Injury Stoppage", "Substitution"})


def _events_after(
    events: Sequence[EventRecord], frame: FrameRecord, window_s: float
) -> list[EventRecord]:
    """Events of the same possession, same half, inside the window, in order.

    A half ending mid-window truncates the window: play does not continue, so
    crediting a shot from after the break to this situation would be wrong.
    """

    anchor = next((e for e in events if e.event_id == frame.event_id), None)
    if anchor is None:
        return []
    return [
        e
        for e in events
        if e.possession_id == frame.possession_id
        and e.period == anchor.period
        and e.index > anchor.index
        and e.ts - anchor.ts <= window_s
    ]


def label_frame(
    frame: FrameRecord,
    possession_events: Sequence[EventRecord],
    *,
    window_s: float = DEFAULT_WINDOW_S,
    possession_team_id: int | None = None,
) -> OutcomeRecord:
    """Label one frame from the events that followed it."""

    anchor = next((e for e in possession_events if e.event_id == frame.event_id), None)
    if anchor is None:
        return OutcomeRecord(
            event_id=frame.event_id,
            shot=False,
            xg_next=0.0,
            goal=False,
            turnover=False,
            box_entry=False,
            seconds_to_end=0.0,
        )
    team = possession_team_id or anchor.possession_team_id or anchor.team_id
    following = _events_after(possession_events, frame, window_s)

    shot = False
    goal = False
    xg_next = 0.0
    box_entry = in_penalty_area(frame.ball_x, frame.ball_y)
    turnover = False
    end_ts = anchor.ts + window_s

    for e in following:
        if e.type == "Shot" and e.team_id == team:
            shot = True
            xg_next += e.xg or 0.0
            goal = goal or e.goal
            if end_ts > e.ts:
                end_ts = e.ts
            break
        if e.x is not None and e.y is not None and e.team_id == team:
            box_entry = box_entry or in_penalty_area(e.x, e.y)
        if e.team_id != team and e.type not in NEUTRAL_TYPES:
            turnover = True
            if end_ts > e.ts:
                end_ts = e.ts
            break

    if not shot and not turnover and following:
        last = following[-1]
        # The possession ended inside the window without a shot: that is a
        # turnover in every sense that matters for "what happened next".
        if last.ts - anchor.ts < window_s:
            turnover = True
            end_ts = last.ts

    return OutcomeRecord(
        event_id=frame.event_id,
        shot=shot,
        xg_next=round(xg_next, 6),
        goal=goal,
        turnover=turnover and not shot,
        box_entry=box_entry,
        seconds_to_end=round(max(0.0, end_ts - anchor.ts), 3),
    )


def label_match(
    frames: Iterable[FrameRecord],
    events: Sequence[EventRecord],
    *,
    window_s: float = DEFAULT_WINDOW_S,
) -> list[OutcomeRecord]:
    """Label every frame of a match. Indexed by possession to stay linear-ish."""

    by_possession: dict[int, list[EventRecord]] = {}
    for e in events:
        by_possession.setdefault(e.possession_id, []).append(e)
    for bucket in by_possession.values():
        bucket.sort(key=lambda e: e.index)

    team_of: dict[int, int | None] = {
        pid: (bucket[0].possession_team_id or bucket[0].team_id)
        for pid, bucket in by_possession.items()
        if bucket
    }

    return [
        label_frame(
            f,
            by_possession.get(f.possession_id, []),
            window_s=window_s,
            possession_team_id=team_of.get(f.possession_id),
        )
        for f in frames
    ]
