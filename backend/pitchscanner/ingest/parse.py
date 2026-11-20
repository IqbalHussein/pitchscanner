"""Parse and validate StatsBomb JSON. Malformed records go to a reject log.

A single bad record in a 3,000-event file must not fail a whole ingestion run, so
every record is validated and anything unusable is recorded with a reason.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pitchscanner.geometry import PlayerPos, in_pitch, polygon_from_raw
from pitchscanner.models import EventRecord, FrameRecord, MatchRecord, Stage

#: StatsBomb's competition_stage names, folded into the four the API exposes.
STAGE_MAP: dict[str, Stage] = {
    "regular season": "regular",
    "group stage": "group",
    "group phase": "group",
    "1st round": "group",
    "round of 16": "knockout",
    "last 16": "knockout",
    "quarter-finals": "knockout",
    "semi-finals": "knockout",
    "3rd place final": "knockout",
    "final": "final",
}
GOAL_OUTCOMES = {"Goal"}
OWN_GOAL_TYPES = {"Own Goal For", "Own Goal Against"}


@dataclass(slots=True)
class Reject:
    source: str
    reason: str
    payload: Any


@dataclass(slots=True)
class ParsedMatch:
    match: MatchRecord
    events: list[EventRecord] = field(default_factory=list)
    frames: list[FrameRecord] = field(default_factory=list)
    rejects: list[Reject] = field(default_factory=list)
    #: (period, ts, scoring_team_id) for every goal, used to derive game state.
    goals: list[tuple[int, float, int]] = field(default_factory=list)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def parse_timestamp(raw: str, period: int) -> float:
    """"00:12:34.567" plus the period offset, in seconds from kickoff."""

    hh, mm, rest = raw.split(":")
    ss = float(rest)
    base = int(hh) * 3600 + int(mm) * 60 + ss
    # Periods restart the clock; offsetting keeps ordering and windows sane.
    offsets = {1: 0.0, 2: 45 * 60.0, 3: 90 * 60.0, 4: 105 * 60.0, 5: 120 * 60.0}
    return base + offsets.get(period, 0.0)


def stage_of(raw: str | None) -> Stage:
    """Fold StatsBomb's stage names into the four the API exposes."""

    return STAGE_MAP.get((raw or "").strip().lower(), "regular")


def parse_matches(
    path: Path, *, competition_id: int, season_id: int
) -> tuple[list[MatchRecord], list[Reject]]:
    out: list[MatchRecord] = []
    rejects: list[Reject] = []
    for raw in load_json(path):
        try:
            out.append(
                MatchRecord(
                    match_id=int(raw["match_id"]),
                    competition_id=int(
                        raw.get("competition", {}).get("competition_id", competition_id)
                    ),
                    competition_name=str(
                        raw.get("competition", {}).get("competition_name", "Unknown")
                    ),
                    season_id=int(raw.get("season", {}).get("season_id", season_id)),
                    season_name=str(raw.get("season", {}).get("season_name", "Unknown")),
                    home_team_id=int(raw["home_team"]["home_team_id"]),
                    home_team=str(raw["home_team"]["home_team_name"]),
                    away_team_id=int(raw["away_team"]["away_team_id"]),
                    away_team=str(raw["away_team"]["away_team_name"]),
                    match_date=raw.get("match_date"),
                    stage=stage_of((raw.get("competition_stage") or {}).get("name")),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            rejects.append(Reject(str(path), f"bad match record: {exc}", raw))
    return out, rejects


def parse_events(
    path: Path, match_id: int
) -> tuple[list[EventRecord], list[tuple[int, float, int]], list[Reject]]:
    events: list[EventRecord] = []
    goals: list[tuple[int, float, int]] = []
    rejects: list[Reject] = []
    for raw in load_json(path):
        try:
            type_name = str(raw["type"]["name"])
            period = int(raw["period"])
            ts = parse_timestamp(str(raw["timestamp"]), period)
            location = raw.get("location") or [None, None]
            shot = raw.get("shot") or {}
            outcome = str((shot.get("outcome") or {}).get("name", ""))
            is_goal = type_name == "Shot" and outcome in GOAL_OUTCOMES
            team_id = int(raw["team"]["id"])
            events.append(
                EventRecord(
                    event_id=str(raw["id"]),
                    match_id=match_id,
                    possession_id=int(raw.get("possession", 0)),
                    index=int(raw.get("index", 0)),
                    team_id=team_id,
                    player_id=(raw.get("player") or {}).get("id"),
                    player_name=(raw.get("player") or {}).get("name"),
                    type=type_name,
                    x=float(location[0]) if location[0] is not None else None,
                    y=float(location[1]) if location[1] is not None else None,
                    ts=ts,
                    period=period,
                    minute=int(raw.get("minute", 0)),
                    xg=float(shot["statsbomb_xg"]) if "statsbomb_xg" in shot else None,
                    goal=is_goal,
                    possession_team_id=(raw.get("possession_team") or {}).get("id"),
                )
            )
            if is_goal:
                goals.append((period, ts, team_id))
            elif type_name in OWN_GOAL_TYPES:
                # An own goal credits the other team; "Own Goal For" is already
                # recorded against the benefiting team by StatsBomb.
                goals.append((period, ts, team_id))
        except (KeyError, TypeError, ValueError) as exc:
            rejects.append(Reject(str(path), f"bad event record: {exc}", raw))
    events.sort(key=lambda e: (e.period, e.index))
    return events, goals, rejects


#: Where a three-sixty frame object begins, whatever the file's whitespace style.
_FRAME_START = re.compile(r'\{\s*"event_uuid"\s*:')


@dataclass(frozen=True, slots=True)
class Salvage:
    """What a tolerant read of a damaged three-sixty file got back."""

    recovered: int
    #: Frames the damage swallowed, estimated from the size of the damaged span
    #: and the mean size of the frames around it.
    lost_estimate: int

    @property
    def reason(self) -> str:
        return (
            f"salvaged three-sixty file: recovered {self.recovered} frames, "
            f"lost ~{self.lost_estimate}"
        )


def salvage_json_array(text: str) -> tuple[list[Any], Salvage] | None:
    """Recover the complete objects of a top-level JSON array that has been damaged.

    Only for a file the strict parser has already rejected. Objects are decoded one
    at a time with ``JSONDecoder.raw_decode``; where one fails (a NUL run, which is
    never valid JSON, lands inside it) the scan resumes at the next frame start. A
    damaged object is dropped whole, never repaired: a number cut in half could
    still parse as a different number, so nothing that touched the damage is kept.

    Returns ``None`` when the text is not an array of objects to begin with.
    """

    decoder = json.JSONDecoder()
    n = len(text)
    pos = 0
    while pos < n and text[pos].isspace():
        pos += 1
    if pos >= n or text[pos] != "[":
        return None
    pos += 1

    items: list[Any] = []
    intact_chars = 0
    damaged_chars = 0
    broke_an_object = False
    while pos < n:
        while pos < n and (text[pos].isspace() or text[pos] == ","):
            pos += 1
        if pos >= n or text[pos] == "]":
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            nxt = _FRAME_START.search(text, pos + 1)
            resume = nxt.start() if nxt else n
            damaged_chars += resume - pos
            broke_an_object = broke_an_object or text[pos] == "{"
            pos = resume
            continue
        if isinstance(obj, dict):
            items.append(obj)
            intact_chars += end - pos
        pos = end

    if not items:
        return None
    mean = intact_chars / len(items)
    lost = round(damaged_chars / mean) if mean else 0
    if broke_an_object:
        lost = max(lost, 1)
    return items, Salvage(recovered=len(items), lost_estimate=lost)


def read_frames_tolerant(path: Path) -> tuple[list[Any], Salvage] | None:
    """The raw frame objects of a three-sixty file whose NUL-damaged JSON will not
    parse, or ``None`` if the file is not NUL damage (a truncation, bad encoding or
    other corruption stays a skipped match)."""

    text = path.read_text(encoding="utf-8")
    if "\x00" not in text:
        return None
    return salvage_json_array(text)


def parse_frames(
    path: Path,
    match_id: int,
    events_by_id: dict[str, EventRecord],
    *,
    raw_frames: list[Any] | None = None,
) -> tuple[list[FrameRecord], list[Reject]]:
    """360 freeze frames, joined to their event for the ball position.

    ``raw_frames`` carries objects already recovered by ``read_frames_tolerant``;
    by default the file is read strictly.
    """

    frames: list[FrameRecord] = []
    rejects: list[Reject] = []
    for raw in load_json(path) if raw_frames is None else raw_frames:
        event_id = str(raw.get("event_uuid", ""))
        event = events_by_id.get(event_id)
        if event is None:
            rejects.append(Reject(str(path), "frame has no matching event", event_id))
            continue
        if event.x is None or event.y is None:
            rejects.append(Reject(str(path), "event has no location", event_id))
            continue
        players: list[PlayerPos] = []
        bad = False
        for p in raw.get("freeze_frame") or []:
            loc = p.get("location") or []
            if len(loc) < 2:
                bad = True
                continue
            x, y = float(loc[0]), float(loc[1])
            if not in_pitch(x, y):
                bad = True
                continue
            players.append(
                PlayerPos(
                    x=x,
                    y=y,
                    teammate=bool(p.get("teammate", False)),
                    keeper=bool(p.get("keeper", False)),
                )
            )
        if not players:
            rejects.append(Reject(str(path), "frame has no usable players", event_id))
            continue
        if bad:
            rejects.append(Reject(str(path), "frame had some unusable players", event_id))
        frames.append(
            FrameRecord(
                event_id=event_id,
                match_id=match_id,
                possession_id=event.possession_id,
                players=tuple(players),
                visible_area=polygon_from_raw(_pairs(raw.get("visible_area"))),
                ball_x=event.x,
                ball_y=event.y,
                mirrored=False,
            )
        )
    return frames, rejects


def _pairs(flat: list[float] | None) -> list[list[float]] | None:
    """StatsBomb stores visible_area as a flat [x1, y1, x2, y2, ...] list."""

    if not flat or len(flat) < 6:
        return None
    return [[flat[i], flat[i + 1]] for i in range(0, len(flat) - 1, 2)]


def iter_competition_seasons(data_dir: Path) -> Iterator[tuple[int, int, str, str]]:
    for raw in load_json(data_dir / "competitions.json"):
        yield (
            int(raw["competition_id"]),
            int(raw["season_id"]),
            str(raw["competition_name"]),
            str(raw["season_name"]),
        )
