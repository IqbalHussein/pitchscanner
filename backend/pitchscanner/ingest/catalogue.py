"""Build the entity catalogue from the data, never from a hard-coded list.

The dropdowns and the text parser both read this, so coverage updates the moment
StatsBomb adds data.
"""

from __future__ import annotations

from collections import Counter

from pitchscanner.models import EventRecord, FrameRecord, MatchRecord
from pitchscanner.store.base import CatalogueEntry


def build_catalogue(
    matches: list[MatchRecord],
    events: list[EventRecord],
    frames: list[FrameRecord],
    metas: dict[str, object],
) -> list[CatalogueEntry]:
    frame_match_ids = Counter(f.match_id for f in frames)
    match_by_id = {m.match_id: m for m in matches}

    comp_frames: Counter[int] = Counter()
    season_frames: Counter[int] = Counter()
    team_frames: Counter[int] = Counter()
    season_parent: dict[int, int] = {}
    comp_names: dict[int, str] = {}
    season_names: dict[int, str] = {}
    team_names: dict[int, str] = {}

    for match_id, n in frame_match_ids.items():
        m = match_by_id.get(match_id)
        if m is None:
            continue
        comp_frames[m.competition_id] += n
        season_frames[m.season_id] += n
        season_parent[m.season_id] = m.competition_id
        comp_names[m.competition_id] = m.competition_name
        season_names[m.season_id] = m.season_name
        team_frames[m.home_team_id] += n
        team_frames[m.away_team_id] += n
        team_names[m.home_team_id] = m.home_team
        team_names[m.away_team_id] = m.away_team

    frame_ids = {f.event_id for f in frames}
    player_frames: Counter[int] = Counter()
    player_names: dict[int, str] = {}
    for e in events:
        if e.player_id is None or e.player_name is None:
            continue
        player_names[e.player_id] = e.player_name
        if e.event_id in frame_ids:
            player_frames[e.player_id] += 1

    entries: list[CatalogueEntry] = []
    # Competitions and seasons with zero frames are still listed: the parser needs
    # them to explain "no positional data for that competition" instead of
    # pretending the name is unknown.
    for m in matches:
        comp_names.setdefault(m.competition_id, m.competition_name)
        season_names.setdefault(m.season_id, m.season_name)
        season_parent.setdefault(m.season_id, m.competition_id)
        team_names.setdefault(m.home_team_id, m.home_team)
        team_names.setdefault(m.away_team_id, m.away_team)

    entries.extend(
        CatalogueEntry("competition", cid, name, frames=comp_frames.get(cid, 0))
        for cid, name in comp_names.items()
    )
    entries.extend(
        CatalogueEntry(
            "season", sid, name, parent_id=season_parent.get(sid),
            frames=season_frames.get(sid, 0),
        )
        for sid, name in season_names.items()
    )
    entries.extend(
        CatalogueEntry("team", tid, name, frames=team_frames.get(tid, 0))
        for tid, name in team_names.items()
    )
    entries.extend(
        CatalogueEntry("player", pid, name, frames=player_frames.get(pid, 0))
        for pid, name in player_names.items()
    )
    return entries
