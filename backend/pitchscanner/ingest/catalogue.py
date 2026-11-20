"""Build the entity catalogue from the data, never from a hard-coded list.

The dropdowns and the text parser both read this, so coverage updates the moment
StatsBomb adds data.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from pitchscanner.models import EventRecord, FrameRecord, MatchRecord
from pitchscanner.store.base import CatalogueEntry


class CatalogueBuilder:
    """Accumulates what the catalogue needs, one match at a time.

    Only counts and names are kept, never the frames themselves, so a full
    ingest does not hold every frame in memory just to count them at the end.
    """

    def __init__(self) -> None:
        self._matches: list[MatchRecord] = []
        self._frames_per_match: Counter[int] = Counter()
        self._player_frames: Counter[int] = Counter()
        self._player_names: dict[int, str] = {}

    def add_match(
        self,
        match: MatchRecord,
        events: Iterable[EventRecord],
        frames: Iterable[FrameRecord],
    ) -> None:
        self._matches.append(match)
        frame_ids = set()
        for f in frames:
            self._frames_per_match[f.match_id] += 1
            frame_ids.add(f.event_id)
        for e in events:
            if e.player_id is None or e.player_name is None:
                continue
            self._player_names[e.player_id] = e.player_name
            if e.event_id in frame_ids:
                self._player_frames[e.player_id] += 1

    def entries(self) -> list[CatalogueEntry]:
        match_by_id = {m.match_id: m for m in self._matches}

        comp_frames: Counter[int] = Counter()
        season_frames: Counter[int] = Counter()
        team_frames: Counter[int] = Counter()
        season_parent: dict[int, int] = {}
        comp_names: dict[int, str] = {}
        season_names: dict[int, str] = {}
        team_names: dict[int, str] = {}

        for match_id, n in self._frames_per_match.items():
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

        entries: list[CatalogueEntry] = []
        # Competitions and seasons with zero frames are still listed: the parser needs
        # them to explain "no positional data for that competition" instead of
        # pretending the name is unknown.
        for m in self._matches:
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
            CatalogueEntry("player", pid, name, frames=self._player_frames.get(pid, 0))
            for pid, name in self._player_names.items()
        )
        return entries


def build_catalogue(
    matches: list[MatchRecord],
    events: list[EventRecord],
    frames: list[FrameRecord],
    metas: dict[str, object],
) -> list[CatalogueEntry]:
    """The whole catalogue from flat lists (``metas`` is unused, kept for callers)."""

    events_by_match: dict[int, list[EventRecord]] = {}
    for e in events:
        events_by_match.setdefault(e.match_id, []).append(e)
    frames_by_match: dict[int, list[FrameRecord]] = {}
    for f in frames:
        frames_by_match.setdefault(f.match_id, []).append(f)
    builder = CatalogueBuilder()
    for m in matches:
        builder.add_match(
            m, events_by_match.get(m.match_id, []), frames_by_match.get(m.match_id, [])
        )
    return builder.entries()
