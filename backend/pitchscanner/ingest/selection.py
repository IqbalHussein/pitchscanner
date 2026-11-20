"""Which competitions (and seasons) an ingest run should touch.

The spec is a comma-separated list of ``competition_id`` or
``competition_id:season_id`` entries, e.g. ``55:282,9``. Both ``fetch`` (to
decide which files to check out) and the pipeline (to decide which matches to
ingest) read it, so it lives apart from either.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CompetitionSelector:
    competition_id: int
    #: None selects every season of the competition.
    season_id: int | None = None

    def matches(self, competition_id: int, season_id: int) -> bool:
        return self.competition_id == competition_id and self.season_id in (None, season_id)

    def __str__(self) -> str:
        return str(self.competition_id) if self.season_id is None else (
            f"{self.competition_id}:{self.season_id}"
        )


def parse_competitions(spec: str) -> tuple[CompetitionSelector, ...]:
    """Parse the ``--competitions`` value. Raises ValueError with the bad entry."""

    out: list[CompetitionSelector] = []
    for raw in spec.split(","):
        entry = raw.strip()
        if not entry:
            continue
        parts = entry.split(":")
        try:
            if len(parts) > 2:
                raise ValueError
            ids = [int(p) for p in parts]
        except ValueError:
            raise ValueError(
                f"bad competition entry {entry!r}: expected competition_id or "
                "competition_id:season_id (integers)"
            ) from None
        selector = CompetitionSelector(ids[0], ids[1] if len(ids) == 2 else None)
        if selector not in out:
            out.append(selector)
    if not out:
        raise ValueError("no competitions given")
    return tuple(out)


def is_selected(
    selectors: Iterable[CompetitionSelector] | None, competition_id: int, season_id: int
) -> bool:
    """No selectors means no filter: everything is selected."""

    if selectors is None:
        return True
    return any(s.matches(competition_id, season_id) for s in selectors)
