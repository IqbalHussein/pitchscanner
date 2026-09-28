"""One definition of what each filter means, shared by every backend.

The in-memory store applies these predicates directly; the Postgres store
compiles the same semantics to SQL. ``tests/unit/test_filter_parity`` checks the
two agree, because a filter that means different things in dev and production is
a bug nobody notices until the numbers are wrong.
"""

from __future__ import annotations

from pitchscanner.models import Filters
from pitchscanner.store.base import FrameMeta


def matches(meta: FrameMeta, filters: Filters | None) -> bool:
    if filters is None:
        return True
    if filters.competition_ids and meta.competition_id not in filters.competition_ids:
        return False
    if filters.season_ids and meta.season_id not in filters.season_ids:
        return False
    if filters.team_ids:
        role = filters.team_role
        if role == "attacking":
            ok = meta.possession_team_id in filters.team_ids
        elif role == "defending":
            ok = meta.opponent_team_id in filters.team_ids
        else:
            ok = (
                meta.possession_team_id in filters.team_ids
                or meta.opponent_team_id in filters.team_ids
            )
        if not ok:
            return False
    if filters.player_ids and meta.player_id not in filters.player_ids:
        return False
    if filters.game_state != "any" and meta.game_state != filters.game_state:
        return False
    if filters.minute_range is not None:
        lo, hi = filters.minute_range
        if not (lo <= meta.minute <= hi):
            return False
    if filters.stages and meta.stage not in filters.stages:
        return False
    return not (filters.ball_zone_ids and meta.zone_id not in filters.ball_zone_ids)


def to_sql(filters: Filters | None, params: dict[str, object]) -> str:
    """Compile filters to a WHERE fragment against ``frames f``."""

    if filters is None or filters.is_empty():
        return ""
    clauses: list[str] = []
    if filters.competition_ids:
        clauses.append("f.competition_id = ANY(%(competition_ids)s)")
        params["competition_ids"] = list(filters.competition_ids)
    if filters.season_ids:
        clauses.append("f.season_id = ANY(%(season_ids)s)")
        params["season_ids"] = list(filters.season_ids)
    if filters.team_ids:
        params["team_ids"] = list(filters.team_ids)
        if filters.team_role == "attacking":
            clauses.append("f.possession_team_id = ANY(%(team_ids)s)")
        elif filters.team_role == "defending":
            clauses.append("f.opponent_team_id = ANY(%(team_ids)s)")
        else:
            clauses.append(
                "(f.possession_team_id = ANY(%(team_ids)s)"
                " OR f.opponent_team_id = ANY(%(team_ids)s))"
            )
    if filters.player_ids:
        clauses.append("f.player_id = ANY(%(player_ids)s)")
        params["player_ids"] = list(filters.player_ids)
    if filters.game_state != "any":
        clauses.append("f.game_state = %(game_state)s")
        params["game_state"] = filters.game_state
    if filters.minute_range is not None:
        clauses.append("f.minute BETWEEN %(minute_lo)s AND %(minute_hi)s")
        params["minute_lo"], params["minute_hi"] = filters.minute_range
    if filters.stages:
        clauses.append("f.stage = ANY(%(stages)s)")
        params["stages"] = list(filters.stages)
    if filters.ball_zone_ids:
        clauses.append("f.zone_id = ANY(%(zone_ids)s)")
        params["zone_ids"] = list(filters.ball_zone_ids)
    return " AND ".join(clauses)
