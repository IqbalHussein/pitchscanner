"""Zone baselines: what happens on average with the ball in this part of the pitch."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from pitchscanner.geometry import N_ZONES, zone_id
from pitchscanner.models import FrameRecord, OutcomeRecord, ZoneBaseline

BASELINE_FIELDS = ("shot_rate", "xg_mean", "goal_rate", "turnover_rate", "box_entry_rate")


def compute_zone_baselines(
    frames: Sequence[FrameRecord],
    outcomes: Mapping[str, OutcomeRecord],
    *,
    encoding_version: str,
) -> list[ZoneBaseline]:
    """Unweighted averages per zone over every labelled frame."""

    buckets: dict[int, list[OutcomeRecord]] = defaultdict(list)
    for f in frames:
        o = outcomes.get(f.event_id)
        if o is not None:
            buckets[zone_id(f.ball_x, f.ball_y)].append(o)

    out: list[ZoneBaseline] = []
    for zone in range(N_ZONES):
        rows = buckets.get(zone, [])
        n = len(rows)
        if n == 0:
            out.append(
                ZoneBaseline(
                    zone_id=zone,
                    encoding_version=encoding_version,
                    n=0,
                    shot_rate=0.0,
                    xg_mean=0.0,
                    goal_rate=0.0,
                    turnover_rate=0.0,
                    box_entry_rate=0.0,
                )
            )
            continue
        out.append(
            ZoneBaseline(
                zone_id=zone,
                encoding_version=encoding_version,
                n=n,
                shot_rate=sum(r.shot for r in rows) / n,
                xg_mean=sum(r.xg_next for r in rows) / n,
                goal_rate=sum(r.goal for r in rows) / n,
                turnover_rate=sum(r.turnover for r in rows) / n,
                box_entry_rate=sum(r.box_entry for r in rows) / n,
            )
        )
    return out


def as_mapping(baselines: Iterable[ZoneBaseline]) -> dict[int, dict[str, float]]:
    return {
        b.zone_id: {
            "shot_rate": b.shot_rate,
            "xg_mean": b.xg_mean,
            "goal_rate": b.goal_rate,
            "turnover_rate": b.turnover_rate,
            "box_entry_rate": b.box_entry_rate,
            "n": float(b.n),
        }
        for b in baselines
    }
