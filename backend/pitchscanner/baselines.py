"""Zone baselines: what happens on average with the ball in this part of the pitch."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from pitchscanner.geometry import N_ZONES, zone_id
from pitchscanner.models import FrameRecord, OutcomeRecord, ZoneBaseline

BASELINE_FIELDS = ("shot_rate", "xg_mean", "goal_rate", "turnover_rate", "box_entry_rate")


class ZoneAccumulator:
    """Per-zone tallies, so baselines need no list of every frame record.

    The yes/no outcomes are exact integer counts. xG is kept as a list of floats per
    zone and summed with ``sum()`` at the end, because Python's compensated ``sum``
    and a running ``+=`` differ in the last digit, and the baselines should be
    bit-for-bit what they were when computed from the full list of frames.
    """

    def __init__(self) -> None:
        self.n = [0] * N_ZONES
        self._counts = {f: [0] * N_ZONES for f in BASELINE_FIELDS if f != "xg_mean"}
        self._xg: list[list[float]] = [[] for _ in range(N_ZONES)]

    def add(self, frames: Iterable[FrameRecord], outcomes: Mapping[str, OutcomeRecord]) -> None:
        for f in frames:
            o = outcomes.get(f.event_id)
            if o is None:
                continue
            z = zone_id(f.ball_x, f.ball_y)
            self.n[z] += 1
            self._counts["shot_rate"][z] += o.shot
            self._xg[z].append(o.xg_next)
            self._counts["goal_rate"][z] += o.goal
            self._counts["turnover_rate"][z] += o.turnover
            self._counts["box_entry_rate"][z] += o.box_entry

    def baselines(self, encoding_version: str) -> list[ZoneBaseline]:
        out: list[ZoneBaseline] = []
        for zone in range(N_ZONES):
            n = self.n[zone]
            rates = {f: (c[zone] / n if n else 0.0) for f, c in self._counts.items()}
            rates["xg_mean"] = sum(self._xg[zone]) / n if n else 0.0
            out.append(
                ZoneBaseline(zone_id=zone, encoding_version=encoding_version, n=n, **rates)
            )
        return out


def compute_zone_baselines(
    frames: Sequence[FrameRecord],
    outcomes: Mapping[str, OutcomeRecord],
    *,
    encoding_version: str,
) -> list[ZoneBaseline]:
    """Unweighted averages per zone over every labelled frame."""

    acc = ZoneAccumulator()
    acc.add(frames, outcomes)
    return acc.baselines(encoding_version)


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
