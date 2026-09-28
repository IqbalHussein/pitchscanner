"""Aggregation with honest uncertainty.

Neighbours are weighted by similarity, rates get Wilson intervals and the xG
mean gets a seeded bootstrap, so a rerun of the benchmark reproduces the numbers
in the committed report exactly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pitchscanner.models import Interval, OutcomeRecord

Z95 = 1.959963984540054
DEFAULT_LOW_CONFIDENCE_N = 30


def similarity_weight(distance: float) -> float:
    """w = 1 / (1 + d). Closer neighbours count more, and nothing counts zero."""

    return 1.0 / (1.0 + max(0.0, distance))


def wilson_interval(successes: float, n: float, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval, valid at the small n this project actually has.

    The normal approximation would happily report a negative shot rate from a
    handful of plays, which is exactly the regime the UI operates in.
    """

    if n <= 0:
        return (0.0, 1.0)
    p = min(max(successes / n, 0.0), 1.0)
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def effective_n(weights: Sequence[float] | np.ndarray) -> float:
    """Kish effective sample size, so weighting cannot fake precision."""

    w = np.asarray(weights, dtype=np.float64)
    if w.size == 0 or w.sum() <= 0:
        return 0.0
    return float(w.sum() ** 2 / np.square(w).sum())


def weighted_proportion(flags: Sequence[bool], weights: Sequence[float]) -> Interval:
    w = np.asarray(weights, dtype=np.float64)
    f = np.asarray(flags, dtype=np.float64)
    if w.size == 0 or w.sum() <= 0:
        return Interval(value=0.0, ci95=(0.0, 1.0))
    value = float((f * w).sum() / w.sum())
    n_eff = effective_n(w)
    lo, hi = wilson_interval(value * n_eff, n_eff)
    return Interval(value=value, ci95=(lo, hi))


def bootstrap_mean(
    values: Sequence[float],
    weights: Sequence[float],
    *,
    resamples: int = 1000,
    seed: int = 20260925,
) -> Interval:
    """Weighted mean with a percentile bootstrap interval.

    Seeded from a fixed value: the same neighbours must always produce the same
    interval, or the benchmark report stops being reproducible.
    """

    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if v.size == 0 or w.sum() <= 0:
        return Interval(value=0.0, ci95=(0.0, 0.0))
    point = float((v * w).sum() / w.sum())
    if v.size == 1:
        return Interval(value=point, ci95=(point, point))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(resamples, v.size))
    vs = v[idx]
    ws = w[idx]
    sums = ws.sum(axis=1)
    sums[sums == 0] = 1.0
    means = (vs * ws).sum(axis=1) / sums
    lo, hi = np.percentile(means, [2.5, 97.5])
    return Interval(value=point, ci95=(float(lo), float(hi)))


@dataclass(frozen=True, slots=True)
class Neighbour:
    outcome: OutcomeRecord
    distance: float


def ratio_to_baseline(value: float, baseline: float | None) -> float | None:
    """A rate on its own says very little; report it against the zone average."""

    if baseline is None or baseline <= 1e-9:
        return None
    return round(value / baseline, 3)


@dataclass(frozen=True, slots=True)
class Aggregate:
    """The aggregated block, typed, so callers do not unpack a bag of objects."""

    n: int
    shot_rate: Interval
    xg_mean: Interval
    turnover_rate: Interval
    box_entry_rate: Interval
    goal_rate: Interval
    unweighted: dict[str, float]
    low_confidence: bool
    notes: list[str]


def aggregate(
    neighbours: Sequence[Neighbour],
    *,
    baseline: dict[str, float] | None = None,
    resamples: int = 1000,
    seed: int = 20260925,
    low_confidence_n: int = DEFAULT_LOW_CONFIDENCE_N,
) -> Aggregate:
    """Turn a neighbour list into the stats block the API returns."""

    weights = [similarity_weight(n.distance) for n in neighbours]
    outcomes = [n.outcome for n in neighbours]
    n = len(outcomes)

    shot = weighted_proportion([o.shot for o in outcomes], weights)
    turnover = weighted_proportion([o.turnover for o in outcomes], weights)
    box = weighted_proportion([o.box_entry for o in outcomes], weights)
    goal = weighted_proportion([o.goal for o in outcomes], weights)
    xg = bootstrap_mean(
        [o.xg_next for o in outcomes], weights, resamples=resamples, seed=seed
    )

    if baseline:
        shot.vs_zone_baseline = ratio_to_baseline(shot.value, baseline.get("shot_rate"))
        turnover.vs_zone_baseline = ratio_to_baseline(
            turnover.value, baseline.get("turnover_rate")
        )
        box.vs_zone_baseline = ratio_to_baseline(box.value, baseline.get("box_entry_rate"))
        goal.vs_zone_baseline = ratio_to_baseline(goal.value, baseline.get("goal_rate"))
        xg.vs_zone_baseline = ratio_to_baseline(xg.value, baseline.get("xg_mean"))

    unweighted = {
        "shot_rate": _mean([o.shot for o in outcomes]),
        "xg_mean": _mean([o.xg_next for o in outcomes]),
        "turnover_rate": _mean([o.turnover for o in outcomes]),
        "box_entry_rate": _mean([o.box_entry for o in outcomes]),
        "goal_rate": _mean([o.goal for o in outcomes]),
        "seconds_to_end": _mean([o.seconds_to_end for o in outcomes]),
        "effective_n": round(effective_n(weights), 2),
    }

    notes: list[str] = []
    if n < low_confidence_n:
        notes.append(
            f"Only {n} distinct possessions matched. Treat these rates as indicative."
        )
    notes.append("Goal rate is noisy at this sample size; xG is the number to read.")

    return Aggregate(
        n=n,
        shot_rate=shot,
        xg_mean=xg,
        turnover_rate=turnover,
        box_entry_rate=box,
        goal_rate=goal,
        unweighted=unweighted,
        low_confidence=n < low_confidence_n,
        notes=notes,
    )


def _mean(values: Sequence[bool] | Sequence[float]) -> float:
    if not values:
        return 0.0
    return round(float(np.mean(np.asarray(values, dtype=np.float64))), 6)
