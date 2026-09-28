"""Wilson intervals against known values, weighting, and the bootstrap's seed."""

from __future__ import annotations

import math

import pytest

from pitchscanner.models import OutcomeRecord
from pitchscanner.stats import (
    Neighbour,
    aggregate,
    bootstrap_mean,
    effective_n,
    ratio_to_baseline,
    similarity_weight,
    weighted_proportion,
    wilson_interval,
)


def _wilson_reference(successes: float, n: float, z: float = 1.959963984540054):
    """The textbook formula, written out independently of the implementation."""

    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return centre - half, centre + half


@pytest.mark.parametrize(("successes", "n"), [(8, 25), (1, 10), (50, 100), (99, 100)])
def test_wilson_matches_reference(successes: int, n: int) -> None:
    got = wilson_interval(successes, n)
    want = _wilson_reference(successes, n)
    assert got[0] == pytest.approx(want[0], abs=1e-9)
    assert got[1] == pytest.approx(want[1], abs=1e-9)


def test_wilson_handles_zero_n() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_stays_inside_zero_one() -> None:
    lo, hi = wilson_interval(0, 3)
    assert lo >= 0.0
    assert hi <= 1.0


def test_similarity_weight_decreases_with_distance() -> None:
    assert similarity_weight(0.0) == 1.0
    assert similarity_weight(1.0) > similarity_weight(3.0) > 0.0


def test_effective_n_penalises_uneven_weights() -> None:
    assert effective_n([1, 1, 1, 1]) == pytest.approx(4.0)
    assert effective_n([1, 0.01, 0.01, 0.01]) < 2.0
    assert effective_n([]) == 0.0


def test_weighted_proportion_respects_weights() -> None:
    # The `True` neighbour is far away, so the rate must stay below one half.
    result = weighted_proportion([True, False], [0.1, 1.0])
    assert result.value < 0.5
    assert result.ci95[0] <= result.value <= result.ci95[1]


def test_bootstrap_is_reproducible_from_the_seed() -> None:
    values = [0.1, 0.0, 0.4, 0.05, 0.2]
    weights = [1.0, 0.8, 0.6, 0.9, 0.7]
    a = bootstrap_mean(values, weights, resamples=200, seed=1)
    b = bootstrap_mean(values, weights, resamples=200, seed=1)
    c = bootstrap_mean(values, weights, resamples=200, seed=2)
    assert a == b
    assert a.ci95 != c.ci95


def test_bootstrap_single_value_is_degenerate() -> None:
    result = bootstrap_mean([0.3], [1.0])
    assert result.value == pytest.approx(0.3)
    assert result.ci95 == (pytest.approx(0.3), pytest.approx(0.3))


def test_ratio_to_baseline_guards_zero() -> None:
    assert ratio_to_baseline(0.3, 0.15) == 2.0
    assert ratio_to_baseline(0.3, 0.0) is None
    assert ratio_to_baseline(0.3, None) is None


def _neighbours(n: int) -> list[Neighbour]:
    return [
        Neighbour(
            outcome=OutcomeRecord(
                event_id=f"e{i}",
                shot=i % 3 == 0,
                xg_next=0.05 * (i % 4),
                goal=i % 9 == 0,
                turnover=i % 3 != 0,
                box_entry=i % 2 == 0,
                seconds_to_end=3.0 + i % 5,
            ),
            distance=0.05 * i,
        )
        for i in range(n)
    ]


def test_aggregate_flags_low_confidence_below_thirty() -> None:
    small = aggregate(_neighbours(12), resamples=50)
    big = aggregate(_neighbours(40), resamples=50)
    assert small.low_confidence is True
    assert big.low_confidence is False
    assert any("12 distinct possessions" in note for note in small.notes)


def test_aggregate_reports_baseline_ratios() -> None:
    result = aggregate(
        _neighbours(30),
        baseline={"shot_rate": 0.1, "xg_mean": 0.02, "turnover_rate": 0.5,
                  "box_entry_rate": 0.25, "goal_rate": 0.01},
        resamples=50,
    )
    assert result.shot_rate.vs_zone_baseline is not None
    assert result.xg_mean.vs_zone_baseline is not None


def test_aggregate_handles_no_neighbours() -> None:
    result = aggregate([], resamples=10)
    assert result.n == 0
    assert result.shot_rate.value == 0.0
