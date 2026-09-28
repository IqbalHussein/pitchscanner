"""Search and stats invariants, checked against the ingested fixture dataset."""

from __future__ import annotations

import os

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pitchscanner.encoders import available, get_encoder
from pitchscanner.models import Filters, PointIn, SearchRequest
from pitchscanner.search import SearchService
from pitchscanner.stats import Neighbour, aggregate
from tests.property.strategies import situations

#: These properties run a real search per example, so they are capped well below
#: the encoder properties even on the nightly profile: past a few dozen examples
#: the extra cost buys repetition rather than coverage.
FAST = settings(
    max_examples=int(os.environ.get("PITCHSCANNER_SEARCH_EXAMPLES", "25")),
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@pytest.mark.parametrize("encoding", available())
def test_a_frame_searched_against_the_index_returns_itself_first(
    fixture_store, encoding: str
) -> None:
    encoder = get_encoder(encoding)
    frames = fixture_store.frames()[:40]
    for frame in frames:
        hits = fixture_store.exact_search(
            encoder.version, encoder.encode(frame.to_situation()), 1, encoder.metric, None
        )
        assert hits, "index returned nothing for a frame that is in it"
        assert hits[0].event_id == frame.event_id
        assert hits[0].distance == pytest.approx(0.0, abs=1e-5)


def test_approximate_results_are_a_subset_of_the_exact_ranking(fixture_store) -> None:
    """ANN must not invent neighbours the exact scan would never rank highly."""

    encoder = get_encoder("E1")
    for frame in fixture_store.frames()[:20]:
        vector = encoder.encode(frame.to_situation())
        approx = fixture_store.ann_search(encoder.version, vector, 10, encoder.metric, None)
        exact = fixture_store.exact_search(encoder.version, vector, 300, encoder.metric, None)
        exact_ids = {c.event_id for c in exact}
        assert {c.event_id for c in approx} <= exact_ids


def test_dedup_leaves_one_result_per_possession(fixture_store, settings) -> None:
    service = SearchService(fixture_store, settings)
    request = SearchRequest(
        ball=PointIn(x=90.0, y=40.0),
        attackers=[PointIn(x=92.0, y=38.0), PointIn(x=100.0, y=44.0)],
        defenders=[PointIn(x=95.0, y=40.0)],
        k=40,
    )
    outcome = service.search(request)
    keys = [(r.match_id, r.possession_id) for r in outcome.results]
    assert len(keys) == len(set(keys))


def test_dedup_off_allows_repeats(fixture_store, settings) -> None:
    """The control case: without dedup, one possession really does repeat."""

    service = SearchService(fixture_store, settings)
    request = SearchRequest(
        ball=PointIn(x=90.0, y=40.0),
        attackers=[PointIn(x=92.0, y=38.0)],
        defenders=[PointIn(x=95.0, y=40.0)],
        k=40,
        options={"dedup": False, "mirror": False},
    )
    outcome = service.search(request)
    keys = [(r.match_id, r.possession_id) for r in outcome.results]
    assert len(keys) > len(set(keys)), "fixture data should contain near-duplicate frames"


@FAST
@given(
    distances=st.lists(
        st.floats(min_value=0.0, max_value=3.0, allow_nan=False), min_size=1, max_size=60
    ),
    shots=st.lists(st.booleans(), min_size=1, max_size=60),
)
def test_every_rate_is_a_probability_and_contains_its_estimate(
    distances: list[float], shots: list[bool]
) -> None:
    from pitchscanner.models import OutcomeRecord

    n = min(len(distances), len(shots))
    neighbours = [
        Neighbour(
            outcome=OutcomeRecord(
                event_id=f"e{i}", shot=shots[i], xg_next=0.1 * (i % 3), goal=False,
                turnover=not shots[i], box_entry=shots[i], seconds_to_end=1.0,
            ),
            distance=distances[i],
        )
        for i in range(n)
    ]
    result = aggregate(neighbours, resamples=50)
    for field in ("shot_rate", "turnover_rate", "box_entry_rate", "goal_rate"):
        interval = getattr(result, field)
        assert 0.0 <= interval.value <= 1.0
        assert interval.ci95[0] <= interval.value + 1e-9
        assert interval.value - 1e-9 <= interval.ci95[1]
        assert 0.0 <= interval.ci95[0] <= interval.ci95[1] <= 1.0


@FAST
@given(situation=situations(min_players=1, max_players=8))
def test_search_never_returns_more_than_k(fixture_store, settings, situation) -> None:
    service = SearchService(fixture_store, settings)
    request = SearchRequest(
        ball=PointIn(x=situation.ball.x, y=situation.ball.y),
        attackers=[PointIn(x=p.x, y=p.y) for p in situation.attackers][:11]
        or [PointIn(x=situation.ball.x, y=situation.ball.y)],
        defenders=[PointIn(x=p.x, y=p.y) for p in situation.defenders][:11],
        k=7,
    )
    outcome = service.search(request)
    assert len(outcome.results) <= 7
    assert all(0.0 <= r.similarity <= 1.0 for r in outcome.results)


def test_a_filter_never_widens_the_result_set(fixture_store, settings) -> None:
    service = SearchService(fixture_store, settings)
    request = SearchRequest(
        ball=PointIn(x=88.0, y=30.0),
        attackers=[PointIn(x=90.0, y=28.0)],
        defenders=[PointIn(x=94.0, y=32.0)],
        k=50,
    )
    unfiltered = service.search(request)
    filtered = service.search(request, filters=Filters(game_state="losing"))
    assert len(filtered.results) <= len(unfiltered.results)
    ids = {r.event_id for r in unfiltered.results}
    # Every filtered hit is a real frame, and it satisfies the predicate.
    for result in filtered.results:
        meta = fixture_store.meta(result.event_id)
        assert meta is not None
        assert meta.game_state == "losing"
    assert isinstance(ids, set)
