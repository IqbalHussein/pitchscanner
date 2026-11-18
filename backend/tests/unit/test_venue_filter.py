"""The venue filter: was the team in possession the home side?"""

from __future__ import annotations

import pytest

from pitchscanner.models import Filters
from pitchscanner.store.filtering import matches, to_sql
from pitchscanner.store.memory import MemoryStore

ENCODING = "E1-v1"


def test_home_and_away_partition_every_frame(fixture_store: MemoryStore) -> None:
    total = fixture_store.count_matching(ENCODING, None)
    home = fixture_store.count_matching(ENCODING, Filters(venue="home"))
    away = fixture_store.count_matching(ENCODING, Filters(venue="away"))
    assert 0 < home < total
    assert home + away == total


def test_venue_means_the_possession_team_is_the_matchs_home_team(
    fixture_store: MemoryStore,
) -> None:
    checked = 0
    for event_id, meta in fixture_store._metas.items():
        match = fixture_store.match(meta.match_id)
        assert match is not None
        is_home = meta.possession_team_id == match.home_team_id
        assert meta.possession_is_home is is_home, event_id
        assert matches(meta, Filters(venue="home")) is is_home
        assert matches(meta, Filters(venue="away")) is (not is_home)
        checked += 1
    assert checked > 0


def test_any_venue_is_not_a_filter() -> None:
    assert Filters(venue="any").is_empty()
    assert not Filters(venue="home").is_empty()
    assert to_sql(Filters(), {}) == ""


@pytest.mark.parametrize(("venue", "flag"), [("home", True), ("away", False)])
def test_venue_compiles_to_the_denormalised_column(venue: str, flag: bool) -> None:
    params: dict[str, object] = {}
    sql = to_sql(Filters(venue=venue), params)  # type: ignore[arg-type]
    assert sql == "f.possession_is_home = %(possession_is_home)s"
    assert params == {"possession_is_home": flag}


def test_api_venue_from_text_and_from_filters(api_client) -> None:
    sketch = {
        "ball": {"x": 84, "y": 22},
        "attackers": [{"x": 86, "y": 20}, {"x": 95, "y": 35}],
        "defenders": [{"x": 92, "y": 25}, {"x": 100, "y": 30}],
        "k": 10,
    }
    by_text = api_client.post("/v1/search", json={**sketch, "filter_text": "at home"}).json()
    by_filter = api_client.post(
        "/v1/search", json={**sketch, "filters": {"venue": "home"}}
    ).json()
    assert by_text["filters_applied"]["venue"] == "home"
    assert [(c["field"], c["value"], c["span"]) for c in by_text["parsed_filters"]["chips"]] == [
        ("venue", "home", "at home")
    ]
    assert by_text["results"]
    assert [r["event_id"] for r in by_text["results"]] == [
        r["event_id"] for r in by_filter["results"]
    ]
    away = api_client.post("/v1/search", json={**sketch, "filters": {"venue": "away"}}).json()
    assert not {r["event_id"] for r in away["results"]} & {
        r["event_id"] for r in by_filter["results"]
    }


def test_api_warns_when_a_selected_filter_overrides_the_text(api_client) -> None:
    sketch = {"ball": {"x": 84, "y": 22}, "attackers": [{"x": 86, "y": 20}], "k": 5}
    teams = api_client.get("/v1/filters").json()["teams"]
    germany = next(t["team_id"] for t in teams if t["team_name"] == "Germany")
    body = api_client.post(
        "/v1/search",
        json={**sketch, "filters": {"team_ids": [germany]}, "filter_text": "Spain"},
    ).json()
    assert body["filters_applied"]["team_ids"] == [germany]
    parsed = body["parsed_filters"]
    assert "Team from the text (Spain) is overridden by the selected filter." in parsed["warnings"]
    assert [c for c in parsed["chips"] if c["field"] == "team"] == []
