"""Selected filters beat the text, and the page is told when they do."""

from __future__ import annotations

import pytest

from pitchscanner.config import Settings
from pitchscanner.filters import FilterService
from pitchscanner.filters.parser import merge, overridden
from pitchscanner.models import Filters
from pitchscanner.store.memory import MemoryStore


@pytest.fixture(scope="module")
def service(fixture_store: MemoryStore, settings: Settings) -> FilterService:
    return FilterService(fixture_store, settings)


def _team_id(service: FilterService, name: str) -> int:
    team = next(e for e in service.catalogue.of_kind("team") if e.name == name)
    return team.entity_id


def test_conflicting_team_is_reported_and_its_chip_dropped(service: FilterService) -> None:
    germany, spain = _team_id(service, "Germany"), _team_id(service, "Spain")
    merged, parsed = service.resolve(Filters(team_ids=[germany]), "Spain")
    assert merged.team_ids == [germany]
    assert parsed is not None
    assert "Team from the text (Spain) is overridden by the selected filter." in parsed.warnings
    assert [c for c in parsed.chips if c.field == "team"] == []
    assert spain not in merged.team_ids


def test_the_overridden_teams_role_does_not_leak_into_the_selected_team(
    service: FilterService,
) -> None:
    germany = _team_id(service, "Germany")
    merged, _ = service.resolve(Filters(team_ids=[germany]), "against Spain")
    assert merged.team_role == "any"


def test_agreeing_filters_are_not_a_conflict(service: FilterService) -> None:
    spain = _team_id(service, "Spain")
    merged, parsed = service.resolve(Filters(team_ids=[spain]), "Spain")
    assert merged.team_ids == [spain]
    assert parsed is not None
    assert parsed.warnings == []
    assert [c.field for c in parsed.chips] == ["team"]


def test_text_fills_gaps_without_a_warning(service: FilterService) -> None:
    germany = _team_id(service, "Germany")
    merged, parsed = service.resolve(Filters(team_ids=[germany]), "when losing")
    assert merged.game_state == "losing"
    assert parsed is not None
    assert parsed.warnings == []


def test_other_fields_warn_too(service: FilterService) -> None:
    merged, parsed = service.resolve(
        Filters(game_state="winning", venue="away", minute_range=(0, 45)),
        "when losing at home in the last 15 minutes",
    )
    assert (merged.game_state, merged.venue, merged.minute_range) == ("winning", "away", (0, 45))
    assert parsed is not None
    assert parsed.chips == []
    assert sorted(parsed.warnings) == [
        "Game state from the text (losing) is overridden by the selected filter.",
        "Minutes from the text (75-98) is overridden by the selected filter.",
        "Venue from the text (home) is overridden by the selected filter.",
    ]


def test_role_conflict_is_reported_without_a_chip(service: FilterService) -> None:
    _, parsed = service.resolve(Filters(team_role="defending"), "Spain")
    assert parsed is not None
    assert any(w.startswith("Team role from the text (attacking)") for w in parsed.warnings)
    assert [c.field for c in parsed.chips] == ["team"]


def test_overridden_ignores_list_order_and_defaults() -> None:
    assert overridden(Filters(team_ids=[1, 2]), Filters(team_ids=[2, 1])) == []
    assert overridden(Filters(), Filters(team_ids=[1])) == []
    assert overridden(Filters(team_ids=[1]), Filters()) == []
    assert overridden(Filters(team_ids=[1]), Filters(team_ids=[2])) == ["team_ids"]


def test_merge_keeps_explicit_values() -> None:
    merged = merge(Filters(venue="home"), Filters(venue="away", game_state="losing"))
    assert (merged.venue, merged.game_state) == ("home", "losing")
