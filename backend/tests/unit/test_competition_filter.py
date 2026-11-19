"""`--competitions`: parsing, pipeline filtering, and narrowing the git fetch."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from pitchscanner.cli import build_parser
from pitchscanner.config import Settings
from pitchscanner.ingest import fetch_open_data, ingest
from pitchscanner.ingest.selection import CompetitionSelector, is_selected, parse_competitions
from pitchscanner.store.memory import MemoryStore
from tests.conftest import FIXTURE_DIR

SEL_55 = parse_competitions("55:282")
SEL_9 = parse_competitions("9")


def test_parse_mixed_entries_and_whitespace() -> None:
    got = parse_competitions(" 55:282 , 9,55:282,")
    assert got == (CompetitionSelector(55, 282), CompetitionSelector(9, None))
    assert [str(s) for s in got] == ["55:282", "9"]


@pytest.mark.parametrize("bad", ["", " , ", "abc", "55:", ":282", "1:2:3", "55:x"])
def test_parse_rejects_bad_entries(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_competitions(bad)


def test_is_selected() -> None:
    assert is_selected(None, 1, 2)
    assert is_selected(SEL_55, 55, 282)
    assert not is_selected(SEL_55, 55, 1)
    assert is_selected(SEL_9, 9, 12345)  # bare competition id = every season
    assert not is_selected(SEL_9, 55, 282)


def test_cli_parses_and_rejects(capsys: pytest.CaptureFixture[str]) -> None:
    args = build_parser().parse_args(["ingest", "--fixtures", "--competitions", "55:282,9"])
    assert args.competitions == (CompetitionSelector(55, 282), CompetitionSelector(9, None))
    assert build_parser().parse_args(["ingest", "--fixtures"]).competitions is None
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ingest", "--competitions", "nope"])
    assert "bad competition entry" in capsys.readouterr().err


def _ingest(selectors, settings: Settings, **kw) -> MemoryStore:
    store = MemoryStore()
    ingest(FIXTURE_DIR, store, settings=settings, competitions=selectors, **kw)
    return store


def test_pipeline_ingests_only_selected_competition(settings: Settings) -> None:
    everything = _ingest(None, settings, encodings=["E1"])
    only_55 = _ingest(SEL_55, settings, encodings=["E1"])
    only_9 = _ingest(SEL_9, settings, encodings=["E1"])
    assert {m.competition_id for m in only_55._matches.values()} == {55}
    assert {m.competition_id for m in only_9._matches.values()} == {9}
    assert len(only_55._matches) + len(only_9._matches) == len(everything._matches)
    assert only_55.counts().frames + only_9.counts().frames == everything.counts().frames
    assert set(only_55._matches).isdisjoint(only_9._matches)


def test_pipeline_season_must_match(settings: Settings) -> None:
    with pytest.raises(ValueError, match="no matches files"):
        _ingest(parse_competitions("55:1"), settings)


def test_limit_counts_within_the_selection(settings: Settings) -> None:
    store = _ingest(SEL_55, settings, encodings=["E1"], limit_matches=1)
    assert len(store._matches) == 1
    assert {m.competition_id for m in store._matches.values()} == {55}


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_fetch_checks_out_only_selected_matches(tmp_path: Path) -> None:
    """Against a local stand-in for statsbomb/open-data: only the chosen
    competition's events and three-sixty files are checked out."""

    upstream = tmp_path / "upstream"
    shutil.copytree(FIXTURE_DIR, upstream / "data")

    def git(*args: str, cwd: Path = upstream) -> str:
        done = subprocess.run(
            ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
            cwd=cwd, check=True, capture_output=True, text=True,
        )
        return done.stdout.strip()

    git("init", "--quiet")
    git("add", "-A")
    git("commit", "--quiet", "-m", "fixtures")
    git("config", "uploadpack.allowFilter", "true")
    git("config", "uploadpack.allowAnySHA1InWant", "true")
    commit = git("rev-parse", "HEAD")

    cfg = Settings(
        store="memory",
        open_data_repo=upstream.as_uri(),
        open_data_commit=commit,
        data_cache=tmp_path / "cache",
    )

    def checked_out(data: Path) -> set[str]:
        return {p.name for p in (data / "events").glob("*.json")}

    all_ids = {p.name for p in (FIXTURE_DIR / "events").glob("*.json")}
    data = fetch_open_data(cfg, competitions=SEL_9)
    nine = checked_out(data)
    assert nine and nine < all_ids
    assert checked_out(data) == {p.name for p in (data / "three-sixty").glob("*.json")}

    # Switching the selection swaps the files; dropping it restores everything.
    other = checked_out(fetch_open_data(cfg, competitions=SEL_55))
    assert other and other.isdisjoint(nine) and other | nine == all_ids
    assert checked_out(fetch_open_data(cfg)) == all_ids

    # And the narrowed checkout really ingests.
    store = MemoryStore()
    fetch_open_data(cfg, competitions=SEL_9)
    report = ingest(data, store, settings=cfg, competitions=SEL_9, encodings=["E1"])
    assert report.matches == len(nine)
