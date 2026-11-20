"""A corrupt events/three-sixty file skips its match; it must not abort the ingest."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from pitchscanner.config import Settings
from pitchscanner.ingest import ingest
from pitchscanner.ingest.pipeline import IngestReport
from pitchscanner.store.memory import MemoryStore
from tests.conftest import FIXTURE_DIR

BAD = 3900002


def _run(data: Path, settings: Settings, **kw) -> tuple[IngestReport, MemoryStore]:
    store = MemoryStore()
    report = ingest(data, store, settings=settings, encodings=["E1"], **kw)
    return report, store


def _nul(path: Path) -> None:
    raw = path.read_bytes()
    mid = len(raw) // 2
    path.write_bytes(raw[:mid] + b"\x00" * 16384 + raw[mid:])


def _truncate(path: Path) -> None:
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])


def _undecodable(path: Path) -> None:
    path.write_bytes(b"[\xff\xfe\xfa]")


@pytest.fixture(scope="module")
def clean(settings: Settings) -> tuple[IngestReport, MemoryStore]:
    return _run(FIXTURE_DIR, settings)


# A NUL run in a three-sixty file is salvaged rather than skipped (test_salvage.py).
_SKIPPED = [
    (folder, corrupt)
    for folder in ("three-sixty", "events")
    for corrupt in (_nul, _truncate, _undecodable)
    if not (folder == "three-sixty" and corrupt is _nul)
]


@pytest.mark.parametrize(("folder", "corrupt"), _SKIPPED)
def test_bad_file_skips_only_that_match(
    folder: str, corrupt, tmp_path: Path, settings: Settings, clean
) -> None:
    data = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, data)
    bad_path = data / folder / f"{BAD}.json"
    corrupt(bad_path)

    report, store = _run(data, settings)
    clean_report, clean_store = clean

    assert report.matches_skipped == 1
    assert report.as_dict()["matches_skipped"] == 1
    assert report.matches == clean_report.matches - 1

    # Nothing for the skipped match was written.
    assert BAD not in store._matches
    assert not [e for e in store._events.values() if e.match_id == BAD]
    assert not [f for f in store._frames.values() if f.match_id == BAD]

    # Every other match is exactly as in a clean ingest.
    def per_match(s: MemoryStore) -> dict[int, tuple[int, int]]:
        out: dict[int, tuple[int, int]] = {}
        for m in s._matches:
            out[m] = (
                sum(e.match_id == m for e in s._events.values()),
                sum(f.match_id == m for f in s._frames.values()),
            )
        return out

    expected = {m: c for m, c in per_match(clean_store).items() if m != BAD}
    assert per_match(store) == expected
    assert report.events == sum(c[0] for c in expected.values())
    assert report.frames == sum(c[1] for c in expected.values())

    kind = "three-sixty" if folder == "three-sixty" else "events"
    hits = [r for r in report.rejects if r.reason.startswith(f"unreadable {kind} file: ")]
    assert len(hits) == 1
    assert hits[0].source == str(bad_path)
    assert hits[0].payload == BAD
    if corrupt is not _undecodable:
        assert "JSONDecodeError at line" in hits[0].reason and "col" in hits[0].reason
    else:
        assert "UnicodeDecodeError" in hits[0].reason


def test_skipped_match_does_not_use_up_the_limit(
    tmp_path: Path, settings: Settings
) -> None:
    data = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, data)
    _truncate(data / "three-sixty" / f"{BAD}.json")
    clean_two, _ = _run(FIXTURE_DIR, settings, limit_matches=2)
    report, store = _run(data, settings, limit_matches=2)
    assert report.matches == 2 and report.matches_skipped <= 1
    assert BAD not in store._matches
    assert clean_two.matches == 2


def test_fail_on_bad_file_aborts(tmp_path: Path, settings: Settings) -> None:
    data = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, data)
    _nul(data / "three-sixty" / f"{BAD}.json")
    with pytest.raises(ValueError):
        _run(data, settings, fail_on_bad_file=True)


def test_non_data_errors_still_propagate(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pitchscanner.ingest import pipeline

    def boom(*a, **k):
        raise RuntimeError("a real bug")

    monkeypatch.setattr(pipeline, "parse_frames", boom)
    with pytest.raises(RuntimeError):
        _run(FIXTURE_DIR, settings)
