"""A NUL-damaged three-sixty file is salvaged; every other bad file still skips its match."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from pitchscanner.config import Settings
from pitchscanner.ingest import ingest
from pitchscanner.ingest.parse import read_frames_tolerant, salvage_json_array
from pitchscanner.store.memory import MemoryStore
from tests.conftest import FIXTURE_DIR

BAD = 3900002
NUL_RUN = 16384


def _overwrite_with_nuls(path: Path, at: float = 0.5, n: int = NUL_RUN) -> None:
    """Replace ``n`` bytes with NULs, the way the upstream file was damaged."""

    raw = bytearray(path.read_bytes())
    start = int(len(raw) * at)
    raw[start : start + n] = b"\x00" * n
    path.write_bytes(bytes(raw))


def _array_text(n: int, *, indent: int | None = 2) -> str:
    objs = [
        {"event_uuid": f"id-{i:04d}", "visible_area": [1.5, 2.5], "freeze_frame": [{"a": i}] * 20}
        for i in range(n)
    ]
    return json.dumps(objs, indent=indent)


@pytest.mark.parametrize("indent", [None, 2])
def test_salvage_drops_only_the_damaged_objects(indent: int | None) -> None:
    text = _array_text(60, indent=indent)
    start = len(text) // 2
    damaged = text[:start] + "\x00" * 900 + text[start + 900 :]

    items, info = salvage_json_array(damaged)  # type: ignore[misc]
    ids = [o["event_uuid"] for o in items]

    original = [f"id-{i:04d}" for i in range(60)]
    assert ids == [i for i in original if i in set(ids)]  # order kept, nothing invented
    assert 0 < len(ids) < 60
    assert info.recovered == len(ids)
    # What was lost is at least what is missing from the middle, and is an estimate.
    assert info.lost_estimate >= 1
    assert abs(info.lost_estimate - (60 - len(ids))) <= 2
    assert info.reason == (
        f"salvaged three-sixty file: recovered {len(ids)} frames, lost ~{info.lost_estimate}"
    )
    # Objects either side of the damage are intact, byte for byte.
    assert items[0] == json.loads(text)[0]
    assert items[-1] == json.loads(text)[-1]


def test_salvage_handles_two_damaged_runs_and_a_damaged_tail() -> None:
    text = _array_text(80)
    n = len(text)
    chars = list(text)
    for at in (0.25, 0.6):
        s = int(n * at)
        chars[s : s + 700] = ["\x00"] * 700
    tail = int(n * 0.97)
    chars[tail : tail + 50] = ["\x00"] * 50
    items, info = salvage_json_array("".join(chars))  # type: ignore[misc]
    assert 55 <= len(items) <= 77
    assert info.lost_estimate >= 3


def test_salvage_refuses_what_is_not_an_array_of_objects() -> None:
    assert salvage_json_array("") is None
    assert salvage_json_array('{"a": 1}') is None
    assert salvage_json_array("[\x00\x00\x00") is None


def test_tolerant_read_is_only_for_nul_damage(tmp_path: Path) -> None:
    good = tmp_path / "t.json"
    text = _array_text(20)

    good.write_text(text[: len(text) // 2])  # truncated, no NULs
    assert read_frames_tolerant(good) is None

    good.write_text(text[:300] + "\x00" * 500 + text[800:])
    got = read_frames_tolerant(good)
    assert got is not None and got[1].recovered >= 1


def _ingest(data: Path, settings: Settings, **kw):
    store = MemoryStore()
    report = ingest(data, store, settings=settings, encodings=["E1"], **kw)
    return report, store


@pytest.fixture(scope="module")
def clean(settings: Settings):
    return _ingest(FIXTURE_DIR, settings)


def test_ingest_salvages_a_nul_damaged_three_sixty_file(
    tmp_path: Path, settings: Settings, clean
) -> None:
    data = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, data)
    path = data / "three-sixty" / f"{BAD}.json"
    _overwrite_with_nuls(path, n=2048)

    report, store = _ingest(data, settings)
    clean_report, clean_store = clean

    assert report.matches_skipped == 0
    assert report.matches_salvaged == 1
    assert report.as_dict()["matches_salvaged"] == 1
    assert report.matches == clean_report.matches
    assert BAD in store._matches

    got = {e for e, f in store._frames.items() if f.match_id == BAD}
    want = {e for e, f in clean_store._frames.items() if f.match_id == BAD}
    assert got and got < want  # recovered some, lost some, invented none
    for eid in got:  # every recovered frame is exactly the clean one
        assert store._frames[eid] == clean_store._frames[eid]
        assert store._outcomes[eid] == clean_store._outcomes[eid]
    # Other matches are untouched, and the events file is whole.
    assert report.events == clean_report.events
    assert report.frames == clean_report.frames - (len(want) - len(got))

    hits = [r for r in report.rejects if r.reason.startswith("salvaged three-sixty file: ")]
    assert len(hits) == 1
    assert hits[0].source == str(path) and hits[0].payload == BAD
    assert "recovered " in hits[0].reason and ", lost ~" in hits[0].reason


def test_salvage_is_not_used_under_fail_on_bad_file(tmp_path: Path, settings: Settings) -> None:
    data = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, data)
    _overwrite_with_nuls(data / "three-sixty" / f"{BAD}.json", n=2048)
    with pytest.raises(json.JSONDecodeError):
        _ingest(data, settings, fail_on_bad_file=True)


def test_events_files_are_never_salvaged(tmp_path: Path, settings: Settings, clean) -> None:
    data = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, data)
    _overwrite_with_nuls(data / "events" / f"{BAD}.json", n=2048)
    report, store = _ingest(data, settings)
    assert report.matches_skipped == 1 and report.matches_salvaged == 0
    assert BAD not in store._matches


def test_truncated_three_sixty_is_still_skipped(tmp_path: Path, settings: Settings) -> None:
    data = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, data)
    p = data / "three-sixty" / f"{BAD}.json"
    p.write_bytes(p.read_bytes()[: p.stat().st_size // 2])
    report, store = _ingest(data, settings)
    assert report.matches_skipped == 1 and report.matches_salvaged == 0
    assert BAD not in store._matches
