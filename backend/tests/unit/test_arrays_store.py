"""Array store mechanics: write mode, id table, detail files (local and over HTTP), CLI."""

from __future__ import annotations

import functools
import http.server
import json
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pitchscanner.api import create_app
from pitchscanner.config import Settings
from pitchscanner.ingest import ingest
from pitchscanner.store import ArrayStore, DetailUnavailableError, make_store
from pitchscanner.store.arrays import _decode_ids, _encode_id, _pack, parse_dtype_spec
from pitchscanner.store.memory import MemoryStore
from tests.conftest import FIXTURE_DIR

F32 = {"E0": "float32", "E1": "float32", "E2": "float32"}


@pytest.fixture(scope="module")
def built(settings: Settings, tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("built")
    store = ArrayStore.create(d, dtypes=F32)
    ingest(FIXTURE_DIR, store, settings=settings, encodings=["E1"], limit_matches=2)
    summary = store.finalize(extra_meta={"source": "fixtures"})
    assert summary["frames"] > 0
    return d


def test_layout_and_meta(built: Path) -> None:
    meta = json.loads((built / "meta.json").read_text())
    assert meta["format"] == 1 and meta["source"] == "fixtures"
    assert meta["counts"]["frames"] == meta["encodings"]["E1-v1"]["rows"]
    assert meta["columns"]["minute"] in ("int8", "int16")  # packed small
    assert meta["columns"]["match_id"] == "int32"
    assert (built / "vectors" / "E1-v1.npy").exists()
    assert not (built / "vectors" / "E2-v1.npy").exists()  # not ingested: absent
    assert len(list((built / "detail").glob("*.json.gz"))) == meta["counts"]["matches"]
    store = ArrayStore.open(built)
    assert store.vector_counts() == {"E1-v1": meta["counts"]["frames"]}
    assert isinstance(store._cols["minute"], np.memmap)
    summary = ArrayStore.summary(built)
    assert summary["searchable_bytes_per_frame"] > 0 and summary["detail_bytes"] > 0


def test_unserved_encoding_is_empty(built: Path) -> None:
    store = ArrayStore.open(built)
    assert store.exact_search("E2-v1", np.zeros(18), 5, "l2") == []
    assert store.baselines("E2-v1") == []


def test_mode_guards(tmp_path: Path, built: Path) -> None:
    w = ArrayStore.create(tmp_path / "w")
    with pytest.raises(RuntimeError, match="finalize"):
        w.counts()
    r = ArrayStore.open(built)
    with pytest.raises(RuntimeError, match="read-only"):
        r.upsert_matches([])
    with pytest.raises(FileNotFoundError):
        ArrayStore.open(tmp_path / "nothing")
    with pytest.raises(ValueError, match="format"):
        (tmp_path / "bad").mkdir()
        (tmp_path / "bad" / "meta.json").write_text('{"format": 99}')
        ArrayStore.open(tmp_path / "bad")


def test_create_refuses_foreign_directories_and_replaces_old_stores(
    tmp_path: Path, built: Path
) -> None:
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "precious.txt").write_text("keep")
    with pytest.raises(ValueError, match="not empty"):
        ArrayStore.create(tmp_path / "x")
    assert (tmp_path / "x" / "precious.txt").exists()

    import shutil

    shutil.copytree(built, tmp_path / "old")
    ArrayStore.create(tmp_path / "old")
    assert not (tmp_path / "old" / "meta.json").exists()
    assert not list((tmp_path / "old" / "detail").glob("*"))


def test_id_codec_and_table_handle_awkward_uuids() -> None:
    ids = [
        "00000000-0000-0000-0000-000000000000",
        "ffffffff-ffff-ffff-ffff-ffffffffffff",
        "12345678-1234-5678-1234-567812345600",  # ends in a NUL byte
        "12345678-1234-5678-1234-567812340000",
    ]
    keys = np.frombuffer(b"".join(_encode_id(i) for i in ids), dtype="S16")
    assert _decode_ids(keys) == ids
    order = np.argsort(keys)
    srt = keys[order]
    pos = np.searchsorted(srt, keys)
    assert (srt[pos] == keys).all()
    for bad in ["x", ids[1].upper(), "g" * 36, ids[0][:-1] + " ", ""]:
        with pytest.raises(ValueError):
            _encode_id(bad)
    rnd = [str(uuid.UUID(int=i * 7919 << 20)) for i in range(2000)]
    k2 = np.frombuffer(b"".join(_encode_id(i) for i in rnd), dtype="S16")
    s2 = np.sort(k2)
    assert (s2[np.searchsorted(s2, k2)] == k2).all()


def test_pack_and_dtype_spec(tmp_path: Path) -> None:
    assert _pack(np.array([0, 100])).dtype == np.int8
    assert _pack(np.array([-1, 300])).dtype == np.int16
    assert _pack(np.array([0, 70000])).dtype == np.int32
    assert _pack(np.array([0, 2**40])).dtype == np.int64
    assert _pack(np.zeros(0, dtype=np.int64)).dtype == np.int8
    assert _pack(np.array([True])).dtype == bool
    assert parse_dtype_spec("")["E1"] == "float32"
    assert parse_dtype_spec("E1=float16, e2=float32") == {
        "E0": "float16", "E1": "float16", "E2": "float32",
    }
    with pytest.raises(ValueError):
        parse_dtype_spec("E1=int8")
    target = tmp_path / "never-created"
    with pytest.raises(ValueError):
        ArrayStore.create(target, dtypes={"E1": "float64"})
    assert not target.exists()


def _tiny(tmp_path: Path, settings: Settings) -> tuple[ArrayStore, MemoryStore]:
    mem = MemoryStore()
    ingest(FIXTURE_DIR, mem, settings=settings, encodings=["E1"], limit_matches=1)
    return ArrayStore.create(tmp_path), mem


def _replay(src: MemoryStore, dst: ArrayStore, *, vectors_first: bool = False) -> None:
    ids, mat = src._vectors["E1-v1"]
    if vectors_first:  # arrive in reverse, before anything else
        dst.upsert_vectors("E1-v1", 72, [(i, mat[k]) for k, i in reversed(list(enumerate(ids)))])
    dst.upsert_matches(src._matches.values())
    dst.upsert_events(src._events.values())
    dst.upsert_frames(src._frames.values(), src._metas.values())
    dst.upsert_outcomes(src._outcomes.values())
    if not vectors_first:
        dst.upsert_vectors("E1-v1", 72, [(i, mat[k]) for k, i in enumerate(ids)])


def test_vectors_may_arrive_in_any_order(tmp_path: Path, settings: Settings) -> None:
    dst, src = _tiny(tmp_path, settings)
    _replay(src, dst, vectors_first=True)
    dst.finalize()
    got = ArrayStore.open(tmp_path)
    ids, mat = src._vectors["E1-v1"]
    hit = got.exact_search("E1-v1", mat[17], 1, "cosine")[0]
    assert hit.event_id == ids[17] and abs(hit.distance) < 1e-5


def test_missing_vectors_and_orphans_are_errors(tmp_path: Path, settings: Settings) -> None:
    dst, src = _tiny(tmp_path / "a", settings)
    ids, mat = src._vectors["E1-v1"]
    _replay(src, dst)
    dst._w_vec_data["E1-v1"][0] = dst._w_vec_data["E1-v1"][0][:-5]
    dst._w_vec_ids["E1-v1"][0] = dst._w_vec_ids["E1-v1"][0][:-80]
    with pytest.raises(ValueError, match="every frame needs a vector"):
        dst.finalize()

    dst, src = _tiny(tmp_path / "b", settings)
    _replay(src, dst)
    dst.upsert_vectors("E1-v1", 72, [(str(uuid.uuid4()), mat[0])])
    with pytest.raises(ValueError, match="not frames"):
        dst.finalize()

    dst, src = _tiny(tmp_path / "c", settings)
    _replay(src, dst)
    o = next(iter(src._outcomes.values()))
    dst.upsert_outcomes([o.model_copy(update={"event_id": str(uuid.uuid4())})])
    with pytest.raises(ValueError, match="outcomes have no frame"):
        dst.finalize()

    dst, src = _tiny(tmp_path / "d", settings)
    dst.upsert_events(src._events.values())  # events for a match that was never upserted
    with pytest.raises(ValueError, match="never upserted"):
        dst.finalize()

    dst, src = _tiny(tmp_path / "e", settings)
    _replay(src, dst)
    with pytest.raises(ValueError, match=r"-d vectors"):
        dst.upsert_vectors("E1-v1", 5, [(ids[0], mat[0])])
    with pytest.raises(ValueError, match="UUID"):
        dst.upsert_outcomes([o.model_copy(update={"event_id": "not-a-uuid"})])
    with pytest.raises(ValueError, match="append-only"):
        dst.upsert_matches([])  # flushes the match ...
        dst.upsert_frames(src._frames.values(), src._metas.values())
    with pytest.raises(ValueError, match="writing to"):
        dst.finalize(tmp_path / "elsewhere")
    assert dst.upsert_vectors("E2-v1", 18, []) == 0


def test_duplicate_frame_in_a_match_keeps_the_last(tmp_path: Path, settings: Settings) -> None:
    dst, src = _tiny(tmp_path, settings)
    frames = list(src._frames.values())
    dup = frames[3].model_copy(update={"ball_x": 1.5})
    dst.upsert_matches(src._matches.values())
    dst.upsert_events(src._events.values())
    dst.upsert_frames([*frames, dup], list(src._metas.values()))
    dst.upsert_outcomes(src._outcomes.values())
    ids, mat = src._vectors["E1-v1"]
    dst.upsert_vectors("E1-v1", 72, [(i, mat[k]) for k, i in enumerate(ids)])
    dst.finalize()
    got = ArrayStore.open(tmp_path)
    assert got.counts().frames == len(frames)
    assert got.frame(dup.event_id).ball_x == 1.5  # type: ignore[union-attr]


# --- detail over HTTP ---------------------------------------------------------
@pytest.fixture
def http_detail(built: Path) -> Iterator[str]:
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(built / "detail")
    )
    handler.log_message = lambda *a, **k: None  # type: ignore[attr-defined]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


def test_detail_over_http_matches_local(built: Path, http_detail: str) -> None:
    local = ArrayStore.open(built)
    remote = ArrayStore.open(built, detail_base_url=http_detail)
    ids = list(remote.metas([e for e, _ in local._frame_rows(
        [i for i in _decode_ids(np.asarray(local._ids[:60]))]
    )]))
    assert ids
    assert remote.frames_by_id(ids) == local.frames_by_id(ids)
    assert remote.possession_events(ids[0]) == local.possession_events(ids[0])


def test_http_failures_raise_detail_unavailable(built: Path, http_detail: str) -> None:
    local = ArrayStore.open(built)
    eid = _decode_ids(np.asarray(local._ids[:1]))[0]
    dead = ArrayStore.open(built, detail_base_url=http_detail + "missing/")
    with pytest.raises(DetailUnavailableError):
        dead.event(eid)
    refused = ArrayStore.open(built, detail_base_url="http://127.0.0.1:9/", detail_timeout_s=1)
    with pytest.raises(DetailUnavailableError):
        refused.event(eid)
    with pytest.raises(ValueError, match="http"):
        ArrayStore.open(built, detail_base_url="file:///etc/")


def test_http_server_that_already_decoded_gzip_is_accepted(built: Path, tmp_path: Path) -> None:
    import gzip

    plain = tmp_path / "plain"
    plain.mkdir()
    for f in (built / "detail").glob("*.json.gz"):
        (plain / f.name).write_bytes(gzip.decompress(f.read_bytes()))
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(plain))
    handler.log_message = lambda *a, **k: None  # type: ignore[attr-defined]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        local = ArrayStore.open(built)
        remote = ArrayStore.open(built, detail_base_url=f"http://127.0.0.1:{server.server_port}")
        eid = _decode_ids(np.asarray(local._ids[:1]))[0]
        assert remote.event(eid) == local.event(eid)
    finally:
        server.shutdown()
        server.server_close()


def test_lru_keeps_only_a_few_matches(built: Path) -> None:
    store = ArrayStore.open(built, detail_cache=1)
    first, second = store._match_ids[:2]
    store._detail(first)
    store._detail(second)
    assert list(store._details._cache) == [second]  # type: ignore[union-attr]
    store._detail(second)
    store.close()
    assert not store._details._cache  # type: ignore[union-attr]


def test_corrupt_and_missing_detail_files(built: Path, tmp_path: Path) -> None:
    import shutil

    d = tmp_path / "copy"
    shutil.copytree(built, d)
    local = ArrayStore.open(d)
    m1, m2 = local._match_ids[:2]
    (d / "detail" / f"{m1}.json.gz").write_bytes(b"\x1f\x8b not gzip")
    (d / "detail" / f"{m2}.json.gz").write_bytes(b"{not json")
    for m in (m1, m2):
        with pytest.raises(DetailUnavailableError):
            local._detail(m)
    (d / "detail" / f"{m1}.json.gz").unlink()
    with pytest.raises(DetailUnavailableError):
        local._detail(m1)


def test_missing_detail_is_a_503_not_a_500(built: Path, tmp_path: Path, settings: Settings) -> None:
    import shutil

    d = tmp_path / "copy"
    shutil.copytree(built, d)
    store = ArrayStore.open(d)
    eid = _decode_ids(np.asarray(store._ids[:1]))[0]
    shutil.rmtree(d / "detail")
    with TestClient(create_app(settings=settings, store=store)) as client:
        assert client.get(f"/v1/situations/{eid}/sequence").status_code == 503
        # The searchable half does not need detail files at all.
        body = client.post("/v1/search", json={
            "ball": {"x": 84, "y": 22}, "attackers": [{"x": 86, "y": 20}],
            "defenders": [{"x": 92, "y": 25}], "k": 5,
        })
        assert body.status_code == 200 and body.json()["results"]


# --- factory and CLI -----------------------------------------------------------------
def test_make_store_reads_settings(built: Path, tmp_path: Path) -> None:
    cfg = Settings(store="arrays", arrays_dir=built)
    store = make_store(cfg)
    assert isinstance(store, ArrayStore) and store.counts().frames > 0
    with pytest.raises(ValueError, match="ARRAYS_DIR"):
        make_store(Settings(store="arrays"))
    remote = make_store(Settings(store="arrays", arrays_dir=built, detail_base_url="http://x:1"))
    assert remote._details._base == "http://x:1"  # type: ignore[union-attr]
