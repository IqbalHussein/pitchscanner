"""Integration tier: the real pipeline against a real Postgres with pgvector.

Skipped unless a database is available. In CI a service container provides one;
locally `make test-integration` starts one with Testcontainers.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from pitchscanner.config import Settings
from pitchscanner.encoders import get_encoder
from pitchscanner.ingest import ingest
from pitchscanner.models import Filters, PointIn, SearchRequest
from pitchscanner.search import SearchService
from pitchscanner.store.memory import MemoryStore
from pitchscanner.store.postgres import PostgresStore
from tests.conftest import FIXTURE_DIR

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pg_dsn(dsn: str | None) -> Iterator[str]:
    """A DSN from the environment, or a throwaway container.

    Both branches have to *yield*: `return` inside a generator fixture ends it
    without producing a value, and pytest reports "pg_dsn did not yield a value"
    - which is what happened for every run that supplied PITCHSCANNER_TEST_DSN,
    i.e. exactly the CI configuration this tier exists for.
    """

    if dsn:
        yield dsn
        return
    try:
        from testcontainers.postgres import PostgresContainer
    except ImportError:  # pragma: no cover - optional dev dependency
        pytest.skip("set PITCHSCANNER_TEST_DSN or install testcontainers")
    container = PostgresContainer("pgvector/pgvector:pg16")
    container.start()
    try:
        yield container.get_connection_url().replace("postgresql+psycopg2", "postgresql")
    finally:
        container.stop()


@pytest.fixture(scope="module")
def pg_store(pg_dsn: str) -> Iterator[PostgresStore]:
    settings = Settings(store="postgres", dsn=pg_dsn, fixture_dir=FIXTURE_DIR)
    store = PostgresStore(pg_dsn)
    ingest(FIXTURE_DIR, store, settings=settings, limit_matches=1)
    yield store
    store.close()


def test_ingest_then_find_a_frame_at_distance_zero(pg_store: PostgresStore) -> None:
    """Ingest one match, search for one of its frames, get it back at distance 0."""

    encoder = get_encoder("E1")
    frame = pg_store.frames(limit=1)[0]
    hits = pg_store.exact_search(
        encoder.version, encoder.encode(frame.to_situation()), 1, encoder.metric
    )
    assert hits
    assert hits[0].event_id == frame.event_id
    assert hits[0].distance == pytest.approx(0.0, abs=1e-5)


def test_ingest_is_idempotent(pg_store: PostgresStore) -> None:
    """Rerunning on the same input must converge, not duplicate."""

    before = pg_store.counts()
    settings = Settings(store="postgres", dsn=pg_store.dsn, fixture_dir=FIXTURE_DIR)
    ingest(FIXTURE_DIR, pg_store, settings=settings, limit_matches=1)
    after = pg_store.counts()
    assert before.as_dict() == after.as_dict()


def test_ann_and_exact_agree_at_this_size(pg_store: PostgresStore) -> None:
    """With an HNSW index over a few thousand vectors, recall should be ~1."""

    encoder = get_encoder("E1")
    frames = pg_store.frames(limit=20)
    recalls = []
    for frame in frames:
        vector = encoder.encode(frame.to_situation())
        approx = {
            c.event_id for c in pg_store.ann_search(encoder.version, vector, 10, encoder.metric)
        }
        exact = {
            c.event_id
            for c in pg_store.exact_search(encoder.version, vector, 10, encoder.metric)
        }
        recalls.append(len(approx & exact) / len(exact))
    assert sum(recalls) / len(recalls) >= 0.95


def test_filters_push_down_to_sql(pg_store: PostgresStore) -> None:
    total = pg_store.count_matching(get_encoder("E1").version, None)
    losing = pg_store.count_matching(
        get_encoder("E1").version, Filters(game_state="losing")
    )
    assert 0 <= losing <= total


def test_venue_filter_is_backfilled_and_agrees_with_the_matches_table(
    pg_store: PostgresStore,
) -> None:
    version = get_encoder("E1").version
    total = pg_store.count_matching(version, None)
    home = pg_store.count_matching(version, Filters(venue="home"))
    away = pg_store.count_matching(version, Filters(venue="away"))
    assert 0 < home < total
    assert home + away == total
    with pg_store.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) AS n FROM frames f JOIN matches m USING (match_id)"
            " WHERE (f.possession_team_id = m.home_team_id) <> f.possession_is_home"
        )
        row = cur.fetchone()
    assert row is not None and row["n"] == 0


def test_search_service_against_postgres(pg_store: PostgresStore) -> None:
    service = SearchService(pg_store, Settings(store="postgres", dsn=pg_store.dsn))
    outcome = service.search(
        SearchRequest(
            ball=PointIn(x=84.0, y=22.0),
            attackers=[PointIn(x=86.0, y=20.0)],
            defenders=[PointIn(x=92.0, y=25.0)],
            k=10,
        )
    )
    assert outcome.results
    assert outcome.stats is not None
    keys = [(r.match_id, r.possession_id) for r in outcome.results]
    assert len(keys) == len(set(keys))


def test_reject_log_table_accepts_rows(pg_store: PostgresStore) -> None:
    pg_store.log_reject("test", "deliberate", {"hello": "world"})
    with pg_store.conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM ingest_rejects WHERE source = 'test'")
        assert (cur.fetchone() or {"n": 0})["n"] >= 1


@pytest.fixture(scope="module")
def twin_memory_store() -> MemoryStore:
    """The same fixture match, ingested into the in-memory store.

    The two backends share one definition of filter semantics
    (``store/filtering.py``: ``matches`` for memory, ``to_sql`` for Postgres) but
    two implementations of it. Divergence between them would be invisible to
    every other test, because each tier only ever looks at one store.
    """

    store = MemoryStore()
    ingest(
        FIXTURE_DIR,
        store,
        settings=Settings(store="memory", fixture_dir=FIXTURE_DIR),
        limit_matches=1,
    )
    return store


#: Filters chosen to hit every branch of `to_sql`: ids, ranges, enums, the
#: combination, and one predicate that should match nothing.
_FILTER_CASES = [
    Filters(),
    Filters(game_state="losing"),
    Filters(game_state="drawing"),
    Filters(minute_range=(0, 15)),
    Filters(minute_range=(75, 90)),
    Filters(stages=["group"]),
    Filters(team_role="attacking"),
    Filters(minute_range=(0, 45), game_state="winning"),
    Filters(venue="home"),
    Filters(venue="away"),
    Filters(venue="home", game_state="losing", minute_range=(45, 98)),
    Filters(competition_ids=[999999]),
]

#: Larger than any fixture corpus: "return everything that matches", so the two
#: stores can be compared as sets rather than as top-k lists.
_ALL = 10**6


def _filter_id(filters: Filters) -> str:
    return filters.model_dump_json(exclude_defaults=True) or "empty"


def _matching_ids(store: object, vector, filters: Filters) -> set[str]:
    encoder = get_encoder("E1")
    return {
        c.event_id
        for c in store.exact_search(  # type: ignore[attr-defined]
            encoder.version, vector, _ALL, encoder.metric, filters
        )
    }


@pytest.mark.parametrize("filters", _FILTER_CASES, ids=_filter_id)
def test_filter_semantics_match_the_memory_store(
    pg_store: PostgresStore, twin_memory_store: MemoryStore, filters: Filters
) -> None:
    """Both stores must keep exactly the same frames, for every filter shape.

    Compared on the frames the two stores share rather than on raw counts: a
    developer's database may already hold more matches than the twin, and that is
    not a semantic difference.
    """

    encoder = get_encoder("E1")
    frame = twin_memory_store.frames()[0]
    vector = encoder.encode(frame.to_situation())
    shared = {f.event_id for f in twin_memory_store.frames()}

    mem_ids = _matching_ids(twin_memory_store, vector, filters)
    pg_ids = _matching_ids(pg_store, vector, filters) & shared
    assert pg_ids == mem_ids


def test_exact_search_agrees_with_the_memory_store(
    pg_store: PostgresStore, twin_memory_store: MemoryStore
) -> None:
    """Same query, same subset, same ranking - to a float32's tolerance.

    This is what makes "the in-memory store is the exact-search ground truth" a
    checked claim rather than an assertion in a README.
    """

    encoder = get_encoder("E1")
    shared = {f.event_id for f in twin_memory_store.frames()}
    for frame in twin_memory_store.frames()[:5]:
        vector = encoder.encode(frame.to_situation())
        for filters in (Filters(), Filters(game_state="drawing")):
            pg_hits = [
                c
                for c in pg_store.exact_search(
                    encoder.version, vector, _ALL, encoder.metric, filters
                )
                if c.event_id in shared
            ][:5]
            mem_hits = twin_memory_store.exact_search(
                encoder.version, vector, 5, encoder.metric, filters
            )
            assert [h.event_id for h in pg_hits] == [h.event_id for h in mem_hits]
            for a, b in zip(pg_hits, mem_hits, strict=True):
                assert a.distance == pytest.approx(b.distance, abs=1e-5)


def test_vector_counts_per_encoding(pg_store: PostgresStore) -> None:
    counts = pg_store.vector_counts()
    assert counts
    assert all(n > 0 for n in counts.values())
    assert sum(counts.values()) == pg_store.counts().vectors
    assert pg_store.effective_index_kind is None


# --- transaction-pooler safety (Supabase port 6543) ---------------------------


def _prepared(store: PostgresStore) -> int:
    with store.conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM pg_prepared_statements")
        row = cur.fetchone()
    return int(row["n"]) if row else -1


def _search_repeatedly(store: PostgresStore, times: int = 12) -> None:
    encoder = get_encoder("E1")
    vector = encoder.encode(store.frames(limit=1)[0].to_situation())
    for _ in range(times):  # well past psycopg's prepare_threshold of 5
        store.ann_search(encoder.version, vector, 5, encoder.metric)
        store.count_matching(encoder.version, None)
        store.counts()


def test_default_connection_prepares_repeated_statements(pg_dsn: str) -> None:
    """The control for the next test: without it the assertion proves nothing."""

    store = PostgresStore(pg_dsn)
    try:
        _search_repeatedly(store)
        assert _prepared(store) > 0
    finally:
        store.close()


def test_prepare_off_never_prepares_even_when_ingesting(pg_dsn: str) -> None:
    settings = Settings(store="postgres", dsn=pg_dsn, fixture_dir=FIXTURE_DIR, db_prepare=False)
    store = PostgresStore.from_settings(settings)
    assert store.prepare is False
    try:
        ingest(FIXTURE_DIR, store, settings=settings, limit_matches=1, encodings=["E1"])
        _search_repeatedly(store)
        assert _prepared(store) == 0
    finally:
        store.close()


def test_search_settings_hold_for_the_query(
    pg_store: PostgresStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """hnsw.ef_search and the exact-scan switches must be live while the search
    runs. With autocommit on, a bare set_config(..., true) is gone before the
    next statement, so this was silently never applied."""

    from pitchscanner.store import postgres as pgmod

    seen: dict[str, str] = {}
    real = pgmod._set_local

    def spy(cur, name, value):  # type: ignore[no-untyped-def]
        real(cur, name, value)
        cur.execute("SELECT current_setting(%s) AS v", (name,))
        seen[name] = cur.fetchone()["v"]

    monkeypatch.setattr(pgmod, "_set_local", spy)
    encoder = get_encoder("E1")
    vector = encoder.encode(pg_store.frames(limit=1)[0].to_situation())
    store = PostgresStore(pg_store.dsn, ef_search=33, probes=7)
    try:
        store.ann_search(encoder.version, vector, 5, encoder.metric)
        assert seen == {"hnsw.ef_search": "33", "ivfflat.probes": "7"}
        seen.clear()
        store.exact_search(encoder.version, vector, 5, encoder.metric)
        assert seen == {"enable_indexscan": "off", "enable_bitmapscan": "off"}
        # And nothing leaks into the connection afterwards.
        with store.conn.cursor() as cur:
            cur.execute("SELECT current_setting('enable_indexscan') AS v")
            assert cur.fetchone()["v"] == "on"
    finally:
        store.close()


@pytest.fixture
def e1_only_dsn(pg_dsn: str) -> Iterator[str]:
    """A scratch database holding the fixtures ingested for E1 only."""

    import uuid

    import psycopg
    from psycopg.conninfo import make_conninfo

    name = f"codewk_e1only_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(pg_dsn, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    dsn = make_conninfo(pg_dsn, dbname=name)
    try:
        yield dsn
    finally:
        with psycopg.connect(pg_dsn, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def test_api_offers_only_encodings_with_vectors(e1_only_dsn: str) -> None:
    from fastapi.testclient import TestClient

    from pitchscanner.api import create_app

    settings = Settings(
        store="postgres", dsn=e1_only_dsn, fixture_dir=FIXTURE_DIR, db_prepare=False
    )
    store = PostgresStore.from_settings(settings)
    ingest(FIXTURE_DIR, store, settings=settings, limit_matches=2, encodings=["E1"])
    assert list(store.vector_counts()) == ["E1-v1"]
    sketch = {
        "ball": {"x": 90.0, "y": 40.0},
        "attackers": [{"x": 92.0, "y": 38.0}],
        "defenders": [{"x": 100.0, "y": 40.0}],
        "k": 5,
    }
    with TestClient(create_app(settings=settings, store=store)) as client:
        assert client.get("/v1/filters").json()["encodings"] == ["E1"]
        assert client.get("/v1/meta").json()["encodings"] == ["E1"]
        assert client.post("/v1/search", json=sketch).status_code == 200
        for name in ("E0", "E2"):
            r = client.post("/v1/search", json={**sketch, "options": {"encoding": name}})
            assert r.status_code == 422
            assert "has no vectors" in r.json()["detail"]
    assert _prepared(store) == 0
