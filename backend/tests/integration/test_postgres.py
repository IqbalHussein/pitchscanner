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
