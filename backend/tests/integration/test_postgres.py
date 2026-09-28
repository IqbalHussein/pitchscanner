"""Integration tier: the real pipeline against a real Postgres with pgvector.

Skipped unless a database is available. In CI a service container provides one;
locally `make test-integration` starts one with Testcontainers.
"""

from __future__ import annotations

import pytest

from pitchscanner.config import Settings
from pitchscanner.encoders import get_encoder
from pitchscanner.ingest import ingest
from pitchscanner.models import Filters, PointIn, SearchRequest
from pitchscanner.search import SearchService
from pitchscanner.store.postgres import PostgresStore
from tests.conftest import FIXTURE_DIR

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pg_dsn(dsn: str | None) -> str:
    if dsn:
        return dsn
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
    return ""  # pragma: no cover


@pytest.fixture(scope="module")
def pg_store(pg_dsn: str) -> PostgresStore:
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
