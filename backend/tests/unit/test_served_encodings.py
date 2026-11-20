"""The API only offers, and only accepts, encodings that have vectors."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pitchscanner.api import create_app
from pitchscanner.config import Settings
from pitchscanner.ingest import ingest
from pitchscanner.store.memory import MemoryStore
from tests.conftest import FIXTURE_DIR

SKETCH = {
    "ball": {"x": 90.0, "y": 40.0},
    "attackers": [{"x": 92.0, "y": 38.0}],
    "defenders": [{"x": 100.0, "y": 40.0}],
    "k": 5,
}


@pytest.fixture(scope="module")
def e1_only_client(settings: Settings):
    store = MemoryStore()
    ingest(FIXTURE_DIR, store, settings=settings, encodings=["E1"], limit_matches=2)
    with TestClient(create_app(settings=settings, store=store)) as client:
        yield client


def test_filters_and_meta_list_only_e1(e1_only_client: TestClient) -> None:
    assert e1_only_client.get("/v1/filters").json()["encodings"] == ["E1"]
    meta = e1_only_client.get("/v1/meta").json()
    assert meta["encodings"] == ["E1"]
    assert meta["default_encoding"] == "E1"


def test_search_accepts_e1_any_case(e1_only_client: TestClient) -> None:
    for name in ("E1", "e1"):
        r = e1_only_client.post("/v1/search", json={**SKETCH, "options": {"encoding": name}})
        assert r.status_code == 200, r.text
        assert r.json()["results"]


@pytest.mark.parametrize("name", ["E0", "E2"])
def test_search_rejects_known_encoding_without_vectors(
    e1_only_client: TestClient, name: str
) -> None:
    r = e1_only_client.post("/v1/search", json={**SKETCH, "options": {"encoding": name}})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert f"encoding {name} has no vectors" in detail
    assert "['E1']" in detail


def test_search_rejects_unknown_encoding(e1_only_client: TestClient) -> None:
    r = e1_only_client.post("/v1/search", json={**SKETCH, "options": {"encoding": "E9"}})
    assert r.status_code == 422
    assert "unknown encoding" in r.json()["detail"]


def test_empty_store_serves_nothing(settings: Settings) -> None:
    with TestClient(create_app(settings=settings, store=MemoryStore())) as client:
        assert client.get("/v1/filters").json()["encodings"] == []
        r = client.post("/v1/search", json=SKETCH)
        assert r.status_code == 422
        assert "nothing ingested yet" in r.json()["detail"]


def test_full_store_still_lists_every_encoding(api_client: TestClient) -> None:
    assert api_client.get("/v1/filters").json()["encodings"] == ["E0", "E1", "E2"]
