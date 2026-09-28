"""API behaviour and the request/response contract."""

from __future__ import annotations

import pytest

SKETCH = {
    "ball": {"x": 84, "y": 22},
    "attackers": [{"x": 86, "y": 20}, {"x": 95, "y": 35}],
    "defenders": [{"x": 92, "y": 25}, {"x": 100, "y": 30}],
    "k": 10,
}


def test_healthz(api_client) -> None:
    assert api_client.get("/healthz").json() == {"status": "ok"}


def test_readyz_reports_counts(api_client) -> None:
    body = api_client.get("/readyz").json()
    assert body["status"] == "ready"
    assert body["counts"]["frames"] > 0


def test_search_returns_results_stats_and_timings(api_client) -> None:
    body = api_client.post("/v1/search", json=SKETCH).json()
    assert 0 < len(body["results"]) <= 10
    assert body["stats"]["n"] == len(body["results"])
    assert set(body["timing_ms"]) == {"encode", "search", "dedup", "aggregate", "total"}
    top = body["results"][0]
    assert top["distance"] <= body["results"][-1]["distance"]
    assert {"event_id", "match", "minute", "outcome", "similarity"} <= set(top)


def test_search_respects_the_spec_example_payload(api_client) -> None:
    """The exact body from the spec has to be accepted as written."""

    payload = {
        "ball": {"x": 84, "y": 22},
        "attackers": [{"x": 86, "y": 20}, {"x": 95, "y": 35}],
        "defenders": [{"x": 92, "y": 25}, {"x": 100, "y": 30}],
        "k": 25,
        "filters": {"competition_ids": [55], "game_state": "any"},
        "options": {"mirror": True, "encoding": "E1", "window_s": 15},
    }
    response = api_client.post("/v1/search", json=payload)
    assert response.status_code == 200
    assert response.json()["encoding"] == "E1"


@pytest.mark.parametrize(
    "payload",
    [
        {**SKETCH, "ball": {"x": 999, "y": 22}},
        {**SKETCH, "ball": {"x": 84, "y": -1}},
        {**SKETCH, "k": 500},
        {**SKETCH, "attackers": [{"x": 1, "y": 1}] * 12},
        {"ball": {"x": 84, "y": 22}},
        {**SKETCH, "options": {"encoding": "E9"}},
        {**SKETCH, "filters": {"nonsense": 1}},
    ],
)
def test_invalid_requests_are_rejected(api_client, payload: dict) -> None:
    assert api_client.post("/v1/search", json=payload).status_code == 422


def test_filter_text_becomes_structured_filters(api_client) -> None:
    body = api_client.post(
        "/v1/search", json={**SKETCH, "filter_text": "when losing in the second half"}
    ).json()
    assert body["filters_applied"]["game_state"] == "losing"
    assert body["filters_applied"]["minute_range"] == [45, 98]
    assert [c["field"] for c in body["parsed_filters"]["chips"]] == [
        "game_state",
        "minute_range",
    ]


def test_explicit_filters_win_over_text(api_client) -> None:
    body = api_client.post(
        "/v1/search",
        json={**SKETCH, "filter_text": "when losing", "filters": {"game_state": "winning"}},
    ).json()
    assert body["filters_applied"]["game_state"] == "winning"


def test_filters_endpoint_only_lists_covered_data(api_client) -> None:
    body = api_client.get("/v1/filters").json()
    assert body["competitions"]
    for competition in body["competitions"]:
        assert competition["frames"] > 0
        assert competition["seasons"]
        for season in competition["seasons"]:
            assert season["frames"] > 0
    assert body["total_frames"] > 0
    assert "E1" in body["encodings"]


def test_parse_endpoint_reports_unmatched_words(api_client) -> None:
    body = api_client.post("/v1/filters/parse", json={"text": "rainy Spain games"}).json()
    assert "rainy" in body["unmatched"]
    assert any(chip["field"] == "team" for chip in body["chips"])


def test_situation_and_sequence_round_trip(api_client) -> None:
    results = api_client.post("/v1/search", json=SKETCH).json()["results"]
    event_id = results[0]["event_id"]

    detail = api_client.get(f"/v1/situations/{event_id}").json()
    assert detail["event_id"] == event_id
    assert detail["attackers"] or detail["defenders"]
    assert 0 <= detail["ball"]["x"] <= 120

    sequence = api_client.get(f"/v1/situations/{event_id}/sequence").json()
    assert sequence["event_id"] == event_id
    offsets = [e["second_offset"] for e in sequence["events"]]
    assert offsets == sorted(offsets)


def test_unknown_ids_are_404(api_client) -> None:
    assert api_client.get("/v1/situations/does-not-exist").status_code == 404
    assert api_client.get("/v1/situations/does-not-exist/sequence").status_code == 404


def test_repeated_query_is_served_from_cache(api_client) -> None:
    first = api_client.post("/v1/search", json=SKETCH).json()
    second = api_client.post("/v1/search", json=SKETCH).json()
    assert first == second
    metrics = api_client.get("/metrics").text
    assert "pitchscanner_cache_hits_total" in metrics


def test_metrics_exposes_stage_timings(api_client) -> None:
    api_client.post("/v1/search", json=SKETCH)
    metrics = api_client.get("/metrics").text
    assert "pitchscanner_stage_seconds" in metrics
    assert 'stage="aggregate"' in metrics


def test_request_id_header_is_echoed(api_client) -> None:
    response = api_client.get("/healthz", headers={"x-request-id": "abc123"})
    assert response.headers["x-request-id"] == "abc123"


def test_openapi_documents_every_route(api_client) -> None:
    paths = api_client.get("/openapi.json").json()["paths"]
    assert {
        "/v1/search",
        "/v1/filters",
        "/v1/filters/parse",
        "/v1/situations/{event_id}",
        "/v1/situations/{event_id}/sequence",
        "/healthz",
        "/readyz",
        "/metrics",
    } <= set(paths)


def test_meta_reports_attribution(api_client) -> None:
    body = api_client.get("/v1/meta").json()
    assert "StatsBomb" in body["attribution"]
    assert body["counts"]["frames"] > 0
