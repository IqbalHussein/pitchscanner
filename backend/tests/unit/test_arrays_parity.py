"""The array store must answer exactly as the in-memory store does.

Both are built from the same fixture ingest. Search results (ids, order, distances to
1e-5), counts, coverage, team options, catalogue, baselines and the API's JSON must agree.
Vectors are stored as float32 here, because float16 (the shipped default for E2) cannot
meet a 1e-5 distance tolerance by construction; that case has its own test.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pitchscanner.api import create_app
from pitchscanner.config import Settings
from pitchscanner.encoders import get_encoder
from pitchscanner.ingest import ingest
from pitchscanner.models import Filters, PointIn, SearchRequest
from pitchscanner.search import SearchService
from pitchscanner.store.arrays import ArrayStore
from pitchscanner.store.filtering import matches as filter_matches
from pitchscanner.store.filtering import to_mask
from pitchscanner.store.memory import MemoryStore
from tests.conftest import FIXTURE_DIR
from tests.golden.test_golden import PINNED_SITUATIONS

F32 = {"E0": "float32", "E1": "float32", "E2": "float32"}
ENCODINGS = ["E0", "E1", "E2"]
TOL = 1e-5


@pytest.fixture(scope="module")
def mem(settings: Settings) -> MemoryStore:
    store = MemoryStore()
    ingest(FIXTURE_DIR, store, settings=settings)
    return store


@pytest.fixture(scope="module")
def arr_dir(settings: Settings, tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("arrays")
    store = ArrayStore.create(d, dtypes=F32)
    ingest(FIXTURE_DIR, store, settings=settings)
    store.finalize()
    return d


@pytest.fixture(scope="module")
def arr(arr_dir: Path) -> Iterator[ArrayStore]:
    store = ArrayStore.open(arr_dir)
    yield store
    store.close()


@pytest.fixture(scope="module")
def services(mem: MemoryStore, arr: ArrayStore, settings: Settings):
    return SearchService(mem, settings), SearchService(arr, settings)


def _req(situation, filters: Filters | None = None, k: int = 15) -> SearchRequest:
    return SearchRequest(
        ball=PointIn(x=situation.ball.x, y=situation.ball.y),
        attackers=[PointIn(x=p.x, y=p.y, keeper=p.keeper) for p in situation.players if p.teammate],
        defenders=[
            PointIn(x=p.x, y=p.y, keeper=p.keeper) for p in situation.players if not p.teammate
        ],
        k=k,
        filters=filters or Filters(),
    )


def _random_request(rng: random.Random, filters: Filters | None = None) -> SearchRequest:
    def pt() -> PointIn:
        return PointIn(x=rng.uniform(50, 118), y=rng.uniform(2, 78))

    return SearchRequest(
        ball=pt(),
        attackers=[pt() for _ in range(rng.randint(1, 6))],
        defenders=[pt() for _ in range(rng.randint(0, 7))] or [pt()],
        k=rng.choice([5, 15, 40]),
        filters=filters or Filters(),
    )


def _filter_variants(mem: MemoryStore) -> dict[str, Filters]:
    metas = list(mem._metas.values())
    m0 = metas[0]
    comp = sorted({m.competition_id for m in metas})
    seasons = sorted({m.season_id for m in metas})
    teams = sorted({m.possession_team_id for m in metas})
    players = [p for p in sorted({m.player_id for m in metas if m.player_id}) if p][:3]
    return {
        "none": Filters(),
        "competition": Filters(competition_ids=[comp[0]]),
        "competitions": Filters(competition_ids=comp),
        "season": Filters(season_ids=[seasons[0]]),
        "team_any": Filters(team_ids=[teams[0]]),
        "team_attacking": Filters(team_ids=teams[:2], team_role="attacking"),
        "team_defending": Filters(team_ids=[teams[1]], team_role="defending"),
        "player": Filters(player_ids=players),
        "player_unknown": Filters(player_ids=[-1, 10**9]),
        "game_state": Filters(game_state="losing"),
        "minute": Filters(minute_range=(60, 90)),
        "minute_out_of_range": Filters(minute_range=(5000, 9000)),
        "stages": Filters(stages=["group", "final"]),
        "stage_absent": Filters(stages=["final"]),
        "zones": Filters(ball_zone_ids=[m0.zone_id, 16]),
        "home": Filters(venue="home"),
        "away": Filters(venue="away"),
        "combined": Filters(
            competition_ids=comp[:1], game_state="winning", minute_range=(10, 80), venue="home"
        ),
        "strict": Filters(
            team_ids=[m0.possession_team_id],
            team_role="attacking",
            game_state=m0.game_state,
            minute_range=(m0.minute, m0.minute),
            ball_zone_ids=[m0.zone_id],
        ),
    }


def _assert_same(a, b) -> None:
    assert [r.event_id for r in a.results] == [r.event_id for r in b.results]
    for x, y in zip(a.results, b.results, strict=True):
        assert abs(x.distance - y.distance) <= TOL
        assert (x.match_id, x.minute, x.period, x.possession_id, x.team, x.mirrored) == (
            y.match_id, y.minute, y.period, y.possession_id, y.team, y.mirrored,
        )
        assert x.outcome == y.outcome
    assert a.candidates_examined == b.candidates_examined
    assert a.relaxed == b.relaxed
    assert (a.stats is None) == (b.stats is None)
    if a.stats is not None and b.stats is not None:
        assert a.stats.n == b.stats.n and a.stats.n_possessions == b.stats.n_possessions
        assert a.stats.shot_rate.value == pytest.approx(b.stats.shot_rate.value, abs=1e-4)


# --- bookkeeping ------------------------------------------------------------
def test_counts_and_aggregates_match(mem: MemoryStore, arr: ArrayStore) -> None:
    assert arr.counts() == mem.counts()
    assert arr.vector_counts() == mem.vector_counts()
    assert arr.coverage() == mem.coverage()
    assert arr.team_options() == mem.team_options()
    assert arr.catalogue() == mem.catalogue()
    for version in mem.vector_counts():
        assert arr.baselines(version) == mem.baselines(version)
    assert arr.effective_index_kind == "exact"
    assert arr.matches_by_id([3900001, 3900004, 1]) == mem.matches_by_id([3900001, 3900004, 1])


def test_every_meta_outcome_and_flag_round_trips(mem: MemoryStore, arr: ArrayStore) -> None:
    ids = list(mem._metas)
    assert arr.metas(ids) == mem._metas
    assert arr.outcomes(ids) == mem._outcomes
    assert arr.mirrored_flags(ids) == {e: f.mirrored for e, f in mem._frames.items()}
    assert arr.metas(["nope", "00000000-0000-0000-0000-000000000000", ids[0].upper()]) == {}
    assert arr.meta(ids[7]) == mem.meta(ids[7])
    assert arr.outcome(ids[7]) == mem.outcome(ids[7])


def test_mask_equals_the_scalar_filter_on_every_row(mem: MemoryStore, arr: ArrayStore) -> None:
    ids = list(mem._metas)
    rows = [arr._frame_rows([e])[0][1] for e in ids]
    order = np.argsort(rows)
    ordered = [mem._metas[ids[i]] for i in order]
    rng = random.Random(7)
    variants = list(_filter_variants(mem).values())
    for _ in range(150):
        m = ordered[rng.randrange(len(ordered))]
        variants.append(
            Filters(
                competition_ids=rng.sample([55, 9, 123], rng.randint(0, 2)),
                team_ids=rng.sample(
                    [m.possession_team_id, m.opponent_team_id, 5], rng.randint(0, 2)
                ),
                team_role=rng.choice(["any", "attacking", "defending"]),
                player_ids=[m.player_id] if m.player_id and rng.random() < 0.4 else [],
                game_state=rng.choice(["any", "winning", "drawing", "losing"]),
                minute_range=rng.choice([None, (0, 45), (46, 120), (m.minute, m.minute)]),
                stages=rng.sample(["group", "knockout", "final", "regular"], rng.randint(0, 2)),
                ball_zone_ids=rng.sample(range(18), rng.randint(0, 3)),
                venue=rng.choice(["any", "home", "away"]),
            )
        )
    for f in variants:
        mask = to_mask(arr._cols, arr._vocab, f)
        expected = np.array([filter_matches(m, f) for m in ordered])
        got = np.ones(len(ordered), dtype=bool) if mask is None else np.asarray(mask)
        assert np.array_equal(got, expected), f
        assert arr.count_matching("E1-v1", f) == int(expected.sum())
        assert mem.count_matching("E1-v1", f) == int(expected.sum())


# --- search -----------------------------------------------------------------
@pytest.mark.parametrize("encoding", ENCODINGS)
def test_random_sketches_without_filters(services, encoding: str) -> None:
    m, a = services
    rng = random.Random(11 + ENCODINGS.index(encoding))
    for _ in range(12):
        req = _random_request(rng)
        req.options.encoding = encoding
        _assert_same(m.search(req), a.search(req))


@pytest.mark.parametrize("encoding", ["E1", "E2"])
def test_every_filter_type(services, mem: MemoryStore, encoding: str) -> None:
    m, a = services
    rng = random.Random(5)
    nonempty = 0
    for name, f in _filter_variants(mem).items():
        for _ in range(3):
            req = _random_request(rng, f)
            req.options.encoding = encoding
            try:
                got = a.search(req)
                nonempty += bool(got.results)
                _assert_same(m.search(req), got)
            except AssertionError as exc:  # name the filter that broke
                raise AssertionError(f"filter {name}: {exc}") from exc
    assert nonempty >= 40  # the comparison is not vacuous: most variants return results


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_golden_situations_with_and_without_filters(services, mem: MemoryStore, encoding) -> None:
    m, a = services
    for situation in PINNED_SITUATIONS.values():
        for f in _filter_variants(mem).values():
            req = _req(situation, f)
            req.options.encoding = encoding
            _assert_same(m.search(req), a.search(req))
    # A frame searched for in its own vector comes back first at distance ~0.
    first = next(iter(mem._frames))
    enc = get_encoder(encoding)
    vec = mem._vectors[enc.version][1][mem._vectors[enc.version][0].index(first)]
    top = a.store.exact_search(enc.version, vec, 1, enc.metric)[0]
    assert abs(top.distance) < 1e-5


def test_raw_exact_search_matches_and_handles_ties(mem: MemoryStore, arr: ArrayStore) -> None:
    _ids, mat = mem._vectors["E2-v1"]
    q = mat[3]
    for limit in (1, 7, 500, 10_000):
        want = mem.exact_search("E2-v1", q, limit, "l2")
        got = arr.exact_search("E2-v1", q, limit, "l2")
        assert [c.event_id for c in got] == [c.event_id for c in want]
        assert max(abs(x.distance - y.distance) for x, y in zip(got, want, strict=True)) < TOL
    assert arr.exact_search("E1-v1", np.zeros(72), 3, "cosine")  # a zero query still ranks
    assert arr.exact_search("E9-v1", q, 3, "l2") == []
    assert arr.count_matching("E9-v1", None) == 0
    with pytest.raises(ValueError):
        arr.exact_search("E2-v1", np.zeros(3), 3, "l2")


def test_float16_e2_is_close_but_not_exact(
    mem: MemoryStore, settings: Settings, tmp_path: Path
) -> None:
    store = ArrayStore.create(tmp_path)  # default dtypes: E2 float16, E1 float32
    ingest(FIXTURE_DIR, store, settings=settings, encodings=["E1", "E2"])
    store.finalize()
    half = ArrayStore.open(tmp_path)
    assert half._vectors["E2-v1"].array.dtype == np.float16
    assert half._vectors["E1-v1"].array.dtype == np.float32
    q = mem._vectors["E2-v1"][1][10]
    want = mem.exact_search("E2-v1", q, 25, "l2")
    got = half.exact_search("E2-v1", q, 25, "l2")
    assert len({c.event_id for c in want} & {c.event_id for c in got}) >= 20
    assert max(abs(x.distance - y.distance) for x, y in zip(got, want, strict=True)) < 5e-2


def test_unnormalised_vectors_are_handled(arr: ArrayStore, mem: MemoryStore) -> None:
    vf = arr._vectors["E1-v1"]
    original = vf.unit_norm
    raw = np.asarray(mem._vectors["E1-v1"][1])
    ids = mem._vectors["E1-v1"][0]
    vf.unit_norm = False
    try:
        # Rows are unit already, so both code paths must agree.
        a = arr.exact_search("E1-v1", raw[5], 10, "cosine")
    finally:
        vf.unit_norm = original
    b = arr.exact_search("E1-v1", raw[5], 10, "cosine")
    assert [c.event_id for c in a] == [c.event_id for c in b]
    assert ids[5] == b[0].event_id


# --- detail ------------------------------------------------------------------
def test_frames_events_and_sequences_match(mem: MemoryStore, arr: ArrayStore) -> None:
    sample = random.Random(3).sample(list(mem._frames), 40)
    assert arr.frames_by_id(sample) == {e: mem._frames[e] for e in sample}
    assert arr.frame(sample[0]) == mem.frame(sample[0])
    assert arr.frame("missing") is None
    for eid in sample[:15]:
        assert arr.event(eid) == mem.event(eid)
        assert arr.possession_events(eid) == mem.possession_events(eid)
    # Events that have no frame are still reachable, as in the other stores.
    bare = next(e for e in mem._events if e not in mem._frames)
    assert arr.event(bare) == mem.event(bare)
    assert arr.possession_events(bare) == mem.possession_events(bare)
    assert arr.meta(bare) is None and arr.frame(bare) is None
    assert arr.event("zzz") is None and arr.possession_events("zzz") == []
    assert arr.frames(limit=10) == mem.frames(limit=10)
    assert len(arr.frames()) == len(mem.frames())


# --- API ---------------------------------------------------------------------
def _scrub(body):
    """Drop timings, which differ run to run."""

    if isinstance(body, dict):
        return {k: _scrub(v) for k, v in body.items() if k != "timing_ms"}
    if isinstance(body, list):
        return [_scrub(v) for v in body]
    return body


def _approx_equal(a, b, path="") -> None:
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _approx_equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _approx_equal(x, y, f"{path}[{i}]")
    elif isinstance(a, float) and not isinstance(a, bool):
        assert a == pytest.approx(b, abs=2e-5), path
    else:
        assert a == b, path


@pytest.fixture(scope="module")
def clients(mem: MemoryStore, arr: ArrayStore, settings: Settings):
    with (
        TestClient(create_app(settings=settings, store=mem)) as c1,
        TestClient(create_app(settings=settings, store=arr)) as c2,
    ):
        yield c1, c2


SKETCH = {
    "ball": {"x": 84, "y": 22},
    "attackers": [{"x": 86, "y": 20}, {"x": 95, "y": 35}],
    "defenders": [{"x": 92, "y": 25}, {"x": 100, "y": 30}],
    "k": 10,
}


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"filters": {"competition_ids": [55], "game_state": "losing"}},
        {"filter_text": "home team in the last fifteen minutes"},
        {"options": {"encoding": "E2"}},
    ],
)
def test_api_search_equals(clients, extra) -> None:
    c_mem, c_arr = clients
    a = c_mem.post("/v1/search", json={**SKETCH, **extra})
    b = c_arr.post("/v1/search", json={**SKETCH, **extra})
    assert a.status_code == b.status_code == 200
    _approx_equal(_scrub(a.json()), _scrub(b.json()))


def test_api_situation_sequence_and_listing_equal(clients, mem: MemoryStore) -> None:
    c_mem, c_arr = clients
    hits = c_mem.post("/v1/search", json=SKETCH).json()["results"]
    for hit in hits[:5]:
        eid = hit["event_id"]
        for suffix in ("", "/sequence", "/sequence?window_s=5"):
            a = c_mem.get(f"/v1/situations/{eid}{suffix}")
            b = c_arr.get(f"/v1/situations/{eid}{suffix}")
            assert a.status_code == b.status_code == 200
            assert a.json() == b.json()
    for path in ("/v1/filters", "/v1/meta", "/readyz"):
        a, b = c_mem.get(path), c_arr.get(path)
        assert a.status_code == b.status_code
        assert a.json() == b.json()
    assert c_arr.get("/v1/situations/nope").status_code == 404
    assert c_arr.get("/v1/situations/nope/sequence").status_code == 404
