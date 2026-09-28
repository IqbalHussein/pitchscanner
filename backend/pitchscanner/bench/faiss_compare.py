"""FAISS as a benchmarked alternative to pgvector.

The architecture picks pgvector, and this file is the evidence for that choice
rather than a claim about it. FAISS is in-process: no network hop, no SQL, no
filter columns - so it should win on raw search latency, and the question is by
how much, and what that buys compared with what it costs.

What it costs is the whole reason pgvector stays: FAISS indexes live in memory,
know nothing about the filter columns the API queries on, and would need a second
process to own them plus a way to keep them in step with the database. An extra
millisecond inside a 150ms budget is a bad trade for that. Optional dependency
(``pip install -e "backend[faiss]"``); everything degrades to "not installed".
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from pitchscanner.bench.queries import BenchQuery
from pitchscanner.encoders import get_encoder
from pitchscanner.store.base import Store


def available() -> bool:
    try:
        import faiss  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(slots=True)
class FaissResult:
    encoding: str
    vectors: int
    dim: int
    metric: str
    build_ms: float
    search_p50_ms: float
    search_p95_ms: float
    recall_at_10: float
    queries: int

    def as_dict(self) -> dict[str, object]:
        return {
            "encoding": self.encoding,
            "vectors": self.vectors,
            "dim": self.dim,
            "metric": self.metric,
            "build_ms": round(self.build_ms, 1),
            "search_p50_ms": round(self.search_p50_ms, 4),
            "search_p95_ms": round(self.search_p95_ms, 4),
            "recall@10": round(self.recall_at_10, 4),
            "queries": self.queries,
        }


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def compare(
    store: Store,
    queries: list[BenchQuery],
    *,
    encoding: str = "E1",
    k: int = 10,
    hnsw_m: int = 16,
    ef_construction: int = 200,
    ef_search: int = 80,
) -> FaissResult | None:
    """Build a FAISS HNSW index over the same vectors and measure it.

    Same graph parameters as the pgvector index, so the comparison is between
    implementations rather than between configurations. Cosine encodings are
    compared by inner product over normalised vectors, which is the same ordering
    cosine distance gives.
    """

    try:
        import faiss
    except ImportError:
        return None

    encoder = get_encoder(encoding)
    frames = store.frames()
    if not frames:
        return None

    matrix = np.stack([encoder.encode(f.to_situation()) for f in frames]).astype(np.float32)
    event_ids = [f.event_id for f in frames]
    cosine = encoder.metric == "cosine"
    if cosine:
        matrix = matrix.copy()
        faiss.normalize_L2(matrix)

    t0 = time.perf_counter()
    space = faiss.METRIC_INNER_PRODUCT if cosine else faiss.METRIC_L2
    index = faiss.IndexHNSWFlat(matrix.shape[1], hnsw_m, space)
    index.hnsw.efConstruction = ef_construction
    index.add(matrix)
    index.hnsw.efSearch = ef_search
    build_ms = (time.perf_counter() - t0) * 1000

    timings: list[float] = []
    recalls: list[float] = []
    for q in queries:
        vector = encoder.encode(q.situation).astype(np.float32).reshape(1, -1)
        if cosine:
            faiss.normalize_L2(vector)
        t1 = time.perf_counter()
        _, idx = index.search(vector, k)
        timings.append((time.perf_counter() - t1) * 1000)

        approx = {event_ids[int(i)] for i in idx[0] if i >= 0}
        exact = {
            c.event_id
            for c in store.exact_search(
                encoder.version, encoder.encode(q.situation), k, encoder.metric, None
            )
        }
        if exact:
            recalls.append(len(approx & exact) / len(exact))

    return FaissResult(
        encoding=encoding,
        vectors=len(event_ids),
        dim=int(matrix.shape[1]),
        metric="inner_product" if cosine else "l2",
        build_ms=build_ms,
        search_p50_ms=_percentile(timings, 50),
        search_p95_ms=_percentile(timings, 95),
        recall_at_10=sum(recalls) / len(recalls) if recalls else 0.0,
        queries=len(timings),
    )
