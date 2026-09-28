"""Retrieval quality: recall against exact search, plus the jitter sanity check."""

from __future__ import annotations

from dataclasses import dataclass

from pitchscanner.bench.queries import BenchQuery
from pitchscanner.encoders import get_encoder
from pitchscanner.models import Filters
from pitchscanner.search.exact import recall_at_k
from pitchscanner.store.base import Store


@dataclass(slots=True)
class RecallResult:
    encoding: str
    k: int
    filtered: bool
    recall: float
    n_queries: int

    def as_dict(self) -> dict[str, object]:
        return {
            "encoding": self.encoding,
            "k": self.k,
            "filtered": self.filtered,
            "recall": round(self.recall, 4),
            "queries": self.n_queries,
        }


def measure_recall(
    store: Store,
    queries: list[BenchQuery],
    *,
    encoding: str = "E1",
    k: int = 10,
    filters: Filters | None = None,
) -> RecallResult:
    """ANN top-k against the exact top-k of the same query set."""

    encoder = get_encoder(encoding)
    scores: list[float] = []
    for q in queries:
        vector = encoder.encode(q.situation)
        approx = store.ann_search(encoder.version, vector, k, encoder.metric, filters)
        exact = store.exact_search(encoder.version, vector, k, encoder.metric, filters)
        scores.append(recall_at_k(approx, exact, k))
    return RecallResult(
        encoding=encoding,
        k=k,
        filtered=filters is not None and not filters.is_empty(),
        recall=sum(scores) / len(scores) if scores else 1.0,
        n_queries=len(scores),
    )


@dataclass(slots=True)
class JitterResult:
    encoding: str
    top1: float
    top5: float
    n_queries: int

    def as_dict(self) -> dict[str, object]:
        return {
            "encoding": self.encoding,
            "original_in_top1": round(self.top1, 4),
            "original_in_top5": round(self.top5, 4),
            "queries": self.n_queries,
        }


def measure_jitter_robustness(
    store: Store, queries: list[BenchQuery], *, encoding: str = "E1"
) -> JitterResult:
    """Perturb a real frame by 1-2m; it should still come back at the top.

    This is the automatic sanity check that catches an encoding which is merely
    memorising, without needing any human labels.
    """

    encoder = get_encoder(encoding)
    top1 = 0
    top5 = 0
    for q in queries:
        if q.source_event_id is None:
            continue
        vector = encoder.encode(q.situation)
        hits = store.exact_search(encoder.version, vector, 5, encoder.metric, None)
        ids = [h.event_id for h in hits]
        if ids and ids[0] == q.source_event_id:
            top1 += 1
        if q.source_event_id in ids:
            top5 += 1
    n = sum(1 for q in queries if q.source_event_id is not None)
    return JitterResult(
        encoding=encoding,
        top1=top1 / n if n else 0.0,
        top5=top5 / n if n else 0.0,
        n_queries=n,
    )
