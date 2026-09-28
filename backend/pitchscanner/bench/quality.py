"""Retrieval quality that needs judgement: the blind evaluation set.

The labelling tool builds a *pooled* set - results from every encoding, shuffled,
with the encoding hidden - so the person labelling cannot favour the encoding
they designed. Metrics are precision@10 and nDCG@10 per encoding.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from pitchscanner.bench.queries import build_query_set
from pitchscanner.encoders import available, get_encoder
from pitchscanner.store.base import Store

EVAL_SEED = 777


@dataclass(slots=True)
class PooledQuery:
    """One query plus the union of candidates from all encodings, shuffled."""

    query_id: str
    source_event_id: str | None
    ball: tuple[float, float]
    candidates: list[dict[str, object]] = field(default_factory=list)


def build_eval_pool(
    store: Store, n_queries: int = 200, *, per_encoding: int = 10, seed: int = EVAL_SEED
) -> list[PooledQuery]:
    """Generate the unlabelled evaluation set. Run this once, then label it."""

    rng = random.Random(seed)
    queries = build_query_set(store, n_queries, seed=seed, frame_share=1.0)
    pool: list[PooledQuery] = []
    for i, q in enumerate(queries):
        seen: dict[str, dict[str, object]] = {}
        for encoding in available():
            encoder = get_encoder(encoding)
            vector = encoder.encode(q.situation)
            for rank, hit in enumerate(
                store.exact_search(encoder.version, vector, per_encoding, encoder.metric, None)
            ):
                row = seen.setdefault(
                    hit.event_id,
                    {"event_id": hit.event_id, "from": [], "label": None},
                )
                # `from` is recorded for scoring but must be hidden while labelling.
                assert isinstance(row["from"], list)
                row["from"].append({"encoding": encoding, "rank": rank, "distance": hit.distance})
        candidates = list(seen.values())
        rng.shuffle(candidates)
        pool.append(
            PooledQuery(
                query_id=f"q{i:04d}",
                source_event_id=q.source_event_id,
                ball=(q.situation.ball.x, q.situation.ball.y),
                candidates=candidates,
            )
        )
    return pool


def write_eval_pool(pool: list[PooledQuery], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for pq in pool:
            fh.write(
                json.dumps(
                    {
                        "query_id": pq.query_id,
                        "source_event_id": pq.source_event_id,
                        "ball": list(pq.ball),
                        "candidates": pq.candidates,
                    }
                )
                + "\n"
            )
    return len(pool)


def load_labels(path: Path) -> dict[str, dict[str, int]]:
    """{query_id: {event_id: relevance}} from the hand-labelled file."""

    labels: dict[str, dict[str, int]] = {}
    if not path.exists():
        return labels
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            labels[str(row["query_id"])] = {
                str(k): int(v) for k, v in (row.get("labels") or {}).items()
            }
    return labels


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked: list[int], k: int) -> float:
    ideal = sorted(ranked, reverse=True)[:k]
    denom = dcg(ideal)
    return dcg(ranked[:k]) / denom if denom > 0 else 0.0


@dataclass(slots=True)
class QualityResult:
    encoding: str
    precision_at_10: float
    ndcg_at_10: float
    labelled_queries: int

    def as_dict(self) -> dict[str, object]:
        return {
            "encoding": self.encoding,
            "precision@10": round(self.precision_at_10, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
            "labelled_queries": self.labelled_queries,
        }


def score_encodings(
    store: Store, pool_path: Path, labels_path: Path, *, k: int = 10
) -> list[QualityResult]:
    """Score each encoding against the labels. Unlabelled queries are skipped."""

    labels = load_labels(labels_path)
    if not labels or not pool_path.exists():
        return []

    results: list[QualityResult] = []
    for encoding in available():
        precisions: list[float] = []
        ndcgs: list[float] = []
        with pool_path.open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                qid = str(row["query_id"])
                if qid not in labels:
                    continue
                ranked = [
                    labels[qid].get(str(c["event_id"]), 0)
                    for c in sorted(
                        row["candidates"],
                        key=lambda c: min(
                            (f["rank"] for f in c["from"] if f["encoding"] == encoding),
                            default=10**6,
                        ),
                    )[:k]
                ]
                if not ranked:
                    continue
                precisions.append(sum(1 for r in ranked if r > 0) / len(ranked))
                ndcgs.append(ndcg_at_k(ranked, k))
        results.append(
            QualityResult(
                encoding=encoding,
                precision_at_10=sum(precisions) / len(precisions) if precisions else 0.0,
                ndcg_at_10=sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
                labelled_queries=len(precisions),
            )
        )
    return results
