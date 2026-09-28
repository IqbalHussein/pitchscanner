"""The FAISS comparison harness.

The comparison is only worth quoting if the harness is honest about three things:
it degrades when the optional dependency is missing, it indexes every vector, and
its recall is measured against the same exact search everything else is.
"""

from __future__ import annotations

import pytest

from pitchscanner.bench.faiss_compare import available, compare
from pitchscanner.bench.queries import build_query_set
from pitchscanner.store.memory import MemoryStore

needs_faiss = pytest.mark.skipif(not available(), reason='needs backend[faiss]')


def test_available_answers_without_raising() -> None:
    assert isinstance(available(), bool)


def test_no_vectors_means_no_comparison() -> None:
    """An empty store yields None rather than a table of zeroes."""

    assert compare(MemoryStore(), [], encoding="E1") is None


@needs_faiss
@pytest.mark.parametrize("encoding", ["E0", "E1", "E2"])
def test_faiss_agrees_with_exact_search(fixture_store: MemoryStore, encoding: str) -> None:
    queries = build_query_set(fixture_store, 15)
    result = compare(fixture_store, queries, encoding=encoding)
    assert result is not None
    assert result.vectors == len(fixture_store.frames())
    assert result.queries == len(queries)
    # At fixture size HNSW is effectively exhaustive; anything below the spec's
    # own 0.95 target would mean the harness is comparing the wrong things.
    assert result.recall_at_10 >= 0.95
    assert result.build_ms > 0.0
    assert result.search_p95_ms >= result.search_p50_ms


@needs_faiss
def test_cosine_encodings_are_compared_by_inner_product(
    fixture_store: MemoryStore,
) -> None:
    """Cosine distance and inner product over normalised vectors rank the same.

    If this ever flipped to plain L2 the recall numbers would quietly drop, so the
    metric the harness reports is pinned.
    """

    queries = build_query_set(fixture_store, 5)
    assert compare(fixture_store, queries, encoding="E1").metric == "inner_product"  # type: ignore[union-attr]
    assert compare(fixture_store, queries, encoding="E2").metric == "l2"  # type: ignore[union-attr]
