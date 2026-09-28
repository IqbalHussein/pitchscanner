"""Exact brute-force search, kept in the codebase as the reference implementation.

Recall is defined against this, so it has to stay simple enough to be obviously
correct: no index, no tuning parameters, one distance computation per vector.
"""

from __future__ import annotations

import numpy as np

from pitchscanner.models import Filters
from pitchscanner.store.base import Candidate, Store


def exact_top_k(
    store: Store,
    encoding_version: str,
    vector: np.ndarray,
    k: int,
    metric: str,
    filters: Filters | None = None,
) -> list[Candidate]:
    return store.exact_search(encoding_version, vector, k, metric, filters)


def recall_at_k(approx: list[Candidate], exact: list[Candidate], k: int) -> float:
    """Fraction of the exact top-k that the approximate run also returned."""

    truth = {c.event_id for c in exact[:k]}
    if not truth:
        return 1.0
    got = {c.event_id for c in approx[:k]}
    return len(truth & got) / len(truth)
