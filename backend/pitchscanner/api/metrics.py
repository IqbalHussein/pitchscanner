"""Prometheus metrics: request latency, per-stage timings, cache hits, errors."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUEST_LATENCY = Histogram(
    "pitchscanner_request_seconds",
    "End-to-end request latency",
    labelnames=("method", "path", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5),
)
STAGE_LATENCY = Histogram(
    "pitchscanner_stage_seconds",
    "Per-stage search latency",
    labelnames=("stage",),
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25),
)
CACHE_HITS = Counter("pitchscanner_cache_hits_total", "Search cache hits")
CACHE_MISSES = Counter("pitchscanner_cache_misses_total", "Search cache misses")
ERRORS = Counter("pitchscanner_errors_total", "Handled errors", labelnames=("kind",))
RATE_LIMITED = Counter("pitchscanner_rate_limited_total", "Requests rejected by rate limit")
EXACT_FALLBACKS = Counter(
    "pitchscanner_exact_fallback_total", "Searches served by an exact scan of a filtered subset"
)
INDEX_VECTORS = Gauge(
    "pitchscanner_index_vectors", "Vectors in the index", labelnames=("encoding_version",)
)


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
