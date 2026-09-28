"""Systems benchmarks: latency percentiles and the per-stage breakdown."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

from pitchscanner.bench.queries import BenchQuery
from pitchscanner.config import get_settings
from pitchscanner.search import SearchService
from pitchscanner.store.base import Store

WARMUP = 20


@dataclass(slots=True)
class LatencyResult:
    encoding: str
    n: int
    p50: float
    p95: float
    p99: float
    mean: float
    stages: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "encoding": self.encoding,
            "queries": self.n,
            "p50_ms": self.p50,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
            "mean_ms": self.mean,
            "stage_p95_ms": self.stages,
        }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return round(ordered[idx], 3)


def measure_latency(
    store: Store, queries: list[BenchQuery], *, encoding: str = "E1", k: int = 25
) -> LatencyResult:
    """Warm up first, then measure. A cold first call is not the number to report."""

    service = SearchService(store, get_settings())
    for q in queries[:WARMUP]:
        service.search(q.to_request(k=k, encoding=encoding))

    totals: list[float] = []
    stages: dict[str, list[float]] = {"encode": [], "search": [], "dedup": [], "aggregate": []}
    for q in queries:
        request = q.to_request(k=k, encoding=encoding)
        started = time.perf_counter()
        outcome = service.search(request)
        totals.append((time.perf_counter() - started) * 1000.0)
        stages["encode"].append(outcome.timing.encode)
        stages["search"].append(outcome.timing.search)
        stages["dedup"].append(outcome.timing.dedup)
        stages["aggregate"].append(outcome.timing.aggregate)

    return LatencyResult(
        encoding=encoding,
        n=len(totals),
        p50=percentile(totals, 0.50),
        p95=percentile(totals, 0.95),
        p99=percentile(totals, 0.99),
        mean=round(statistics.fmean(totals), 3) if totals else 0.0,
        stages={name: percentile(vals, 0.95) for name, vals in stages.items()},
    )
