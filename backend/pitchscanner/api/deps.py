"""Application state: one store, one search service, one filter service.

Built at startup, because the catalogue and the baselines are read-mostly and
rebuilding them per request would dominate the latency budget.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic

from fastapi import Request

from pitchscanner.config import Settings, get_settings
from pitchscanner.encoders import available as available_encodings
from pitchscanner.encoders import get_encoder
from pitchscanner.filters import FilterService
from pitchscanner.search import SearchService
from pitchscanner.store import make_store
from pitchscanner.store.base import Store


@dataclass
class QueryCache:
    """Tiny LRU keyed on a hash of the request body."""

    capacity: int = 512
    _data: OrderedDict[str, object] = field(default_factory=OrderedDict)
    _lock: Lock = field(default_factory=Lock)

    def get(self, key: str) -> object | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: str, value: object) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


@dataclass
class RateLimiter:
    """Fixed-window limiter per client. Enough for a public demo."""

    per_minute: int = 120
    _hits: dict[str, list[float]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)
    _last_sweep: float = 0.0

    def allow(self, client: str) -> bool:
        now = monotonic()
        with self._lock:
            if now - self._last_sweep >= 60.0:
                self._sweep(now)
            window = [t for t in self._hits.get(client, []) if now - t < 60.0]
            if len(window) >= self.per_minute:
                self._hits[client] = window
                return False
            window.append(now)
            self._hits[client] = window
            return True

    def _sweep(self, now: float) -> None:
        """Drop clients with nothing left in their window, so unique IPs cannot grow
        the dict without bound. Runs at most once a minute, under the lock."""

        self._last_sweep = now
        for client in [c for c, ts in self._hits.items() if not ts or now - ts[-1] >= 60.0]:
            del self._hits[client]


@dataclass
class AppState:
    settings: Settings
    store: Store
    search: SearchService
    filters: FilterService
    cache: QueryCache
    limiter: RateLimiter
    ready: bool = False
    _served: tuple[float, tuple[str, ...]] | None = field(default=None, repr=False)

    #: How long the list of served encodings is reused. It only changes when an
    #: ingest runs, and on Postgres it costs a count over every vector table.
    SERVED_TTL_S = 30.0

    def served_encodings(self) -> tuple[str, ...]:
        """Encodings this database can actually answer queries for.

        An encoding is served when its *current* version (``E1-v1``) has vectors
        in the store. The registry (`available_encodings`) lists what the code
        can compute, which on a partially loaded database (the free-tier
        deployment holds E1 only) includes encodings whose table is empty or
        does not exist.
        """

        now = monotonic()
        if self._served is not None and now - self._served[0] < self.SERVED_TTL_S:
            return self._served[1]
        counts = self.store.vector_counts()
        served = tuple(
            e for e in available_encodings() if counts.get(get_encoder(e).version, 0) > 0
        )
        self._served = (now, served)
        return served

    @classmethod
    def build(cls, settings: Settings | None = None, store: Store | None = None) -> AppState:
        cfg = settings or get_settings()
        st = store or make_store(cfg)
        return cls(
            settings=cfg,
            store=st,
            search=SearchService(st, cfg),
            filters=FilterService(st, cfg),
            cache=QueryCache(capacity=cfg.cache_size),
            limiter=RateLimiter(per_minute=cfg.rate_limit_per_minute),
        )


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.app_state
    return state
