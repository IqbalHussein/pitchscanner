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

    def allow(self, client: str) -> bool:
        now = monotonic()
        with self._lock:
            window = [t for t in self._hits.get(client, []) if now - t < 60.0]
            if len(window) >= self.per_minute:
                self._hits[client] = window
                return False
            window.append(now)
            self._hits[client] = window
            return True


@dataclass
class AppState:
    settings: Settings
    store: Store
    search: SearchService
    filters: FilterService
    cache: QueryCache
    limiter: RateLimiter
    ready: bool = False

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
