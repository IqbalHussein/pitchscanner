"""RateLimiter: limits still apply, and idle clients are forgotten."""

from __future__ import annotations

import pytest

from pitchscanner.api import deps
from pitchscanner.api.deps import RateLimiter


def test_limit_applies_per_client() -> None:
    limiter = RateLimiter(per_minute=2)
    assert limiter.allow("a") and limiter.allow("a")
    assert not limiter.allow("a")
    assert limiter.allow("b")


def test_idle_clients_are_evicted(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(deps, "monotonic", lambda: clock[0])
    limiter = RateLimiter(per_minute=5)
    for i in range(100):
        limiter.allow(f"10.0.0.{i}")
    assert len(limiter._hits) == 100
    clock[0] += 61.0
    assert limiter.allow("fresh")
    assert set(limiter._hits) == {"fresh"}


def test_limit_still_applies_after_a_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(deps, "monotonic", lambda: clock[0])
    limiter = RateLimiter(per_minute=2)
    limiter.allow("a")
    clock[0] += 61.0
    assert limiter.allow("a") and limiter.allow("a")
    assert not limiter.allow("a")
