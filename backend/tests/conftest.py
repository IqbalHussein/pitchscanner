"""Shared fixtures. The whole suite runs offline against the checked-in fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from hypothesis import HealthCheck
from hypothesis import settings as hypothesis_settings

from pitchscanner.config import Settings
from pitchscanner.ingest import ingest
from pitchscanner.store.memory import MemoryStore

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "data" / "fixtures" / "statsbomb"

# Two Hypothesis budgets. Pull requests get the cheap one; the nightly job gets
# one large enough to actually explore the space - today's mirroring bug in E2
# took a few hundred examples to surface, which is more than a PR should wait for.
# Select with HYPOTHESIS_PROFILE=nightly (the Makefile's `test-slow` does).
_COMMON = {"deadline": None, "suppress_health_check": [HealthCheck.too_slow]}
hypothesis_settings.register_profile("default", max_examples=60, **_COMMON)
hypothesis_settings.register_profile("nightly", max_examples=1000, **_COMMON)
hypothesis_settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(
        store="memory",
        fixture_dir=FIXTURE_DIR,
        bootstrap_resamples=200,  # the suite does not need 1,000 resamples
        mirror_augmentation=True,
    )


@pytest.fixture(scope="session")
def fixture_store(settings: Settings) -> MemoryStore:
    """Ingest two fixture matches once per session; every test reads this store."""

    store = MemoryStore()
    ingest(FIXTURE_DIR, store, settings=settings, limit_matches=2)
    return store


@pytest.fixture(scope="session")
def api_client(fixture_store: MemoryStore, settings: Settings):
    from fastapi.testclient import TestClient

    from pitchscanner.api import create_app

    with TestClient(create_app(settings=settings, store=fixture_store)) as client:
        yield client


@pytest.fixture(scope="session")
def dsn() -> str | None:
    """A Postgres DSN for the integration tier, or None to skip it."""

    return os.environ.get("PITCHSCANNER_TEST_DSN")
