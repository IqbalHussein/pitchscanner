"""Store factory."""

from __future__ import annotations

from pathlib import Path

from pitchscanner.config import Settings, get_settings
from pitchscanner.store.base import Candidate, CatalogueEntry, Counts, FrameMeta, Store
from pitchscanner.store.memory import MemoryStore
from pitchscanner.store.postgres import PostgresStore

#: Where `pitchscanner ingest --store memory` leaves its snapshot, so the API can
#: serve the fixture dataset with no database at all.
MEMORY_SNAPSHOT = Path(".data") / "memory-store.pkl"


def make_store(settings: Settings | None = None) -> Store:
    cfg = settings or get_settings()
    if cfg.store == "memory":
        path = Path(cfg.data_cache).parent / MEMORY_SNAPSHOT.name
        if path.exists():
            return MemoryStore.load(path)
        return MemoryStore()
    return PostgresStore(cfg.dsn, ef_search=cfg.hnsw_ef_search, probes=cfg.ivfflat_probes)


__all__ = [
    "MEMORY_SNAPSHOT",
    "Candidate",
    "CatalogueEntry",
    "Counts",
    "FrameMeta",
    "MemoryStore",
    "PostgresStore",
    "Store",
    "make_store",
]
