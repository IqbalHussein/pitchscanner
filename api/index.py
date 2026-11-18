"""Vercel entrypoint: the PitchScanner API on the in-memory store.

Only Vercel loads this file. It serves the checked-in fixture dataset from a
pickle snapshot bundled with the function (see vercel.json), so no database is
needed. If the snapshot is missing at cold start (for example a Git-based
deploy, where the gitignored `.data/` is absent), it is rebuilt from
`data/fixtures` into /tmp, which takes a few seconds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

os.environ["PITCHSCANNER_STORE"] = "memory"

# The memory store reads `<PITCHSCANNER_DATA_CACHE>/../memory-store.pkl`.
_BUNDLED = ROOT / ".data"
_TMP = Path("/tmp/pitchscanner")
if (_BUNDLED / "memory-store.pkl").exists():
    os.environ["PITCHSCANNER_DATA_CACHE"] = str(_BUNDLED / "open-data")
else:
    os.environ["PITCHSCANNER_DATA_CACHE"] = str(_TMP / "open-data")
    if not (_TMP / "memory-store.pkl").exists():
        from pitchscanner.config import get_settings
        from pitchscanner.ingest import ingest
        from pitchscanner.store.memory import MemoryStore

        get_settings.cache_clear()
        _cfg = get_settings()
        _store = MemoryStore()
        ingest(ROOT / "data" / "fixtures" / "statsbomb", _store, settings=_cfg)
        _TMP.mkdir(parents=True, exist_ok=True)
        _store.save(_TMP / "memory-store.pkl")

from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware  # noqa: E402

from pitchscanner.api.app import app as _api  # noqa: E402

# Vercel's edge sets x-forwarded-for to the real client address; trust it so the
# per-client rate limiter does not see every visitor as one address.
app = ProxyHeadersMiddleware(_api, trusted_hosts="*")
