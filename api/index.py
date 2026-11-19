"""Vercel entrypoint: the PitchScanner API, on Postgres or in memory.

Only Vercel loads this file. Two modes, chosen by one environment variable:

* `PITCHSCANNER_DSN` set (the Supabase transaction-pooler URI, port 6543): serve
  from Postgres, E1 only. Prepared statements are switched off because a
  transaction-mode pooler cannot route them. Nothing connects at import time;
  the store opens its connection on the first request.
* Not set: serve the checked-in fixture dataset from a pickle snapshot bundled
  with the function (see vercel.json), so no database is needed. If the
  snapshot is missing at cold start (for example a Git-based deploy, where the
  gitignored `.data/` is absent), it is rebuilt from `data/fixtures` into /tmp,
  which takes a few seconds.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

if os.environ.get("PITCHSCANNER_DSN"):
    os.environ["PITCHSCANNER_STORE"] = "postgres"
    # The free-plan database holds E1 only (see the README), and the API lists
    # only encodings that have vectors.
    os.environ["PITCHSCANNER_ENCODING"] = "E1"
    # Transaction-mode pooling (port 6543) does not support prepared statements.
    os.environ["PITCHSCANNER_DB_PREPARE"] = "false"
else:
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
