"""FastAPI application factory."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pitchscanner.api import metrics
from pitchscanner.api.deps import AppState
from pitchscanner.api.routes import router
from pitchscanner.config import Settings, get_settings
from pitchscanner.logging_setup import configure_logging, get_logger, request_id_var
from pitchscanner.store.base import Store

log = get_logger(__name__)

DESCRIPTION = """
Query-by-sketch search over StatsBomb positional data.

Draw a situation, get the most similar real moments from professional matches and
what those moments led to, against a zone baseline.

Data: [StatsBomb Open Data](https://github.com/statsbomb/open-data), free for
non-commercial use with attribution.
"""


def create_app(settings: Settings | None = None, store: Store | None = None) -> FastAPI:
    cfg = settings or get_settings()
    configure_logging()

    # Built here rather than in the lifespan hook, so the app is usable the
    # moment it is constructed (tests, scripts) and not only inside an ASGI
    # lifespan context.
    app_state = AppState.build(cfg, store)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state: AppState = app.state.app_state
        counts = state.store.counts()
        entities = state.filters.warm()
        for version, _ in getattr(state.store, "_vectors", {}).items():
            metrics.INDEX_VECTORS.labels(encoding_version=version).set(counts.frames)
        state.ready = True
        log.info("startup", store=cfg.store, entities=entities, **counts.as_dict())
        yield
        state.store.close()

    app = FastAPI(
        title="Match Situation Search Engine",
        description=DESCRIPTION,
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.app_state = app_state
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origin_list,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def observe(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(request_id)
        started = time.perf_counter()

        state: AppState | None = getattr(request.app.state, "app_state", None)
        if state is not None and request.url.path.startswith("/v1/"):
            client = request.client.host if request.client else "unknown"
            if not state.limiter.allow(client):
                metrics.RATE_LIMITED.inc()
                request_id_var.reset(token)
                return JSONResponse(
                    {"detail": "rate limit exceeded"},
                    status_code=429,
                    headers={"retry-after": "60", "x-request-id": request_id},
                )
        try:
            response = await call_next(request)
        except Exception:
            metrics.ERRORS.labels(kind="unhandled").inc()
            log.exception("request_failed", path=request.url.path)
            request_id_var.reset(token)
            raise
        elapsed = time.perf_counter() - started
        metrics.REQUEST_LATENCY.labels(
            method=request.method,
            path=(
                request.scope["route"].path
                if request.scope.get("route")
                else request.url.path
            ),
            status=str(response.status_code),
        ).observe(elapsed)
        response.headers["x-request-id"] = request_id
        response.headers["server-timing"] = f"total;dur={elapsed * 1000:.1f}"
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(elapsed * 1000, 2),
        )
        request_id_var.reset(token)
        return response

    app.include_router(router)
    return app


app = create_app()
