"""FastAPI application factory."""

from __future__ import annotations

import math
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pitchscanner.api import metrics
from pitchscanner.api.deps import AppState
from pitchscanner.api.routes import router
from pitchscanner.config import Settings, get_settings
from pitchscanner.logging_setup import configure_logging, get_logger, request_id_var
from pitchscanner.store.arrays import DetailUnavailableError
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
        for version, n in state.store.vector_counts().items():
            metrics.INDEX_VECTORS.labels(encoding_version=version).set(n)
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
        total = f"total;dur={elapsed * 1000:.1f}"
        existing = response.headers.get("server-timing")
        response.headers["server-timing"] = f"{existing}, {total}" if existing else total
        log.info(
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(elapsed * 1000, 2),
        )
        request_id_var.reset(token)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default handler echoes the offending input back, and a NaN or
        # an infinity in it makes the JSON encoder raise: a 422 became a 500.
        return JSONResponse(
            status_code=422, content={"detail": _json_safe(jsonable_encoder(exc.errors()))}
        )

    @app.exception_handler(DetailUnavailableError)
    async def detail_unavailable(request: Request, exc: DetailUnavailableError) -> JSONResponse:
        # The searchable arrays are fine; only the per-match detail file is not.
        metrics.ERRORS.labels(kind="detail").inc()
        log.error("detail_unavailable", error=str(exc))
        return JSONResponse(status_code=503, content={"detail": "match detail unavailable"})

    app.include_router(router)
    return app


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


app = create_app()
