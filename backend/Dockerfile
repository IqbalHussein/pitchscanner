# API image. Multi-stage so the runtime layer carries no build tooling.
FROM python:3.12-slim AS builder

WORKDIR /build
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml ./
COPY pitchscanner ./pitchscanner
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip wheel \
 && /opt/venv/bin/pip install .

FROM python:3.12-slim AS runtime

# Non-root, because the container has no business writing to its own image.
RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PITCHSCANNER_STORE=postgres

WORKDIR /app
COPY --chown=app:app pitchscanner ./pitchscanner
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations
USER app

EXPOSE 8000
# /readyz, not /healthz: a process that is up but has no index is not ready to
# serve a search.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/readyz').status==200 else 1)"

CMD ["uvicorn", "pitchscanner.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
