# One command per thing you actually want to do.
#
# `make up` brings the full stack up locally with the sample dataset.
# `make bench` writes the benchmark report the README quotes from.

SHELL := /bin/bash
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
BACKEND := backend
FRONTEND := frontend
export PYTHONPATH := $(BACKEND)

.DEFAULT_GOAL := help
.PHONY: help venv install install-faiss fixtures ingest ingest-real ingest-docker serve web \
        up down logs monitoring test test-unit test-property test-golden test-integration \
        test-slow test-e2e bench bench-quick perf-gate eval-pool lint format typecheck openapi \
        migrate clean ci

help: ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	 | awk 'BEGIN{FS=":.*?## "}{printf "  \033[1m%-18s\033[0m %s\n", $$1, $$2}'

# --- setup ----------------------------------------------------------------
venv: ## Create the virtualenv
	$(PY) -m venv $(VENV)

install: venv ## Install backend (with dev extras) and frontend dependencies
	$(BIN)/pip install --upgrade pip wheel
	$(BIN)/pip install -e "$(BACKEND)[dev]"
	cd $(FRONTEND) && npm install

install-faiss: ## Add the optional FAISS comparison dependency
	$(BIN)/pip install -e "$(BACKEND)[faiss]"

# --- data -----------------------------------------------------------------
fixtures: ## Regenerate the checked-in fixture dataset (deterministic)
	$(BIN)/python scripts/make_fixture.py
	$(BIN)/python scripts/make_filter_phrases.py

ingest: ## Ingest the fixtures into the in-memory store (no database needed)
	$(BIN)/python -m pitchscanner.cli --store memory ingest --fixtures

ingest-pg: ## Ingest the fixtures into Postgres
	$(BIN)/python -m pitchscanner.cli --store postgres ingest --fixtures

ingest-real: ## Fetch StatsBomb open data at the pinned commit and ingest it
	$(BIN)/python -m pitchscanner.cli --store postgres ingest

ingest-docker: ## Ingest the fixtures inside Compose
	docker compose run --rm ingest

# --- run ------------------------------------------------------------------
serve: ## Run the API against the in-memory fixture store
	PITCHSCANNER_STORE=memory $(BIN)/python -m pitchscanner.cli --store memory serve --reload

web: ## Run the frontend dev server
	cd $(FRONTEND) && npm run dev

up: ## Bring up the whole stack (db, api, web) and load the sample dataset
	docker compose up -d --build
	docker compose run --rm ingest
	@echo "app: http://localhost:8080   api docs: http://localhost:8000/docs"

monitoring: ## Add Prometheus, Grafana (dashboard provisioned) and the /readyz probe
	docker compose --profile monitoring up -d
	@echo "grafana: http://localhost:3000   prometheus: http://localhost:9090"

down: ## Stop the stack
	docker compose --profile monitoring --profile tools down

logs: ## Tail the stack's logs
	docker compose logs -f --tail=100

# --- tests ----------------------------------------------------------------
test: ## Unit, property, golden and API tests (the fast tiers)
	cd $(BACKEND) && ../$(BIN)/python -m pytest tests -q -m "not integration"

test-cov: ## The fast tiers with coverage on the core packages
	cd $(BACKEND) && ../$(BIN)/python -m pytest tests -q -m "not integration" \
	  --cov=pitchscanner --cov-report=term-missing:skip-covered

test-integration: ## Pipeline and API against a real Postgres
	# Set PITCHSCANNER_TEST_DSN to use a database you already have; otherwise the
	# tier starts one with Testcontainers.
	cd $(BACKEND) && ../$(BIN)/python -m pytest tests/integration -q -m integration -ra

test-slow: ## The nightly tier: deep property tests (slow on purpose)
	cd $(BACKEND) && HYPOTHESIS_PROFILE=nightly PITCHSCANNER_SEARCH_EXAMPLES=50 \
	  ../$(BIN)/python -m pytest tests/property -q

test-e2e: ## Playwright against the built frontend and the fixture API
	# Playwright starts the API with a bare `python`, which is what CI provides;
	# locally that has to be the venv's interpreter.
	cd $(FRONTEND) && npm run build && \
	  PATH="$(CURDIR)/$(BIN):$$PATH" npm run test:e2e

# --- benchmarks -----------------------------------------------------------
bench: ## Full benchmark report into bench/reports/ (postgres/ for a real index)
	# Two committed reports, one per backend: a postgres run must not overwrite
	# the memory one, because only the postgres one has meaningful recall.
	@STORE=$${PITCHSCANNER_STORE:-memory}; OUT=bench/reports; \
	 if [ "$$STORE" = postgres ]; then OUT=bench/reports/postgres; fi; \
	 set -x; $(BIN)/python -m pitchscanner.cli --store $$STORE bench --out $$OUT

bench-quick: ## Short benchmark, for CI regression checks
	$(BIN)/python -m pitchscanner.cli --store memory bench --quick --out bench/reports

perf-gate: ingest ## Compare a fresh quick benchmark against the committed report
	$(BIN)/python -m pitchscanner.cli --store memory bench --quick --out bench/ci
	$(BIN)/python scripts/check_perf.py --baseline bench/reports/benchmark.json \
	  --candidate bench/ci/benchmark.json --tolerance 2.5

eval-pool: ## Build the blind evaluation pool for hand labelling
	$(BIN)/python -m pitchscanner.cli --store memory eval-pool --out bench/reports

# --- quality --------------------------------------------------------------
lint: ## ruff + eslint
	cd $(BACKEND) && ../$(BIN)/python -m ruff check .
	cd $(FRONTEND) && npm run lint

format: ## Apply ruff's formatter and fixes
	cd $(BACKEND) && ../$(BIN)/python -m ruff check --fix . && ../$(BIN)/python -m ruff format .

typecheck: ## mypy (strict on the core packages) + tsc
	cd $(BACKEND) && ../$(BIN)/python -m mypy pitchscanner
	cd $(FRONTEND) && npm run gen:api && npm run typecheck

openapi: ## Regenerate openapi.json and the generated TypeScript types
	$(BIN)/python -m pitchscanner.cli --store memory openapi --out $(FRONTEND)/openapi.json
	cd $(FRONTEND) && npm run gen:api

migrate: ## Apply database migrations
	cd $(BACKEND) && ../$(BIN)/alembic upgrade head

ci: lint typecheck test perf-gate ## What CI runs, minus the containers

clean: ## Remove caches and build artefacts
	rm -rf .data bench/ci $(FRONTEND)/dist $(FRONTEND)/test-results $(FRONTEND)/playwright-report
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf $(BACKEND)/.pytest_cache $(BACKEND)/.mypy_cache $(BACKEND)/.ruff_cache
