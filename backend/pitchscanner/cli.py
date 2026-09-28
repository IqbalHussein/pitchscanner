"""Command line entry points: `pitchscanner <command>`.

Argparse rather than a CLI framework: the surface is small, and one less
dependency in the API image is worth more than subcommand sugar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pitchscanner.config import get_settings
from pitchscanner.logging_setup import configure_logging, get_logger
from pitchscanner.store.base import Store

log = get_logger(__name__)


def _store(args: argparse.Namespace) -> tuple[Store, Path | None]:
    from pitchscanner.store import MEMORY_SNAPSHOT, MemoryStore, PostgresStore

    cfg = get_settings()
    if args.store == "memory":
        path = Path(cfg.data_cache).parent / MEMORY_SNAPSHOT.name
        if getattr(args, "reuse", False) and path.exists():
            return MemoryStore.load(path), path
        return MemoryStore(), path
    return PostgresStore(cfg.dsn, ef_search=cfg.hnsw_ef_search, probes=cfg.ivfflat_probes), None


def cmd_ingest(args: argparse.Namespace) -> int:
    from pitchscanner.ingest import fetch_open_data, ingest, write_reject_log
    from pitchscanner.store.memory import MemoryStore

    cfg = get_settings()
    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir is None:
        data_dir = Path(cfg.fixture_dir) if args.fixtures else fetch_open_data(cfg)
    store, snapshot = _store(args)

    report = ingest(
        data_dir,
        store,
        settings=cfg,
        encodings=args.encodings.split(",") if args.encodings else None,
        limit_matches=args.limit,
    )
    rejects = write_reject_log(report.rejects, Path(args.reject_log))
    if snapshot is not None and isinstance(store, MemoryStore):
        store.save(snapshot)
        log.info("snapshot_written", path=str(snapshot))
    print(
        json.dumps(
            {**report.as_dict(), "reject_log": args.reject_log, "rejects": rejects}, indent=2
        )
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "pitchscanner.api.app:app", host=args.host, port=args.port, reload=args.reload
    )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Brute-force CLI search - the week 1 deliverable, still useful for debugging."""

    from pitchscanner.models import SearchRequest
    from pitchscanner.search import SearchService

    store, _ = _store(args)
    request = SearchRequest.model_validate(json.loads(Path(args.query).read_text()))
    outcome = SearchService(store, get_settings()).search(request)
    print(
        json.dumps(
            {
                "results": [r.model_dump() for r in outcome.results],
                "stats": outcome.stats.model_dump() if outcome.stats else None,
                "timing_ms": outcome.timing.model_dump(),
            },
            indent=2,
            default=str,
        )
    )
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from pitchscanner.bench.report import run_all

    store, _ = _store(args)
    path = run_all(
        store,
        out_dir=Path(args.out),
        queries=args.queries,
        encodings=args.encodings.split(",") if args.encodings else None,
        quick=args.quick,
    )
    print(f"report written to {path}")
    return 0


def cmd_eval_pool(args: argparse.Namespace) -> int:
    """Generate the blind evaluation pool for hand labelling."""

    from pitchscanner.bench.quality import build_eval_pool, write_eval_pool

    store, _ = _store(args)
    pool = build_eval_pool(store, n_queries=args.queries, per_encoding=args.per_encoding)
    path = Path(args.out) / "eval-pool.jsonl"
    n = write_eval_pool(pool, path)
    labels = Path(args.out) / "eval-labels.jsonl"
    if not labels.exists():
        labels.write_text(
            "\n".join(
                json.dumps({"query_id": pq.query_id, "labels": {}}) for pq in pool
            )
            + "\n"
        )
    print(
        json.dumps(
            {
                "pool": str(path),
                "queries": n,
                "labels_template": str(labels),
                "how_to": "Set each candidate's relevance to 0 or 1 (or 0-3 for nDCG) "
                "in the labels file. The pool hides which encoding produced each "
                "candidate; keep it that way while labelling.",
            },
            indent=2,
        )
    )
    return 0


def cmd_openapi(args: argparse.Namespace) -> int:
    from pitchscanner.api import create_app

    spec = create_app().openapi()
    Path(args.out).write_text(json.dumps(spec, indent=2))
    print(f"openapi written to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pitchscanner")
    parser.add_argument(
        "--store", choices=("postgres", "memory"), default=get_settings().store
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="run the offline pipeline")
    p.add_argument("--data-dir", help="a local StatsBomb data directory")
    p.add_argument("--fixtures", action="store_true", help="use the checked-in fixtures")
    p.add_argument("--limit", type=int, default=None, help="stop after N matches")
    p.add_argument("--encodings", default=None, help="comma-separated, default all")
    p.add_argument("--reject-log", default="bench/reports/rejects.jsonl")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("serve", help="run the API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("search", help="run one query from a JSON file")
    p.add_argument("query")
    p.add_argument("--reuse", action="store_true", default=True)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("bench", help="systems and retrieval benchmarks")
    p.add_argument("--out", default="bench/reports")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--encodings", default=None)
    p.add_argument("--quick", action="store_true", help="small run, for CI")
    p.add_argument("--reuse", action="store_true", default=True)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("eval-pool", help="build the blind evaluation pool")
    p.add_argument("--out", default="bench/reports")
    p.add_argument("--queries", type=int, default=200)
    p.add_argument("--per-encoding", type=int, default=10)
    p.add_argument("--reuse", action="store_true", default=True)
    p.set_defaults(func=cmd_eval_pool)

    p = sub.add_parser("openapi", help="write the OpenAPI schema")
    p.add_argument("--out", default="frontend/openapi.json")
    p.set_defaults(func=cmd_openapi)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
