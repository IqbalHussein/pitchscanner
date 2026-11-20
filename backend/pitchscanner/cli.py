"""Command line entry points: `pitchscanner <command>`.

Argparse rather than a CLI framework: the surface is small, and one less
dependency in the API image is worth more than subcommand sugar.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pitchscanner.config import get_settings
from pitchscanner.ingest.selection import CompetitionSelector, parse_competitions
from pitchscanner.logging_setup import configure_logging, get_logger
from pitchscanner.store.base import Store

log = get_logger(__name__)


def _store(args: argparse.Namespace) -> tuple[Store, Path | None]:
    from pitchscanner.store import MEMORY_SNAPSHOT, ArrayStore, MemoryStore, PostgresStore
    from pitchscanner.store.arrays import parse_dtype_spec

    cfg = get_settings()
    if args.store == "arrays":
        # Ingest streams into a new store directory; every other command reads one.
        if args.command == "ingest":
            if not args.out:
                raise SystemExit("--store arrays ingest needs --out DIR")
            return ArrayStore.create(args.out, dtypes=parse_dtype_spec(cfg.arrays_dtypes)), None
        return ArrayStore.from_settings(cfg), None
    if args.store == "memory":
        path = Path(cfg.data_cache).parent / MEMORY_SNAPSHOT.name
        if getattr(args, "reuse", False) and path.exists():
            return MemoryStore.load(path), path
        return MemoryStore(), path
    return PostgresStore.from_settings(cfg), None


def _competitions_arg(value: str) -> tuple[CompetitionSelector, ...]:
    try:
        return parse_competitions(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def cmd_ingest(args: argparse.Namespace) -> int:
    from pitchscanner.ingest import fetch_open_data, ingest, write_reject_log
    from pitchscanner.store.arrays import ArrayStore
    from pitchscanner.store.memory import MemoryStore

    cfg = get_settings()
    competitions = args.competitions
    data_dir = Path(args.data_dir) if args.data_dir else None
    if data_dir is None:
        data_dir = (
            Path(cfg.fixture_dir)
            if args.fixtures
            else fetch_open_data(cfg, competitions=competitions)
        )
    store, snapshot = _store(args)

    report = ingest(
        data_dir,
        store,
        settings=cfg,
        encodings=args.encodings.split(",") if args.encodings else None,
        limit_matches=args.limit,
        competitions=competitions,
        fail_on_bad_file=args.fail_on_bad_file,
    )
    rejects = write_reject_log(report.rejects, Path(args.reject_log))
    if snapshot is not None and isinstance(store, MemoryStore):
        store.save(snapshot)
        log.info("snapshot_written", path=str(snapshot))
    arrays_summary: dict[str, object] | None = None
    if isinstance(store, ArrayStore):
        source = "fixtures" if args.fixtures else str(data_dir)
        arrays_summary = store.finalize(
            extra_meta={
                "source": source,
                "open_data_commit": None if args.fixtures else cfg.open_data_commit,
                "encodings_requested": args.encodings,
            }
        )
        arrays_summary.pop("files", None)
    print(
        json.dumps(
            {
                **report.as_dict(),
                "reject_log": args.reject_log,
                "rejects": rejects,
                **({"arrays": arrays_summary} if arrays_summary else {}),
            },
            indent=2,
        )
    )
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "pitchscanner.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
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
        "--store", choices=("postgres", "memory", "arrays"), default=get_settings().store
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="run the offline pipeline")
    p.add_argument("--data-dir", help="a local StatsBomb data directory")
    p.add_argument("--fixtures", action="store_true", help="use the checked-in fixtures")
    p.add_argument("--limit", type=int, default=None, help="stop after N matches")
    p.add_argument("--encodings", default=None, help="comma-separated, default all")
    p.add_argument(
        "--competitions",
        type=_competitions_arg,
        default=None,
        help="comma-separated competition_id or competition_id:season_id entries "
        "(e.g. 55:282,9); only those matches are fetched and ingested",
    )
    p.add_argument("--out", help="with --store arrays: the directory to write the store into")
    p.add_argument("--reject-log", default="bench/reports/rejects.jsonl")
    p.add_argument(
        "--fail-on-bad-file",
        action="store_true",
        help="abort on an unreadable events/three-sixty file instead of skipping its match",
    )
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
