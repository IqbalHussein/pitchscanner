"""`make bench` - one command, one Markdown report, committed to the repo.

Every number the README or a resume bullet quotes has to come from here, and the
report records the hardware and the commit it ran on so a stale number is
obvious.
"""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from pitchscanner.bench.latency import measure_latency
from pitchscanner.bench.quality import score_encodings
from pitchscanner.bench.queries import build_jitter_set, build_query_set
from pitchscanner.bench.recall import measure_jitter_robustness, measure_recall
from pitchscanner.config import get_settings
from pitchscanner.encoders import available, get_encoder
from pitchscanner.models import Filters
from pitchscanner.store.base import Store


class BenchmarkPayload(TypedDict):
    """The shape `benchmark.json` holds, and the renderer reads."""

    generated_at: str
    commit: str
    hardware: dict[str, str]
    settings: dict[str, Any]
    dataset: dict[str, int]
    latency: list[dict[str, Any]]
    recall: list[dict[str, Any]]
    jitter: list[dict[str, Any]]
    index: list[dict[str, Any]]
    quality: list[dict[str, Any]]


P95_TARGET_MS = 150.0
AGGREGATE_TARGET_MS = 10.0
RECALL_TARGET = 0.95


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _hardware() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
    }


def run_all(
    store: Store,
    *,
    out_dir: Path,
    queries: int = 1000,
    encodings: Sequence[str] | None = None,
    quick: bool = False,
) -> Path:
    from pitchscanner.store.memory import MemoryStore

    cfg = get_settings()
    # The report has to name the store it actually ran against, not the one the
    # environment happens to default to: the recall caveat below turns on it.
    store_name = "memory" if isinstance(store, MemoryStore) else cfg.store
    encodings = tuple(encodings or available())
    n_queries = 60 if quick else queries
    n_recall = 30 if quick else 200

    query_set = build_query_set(store, n_queries)
    recall_set = query_set[:n_recall]
    jitter_set = build_jitter_set(store, 30 if quick else 200)

    counts = store.counts()
    payload: BenchmarkPayload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "hardware": _hardware(),
        "settings": {
            "store": store_name,
            "index_kind": cfg.index_kind,
            "hnsw_ef_search": cfg.hnsw_ef_search,
            "ivfflat_probes": cfg.ivfflat_probes,
            "candidate_multiplier": cfg.candidate_multiplier,
            "outcome_window_s": cfg.outcome_window_s,
        },
        "dataset": counts.as_dict(),
        "latency": [],
        "recall": [],
        "jitter": [],
        "index": [],
        "quality": [],
    }

    for encoding in encodings:
        latency = measure_latency(store, query_set, encoding=encoding)
        payload["latency"].append(latency.as_dict())
        for k in (10, 50):
            payload["recall"].append(
                measure_recall(store, recall_set, encoding=encoding, k=k).as_dict()
            )
        # Recall with a filter applied is the number that actually matters: a
        # strict filter is where ANN quality falls apart.
        payload["recall"].append(
            measure_recall(
                store,
                recall_set,
                encoding=encoding,
                k=10,
                filters=Filters(game_state="losing"),
            ).as_dict()
        )
        payload["jitter"].append(
            measure_jitter_robustness(store, jitter_set, encoding=encoding).as_dict()
        )
        encoder = get_encoder(encoding)
        payload["index"].append(
            {
                "encoding": encoding,
                "dim": encoder.dim,
                "metric": encoder.metric,
                **store.build_index(encoder.version, cfg.index_kind),
            }
        )

    payload["quality"] = [
        r.as_dict()
        for r in score_encodings(
            Path(out_dir) / "eval-pool.jsonl",
            Path(out_dir) / "eval-labels.jsonl",
        )
    ]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "benchmark.json").write_text(json.dumps(payload, indent=2))
    report_path = out_dir / "benchmark.md"
    report_path.write_text(render_markdown(payload))
    return report_path


def _recall_caveat(payload: BenchmarkPayload) -> str:
    """Say plainly when recall is measured against a backend that is already exact."""

    settings = payload["settings"]
    if settings.get("store") == "memory":
        return (
            "> These recall numbers were produced against the in-memory backend, "
            "which is exact by construction, so they are 1.0 by definition and mean "
            "nothing. Recall is only informative against `store=postgres` with an "
            "ANN index. Do not quote the numbers below from a memory run."
        )
    return (
        f"Measured with `index_kind={settings.get('index_kind')}`, "
        f"`ef_search={settings.get('hnsw_ef_search')}`, "
        f"`probes={settings.get('ivfflat_probes')}`."
    )


def render_markdown(payload: BenchmarkPayload) -> str:
    """Markdown with tables, targets and a verdict per target."""

    dataset = payload["dataset"]
    hardware = payload["hardware"]
    lines: list[str] = [
        "# Benchmark report",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Commit: `{payload['commit']}`",
        f"- Hardware: `{hardware.get('platform')}` / Python `{hardware.get('python')}`",
        f"- Settings: `{json.dumps(payload['settings'])}`",
        "",
        "Rerun with `make bench`. Every number quoted anywhere else in this "
        "repo has to come from this file.",
        "",
        "## Dataset",
        "",
        "| matches | events | frames | possessions | outcomes |",
        "| --- | --- | --- | --- | --- |",
        f"| {dataset['matches']} | {dataset['events']} | {dataset['frames']} "
        f"| {dataset['possessions']} | {dataset['outcomes']} |",
        "",
        "## Query latency",
        "",
        f"Target: p95 under {P95_TARGET_MS:.0f}ms end to end, "
        f"aggregate under {AGGREGATE_TARGET_MS:.0f}ms.",
        "",
        "| encoding | queries | p50 ms | p95 ms | p99 ms | encode | search | dedup "
        "| aggregate | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["latency"]:
        stages = row["stage_p95_ms"]
        ok = row["p95_ms"] <= P95_TARGET_MS and stages["aggregate"] <= AGGREGATE_TARGET_MS
        lines.append(
            f"| {row['encoding']} | {row['queries']} | {row['p50_ms']} | {row['p95_ms']} "
            f"| {row['p99_ms']} | {stages['encode']} | {stages['search']} | {stages['dedup']} "
            f"| {stages['aggregate']} | {'PASS' if ok else 'FAIL'} |"
        )

    lines += [
        "",
        "## Recall against exact search",
        "",
        f"Target: {RECALL_TARGET:.2f} or higher at the chosen settings.",
        "",
        _recall_caveat(payload),
        "",
        "| encoding | k | filtered | recall | queries | verdict |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["recall"]:
        ok = row["recall"] >= RECALL_TARGET
        lines.append(
            f"| {row['encoding']} | {row['k']} | {row['filtered']} | {row['recall']} "
            f"| {row['queries']} | {'PASS' if ok else 'FAIL'} |"
        )

    lines += [
        "",
        "## Jitter sanity check",
        "",
        "Every player nudged 1-2m; the original frame should still rank first.",
        "",
        "| encoding | original in top 1 | original in top 5 | queries |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["jitter"]:
        lines.append(
            f"| {row['encoding']} | {row['original_in_top1']} | {row['original_in_top5']} "
            f"| {row['queries']} |"
        )

    lines += [
        "",
        "## Index size and build",
        "",
        "| encoding | dim | metric | kind | vectors | bytes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["index"]:
        lines.append(
            f"| {row['encoding']} | {row['dim']} | {row['metric']} | {row['kind']} "
            f"| {row['vectors']} | {row['bytes']} |"
        )

    quality = payload["quality"]
    lines += ["", "## Retrieval quality (blind, hand-labelled)", ""]
    if quality:
        lines += [
            "| encoding | precision@10 | nDCG@10 | labelled queries |",
            "| --- | --- | --- | --- |",
        ]
        for row in quality:
            lines.append(
                f"| {row['encoding']} | {row['precision@10']} | {row['ndcg@10']} "
                f"| {row['labelled_queries']} |"
            )
    else:
        lines.append(
            "No labels found. Generate the pool with `make eval-pool`, label "
            "`bench/reports/eval-labels.jsonl`, then rerun `make bench`. "
            "Until then there are no precision numbers, and none should be quoted."
        )
    lines.append("")
    return "\n".join(lines)
