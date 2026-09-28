#!/usr/bin/env python3
"""Fail the build on a large p95 regression.

Two guards, both needed:

* an absolute one, because the spec commits to p95 under 150ms end to end and
  under 10ms for the aggregate step, and that holds regardless of history;
* a relative one against the committed report, which is what catches an
  algorithmic regression that is still comfortably inside the absolute budget.

The relative tolerance is a *ratio*, not a percentage, and it is deliberately
loose. Shared CI runners vary by more than 2x on a bad morning, so anything
tighter would fail for reasons that have nothing to do with the code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

P95_BUDGET_MS = 150.0
AGGREGATE_BUDGET_MS = 10.0


def _by_encoding(report: dict[str, object]) -> dict[str, dict[str, float]]:
    rows = report.get("latency") or []
    assert isinstance(rows, list)
    return {str(r["encoding"]): r for r in rows}  # type: ignore[misc]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=2.5)
    args = ap.parse_args()

    candidate = _by_encoding(json.loads(args.candidate.read_text()))
    if not candidate:
        print(f"no latency rows in {args.candidate}", file=sys.stderr)
        return 1

    baseline: dict[str, dict[str, float]] = {}
    if args.baseline.exists():
        baseline = _by_encoding(json.loads(args.baseline.read_text()))
    else:
        print(f"note: no baseline at {args.baseline}; absolute budgets only")

    failures: list[str] = []
    for encoding, row in sorted(candidate.items()):
        p95 = float(row["p95_ms"])
        stages = row.get("stage_p95_ms") or {}
        aggregate = float(stages.get("aggregate", 0.0))  # type: ignore[union-attr]

        line = f"{encoding}: p95 {p95:.3f}ms, aggregate {aggregate:.3f}ms"
        if p95 > P95_BUDGET_MS:
            failures.append(
                f"{encoding}: p95 {p95:.3f}ms over the {P95_BUDGET_MS}ms budget"
            )
        if aggregate > AGGREGATE_BUDGET_MS:
            failures.append(
                f"{encoding}: aggregate {aggregate:.3f}ms over the {AGGREGATE_BUDGET_MS}ms budget"
            )

        prior = baseline.get(encoding)
        if prior is not None:
            was = float(prior["p95_ms"])
            ratio = p95 / was if was > 0 else 1.0
            line += f" (baseline {was:.3f}ms, {ratio:.2f}x)"
            if ratio > args.tolerance:
                failures.append(
                    f"{encoding}: p95 regressed {ratio:.2f}x against the committed report "
                    f"({was:.3f}ms -> {p95:.3f}ms), over the {args.tolerance}x tolerance"
                )
        print(line)

    if failures:
        print("\nFAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
