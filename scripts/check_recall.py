#!/usr/bin/env python3
"""Fail the nightly build when ANN recall drops below the spec's target.

Only meaningful against a real index. A report generated with the in-memory
store is exact by construction, so its recall is 1.0 by definition; this script
refuses such a report rather than passing it, because a gate that cannot fail is
worse than no gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--min", type=float, default=0.95)
    args = ap.parse_args()

    report = json.loads(args.report.read_text())
    store = str(report.get("settings", {}).get("store", "unknown"))
    if store != "postgres":
        print(
            f"{args.report} was generated with store={store}; recall from an exact "
            "store is 1.0 by definition and cannot be gated on.",
            file=sys.stderr,
        )
        return 1

    rows = report.get("recall") or []
    if not rows:
        print(f"no recall rows in {args.report}", file=sys.stderr)
        return 1

    failures: list[str] = []
    for row in rows:
        recall = float(row["recall"])
        label = (
            f"{row['encoding']} k={row['k']}"
            f"{' filtered' if row.get('filtered') else ''}"
        )
        print(f"{label}: recall {recall:.4f} over {row.get('queries', '?')} queries")
        if recall < args.min:
            failures.append(f"{label}: recall {recall:.4f} below {args.min}")

    if failures:
        print("\nFAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"\nPASS (index_kind={report['settings'].get('index_kind')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
