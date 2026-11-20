"""`ingest --store arrays --out DIR` and `bench` against an array store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pitchscanner.cli import main as cli_main
from pitchscanner.store import ArrayStore


def test_cli_ingest_streams_into_arrays_and_bench_reads_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "store"
    rc = cli_main([
        "--store", "arrays", "ingest", "--fixtures", "--limit", "1", "--encodings", "E1,E2",
        "--out", str(out), "--reject-log", str(tmp_path / "rej.jsonl"),
    ])
    assert rc == 0
    printed = capsys.readouterr().out
    assert '"arrays"' in printed and '"searchable_bytes_per_frame"' in printed
    store = ArrayStore.open(out)
    assert set(store.vector_counts()) == {"E1-v1", "E2-v1"}
    meta = json.loads((out / "meta.json").read_text())
    assert meta["source"] == "fixtures" and meta["open_data_commit"] is None

    with pytest.raises(SystemExit, match="--out"):
        cli_main(["--store", "arrays", "ingest", "--fixtures", "--limit", "1"])

    from pitchscanner.config import get_settings

    monkeypatch.setenv("PITCHSCANNER_ARRAYS_DIR", str(out))
    get_settings.cache_clear()
    try:
        rc = cli_main(["--store", "arrays", "bench", "--quick", "--encodings", "E1,E2",
                       "--out", str(tmp_path / "bench")])
    finally:
        get_settings.cache_clear()
    assert rc == 0
    report = json.loads((tmp_path / "bench" / "benchmark.json").read_text())
    assert report["settings"]["store"] == "arrays"
    assert report["latency"][0]["p95_ms"] > 0
