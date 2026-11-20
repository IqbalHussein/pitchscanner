"""api/index.py picks the array store from the environment or a bundled api/_data."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pitchscanner.config import Settings
from pitchscanner.ingest import ingest
from pitchscanner.store.arrays import ArrayStore
from tests.conftest import FIXTURE_DIR, REPO_ROOT

DRIVER = """
import runpy, sys
from fastapi.testclient import TestClient
ns = runpy.run_path(sys.argv[1])
import os
print("STORE", os.environ["PITCHSCANNER_STORE"])
with TestClient(ns["app"].app if hasattr(ns["app"], "app") else ns["app"]) as c:
    print("READY", c.get("/readyz").json()["status"])
    body = {"ball": {"x": 84, "y": 22}, "attackers": [{"x": 86, "y": 20}],
            "defenders": [{"x": 92, "y": 25}], "k": 3}
    r = c.post("/v1/search", json=body)
    print("SEARCH", r.status_code, len(r.json()["results"]))
    eid = r.json()["results"][0]["event_id"]
    print("SEQ", c.get(f"/v1/situations/{eid}/sequence").status_code)
    print("META", c.get("/v1/meta").json()["index_kind"])
"""


@pytest.fixture(scope="module")
def store_dir(settings: Settings, tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("vercel-arrays")
    store = ArrayStore.create(d)
    ingest(FIXTURE_DIR, store, settings=settings, encodings=["E1"], limit_matches=2)
    store.finalize()
    return d


def _run(index: Path, env: dict[str, str]) -> dict[str, str]:
    full = {k: v for k, v in os.environ.items() if not k.startswith("PITCHSCANNER_")}
    full.update(env)
    full["PYTHONPATH"] = str(REPO_ROOT / "backend")
    out = subprocess.run(
        [sys.executable, "-c", DRIVER, str(index)],
        env=full, capture_output=True, text=True, timeout=120, check=False,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return dict(line.split(" ", 1) for line in out.stdout.splitlines() if " " in line)


def test_env_var_selects_the_array_store(store_dir: Path) -> None:
    got = _run(REPO_ROOT / "api" / "index.py", {"PITCHSCANNER_ARRAYS_DIR": str(store_dir)})
    assert got["STORE"] == "arrays" and got["READY"] == "ready"
    assert got["SEARCH"] == "200 3" and got["SEQ"] == "200" and got["META"] == "exact"


def test_a_bundled_store_is_found_without_any_env(store_dir: Path, tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    shutil.copy(REPO_ROOT / "api" / "index.py", tmp_path / "api" / "index.py")
    shutil.copytree(store_dir, tmp_path / "api" / "_data")
    got = _run(tmp_path / "api" / "index.py", {})
    assert got["STORE"] == "arrays" and got["SEARCH"] == "200 3"


def test_no_arrays_keeps_the_fixture_mode(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    shutil.copy(REPO_ROOT / "api" / "index.py", tmp_path / "api" / "index.py")
    (tmp_path / "data" / "fixtures").mkdir(parents=True)
    shutil.copytree(FIXTURE_DIR, tmp_path / "data" / "fixtures" / "statsbomb")
    got = _run(tmp_path / "api" / "index.py", {})
    assert got["STORE"] == "memory" and got["SEARCH"] == "200 3"
