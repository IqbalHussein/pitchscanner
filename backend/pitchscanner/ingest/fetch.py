"""Fetch the open data at a pinned commit.

Pinning is what makes the pipeline reproducible: the same command a month from
now ingests byte-identical input, so a changed benchmark number means the code
changed, not the dataset.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pitchscanner.config import Settings, get_settings
from pitchscanner.logging_setup import get_logger

log = get_logger(__name__)
#: Only these folders are needed; the repo is large and the rest is noise.
SPARSE_PATHS = ("data/competitions.json", "data/matches", "data/events", "data/three-sixty")


def _run(args: list[str], cwd: Path | None = None) -> None:
    log.info("git", args=" ".join(args), cwd=str(cwd) if cwd else None)
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def fetch_open_data(settings: Settings | None = None, *, force: bool = False) -> Path:
    """Sparse-checkout statsbomb/open-data at the pinned commit. Idempotent."""

    cfg = settings or get_settings()
    target = Path(cfg.data_cache)
    if force and target.exists():
        shutil.rmtree(target)

    if not (target / ".git").exists():
        target.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "--quiet"], cwd=target)
        _run(["git", "remote", "add", "origin", cfg.open_data_repo], cwd=target)
        _run(["git", "config", "core.sparseCheckout", "true"], cwd=target)
        sparse = target / ".git" / "info" / "sparse-checkout"
        sparse.parent.mkdir(parents=True, exist_ok=True)
        sparse.write_text("\n".join(SPARSE_PATHS) + "\n")

    _run(["git", "fetch", "--depth", "1", "origin", cfg.open_data_commit], cwd=target)
    _run(["git", "checkout", "--quiet", "FETCH_HEAD"], cwd=target)
    log.info("open_data_ready", path=str(target), commit=cfg.open_data_commit)
    return target / "data"
