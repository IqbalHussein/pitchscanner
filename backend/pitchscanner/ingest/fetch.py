"""Fetch the open data at a pinned commit.

Pinning is what makes the pipeline reproducible: the same command a month from
now ingests byte-identical input, so a changed benchmark number means the code
changed, not the dataset.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

from pitchscanner.config import Settings, get_settings
from pitchscanner.ingest.parse import load_json
from pitchscanner.ingest.selection import CompetitionSelector, is_selected
from pitchscanner.logging_setup import get_logger

log = get_logger(__name__)
#: Only these folders are needed; the repo is large and the rest is noise.
SPARSE_PATHS = ("data/competitions.json", "data/matches", "data/events", "data/three-sixty")
#: What a competition-filtered run needs before it knows which matches it wants.
_INDEX_PATHS = ("data/competitions.json", "data/matches")


def _run(args: list[str], cwd: Path | None = None) -> None:
    log.info("git", args=" ".join(args), cwd=str(cwd) if cwd else None)
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def _set_sparse(target: Path, paths: Sequence[str]) -> None:
    sparse = target / ".git" / "info" / "sparse-checkout"
    sparse.parent.mkdir(parents=True, exist_ok=True)
    sparse.write_text("\n".join(paths) + "\n")


def _selected_match_ids(data_dir: Path, competitions: Sequence[CompetitionSelector]) -> list[int]:
    ids: list[int] = []
    matches_dir = data_dir / "matches"
    for comp_dir in sorted(matches_dir.iterdir()) if matches_dir.exists() else []:
        for season_file in sorted(comp_dir.glob("*.json")):
            try:
                comp_id, season_id = int(comp_dir.name), int(season_file.stem)
            except ValueError:
                continue
            if not is_selected(competitions, comp_id, season_id):
                continue
            ids.extend(int(m["match_id"]) for m in load_json(season_file))
    return ids


def fetch_open_data(
    settings: Settings | None = None,
    *,
    force: bool = False,
    competitions: Sequence[CompetitionSelector] | None = None,
) -> Path:
    """Sparse-checkout statsbomb/open-data at the pinned commit. Idempotent.

    With ``competitions``, the fetch is narrowed in two steps. The repository is
    cloned without file contents (``--filter=blob:none``) and only the small
    ``matches/`` index is checked out; the match ids found there decide which
    ``events/<id>.json`` and ``three-sixty/<id>.json`` files are checked out
    next, and git downloads only those blobs. The full ``events/`` and
    ``three-sixty/`` folders are several GB; a handful of competitions is a
    small fraction of that. Without it, behaviour is unchanged.
    """

    cfg = settings or get_settings()
    target = Path(cfg.data_cache)
    if force and target.exists():
        shutil.rmtree(target)

    if not (target / ".git").exists():
        target.mkdir(parents=True, exist_ok=True)
        _run(["git", "init", "--quiet"], cwd=target)
        _run(["git", "remote", "add", "origin", cfg.open_data_repo], cwd=target)
        _run(["git", "config", "core.sparseCheckout", "true"], cwd=target)
        if competitions:
            _run(["git", "config", "remote.origin.promisor", "true"], cwd=target)
            _run(["git", "config", "remote.origin.partialclonefilter", "blob:none"], cwd=target)

    fetch = ["git", "fetch", "--depth", "1"]
    if competitions:
        fetch.append("--filter=blob:none")
    _run([*fetch, "origin", cfg.open_data_commit], cwd=target)

    if competitions:
        _set_sparse(target, _INDEX_PATHS)
        _run(["git", "checkout", "--quiet", "FETCH_HEAD"], cwd=target)
        ids = _selected_match_ids(target / "data", competitions)
        # Leading slash anchors each pattern to the repository root. Ids with no
        # three-sixty file simply match nothing.
        paths = [*(f"/{p}" for p in _INDEX_PATHS)]
        for match_id in ids:
            paths += [f"/data/events/{match_id}.json", f"/data/three-sixty/{match_id}.json"]
        _set_sparse(target, paths)
        # Already on FETCH_HEAD, so `checkout` would do nothing; re-reading the
        # tree is what applies the widened sparse patterns.
        _run(["git", "read-tree", "-mu", "HEAD"], cwd=target)
        log.info("open_data_narrowed", competitions=len(competitions), matches=len(ids))
    else:
        _set_sparse(target, SPARSE_PATHS)
        _run(["git", "checkout", "--quiet", "FETCH_HEAD"], cwd=target)
        _run(["git", "read-tree", "-mu", "HEAD"], cwd=target)
    log.info("open_data_ready", path=str(target), commit=cfg.open_data_commit)
    return target / "data"
