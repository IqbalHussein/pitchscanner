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
#: What a competition-filtered run needs before it knows which matches it wants.
_INDEX_PATHS = ("data/competitions.json", "data/matches")


def _run(args: list[str], cwd: Path | None = None) -> None:
    log.info("git", args=" ".join(args), cwd=str(cwd) if cwd else None)
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def _set_sparse(target: Path, paths: Sequence[str]) -> None:
    sparse = target / ".git" / "info" / "sparse-checkout"
    sparse.parent.mkdir(parents=True, exist_ok=True)
    sparse.write_text("\n".join(paths) + "\n")


def _selected_match_ids(
    data_dir: Path, competitions: Sequence[CompetitionSelector] | None
) -> list[int]:
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


def _three_sixty_ids(target: Path) -> set[int]:
    """Match ids that have a three-sixty file at FETCH_HEAD.

    Reads the tree listing only, which a blobless fetch already has, so nothing
    but the (small) tree objects is downloaded.
    """

    done = subprocess.run(
        ["git", "ls-tree", "--name-only", "FETCH_HEAD", "data/three-sixty/"],
        cwd=target, check=True, capture_output=True, text=True,
    )
    ids: set[int] = set()
    for line in done.stdout.splitlines():
        name = Path(line).name
        if name.endswith(".json") and name[:-5].isdigit():
            ids.add(int(name[:-5]))
    return ids


def fetch_open_data(
    settings: Settings | None = None,
    *,
    force: bool = False,
    competitions: Sequence[CompetitionSelector] | None = None,
) -> Path:
    """Sparse-checkout statsbomb/open-data at the pinned commit. Idempotent.

    The fetch is always narrowed, in two steps. The repository is cloned without
    file contents (``--filter=blob:none``) and only the small ``matches/`` index is
    checked out. The ``data/three-sixty`` tree listing says which matches have
    frames; those ids, restricted to ``competitions`` when given, decide which
    ``events/<id>.json`` and ``three-sixty/<id>.json`` files are checked out next,
    and git downloads only those blobs. A match without a three-sixty file is never
    ingested, and the full ``events/`` folder (about 3,400 matches) is several GB of
    files that would go unused.
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
        _run(["git", "config", "remote.origin.promisor", "true"], cwd=target)
        _run(["git", "config", "remote.origin.partialclonefilter", "blob:none"], cwd=target)

    _run(["git", "fetch", "--depth", "1", "--filter=blob:none", "origin", cfg.open_data_commit],
         cwd=target)

    _set_sparse(target, _INDEX_PATHS)
    _run(["git", "checkout", "--quiet", "FETCH_HEAD"], cwd=target)
    wanted = _selected_match_ids(target / "data", competitions or None)
    with_frames = _three_sixty_ids(target)
    ids = [i for i in wanted if i in with_frames]
    # Leading slash anchors each pattern to the repository root.
    paths = [f"/{p}" for p in _INDEX_PATHS]
    for match_id in ids:
        paths += [f"/data/events/{match_id}.json", f"/data/three-sixty/{match_id}.json"]
    _set_sparse(target, paths)
    # Already on FETCH_HEAD, so `checkout` would do nothing; re-reading the
    # tree is what applies the widened sparse patterns.
    _run(["git", "read-tree", "-mu", "HEAD"], cwd=target)
    log.info(
        "open_data_narrowed",
        competitions=len(competitions or ()),
        matches=len(ids),
        without_frames=len(wanted) - len(ids),
    )
    log.info("open_data_ready", path=str(target), commit=cfg.open_data_commit)
    return target / "data"
