"""In-process store. Also the exact-search reference implementation.

Brute force over a NumPy matrix is the ground truth recall is measured against,
and at fixture size it is genuinely fast enough to serve the demo without any
infrastructure at all (``PITCHSCANNER_STORE=memory``).
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from pitchscanner.models import (
    EventRecord,
    Filters,
    FrameRecord,
    MatchRecord,
    OutcomeRecord,
    ZoneBaseline,
)
from pitchscanner.store.base import (
    Candidate,
    CatalogueEntry,
    Counts,
    CoverageRow,
    FrameMeta,
    TeamRow,
)
from pitchscanner.store.filtering import matches as filter_matches


def _distances(matrix: np.ndarray, vector: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        norms = np.linalg.norm(matrix, axis=1)
        norms[norms == 0] = 1e-9
        qn = float(np.linalg.norm(vector)) or 1e-9
        return np.asarray(1.0 - (matrix @ vector) / (norms * qn), dtype=np.float64)
    return np.asarray(np.linalg.norm(matrix - vector[None, :], axis=1), dtype=np.float64)


class MemoryStore:
    def __init__(self) -> None:
        self._matches: dict[int, MatchRecord] = {}
        self._events: dict[str, EventRecord] = {}
        self._by_possession: dict[tuple[int, int], list[str]] = {}
        self._frames: dict[str, FrameRecord] = {}
        self._metas: dict[str, FrameMeta] = {}
        self._outcomes: dict[str, OutcomeRecord] = {}
        self._baselines: dict[str, list[ZoneBaseline]] = {}
        self._catalogue: dict[tuple[str, int], CatalogueEntry] = {}
        self._vectors: dict[str, tuple[list[str], np.ndarray]] = {}

    # --- writes ------------------------------------------------------------
    def init_schema(self) -> None:  # nothing to create
        return None

    def upsert_matches(self, matches: Iterable[MatchRecord]) -> int:
        n = 0
        for m in matches:
            self._matches[m.match_id] = m
            n += 1
        return n

    def upsert_events(self, events: Iterable[EventRecord]) -> int:
        n = 0
        for e in events:
            self._events[e.event_id] = e
            self._by_possession.setdefault((e.match_id, e.possession_id), []).append(e.event_id)
            n += 1
        for bucket in self._by_possession.values():
            bucket.sort(key=lambda eid: self._events[eid].index)
        return n

    def upsert_frames(
        self, frames: Iterable[FrameRecord], metas: Iterable[FrameMeta]
    ) -> int:
        n = 0
        for f in frames:
            self._frames[f.event_id] = f
            n += 1
        for m in metas:
            self._metas[m.event_id] = m
        return n

    def upsert_outcomes(self, outcomes: Iterable[OutcomeRecord]) -> int:
        n = 0
        for o in outcomes:
            self._outcomes[o.event_id] = o
            n += 1
        return n

    def upsert_vectors(
        self, encoding_version: str, dim: int, rows: Sequence[tuple[str, np.ndarray]]
    ) -> int:
        ids, mat = self._vectors.get(encoding_version, ([], np.zeros((0, dim), np.float32)))
        index = {eid: i for i, eid in enumerate(ids)}
        new_ids = list(ids)
        blocks = [mat]
        extra: list[np.ndarray] = []
        for eid, vec in rows:
            v = np.asarray(vec, dtype=np.float32)
            if eid in index:
                mat[index[eid]] = v
            else:
                index[eid] = len(new_ids)
                new_ids.append(eid)
                extra.append(v)
        if extra:
            blocks.append(np.stack(extra))
        self._vectors[encoding_version] = (new_ids, np.concatenate(blocks, axis=0))
        return len(rows)

    def upsert_baselines(self, baselines: Iterable[ZoneBaseline]) -> int:
        rows = list(baselines)
        if rows:
            self._baselines[rows[0].encoding_version] = rows
        return len(rows)

    def upsert_catalogue(self, entries: Iterable[CatalogueEntry]) -> int:
        n = 0
        for e in entries:
            key = (e.kind, e.entity_id)
            existing = self._catalogue.get(key)
            if existing is None:
                self._catalogue[key] = e
            else:
                existing.frames += e.frames
                existing.aliases = sorted(set(existing.aliases) | set(e.aliases))
            n += 1
        return n

    def build_index(self, encoding_version: str, kind: str) -> dict[str, object]:
        ids, mat = self._vectors.get(encoding_version, ([], np.zeros((0, 0), np.float32)))
        return {
            "kind": "exact",
            "encoding_version": encoding_version,
            "vectors": len(ids),
            "bytes": int(mat.nbytes),
            "requested": kind,
        }

    # --- reads -------------------------------------------------------------
    def counts(self) -> Counts:
        return Counts(
            matches=len(self._matches),
            events=len(self._events),
            frames=len(self._frames),
            vectors=sum(len(ids) for ids, _ in self._vectors.values()),
            outcomes=len(self._outcomes),
            possessions=len({(f.match_id, f.possession_id) for f in self._frames.values()}),
        )

    def match(self, match_id: int) -> MatchRecord | None:
        return self._matches.get(match_id)

    def frame(self, event_id: str) -> FrameRecord | None:
        return self._frames.get(event_id)

    def event(self, event_id: str) -> EventRecord | None:
        return self._events.get(event_id)

    def meta(self, event_id: str) -> FrameMeta | None:
        return self._metas.get(event_id)

    def outcome(self, event_id: str) -> OutcomeRecord | None:
        return self._outcomes.get(event_id)

    def outcomes(self, event_ids: Sequence[str]) -> dict[str, OutcomeRecord]:
        return {e: self._outcomes[e] for e in event_ids if e in self._outcomes}

    def metas(self, event_ids: Sequence[str]) -> dict[str, FrameMeta]:
        return {e: self._metas[e] for e in event_ids if e in self._metas}

    def frames(self, limit: int | None = None) -> list[FrameRecord]:
        rows = list(self._frames.values())
        return rows[:limit] if limit else rows

    def possession_events(self, event_id: str) -> list[EventRecord]:
        ev = self._events.get(event_id)
        if ev is None:
            return []
        ids = self._by_possession.get((ev.match_id, ev.possession_id), [])
        return [self._events[i] for i in ids]

    def baselines(self, encoding_version: str) -> list[ZoneBaseline]:
        return self._baselines.get(encoding_version, [])

    def catalogue(self) -> list[CatalogueEntry]:
        return list(self._catalogue.values())

    def coverage(self) -> list[CoverageRow]:
        """Only competition/season pairs that actually have 360 frames."""

        frames: dict[tuple[int, int], int] = {}
        match_ids: dict[tuple[int, int], set[int]] = {}
        names: dict[tuple[int, int], tuple[str, str]] = {}
        for meta in self._metas.values():
            m = self._matches[meta.match_id]
            key = (m.competition_id, m.season_id)
            frames[key] = frames.get(key, 0) + 1
            match_ids.setdefault(key, set()).add(m.match_id)
            names[key] = (m.competition_name, m.season_name)

        rows: list[CoverageRow] = [
            {
                "competition_id": key[0],
                "competition_name": names[key][0],
                "season_id": key[1],
                "season_name": names[key][1],
                "frames": n,
                "matches": len(match_ids[key]),
            }
            for key, n in frames.items()
        ]
        return sorted(rows, key=lambda r: (-r["frames"], r["competition_name"]))

    def team_options(self) -> list[TeamRow]:
        counts: dict[int, int] = {}
        names: dict[int, str] = {}
        for meta in self._metas.values():
            m = self._matches[meta.match_id]
            for tid, name in ((m.home_team_id, m.home_team), (m.away_team_id, m.away_team)):
                names[tid] = name
            counts[meta.possession_team_id] = counts.get(meta.possession_team_id, 0) + 1
            counts[meta.opponent_team_id] = counts.get(meta.opponent_team_id, 0) + 1
        rows: list[TeamRow] = [
            {"team_id": tid, "team_name": names.get(tid, str(tid)), "frames": n}
            for tid, n in counts.items()
        ]
        return sorted(rows, key=lambda r: (-r["frames"], r["team_name"]))

    # --- search ------------------------------------------------------------
    def ann_search(
        self,
        encoding_version: str,
        vector: np.ndarray,
        limit: int,
        metric: str,
        filters: Filters | None = None,
    ) -> list[Candidate]:
        # No approximation here: the in-memory backend is exact by construction,
        # which is precisely why it is the recall reference.
        return self.exact_search(encoding_version, vector, limit, metric, filters)

    def exact_search(
        self,
        encoding_version: str,
        vector: np.ndarray,
        limit: int,
        metric: str,
        filters: Filters | None = None,
    ) -> list[Candidate]:
        entry = self._vectors.get(encoding_version)
        if entry is None:
            return []
        ids, mat = entry
        if not ids:
            return []
        keep = np.ones(len(ids), dtype=bool)
        if filters is not None and not filters.is_empty():
            for i, eid in enumerate(ids):
                meta = self._metas.get(eid)
                keep[i] = meta is not None and filter_matches(meta, filters)
        if not keep.any():
            return []
        idx = np.flatnonzero(keep)
        d = _distances(mat[idx], np.asarray(vector, dtype=np.float32), metric)
        order = np.argsort(d, kind="stable")[:limit]
        return [Candidate(ids[int(idx[i])], float(d[int(i)])) for i in order]

    def count_matching(self, encoding_version: str, filters: Filters | None) -> int:
        entry = self._vectors.get(encoding_version)
        if entry is None:
            return 0
        ids, _ = entry
        if filters is None or filters.is_empty():
            return len(ids)
        return sum(
            1
            for eid in ids
            if (meta := self._metas.get(eid)) is not None and filter_matches(meta, filters)
        )

    def close(self) -> None:
        return None

    # --- persistence for the offline demo ----------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "matches": self._matches,
                    "events": self._events,
                    "by_possession": self._by_possession,
                    "frames": self._frames,
                    "metas": self._metas,
                    "outcomes": self._outcomes,
                    "baselines": self._baselines,
                    "catalogue": self._catalogue,
                    "vectors": self._vectors,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: Path) -> MemoryStore:
        store = cls()
        with path.open("rb") as fh:
            blob = pickle.load(fh)
        store._matches = blob["matches"]
        store._events = blob["events"]
        store._by_possession = blob["by_possession"]
        store._frames = blob["frames"]
        store._metas = blob["metas"]
        store._outcomes = blob["outcomes"]
        store._baselines = blob["baselines"]
        store._catalogue = blob["catalogue"]
        store._vectors = blob["vectors"]
        return store
