"""The offline batch side: ingest, label, encode, index.

Idempotent by construction: every write is an upsert keyed on the natural key, so
rerunning after a failure converges on the same database instead of duplicating.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pitchscanner.baselines import compute_zone_baselines
from pitchscanner.config import Settings, get_settings
from pitchscanner.encoders import available, encode_many, get_encoder
from pitchscanner.ingest.catalogue import build_catalogue
from pitchscanner.ingest.normalize import frame_meta, normalise_frame
from pitchscanner.ingest.parse import (
    Reject,
    parse_events,
    parse_frames,
    parse_matches,
)
from pitchscanner.labeller import label_match
from pitchscanner.logging_setup import get_logger
from pitchscanner.models import EventRecord, FrameRecord, MatchRecord
from pitchscanner.store.base import FrameMeta, Store

log = get_logger(__name__)


@dataclass(slots=True)
class IngestReport:
    matches: int = 0
    events: int = 0
    frames: int = 0
    outcomes: int = 0
    vectors: dict[str, int] = field(default_factory=dict)
    rejects: list[Reject] = field(default_factory=list)
    seconds: float = 0.0
    indexes: list[dict[str, object]] = field(default_factory=list)

    @property
    def frames_per_second(self) -> float:
        return round(self.frames / self.seconds, 1) if self.seconds else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "matches": self.matches,
            "events": self.events,
            "frames": self.frames,
            "outcomes": self.outcomes,
            "vectors": self.vectors,
            "rejects": len(self.rejects),
            "seconds": round(self.seconds, 2),
            "frames_per_second": self.frames_per_second,
            "indexes": self.indexes,
        }


def _match_files(data_dir: Path) -> list[tuple[Path, int, int]]:
    """(matches file, competition_id, season_id) for everything on disk."""

    out: list[tuple[Path, int, int]] = []
    matches_dir = data_dir / "matches"
    if not matches_dir.exists():
        return out
    for comp_dir in sorted(matches_dir.iterdir()):
        if not comp_dir.is_dir():
            continue
        for season_file in sorted(comp_dir.glob("*.json")):
            try:
                out.append((season_file, int(comp_dir.name), int(season_file.stem)))
            except ValueError:
                continue
    return out


def ingest(
    data_dir: Path,
    store: Store,
    *,
    settings: Settings | None = None,
    encodings: Sequence[str] | None = None,
    limit_matches: int | None = None,
    only_with_frames: bool = True,
) -> IngestReport:
    """Run the whole offline side over a data directory."""

    cfg = settings or get_settings()
    started = time.perf_counter()
    report = IngestReport()
    store.init_schema()

    all_matches: list[MatchRecord] = []
    all_events: list[EventRecord] = []
    all_frames: list[FrameRecord] = []
    all_metas: list[FrameMeta] = []

    for season_file, competition_id, season_id in _match_files(data_dir):
        matches, rejects = parse_matches(
            season_file, competition_id=competition_id, season_id=season_id
        )
        report.rejects.extend(rejects)
        for match in matches:
            events_path = data_dir / "events" / f"{match.match_id}.json"
            frames_path = data_dir / "three-sixty" / f"{match.match_id}.json"
            if not events_path.exists():
                continue
            if only_with_frames and not frames_path.exists():
                continue
            if limit_matches is not None and report.matches >= limit_matches:
                break

            events, goals, ev_rejects = parse_events(events_path, match.match_id)
            report.rejects.extend(ev_rejects)
            events_by_id = {e.event_id: e for e in events}

            frames: list[FrameRecord] = []
            if frames_path.exists():
                frames, fr_rejects = parse_frames(frames_path, match.match_id, events_by_id)
                report.rejects.extend(fr_rejects)

            # Normalise, then optionally add the mirrored copy. The mirror is a
            # real row with its own event_id suffix so dedup and outcomes still
            # resolve to the original event.
            normalised: list[FrameRecord] = []
            metas: list[FrameMeta] = []
            for frame in frames:
                event = events_by_id[frame.event_id]
                nf = normalise_frame(frame)
                normalised.append(nf)
                metas.append(frame_meta(nf, event, match, goals))

            outcomes = label_match(normalised, events, window_s=cfg.outcome_window_s)

            store.upsert_matches([match])
            store.upsert_events(events)
            store.upsert_frames(normalised, metas)
            store.upsert_outcomes(outcomes)

            report.matches += 1
            report.events += len(events)
            report.frames += len(normalised)
            report.outcomes += len(outcomes)
            all_matches.append(match)
            all_events.extend(events)
            all_frames.extend(normalised)
            all_metas.extend(metas)
            log.info(
                "match_ingested",
                match_id=match.match_id,
                events=len(events),
                frames=len(normalised),
            )
        if limit_matches is not None and report.matches >= limit_matches:
            break

    # --- derive -----------------------------------------------------------
    encodings = tuple(encodings or available())
    for encoding in encodings:
        encoder = get_encoder(encoding)
        situations = [f.to_situation() for f in all_frames]
        rows: list[tuple[str, np.ndarray]] = []
        if situations:
            vectors = encode_many(encoder, situations)
            rows = [(f.event_id, vectors[i]) for i, f in enumerate(all_frames)]
        # Mirroring is applied to the *query* instead of the index: encoding the
        # sketch twice costs microseconds, while a mirrored copy of every frame
        # would double index size and build time for the same reach. See
        # docs/design-notes.md.
        n = store.upsert_vectors(encoder.version, encoder.dim, rows)
        report.vectors[encoder.version] = n

        outcome_map = store.outcomes([f.event_id for f in all_frames])
        store.upsert_baselines(
            compute_zone_baselines(
                all_frames, outcome_map, encoding_version=encoder.version
            )
        )
        report.indexes.append(store.build_index(encoder.version, cfg.index_kind))

    store.upsert_catalogue(
        build_catalogue(all_matches, all_events, all_frames, {m.event_id: m for m in all_metas})
    )

    report.seconds = time.perf_counter() - started
    log.info("ingest_complete", **report.as_dict())
    return report


def write_reject_log(rejects: Iterable[Reject], path: Path) -> int:
    """Rejects go to a file, not to a crash."""

    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rejects:
            fh.write(json.dumps({"source": r.source, "reason": r.reason}) + "\n")
            count += 1
    return count
