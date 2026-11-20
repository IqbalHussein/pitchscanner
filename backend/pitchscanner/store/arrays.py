"""A database-free store: memory-mapped numpy arrays plus one small file per match.

All of StatsBomb's 360 data is about 1.3M frames. As pydantic objects that is many
GB, and Postgres does not fit on any free host, but the *searchable* part of a
frame is small: a vector, a dozen small integers and an outcome. This store keeps
exactly that in ``.npy`` files, opened with ``mmap_mode="r"``, and moves everything
bulky (players, camera polygon, events) into ``detail/<match_id>.json.gz``, which
only the situation and sequence endpoints read.

Layout of a store directory::

    meta.json           counts, versions, dtypes, vocabularies, provenance (written last)
    matches.json        every MatchRecord
    catalogue.json      the entity catalogue
    baselines.json      zone baselines per encoding version
    columns/<name>.npy  one packed column per frame row (FrameMeta fields and more)
    vectors/<ver>.npy   (rows, dim) float32 or float16, one file per encoding
    ids/*.npy           sorted event_id table: UUID bytes, frame row, match index
    detail/<id>.json.gz players, camera polygon, ball and events for one match

Write mode (``ArrayStore.create``) is what ingest talks to: each ``upsert_*`` call
appends one match's data to column buffers, so nothing per frame stays alive as an
object. ``finalize()`` writes the files. Read mode (``ArrayStore.open``) is what the
API talks to.

Search is exact. A boolean mask is built from the packed columns by
``filtering.to_mask`` (the vectorised twin of ``filtering.matches``), then distances
are computed chunk by chunk over the masked rows only and reduced with
``argpartition``. ``ann_search`` is therefore the same exact scan, and
``effective_index_kind`` is ``"exact"``.
"""

from __future__ import annotations

import gzip
import json
import shutil
import threading
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from pitchscanner.config import Settings
from pitchscanner.geometry import PlayerPos, Point
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
from pitchscanner.store.filtering import to_mask

FORMAT_VERSION = 1

#: Rows scanned per distance computation. Float16 vectors are upcast one chunk at
#: a time, so peak extra memory is CHUNK x dim x 4 bytes (37 MB for E1, 153 MB for E0).
CHUNK = 131_072

DEFAULT_DTYPES = {"E0": "float16", "E1": "float32", "E2": "float16"}
VECTOR_DTYPES = ("float32", "float16")

#: Per-frame columns, in the order they are listed in meta.json.
_FRAME_COLUMNS = (
    "event_index",
    "match_id",
    "possession_id",
    "competition_id",
    "season_id",
    "possession_team_id",
    "opponent_team_id",
    "player_id",
    "minute",
    "period",
    "stage",
    "game_state",
    "zone_id",
    "possession_is_home",
    "mirrored",
    "outcome_bits",
    "xg_next",
    "seconds_to_end",
)
_CATEGORICAL = ("stage", "game_state")

# outcome_bits layout
_HAS, _SHOT, _GOAL, _TURNOVER, _BOX = 1, 2, 4, 8, 16

_EVENT_FIELDS = (
    "event_id",
    "possession_id",
    "index",
    "team_id",
    "player_id",
    "player_name",
    "type",
    "x",
    "y",
    "ts",
    "period",
    "minute",
    "xg",
    "goal",
    "possession_team_id",
)


class DetailUnavailableError(RuntimeError):
    """A match's detail file could not be read (missing file, HTTP error, timeout)."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def parse_dtype_spec(spec: str) -> dict[str, str]:
    """``"E1=float32,E2=float16"`` -> ``{"E1": "float32", "E2": "float16"}``, over the defaults."""

    out = dict(DEFAULT_DTYPES)
    for part in filter(None, (p.strip() for p in spec.split(","))):
        key, _, value = part.partition("=")
        key, value = key.strip().upper(), value.strip().lower()
        if value not in VECTOR_DTYPES:
            raise ValueError(f"vector dtype for {key or '?'} must be one of {VECTOR_DTYPES}")
        out[key] = value
    return out


def _encode_id(event_id: str) -> bytes:
    """A canonical lower-case UUID string as its 16 bytes."""

    if (
        len(event_id) == 36
        and event_id == event_id.lower()
        and event_id[8] == event_id[13] == event_id[18] == event_id[23] == "-"
    ):
        try:
            raw = bytes.fromhex(event_id.replace("-", ""))
        except ValueError:
            raw = b""
        if len(raw) == 16:
            return raw
    raise ValueError(f"the array store needs canonical lower-case UUID event ids, got {event_id!r}")


def _try_encode_id(event_id: str) -> bytes | None:
    try:
        return _encode_id(event_id)
    except ValueError:
        return None


def _decode_ids(keys: np.ndarray) -> list[str]:
    """S16 array -> UUID strings. Goes through raw bytes: numpy drops trailing NULs."""

    raw = np.ascontiguousarray(keys).tobytes()
    out = []
    for i in range(0, len(raw), 16):
        h = raw[i : i + 16].hex()
        out.append(f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}")
    return out


def _pack(values: np.ndarray) -> np.ndarray:
    """The smallest signed integer dtype that holds every value."""

    if values.dtype.kind not in "iu":
        return values
    if values.size == 0:
        return values.astype(np.int8)
    lo, hi = int(values.min()), int(values.max())
    for dt in (np.int8, np.int16, np.int32):
        info = np.iinfo(dt)
        if info.min <= lo and hi <= info.max:
            return values.astype(dt)
    return values.astype(np.int64)


def _metric_of(encoding_version: str) -> str:
    from pitchscanner.encoders import get_encoder

    try:
        return str(get_encoder(encoding_version.split("-")[0]).metric)
    except KeyError:
        return "cosine"


def _frame_payload(frame: FrameRecord) -> dict[str, Any]:
    return {
        "id": frame.event_id,
        "p": frame.possession_id,
        "b": [frame.ball_x, frame.ball_y],
        "m": 1 if frame.mirrored else 0,
        "pl": [[p.x, p.y, int(p.teammate), int(p.keeper)] for p in frame.players],
        "va": [[q.x, q.y] for q in frame.visible_area] if frame.visible_area else None,
    }


def _frame_from_payload(match_id: int, raw: Mapping[str, Any]) -> FrameRecord:
    va = raw["va"]
    return FrameRecord(
        event_id=raw["id"],
        match_id=match_id,
        possession_id=raw["p"],
        players=tuple(PlayerPos(x, y, bool(t), bool(k)) for x, y, t, k in raw["pl"]),
        visible_area=tuple(Point(x, y) for x, y in va) if va else None,
        ball_x=raw["b"][0],
        ball_y=raw["b"][1],
        mirrored=bool(raw["m"]),
    )


# --------------------------------------------------------------------------
# detail files
# --------------------------------------------------------------------------
class _Detail:
    """One match's events and frames, parsed from its detail file."""

    def __init__(self, match_id: int, payload: Mapping[str, Any]) -> None:
        self.match_id = match_id
        self._event_rows: list[list[Any]] = payload["events"]
        self._frames: dict[str, Mapping[str, Any]] = {f["id"]: f for f in payload["frames"]}
        self._frame_order: list[str] = [f["id"] for f in payload["frames"]]
        self._by_id: dict[str, int] | None = None

    def _event(self, row: Sequence[Any]) -> EventRecord:
        values = dict(zip(_EVENT_FIELDS, row, strict=True))
        values["match_id"] = self.match_id
        return EventRecord.model_construct(**values)

    def event(self, event_id: str) -> EventRecord | None:
        if self._by_id is None:
            self._by_id = {row[0]: i for i, row in enumerate(self._event_rows)}
        i = self._by_id.get(event_id)
        return None if i is None else self._event(self._event_rows[i])

    def possession_events(self, possession_id: int) -> list[EventRecord]:
        rows = [r for r in self._event_rows if r[1] == possession_id]
        # Same order the in-memory store gives: by index, ties in stored order.
        rows.sort(key=lambda r: r[2])
        return [self._event(r) for r in rows]

    def frame(self, event_id: str) -> FrameRecord | None:
        raw = self._frames.get(event_id)
        return None if raw is None else _frame_from_payload(self.match_id, raw)

    def frames(self) -> list[FrameRecord]:
        return [_frame_from_payload(self.match_id, self._frames[i]) for i in self._frame_order]


class _DetailSource:
    """Where ``<match_id>.json.gz`` files come from: a directory, or an http(s) base URL."""

    def __init__(
        self, directory: Path | None, base_url: str | None, timeout: float, cache_size: int
    ) -> None:
        if base_url is not None:
            scheme = urllib.parse.urlparse(base_url).scheme
            if scheme not in ("http", "https"):
                raise ValueError(f"detail base URL must be http(s), got {base_url!r}")
        self._dir = directory
        self._base = base_url.rstrip("/") if base_url else None
        self._timeout = timeout
        self._cache: OrderedDict[int, _Detail] = OrderedDict()
        self._size = max(1, cache_size)
        self._lock = threading.Lock()

    def _fetch(self, match_id: int) -> bytes:
        name = f"{int(match_id)}.json.gz"
        if self._base is not None:
            url = f"{self._base}/{name}"
            req = urllib.request.Request(
                url, headers={"Accept-Encoding": "identity", "User-Agent": "pitchscanner"}
            )
            try:
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    body: bytes = resp.read()
                    return body
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise DetailUnavailableError(f"cannot fetch {url}: {exc}") from exc
        assert self._dir is not None
        path = self._dir / name
        try:
            return path.read_bytes()
        except OSError as exc:
            raise DetailUnavailableError(f"cannot read {path}: {exc}") from exc

    def get(self, match_id: int) -> _Detail:
        with self._lock:
            hit = self._cache.get(match_id)
            if hit is not None:
                self._cache.move_to_end(match_id)
                return hit
        body = self._fetch(match_id)
        # A server (or CDN) may already have undone the gzip; accept either.
        if body[:2] == b"\x1f\x8b":
            try:
                body = gzip.decompress(body)
            except (OSError, EOFError) as exc:
                raise DetailUnavailableError(f"corrupt detail file for match {match_id}") from exc
        try:
            detail = _Detail(match_id, orjson.loads(body))
        except (orjson.JSONDecodeError, KeyError, TypeError) as exc:
            raise DetailUnavailableError(f"unreadable detail file for match {match_id}") from exc
        with self._lock:
            self._cache[match_id] = detail
            while len(self._cache) > self._size:
                self._cache.popitem(last=False)
        return detail

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------
@dataclass(slots=True)
class _VectorFile:
    array: np.ndarray
    dim: int
    unit_norm: bool
    dtype: str


class ArrayStore:
    #: ``ingest`` hands this store each match's vectors as they are encoded.
    streaming_vectors = True

    def __init__(self) -> None:
        self._writing = False
        self._dir: Path | None = None
        # --- read mode ---
        self._meta: dict[str, Any] = {}
        self._cols: dict[str, np.ndarray] = {}
        self._vocab: dict[str, list[str]] = {}
        self._ids: np.ndarray | None = None
        self._id_frame_row: np.ndarray | None = None
        self._id_match_ix: np.ndarray | None = None
        self._row_pos: np.ndarray | None = None
        self._match_ids: list[int] = []
        self._vectors: dict[str, _VectorFile] = {}
        self._matches: dict[int, MatchRecord] = {}
        self._catalogue: list[CatalogueEntry] = []
        self._baselines: dict[str, list[ZoneBaseline]] = {}
        self._details: _DetailSource | None = None
        self._mask_cache: OrderedDict[str, np.ndarray | None] = OrderedDict()
        self._mask_lock = threading.Lock()
        self._derived: dict[str, Any] = {}
        # --- write mode ---
        self._w_dtypes: dict[str, str] = dict(DEFAULT_DTYPES)
        self._w_matches: dict[int, MatchRecord] = {}
        self._w_match_order: list[int] = []
        self._w_events: dict[int, list[EventRecord]] = {}
        self._w_frames: dict[int, dict[str, tuple[FrameRecord, FrameMeta]]] = {}
        self._w_outcomes: dict[str, OutcomeRecord] = {}
        self._w_flushed: set[int] = set()
        self._w_cols: dict[str, list[np.ndarray]] = {c: [] for c in _FRAME_COLUMNS}
        self._w_vocab: dict[str, list[str]] = {c: [] for c in _CATEGORICAL}
        self._w_id_keys: list[bytes] = []
        self._w_id_frame_row: list[np.ndarray] = []
        self._w_id_match_ix: list[np.ndarray] = []
        self._w_rows = 0
        self._w_event_count = 0
        self._w_vec_ids: dict[str, list[bytes]] = {}
        self._w_vec_data: dict[str, list[np.ndarray]] = {}
        self._w_vec_info: dict[str, dict[str, Any]] = {}
        self._w_baselines: dict[str, list[ZoneBaseline]] = {}
        self._w_catalogue: dict[tuple[str, int], CatalogueEntry] = {}

    # ===================================================================== write
    @classmethod
    def create(
        cls, out_dir: Path | str, *, dtypes: Mapping[str, str] | None = None
    ) -> ArrayStore:
        """A store that ingest writes into. ``dtypes`` maps encoding id to float32/float16."""

        # Validate everything before touching the filesystem: a bad spec must create nothing.
        merged = dict(DEFAULT_DTYPES)
        merged.update({k.upper(): v for k, v in (dtypes or {}).items()})
        bad = {v for v in merged.values() if v not in VECTOR_DTYPES}
        if bad:
            raise ValueError(f"vector dtype must be one of {VECTOR_DTYPES}, got {sorted(bad)}")
        store = cls()
        d = Path(out_dir)
        if d.exists() and any(d.iterdir()):
            if not (d / "meta.json").exists():
                raise ValueError(
                    f"{d} is not empty and is not an array store; refusing to write into it"
                )
            for name in ("columns", "vectors", "ids", "detail"):
                shutil.rmtree(d / name, ignore_errors=True)
            for name in ("meta.json", "matches.json", "catalogue.json", "baselines.json"):
                (d / name).unlink(missing_ok=True)
        (d / "detail").mkdir(parents=True, exist_ok=True)
        store._writing = True
        store._dir = d
        store._w_dtypes = merged
        return store

    def _need_write(self) -> None:
        if not self._writing:
            raise RuntimeError("this ArrayStore is open read-only")

    def _need_read(self) -> None:
        if self._writing:
            raise RuntimeError("this ArrayStore is still being written; call finalize() first")

    def init_schema(self) -> None:
        return None

    def upsert_matches(self, matches: Iterable[MatchRecord]) -> int:
        self._need_write()
        self._flush_all()
        n = 0
        for m in matches:
            if m.match_id not in self._w_matches:
                self._w_match_order.append(m.match_id)
            self._w_matches[m.match_id] = m
            n += 1
        return n

    def upsert_events(self, events: Iterable[EventRecord]) -> int:
        self._need_write()
        n = 0
        for e in events:
            if e.match_id in self._w_flushed:
                raise ValueError(f"match {e.match_id} is already written; events are append-only")
            self._w_events.setdefault(e.match_id, []).append(e)
            n += 1
        return n

    def upsert_frames(self, frames: Iterable[FrameRecord], metas: Iterable[FrameMeta]) -> int:
        self._need_write()
        frame_list = list(frames)
        meta_by_id = {m.event_id: m for m in metas}
        for f in frame_list:
            if f.match_id in self._w_flushed:
                raise ValueError(f"match {f.match_id} is already written; frames are append-only")
            meta = meta_by_id.get(f.event_id)
            if meta is None:
                raise ValueError(f"frame {f.event_id} has no FrameMeta")
            _encode_id(f.event_id)
            # Same event_id twice in one match: the later one wins, as in the other stores.
            self._w_frames.setdefault(f.match_id, {})[f.event_id] = (f, meta)
        return len(frame_list)

    def upsert_outcomes(self, outcomes: Iterable[OutcomeRecord]) -> int:
        self._need_write()
        n = 0
        for o in outcomes:
            _encode_id(o.event_id)
            self._w_outcomes[o.event_id] = o
            n += 1
        return n

    def upsert_vectors(
        self, encoding_version: str, dim: int, rows: Sequence[tuple[str, np.ndarray]]
    ) -> int:
        self._need_write()
        if not rows:
            return 0
        mat = np.stack([np.asarray(v, dtype=np.float32) for _, v in rows])
        if mat.shape[1] != dim:
            raise ValueError(f"expected {dim}-d vectors for {encoding_version}, got {mat.shape[1]}")
        info = self._w_vec_info.setdefault(
            encoding_version,
            {
                "dim": dim,
                "metric": _metric_of(encoding_version),
                "dtype": self._w_dtypes.get(encoding_version.split("-")[0].upper(), "float32"),
            },
        )
        if info["dim"] != dim:
            raise ValueError(f"{encoding_version} already has {info['dim']}-d vectors")
        if info["metric"] == "cosine":
            # Unit rows turn cosine distance into 1 - dot/|q|, so a query never
            # recomputes row norms. A zero row stays zero (distance 1, as before).
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            np.divide(mat, norms, out=mat, where=norms > 1e-9)
        self._w_vec_ids.setdefault(encoding_version, []).append(
            b"".join(_encode_id(eid) for eid, _ in rows)
        )
        self._w_vec_data.setdefault(encoding_version, []).append(mat.astype(info["dtype"]))
        return len(rows)

    def upsert_baselines(self, baselines: Iterable[ZoneBaseline]) -> int:
        self._need_write()
        rows = list(baselines)
        if rows:
            self._w_baselines[rows[0].encoding_version] = rows
        return len(rows)

    def upsert_catalogue(self, entries: Iterable[CatalogueEntry]) -> int:
        self._need_write()
        n = 0
        for e in entries:
            key = (e.kind, e.entity_id)
            existing = self._w_catalogue.get(key)
            if existing is None:
                self._w_catalogue[key] = e
            else:
                existing.frames += e.frames
                existing.aliases = sorted(set(existing.aliases) | set(e.aliases))
            n += 1
        return n

    def build_index(self, encoding_version: str, kind: str) -> dict[str, object]:
        if self._writing:
            chunks = self._w_vec_data.get(encoding_version, [])
            vectors, nbytes = sum(len(c) for c in chunks), sum(c.nbytes for c in chunks)
        else:
            vf = self._vectors.get(encoding_version)
            vectors, nbytes = (len(vf.array), int(vf.array.nbytes)) if vf else (0, 0)
        return {
            "kind": "exact",
            "encoding_version": encoding_version,
            "vectors": vectors,
            "bytes": nbytes,
            "requested": kind,
        }

    # -- flushing one match's buffered rows into the column buffers ----------
    def _code(self, column: str, value: str) -> int:
        vocab = self._w_vocab[column]
        if value not in vocab:
            if len(vocab) >= 127:
                raise ValueError(f"too many distinct {column} values for the array store")
            vocab.append(value)
        return vocab.index(value)

    def _flush_all(self) -> None:
        pending = sorted(set(self._w_frames) | set(self._w_events))
        for match_id in pending:
            self._flush_match(match_id)

    def _flush_match(self, match_id: int) -> None:
        assert self._dir is not None
        if match_id not in self._w_matches:
            raise ValueError(f"match {match_id} has events or frames but was never upserted")
        events = self._w_events.pop(match_id, [])
        framed = self._w_frames.pop(match_id, {})
        self._w_flushed.add(match_id)
        match_ix = self._w_match_order.index(match_id)
        event_index = {e.event_id: e.index for e in events}

        cols: dict[str, list[Any]] = {c: [] for c in _FRAME_COLUMNS}
        frame_payloads = []
        base = self._w_rows
        row_of: dict[str, int] = {}
        for i, (frame, meta) in enumerate(framed.values()):
            row_of[frame.event_id] = base + i
            out = self._w_outcomes.pop(frame.event_id, None)
            bits = 0
            xg = secs = 0.0
            if out is not None:
                bits = (
                    _HAS
                    | (_SHOT if out.shot else 0)
                    | (_GOAL if out.goal else 0)
                    | (_TURNOVER if out.turnover else 0)
                    | (_BOX if out.box_entry else 0)
                )
                xg, secs = out.xg_next, out.seconds_to_end
            cols["event_index"].append(event_index.get(frame.event_id, -1))
            cols["match_id"].append(meta.match_id)
            cols["possession_id"].append(meta.possession_id)
            cols["competition_id"].append(meta.competition_id)
            cols["season_id"].append(meta.season_id)
            cols["possession_team_id"].append(meta.possession_team_id)
            cols["opponent_team_id"].append(meta.opponent_team_id)
            cols["player_id"].append(-1 if meta.player_id is None else meta.player_id)
            cols["minute"].append(meta.minute)
            cols["period"].append(meta.period)
            cols["stage"].append(self._code("stage", meta.stage))
            cols["game_state"].append(self._code("game_state", meta.game_state))
            cols["zone_id"].append(meta.zone_id)
            cols["possession_is_home"].append(meta.possession_is_home)
            cols["mirrored"].append(frame.mirrored)
            cols["outcome_bits"].append(bits)
            cols["xg_next"].append(xg)
            cols["seconds_to_end"].append(secs)
            frame_payloads.append(_frame_payload(frame))
        n = len(framed)
        for name in _FRAME_COLUMNS:
            if name in ("possession_is_home", "mirrored"):
                arr = np.asarray(cols[name], dtype=bool)
            elif name in ("xg_next", "seconds_to_end"):
                arr = np.asarray(cols[name], dtype=np.float64)
            elif name == "outcome_bits":
                arr = np.asarray(cols[name], dtype=np.uint8)
            else:
                arr = np.asarray(cols[name], dtype=np.int64)
            self._w_cols[name].append(arr)
        self._w_rows += n

        # The id table covers every event (so the sequence endpoint can start from
        # any of them) and any frame whose event is missing.
        seen: set[str] = set()
        keys: list[bytes] = []
        frame_rows: list[int] = []
        for e in events:
            if e.event_id in seen:
                continue
            seen.add(e.event_id)
            keys.append(_encode_id(e.event_id))
            frame_rows.append(row_of.get(e.event_id, -1))
        for eid, row in row_of.items():
            if eid not in seen:
                keys.append(_encode_id(eid))
                frame_rows.append(row)
        self._w_event_count += len(seen)
        self._w_id_keys.append(b"".join(keys))
        self._w_id_frame_row.append(np.asarray(frame_rows, dtype=np.int64))
        self._w_id_match_ix.append(np.full(len(keys), match_ix, dtype=np.int64))

        payload = {
            "v": FORMAT_VERSION,
            "match_id": match_id,
            "event_fields": list(_EVENT_FIELDS),
            "events": [[getattr(e, f) for f in _EVENT_FIELDS] for e in events],
            "frames": frame_payloads,
        }
        blob = gzip.compress(orjson.dumps(payload), compresslevel=6, mtime=0)
        (self._dir / "detail" / f"{int(match_id)}.json.gz").write_bytes(blob)

    # -- finalize -------------------------------------------------------------
    def finalize(
        self, out_dir: Path | str | None = None, *, extra_meta: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Write every file. Returns a summary including sizes in bytes."""

        self._need_write()
        assert self._dir is not None
        d = self._dir
        if out_dir is not None and Path(out_dir) != d:
            raise ValueError(f"this store is writing to {d}; create() it at {out_dir} instead")
        self._flush_all()
        if self._w_outcomes:
            raise ValueError(
                f"{len(self._w_outcomes)} outcomes have no frame (e.g. "
                f"{next(iter(self._w_outcomes))})"
            )

        for sub in ("columns", "vectors", "ids"):
            (d / sub).mkdir(exist_ok=True)
        n_rows = self._w_rows

        columns_meta: dict[str, str] = {}
        for name in _FRAME_COLUMNS:
            parts = self._w_cols[name]
            arr = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
            arr = _pack(arr)
            np.save(d / "columns" / f"{name}.npy", arr)
            columns_meta[name] = arr.dtype.name
            self._w_cols[name] = []

        # --- id table: sorted UUID bytes -> frame row / match -------------------
        raw = b"".join(self._w_id_keys)
        keys = np.frombuffer(raw, dtype="S16") if raw else np.zeros(0, dtype="S16")
        order = np.argsort(keys, kind="stable")
        sorted_keys = np.ascontiguousarray(keys[order])
        if len(sorted_keys) > 1 and bool((sorted_keys[1:] == sorted_keys[:-1]).any()):
            raise ValueError("duplicate event_id across matches")
        frame_row = (
            np.concatenate(self._w_id_frame_row) if self._w_id_frame_row else np.zeros(0, np.int64)
        )[order]
        match_ix = (
            np.concatenate(self._w_id_match_ix) if self._w_id_match_ix else np.zeros(0, np.int64)
        )[order]
        row_pos = np.full(n_rows, -1, dtype=np.int64)
        framed = np.flatnonzero(frame_row >= 0)
        row_pos[frame_row[framed]] = framed
        if n_rows and bool((row_pos < 0).any()):
            raise ValueError("internal error: a frame row has no id")
        np.save(d / "ids" / "sorted.npy", sorted_keys)
        np.save(d / "ids" / "frame_row.npy", _pack(frame_row))
        np.save(d / "ids" / "match_ix.npy", _pack(match_ix))
        np.save(d / "ids" / "row_pos.npy", _pack(row_pos))
        n_events = self._w_event_count

        # --- vectors: placed by event id, so any upsert order works -------------
        encodings: dict[str, Any] = {}
        for version, chunks in self._w_vec_data.items():
            info = self._w_vec_info[version]
            out = np.lib.format.open_memmap(
                d / "vectors" / f"{version}.npy",
                mode="w+",
                dtype=np.dtype(info["dtype"]),
                shape=(n_rows, info["dim"]),
            )
            have = np.zeros(n_rows, dtype=bool)
            for ids_blob, mat in zip(self._w_vec_ids[version], chunks, strict=True):
                vec_keys = np.frombuffer(ids_blob, dtype="S16")
                pos = np.searchsorted(sorted_keys, vec_keys)
                pos = np.minimum(pos, max(len(sorted_keys) - 1, 0))
                found = sorted_keys[pos] == vec_keys if len(sorted_keys) else np.zeros(
                    len(vec_keys), dtype=bool
                )
                rows = frame_row[pos]
                if not bool(found.all()) or bool((rows < 0).any()):
                    raise ValueError(f"{version} has vectors for ids that are not frames")
                out[rows] = mat  # a later vector for the same id overwrites the earlier
                have[rows] = True
            if not bool(have.all()):
                raise ValueError(
                    f"{version} has vectors for {int(have.sum())} of {n_rows} frames; "
                    "every frame needs a vector in every stored encoding"
                )
            out.flush()
            del out
            encodings[version] = {
                "file": f"vectors/{version}.npy",
                "dim": info["dim"],
                "dtype": info["dtype"],
                "metric": info["metric"],
                "unit_norm": info["metric"] == "cosine",
                "rows": n_rows,
            }
        self._w_vec_data.clear()

        # --- json ------------------------------------------------------------
        matches = [self._w_matches[i].model_dump(mode="json") for i in self._w_match_order]
        (d / "matches.json").write_bytes(orjson.dumps(matches))
        cat = [
            {
                "kind": e.kind,
                "entity_id": e.entity_id,
                "name": e.name,
                "aliases": e.aliases,
                "parent_id": e.parent_id,
                "frames": e.frames,
            }
            for e in self._w_catalogue.values()
        ]
        (d / "catalogue.json").write_bytes(orjson.dumps(cat))
        base = {v: [b.model_dump() for b in rows] for v, rows in self._w_baselines.items()}
        (d / "baselines.json").write_bytes(orjson.dumps(base))

        match_col = np.load(d / "columns" / "match_id.npy")
        poss_col = np.load(d / "columns" / "possession_id.npy")
        possessions = (
            len(np.unique(match_col.astype(np.int64) * (1 << 32) + poss_col.astype(np.int64)))
            if n_rows
            else 0
        )
        bits = np.load(d / "columns" / "outcome_bits.npy")
        meta: dict[str, Any] = {
            "format": FORMAT_VERSION,
            "counts": {
                "matches": len(self._w_match_order),
                "events": n_events,
                "frames": n_rows,
                "outcomes": int(((bits & _HAS) != 0).sum()),
                "possessions": possessions,
                "vectors": {v: int(e["rows"]) for v, e in encodings.items()},
            },
            "encodings": encodings,
            "columns": columns_meta,
            "vocab": self._w_vocab,
            "match_ids": self._w_match_order,
            "ids": {"format": "uuid16", "dtype": "S16"},
            "detail": {"pattern": "detail/{match_id}.json.gz", "format": "orjson+gzip"},
            **dict(extra_meta or {}),
        }
        (d / "meta.json").write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
        self._writing = False
        return self.summary(d)

    @staticmethod
    def summary(directory: Path | str) -> dict[str, Any]:
        """File sizes for a finished store, in bytes, and the per-frame cost."""

        d = Path(directory)
        sizes: dict[str, int] = {}
        for sub in ("columns", "vectors", "ids"):
            for f in sorted((d / sub).glob("*.npy")):
                sizes[f"{sub}/{f.name}"] = f.stat().st_size
        for name in ("meta.json", "matches.json", "catalogue.json", "baselines.json"):
            sizes[name] = (d / name).stat().st_size
        detail = sum(f.stat().st_size for f in (d / "detail").glob("*.json.gz"))
        meta = json.loads((d / "meta.json").read_text())
        frames = int(meta["counts"]["frames"])
        searchable = sum(sizes.values())
        return {
            "path": str(d),
            "frames": frames,
            "events": int(meta["counts"]["events"]),
            "searchable_bytes": searchable,
            "detail_bytes": detail,
            "searchable_bytes_per_frame": round(searchable / frames, 2) if frames else 0.0,
            "detail_bytes_per_frame": round(detail / frames, 2) if frames else 0.0,
            "files": sizes,
        }

    # ====================================================================== read
    @classmethod
    def open(
        cls,
        directory: Path | str,
        *,
        detail_dir: Path | str | None = None,
        detail_base_url: str | None = None,
        detail_timeout_s: float = 10.0,
        detail_cache: int = 6,
    ) -> ArrayStore:
        d = Path(directory)
        meta_path = d / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{d} is not an array store (no meta.json)")
        store = cls()
        store._dir = d
        meta = json.loads(meta_path.read_text())
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(f"unsupported array store format {meta.get('format')!r}")
        store._meta = meta
        store._vocab = {k: list(v) for k, v in meta["vocab"].items()}
        store._match_ids = list(meta["match_ids"])
        for name in meta["columns"]:
            store._cols[name] = np.load(d / "columns" / f"{name}.npy", mmap_mode="r")
        store._ids = np.load(d / "ids" / "sorted.npy", mmap_mode="r")
        store._id_frame_row = np.load(d / "ids" / "frame_row.npy", mmap_mode="r")
        store._id_match_ix = np.load(d / "ids" / "match_ix.npy", mmap_mode="r")
        store._row_pos = np.load(d / "ids" / "row_pos.npy", mmap_mode="r")
        for version, info in meta["encodings"].items():
            arr = np.load(d / info["file"], mmap_mode="r")
            store._vectors[version] = _VectorFile(
                arr, int(info["dim"]), bool(info["unit_norm"]), str(info["dtype"])
            )
        for raw in json.loads((d / "matches.json").read_text()):
            m = MatchRecord.model_validate(raw)
            store._matches[m.match_id] = m
        store._catalogue = [CatalogueEntry(**raw) for raw in orjson.loads(
            (d / "catalogue.json").read_bytes()
        )]
        store._baselines = {
            v: [ZoneBaseline.model_validate(b) for b in rows]
            for v, rows in orjson.loads((d / "baselines.json").read_bytes()).items()
        }
        store._details = _DetailSource(
            None if detail_base_url else Path(detail_dir) if detail_dir else d / "detail",
            detail_base_url,
            detail_timeout_s,
            detail_cache,
        )
        return store

    @classmethod
    def from_settings(cls, settings: Settings) -> ArrayStore:
        if settings.arrays_dir is None:
            raise ValueError("store=arrays needs PITCHSCANNER_ARRAYS_DIR")
        return cls.open(
            settings.arrays_dir,
            detail_base_url=settings.detail_base_url,
            detail_timeout_s=settings.detail_timeout_s,
        )

    # -- id lookups -------------------------------------------------------------
    def _locate(self, event_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """(sorted-table position, found) for each id. Ids that are not UUIDs are not found."""

        assert self._ids is not None
        n = len(event_ids)
        pos = np.zeros(n, dtype=np.int64)
        found = np.zeros(n, dtype=bool)
        if n == 0 or len(self._ids) == 0:
            return pos, found
        keys: list[bytes] = []
        idx: list[int] = []
        for i, eid in enumerate(event_ids):
            k = _try_encode_id(eid)
            if k is not None:
                keys.append(k)
                idx.append(i)
        if not keys:
            return pos, found
        query = np.frombuffer(b"".join(keys), dtype="S16")
        p = np.minimum(np.searchsorted(self._ids, query), len(self._ids) - 1)
        ok = np.asarray(self._ids[p] == query)
        sel = np.asarray(idx)
        pos[sel] = p
        found[sel] = ok
        return pos, found

    def _frame_rows(self, event_ids: Sequence[str]) -> list[tuple[str, int]]:
        """(event_id, row) for the ids that are frames, in the order asked (duplicates kept)."""

        self._need_read()
        assert self._id_frame_row is not None
        pos, found = self._locate(event_ids)
        rows = np.asarray(self._id_frame_row[pos]) if len(pos) else np.zeros(0, dtype=np.int64)
        return [
            (eid, int(r))
            for eid, r, ok in zip(event_ids, rows, found, strict=True)
            if ok and r >= 0
        ]

    def _match_of_event(self, event_id: str) -> int | None:
        assert self._id_match_ix is not None
        pos, found = self._locate([event_id])
        if not bool(found[0]):
            return None
        return self._match_ids[int(self._id_match_ix[pos[0]])]

    def _event_id_of_rows(self, rows: np.ndarray) -> list[str]:
        assert self._ids is not None and self._row_pos is not None
        return _decode_ids(np.asarray(self._ids[np.asarray(self._row_pos[rows])]))

    # -- counts and catalogue -----------------------------------------------------
    def counts(self) -> Counts:
        self._need_read()
        c = self._meta["counts"]
        return Counts(
            matches=int(c["matches"]),
            events=int(c["events"]),
            frames=int(c["frames"]),
            vectors=int(sum(c["vectors"].values())),
            outcomes=int(c["outcomes"]),
            possessions=int(c["possessions"]),
        )

    def vector_counts(self) -> dict[str, int]:
        if self._writing:
            return {
                v: sum(len(c) for c in chunks) for v, chunks in self._w_vec_data.items()
            }
        return {v: int(c) for v, c in self._meta["counts"]["vectors"].items()}

    @property
    def effective_index_kind(self) -> str | None:
        return "exact"

    def match(self, match_id: int) -> MatchRecord | None:
        self._need_read()
        return self._matches.get(match_id)

    def matches_by_id(self, match_ids: Sequence[int]) -> dict[int, MatchRecord]:
        self._need_read()
        return {i: self._matches[i] for i in match_ids if i in self._matches}

    def baselines(self, encoding_version: str) -> list[ZoneBaseline]:
        self._need_read()
        return self._baselines.get(encoding_version, [])

    def catalogue(self) -> list[CatalogueEntry]:
        self._need_read()
        return list(self._catalogue)

    # -- per-frame reads ------------------------------------------------------------
    def _meta_at(self, event_id: str, row: int) -> FrameMeta:
        c = self._cols
        player = int(c["player_id"][row])
        return FrameMeta(
            event_id=event_id,
            match_id=int(c["match_id"][row]),
            possession_id=int(c["possession_id"][row]),
            competition_id=int(c["competition_id"][row]),
            season_id=int(c["season_id"][row]),
            possession_team_id=int(c["possession_team_id"][row]),
            opponent_team_id=int(c["opponent_team_id"][row]),
            player_id=None if player == -1 else player,
            minute=int(c["minute"][row]),
            period=int(c["period"][row]),
            stage=self._vocab["stage"][int(c["stage"][row])],
            game_state=self._vocab["game_state"][int(c["game_state"][row])],
            zone_id=int(c["zone_id"][row]),
            possession_is_home=bool(c["possession_is_home"][row]),
        )

    def _outcome_at(self, event_id: str, row: int) -> OutcomeRecord | None:
        bits = int(self._cols["outcome_bits"][row])
        if not bits & _HAS:
            return None
        return OutcomeRecord(
            event_id=event_id,
            shot=bool(bits & _SHOT),
            xg_next=float(self._cols["xg_next"][row]),
            goal=bool(bits & _GOAL),
            turnover=bool(bits & _TURNOVER),
            box_entry=bool(bits & _BOX),
            seconds_to_end=float(self._cols["seconds_to_end"][row]),
        )

    def meta(self, event_id: str) -> FrameMeta | None:
        return self.metas([event_id]).get(event_id)

    def metas(self, event_ids: Sequence[str]) -> dict[str, FrameMeta]:
        return {eid: self._meta_at(eid, row) for eid, row in self._frame_rows(event_ids)}

    def outcome(self, event_id: str) -> OutcomeRecord | None:
        return self.outcomes([event_id]).get(event_id)

    def outcomes(self, event_ids: Sequence[str]) -> dict[str, OutcomeRecord]:
        out: dict[str, OutcomeRecord] = {}
        for eid, row in self._frame_rows(event_ids):
            o = self._outcome_at(eid, row)
            if o is not None:
                out[eid] = o
        return out

    def mirrored_flags(self, event_ids: Sequence[str]) -> dict[str, bool]:
        mirrored = self._cols["mirrored"]
        return {eid: bool(mirrored[row]) for eid, row in self._frame_rows(event_ids)}

    # -- detail reads ------------------------------------------------------------
    def _detail(self, match_id: int) -> _Detail:
        assert self._details is not None
        return self._details.get(match_id)

    def frame(self, event_id: str) -> FrameRecord | None:
        return self.frames_by_id([event_id]).get(event_id)

    def frames_by_id(self, event_ids: Sequence[str]) -> dict[str, FrameRecord]:
        out: dict[str, FrameRecord] = {}
        by_match: dict[int, list[str]] = {}
        match_col = self._cols["match_id"]
        for eid, row in self._frame_rows(event_ids):
            by_match.setdefault(int(match_col[row]), []).append(eid)
        for match_id, ids in by_match.items():
            detail = self._detail(match_id)
            for eid in ids:
                frame = detail.frame(eid)
                if frame is not None:
                    out[eid] = frame
        return out

    def event(self, event_id: str) -> EventRecord | None:
        self._need_read()
        match_id = self._match_of_event(event_id)
        return None if match_id is None else self._detail(match_id).event(event_id)

    def possession_events(self, event_id: str) -> list[EventRecord]:
        self._need_read()
        match_id = self._match_of_event(event_id)
        if match_id is None:
            return []
        detail = self._detail(match_id)
        anchor = detail.event(event_id)
        return [] if anchor is None else detail.possession_events(anchor.possession_id)

    def frames(self, limit: int | None = None) -> list[FrameRecord]:
        self._need_read()
        out: list[FrameRecord] = []
        for match_id in self._match_ids:
            if limit and len(out) >= limit:
                break
            out.extend(self._detail(match_id).frames())
        return out[:limit] if limit else out

    # -- aggregates over the columns --------------------------------------------------
    def _per_match(self) -> list[tuple[int, int, int, int]]:
        """(match_id, frames, first_row, last_row) for each match with frames, by first row."""

        if "per_match" not in self._derived:
            col = np.asarray(self._cols["match_id"])
            ids, first, counts = np.unique(col, return_index=True, return_counts=True)
            last = len(col) - 1 - np.unique(col[::-1], return_index=True)[1]
            rows = sorted(
                zip(
                    ids.tolist(), counts.tolist(), first.tolist(), last.tolist(), strict=True
                ),
                key=lambda r: r[2],
            )
            self._derived["per_match"] = [(m, n, f, last_) for m, n, f, last_ in rows]
        return list(self._derived["per_match"])

    def coverage(self) -> list[CoverageRow]:
        """Only competition/season pairs that actually have 360 frames."""

        self._need_read()
        frames: dict[tuple[int, int], int] = {}
        match_sets: dict[tuple[int, int], set[int]] = {}
        names: dict[tuple[int, int], tuple[int, tuple[str, str]]] = {}
        for match_id, n, _first, last in self._per_match():
            m = self._matches[match_id]
            key = (m.competition_id, m.season_id)
            frames[key] = frames.get(key, 0) + n
            match_sets.setdefault(key, set()).add(match_id)
            # The in-memory store keeps the names of the last frame seen for a pair.
            if key not in names or last > names[key][0]:
                names[key] = (last, (m.competition_name, m.season_name))
        rows: list[CoverageRow] = [
            {
                "competition_id": key[0],
                "competition_name": names[key][1][0],
                "season_id": key[1],
                "season_name": names[key][1][1],
                "frames": n,
                "matches": len(match_sets[key]),
            }
            for key, n in frames.items()
        ]
        return sorted(rows, key=lambda r: (-r["frames"], r["competition_name"]))

    def team_options(self) -> list[TeamRow]:
        self._need_read()
        poss = np.asarray(self._cols["possession_team_id"])
        opp = np.asarray(self._cols["opponent_team_id"])
        both = np.empty(len(poss) * 2, dtype=np.int64)
        both[0::2], both[1::2] = poss, opp
        ids, first, counts = np.unique(both, return_index=True, return_counts=True)
        by_first = np.argsort(first, kind="stable")  # insertion order of the in-memory dict
        names: dict[int, str] = {}
        for match_id, _n, _first, _last in sorted(self._per_match(), key=lambda r: r[3]):
            m = self._matches[match_id]
            names[m.home_team_id] = m.home_team
            names[m.away_team_id] = m.away_team
        rows: list[TeamRow] = [
            {
                "team_id": int(ids[i]),
                "team_name": names.get(int(ids[i]), str(int(ids[i]))),
                "frames": int(counts[i]),
            }
            for i in by_first
        ]
        return sorted(rows, key=lambda r: (-r["frames"], r["team_name"]))

    # -- search --------------------------------------------------------------------------
    def _mask(self, filters: Filters | None) -> np.ndarray | None:
        if filters is None or filters.is_empty():
            return None
        key = filters.model_dump_json()
        with self._mask_lock:
            if key in self._mask_cache:
                self._mask_cache.move_to_end(key)
                return self._mask_cache[key]
        mask = to_mask(self._cols, self._vocab, filters)
        with self._mask_lock:
            self._mask_cache[key] = mask
            while len(self._mask_cache) > 8:
                self._mask_cache.popitem(last=False)
        return mask

    def count_matching(self, encoding_version: str, filters: Filters | None) -> int:
        self._need_read()
        vf = self._vectors.get(encoding_version)
        if vf is None:
            return 0
        mask = self._mask(filters)
        return len(vf.array) if mask is None else int(np.count_nonzero(mask))

    def ann_search(
        self,
        encoding_version: str,
        vector: np.ndarray,
        limit: int,
        metric: str,
        filters: Filters | None = None,
    ) -> list[Candidate]:
        # Exact by construction, so there is nothing approximate to call.
        return self.exact_search(encoding_version, vector, limit, metric, filters)

    def exact_search(
        self,
        encoding_version: str,
        vector: np.ndarray,
        limit: int,
        metric: str,
        filters: Filters | None = None,
    ) -> list[Candidate]:
        self._need_read()
        vf = self._vectors.get(encoding_version)
        if vf is None or len(vf.array) == 0 or limit <= 0:
            return []
        q = np.asarray(vector, dtype=np.float32).ravel()
        if q.shape[0] != vf.dim:
            raise ValueError(f"query has {q.shape[0]} dims, {encoding_version} has {vf.dim}")
        mask = self._mask(filters)
        if mask is not None and not mask.any():
            return []

        cosine = metric == "cosine"
        qn = float(np.linalg.norm(q)) or 1e-9
        n = len(vf.array)
        best_d: list[np.ndarray] = []
        best_i: list[np.ndarray] = []
        for start in range(0, n, CHUNK):
            stop = min(start + CHUNK, n)
            block = vf.array[start:stop]
            if mask is None:
                rows, idx = block, None
            else:
                m = mask[start:stop]
                kept = int(np.count_nonzero(m))
                if kept == 0:
                    continue
                if kept == stop - start:
                    rows, idx = block, None
                else:
                    sel = np.flatnonzero(m)
                    rows, idx = block[sel], sel
            rows32 = rows.astype(np.float32, copy=False)
            if cosine:
                dots = (rows32 @ q).astype(np.float64)
                if vf.unit_norm:
                    d = 1.0 - dots / qn
                else:
                    norms = np.linalg.norm(rows32, axis=1).astype(np.float64)
                    norms[norms == 0] = 1e-9
                    d = 1.0 - dots / (norms * qn)
            else:
                d = np.linalg.norm(rows32 - q[None, :], axis=1).astype(np.float64)
            gidx = (np.arange(len(d)) if idx is None else idx) + start
            if len(d) > limit:
                # Everything at or below the limit-th smallest, so ties survive the cut.
                cut = np.partition(d, limit - 1)[limit - 1]
                keep = np.flatnonzero(d <= cut)
                d, gidx = d[keep], gidx[keep]
            best_d.append(d)
            best_i.append(gidx)
        if not best_d:
            return []
        dist = np.concatenate(best_d)
        gi = np.concatenate(best_i)
        order = np.lexsort((gi, dist))[:limit]  # distance, then row: the stable order
        ids = self._event_id_of_rows(gi[order])
        return [Candidate(eid, float(dist[o])) for eid, o in zip(ids, order, strict=True)]

    def close(self) -> None:
        if self._details is not None:
            self._details.clear()
        self._mask_cache.clear()
