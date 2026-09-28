"""Postgres + pgvector store: one system of record for rows and vectors.

Vectors live in a table per encoding (``frame_vectors_e1``), because each
encoding has its own dimensionality and needs its own index. ``encoding_version``
is still carried on every row so a rebuilt encoding is never silently mixed with
the vectors of the previous one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from importlib import resources
from typing import Any

import numpy as np
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

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
from pitchscanner.store.filtering import to_sql

#: pgvector operator per metric, and the matching `ORDER BY` direction.
_OPERATOR = {"cosine": "<=>", "l2": "<->"}


def _table_name(encoding_version: str) -> str:
    """One vector table per encoding, named from a sanitised version string."""

    safe = "".join(c for c in encoding_version.lower() if c.isalnum() or c == "_")
    if not safe:
        raise ValueError(f"unusable encoding_version {encoding_version!r}")
    return f"frame_vectors_{safe}"


def _table(encoding_version: str) -> sql.Identifier:
    return sql.Identifier(_table_name(encoding_version))


def _literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(v):.6g}" for v in np.asarray(vector).ravel()) + "]"


class PostgresStore:
    def __init__(self, dsn: str, *, ef_search: int = 80, probes: int = 20) -> None:
        self.dsn = dsn
        self.ef_search = ef_search
        self.probes = probes
        self._conn: psycopg.Connection[Any] | None = None

    # --- plumbing ----------------------------------------------------------
    @property
    def conn(self) -> psycopg.Connection[Any]:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.dsn, row_factory=dict_row, autocommit=True)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None

    def init_schema(self) -> None:
        ddl = resources.files("pitchscanner.store").joinpath("schema.sql").read_text()
        with self.conn.cursor() as cur:
            cur.execute(ddl)

    def ensure_vector_table(self, encoding_version: str, dim: int) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {tbl} (
                        event_id UUID PRIMARY KEY REFERENCES frames(event_id) ON DELETE CASCADE,
                        encoding_version TEXT NOT NULL,
                        embedding vector({dim}) NOT NULL
                    )
                    """
                ).format(tbl=_table(encoding_version), dim=sql.Literal(dim))
            )

    # --- writes ------------------------------------------------------------
    def upsert_matches(self, matches: Iterable[MatchRecord]) -> int:
        rows = [
            (
                m.match_id, m.competition_id, m.competition_name, m.season_id, m.season_name,
                m.home_team_id, m.home_team, m.away_team_id, m.away_team, m.match_date, m.stage,
            )
            for m in matches
        ]
        if not rows:
            return 0
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO matches (match_id, competition_id, competition_name, season_id,
                    season_name, home_team_id, home_team, away_team_id, away_team,
                    match_date, stage)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (match_id) DO UPDATE SET
                    competition_name = EXCLUDED.competition_name,
                    season_name = EXCLUDED.season_name,
                    stage = EXCLUDED.stage
                """,
                rows,
            )
        return len(rows)

    def upsert_events(self, events: Iterable[EventRecord]) -> int:
        """Bulk insert with COPY into a temp table, then one merge. Idempotent."""

        rows = list(events)
        if not rows:
            return 0
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                "CREATE TEMP TABLE tmp_events (LIKE events INCLUDING DEFAULTS) ON COMMIT DROP"
            )
            with cur.copy(
                "COPY tmp_events (event_id, match_id, possession_id, idx, team_id,"
                " possession_team_id, player_id, player_name, type, x, y, ts, period,"
                " minute, xg, goal) FROM STDIN"
            ) as copy:
                for e in rows:
                    copy.write_row(
                        (
                            e.event_id, e.match_id, e.possession_id, e.index, e.team_id,
                            e.possession_team_id, e.player_id, e.player_name, e.type,
                            e.x, e.y, e.ts, e.period, e.minute, e.xg, e.goal,
                        )
                    )
            cur.execute(
                """
                INSERT INTO events SELECT * FROM tmp_events
                ON CONFLICT (event_id) DO UPDATE SET
                    type = EXCLUDED.type, xg = EXCLUDED.xg, goal = EXCLUDED.goal
                """
            )
        return len(rows)

    def upsert_frames(
        self, frames: Iterable[FrameRecord], metas: Iterable[FrameMeta]
    ) -> int:
        frame_list = list(frames)
        meta_by_id = {m.event_id: m for m in metas}
        if not frame_list:
            return 0
        payload = []
        for f in frame_list:
            m = meta_by_id[f.event_id]
            payload.append(
                (
                    f.event_id, f.match_id, f.possession_id,
                    Jsonb([{"x": p.x, "y": p.y, "teammate": p.teammate, "keeper": p.keeper}
                           for p in f.players]),
                    Jsonb([[p.x, p.y] for p in f.visible_area]) if f.visible_area else None,
                    f.ball_x, f.ball_y, f.mirrored,
                    m.competition_id, m.season_id, m.possession_team_id, m.opponent_team_id,
                    m.player_id, m.minute, m.period, m.stage, m.game_state, m.zone_id,
                )
            )
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO frames (event_id, match_id, possession_id, players, visible_area,
                    ball_x, ball_y, mirrored, competition_id, season_id, possession_team_id,
                    opponent_team_id, player_id, minute, period, stage, game_state, zone_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (event_id) DO UPDATE SET
                    players = EXCLUDED.players, visible_area = EXCLUDED.visible_area,
                    ball_x = EXCLUDED.ball_x, ball_y = EXCLUDED.ball_y,
                    game_state = EXCLUDED.game_state, zone_id = EXCLUDED.zone_id
                """,
                payload,
            )
        return len(frame_list)

    def upsert_outcomes(self, outcomes: Iterable[OutcomeRecord]) -> int:
        rows = [
            (o.event_id, o.shot, o.xg_next, o.goal, o.turnover, o.box_entry, o.seconds_to_end)
            for o in outcomes
        ]
        if not rows:
            return 0
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO outcomes (event_id, shot, xg_next, goal, turnover, box_entry,
                    seconds_to_end)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (event_id) DO UPDATE SET
                    shot = EXCLUDED.shot, xg_next = EXCLUDED.xg_next, goal = EXCLUDED.goal,
                    turnover = EXCLUDED.turnover, box_entry = EXCLUDED.box_entry,
                    seconds_to_end = EXCLUDED.seconds_to_end
                """,
                rows,
            )
        return len(rows)

    def upsert_vectors(
        self, encoding_version: str, dim: int, rows: Sequence[tuple[str, np.ndarray]]
    ) -> int:
        if not rows:
            return 0
        self.ensure_vector_table(encoding_version, dim)
        tbl = _table(encoding_version)
        with self.conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {tbl} (event_id, encoding_version, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (event_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        encoding_version = EXCLUDED.encoding_version
                    """
                ).format(tbl=tbl),
                [(eid, encoding_version, _literal(vec)) for eid, vec in rows],
            )
        return len(rows)

    def upsert_baselines(self, baselines: Iterable[ZoneBaseline]) -> int:
        rows = [
            (b.zone_id, b.encoding_version, b.n, b.shot_rate, b.xg_mean, b.goal_rate,
             b.turnover_rate, b.box_entry_rate)
            for b in baselines
        ]
        if not rows:
            return 0
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO zone_baselines (zone_id, encoding_version, n, shot_rate, xg_mean,
                    goal_rate, turnover_rate, box_entry_rate)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (zone_id, encoding_version) DO UPDATE SET
                    n = EXCLUDED.n, shot_rate = EXCLUDED.shot_rate,
                    xg_mean = EXCLUDED.xg_mean, goal_rate = EXCLUDED.goal_rate,
                    turnover_rate = EXCLUDED.turnover_rate,
                    box_entry_rate = EXCLUDED.box_entry_rate
                """,
                rows,
            )
        return len(rows)

    def upsert_catalogue(self, entries: Iterable[CatalogueEntry]) -> int:
        rows = [(e.kind, e.entity_id, e.name, e.aliases, e.parent_id, e.frames) for e in entries]
        if not rows:
            return 0
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO entities (kind, entity_id, name, aliases, parent_id, frames)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (kind, entity_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    aliases = EXCLUDED.aliases,
                    parent_id = EXCLUDED.parent_id,
                    frames = EXCLUDED.frames
                """,
                rows,
            )
        return len(rows)

    def build_index(self, encoding_version: str, kind: str) -> dict[str, object]:
        """(Re)build the ANN index. HNSW by default; IVFFlat and exact are benchmarked."""

        name_str = _table_name(encoding_version)
        tbl = sql.Identifier(name_str)
        name = sql.Identifier(f"{name_str}_ann")
        from pitchscanner.config import get_settings

        cfg = get_settings()
        with self.conn.cursor() as cur:
            cur.execute(sql.SQL("DROP INDEX IF EXISTS {name}").format(name=name))
            if kind == "hnsw":
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX {name} ON {tbl} USING hnsw (embedding vector_cosine_ops)"
                        " WITH (m = {m}, ef_construction = {efc})"
                    ).format(
                        name=name, tbl=tbl,
                        m=sql.Literal(cfg.hnsw_m),
                        efc=sql.Literal(cfg.hnsw_ef_construction),
                    )
                )
            elif kind == "ivfflat":
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX {name} ON {tbl} USING ivfflat"
                        " (embedding vector_cosine_ops) WITH (lists = {lists})"
                    ).format(name=name, tbl=tbl, lists=sql.Literal(cfg.ivfflat_lists))
                )
            cur.execute(sql.SQL("ANALYZE {tbl}").format(tbl=tbl))
            cur.execute(
                sql.SQL("SELECT count(*) AS n, pg_total_relation_size({t}) AS bytes FROM {tbl}")
                .format(t=sql.Literal(name_str), tbl=tbl)
            )
            row = cur.fetchone() or {"n": 0, "bytes": 0}
        return {
            "kind": kind,
            "encoding_version": encoding_version,
            "vectors": int(row["n"]),
            "bytes": int(row["bytes"]),
        }

    # --- reads -------------------------------------------------------------
    def counts(self) -> Counts:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT (SELECT count(*) FROM matches)   AS matches,
                       (SELECT count(*) FROM events)    AS events,
                       (SELECT count(*) FROM frames)    AS frames,
                       (SELECT count(*) FROM outcomes)  AS outcomes,
                       (SELECT count(DISTINCT (match_id, possession_id)) FROM frames)
                           AS possessions
                """
            )
            row = cur.fetchone() or {}
        return Counts(
            matches=int(row.get("matches", 0)),
            events=int(row.get("events", 0)),
            frames=int(row.get("frames", 0)),
            vectors=0,
            outcomes=int(row.get("outcomes", 0)),
            possessions=int(row.get("possessions", 0)),
        )

    def match(self, match_id: int) -> MatchRecord | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM matches WHERE match_id = %s", (match_id,))
            row = cur.fetchone()
        return MatchRecord(**row) if row else None

    def frame(self, event_id: str) -> FrameRecord | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM frames WHERE event_id = %s", (event_id,))
            row = cur.fetchone()
        return _frame_from_row(row) if row else None

    def event(self, event_id: str) -> EventRecord | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM events WHERE event_id = %s", (event_id,))
            row = cur.fetchone()
        return _event_from_row(row) if row else None

    def meta(self, event_id: str) -> FrameMeta | None:
        return self.metas([event_id]).get(event_id)

    def metas(self, event_ids: Sequence[str]) -> dict[str, FrameMeta]:
        if not event_ids:
            return {}
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_id, match_id, possession_id, competition_id, season_id,
                       possession_team_id, opponent_team_id, player_id, minute, period,
                       stage, game_state, zone_id
                FROM frames WHERE event_id = ANY(%s)
                """,
                (list(event_ids),),
            )
            rows = cur.fetchall()
        return {
            str(r["event_id"]): FrameMeta(
                event_id=str(r["event_id"]), match_id=r["match_id"],
                possession_id=r["possession_id"], competition_id=r["competition_id"],
                season_id=r["season_id"], possession_team_id=r["possession_team_id"],
                opponent_team_id=r["opponent_team_id"], player_id=r["player_id"],
                minute=r["minute"], period=r["period"], stage=r["stage"],
                game_state=r["game_state"], zone_id=r["zone_id"],
            )
            for r in rows
        }

    def outcome(self, event_id: str) -> OutcomeRecord | None:
        return self.outcomes([event_id]).get(event_id)

    def outcomes(self, event_ids: Sequence[str]) -> dict[str, OutcomeRecord]:
        if not event_ids:
            return {}
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM outcomes WHERE event_id = ANY(%s)", (list(event_ids),))
            rows = cur.fetchall()
        return {
            str(r["event_id"]): OutcomeRecord(
                event_id=str(r["event_id"]), shot=r["shot"], xg_next=r["xg_next"],
                goal=r["goal"], turnover=r["turnover"], box_entry=r["box_entry"],
                seconds_to_end=r["seconds_to_end"],
            )
            for r in rows
        }

    def frames(self, limit: int | None = None) -> list[FrameRecord]:
        q = "SELECT * FROM frames ORDER BY match_id, possession_id"
        if limit:
            q += f" LIMIT {int(limit)}"
        with self.conn.cursor() as cur:
            cur.execute(q)
            return [_frame_from_row(r) for r in cur.fetchall()]

    def possession_events(self, event_id: str) -> list[EventRecord]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT e.* FROM events e
                JOIN events a ON a.event_id = %s
                WHERE e.match_id = a.match_id AND e.possession_id = a.possession_id
                ORDER BY e.idx
                """,
                (event_id,),
            )
            return [_event_from_row(r) for r in cur.fetchall()]

    def baselines(self, encoding_version: str) -> list[ZoneBaseline]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM zone_baselines WHERE encoding_version = %s ORDER BY zone_id",
                (encoding_version,),
            )
            return [ZoneBaseline(**r) for r in cur.fetchall()]

    def catalogue(self) -> list[CatalogueEntry]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM entities")
            return [
                CatalogueEntry(
                    kind=r["kind"], entity_id=r["entity_id"], name=r["name"],
                    aliases=list(r["aliases"] or []), parent_id=r["parent_id"],
                    frames=r["frames"],
                )
                for r in cur.fetchall()
            ]

    def coverage(self) -> list[CoverageRow]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.competition_id, m.competition_name, m.season_id, m.season_name,
                       count(*) AS frames, count(DISTINCT f.match_id) AS matches
                FROM frames f JOIN matches m ON m.match_id = f.match_id
                GROUP BY 1,2,3,4
                HAVING count(*) > 0
                ORDER BY frames DESC, m.competition_name
                """
            )
            return [
                CoverageRow(
                    competition_id=int(r["competition_id"]),
                    competition_name=str(r["competition_name"]),
                    season_id=int(r["season_id"]),
                    season_name=str(r["season_name"]),
                    frames=int(r["frames"]),
                    matches=int(r["matches"]),
                )
                for r in cur.fetchall()
            ]

    def team_options(self) -> list[TeamRow]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT team_id, max(team_name) AS team_name, sum(frames)::int AS frames
                FROM (
                    SELECT f.possession_team_id AS team_id,
                           CASE WHEN f.possession_team_id = m.home_team_id
                                THEN m.home_team ELSE m.away_team END AS team_name,
                           count(*) AS frames
                    FROM frames f JOIN matches m ON m.match_id = f.match_id
                    GROUP BY 1,2
                    UNION ALL
                    SELECT f.opponent_team_id,
                           CASE WHEN f.opponent_team_id = m.home_team_id
                                THEN m.home_team ELSE m.away_team END,
                           count(*)
                    FROM frames f JOIN matches m ON m.match_id = f.match_id
                    GROUP BY 1,2
                ) t
                GROUP BY team_id
                ORDER BY frames DESC, team_name
                """
            )
            return [
                TeamRow(
                    team_id=int(r["team_id"]),
                    team_name=str(r["team_name"]),
                    frames=int(r["frames"]),
                )
                for r in cur.fetchall()
            ]

    # --- search ------------------------------------------------------------
    def _search(
        self,
        encoding_version: str,
        vector: np.ndarray,
        limit: int,
        metric: str,
        filters: Filters | None,
        *,
        exact: bool,
    ) -> list[Candidate]:
        tbl = _table(encoding_version)
        op = _OPERATOR.get(metric, "<=>")
        params: dict[str, object] = {"q": _literal(vector), "limit": limit}
        where = to_sql(filters, params)
        query = sql.SQL(
            "SELECT v.event_id, v.embedding {op} %(q)s::vector AS distance"
            " FROM {tbl} v JOIN frames f ON f.event_id = v.event_id"
            " {where} ORDER BY distance ASC LIMIT %(limit)s"
        ).format(
            op=sql.SQL(op),
            tbl=tbl,
            where=sql.SQL("WHERE " + where) if where else sql.SQL(""),
        )
        with self.conn.cursor() as cur:
            if exact:
                # Turning the scan off is what makes this the ground truth.
                cur.execute("SET LOCAL enable_indexscan = off")
                cur.execute("SET LOCAL enable_bitmapscan = off")
            else:
                cur.execute("SET LOCAL hnsw.ef_search = %s", (self.ef_search,))
                cur.execute("SET LOCAL ivfflat.probes = %s", (self.probes,))
            cur.execute(query, params)
            return [Candidate(str(r["event_id"]), float(r["distance"])) for r in cur.fetchall()]

    def ann_search(
        self,
        encoding_version: str,
        vector: np.ndarray,
        limit: int,
        metric: str,
        filters: Filters | None = None,
    ) -> list[Candidate]:
        return self._search(encoding_version, vector, limit, metric, filters, exact=False)

    def exact_search(
        self,
        encoding_version: str,
        vector: np.ndarray,
        limit: int,
        metric: str,
        filters: Filters | None = None,
    ) -> list[Candidate]:
        return self._search(encoding_version, vector, limit, metric, filters, exact=True)

    def count_matching(self, encoding_version: str, filters: Filters | None) -> int:
        params: dict[str, object] = {}
        where = to_sql(filters, params)
        query = sql.SQL(
            "SELECT count(*) AS n FROM {tbl} v JOIN frames f ON f.event_id = v.event_id {where}"
        ).format(
            tbl=_table(encoding_version),
            where=sql.SQL("WHERE " + where) if where else sql.SQL(""),
        )
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone() or {"n": 0}
        return int(row["n"])

    def log_reject(self, source: str, reason: str, payload: object) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ingest_rejects (source, reason, payload) VALUES (%s,%s,%s)",
                (source, reason, Jsonb(payload)),
            )


def _frame_from_row(row: dict[str, Any]) -> FrameRecord:
    players = row["players"]
    if isinstance(players, str):
        players = json.loads(players)
    area = row["visible_area"]
    if isinstance(area, str):
        area = json.loads(area)
    return FrameRecord(
        event_id=str(row["event_id"]),
        match_id=row["match_id"],
        possession_id=row["possession_id"],
        players=tuple(
            PlayerPos(p["x"], p["y"], bool(p["teammate"]), bool(p.get("keeper", False)))
            for p in players
        ),
        visible_area=tuple(Point(p[0], p[1]) for p in area) if area else None,
        ball_x=row["ball_x"],
        ball_y=row["ball_y"],
        mirrored=bool(row["mirrored"]),
    )


def _event_from_row(row: dict[str, Any]) -> EventRecord:
    return EventRecord(
        event_id=str(row["event_id"]), match_id=row["match_id"],
        possession_id=row["possession_id"], index=row["idx"], team_id=row["team_id"],
        player_id=row["player_id"], player_name=row["player_name"], type=row["type"],
        x=row["x"], y=row["y"], ts=row["ts"], period=row["period"], minute=row["minute"],
        xg=row["xg"], goal=row["goal"], possession_team_id=row["possession_team_id"],
    )
