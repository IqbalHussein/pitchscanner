-- Base schema. Vector tables are created per encoding by PostgresStore, because
-- each encoding has its own dimensionality and its own index.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS matches (
    match_id          BIGINT PRIMARY KEY,
    competition_id    INT NOT NULL,
    competition_name  TEXT NOT NULL,
    season_id         INT NOT NULL,
    season_name       TEXT NOT NULL,
    home_team_id      INT NOT NULL,
    home_team         TEXT NOT NULL,
    away_team_id      INT NOT NULL,
    away_team         TEXT NOT NULL,
    match_date        DATE,
    stage             TEXT NOT NULL DEFAULT 'regular'
);

CREATE TABLE IF NOT EXISTS events (
    event_id            UUID PRIMARY KEY,
    match_id            BIGINT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    possession_id       INT NOT NULL,
    idx                 INT NOT NULL,
    team_id             INT NOT NULL,
    possession_team_id  INT,
    player_id           INT,
    player_name         TEXT,
    type                TEXT NOT NULL,
    x                   REAL,
    y                   REAL,
    ts                  REAL NOT NULL,
    period              SMALLINT NOT NULL,
    minute              SMALLINT NOT NULL,
    xg                  REAL,
    goal                BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS events_possession_idx
    ON events (match_id, possession_id, idx);
CREATE INDEX IF NOT EXISTS events_type_idx ON events (type);

-- Frames carry the filter columns as well as the positions. That denormalisation
-- is deliberate: filtering an ANN candidate set must not need a join.
CREATE TABLE IF NOT EXISTS frames (
    event_id            UUID PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
    match_id            BIGINT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    possession_id       INT NOT NULL,
    players             JSONB NOT NULL,
    visible_area        JSONB,
    ball_x              REAL NOT NULL,
    ball_y              REAL NOT NULL,
    mirrored            BOOLEAN NOT NULL DEFAULT FALSE,
    competition_id      INT NOT NULL,
    season_id           INT NOT NULL,
    possession_team_id  INT NOT NULL,
    opponent_team_id    INT NOT NULL,
    player_id           INT,
    minute              SMALLINT NOT NULL,
    period              SMALLINT NOT NULL,
    stage               TEXT NOT NULL,
    game_state          TEXT NOT NULL,
    zone_id             SMALLINT NOT NULL
);
CREATE INDEX IF NOT EXISTS frames_possession_idx ON frames (match_id, possession_id);
CREATE INDEX IF NOT EXISTS frames_competition_idx ON frames (competition_id, season_id);
CREATE INDEX IF NOT EXISTS frames_team_idx ON frames (possession_team_id);
CREATE INDEX IF NOT EXISTS frames_opponent_idx ON frames (opponent_team_id);
CREATE INDEX IF NOT EXISTS frames_zone_idx ON frames (zone_id);
CREATE INDEX IF NOT EXISTS frames_state_idx ON frames (game_state, minute);

CREATE TABLE IF NOT EXISTS outcomes (
    event_id        UUID PRIMARY KEY REFERENCES events(event_id) ON DELETE CASCADE,
    shot            BOOLEAN NOT NULL,
    xg_next         REAL NOT NULL,
    goal            BOOLEAN NOT NULL,
    turnover        BOOLEAN NOT NULL,
    box_entry       BOOLEAN NOT NULL,
    seconds_to_end  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS zone_baselines (
    zone_id          SMALLINT NOT NULL,
    encoding_version TEXT NOT NULL,
    n                INT NOT NULL,
    shot_rate        REAL NOT NULL,
    xg_mean          REAL NOT NULL,
    goal_rate        REAL NOT NULL,
    turnover_rate    REAL NOT NULL,
    box_entry_rate   REAL NOT NULL,
    PRIMARY KEY (zone_id, encoding_version)
);

-- Built from the data at ingestion, never hard-coded: the natural-language
-- filter parser may only return IDs that exist here.
CREATE TABLE IF NOT EXISTS entities (
    kind       TEXT NOT NULL,
    entity_id  INT NOT NULL,
    name       TEXT NOT NULL,
    aliases    TEXT[] NOT NULL DEFAULT '{}',
    parent_id  INT,
    frames     INT NOT NULL DEFAULT 0,
    PRIMARY KEY (kind, entity_id)
);

CREATE TABLE IF NOT EXISTS ingest_rejects (
    id         BIGSERIAL PRIMARY KEY,
    source     TEXT NOT NULL,
    reason     TEXT NOT NULL,
    payload    JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
