"""Initial schema: matches, events, frames, outcomes, baselines, entities.

The DDL lives in `pitchscanner/store/schema.sql` so the same statements create the
database in tests, in Compose and in production. This migration applies that file,
which keeps one source of truth instead of two drifting copies.

Vector tables are created per encoding at write time (see
`PostgresStore.ensure_vector_table`), because each encoding has its own
dimensionality; they are dropped here on downgrade.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

from importlib import resources

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

TABLES = (
    "ingest_rejects",
    "entities",
    "zone_baselines",
    "outcomes",
    "frames",
    "events",
    "matches",
)


def upgrade() -> None:
    ddl = resources.files("pitchscanner.store").joinpath("schema.sql").read_text()
    op.execute(ddl)


def downgrade() -> None:
    op.execute(
        "DO $$ DECLARE t text; BEGIN "
        "FOR t IN SELECT tablename FROM pg_tables WHERE tablename LIKE 'frame_vectors_%' "
        "LOOP EXECUTE format('DROP TABLE IF EXISTS %I CASCADE', t); END LOOP; END $$;"
    )
    for table in TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
