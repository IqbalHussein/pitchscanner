"""Denormalise "was the team in possession the home side" onto frames.

The venue filter must be answerable from the frame row, like every other filter,
so an ANN candidate set can be checked without a join. A fresh database already
has the column (0001 applies the current schema.sql); an older one gets it here,
backfilled from the matches table.

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE frames ADD COLUMN IF NOT EXISTS possession_is_home BOOLEAN"
        " NOT NULL DEFAULT FALSE"
    )
    op.execute(
        "UPDATE frames f SET possession_is_home = TRUE FROM matches m"
        " WHERE m.match_id = f.match_id AND m.home_team_id = f.possession_team_id"
        " AND NOT f.possession_is_home"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE frames DROP COLUMN IF EXISTS possession_is_home")
