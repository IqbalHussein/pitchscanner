"""Alembic environment. The URL comes from settings, never from alembic.ini."""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from pitchscanner.config import get_settings


def sqlalchemy_url(dsn: str) -> str:
    """Turn a libpq-style DSN into one SQLAlchemy will open with psycopg 3.

    Supabase hands out ``postgresql://`` (or the older ``postgres://``) URIs.
    SQLAlchemy reads a bare ``postgresql://`` as the psycopg2 driver, which is
    not installed here, so name the driver explicitly. Query parameters such as
    ``sslmode=require`` pass through to libpq untouched.
    """

    for prefix in ("postgresql://", "postgres://"):
        if dsn.startswith(prefix):
            return "postgresql+psycopg://" + dsn[len(prefix) :]
    return dsn


config = context.config
# configparser treats "%" as interpolation syntax; a percent-encoded password
# (common in Supabase URIs) would otherwise raise at this line.
config.set_main_option("sqlalchemy.url", sqlalchemy_url(get_settings().dsn).replace("%", "%%"))
target_metadata = None  # the schema is hand-written SQL, not an ORM model


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
