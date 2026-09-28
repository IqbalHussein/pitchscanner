"""Alembic environment. The URL comes from settings, never from alembic.ini."""

from __future__ import annotations

from alembic import context
from sqlalchemy import engine_from_config, pool

from pitchscanner.config import get_settings

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().dsn)
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
