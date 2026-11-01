"""Alembic environment: the URL comes from the caller (`db.upgrade`) or DATABASE_URL (CLI)."""

import os

from alembic import context
from sqlalchemy import Connection, Engine, create_engine, pool

from expense_tracker.config import DEFAULT_DATABASE_URL
from expense_tracker.db import sqlalchemy_url
from expense_tracker.models import Base

url: str = context.config.attributes.get("url") or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
engine: Engine = create_engine(sqlalchemy_url(url), poolclass=pool.NullPool)
connection: Connection
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()
