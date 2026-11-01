"""Database tests run in their own database next to DATABASE_URL's, never in the real one.

CI starts Postgres as a service; locally `docker compose up -d db` and set DATABASE_URL.
"""

import os
from collections.abc import AsyncIterator, Iterator
from urllib.parse import SplitResult, urlsplit

import psycopg
import pytest
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    AsyncTransaction,
    async_sessionmaker,
)

from expense_tracker.db import make_engine, upgrade

DATABASE_URL: str | None = os.environ.get("DATABASE_URL")
requires_db: pytest.MarkDecorator = pytest.mark.skipif(DATABASE_URL is None, reason="DATABASE_URL is not set")


@pytest.fixture(scope="session")
def test_db_url() -> Iterator[str]:
    """A fresh `<name>_test` database, migrated to head, dropped afterwards."""
    assert DATABASE_URL is not None
    parts: SplitResult = urlsplit(DATABASE_URL)
    name: str = f"{parts.path.lstrip('/')}_test"
    url: str = parts._replace(path=f"/{name}").geturl()
    admin: psycopg.Connection[tuple[object, ...]]
    with psycopg.connect(DATABASE_URL, autocommit=True) as admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        admin.execute(f'CREATE DATABASE "{name}"')
    upgrade(url)
    yield url
    with psycopg.connect(DATABASE_URL, autocommit=True) as admin:
        admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


@pytest.fixture
async def connection(test_db_url: str) -> AsyncIterator[AsyncConnection]:
    """One outer transaction per test, rolled back afterwards."""
    engine: AsyncEngine = make_engine(test_db_url)
    conn: AsyncConnection
    async with engine.connect() as conn:
        outer: AsyncTransaction = await conn.begin()
        yield conn
        await outer.rollback()
    await engine.dispose()


@pytest.fixture
def sessions(connection: AsyncConnection) -> async_sessionmaker[AsyncSession]:
    """Commits inside the code under test become savepoints of the outer transaction."""
    return async_sessionmaker(
        bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
    )


@pytest.fixture
async def session(sessions: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    db: AsyncSession
    async with sessions() as db:
        yield db
