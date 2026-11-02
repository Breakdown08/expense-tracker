"""Database tests run in their own database next to DATABASE_URL's, never in the real one.

CI starts Postgres as a service; locally `docker compose up -d db` and set DATABASE_URL.
"""

import os
from collections.abc import AsyncIterator, Iterator
from datetime import date
from pathlib import Path
from urllib.parse import SplitResult, urlsplit

import psycopg
import pytest
from psycopg import sql
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    AsyncTransaction,
    async_sessionmaker,
)

from expense_tracker.db import make_engine, upgrade
from expense_tracker.rates import Rates, parse

# XML_daily.asp of 31.10.2026, cut down to VND, USD, EUR and THB; windows-1251 as the Bank sends it.
CBR_DAILY: bytes = (Path(__file__).parent / "fixtures" / "cbr_daily.xml").read_bytes()
RATES: Rates = parse(CBR_DAILY)


class FixedRates:
    """The Bank's rates of 31.10.2026, whatever day it is; `None` plays the Bank being down."""

    def __init__(self, rates: Rates | None = RATES) -> None:
        self.rates: Rates | None = rates

    async def latest(self, today: date) -> Rates:
        if self.rates is None:
            raise ConnectionError("Bank of Russia is down")
        return self.rates


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
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    upgrade(url)
    yield url
    with psycopg.connect(DATABASE_URL, autocommit=True) as admin:
        admin.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


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
