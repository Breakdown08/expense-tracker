"""Needs Postgres: CI starts it as a service; locally `docker compose up -d db`."""

import os

import pytest

from expense_tracker.bot import check_database

DATABASE_URL: str | None = os.environ.get("DATABASE_URL")


@pytest.mark.skipif(DATABASE_URL is None, reason="DATABASE_URL is not set")
async def test_database_reachable() -> None:
    assert DATABASE_URL is not None
    await check_database(DATABASE_URL)
