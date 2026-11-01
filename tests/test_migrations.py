import pytest
from alembic import command

from expense_tracker.db import alembic_config
from tests.conftest import requires_db

pytestmark: pytest.MarkDecorator = requires_db


def test_models_match_migrations(test_db_url: str) -> None:
    """A model changed without a revision fails here, not on the laptop after `docker compose up`."""
    command.check(alembic_config(test_db_url))


def test_downgrade_and_upgrade_again(test_db_url: str) -> None:
    command.downgrade(alembic_config(test_db_url), "base")
    command.upgrade(alembic_config(test_db_url), "head")
