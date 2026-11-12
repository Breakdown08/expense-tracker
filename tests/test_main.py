import logging
import sys
from pathlib import Path

import psycopg
import pytest
from alembic.script import ScriptDirectory

from expense_tracker import __main__ as entry
from expense_tracker.config import DEFAULT_DATABASE_URL
from expense_tracker.db import alembic_config
from tests.conftest import requires_db

SECRETS: tuple[str, ...] = (
    "TELEGRAM_BOT_TOKEN",
    "ALLOWED_USER_IDS",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "LLM_MODEL",
)


class Calls:
    """Records what `upgrade` and `run` were called with, in place of the real ones."""

    def __init__(self) -> None:
        self.upgrade: list[str] = []
        self.run: list[object] = []

    def fake_upgrade(self, url: str) -> None:
        self.upgrade.append(url)

    def fake_run(self, settings: object) -> None:
        self.run.append(settings)


@requires_db
def test_migrate_brings_schema_to_head_without_bot_config(
    test_db_url: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from alembic import command

    name: str
    for name in SECRETS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", test_db_url)
    monkeypatch.chdir(tmp_path)
    command.downgrade(alembic_config(test_db_url), "base")

    entry.main(["migrate"])

    head: str = ScriptDirectory.from_config(alembic_config(test_db_url)).get_current_head() or ""
    assert head
    conn: psycopg.Connection[tuple[object, ...]]
    with psycopg.connect(test_db_url) as conn:
        rows: list[tuple[object, ...]] = conn.execute("select version_num from alembic_version").fetchall()
    assert rows == [(head,)]


def test_unknown_argument_exits_2_with_usage(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: Calls = Calls()
    monkeypatch.setattr(entry, "upgrade", calls.fake_upgrade)
    monkeypatch.setattr(entry, "run", calls.fake_run)

    raised: pytest.ExceptionInfo[SystemExit]
    with pytest.raises(SystemExit) as raised:
        entry.main(["bogus"])

    assert raised.value.code == 2
    assert capsys.readouterr().err.startswith("usage: expense-tracker [migrate]")
    assert calls.upgrade == []
    assert calls.run == []


def test_migrate_without_database_url_uses_default_and_does_not_run_bot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: Calls = Calls()
    monkeypatch.setattr(entry, "upgrade", calls.fake_upgrade)
    monkeypatch.setattr(entry, "run", calls.fake_run)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    name: str
    for name in SECRETS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)

    entry.main(["migrate"])

    assert calls.upgrade == [DEFAULT_DATABASE_URL]
    assert calls.run == []


def test_migrate_logs_one_info_line(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls: Calls = Calls()
    monkeypatch.setattr(entry, "upgrade", calls.fake_upgrade)
    monkeypatch.setattr(entry, "run", calls.fake_run)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with caplog.at_level(logging.INFO):
        entry.main(["migrate"])

    assert "migrations applied" in [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]


def test_main_without_argv_reads_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Calls = Calls()
    monkeypatch.setattr(entry, "upgrade", calls.fake_upgrade)
    monkeypatch.setattr(entry, "run", calls.fake_run)
    monkeypatch.setattr(sys, "argv", ["expense-tracker", "migrate"])
    monkeypatch.delenv("DATABASE_URL", raising=False)

    entry.main()

    assert calls.upgrade == [DEFAULT_DATABASE_URL]
    assert calls.run == []
