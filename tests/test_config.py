from pathlib import Path

import pytest
from pydantic import ValidationError

from expense_tracker.config import Settings, explain

REQUIRED: tuple[str, ...] = (
    "TELEGRAM_BOT_TOKEN",
    "ALLOWED_USER_IDS",
    "LLM_MODEL",
    "GROQ_API_KEY",
    "ANTHROPIC_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Neither the developer's environment nor their `.env` leaks into a test."""
    name: str
    for name in REQUIRED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def load(monkeypatch: pytest.MonkeyPatch, **env: str) -> Settings:
    name: str
    value: str
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return Settings()


def error_text(monkeypatch: pytest.MonkeyPatch, **env: str) -> str:
    info: pytest.ExceptionInfo[ValidationError]
    with pytest.raises(ValidationError) as info:
        load(monkeypatch, **env)
    return explain(info.value)


def test_defaults_to_groq(monkeypatch: pytest.MonkeyPatch) -> None:
    settings: Settings = load(
        monkeypatch, TELEGRAM_BOT_TOKEN="t", ALLOWED_USER_IDS="111, 222", GROQ_API_KEY="k"
    )
    assert settings.llm_model == "groq:openai/gpt-oss-120b"
    assert settings.allowed_user_ids == [111, 222]
    assert not settings.threads_enabled
    assert not settings.channel_enabled


def test_missing_keys_are_named_as_in_env(monkeypatch: pytest.MonkeyPatch) -> None:
    text: str = error_text(monkeypatch)
    assert "TELEGRAM_BOT_TOKEN is required" in text
    assert "ALLOWED_USER_IDS is required" in text


def test_provider_key_follows_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    text: str = error_text(
        monkeypatch,
        TELEGRAM_BOT_TOKEN="t",
        ALLOWED_USER_IDS="1",
        GROQ_API_KEY="k",
        LLM_MODEL="anthropic:claude-haiku-5-5",
    )
    assert "ANTHROPIC_API_KEY is required for LLM_MODEL=anthropic:claude-haiku-5-5" in text


def test_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    text: str = error_text(monkeypatch, TELEGRAM_BOT_TOKEN="t", ALLOWED_USER_IDS="1", LLM_MODEL="openai:gpt")
    assert "unknown provider 'openai'" in text


def test_empty_whitelist_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    text: str = error_text(monkeypatch, TELEGRAM_BOT_TOKEN="t", ALLOWED_USER_IDS=" , ", GROQ_API_KEY="k")
    assert "ALLOWED_USER_IDS" in text
