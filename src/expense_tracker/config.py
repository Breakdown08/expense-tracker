"""Settings from the environment and `.env`; a missing key stops the bot with a readable message."""

from typing import Annotated, ClassVar, Self

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_core import ErrorDetails
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Provider prefix of LLM_MODEL (`init_chat_model("provider:model")`) -> the key it needs.
PROVIDER_KEYS: dict[str, str] = {"groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
DEFAULT_DATABASE_URL: str = "postgresql://tracker:tracker@localhost:5432/tracker"


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(env_file=".env", extra="ignore")

    telegram_bot_token: SecretStr
    # Comma-separated Telegram user ids; anyone else is ignored, or a stranger fills the database.
    allowed_user_ids: Annotated[list[int], NoDecode] = Field(min_length=1)
    llm_model: str = "groq:openai/gpt-oss-120b"
    groq_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    database_url: str = DEFAULT_DATABASE_URL

    # Publishing: each platform is on when its keys are set; otherwise the bot sends the text to post by hand.
    threads_access_token: SecretStr | None = None
    threads_user_id: str | None = None
    telegram_channel_id: str | None = None

    @field_validator("allowed_user_ids", mode="before")
    @classmethod
    def split_ids(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @model_validator(mode="after")
    def provider_key_present(self) -> Self:
        provider: str = self.llm_model.split(":", 1)[0]
        if provider not in PROVIDER_KEYS:
            raise ValueError(
                f"LLM_MODEL: unknown provider {provider!r}, expected one of {sorted(PROVIDER_KEYS)}"
            )
        key: str = PROVIDER_KEYS[provider]
        if getattr(self, key.lower()) is None:
            raise ValueError(f"{key} is required for LLM_MODEL={self.llm_model}")
        return self

    @property
    def threads_enabled(self) -> bool:
        return self.threads_access_token is not None and self.threads_user_id is not None

    @property
    def channel_enabled(self) -> bool:
        return self.telegram_channel_id is not None


def explain(error: ValidationError) -> str:
    """One line per problem, named as in `.env` — not pydantic's field paths."""
    lines: list[str] = ["Configuration error (see .env.example):"]
    item: ErrorDetails
    for item in error.errors():
        loc: tuple[int | str, ...] = item["loc"]
        name: str = str(loc[0]).upper() if loc else ""
        message: str = "is required" if item["type"] == "missing" else item["msg"]
        lines.append(f"  {name} {message}".rstrip() if name else f"  {message.removeprefix('Value error, ')}")
    return "\n".join(lines)
