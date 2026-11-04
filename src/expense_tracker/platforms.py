"""Where a post goes. Each platform is one small class, on when its keys are in `.env`."""

import httpx
from aiogram import Bot
from aiogram.types import Message

from expense_tracker.config import Settings
from expense_tracker.post import Publisher

THREADS_API: str = "https://graph.threads.net/v1.0"
THREADS_LIMIT: int = 500  # characters in a text post


class TelegramChannel:
    name: str = "Telegram"

    def __init__(self, bot: Bot, channel: str) -> None:
        self.bot: Bot = bot
        self.channel: str = channel

    async def publish(self, text: str) -> str:
        sent: Message = await self.bot.send_message(self.channel, text)
        return f"{self.channel}, message {sent.message_id}"


class Threads:
    """Two steps by Meta's API: a media container, then publishing it."""

    name: str = "Threads"

    def __init__(self, token: str, user_id: str, client: httpx.AsyncClient) -> None:
        self.token: str = token
        self.user_id: str = user_id
        self.client: httpx.AsyncClient = client

    async def call(self, path: str, params: dict[str, str]) -> str:
        response: httpx.Response = await self.client.post(
            f"{THREADS_API}/{self.user_id}/{path}", params={**params, "access_token": self.token}
        )
        if response.is_error:  # httpx's own error would print the URL with the token in it
            raise RuntimeError(f"Threads answered {response.status_code}: {response.text[:200]}")
        return str(response.json()["id"])

    async def publish(self, text: str) -> str:
        if len(text) > THREADS_LIMIT:
            raise ValueError(f"{len(text)} characters, Threads takes {THREADS_LIMIT}")
        container: str = await self.call("threads", {"media_type": "TEXT", "text": text})
        return f"post {await self.call('threads_publish', {'creation_id': container})}"


def publishers(settings: Settings, bot: Bot, client: httpx.AsyncClient) -> list[Publisher]:
    enabled: list[Publisher] = []
    if settings.threads_access_token is not None and settings.threads_user_id is not None:
        token: str = settings.threads_access_token.get_secret_value()
        enabled.append(Threads(token, settings.threads_user_id, client))
    if settings.telegram_channel_id is not None:
        enabled.append(TelegramChannel(bot, settings.telegram_channel_id))
    return enabled
