"""Telegram over long polling: no public IP, domain or VPS — the bot runs on a laptop at home."""

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import date, datetime
from typing import Any, override
from zoneinfo import ZoneInfo

from aiogram import BaseMiddleware, Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Chat, Message, TelegramObject
from aiogram.utils.chat_action import ChatActionSender
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from expense_tracker.agent import Agent, Context, build_agent, reply
from expense_tracker.config import Settings
from expense_tracker.db import make_engine, make_sessions

log: logging.Logger = logging.getLogger(__name__)

TELEGRAM_LIMIT: int = 4096  # characters in one message; Telegram rejects longer ones

GREETING: str = """Пиши траты как в тетрадку: «фо 60к», «grab 45k, кофе вчера 35к».
Поправить: «нет, было 50». Спросить: «сколько за неделю?», «что записано сегодня»."""

FAILED: str = "Модель не ответила. Попробуй позже; что успело записаться — спроси «что записано сегодня»."


def signed(message: Message) -> str:
    """The group is one memory thread for two: the model needs to know who said what."""
    assert message.from_user is not None and message.text is not None
    return f"[{message.from_user.first_name}] {message.text}"


def chunks(text: str, size: int = TELEGRAM_LIMIT) -> list[str]:
    """Split at line breaks so that a month's list arrives whole; a single overlong line is cut."""
    parts: list[str] = []
    current: str = ""
    line: str
    for line in text.splitlines():
        while len(line) > size:
            if current:
                parts.append(current)
                current = ""
            parts.append(line[:size])
            line = line[size:]
        if current and len(current) + 1 + len(line) > size:
            parts.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        parts.append(current)
    return parts or ["…"]


class InTurn(BaseMiddleware):
    """A chat's updates are handled one at a time, in the order they came.

    Two quick messages would otherwise run the agent on one memory thread at once, and the
    second turn vanished from the history. The lock is taken before the filters: aiogram runs
    sync filters in threads, so by the handler two messages may already be out of order.
    """

    def __init__(self) -> None:
        self.locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    @override
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        chat: Chat | None = data.get("event_chat")
        if chat is None:
            return await handler(event, data)
        async with self.locks[chat.id]:
            return await handler(event, data)


def build_dispatcher(
    settings: Settings, agent: Agent, sessions: async_sessionmaker[AsyncSession]
) -> Dispatcher:
    dp: Dispatcher = Dispatcher()
    dp.update.outer_middleware(InTurn())  # after aiogram's own, which sets event_chat
    # Messages from anyone outside the whitelist are dropped without a reply.
    dp.message.filter(F.from_user.id.in_(settings.allowed_user_ids))
    zone: ZoneInfo = ZoneInfo(settings.timezone)

    @dp.message(CommandStart())
    async def on_start(message: Message) -> None:
        await message.answer(GREETING)

    @dp.message(F.text)
    async def on_text(message: Message, bot: Bot) -> None:
        assert message.from_user is not None and message.text is not None  # the filters above
        answer: str
        async with ChatActionSender.typing(message.chat.id, bot):
            today: date = datetime.now(zone).date()
            context: Context = Context(user_id=message.from_user.id, sessions=sessions, today=today)
            try:
                answer = await reply(agent, message.chat.id, signed(message), context)
            except Exception:
                log.exception("agent failed")
                answer = FAILED
        part: str
        for part in chunks(answer):
            await message.answer(part)

    @dp.message()
    async def on_other(message: Message) -> None:
        await message.answer("Пока понимаю только текст.")

    return dp


async def run(settings: Settings) -> None:
    # No temperature: recent Claude models reject the parameter, and gpt-oss is meant to run at its default.
    model: BaseChatModel = init_chat_model(
        settings.llm_model, api_key=settings.llm_api_key.get_secret_value()
    )
    engine: AsyncEngine = make_engine(settings.database_url)
    saver: AsyncPostgresSaver
    async with AsyncPostgresSaver.from_conn_string(settings.database_url) as saver:
        await saver.setup()  # LangGraph's own tables; idempotent
        agent: Agent = build_agent(model, saver)
        log.info("model %s", settings.llm_model)
        bot: Bot = Bot(settings.telegram_bot_token.get_secret_value())
        try:
            await build_dispatcher(settings, agent, make_sessions(engine)).start_polling(bot)
        finally:
            await engine.dispose()
